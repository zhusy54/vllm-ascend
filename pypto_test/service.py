# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Fixed-program pseudo-PyPTO distributed service.

The service is the only object exposed to the hypothetical upper layer.  Its
normal lifecycle is NEW -> INITIALIZING -> READY -> EXECUTING -> READY, then
DRAINING -> DRAINED -> CLOSED.  It borrows a prepared EndpointBundle and never
allocates or maps communication memory.

One execute call follows this end-to-end path:

    Host input H2D + request publication
      -> resident Attention driver executes A
      -> Device P2P submission wakes resident surrogate B
      -> Device P2P completion wakes the Attention driver
      -> Attention driver executes C and publishes final completion
      -> Host observes only final completion and copies final output D2H

There is no Host read/copy/RPC between A, B, and C.  This fixed protocol proves
the service and dependency path, not a general PyPTO compiler or scheduler.
"""

from __future__ import annotations

import ctypes
import time
from pathlib import Path
from typing import Any

from pypto_test.contracts import (
    CACHE_LINE_BYTES,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_ELEMENTS,
    NPU_B_COMPLETION_DESC_OFFSET,
    NPU_B_COMPLETION_SIGNAL_OFFSET,
    NPU_B_OUTPUT_OFFSET,
    NPU_CONTROL_OFFSET,
    NPU_FINAL_OUTPUT_OFFSET,
    NPU_HOST_REQUEST_DESC_OFFSET,
    NPU_HOST_REQUEST_SIGNAL_OFFSET,
    NPU_HOST_RESULT_DESC_OFFSET,
    NPU_HOST_RESULT_SIGNAL_OFFSET,
    NPU_INPUT_OFFSET,
    NPU_LIFECYCLE_OFFSET,
    NPU_REPORT_OFFSET,
    WSE_B_INPUT_OFFSET,
    WSE_B_SUBMISSION_DESC_OFFSET,
    WSE_B_SUBMISSION_SIGNAL_OFFSET,
    CompletionDescriptor,
    DriverReport,
    EndpointBundle,
    ExecutionResult,
    HostRequestDescriptor,
    LifecycleLine,
    PseudoProgramSpec,
    ServiceError,
    ServiceState,
    SignalLine,
    checksum_u32,
)


class DriverKernelArguments(ctypes.Structure):
    """Host ABI matching ``pypto_abc_driver_0_mix_aiv`` exactly.

    Local addresses cover Host ingress, B return, final output, and all local
    control lines.  The three remote addresses are imported views of the
    surrogate input/submission area used directly by the Attention Device.
    """

    _fields_ = [
        ("local_input", ctypes.c_void_p),
        ("local_b_output", ctypes.c_void_p),
        ("local_final_output", ctypes.c_void_p),
        ("host_request_signal", ctypes.c_void_p),
        ("host_request_descriptor", ctypes.c_void_p),
        ("host_result_signal", ctypes.c_void_p),
        ("host_result_descriptor", ctypes.c_void_p),
        ("b_completion_signal", ctypes.c_void_p),
        ("b_completion_descriptor", ctypes.c_void_p),
        ("lifecycle", ctypes.c_void_p),
        ("report", ctypes.c_void_p),
        ("remote_b_input", ctypes.c_void_p),
        ("remote_submission_signal", ctypes.c_void_p),
        ("remote_submission_descriptor", ctypes.c_void_p),
        ("generation", ctypes.c_uint64),
        ("max_elements", ctypes.c_uint64),
    ]


class FinalCompletionObserver:
    """Poll only the final Host-visible completion.

    Blocking here is valid service behavior.  The observer never examines the
    B-completion line and never synchronizes the resident driver stream, so it
    cannot be responsible for advancing B -> C.
    """

    def __init__(self, bundle: EndpointBundle, *, timeout_seconds: float) -> None:
        self._bundle = bundle
        self._timeout_seconds = timeout_seconds

    def wait(self, sequence: int) -> tuple[CompletionDescriptor, int]:
        address = self._bundle.npu_local_window.address
        port = self._bundle.execution_port
        deadline = time.monotonic() + self._timeout_seconds
        poll_reads = 0
        while True:
            # Count every cache-line D2H read independently from the final
            # payload D2H.  This makes Host involvement measurable.
            signal = SignalLine.from_bytes(
                port.copy_device_to_host(address + NPU_HOST_RESULT_SIGNAL_OFFSET, CACHE_LINE_BYTES)
            )
            poll_reads += 1
            if signal.sequence == sequence:
                # The driver's signal is written only after final output and
                # this descriptor are globally visible.
                descriptor = CompletionDescriptor.from_bytes(
                    port.copy_device_to_host(address + NPU_HOST_RESULT_DESC_OFFSET, CACHE_LINE_BYTES)
                )
                return descriptor, poll_reads
            if signal.sequence > sequence:
                raise ServiceError("final completion sequence advanced beyond the request")
            if time.monotonic() >= deadline:
                raise ServiceError("timed out waiting for final device completion")
            time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)


class PseudoPyptoDistributedService:
    """Synchronous, single-request proxy with PyPTO-compatible assumed APIs.

    ``initialize/execute/health/drain/close`` are compatibility assumptions for
    this experiment.  They are not imported from, or registered into, vLLM.
    """

    def __init__(
        self,
        bundle: EndpointBundle,
        *,
        driver_binary: Path,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        bundle.validate()
        self._bundle = bundle
        self._driver_binary = driver_binary
        self._timeout_seconds = timeout_seconds
        self._observer = FinalCompletionObserver(bundle, timeout_seconds=timeout_seconds)
        self._state = ServiceState.NEW
        self._kernel = None
        self._sequence = 0
        self._request_id = 0
        self._driver_drain: dict[str, Any] | None = None
        self._remote_drain: dict[str, Any] | None = None
        # These counters distinguish permitted Host ingress/egress from the
        # forbidden case where Host transports an A/B intermediate value.
        self._traffic = {
            "final_control_d2h_bytes": 0,
            "final_payload_d2h_bytes": 0,
            "host_intermediate_bytes": 0,
            "input_h2d_bytes": 0,
            "request_control_h2d_bytes": 0,
        }

    @property
    def state(self) -> ServiceState:
        return self._state

    def initialize(self, program: PseudoProgramSpec | None = None) -> dict[str, Any]:
        """Validate the bundle, start both resident kernels, and reach READY."""

        if self._state is not ServiceState.NEW:
            raise ServiceError("initialize is only valid in NEW state")
        self._state = ServiceState.INITIALIZING
        spec = program if program is not None else PseudoProgramSpec()
        spec.validate()
        self._bundle.validate()
        local = self._bundle.npu_local_window.address
        port = self._bundle.execution_port
        # Step 1: clear Attention control lines.  Payload regions are populated
        # only by execute or by the two Device kernels.
        port.copy_host_to_device(local + NPU_CONTROL_OFFSET, bytes(10 * CACHE_LINE_BYTES))
        # Step 2: start B first, so it is already waiting when the driver is
        # launched.  This RPC is generation-level and carries no request data.
        remote_ready = self._bundle.wse_control.start()
        # Step 3: bind local and imported peer VAs into the resident driver ABI.
        arguments = DriverKernelArguments(
            local + NPU_INPUT_OFFSET,
            local + NPU_B_OUTPUT_OFFSET,
            local + NPU_FINAL_OUTPUT_OFFSET,
            local + NPU_HOST_REQUEST_SIGNAL_OFFSET,
            local + NPU_HOST_REQUEST_DESC_OFFSET,
            local + NPU_HOST_RESULT_SIGNAL_OFFSET,
            local + NPU_HOST_RESULT_DESC_OFFSET,
            local + NPU_B_COMPLETION_SIGNAL_OFFSET,
            local + NPU_B_COMPLETION_DESC_OFFSET,
            local + NPU_LIFECYCLE_OFFSET,
            local + NPU_REPORT_OFFSET,
            self._bundle.wse_peer_window.address + WSE_B_INPUT_OFFSET,
            self._bundle.wse_peer_window.address + WSE_B_SUBMISSION_SIGNAL_OFFSET,
            self._bundle.wse_peer_window.address + WSE_B_SUBMISSION_DESC_OFFSET,
            self._bundle.generation,
            MAX_ELEMENTS,
        )
        self._kernel = port.launch_kernel(self._driver_binary, arguments)
        # Step 4: READY is written by the Device kernel itself; observing it
        # proves the resident loop is live before admission begins.
        lifecycle, polls = self._wait_for_lifecycle("ready")
        self._state = ServiceState.READY
        return {
            "driver_binary_sha256": self._kernel.binary_sha256,
            "driver_ready_poll_reads": polls,
            "generation": self._bundle.generation,
            "lifecycle": lifecycle.__dict__,
            "program": spec.to_dict(),
            "remote": remote_ready,
            "state": self._state.value,
        }

    def execute(self, payload: bytes) -> ExecutionResult:
        """Submit one input and block until the driver's final C completion."""

        if self._state is not ServiceState.READY:
            raise ServiceError(f"execute is not valid in {self._state.value} state")
        input_checksum = checksum_u32(payload)
        element_count = len(payload) // 4
        self._state = ServiceState.EXECUTING
        self._sequence += 1
        self._request_id += 1
        sequence = self._sequence
        request_id = self._request_id
        local = self._bundle.npu_local_window.address
        port = self._bundle.execution_port
        descriptor = HostRequestDescriptor(
            self._bundle.generation,
            request_id,
            element_count,
            input_checksum,
            sequence,
        )
        try:
            # Publish request in payload -> descriptor -> signal order.  The
            # signal is the only event that admits work into the resident
            # Device graph, so the driver cannot observe a partial request.
            port.copy_host_to_device(local + NPU_INPUT_OFFSET, payload)
            self._traffic["input_h2d_bytes"] += len(payload)
            port.copy_host_to_device(local + NPU_HOST_REQUEST_DESC_OFFSET, descriptor.to_bytes())
            port.copy_host_to_device(local + NPU_HOST_REQUEST_SIGNAL_OFFSET, SignalLine(sequence).to_bytes())
            self._traffic["request_control_h2d_bytes"] += 2 * CACHE_LINE_BYTES

            # From here until final completion, Host performs no operation on
            # A output, B submission, B payload, or B completion.  Both Device
            # kernels advance that dependency chain by remote memory signals.
            completion, poll_reads = self._observer.wait(sequence)
            self._traffic["final_control_d2h_bytes"] += (poll_reads + 1) * CACHE_LINE_BYTES
            if completion.generation != self._bundle.generation:
                raise ServiceError("completion generation mismatch")
            if completion.request_id != request_id or completion.sequence != sequence:
                raise ServiceError("completion identity mismatch")
            if completion.element_count != element_count or completion.status != 0:
                raise ServiceError(f"device task failed with status {completion.status}")
            # Copy C output only after validating the final completion identity.
            # This is the service result-return path to the upper layer.
            output = port.copy_device_to_host(local + NPU_FINAL_OUTPUT_OFFSET, len(payload))
            self._traffic["final_payload_d2h_bytes"] += len(output)
            if checksum_u32(output) != completion.output_checksum:
                raise ServiceError("final output checksum mismatch")
            return ExecutionResult(
                self._bundle.generation,
                request_id,
                sequence,
                element_count,
                output,
                completion.output_checksum,
                poll_reads,
                (poll_reads + 1) * CACHE_LINE_BYTES,
                len(output),
            )
        finally:
            # The first version validates only synchronous normal execution and
            # max_inflight=1.  Returning to READY admits the next serial call.
            self._state = ServiceState.READY

    def health(self) -> dict[str, Any]:
        return {
            "generation": self._bundle.generation,
            "remote": self._bundle.wse_control.health(),
            "state": self._state.value,
        }

    def drain(self) -> dict[str, Any]:
        """Stop admission and both resident loops while mappings remain valid."""

        if self._state is ServiceState.DRAINED:
            return self._drain_evidence()
        if self._state is not ServiceState.READY:
            raise ServiceError(f"drain is not valid in {self._state.value} state")
        if self._kernel is None:
            raise ServiceError("driver kernel is not initialized")
        self._state = ServiceState.DRAINING
        local = self._bundle.npu_local_window.address
        # With no in-flight request, the Attention driver can leave its wait
        # loop immediately.  The surrogate receives the equivalent STOP via
        # its generation-level control endpoint below.
        self._bundle.execution_port.copy_host_to_device(
            local + NPU_LIFECYCLE_OFFSET,
            LifecycleLine(stop_requested=1).to_bytes(),
        )
        lifecycle, polls = self._wait_for_lifecycle("stopped")
        self._remote_drain = self._bundle.wse_control.drain()
        # Synchronize only during drain.  Doing this in execute would block on
        # the resident loop and prevent multiple requests per generation.
        elapsed_ns = self._kernel.synchronize()
        report = DriverReport.from_bytes(
            self._bundle.execution_port.copy_device_to_host(local + NPU_REPORT_OFFSET, 2 * CACHE_LINE_BYTES)
        )
        self._driver_drain = {
            "device_stopped_poll_reads": polls,
            "kernel_elapsed_ns": elapsed_ns,
            "lifecycle": lifecycle.__dict__,
            "report": report.to_dict(),
        }
        self._state = ServiceState.DRAINED
        return self._drain_evidence()

    def close(self) -> None:
        """Unload service-owned kernels and return the lease quiesced."""

        if self._state is ServiceState.CLOSED:
            return
        if self._state is ServiceState.READY:
            self.drain()
        if self._state is not ServiceState.DRAINED:
            raise ServiceError(f"close is not valid in {self._state.value} state")
        self._bundle.wse_control.close()
        if self._kernel is not None:
            self._kernel.close()
            self._kernel = None
        # Quiescing transfers no ownership; it only tells Bootstrap that all
        # borrowers are done and physical mappings may now be released.
        self._bundle.lease.quiesce()
        self._state = ServiceState.CLOSED

    def evidence(self) -> dict[str, Any]:
        return {
            "driver": self._driver_drain,
            "generation": self._bundle.generation,
            "remote": self._remote_drain,
            "state": self._state.value,
            "traffic": dict(self._traffic),
        }

    def _read_lifecycle(self) -> LifecycleLine:
        return LifecycleLine.from_bytes(
            self._bundle.execution_port.copy_device_to_host(
                self._bundle.npu_local_window.address + NPU_LIFECYCLE_OFFSET,
                CACHE_LINE_BYTES,
            )
        )

    def _wait_for_lifecycle(self, field: str) -> tuple[LifecycleLine, int]:
        deadline = time.monotonic() + self._timeout_seconds
        polls = 0
        while True:
            lifecycle = self._read_lifecycle()
            polls += 1
            if getattr(lifecycle, field) == 1:
                return lifecycle, polls
            if time.monotonic() >= deadline:
                raise ServiceError(f"timed out waiting for driver {field}")
            time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)

    def _drain_evidence(self) -> dict[str, Any]:
        return {"driver": self._driver_drain, "remote": self._remote_drain, "state": self._state.value}
