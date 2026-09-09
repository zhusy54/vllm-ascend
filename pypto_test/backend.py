# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Host control backend for the resident surrogate B service."""

from __future__ import annotations

import ctypes
import time
from pathlib import Path
from typing import Any, Protocol

from pypto_test.contracts import (
    CACHE_LINE_BYTES,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_ELEMENTS,
    NPU_B_COMPLETION_DESC_OFFSET,
    NPU_B_COMPLETION_SIGNAL_OFFSET,
    NPU_B_OUTPUT_OFFSET,
    WSE_B_INPUT_OFFSET,
    WSE_B_SUBMISSION_DESC_OFFSET,
    WSE_B_SUBMISSION_SIGNAL_OFFSET,
    WSE_CONTROL_OFFSET,
    WSE_LIFECYCLE_OFFSET,
    WSE_REPORT_OFFSET,
    BorrowedWindowView,
    DeviceExecutionPort,
    LifecycleLine,
    ServiceReport,
)


class BackendError(RuntimeError):
    """Raised when the remote compute service cannot transition state."""


class WseExecutionBackend(Protocol):
    backend_kind: str

    def initialize(self) -> dict[str, Any]: ...

    def health(self) -> dict[str, Any]: ...

    def drain(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class BServiceKernelArguments(ctypes.Structure):
    _fields_ = [
        ("local_b_input", ctypes.c_void_p),
        ("submission_signal", ctypes.c_void_p),
        ("submission_descriptor", ctypes.c_void_p),
        ("lifecycle", ctypes.c_void_p),
        ("report", ctypes.c_void_p),
        ("remote_b_output", ctypes.c_void_p),
        ("remote_completion_signal", ctypes.c_void_p),
        ("remote_completion_descriptor", ctypes.c_void_p),
        ("generation", ctypes.c_uint64),
        ("max_elements", ctypes.c_uint64),
    ]


class NpuSurrogateBackend:
    """Control-plane owner of a surrogate's resident B kernel, not its memory."""

    backend_kind = "NPU_SURROGATE"

    def __init__(
        self,
        *,
        execution_port: DeviceExecutionPort,
        local_window: BorrowedWindowView,
        npu_peer_window: BorrowedWindowView,
        kernel_binary: Path,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._port = execution_port
        self._local = local_window
        self._peer = npu_peer_window
        self._kernel_binary = kernel_binary
        self._timeout_seconds = timeout_seconds
        self._kernel = None
        self._drain_result: dict[str, Any] | None = None

    def initialize(self) -> dict[str, Any]:
        if self._kernel is not None:
            raise BackendError("backend is already initialized")
        self._port.copy_host_to_device(self._local.address + WSE_CONTROL_OFFSET, bytes(6 * CACHE_LINE_BYTES))
        arguments = BServiceKernelArguments(
            self._local.address + WSE_B_INPUT_OFFSET,
            self._local.address + WSE_B_SUBMISSION_SIGNAL_OFFSET,
            self._local.address + WSE_B_SUBMISSION_DESC_OFFSET,
            self._local.address + WSE_LIFECYCLE_OFFSET,
            self._local.address + WSE_REPORT_OFFSET,
            self._peer.address + NPU_B_OUTPUT_OFFSET,
            self._peer.address + NPU_B_COMPLETION_SIGNAL_OFFSET,
            self._peer.address + NPU_B_COMPLETION_DESC_OFFSET,
            self._port.generation,
            MAX_ELEMENTS,
        )
        self._kernel = self._port.launch_kernel(self._kernel_binary, arguments)
        lifecycle, polls = self._wait_for_lifecycle("ready")
        return {
            "backend_kind": self.backend_kind,
            "binary_sha256": self._kernel.binary_sha256,
            "device_ready_poll_reads": polls,
            "lifecycle": lifecycle.__dict__,
        }

    def health(self) -> dict[str, Any]:
        lifecycle = self._read_lifecycle()
        return {
            "backend_kind": self.backend_kind,
            "ready": lifecycle.ready == 1 and lifecycle.stopped == 0,
        }

    def drain(self) -> dict[str, Any]:
        if self._drain_result is not None:
            return dict(self._drain_result)
        if self._kernel is None:
            raise BackendError("backend is not initialized")
        self._port.copy_host_to_device(
            self._local.address + WSE_LIFECYCLE_OFFSET,
            LifecycleLine(stop_requested=1).to_bytes(),
        )
        lifecycle, polls = self._wait_for_lifecycle("stopped")
        elapsed_ns = self._kernel.synchronize()
        report = ServiceReport.from_bytes(
            self._port.copy_device_to_host(self._local.address + WSE_REPORT_OFFSET, 2 * CACHE_LINE_BYTES)
        )
        self._drain_result = {
            "device_stopped_poll_reads": polls,
            "kernel_elapsed_ns": elapsed_ns,
            "lifecycle": lifecycle.__dict__,
            "report": report.to_dict(),
        }
        return dict(self._drain_result)

    def close(self) -> None:
        if self._kernel is None:
            return
        if self._drain_result is None:
            self.drain()
        self._kernel.close()
        self._kernel = None

    def _read_lifecycle(self) -> LifecycleLine:
        return LifecycleLine.from_bytes(
            self._port.copy_device_to_host(self._local.address + WSE_LIFECYCLE_OFFSET, CACHE_LINE_BYTES)
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
                raise BackendError(f"timed out waiting for backend {field}")
            time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)


class RealWseBackend:
    """Reserved contract point; real WSE runtime integration is out of scope."""

    backend_kind = "REAL_WSE_UNIMPLEMENTED"

    def initialize(self) -> dict[str, Any]:
        raise NotImplementedError("real WSE backend is not part of this prototype")

    def health(self) -> dict[str, Any]:
        raise NotImplementedError("real WSE backend is not part of this prototype")

    def drain(self) -> dict[str, Any]:
        raise NotImplementedError("real WSE backend is not part of this prototype")

    def close(self) -> None:
        raise NotImplementedError("real WSE backend is not part of this prototype")
