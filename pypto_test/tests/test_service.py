# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Typed service API and PyPTO-owned Host-IO tests."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from pypto_test.pseudo_pypto.api import ExecuteRequest
from pypto_test.pseudo_pypto.communication import (
    DEFAULT_LAYOUT,
    NPU_LOCAL_FINAL_OUTPUT_OFFSET,
    NPU_LOCAL_HOST_REQUEST_DESC_OFFSET,
    NPU_LOCAL_HOST_REQUEST_SIGNAL_OFFSET,
    NPU_LOCAL_HOST_RESULT_DESC_OFFSET,
    NPU_LOCAL_HOST_RESULT_SIGNAL_OFFSET,
    NPU_LOCAL_INPUT_OFFSET,
    NPU_LOCAL_LIFECYCLE_OFFSET,
    NPU_LOCAL_REPORT_OFFSET,
    NPU_SHARED_WINDOW_BYTES,
    WSE_SHARED_WINDOW_BYTES,
    BootstrapLease,
    CompletionDescriptor,
    EndpointBundle,
    HostRequestDescriptor,
    LeaseState,
    LifecycleLine,
    NpuCommunicationBinding,
    ServiceError,
    SignalLine,
    checksum_u32,
)
from pypto_test.pseudo_pypto.service import PseudoPyptoDistributedService
from pypto_test.validation.validation_utils import expected_abc, get_input_payload


class FakeBuffer:
    def __init__(self, address, size):
        self.address = address
        self.size = size
        self.closed = False

    def close(self):
        self.closed = True


class FakeKernel:
    binary_sha256 = "driver-sha"

    def __init__(self):
        self.closed = False

    def synchronize(self):
        return 456

    def close(self):
        self.closed = True


class FakeRemoteControl:
    def start(self):
        return {"backend_kind": "WSE"}

    def health(self):
        return {"ready": True}

    def drain(self):
        return {"report": {"completed": 1}}

    def close(self):
        self.closed = True


class CompletingRuntime:
    """Models final Device publication, not the intermediate A/B path."""

    def __init__(self):
        self.base = 1_000_000
        self.memory = bytearray(16 * 1024 * 1024)
        self.kernel = FakeKernel()
        self.writes = []
        self.ready = False
        self.stopped = False
        self.auto_complete = True
        self.request_submitted = threading.Event()

    def allocate_local(self, size):
        self.buffer = FakeBuffer(self.base, size)
        return self.buffer

    def copy_host_to_device(self, destination, payload):
        self.memory[destination : destination + len(payload)] = payload
        self.writes.append((destination, len(payload)))
        if destination == self.base + NPU_LOCAL_HOST_REQUEST_SIGNAL_OFFSET and len(payload) == 64:
            self.request_submitted.set()
            if self.auto_complete:
                self.complete_request()
        elif destination == self.base + NPU_LOCAL_LIFECYCLE_OFFSET and len(payload) == 64:
            self.stopped = True

    def copy_device_to_host(self, source, size):
        if source == self.base + NPU_LOCAL_LIFECYCLE_OFFSET:
            return LifecycleLine(
                stop_requested=int(self.stopped),
                stopped=int(self.stopped),
                ready=int(self.ready),
            ).to_bytes()
        return bytes(self.memory[source : source + size])

    def launch_kernel(self, binary_path, arguments):
        self.binary_path = binary_path
        self.arguments = arguments
        self.ready = True
        return self.kernel

    def complete_request(self):
        descriptor = HostRequestDescriptor.from_bytes(
            bytes(
                self.memory[
                    self.base
                    + NPU_LOCAL_HOST_REQUEST_DESC_OFFSET : self.base
                    + NPU_LOCAL_HOST_REQUEST_DESC_OFFSET
                    + 64
                ]
            )
        )
        size = descriptor.element_count * 4
        payload = bytes(self.memory[self.base + NPU_LOCAL_INPUT_OFFSET : self.base + NPU_LOCAL_INPUT_OFFSET + size])
        output = expected_abc(payload)
        final_start = self.base + NPU_LOCAL_FINAL_OUTPUT_OFFSET
        self.memory[final_start : final_start + size] = output
        completion = CompletionDescriptor(
            descriptor.generation,
            descriptor.request_id,
            descriptor.element_count,
            0,
            checksum_u32(output),
            descriptor.sequence,
        )
        descriptor_start = self.base + NPU_LOCAL_HOST_RESULT_DESC_OFFSET
        signal_start = self.base + NPU_LOCAL_HOST_RESULT_SIGNAL_OFFSET
        self.memory[descriptor_start : descriptor_start + 64] = completion.to_bytes()
        self.memory[signal_start : signal_start + 64] = SignalLine(descriptor.sequence).to_bytes()
        report_start = self.base + NPU_LOCAL_REPORT_OFFSET
        self.memory[report_start : report_start + 128] = bytes(128)


def make_service():
    runtime = CompletingRuntime()
    remote = FakeRemoteControl()
    lease = BootstrapLease("lease", 1)
    lease.borrow()
    bundle = EndpointBundle(
        1,
        "attention",
        "WSE",
        "FAKE",
        "HOST_LOCAL",
        DEFAULT_LAYOUT,
        NpuCommunicationBinding(1, 4_000_000, NPU_SHARED_WINDOW_BYTES, 6_000_000, WSE_SHARED_WINDOW_BYTES),
        remote,
        lease,
    )
    service = PseudoPyptoDistributedService(
        bundle,
        driver_binary=Path("driver.o"),
        execution_runtime=runtime,
    )
    return service, runtime, lease


def test_service_runs_fixed_abc_with_no_host_intermediate_progress():
    service, runtime, lease = make_service()
    assert service.initialize().state.value == "READY"
    payload = get_input_payload(generation=1, request_id=1, element_count=32)
    result = service.execute(ExecuteRequest(payload)).result
    assert result.output == expected_abc(payload)
    request_signal = runtime.base + NPU_LOCAL_HOST_REQUEST_SIGNAL_OFFSET
    request_submissions = [item for item in runtime.writes if item == (request_signal, 64)]
    assert len(request_submissions) == 1
    assert service.evidence()["traffic"]["host_intermediate_bytes"] == 0
    assert service.health().state.value == "READY"
    service.drain()
    service.close()
    assert lease.state is LeaseState.QUIESCED
    assert runtime.kernel.closed
    assert runtime.buffer.closed


def test_service_rejects_execute_before_initialize():
    service, _, _ = make_service()
    with pytest.raises(ServiceError, match="NEW"):
        service.execute(ExecuteRequest(bytes(4)))


def test_service_rejects_second_request_while_busy():
    service, runtime, _ = make_service()
    service.initialize()
    runtime.auto_complete = False
    payload = get_input_payload(generation=1, request_id=1, element_count=4)
    results = []
    worker = threading.Thread(target=lambda: results.append(service.execute(ExecuteRequest(payload))))
    worker.start()
    assert runtime.request_submitted.wait(1)
    with pytest.raises(ServiceError, match="EXECUTING"):
        service.execute(ExecuteRequest(payload))
    runtime.complete_request()
    worker.join(1)
    assert not worker.is_alive()
    assert results[0].result.output == expected_abc(payload)
    service.close()


def test_close_is_idempotent_and_prevents_execute():
    service, _, _ = make_service()
    service.initialize()
    service.close()
    service.close()
    with pytest.raises(ServiceError, match="CLOSED"):
        service.execute(ExecuteRequest(bytes(4)))
