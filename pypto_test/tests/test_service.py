# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Proxy API/state tests with a deterministic final-completion fake.

The fake collapses Device computation into the test oracle; it does not prove
P2P communication.  It does prove Host submission order, final-only result
observation, single-request BUSY behavior, and lease quiescing.  The separate
hardware matrix supplies the actual AIV/P2P evidence.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from pypto_test.contracts import (
    DEFAULT_LAYOUT,
    NPU_FINAL_OUTPUT_OFFSET,
    NPU_HOST_REQUEST_DESC_OFFSET,
    NPU_HOST_REQUEST_SIGNAL_OFFSET,
    NPU_HOST_RESULT_DESC_OFFSET,
    NPU_HOST_RESULT_SIGNAL_OFFSET,
    NPU_LIFECYCLE_OFFSET,
    NPU_REPORT_OFFSET,
    BootstrapLease,
    BorrowedWindowView,
    CompletionDescriptor,
    EndpointBundle,
    EndpointRole,
    HostRequestDescriptor,
    LeaseState,
    LifecycleLine,
    ServiceError,
    SignalLine,
    checksum_u32,
    deterministic_input,
    expected_abc,
)
from pypto_test.service import PseudoPyptoDistributedService


class FakeKernel:
    binary_sha256 = "driver-sha"
    closed = False

    def synchronize(self):
        return 456

    def close(self):
        self.closed = True


class FakeRemoteControl:
    def start(self):
        return {"backend_kind": "NPU_SURROGATE"}

    def health(self):
        return {"ready": True}

    def drain(self):
        return {"report": {"completed": 1}}

    def close(self):
        self.closed = True


class FakePort:
    generation = 1

    def __init__(self, base):
        self.base = base
        self.memory = bytearray(4 * 1024 * 1024)
        self.kernel = FakeKernel()
        self.writes = []
        self.ready = False
        self.stopped = False
        self.auto_complete = True
        self.request_submitted = threading.Event()

    def copy_host_to_device(self, destination, payload):
        offset = destination - self.base
        self.memory[offset : offset + len(payload)] = payload
        self.writes.append((offset, len(payload)))
        if offset == NPU_HOST_REQUEST_SIGNAL_OFFSET and len(payload) == 64:
            self.request_submitted.set()
            if self.auto_complete:
                self._complete_request()
        elif offset == NPU_LIFECYCLE_OFFSET:
            self.stopped = True

    def copy_device_to_host(self, source, size):
        offset = source - self.base
        if offset == NPU_LIFECYCLE_OFFSET:
            return LifecycleLine(
                stop_requested=int(self.stopped),
                stopped=int(self.stopped),
                ready=int(self.ready),
            ).to_bytes()
        return bytes(self.memory[offset : offset + size])

    def launch_kernel(self, binary_path, arguments):
        self.binary_path = binary_path
        self.arguments = arguments
        self.ready = True
        return self.kernel

    def invalidate(self):
        pass

    def _complete_request(self):
        descriptor = HostRequestDescriptor.from_bytes(
            bytes(self.memory[NPU_HOST_REQUEST_DESC_OFFSET : NPU_HOST_REQUEST_DESC_OFFSET + 64])
        )
        size = descriptor.element_count * 4
        output = expected_abc(bytes(self.memory[:size]))
        self.memory[NPU_FINAL_OUTPUT_OFFSET : NPU_FINAL_OUTPUT_OFFSET + size] = output
        completion = CompletionDescriptor(
            descriptor.generation,
            descriptor.request_id,
            descriptor.element_count,
            0,
            checksum_u32(output),
            descriptor.sequence,
        )
        self.memory[NPU_HOST_RESULT_DESC_OFFSET : NPU_HOST_RESULT_DESC_OFFSET + 64] = completion.to_bytes()
        self.memory[NPU_HOST_RESULT_SIGNAL_OFFSET : NPU_HOST_RESULT_SIGNAL_OFFSET + 64] = SignalLine(
            descriptor.sequence
        ).to_bytes()
        self.memory[NPU_REPORT_OFFSET : NPU_REPORT_OFFSET + 128] = bytes(128)


def make_service():
    base = 1000
    port = FakePort(base)
    remote = FakeRemoteControl()
    lease = BootstrapLease("lease", 1)
    lease.borrow()
    bundle = EndpointBundle(
        1,
        "attention",
        "NPU_SURROGATE",
        "FAKE",
        "HOST_LOCAL",
        DEFAULT_LAYOUT,
        port,
        BorrowedWindowView(EndpointRole.ATTENTION, "npu", 1, base, 4 * 1024 * 1024, 4 * 1024 * 1024),
        BorrowedWindowView(EndpointRole.WSE_SURROGATE, "wse", 1, 8_000_000, 2 * 1024 * 1024, 2 * 1024 * 1024),
        remote,
        lease,
    )
    return PseudoPyptoDistributedService(bundle, driver_binary=Path("driver.o"), timeout_seconds=0.1), port, lease


def test_service_runs_fixed_abc_with_no_host_intermediate_progress():
    service, port, lease = make_service()
    initialized = service.initialize()
    assert initialized["state"] == "READY"
    payload = deterministic_input(generation=1, request_id=1, element_count=32)
    result = service.execute(payload)
    assert result.output == expected_abc(payload)
    request_submissions = [item for item in port.writes if item == (NPU_HOST_REQUEST_SIGNAL_OFFSET, 64)]
    assert len(request_submissions) == 1
    assert service.evidence()["traffic"]["host_intermediate_bytes"] == 0
    assert service.health()["state"] == "READY"
    service.drain()
    service.close()
    assert lease.state is LeaseState.QUIESCED
    assert port.kernel.closed


def test_service_rejects_execute_before_initialize():
    service, _, _ = make_service()
    with pytest.raises(ServiceError, match="NEW"):
        service.execute(bytes(4))


def test_service_rejects_second_request_while_busy():
    service, port, _ = make_service()
    service.initialize()
    port.auto_complete = False
    payload = deterministic_input(generation=1, request_id=1, element_count=4)
    result = []
    worker = threading.Thread(target=lambda: result.append(service.execute(payload)))
    worker.start()
    assert port.request_submitted.wait(1)
    with pytest.raises(ServiceError, match="EXECUTING"):
        service.execute(payload)
    port._complete_request()
    worker.join(1)
    assert not worker.is_alive()
    assert result[0].output == expected_abc(payload)
    service.close()


def test_close_is_idempotent_and_prevents_execute():
    service, _, _ = make_service()
    service.initialize()
    service.close()
    service.close()
    with pytest.raises(ServiceError, match="CLOSED"):
        service.execute(bytes(4))
