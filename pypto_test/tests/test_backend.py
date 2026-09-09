# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Backend boundary tests using Host-only fakes.

The fake port models lifecycle reads and kernel ownership, not Device P2P
execution.  Assertions focus on the important boundary: the backend binds
borrowed addresses and controls one resident kernel, but has no execute or VMM
allocation API.
"""

from __future__ import annotations

from pathlib import Path

from pypto_test.pseudo_pypto.backend import WseBackend
from pypto_test.pseudo_pypto.contracts import (
    CACHE_LINE_BYTES,
    WSE_LIFECYCLE_OFFSET,
    BorrowedWindowView,
    EndpointRole,
    LifecycleLine,
)


class FakeKernel:
    binary_sha256 = "abc"
    closed = False

    def synchronize(self):
        return 123

    def close(self):
        self.closed = True


class FakePort:
    generation = 7

    def __init__(self):
        self.kernel = FakeKernel()
        self.writes = []
        self.stopping = False

    def copy_host_to_device(self, destination, payload):
        self.writes.append((destination, payload))
        if destination == 1000 + WSE_LIFECYCLE_OFFSET:
            self.stopping = True

    def copy_device_to_host(self, source, size):
        if source == 1000 + WSE_LIFECYCLE_OFFSET:
            return LifecycleLine(stop_requested=int(self.stopping), stopped=int(self.stopping), ready=1).to_bytes()
        return bytes(size)

    def launch_kernel(self, binary_path, arguments):
        self.binary_path = binary_path
        self.arguments = arguments
        return self.kernel

    def invalidate(self):
        pass


def test_wse_backend_only_controls_resident_kernel():
    port = FakePort()
    local = BorrowedWindowView(EndpointRole.WSE, "wse", 7, 1000, 2**20, 2**21)
    peer = BorrowedWindowView(EndpointRole.ATTENTION, "npu", 7, 10_000_000, 4 * 2**20, 4 * 2**20)
    backend = WseBackend(
        execution_port=port,
        local_window=local,
        npu_peer_window=peer,
        kernel_binary=Path("b_service.o"),
        timeout_seconds=0.1,
    )
    ready = backend.initialize()
    assert ready["backend_kind"] == "WSE"
    assert port.arguments.local_b_input == local.address
    assert not hasattr(backend, "execute")
    assert not hasattr(backend, "allocate_window")
    drained = backend.drain()
    assert drained["kernel_elapsed_ns"] == 123
    assert len(port.writes[0][1]) == 6 * CACHE_LINE_BYTES
    backend.close()
    assert port.kernel.closed
