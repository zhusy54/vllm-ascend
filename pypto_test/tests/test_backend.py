# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""PyPTO execution-backend tests using an addressable Host-only runtime."""

from __future__ import annotations

from pathlib import Path

from pypto_test.pseudo_pypto.backend import NpuExecutionBackend, WseBackend
from pypto_test.pseudo_pypto.communication import (
    CACHE_LINE_BYTES,
    NPU_LOCAL_LIFECYCLE_OFFSET,
    NPU_SHARED_WINDOW_BYTES,
    WSE_LOCAL_LIFECYCLE_OFFSET,
    WSE_SHARED_WINDOW_BYTES,
    LifecycleLine,
    NpuCommunicationBinding,
    WseCommunicationBinding,
)


class FakeBuffer:
    def __init__(self, address, size):
        self.address = address
        self.size = size
        self.closed = False

    def close(self):
        self.closed = True


class FakeKernel:
    binary_sha256 = "kernel-sha"

    def __init__(self):
        self.closed = False

    def synchronize(self):
        return 123

    def close(self):
        self.closed = True


class FakeRuntime:
    def __init__(self, local_base):
        self.local_base = local_base
        self.memory = bytearray(16 * 1024 * 1024)
        self.kernel = FakeKernel()
        self.writes = []
        self.stopping = False

    def allocate_local(self, size):
        self.buffer = FakeBuffer(self.local_base, size)
        return self.buffer

    def copy_host_to_device(self, destination, payload):
        self.memory[destination : destination + len(payload)] = payload
        self.writes.append((destination, len(payload)))
        if len(payload) == CACHE_LINE_BYTES and destination in (
            self.local_base + NPU_LOCAL_LIFECYCLE_OFFSET,
            self.local_base + WSE_LOCAL_LIFECYCLE_OFFSET,
        ) and LifecycleLine.from_bytes(payload).stop_requested:
            self.stopping = True

    def copy_device_to_host(self, source, size):
        if source in (
            self.local_base + NPU_LOCAL_LIFECYCLE_OFFSET,
            self.local_base + WSE_LOCAL_LIFECYCLE_OFFSET,
        ):
            return LifecycleLine(
                stop_requested=int(self.stopping),
                stopped=int(self.stopping),
                ready=1,
            ).to_bytes()
        return bytes(self.memory[source : source + size])

    def launch_kernel(self, binary_path, arguments):
        self.binary_path = binary_path
        self.arguments = arguments
        return self.kernel


def test_npu_backend_owns_local_memory_io_and_driver():
    runtime = FakeRuntime(local_base=1_000_000)
    binding = NpuCommunicationBinding(7, 4_000_000, NPU_SHARED_WINDOW_BYTES, 6_000_000, WSE_SHARED_WINDOW_BYTES)
    backend = NpuExecutionBackend(binding=binding, kernel_binary=Path("abc_driver.o"), runtime=runtime)
    ready = backend.initialize()
    assert ready["binary_sha256"] == "kernel-sha"
    assert runtime.arguments.local_b_output == binding.local_shared_base
    assert runtime.arguments.remote_b_input == binding.peer_shared_base
    assert not hasattr(backend, "allocate_shared_window")
    backend.drain()
    backend.close()
    assert runtime.buffer.closed
    assert runtime.kernel.closed


def test_wse_backend_binds_injected_shared_addresses_and_local_control():
    runtime = FakeRuntime(local_base=2_000_000)
    binding = WseCommunicationBinding(7, 6_000_000, WSE_SHARED_WINDOW_BYTES, 4_000_000, NPU_SHARED_WINDOW_BYTES)
    backend = WseBackend(binding=binding, kernel_binary=Path("b_service.o"), runtime=runtime)
    ready = backend.initialize()
    assert ready["backend_kind"] == "WSE"
    assert runtime.arguments.local_b_input == binding.local_shared_base
    assert runtime.arguments.remote_b_output == binding.peer_shared_base
    assert not hasattr(backend, "execute")
    assert not hasattr(backend, "allocate_shared_window")
    drained = backend.drain()
    assert drained["kernel_elapsed_ns"] == 123
    assert any(size == 3 * CACHE_LINE_BYTES for _, size in runtime.writes)
    backend.close()
    assert runtime.buffer.closed
