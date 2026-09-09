# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import socket
from dataclasses import dataclass

import pytest

from pypto_test.bootstrap import (
    BootstrapError,
    BorrowedDeviceExecutionPort,
    NpuDeviceMemoryManager,
    ProxyControlChannel,
    WindowManifest,
)
from pypto_test.contracts import EndpointRole
from tools.pypto_wse_validation.acl_vmm import VmmExport


@dataclass
class FakeWindow:
    address: int
    logical_bytes: int
    mapping_bytes: int
    shareable_handle: int = 77
    closed: bool = False

    @property
    def export(self):
        return VmmExport(0, self.mapping_bytes, self.shareable_handle)

    def close(self):
        self.closed = True


class FakeRuntime:
    device_id = 0

    def __init__(self):
        self.memory = bytearray(4096)
        self.initialized = False
        self.closed = False

    def initialize(self):
        self.initialized = True

    def allocate_window(self, logical_bytes):
        self.local_window = FakeWindow(1000, logical_bytes, logical_bytes)
        return self.local_window

    def import_window(self, exported, *, peer_device_id):
        del peer_device_id
        self.peer_window = FakeWindow(2000, exported.mapping_bytes, exported.mapping_bytes)
        return self.peer_window

    def copy_host_to_device(self, destination, payload):
        self.memory[destination - 1000 : destination - 1000 + len(payload)] = payload

    def copy_device_to_host(self, source, size):
        return bytes(self.memory[source - 1000 : source - 1000 + size])

    def close(self):
        self.closed = True


def test_window_manifest_round_trip_and_redacts_handle():
    manifest = WindowManifest("attention", EndpointRole.ATTENTION, 0, 2, "npu-window", 10, 16, 99)
    decoded = WindowManifest.from_dict(manifest.to_dict())
    assert decoded == manifest
    assert "shareable_handle" not in manifest.evidence()


def test_control_channel_rejects_payload_keys_and_stale_generation():
    left, right = socket.socketpair()
    try:
        sender = ProxyControlChannel(left, run_id="run", generation=1)
        receiver = ProxyControlChannel(right, run_id="run", generation=2)
        with pytest.raises(BootstrapError, match="data-plane"):
            sender.send("READY", EndpointRole.ATTENTION, payload="forbidden")
        sender.send("READY", EndpointRole.ATTENTION)
        with pytest.raises(BootstrapError, match="generation"):
            receiver.receive("READY", EndpointRole.ATTENTION)
    finally:
        left.close()
        right.close()


def test_execution_port_enforces_borrowed_ranges_and_invalidation():
    runtime = FakeRuntime()
    port = BorrowedDeviceExecutionPort(runtime, generation=3, permitted_ranges=((1000, 128),))
    port.copy_host_to_device(1004, b"abcd")
    assert port.copy_device_to_host(1004, 4) == b"abcd"
    with pytest.raises(BootstrapError, match="outside"):
        port.copy_host_to_device(999, b"x")
    assert not hasattr(port, "allocate_window")
    assert not hasattr(port, "import_window")
    port.invalidate()
    with pytest.raises(BootstrapError, match="invalidated"):
        port.copy_device_to_host(1004, 4)


def test_memory_manager_is_sole_allocator_and_releaser():
    runtime = FakeRuntime()
    manager = NpuDeviceMemoryManager(endpoint_id="attention", device_id=0, generation=1, runtime=runtime)
    local = manager.initialize()
    peer = WindowManifest("surrogate", EndpointRole.WSE_SURROGATE, 1, 1, "wse-window", 64, 64, 88)
    manager.attach(peer)
    _, _, port = manager.borrowed_resources(peer)
    assert manager.audit.actors() == {"NpuDeviceMemoryManager"}
    assert [item["operation"] for item in manager.audit.operations] == ["runtime_initialize", "allocate", "attach"]
    assert local.buffer_id == "npu-window"
    assert not hasattr(port, "release")
    manager.release()
    assert runtime.local_window.closed
    assert runtime.peer_window.closed
    assert runtime.closed
