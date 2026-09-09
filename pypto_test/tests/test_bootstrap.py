# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Bootstrap contract tests that do not require an NPU.

Socket pairs exercise real control framing.  Fake VMM objects verify ownership,
range restriction, invalidation, and cleanup ordering without pretending to
validate physical Device communication.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

import pytest

from pypto_test.bootstrap import (
    BootstrapError,
    BootstrapManager,
    BorrowedDeviceExecutionPort,
    NpuDeviceMemoryManager,
    ProxyControlChannel,
    WindowManifest,
)
from pypto_test.contracts import WSE_WINDOW_BYTES, EndpointRole
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


class FakeControlChannel:
    def __init__(self, peer_manifest):
        self.peer_manifest = peer_manifest
        self.sent = []

    def send(self, message_type, role, **fields):
        self.sent.append((message_type, role, fields))

    def receive(self, expected_type, expected_role):
        del expected_role
        if expected_type == "MANIFEST":
            return {"manifest": self.peer_manifest.to_dict()}
        return {"type": expected_type}


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
    manager.initialize_runtime()
    local = manager.allocate_window()
    peer = WindowManifest("wse", EndpointRole.WSE, 1, 1, "wse-window", 64, 64, 88)
    manager.attach(peer)
    _, _, port = manager.borrowed_resources(peer)
    audit = manager.evidence()["audit"]
    assert {item["actor"] for item in audit} == {"NpuDeviceMemoryManager"}
    assert [item["operation"] for item in audit] == ["runtime_initialize", "allocate", "attach"]
    assert local.buffer_id == "npu-window"
    assert not hasattr(port, "release")
    manager.release()
    assert runtime.local_window.closed
    assert runtime.peer_window.closed
    assert runtime.closed
    assert manager.evidence()["live_mapping_count"] == 0
    assert manager.evidence()["mapping_count"] == 2


def test_bootstrap_keeps_host_initialization_separate_from_communication(monkeypatch):
    manager = BootstrapManager(attention_device=0, wse_device=1, run_id="run", generation=1)
    with pytest.raises(BootstrapError, match="WSE Host"):
        manager.launch_npu_host()

    runtime = FakeRuntime()
    local_manager = NpuDeviceMemoryManager(
        endpoint_id="attention",
        device_id=0,
        generation=1,
        runtime=runtime,
    )
    peer_manifest = WindowManifest(
        "wse",
        EndpointRole.WSE,
        1,
        1,
        "wse-window",
        WSE_WINDOW_BYTES,
        WSE_WINDOW_BYTES,
        88,
    )
    manager.channel = FakeControlChannel(peer_manifest)
    manager._process_controller = object()
    monkeypatch.setattr("pypto_test.bootstrap.NpuDeviceMemoryManager", lambda **kwargs: local_manager)

    manager.launch_npu_host()
    assert runtime.initialized
    assert not hasattr(runtime, "local_window")

    bundle = manager.build_communication()
    assert runtime.local_window is not None
    assert bundle.backend_kind == "WSE"
    assert [item[0] for item in manager.channel.sent[:2]] == ["BUILD_COMMUNICATION", "MANIFEST"]
    local_manager.release()
