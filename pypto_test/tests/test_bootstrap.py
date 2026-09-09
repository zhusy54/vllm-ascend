# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""External RPC/VMM ownership and three-stage Bootstrap tests."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest

from pypto_test.infrastructure.bootstrap import BootstrapError, BootstrapManager
from pypto_test.infrastructure.memory import AscendVmmMemoryProvider, WindowManifest
from pypto_test.infrastructure.rpc import HostControlRpcError, MultiprocessingSocketRpc
from pypto_test.pseudo_pypto.communication import (
    NPU_SHARED_WINDOW_BYTES,
    WSE_SHARED_WINDOW_BYTES,
    EndpointRole,
)
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


class FakeVmmRuntime:
    device_id = 0

    def __init__(self):
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

    def close(self):
        self.closed = True


class FakeRpc:
    def __init__(self, peer_manifest):
        self.peer_manifest = peer_manifest
        self.calls = []
        self.launched = False

    def launch_worker(self, worker, *, worker_config, start_order):
        del worker, worker_config, start_order
        self.launched = True

    def receive_event(self, event):
        assert event == "WSE_HOST_READY"
        return {}

    def call(self, method, **fields):
        self.calls.append((method, fields))
        if method == "BUILD_COMMUNICATION":
            return {"manifest": self.peer_manifest.to_dict()}
        return {"details": {}}

    def close_worker(self):
        return {"status": "PASS"}

    def abort(self):
        pass

    def evidence(self):
        return {"calls": len(self.calls)}


def _echo_rpc_worker(*, server, config):
    server.emit("READY", value=config["value"])
    while True:
        method, fields = server.receive_call()
        if method == "ECHO":
            server.reply(method, value=fields["value"])
        elif method == "STOP":
            server.reply(method)
            return {"control": server.evidence()}


def test_window_manifest_round_trip_and_redacts_handle():
    manifest = WindowManifest("attention", EndpointRole.ATTENTION, 0, 2, "npu", 10, 16, 99)
    assert WindowManifest.from_dict(manifest.to_dict()) == manifest
    assert "shareable_handle" not in manifest.evidence()


@pytest.mark.parametrize("start_order", ("attention-first", "wse-first"))
def test_multiprocessing_socket_provider_behaves_as_rpc(start_order):
    rpc = MultiprocessingSocketRpc(run_id=f"rpc-{start_order}", generation=1)
    try:
        rpc.launch_worker(_echo_rpc_worker, worker_config={"value": 7}, start_order=start_order)
        assert rpc.receive_event("READY") == {"value": 7}
        assert rpc.call("ECHO", value=9) == {"value": 9}
        with pytest.raises(HostControlRpcError, match="data-plane"):
            rpc.call("ECHO", input="forbidden")
        rpc.call("STOP")
        assert rpc.close_worker()["status"] == "PASS"
    finally:
        rpc.abort()


def test_memory_provider_is_sole_shared_allocator_and_releaser():
    runtime = FakeVmmRuntime()
    provider = AscendVmmMemoryProvider(
        role=EndpointRole.ATTENTION,
        endpoint_id="attention",
        device_id=0,
        generation=1,
        logical_bytes=NPU_SHARED_WINDOW_BYTES,
        runtime=runtime,
    )
    provider.initialize_host()
    provider.allocate_shared_window()
    peer = WindowManifest("wse", EndpointRole.WSE, 1, 1, "wse", WSE_SHARED_WINDOW_BYTES, WSE_SHARED_WINDOW_BYTES, 88)
    provider.attach_peer(peer)
    binding = provider.binding(peer)
    assert binding.local.address == 1000
    assert binding.peer.address == 2000
    assert not hasattr(provider, "copy_host_to_device")
    assert not hasattr(provider, "launch_kernel")
    provider.release()
    assert runtime.local_window.closed and runtime.peer_window.closed and runtime.closed
    assert provider.evidence()["live_mapping_count"] == 0


def test_bootstrap_keeps_three_host_and_communication_stages_separate():
    runtime = FakeVmmRuntime()
    provider = AscendVmmMemoryProvider(
        role=EndpointRole.ATTENTION,
        endpoint_id="attention",
        device_id=0,
        generation=1,
        logical_bytes=NPU_SHARED_WINDOW_BYTES,
        runtime=runtime,
    )
    peer = WindowManifest("wse", EndpointRole.WSE, 1, 1, "wse", WSE_SHARED_WINDOW_BYTES, WSE_SHARED_WINDOW_BYTES, 88)
    rpc = FakeRpc(peer)
    manager = BootstrapManager(
        attention_device=0,
        wse_device=1,
        run_id="run",
        generation=1,
        rpc=rpc,
        memory_provider_factory=lambda **kwargs: provider,
    )
    with pytest.raises(BootstrapError, match="WSE Host"):
        manager.launch_npu_host()
    manager.launch_wse_host(endpoint_target=lambda: None, kernel_binary=Path("b.o"), start_order="attention-first")
    assert rpc.launched and not runtime.initialized
    manager.launch_npu_host()
    assert runtime.initialized and not hasattr(runtime, "local_window")
    bundle = manager.build_communication()
    assert bundle.npu_communication.local_shared_base == 1000
    assert rpc.calls[0][0] == "BUILD_COMMUNICATION"


def test_bootstrap_source_does_not_own_transport_or_device_execution():
    source = inspect.getsource(BootstrapManager)
    assert "multiprocessing" not in source
    assert "socket." not in source
    assert "copy_host_to_device" not in source
    assert "launch_kernel" not in source
