# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Three-stage Host bootstrap and PyPTO communication-resource injection.

The public sequence is deliberately explicit:

``launch_wse_host`` -> ``launch_npu_host`` -> ``build_communication``.

The manager delegates process transport to ``HostControlRpc`` and VMM work to
``CrossDeviceMemoryProvider``.  It never gives PyPTO an allocator, an
``AclVmmRuntime``, or Host copy/kernel-launch methods.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from pypto_test.infrastructure.memory import (
    AscendVmmMemoryProvider,
    CrossDeviceMemoryBinding,
    CrossDeviceMemoryProvider,
    WindowManifest,
)
from pypto_test.infrastructure.rpc import HostControlRpc, MultiprocessingSocketRpc
from pypto_test.pseudo_pypto.communication import (
    DEFAULT_LAYOUT,
    NPU_SHARED_WINDOW_BYTES,
    WSE_SHARED_WINDOW_BYTES,
    BootstrapLease,
    EndpointBundle,
    EndpointRole,
    LeaseState,
    NpuCommunicationBinding,
    WseCommunicationBinding,
    WseServiceControl,
)


class BootstrapError(RuntimeError):
    """Raised when resource setup or teardown violates the bootstrap order."""


class RemoteWseServiceControl(WseServiceControl):
    """Generation-level WSE control; per-request execution never uses RPC."""

    def __init__(self, rpc: HostControlRpc) -> None:
        self._rpc = rpc
        self._closed = False

    def start(self) -> dict[str, Any]:
        return self._rpc.call("START").get("details", {})

    def health(self) -> dict[str, Any]:
        return self._rpc.call("HEALTH").get("details", {})

    def drain(self) -> dict[str, Any]:
        return self._rpc.call("DRAIN").get("details", {})

    def close(self) -> None:
        if self._closed:
            return
        self._rpc.call("CLOSE")
        self._closed = True


class BootstrapManager:
    """External owner of Host control and cross-Device communication memory."""

    def __init__(
        self,
        *,
        attention_device: int,
        wse_device: int,
        run_id: str,
        generation: int,
        rpc: HostControlRpc | None = None,
    ) -> None:
        self.attention_device = attention_device
        self.wse_device = wse_device
        self.run_id = run_id
        self.generation = generation
        self._rpc = rpc if rpc is not None else MultiprocessingSocketRpc(run_id=run_id, generation=generation)
        self._npu_memory_provider: CrossDeviceMemoryProvider | None = None
        self._lease: BootstrapLease | None = None
        self._lifecycle_events: list[str] = []
        self._wse_evidence: dict[str, Any] | None = None

    def __enter__(self) -> BootstrapManager:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.abort()

    def launch_wse_host(
        self,
        *,
        endpoint_target: Callable[..., dict[str, Any]],
        kernel_binary: Path,
        start_order: str,
    ) -> None:
        """Launch the WSE Host control worker and its non-communication setup."""

        if self._lifecycle_events:
            raise BootstrapError("WSE Host is already launched")
        try:
            self._rpc.launch_worker(
                endpoint_target,
                worker_config={
                    "device_id": self.wse_device,
                    "generation": self.generation,
                    "kernel_binary": str(kernel_binary),
                },
                start_order=start_order,
            )
            self._rpc.receive_event("WSE_HOST_READY")
            self._lifecycle_events.append("wse_host_ready")
        except BaseException:
            self.abort()
            raise

    def launch_npu_host(self) -> None:
        """Initialize external ACL/device state; do not allocate PyPTO memory."""

        if self._lifecycle_events != ["wse_host_ready"]:
            raise BootstrapError("WSE Host must be ready before NPU Host")
        provider = create_npu_memory_provider(
            device_id=self.attention_device,
            generation=self.generation,
        )
        try:
            provider.initialize_host()
        except BaseException:
            provider.release()
            raise
        self._npu_memory_provider = provider
        self._lifecycle_events.append("npu_host_ready")

    def build_communication(self) -> EndpointBundle:
        """Provision shared mappings and inject NPU-process Device addresses."""

        if self._lifecycle_events != ["wse_host_ready", "npu_host_ready"]:
            raise BootstrapError("both Hosts must be ready before communication")
        if self._npu_memory_provider is None or self._lease is not None:
            raise BootstrapError("communication cannot build in the current state")

        npu_binding, local_manifest = build_npu_communication(self._npu_memory_provider, self._rpc)

        lease = BootstrapLease(f"lease-{uuid4().hex}", self.generation)
        lease.borrow()
        self._lease = lease
        self._lifecycle_events.extend(("communication_built", "lease_borrowed"))
        bundle = EndpointBundle(
            generation=self.generation,
            endpoint_id=local_manifest.endpoint_id,
            backend_kind="WSE",
            transport_kind="ASCEND_VMM_P2P",
            transport_scope="HOST_LOCAL",
            layout=DEFAULT_LAYOUT,
            npu_communication=npu_binding,
            wse_control=RemoteWseServiceControl(self._rpc),
            lease=lease,
        )
        bundle.validate()
        return bundle

    def release(self) -> dict[str, Any]:
        """Release external mappings only after PyPTO has stopped all kernels."""

        if self._lease is None or self._npu_memory_provider is None:
            raise BootstrapError("communication was not built")
        if self._lease.state is not LeaseState.QUIESCED:
            raise BootstrapError("service must quiesce its lease before bootstrap release")
        self._rpc.call("RELEASE")
        self._lifecycle_events.append("wse_released")
        self._npu_memory_provider.release()
        self._lease.release()
        self._lifecycle_events.extend(("attention_released", "lease_released"))
        self._wse_evidence = self._rpc.close_worker()
        return dict(self._wse_evidence)

    def abort(self) -> None:
        self._rpc.abort()
        if self._npu_memory_provider is not None:
            self._npu_memory_provider.release()

    def evidence(self) -> dict[str, Any]:
        return {
            "control": self._rpc.evidence(),
            "lease_id": self._lease.lease_id if self._lease is not None else None,
            "lease_state": self._lease.state.value if self._lease is not None else None,
            "lifecycle_events": list(self._lifecycle_events),
            "memory": self._npu_memory_provider.evidence() if self._npu_memory_provider is not None else None,
            "run_id": self.run_id,
        }


def build_npu_communication(
    provider: CrossDeviceMemoryProvider,
    rpc: HostControlRpc,
) -> tuple[NpuCommunicationBinding, WindowManifest]:
    """Build NPU-local mappings and return its PyPTO binding plus manifest."""

    local_manifest = provider.allocate_shared_window()
    response = rpc.call("BUILD_COMMUNICATION", manifest=local_manifest.to_dict())
    peer_manifest = WindowManifest.from_dict(response["manifest"])
    provider.attach_peer(peer_manifest)
    binding = _to_npu_binding(provider.binding(peer_manifest))
    return binding, local_manifest


def build_wse_communication(
    provider: CrossDeviceMemoryProvider,
    peer_manifest_payload: dict[str, Any],
) -> tuple[WseCommunicationBinding, WindowManifest]:
    """Build WSE-local mappings and return its PyPTO binding plus manifest."""

    peer_manifest = WindowManifest.from_dict(peer_manifest_payload)
    local_manifest = provider.allocate_shared_window()
    provider.attach_peer(peer_manifest)
    binding = _to_wse_binding(provider.binding(peer_manifest))
    return binding, local_manifest


def create_npu_memory_provider(
    *,
    device_id: int,
    generation: int,
) -> CrossDeviceMemoryProvider:
    """Create the NPU Host provider with the NPU shared-window contract."""

    return AscendVmmMemoryProvider(
        role=EndpointRole.ATTENTION,
        endpoint_id="attention",
        device_id=device_id,
        generation=generation,
        logical_bytes=NPU_SHARED_WINDOW_BYTES,
    )


def create_wse_memory_provider(
    *,
    device_id: int,
    generation: int,
) -> CrossDeviceMemoryProvider:
    """Create the WSE Host provider with the WSE shared-window contract."""

    return AscendVmmMemoryProvider(
        role=EndpointRole.WSE,
        endpoint_id="wse",
        device_id=device_id,
        generation=generation,
        logical_bytes=WSE_SHARED_WINDOW_BYTES,
    )


def _to_npu_binding(binding: CrossDeviceMemoryBinding) -> NpuCommunicationBinding:
    return NpuCommunicationBinding(
        binding.generation,
        binding.local.address,
        binding.local.logical_bytes,
        binding.peer.address,
        binding.peer.logical_bytes,
    )


def _to_wse_binding(binding: CrossDeviceMemoryBinding) -> WseCommunicationBinding:
    return WseCommunicationBinding(
        binding.generation,
        binding.local.address,
        binding.local.logical_bytes,
        binding.peer.address,
        binding.peer.logical_bytes,
    )
