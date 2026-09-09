# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Bootstrap-owned device memory and host-local control protocol."""

from __future__ import annotations

import json
import socket
import struct
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pypto_test.contracts import (
    DEFAULT_LAYOUT,
    EXPECTED_LAYOUT_HASH,
    NPU_WINDOW_BYTES,
    PROTOCOL_VERSION,
    WSE_WINDOW_BYTES,
    BootstrapLease,
    BorrowedWindowView,
    ContractError,
    DeviceExecutionPort,
    EndpointBundle,
    EndpointRole,
    KernelSession,
    LeaseState,
    MemoryOperationAudit,
)
from tools.pypto_wse_validation.acl_kernel import AclDeviceKernel
from tools.pypto_wse_validation.acl_vmm import AclVmmRuntime, VmmExport

MAX_CONTROL_FRAME_BYTES = 64 * 1024
_FRAME_LENGTH = struct.Struct("!I")
_FORBIDDEN_CONTROL_KEYS = frozenset({"input", "output", "payload", "tensor", "token"})
_MESSAGE_TYPES = frozenset(
    {
        "MANIFEST",
        "ATTACHED",
        "START",
        "READY",
        "HEALTH",
        "HEALTH_REPLY",
        "DRAIN",
        "DRAINED",
        "CLOSE",
        "CLOSED",
        "RELEASE",
        "RELEASED",
        "ERROR",
    }
)


class BootstrapError(RuntimeError):
    """Raised when bootstrap ownership or protocol rules are violated."""


class _OwnedWindow(Protocol):
    address: int
    logical_bytes: int
    mapping_bytes: int
    export: VmmExport

    def close(self) -> None: ...


class _ImportedWindow(Protocol):
    address: int
    mapping_bytes: int

    def close(self) -> None: ...


class _Runtime(Protocol):
    device_id: int

    def initialize(self) -> None: ...

    def allocate_window(self, logical_bytes: int) -> _OwnedWindow: ...

    def import_window(self, exported: VmmExport, *, peer_device_id: int) -> _ImportedWindow: ...

    def copy_host_to_device(self, destination: int, payload: bytes) -> None: ...

    def copy_device_to_host(self, source: int, size: int) -> bytes: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class WindowManifest:
    endpoint_id: str
    role: EndpointRole
    device_id: int
    generation: int
    buffer_id: str
    logical_bytes: int
    mapping_bytes: int
    shareable_handle: int
    layout_hash: str = EXPECTED_LAYOUT_HASH

    def validate(self, *, generation: int) -> None:
        if self.generation != generation:
            raise BootstrapError("manifest generation mismatch")
        if not self.endpoint_id or not self.buffer_id:
            raise BootstrapError("manifest identifiers must be non-empty")
        if self.device_id < 0:
            raise BootstrapError("manifest device_id must be non-negative")
        if self.logical_bytes <= 0 or self.mapping_bytes < self.logical_bytes:
            raise BootstrapError("manifest window size is invalid")
        if self.shareable_handle <= 0:
            raise BootstrapError("manifest shareable handle is invalid")
        if self.layout_hash != EXPECTED_LAYOUT_HASH:
            raise BootstrapError("manifest layout hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["role"] = self.role.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WindowManifest:
        try:
            values = dict(payload)
            values["role"] = EndpointRole(values["role"])
            manifest = cls(**values)
        except (KeyError, TypeError, ValueError) as exc:
            raise BootstrapError("invalid window manifest") from exc
        manifest.validate(generation=manifest.generation)
        return manifest

    def evidence(self) -> dict[str, Any]:
        result = self.to_dict()
        result.pop("shareable_handle")
        result["opaque_handle"] = "PRESENT_REDACTED"
        return result


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise BootstrapError("control connection closed during frame receive")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class ProxyControlChannel:
    """Length-prefixed JSON channel restricted to lifecycle control data."""

    def __init__(self, connection: socket.socket, *, run_id: str, generation: int) -> None:
        self.connection = connection
        self.run_id = run_id
        self.generation = generation
        self.sent_messages: Counter[str] = Counter()
        self.received_messages: Counter[str] = Counter()
        self.sent_bytes = 0
        self.received_bytes = 0

    def send(self, message_type: str, role: EndpointRole, **fields: Any) -> None:
        if message_type not in _MESSAGE_TYPES:
            raise BootstrapError(f"unsupported control message: {message_type}")
        forbidden = _find_forbidden_keys(fields)
        if forbidden:
            raise BootstrapError(f"control message contains data-plane keys: {sorted(forbidden)}")
        message = {
            "generation": self.generation,
            "protocol_version": PROTOCOL_VERSION,
            "role": role.value,
            "run_id": self.run_id,
            "type": message_type,
            **fields,
        }
        encoded = json.dumps(message, separators=(",", ":"), sort_keys=True).encode()
        if len(encoded) > MAX_CONTROL_FRAME_BYTES:
            raise BootstrapError("control frame is too large")
        self.connection.sendall(_FRAME_LENGTH.pack(len(encoded)) + encoded)
        self.sent_messages[message_type] += 1
        self.sent_bytes += len(encoded)

    def receive(self, expected_type: str, expected_role: EndpointRole) -> dict[str, Any]:
        message = self.receive_any(expected_role)
        if message.get("type") != expected_type:
            raise BootstrapError(f"control type mismatch: expected {expected_type}, received {message.get('type')}")
        return message

    def receive_any(self, expected_role: EndpointRole) -> dict[str, Any]:
        (size,) = _FRAME_LENGTH.unpack(_receive_exact(self.connection, _FRAME_LENGTH.size))
        if size <= 0 or size > MAX_CONTROL_FRAME_BYTES:
            raise BootstrapError("invalid control frame size")
        encoded = _receive_exact(self.connection, size)
        try:
            message = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BootstrapError("invalid control JSON") from exc
        if not isinstance(message, dict):
            raise BootstrapError("control frame must be an object")
        expected = {
            "generation": self.generation,
            "protocol_version": PROTOCOL_VERSION,
            "role": expected_role.value,
            "run_id": self.run_id,
        }
        for key, value in expected.items():
            if message.get(key) != value:
                raise BootstrapError(f"control {key} mismatch")
        forbidden = _find_forbidden_keys(message)
        if forbidden:
            raise BootstrapError(f"control message contains data-plane keys: {sorted(forbidden)}")
        message_type = message.get("type")
        if message_type not in _MESSAGE_TYPES:
            raise BootstrapError(f"unsupported control message: {message_type}")
        self.received_messages[str(message_type)] += 1
        self.received_bytes += size
        return message

    def evidence(self) -> dict[str, Any]:
        return {
            "received_bytes": self.received_bytes,
            "received_messages": dict(sorted(self.received_messages.items())),
            "sent_bytes": self.sent_bytes,
            "sent_messages": dict(sorted(self.sent_messages.items())),
        }


def _find_forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_CONTROL_KEYS:
                found.add(normalized)
            found.update(_find_forbidden_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_find_forbidden_keys(child))
    return found


class BorrowedDeviceExecutionPort(DeviceExecutionPort):
    """Narrow execution capability; it intentionally has no VMM lifecycle API."""

    def __init__(
        self,
        runtime: _Runtime,
        *,
        generation: int,
        permitted_ranges: tuple[tuple[int, int], ...],
    ) -> None:
        self._runtime = runtime
        self._generation = generation
        self._permitted_ranges = permitted_ranges
        self._valid = True

    @property
    def generation(self) -> int:
        return self._generation

    def _require_range(self, address: int, size: int) -> None:
        if not self._valid:
            raise BootstrapError("execution port is invalidated")
        in_range = any(
            start <= address and address + size <= start + length for start, length in self._permitted_ranges
        )
        if size <= 0 or not in_range:
            raise BootstrapError("device access is outside the borrowed windows")

    def copy_host_to_device(self, destination: int, payload: bytes) -> None:
        self._require_range(destination, len(payload))
        self._runtime.copy_host_to_device(destination, payload)

    def copy_device_to_host(self, source: int, size: int) -> bytes:
        self._require_range(source, size)
        return self._runtime.copy_device_to_host(source, size)

    def launch_kernel(self, binary_path: Path, arguments: Any) -> KernelSession:
        if not self._valid:
            raise BootstrapError("execution port is invalidated")
        kernel = AclDeviceKernel(self._runtime, binary_path)  # type: ignore[arg-type]
        kernel.launch(arguments)
        return kernel

    def invalidate(self) -> None:
        self._valid = False


class DeviceMemoryManager:
    """The sole owner of one endpoint's runtime and communication windows."""

    def __init__(
        self,
        *,
        role: EndpointRole,
        endpoint_id: str,
        device_id: int,
        generation: int,
        logical_bytes: int,
        audit: MemoryOperationAudit | None = None,
        runtime: _Runtime | None = None,
    ) -> None:
        self.role = role
        self.endpoint_id = endpoint_id
        self.device_id = device_id
        self.generation = generation
        self.logical_bytes = logical_bytes
        self.audit = audit if audit is not None else MemoryOperationAudit()
        self._runtime = runtime if runtime is not None else AclVmmRuntime(device_id)
        self._local: _OwnedWindow | None = None
        self._peer: _ImportedWindow | None = None
        self._port: BorrowedDeviceExecutionPort | None = None
        self._closed = False
        self._allocated_window_count = 0
        self._mapping_count = 0

    def initialize(self) -> WindowManifest:
        if self._closed or self._local is not None:
            raise BootstrapError("memory manager cannot initialize in its current state")
        self._runtime.initialize()
        self.audit.record(actor=self.__class__.__name__, operation="runtime_initialize", buffer_id=self.endpoint_id)
        self._local = self._runtime.allocate_window(self.logical_bytes)
        self._allocated_window_count += 1
        self._mapping_count += 1
        self.audit.record(actor=self.__class__.__name__, operation="allocate", buffer_id=self.local_buffer_id)
        export = self._local.export
        return WindowManifest(
            endpoint_id=self.endpoint_id,
            role=self.role,
            device_id=self.device_id,
            generation=self.generation,
            buffer_id=self.local_buffer_id,
            logical_bytes=self.logical_bytes,
            mapping_bytes=export.mapping_bytes,
            shareable_handle=export.shareable_handle,
        )

    @property
    def local_buffer_id(self) -> str:
        return "npu-window" if self.role is EndpointRole.ATTENTION else "wse-window"

    def attach(self, peer: WindowManifest) -> None:
        peer.validate(generation=self.generation)
        if self._local is None or self._peer is not None:
            raise BootstrapError("memory manager cannot attach in its current state")
        if peer.role is self.role:
            raise BootstrapError("cannot attach a manifest with the local role")
        exported = VmmExport(peer.device_id, peer.mapping_bytes, peer.shareable_handle)
        self._peer = self._runtime.import_window(exported, peer_device_id=peer.device_id)
        self._mapping_count += 1
        self.audit.record(actor=self.__class__.__name__, operation="attach", buffer_id=peer.buffer_id)

    def borrowed_resources(
        self, peer: WindowManifest
    ) -> tuple[BorrowedWindowView, BorrowedWindowView, DeviceExecutionPort]:
        if self._local is None or self._peer is None:
            raise BootstrapError("both local and peer windows must be ready")
        local_view = BorrowedWindowView(
            self.role,
            self.local_buffer_id,
            self.generation,
            self._local.address,
            self.logical_bytes,
            self._local.mapping_bytes,
        )
        peer_view = BorrowedWindowView(
            peer.role,
            peer.buffer_id,
            self.generation,
            self._peer.address,
            peer.logical_bytes,
            self._peer.mapping_bytes,
        )
        self._port = BorrowedDeviceExecutionPort(
            self._runtime,
            generation=self.generation,
            permitted_ranges=(
                (local_view.address, local_view.mapping_bytes),
                (peer_view.address, peer_view.mapping_bytes),
            ),
        )
        return local_view, peer_view, self._port

    def release(self) -> None:
        if self._closed:
            return
        if self._port is not None:
            self._port.invalidate()
        if self._peer is not None:
            self._peer.close()
            self.audit.record(actor=self.__class__.__name__, operation="detach", buffer_id="peer-window")
            self._peer = None
        if self._local is not None:
            self._local.close()
            self.audit.record(actor=self.__class__.__name__, operation="free", buffer_id=self.local_buffer_id)
            self._local = None
        self._runtime.close()
        self.audit.record(actor=self.__class__.__name__, operation="runtime_close", buffer_id=self.endpoint_id)
        self._closed = True

    def evidence(self) -> dict[str, Any]:
        return {
            "audit": self.audit.to_dict(),
            "device_id": self.device_id,
            "endpoint_id": self.endpoint_id,
            "generation": self.generation,
            "allocated_window_count": self._allocated_window_count,
            "live_mapping_count": int(self._local is not None) + int(self._peer is not None),
            "live_owned_window_count": int(self._local is not None),
            "mapping_count": self._mapping_count,
            "role": self.role.value,
        }


class NpuDeviceMemoryManager(DeviceMemoryManager):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(role=EndpointRole.ATTENTION, logical_bytes=NPU_WINDOW_BYTES, **kwargs)


class SurrogateDeviceMemoryManager(DeviceMemoryManager):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(role=EndpointRole.WSE_SURROGATE, logical_bytes=WSE_WINDOW_BYTES, **kwargs)


class RemoteWseServiceControl:
    def __init__(self, channel: ProxyControlChannel) -> None:
        self._channel = channel
        self._closed = False

    def start(self) -> dict[str, Any]:
        self._channel.send("START", EndpointRole.ATTENTION)
        return self._channel.receive("READY", EndpointRole.WSE_SURROGATE)

    def health(self) -> dict[str, Any]:
        self._channel.send("HEALTH", EndpointRole.ATTENTION)
        return self._channel.receive("HEALTH_REPLY", EndpointRole.WSE_SURROGATE)

    def drain(self) -> dict[str, Any]:
        self._channel.send("DRAIN", EndpointRole.ATTENTION)
        return self._channel.receive("DRAINED", EndpointRole.WSE_SURROGATE)

    def close(self) -> None:
        if self._closed:
            return
        self._channel.send("CLOSE", EndpointRole.ATTENTION)
        self._channel.receive("CLOSED", EndpointRole.WSE_SURROGATE)
        self._closed = True


@dataclass(frozen=True)
class SurrogateBootstrapResources:
    local_window: BorrowedWindowView
    npu_peer_window: BorrowedWindowView
    execution_port: DeviceExecutionPort
    peer_manifest: WindowManifest


class BootstrapManager:
    """Attention-side coordinator that lends, but never transfers, memory ownership."""

    def __init__(
        self,
        *,
        channel: ProxyControlChannel,
        memory_manager: NpuDeviceMemoryManager,
        run_id: str,
        generation: int,
    ) -> None:
        self.channel = channel
        self.memory_manager = memory_manager
        self.run_id = run_id
        self.generation = generation
        self._lease: BootstrapLease | None = None

    def prepare(self) -> EndpointBundle:
        local_manifest = self.memory_manager.initialize()
        self.channel.send("MANIFEST", EndpointRole.ATTENTION, manifest=local_manifest.to_dict())
        message = self.channel.receive("MANIFEST", EndpointRole.WSE_SURROGATE)
        peer_manifest = WindowManifest.from_dict(message["manifest"])
        peer_manifest.validate(generation=self.generation)
        self.memory_manager.attach(peer_manifest)
        self.channel.send("ATTACHED", EndpointRole.ATTENTION)
        self.channel.receive("ATTACHED", EndpointRole.WSE_SURROGATE)
        local, peer, port = self.memory_manager.borrowed_resources(peer_manifest)
        lease = BootstrapLease(f"lease-{uuid4().hex}", self.generation)
        lease.borrow()
        self._lease = lease
        bundle = EndpointBundle(
            generation=self.generation,
            endpoint_id=local_manifest.endpoint_id,
            backend_kind="NPU_SURROGATE",
            transport_kind="ASCEND_VMM_P2P",
            transport_scope="HOST_LOCAL",
            layout=DEFAULT_LAYOUT,
            execution_port=port,
            npu_local_window=local,
            wse_peer_window=peer,
            wse_control=RemoteWseServiceControl(self.channel),
            lease=lease,
        )
        bundle.validate()
        return bundle

    def release(self) -> None:
        if self._lease is None:
            raise BootstrapError("bootstrap resources were not prepared")
        if self._lease.state is not LeaseState.QUIESCED:
            raise BootstrapError("service must quiesce its lease before bootstrap release")
        self.channel.send("RELEASE", EndpointRole.ATTENTION)
        self.channel.receive("RELEASED", EndpointRole.WSE_SURROGATE)
        self.memory_manager.release()
        self._lease.release()

    def evidence(self) -> dict[str, Any]:
        return {
            "control": self.channel.evidence(),
            "memory": self.memory_manager.evidence(),
            "run_id": self.run_id,
        }


def accept_surrogate_bootstrap(
    *,
    channel: ProxyControlChannel,
    memory_manager: SurrogateDeviceMemoryManager,
) -> SurrogateBootstrapResources:
    local_manifest = memory_manager.initialize()
    channel.send("MANIFEST", EndpointRole.WSE_SURROGATE, manifest=local_manifest.to_dict())
    message = channel.receive("MANIFEST", EndpointRole.ATTENTION)
    peer_manifest = WindowManifest.from_dict(message["manifest"])
    memory_manager.attach(peer_manifest)
    channel.receive("ATTACHED", EndpointRole.ATTENTION)
    channel.send("ATTACHED", EndpointRole.WSE_SURROGATE)
    local, peer, port = memory_manager.borrowed_resources(peer_manifest)
    return SurrogateBootstrapResources(local, peer, port, peer_manifest)
