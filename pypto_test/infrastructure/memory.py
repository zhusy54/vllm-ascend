# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""External cross-Device directly accessible memory provisioning.

This module owns ACL initialization/device selection and the full VMM
allocation, handle exchange, import, VA mapping, peer-access, and release
lifecycle.  It injects only process-local mapped addresses into pseudo-PyPTO;
it provides no H2D/D2H or kernel-launch capability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from pypto_test.pseudo_pypto.communication import EXPECTED_LAYOUT_HASH, EndpointRole
from tools.pypto_wse_validation.acl_vmm import AclVmmRuntime, VmmExport


class CrossDeviceMemoryError(RuntimeError):
    """Raised when a shared-memory provider violates its lifecycle contract."""


@dataclass(frozen=True)
class WindowManifest:
    """Serializable opaque allocation identity exchanged by Host control RPC."""

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
            raise CrossDeviceMemoryError("manifest generation mismatch")
        if not self.endpoint_id or not self.buffer_id:
            raise CrossDeviceMemoryError("manifest identifiers must be non-empty")
        if self.device_id < 0 or self.logical_bytes <= 0 or self.mapping_bytes < self.logical_bytes:
            raise CrossDeviceMemoryError("manifest Device or size is invalid")
        if self.shareable_handle <= 0:
            raise CrossDeviceMemoryError("manifest shareable handle is invalid")
        if self.layout_hash != EXPECTED_LAYOUT_HASH:
            raise CrossDeviceMemoryError("manifest layout hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["role"] = self.role.value
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WindowManifest:
        try:
            values = dict(payload)
            values["role"] = EndpointRole(values["role"])
            manifest = cls(**values)
        except (KeyError, TypeError, ValueError) as exc:
            raise CrossDeviceMemoryError("invalid window manifest") from exc
        manifest.validate(generation=manifest.generation)
        return manifest

    def evidence(self) -> dict[str, Any]:
        result = self.to_dict()
        result.pop("shareable_handle")
        result["opaque_handle"] = "PRESENT_REDACTED"
        return result


@dataclass(frozen=True)
class ProcessLocalWindow:
    """One mapped Device VA, meaningful only inside the receiving process."""

    owner: EndpointRole
    buffer_id: str
    generation: int
    address: int
    logical_bytes: int
    mapping_bytes: int


@dataclass(frozen=True)
class CrossDeviceMemoryBinding:
    """External resource injection: local and imported peer mappings only."""

    generation: int
    local: ProcessLocalWindow
    peer: ProcessLocalWindow


@runtime_checkable
class CrossDeviceMemoryProvider(Protocol):
    """Capability required from an external shared-memory implementation."""

    def initialize_host(self) -> None: ...

    def allocate_shared_window(self) -> WindowManifest: ...

    def attach_peer(self, peer: WindowManifest) -> None: ...

    def binding(self, peer: WindowManifest) -> CrossDeviceMemoryBinding: ...

    def release(self) -> None: ...

    def evidence(self) -> dict[str, Any]: ...


@dataclass
class MemoryOperationAudit:
    operations: list[dict[str, Any]] = field(default_factory=list)

    def record(self, *, actor: str, operation: str, buffer_id: str) -> None:
        self.operations.append({"actor": actor, "buffer_id": buffer_id, "operation": operation})


class AscendVmmMemoryProvider:
    """AscendCL/VMM implementation of ``CrossDeviceMemoryProvider``."""

    def __init__(
        self,
        *,
        role: EndpointRole,
        endpoint_id: str,
        device_id: int,
        generation: int,
        logical_bytes: int,
        runtime: Any | None = None,
        audit: MemoryOperationAudit | None = None,
    ) -> None:
        self.role = role
        self.endpoint_id = endpoint_id
        self.device_id = device_id
        self.generation = generation
        self.logical_bytes = logical_bytes
        self._runtime = runtime if runtime is not None else AclVmmRuntime(device_id)
        self._audit = audit if audit is not None else MemoryOperationAudit()
        self._local: Any | None = None
        self._peer: Any | None = None
        self._initialized = False
        self._closed = False
        self._allocated_window_count = 0
        self._mapping_count = 0

    @property
    def local_buffer_id(self) -> str:
        return "npu-shared-window" if self.role is EndpointRole.ATTENTION else "wse-shared-window"

    def initialize_host(self) -> None:
        if self._closed or self._initialized:
            raise CrossDeviceMemoryError("provider cannot initialize in its current state")
        self._runtime.initialize()
        self._initialized = True
        self._audit.record(actor=self.__class__.__name__, operation="runtime_initialize", buffer_id=self.endpoint_id)

    def allocate_shared_window(self) -> WindowManifest:
        if self._closed or not self._initialized or self._local is not None:
            raise CrossDeviceMemoryError("shared window cannot allocate in its current state")
        self._local = self._runtime.allocate_window(self.logical_bytes)
        self._allocated_window_count += 1
        self._mapping_count += 1
        self._audit.record(actor=self.__class__.__name__, operation="allocate", buffer_id=self.local_buffer_id)
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

    def attach_peer(self, peer: WindowManifest) -> None:
        peer.validate(generation=self.generation)
        if self._local is None or self._peer is not None or peer.role is self.role:
            raise CrossDeviceMemoryError("provider cannot attach peer in its current state")
        exported = VmmExport(peer.device_id, peer.mapping_bytes, peer.shareable_handle)
        self._peer = self._runtime.import_window(exported, peer_device_id=peer.device_id)
        self._mapping_count += 1
        self._audit.record(actor=self.__class__.__name__, operation="attach", buffer_id=peer.buffer_id)

    def binding(self, peer: WindowManifest) -> CrossDeviceMemoryBinding:
        if self._local is None or self._peer is None:
            raise CrossDeviceMemoryError("local and peer mappings must both exist")
        return CrossDeviceMemoryBinding(
            generation=self.generation,
            local=ProcessLocalWindow(
                self.role,
                self.local_buffer_id,
                self.generation,
                self._local.address,
                self.logical_bytes,
                self._local.mapping_bytes,
            ),
            peer=ProcessLocalWindow(
                peer.role,
                peer.buffer_id,
                self.generation,
                self._peer.address,
                peer.logical_bytes,
                self._peer.mapping_bytes,
            ),
        )

    def release(self) -> None:
        if self._closed:
            return
        if self._peer is not None:
            self._peer.close()
            self._audit.record(actor=self.__class__.__name__, operation="detach", buffer_id="peer-shared-window")
            self._peer = None
        if self._local is not None:
            self._local.close()
            self._audit.record(actor=self.__class__.__name__, operation="free", buffer_id=self.local_buffer_id)
            self._local = None
        self._runtime.close()
        self._audit.record(actor=self.__class__.__name__, operation="runtime_close", buffer_id=self.endpoint_id)
        self._closed = True

    def evidence(self) -> dict[str, Any]:
        return {
            "allocated_window_count": self._allocated_window_count,
            "audit": [dict(item) for item in self._audit.operations],
            "device_id": self.device_id,
            "endpoint_id": self.endpoint_id,
            "generation": self.generation,
            "live_mapping_count": int(self._local is not None) + int(self._peer is not None),
            "live_owned_window_count": int(self._local is not None),
            "mapping_count": self._mapping_count,
            "role": self.role.value,
        }
