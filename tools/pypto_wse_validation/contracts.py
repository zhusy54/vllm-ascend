# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Frozen stage-0 contracts for the PyPTO WSE NPU-surrogate profile."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum, IntEnum, IntFlag
from typing import Any, ClassVar

PROTOCOL_VERSION = 1
SLOT_COUNT = 2
MAX_PAYLOAD_BYTES = 1024 * 1024
MAX_INFLIGHT = SLOT_COUNT
SUBMISSION_ENTRY_BYTES = 48
COMPLETION_ENTRY_BYTES = 40
U32_MAX = (1 << 32) - 1
U64_MAX = (1 << 64) - 1
I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1


class ContractError(ValueError):
    """Raised when a stage-0 wire or manifest contract is violated."""


class EndpointRole(str, Enum):
    ATTENTION = "ATTENTION"
    WSE_SURROGATE = "WSE_SURROGATE"


class DeviceKind(str, Enum):
    ASCEND_NPU = "ASCEND_NPU"


class TransportScope(str, Enum):
    HOST_LOCAL = "HOST_LOCAL"
    NETWORK_REMOTE = "NETWORK_REMOTE"
    SIMULATION = "SIMULATION"


class AccessFlag(IntFlag):
    READ = 1
    WRITE = 2


class PatternKind(IntEnum):
    HASH_XOR_SEQUENCE_LOW8 = 1


class SlotState(str, Enum):
    FREE = "FREE"
    RESERVED = "RESERVED"
    INPUT_WRITTEN = "INPUT_WRITTEN"
    COMMAND_PUBLISHED = "COMMAND_PUBLISHED"
    REMOTE_RUNNING = "REMOTE_RUNNING"
    OUTPUT_WRITTEN = "OUTPUT_WRITTEN"
    COMPLETION_PUBLISHED = "COMPLETION_PUBLISHED"
    VALIDATED = "VALIDATED"


class PublishEvent(str, Enum):
    INPUT_WRITE = "INPUT_WRITE"
    INPUT_FENCE = "INPUT_FENCE"
    COMMAND_PUBLISH = "COMMAND_PUBLISH"
    OUTPUT_WRITE = "OUTPUT_WRITE"
    OUTPUT_FENCE = "OUTPUT_FENCE"
    COMPLETION_PUBLISH = "COMPLETION_PUBLISH"


_SLOT_TRANSITIONS: Mapping[SlotState, SlotState] = {
    SlotState.FREE: SlotState.RESERVED,
    SlotState.RESERVED: SlotState.INPUT_WRITTEN,
    SlotState.INPUT_WRITTEN: SlotState.COMMAND_PUBLISHED,
    SlotState.COMMAND_PUBLISHED: SlotState.REMOTE_RUNNING,
    SlotState.REMOTE_RUNNING: SlotState.OUTPUT_WRITTEN,
    SlotState.OUTPUT_WRITTEN: SlotState.COMPLETION_PUBLISHED,
    SlotState.COMPLETION_PUBLISHED: SlotState.VALIDATED,
    SlotState.VALIDATED: SlotState.FREE,
}

_LAYOUT_SPEC: Mapping[str, Any] = {
    "completion_entry_bytes": COMPLETION_ENTRY_BYTES,
    "completion_ring_entries": SLOT_COUNT,
    "input_slot_bytes": MAX_PAYLOAD_BYTES,
    "input_slots": SLOT_COUNT,
    "max_inflight": MAX_INFLIGHT,
    "output_slot_bytes": MAX_PAYLOAD_BYTES,
    "output_slots": SLOT_COUNT,
    "submission_entry_bytes": SUBMISSION_ENTRY_BYTES,
    "submission_ring_entries": SLOT_COUNT,
}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


EXPECTED_LAYOUT_HASH = hashlib.sha256(_canonical_json(_LAYOUT_SPEC).encode()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _require_int_range(name: str, value: int, lower: int, upper: int) -> None:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer")
    _require(lower <= value <= upper, f"{name} must be in [{lower}, {upper}], got {value}")


def _enum_value(enum_type: type[Enum], value: object, field: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"invalid {field}: {value!r}") from exc


@dataclass(frozen=True)
class RemoteMemoryHandle:
    owner_endpoint_id: str
    buffer_id: str
    generation: int
    size_bytes: int
    alignment: int
    access_flags: int
    handle_kind: str
    opaque_handle: str

    def validate(self) -> None:
        _require(bool(self.owner_endpoint_id), "handle owner_endpoint_id must be non-empty")
        _require(bool(self.buffer_id), "handle buffer_id must be non-empty")
        _require_int_range("handle generation", self.generation, 1, U64_MAX)
        _require_int_range("handle size_bytes", self.size_bytes, 1, U64_MAX)
        _require_int_range("handle alignment", self.alignment, 1, U32_MAX)
        _require(self.alignment & (self.alignment - 1) == 0, "handle alignment must be a power of two")
        _require_int_range("handle access_flags", self.access_flags, 1, int(AccessFlag.READ | AccessFlag.WRITE))
        _require(bool(self.handle_kind), "handle handle_kind must be non-empty")
        _require(bool(self.opaque_handle), "handle opaque_handle must be non-empty")

    def allows(self, flag: AccessFlag) -> bool:
        return bool(AccessFlag(self.access_flags) & flag)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RemoteMemoryHandle:
        try:
            handle = cls(**dict(value))
        except TypeError as exc:
            raise ContractError(f"invalid remote handle fields: {exc}") from exc
        handle.validate()
        return handle


@dataclass(frozen=True)
class CommunicationManifest:
    protocol_version: int
    run_id: str
    endpoint_id: str
    generation: int
    role: EndpointRole
    device_kind: DeviceKind
    transport_kind: str
    transport_scope: TransportScope
    capabilities: tuple[str, ...]
    layout_hash: str
    slot_count: int
    slot_bytes: int
    max_inflight: int
    input_window: RemoteMemoryHandle | None = None
    output_window: RemoteMemoryHandle | None = None
    submission_queue: RemoteMemoryHandle | None = None
    completion_queue: RemoteMemoryHandle | None = None

    _HANDLE_FIELDS: ClassVar[tuple[str, ...]] = (
        "input_window",
        "output_window",
        "submission_queue",
        "completion_queue",
    )

    def validate(self) -> None:
        _require(self.protocol_version == PROTOCOL_VERSION, f"unsupported protocol_version {self.protocol_version}")
        _require(bool(self.run_id), "run_id must be non-empty")
        _require(bool(self.endpoint_id), "endpoint_id must be non-empty")
        _require_int_range("generation", self.generation, 1, U64_MAX)
        _enum_value(EndpointRole, self.role, "role")
        _enum_value(DeviceKind, self.device_kind, "device_kind")
        _enum_value(TransportScope, self.transport_scope, "transport_scope")
        _require(bool(self.transport_kind), "transport_kind must be non-empty")
        _require(len(set(self.capabilities)) == len(self.capabilities), "capabilities must be unique")
        _require(all(isinstance(item, str) and item for item in self.capabilities), "capabilities must be strings")
        _require(self.layout_hash == EXPECTED_LAYOUT_HASH, "layout_hash does not match the frozen layout")
        _require(self.slot_count == SLOT_COUNT, f"slot_count must be {SLOT_COUNT}")
        _require(self.slot_bytes == MAX_PAYLOAD_BYTES, f"slot_bytes must be {MAX_PAYLOAD_BYTES}")
        _require(self.max_inflight == MAX_INFLIGHT, f"max_inflight must be {MAX_INFLIGHT}")
        _require(self.transport_scope is TransportScope.SIMULATION, "NPU surrogate profile requires SIMULATION scope")
        _require(self.device_kind is DeviceKind.ASCEND_NPU, "NPU surrogate endpoint must use ASCEND_NPU")

        handles = {name: getattr(self, name) for name in self._HANDLE_FIELDS}
        expected = (
            {"output_window", "completion_queue"}
            if self.role is EndpointRole.ATTENTION
            else {"input_window", "submission_queue"}
        )
        present = {name for name, handle in handles.items() if handle is not None}
        _require(present == expected, f"{self.role.value} must export {sorted(expected)}, got {sorted(present)}")
        for name, handle in handles.items():
            if handle is None:
                continue
            handle.validate()
            _require(handle.owner_endpoint_id == self.endpoint_id, f"{name} owner does not match endpoint")
            _require(handle.generation == self.generation, f"{name} generation does not match manifest")
            _require(handle.allows(AccessFlag.WRITE), f"{name} does not permit peer writes")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["role"] = self.role.value
        result["device_kind"] = self.device_kind.value
        result["transport_scope"] = self.transport_scope.value
        result["capabilities"] = list(self.capabilities)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CommunicationManifest:
        data = dict(value)
        try:
            data["role"] = _enum_value(EndpointRole, data.get("role"), "role")
            data["device_kind"] = _enum_value(DeviceKind, data.get("device_kind"), "device_kind")
            data["transport_scope"] = _enum_value(TransportScope, data.get("transport_scope"), "transport_scope")
            data["capabilities"] = tuple(data.get("capabilities", ()))
            for name in cls._HANDLE_FIELDS:
                raw = data.get(name)
                data[name] = None if raw is None else RemoteMemoryHandle.from_dict(raw)
            manifest = cls(**data)
        except TypeError as exc:
            raise ContractError(f"invalid manifest fields: {exc}") from exc
        manifest.validate()
        return manifest

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


def validate_manifest_pair(left: CommunicationManifest, right: CommunicationManifest) -> None:
    left.validate()
    right.validate()
    _require(left.endpoint_id != right.endpoint_id, "endpoint ids must be distinct")
    _require(left.run_id == right.run_id, "manifest run_id mismatch")
    _require(left.generation == right.generation, "manifest generation mismatch")
    _require(left.layout_hash == right.layout_hash, "manifest layout mismatch")
    _require(left.transport_kind == right.transport_kind, "manifest transport_kind mismatch")
    _require(left.transport_scope == right.transport_scope, "manifest transport_scope mismatch")
    _require({left.role, right.role} == set(EndpointRole), "manifest roles must be Attention and WSE surrogate")


@dataclass(frozen=True)
class TestDescriptor:
    __test__: ClassVar[bool] = False

    protocol_version: int
    flags: int
    generation: int
    sequence_id: int
    input_slot: int
    output_slot: int
    payload_bytes: int
    pattern_kind: int
    expected_checksum: int

    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<IIQQIIIIQ")

    def validate(self) -> None:
        _require(self.protocol_version == PROTOCOL_VERSION, "descriptor protocol_version mismatch")
        _require_int_range("descriptor flags", self.flags, 0, U32_MAX)
        _require_int_range("descriptor generation", self.generation, 1, U64_MAX)
        _require_int_range("descriptor sequence_id", self.sequence_id, 1, U64_MAX)
        _require_int_range("descriptor input_slot", self.input_slot, 0, SLOT_COUNT - 1)
        _require_int_range("descriptor output_slot", self.output_slot, 0, SLOT_COUNT - 1)
        _require_int_range("descriptor payload_bytes", self.payload_bytes, 1, MAX_PAYLOAD_BYTES)
        _enum_value(PatternKind, self.pattern_kind, "descriptor pattern_kind")
        _require_int_range("descriptor expected_checksum", self.expected_checksum, 0, U64_MAX)

    def to_bytes(self) -> bytes:
        self.validate()
        return self._STRUCT.pack(*asdict(self).values())

    @classmethod
    def from_bytes(cls, payload: bytes) -> TestDescriptor:
        _require(len(payload) == cls._STRUCT.size, f"descriptor must be {cls._STRUCT.size} bytes")
        descriptor = cls(*cls._STRUCT.unpack(payload))
        descriptor.validate()
        return descriptor


@dataclass(frozen=True)
class TestCompletion:
    __test__: ClassVar[bool] = False

    protocol_version: int
    status: int
    generation: int
    sequence_id: int
    output_slot: int
    output_bytes: int
    output_checksum: int

    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<IiQQIIQ")

    def validate(self) -> None:
        _require(self.protocol_version == PROTOCOL_VERSION, "completion protocol_version mismatch")
        _require_int_range("completion status", self.status, I32_MIN, I32_MAX)
        _require_int_range("completion generation", self.generation, 1, U64_MAX)
        _require_int_range("completion sequence_id", self.sequence_id, 1, U64_MAX)
        _require_int_range("completion output_slot", self.output_slot, 0, SLOT_COUNT - 1)
        _require_int_range("completion output_bytes", self.output_bytes, 0, MAX_PAYLOAD_BYTES)
        _require_int_range("completion output_checksum", self.output_checksum, 0, U64_MAX)
        if self.status == 0:
            _require(self.output_bytes > 0, "successful completion must contain output bytes")

    def to_bytes(self) -> bytes:
        self.validate()
        return self._STRUCT.pack(*asdict(self).values())

    @classmethod
    def from_bytes(cls, payload: bytes) -> TestCompletion:
        _require(len(payload) == cls._STRUCT.size, f"completion must be {cls._STRUCT.size} bytes")
        completion = cls(*cls._STRUCT.unpack(payload))
        completion.validate()
        return completion


def validate_slot_transition(current: SlotState, next_state: SlotState) -> None:
    expected = _SLOT_TRANSITIONS.get(current)
    _require(expected is next_state, f"invalid slot transition {current.value} -> {next_state.value}")


def validate_publish_order(events: Iterable[PublishEvent]) -> None:
    observed = tuple(events)
    expected = (
        PublishEvent.INPUT_WRITE,
        PublishEvent.INPUT_FENCE,
        PublishEvent.COMMAND_PUBLISH,
        PublishEvent.OUTPUT_WRITE,
        PublishEvent.OUTPUT_FENCE,
        PublishEvent.COMPLETION_PUBLISH,
    )
    _require(observed == expected, f"invalid publish order: {[item.value for item in observed]}")
