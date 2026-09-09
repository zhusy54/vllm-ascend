# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Contracts for the isolated fixed-ABC proxy validation prototype."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

PROTOCOL_VERSION = 1
CACHE_LINE_BYTES = 64
UINT32_BYTES = 4
MAX_PAYLOAD_BYTES = 1024 * 1024
MAX_ELEMENTS = MAX_PAYLOAD_BYTES // UINT32_BYTES
MAX_INFLIGHT = 1
DEFAULT_PROGRAM_ID = "fixed-abc-v1"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_POLL_INTERVAL_SECONDS = 0.001

NPU_INPUT_OFFSET = 0
NPU_B_OUTPUT_OFFSET = MAX_PAYLOAD_BYTES
NPU_FINAL_OUTPUT_OFFSET = 2 * MAX_PAYLOAD_BYTES
NPU_CONTROL_OFFSET = 3 * MAX_PAYLOAD_BYTES
NPU_HOST_REQUEST_SIGNAL_OFFSET = NPU_CONTROL_OFFSET
NPU_HOST_REQUEST_DESC_OFFSET = NPU_CONTROL_OFFSET + CACHE_LINE_BYTES
NPU_HOST_RESULT_SIGNAL_OFFSET = NPU_CONTROL_OFFSET + (2 * CACHE_LINE_BYTES)
NPU_HOST_RESULT_DESC_OFFSET = NPU_CONTROL_OFFSET + (3 * CACHE_LINE_BYTES)
NPU_B_COMPLETION_SIGNAL_OFFSET = NPU_CONTROL_OFFSET + (4 * CACHE_LINE_BYTES)
NPU_B_COMPLETION_DESC_OFFSET = NPU_CONTROL_OFFSET + (5 * CACHE_LINE_BYTES)
NPU_LIFECYCLE_OFFSET = NPU_CONTROL_OFFSET + (6 * CACHE_LINE_BYTES)
NPU_REPORT_OFFSET = NPU_CONTROL_OFFSET + (7 * CACHE_LINE_BYTES)
NPU_WINDOW_BYTES = NPU_CONTROL_OFFSET + (10 * CACHE_LINE_BYTES)

WSE_B_INPUT_OFFSET = 0
WSE_CONTROL_OFFSET = MAX_PAYLOAD_BYTES
WSE_B_SUBMISSION_SIGNAL_OFFSET = WSE_CONTROL_OFFSET
WSE_B_SUBMISSION_DESC_OFFSET = WSE_CONTROL_OFFSET + CACHE_LINE_BYTES
WSE_LIFECYCLE_OFFSET = WSE_CONTROL_OFFSET + (2 * CACHE_LINE_BYTES)
WSE_REPORT_OFFSET = WSE_CONTROL_OFFSET + (3 * CACHE_LINE_BYTES)
WSE_WINDOW_BYTES = WSE_CONTROL_OFFSET + (6 * CACHE_LINE_BYTES)

_U64_LINE = struct.Struct("<QQQQQQQQ")
_REPORT = struct.Struct("<" + ("Q" * 16))


class ContractError(ValueError):
    """Raised when a proxy contract is malformed or stale."""


class ServiceError(RuntimeError):
    """Raised when a proxy service operation violates its state contract."""


class EndpointRole(str, Enum):
    ATTENTION = "ATTENTION"
    WSE_SURROGATE = "WSE_SURROGATE"


class ServiceState(str, Enum):
    NEW = "NEW"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    EXECUTING = "EXECUTING"
    DRAINING = "DRAINING"
    DRAINED = "DRAINED"
    CLOSED = "CLOSED"


class LeaseState(str, Enum):
    CREATED = "CREATED"
    BORROWED = "BORROWED"
    QUIESCED = "QUIESCED"
    RELEASED = "RELEASED"


@dataclass(frozen=True)
class PseudoTask:
    name: str
    placement: str
    dependencies: tuple[str, ...] = ()


ABC_TASKS = (
    PseudoTask("A", "NPU"),
    PseudoTask("B", "WSE", ("A",)),
    PseudoTask("C", "NPU", ("B",)),
)


@dataclass(frozen=True)
class PseudoProgramSpec:
    program_id: str = DEFAULT_PROGRAM_ID
    dtype: str = "uint32"
    max_elements: int = MAX_ELEMENTS
    tasks: tuple[PseudoTask, ...] = ABC_TASKS

    def validate(self) -> None:
        if self.program_id != DEFAULT_PROGRAM_ID:
            raise ContractError(f"unsupported program_id: {self.program_id}")
        if self.dtype != "uint32":
            raise ContractError("fixed ABC program requires uint32")
        if self.max_elements != MAX_ELEMENTS:
            raise ContractError(f"max_elements must be {MAX_ELEMENTS}")
        if self.tasks != ABC_TASKS:
            raise ContractError("only A(NPU) -> B(WSE) -> C(NPU) is supported")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ProxyBufferLayout:
    npu_window_bytes: int = NPU_WINDOW_BYTES
    wse_window_bytes: int = WSE_WINDOW_BYTES
    max_payload_bytes: int = MAX_PAYLOAD_BYTES
    max_inflight: int = MAX_INFLIGHT

    @property
    def layout_hash(self) -> str:
        payload = json.dumps(asdict(self), separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def validate(self) -> None:
        if self != ProxyBufferLayout():
            raise ContractError("buffer layout does not match fixed ABC v1")


DEFAULT_LAYOUT = ProxyBufferLayout()
EXPECTED_LAYOUT_HASH = DEFAULT_LAYOUT.layout_hash


@dataclass(frozen=True)
class BorrowedWindowView:
    owner: EndpointRole
    buffer_id: str
    generation: int
    address: int
    logical_bytes: int
    mapping_bytes: int

    def validate(self, *, generation: int, minimum_bytes: int) -> None:
        if self.generation != generation:
            raise ContractError(f"{self.buffer_id} generation mismatch")
        if self.address <= 0:
            raise ContractError(f"{self.buffer_id} address must be positive")
        if self.logical_bytes < minimum_bytes or self.mapping_bytes < self.logical_bytes:
            raise ContractError(f"{self.buffer_id} is smaller than the required layout")


@runtime_checkable
class KernelSession(Protocol):
    @property
    def binary_sha256(self) -> str: ...

    @property
    def closed(self) -> bool: ...

    def synchronize(self) -> int: ...

    def close(self) -> None: ...


@runtime_checkable
class DeviceExecutionPort(Protocol):
    @property
    def generation(self) -> int: ...

    def copy_host_to_device(self, destination: int, payload: bytes) -> None: ...

    def copy_device_to_host(self, source: int, size: int) -> bytes: ...

    def launch_kernel(self, binary_path: Path, arguments: Any) -> KernelSession: ...

    def invalidate(self) -> None: ...


@runtime_checkable
class WseServiceControl(Protocol):
    def start(self) -> dict[str, Any]: ...

    def health(self) -> dict[str, Any]: ...

    def drain(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass
class BootstrapLease:
    lease_id: str
    generation: int
    state: LeaseState = LeaseState.CREATED

    def borrow(self) -> None:
        if self.state is not LeaseState.CREATED:
            raise ContractError(f"cannot borrow lease in state {self.state.value}")
        self.state = LeaseState.BORROWED

    def quiesce(self) -> None:
        if self.state not in (LeaseState.CREATED, LeaseState.BORROWED):
            raise ContractError(f"cannot quiesce lease in state {self.state.value}")
        self.state = LeaseState.QUIESCED

    def release(self) -> None:
        if self.state is not LeaseState.QUIESCED:
            raise ContractError("lease must be quiesced before release")
        self.state = LeaseState.RELEASED

    @property
    def active(self) -> bool:
        return self.state in (LeaseState.CREATED, LeaseState.BORROWED)


@dataclass(frozen=True)
class EndpointBundle:
    generation: int
    endpoint_id: str
    backend_kind: str
    transport_kind: str
    transport_scope: str
    layout: ProxyBufferLayout
    execution_port: DeviceExecutionPort
    npu_local_window: BorrowedWindowView
    wse_peer_window: BorrowedWindowView
    wse_control: WseServiceControl
    lease: BootstrapLease

    def validate(self) -> None:
        if self.generation <= 0:
            raise ContractError("generation must be positive")
        if not self.endpoint_id:
            raise ContractError("endpoint_id must be non-empty")
        if self.backend_kind != "NPU_SURROGATE":
            raise ContractError("only NPU_SURROGATE backend is supported")
        if self.transport_scope != "HOST_LOCAL":
            raise ContractError("only HOST_LOCAL transport scope is supported")
        if self.execution_port.generation != self.generation:
            raise ContractError("execution port generation mismatch")
        if self.lease.generation != self.generation or not self.lease.active:
            raise ContractError("bootstrap lease is stale or released")
        self.layout.validate()
        self.npu_local_window.validate(generation=self.generation, minimum_bytes=NPU_WINDOW_BYTES)
        self.wse_peer_window.validate(generation=self.generation, minimum_bytes=WSE_WINDOW_BYTES)


@dataclass(frozen=True)
class SignalLine:
    sequence: int

    def to_bytes(self) -> bytes:
        _require_u64("sequence", self.sequence)
        return _U64_LINE.pack(self.sequence, 0, 0, 0, 0, 0, 0, 0)

    @classmethod
    def from_bytes(cls, payload: bytes) -> SignalLine:
        values = _unpack_line(payload)
        return cls(sequence=values[0])


@dataclass(frozen=True)
class HostRequestDescriptor:
    generation: int
    request_id: int
    element_count: int
    input_checksum: int
    sequence: int
    status: int = 0

    def to_bytes(self) -> bytes:
        self.validate()
        return _U64_LINE.pack(
            self.generation,
            self.request_id,
            self.element_count,
            self.input_checksum,
            self.sequence,
            self.status,
            0,
            0,
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> HostRequestDescriptor:
        values = _unpack_line(payload)
        descriptor = cls(*values[:6])
        descriptor.validate()
        return descriptor

    def validate(self) -> None:
        _validate_common_descriptor(self.generation, self.request_id, self.element_count, self.sequence)
        _require_u64("input_checksum", self.input_checksum)
        _require_u64("status", self.status)


@dataclass(frozen=True)
class RemoteTaskDescriptor:
    generation: int
    request_id: int
    element_count: int
    input_checksum: int
    sequence: int
    status: int = 0

    def to_bytes(self) -> bytes:
        self.validate()
        return _U64_LINE.pack(
            self.generation,
            self.request_id,
            self.element_count,
            self.input_checksum,
            self.sequence,
            self.status,
            0,
            0,
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> RemoteTaskDescriptor:
        values = _unpack_line(payload)
        descriptor = cls(*values[:6])
        descriptor.validate()
        return descriptor

    def validate(self) -> None:
        _validate_common_descriptor(self.generation, self.request_id, self.element_count, self.sequence)
        _require_u64("input_checksum", self.input_checksum)
        _require_u64("status", self.status)


@dataclass(frozen=True)
class CompletionDescriptor:
    generation: int
    request_id: int
    element_count: int
    status: int
    output_checksum: int
    sequence: int

    def to_bytes(self) -> bytes:
        self.validate()
        return _U64_LINE.pack(
            self.generation,
            self.request_id,
            self.element_count,
            self.status,
            self.output_checksum,
            self.sequence,
            0,
            0,
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> CompletionDescriptor:
        values = _unpack_line(payload)
        descriptor = cls(*values[:6])
        descriptor.validate()
        return descriptor

    def validate(self) -> None:
        _validate_common_descriptor(self.generation, self.request_id, self.element_count, self.sequence)
        _require_u64("status", self.status)
        _require_u64("output_checksum", self.output_checksum)


@dataclass(frozen=True)
class LifecycleLine:
    stop_requested: int = 0
    stopped: int = 0
    ready: int = 0

    def to_bytes(self) -> bytes:
        for name, value in asdict(self).items():
            if value not in (0, 1):
                raise ContractError(f"{name} must be 0 or 1")
        return _U64_LINE.pack(self.stop_requested, self.stopped, self.ready, 0, 0, 0, 0, 0)

    @classmethod
    def from_bytes(cls, payload: bytes) -> LifecycleLine:
        values = _unpack_line(payload)
        return cls(*values[:3])


@dataclass(frozen=True)
class DriverReport:
    accepted: int
    completed: int
    a_runs: int
    b_submissions: int
    b_completions: int
    c_runs: int
    validation_errors: int
    generation_errors: int
    request_errors: int
    sequence_errors: int
    checksum_errors: int
    input_fences: int
    output_fences: int
    host_wait_cycles: int
    b_wait_cycles: int
    stopped: int

    @classmethod
    def from_bytes(cls, payload: bytes) -> DriverReport:
        if len(payload) != _REPORT.size:
            raise ContractError(f"driver report must be {_REPORT.size} bytes")
        return cls(*_REPORT.unpack(payload))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ServiceReport:
    accepted: int
    completed: int
    b_runs: int
    validation_errors: int
    generation_errors: int
    request_errors: int
    sequence_errors: int
    checksum_errors: int
    input_fences: int
    output_fences: int
    submission_wait_cycles: int
    stopped: int
    reserved0: int
    reserved1: int
    reserved2: int
    reserved3: int

    @classmethod
    def from_bytes(cls, payload: bytes) -> ServiceReport:
        if len(payload) != _REPORT.size:
            raise ContractError(f"service report must be {_REPORT.size} bytes")
        return cls(*_REPORT.unpack(payload))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionResult:
    generation: int
    request_id: int
    sequence: int
    element_count: int
    output: bytes
    output_checksum: int
    final_signal_poll_reads: int
    final_control_d2h_bytes: int
    final_payload_d2h_bytes: int


@dataclass
class MemoryOperationAudit:
    operations: list[dict[str, Any]] = field(default_factory=list)

    def record(self, *, actor: str, operation: str, buffer_id: str) -> None:
        self.operations.append({"actor": actor, "buffer_id": buffer_id, "operation": operation})

    def actors(self) -> set[str]:
        return {str(item["actor"]) for item in self.operations}

    def to_dict(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.operations]


def checksum_u32(payload: bytes) -> int:
    _validate_uint32_payload(payload)
    return sum(value[0] for value in struct.iter_unpack("<I", payload))


def expected_abc(payload: bytes) -> bytes:
    _validate_uint32_payload(payload)
    values = (((value[0] + 1) * 2 + 3) & 0xFFFFFFFF for value in struct.iter_unpack("<I", payload))
    return b"".join(struct.pack("<I", value) for value in values)


def deterministic_input(*, generation: int, request_id: int, element_count: int) -> bytes:
    if generation <= 0 or request_id <= 0:
        raise ContractError("generation and request_id must be positive")
    if element_count <= 0 or element_count > MAX_ELEMENTS:
        raise ContractError(f"element_count must be in [1, {MAX_ELEMENTS}]")
    values = (
        ((generation * 0x9E3779B1) ^ (request_id * 0x85EBCA77) ^ (index * 0xC2B2AE3D)) & 0xFFFFFFFF
        for index in range(element_count)
    )
    return b"".join(struct.pack("<I", value) for value in values)


def _unpack_line(payload: bytes) -> tuple[int, ...]:
    if len(payload) != _U64_LINE.size:
        raise ContractError(f"control line must be {_U64_LINE.size} bytes")
    return _U64_LINE.unpack(payload)


def _require_u64(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value >= 1 << 64:
        raise ContractError(f"{name} must be an unsigned 64-bit integer")


def _validate_common_descriptor(generation: int, request_id: int, element_count: int, sequence: int) -> None:
    for name, value in (("generation", generation), ("request_id", request_id), ("sequence", sequence)):
        _require_u64(name, value)
        if value == 0:
            raise ContractError(f"{name} must be positive")
    if element_count <= 0 or element_count > MAX_ELEMENTS:
        raise ContractError(f"element_count must be in [1, {MAX_ELEMENTS}]")


def _validate_uint32_payload(payload: bytes) -> None:
    if not payload or len(payload) > MAX_PAYLOAD_BYTES or len(payload) % UINT32_BYTES:
        raise ContractError("payload must contain 1..MAX_ELEMENTS little-endian uint32 values")
