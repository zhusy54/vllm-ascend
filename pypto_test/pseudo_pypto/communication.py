# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Device communication ABI for the fixed-ABC pseudo-PyPTO prototype.

This file contains data descriptions and capability interfaces; none of the
classes here creates a process, allocates device memory, or executes a task.
Keeping those concerns out of the contracts makes the software boundary under
test explicit: Bootstrap owns resources, while the proxy service only borrows
the capabilities collected in :class:`EndpointBundle`.

The prototype intentionally describes one fixed program rather than a generic
PyPTO graph.  A runs on the Attention NPU, B runs on an NPU standing in for the
WSE, and C runs back on the Attention NPU.  A real compiler and scheduler are
therefore outside the conclusion supported by this test.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from enum import Enum
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

# PyPTO-local NPU memory.  Input/final payloads and Host-facing control never
# need remote visibility and are allocated by the NPU execution backend.
NPU_LOCAL_INPUT_OFFSET = 0
NPU_LOCAL_FINAL_OUTPUT_OFFSET = MAX_PAYLOAD_BYTES
NPU_LOCAL_CONTROL_OFFSET = 2 * MAX_PAYLOAD_BYTES
NPU_LOCAL_HOST_REQUEST_SIGNAL_OFFSET = NPU_LOCAL_CONTROL_OFFSET
NPU_LOCAL_HOST_REQUEST_DESC_OFFSET = NPU_LOCAL_CONTROL_OFFSET + CACHE_LINE_BYTES
NPU_LOCAL_HOST_RESULT_SIGNAL_OFFSET = NPU_LOCAL_CONTROL_OFFSET + (2 * CACHE_LINE_BYTES)
NPU_LOCAL_HOST_RESULT_DESC_OFFSET = NPU_LOCAL_CONTROL_OFFSET + (3 * CACHE_LINE_BYTES)
NPU_LOCAL_LIFECYCLE_OFFSET = NPU_LOCAL_CONTROL_OFFSET + (4 * CACHE_LINE_BYTES)
NPU_LOCAL_REPORT_OFFSET = NPU_LOCAL_CONTROL_OFFSET + (5 * CACHE_LINE_BYTES)
NPU_LOCAL_WINDOW_BYTES = NPU_LOCAL_CONTROL_OFFSET + (8 * CACHE_LINE_BYTES)

# Externally provisioned NPU-owned communication memory.  The WSE Device
# writes B output then completion descriptor/signal directly into this window.
NPU_SHARED_B_OUTPUT_OFFSET = 0
NPU_SHARED_B_COMPLETION_SIGNAL_OFFSET = MAX_PAYLOAD_BYTES
NPU_SHARED_B_COMPLETION_DESC_OFFSET = MAX_PAYLOAD_BYTES + CACHE_LINE_BYTES
NPU_SHARED_WINDOW_BYTES = MAX_PAYLOAD_BYTES + (3 * CACHE_LINE_BYTES)

# Externally provisioned WSE-owned communication memory.  The NPU Device
# writes A output then B submission descriptor/signal directly into this window.
WSE_SHARED_B_INPUT_OFFSET = 0
WSE_SHARED_B_SUBMISSION_SIGNAL_OFFSET = MAX_PAYLOAD_BYTES
WSE_SHARED_B_SUBMISSION_DESC_OFFSET = MAX_PAYLOAD_BYTES + CACHE_LINE_BYTES
WSE_SHARED_WINDOW_BYTES = MAX_PAYLOAD_BYTES + (3 * CACHE_LINE_BYTES)

# PyPTO-local WSE lifecycle/report memory is not remotely accessible.
WSE_LOCAL_LIFECYCLE_OFFSET = 0
WSE_LOCAL_REPORT_OFFSET = CACHE_LINE_BYTES
WSE_LOCAL_WINDOW_BYTES = 3 * CACHE_LINE_BYTES

_U64_LINE = struct.Struct("<QQQQQQQQ")
_REPORT = struct.Struct("<" + ("Q" * 16))


class ContractError(ValueError):
    """Raised when a proxy contract is malformed or stale."""


class ServiceError(RuntimeError):
    """Raised when a proxy service operation violates its state contract."""


class EndpointRole(str, Enum):
    ATTENTION = "ATTENTION"
    WSE = "WSE"


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
    """Human-readable placement metadata for the fixed validation program."""

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
    """The single accepted program; this is not a general dependency graph."""

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
    """Versioned sizes shared by both processes and both AIV kernels.

    The hash travels in each manifest.  Rejecting a mismatched hash prevents a
    kernel compiled for one offset scheme from attaching to another scheme.
    """

    npu_local_window_bytes: int = NPU_LOCAL_WINDOW_BYTES
    npu_shared_window_bytes: int = NPU_SHARED_WINDOW_BYTES
    wse_local_window_bytes: int = WSE_LOCAL_WINDOW_BYTES
    wse_shared_window_bytes: int = WSE_SHARED_WINDOW_BYTES
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
class NpuCommunicationBinding:
    """NPU-process Device addresses injected by the external provider.

    Both bases are local to the NPU Host process: ``local_shared_base`` maps
    NPU-owned physical memory and ``peer_shared_base`` maps WSE-owned memory.
    PyPTO derives its fixed ABI addresses from these bases and never frees them.
    """

    generation: int
    local_shared_base: int
    local_shared_bytes: int
    peer_shared_base: int
    peer_shared_bytes: int

    def validate(self) -> None:
        _validate_binding(
            self.generation,
            self.local_shared_base,
            self.local_shared_bytes,
            NPU_SHARED_WINDOW_BYTES,
            self.peer_shared_base,
            self.peer_shared_bytes,
            WSE_SHARED_WINDOW_BYTES,
        )


@dataclass(frozen=True)
class WseCommunicationBinding:
    """WSE-process Device addresses injected by the external provider."""

    generation: int
    local_shared_base: int
    local_shared_bytes: int
    peer_shared_base: int
    peer_shared_bytes: int

    def validate(self) -> None:
        _validate_binding(
            self.generation,
            self.local_shared_base,
            self.local_shared_bytes,
            WSE_SHARED_WINDOW_BYTES,
            self.peer_shared_base,
            self.peer_shared_bytes,
            NPU_SHARED_WINDOW_BYTES,
        )


@runtime_checkable
class WseServiceControl(Protocol):
    """Generation-level control RPC; it never carries request tensors."""

    def start(self) -> dict[str, Any]: ...

    def health(self) -> dict[str, Any]: ...

    def drain(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass
class BootstrapLease:
    """Enforces CREATED -> BORROWED -> QUIESCED -> RELEASED ordering.

    ``quiesce`` means the proxy has stopped both resident kernels and released
    its logical use of the bundle.  Only then may Bootstrap destroy mappings.
    """

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
    """External resources injected into one NPU-side PyPTO service."""

    generation: int
    endpoint_id: str
    backend_kind: str
    transport_kind: str
    transport_scope: str
    layout: ProxyBufferLayout
    npu_communication: NpuCommunicationBinding
    wse_control: WseServiceControl
    lease: BootstrapLease

    def validate(self) -> None:
        if self.generation <= 0:
            raise ContractError("generation must be positive")
        if not self.endpoint_id:
            raise ContractError("endpoint_id must be non-empty")
        if self.backend_kind != "WSE":
            raise ContractError("only the WSE backend is supported")
        if self.transport_scope != "HOST_LOCAL":
            raise ContractError("only HOST_LOCAL transport scope is supported")
        if self.lease.generation != self.generation or not self.lease.active:
            raise ContractError("bootstrap lease is stale or released")
        self.layout.validate()
        self.npu_communication.validate()
        if self.npu_communication.generation != self.generation:
            raise ContractError("communication binding generation mismatch")


@dataclass(frozen=True)
class SignalLine:
    """One monotonically increasing sequence used as a publication signal."""

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
    """Host-to-driver request metadata published before the request signal."""

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
    """A-to-B metadata written by the Attention Device into the peer window."""

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
    """Completion metadata used for both B-to-C and final Host completion.

    ``generation`` rejects a stale service instance, ``request_id`` identifies
    the logical call, and ``sequence`` orders reuse of the single shared slot.
    """

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
    """Host-visible ready/stop handshake for one resident kernel."""

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
        lifecycle = cls(*values[:3])
        lifecycle.to_bytes()
        return lifecycle


@dataclass(frozen=True)
class DriverReport:
    """Device-written counters proving the driver's A -> wait-B -> C path."""

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
    """Device-written counters proving the WSE-side Device accepted and ran B."""

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
    """Final proxy result plus Host-visible transfer accounting."""

    generation: int
    request_id: int
    sequence: int
    element_count: int
    output: bytes
    output_checksum: int
    final_signal_poll_reads: int
    final_control_d2h_bytes: int
    final_payload_d2h_bytes: int


def checksum_u32(payload: bytes) -> int:
    """Return a deterministic validation checksum, not a production integrity code."""

    _validate_uint32_payload(payload)
    return sum(value[0] for value in struct.iter_unpack("<I", payload))


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


def _validate_binding(
    generation: int,
    local_base: int,
    local_bytes: int,
    minimum_local_bytes: int,
    peer_base: int,
    peer_bytes: int,
    minimum_peer_bytes: int,
) -> None:
    if generation <= 0:
        raise ContractError("communication generation must be positive")
    if local_base <= 0 or peer_base <= 0:
        raise ContractError("communication Device addresses must be positive")
    if local_bytes < minimum_local_bytes or peer_bytes < minimum_peer_bytes:
        raise ContractError("communication window is smaller than the fixed ABI")


def _validate_binding(
    generation: int,
    local_base: int,
    local_bytes: int,
    required_local_bytes: int,
    peer_base: int,
    peer_bytes: int,
    required_peer_bytes: int,
) -> None:
    if generation <= 0:
        raise ContractError("communication generation must be positive")
    if local_base <= 0 or peer_base <= 0:
        raise ContractError("communication addresses must be positive process-local Device VAs")
    if local_bytes < required_local_bytes or peer_bytes < required_peer_bytes:
        raise ContractError("communication mapping is smaller than the fixed ABI")
