# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Evidence contracts for stage-1A host-initiated Device Memory transfer."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from tools.pypto_wse_validation.contracts import ContractError, TransportScope

BASE_PAYLOAD_SIZES = (64, 4 * 1024, 64 * 1024, 1024 * 1024)
STAGE1A_BACKEND = "cann_acl_vmm_p2p"
STAGE1A_HANDLE_KIND = "ACL_VMM_SHAREABLE_HANDLE"
STAGE1A_TRANSFER_API = "aclrtMemcpy:ACL_MEMCPY_DEVICE_TO_DEVICE"
STAGE1A_FENCE_API = "blocking_aclrtMemcpy_completion"
VALIDATION_P2P_CHUNK_BYTES = 64 * 1024
ROUND_TRIP_CASE_ID = "T03"
ROUND_TRIP_TRANSFORM_API = "aclnnInplaceBitwiseXorScalar"


class TransferDirection(str, Enum):
    NPU_TO_WSE_SURROGATE = "NPU_TO_WSE_SURROGATE"
    WSE_SURROGATE_TO_NPU = "WSE_SURROGATE_TO_NPU"

    @property
    def case_id(self) -> str:
        if self is TransferDirection.NPU_TO_WSE_SURROGATE:
            return "T01"
        return "T02"


class MemoryKind(str, Enum):
    DEVICE = "DEVICE_MEMORY"
    HOST = "HOST_MEMORY"


def deterministic_payload(run_id: str, generation: int, sequence_id: int, size: int) -> bytes:
    """Build reproducible bytes without relying on model or runtime state."""
    if not run_id:
        raise ContractError("run_id must be non-empty")
    if generation < 1 or sequence_id < 1:
        raise ContractError("generation and sequence_id must be positive")
    if size < 1:
        raise ContractError("payload size must be positive")
    seed = f"{run_id}:{generation}:{sequence_id}".encode()
    output = bytearray()
    block = 0
    while len(output) < size:
        output.extend(hashlib.sha256(seed + block.to_bytes(8, "little")).digest())
        block += 1
    return bytes(output[:size])


def payload_checksum(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def transform_payload(payload: bytes, sequence_id: int) -> bytes:
    if sequence_id < 1:
        raise ContractError("sequence_id must be positive")
    xor_value = sequence_id & 0xFF
    return bytes(value ^ xor_value for value in payload)


def opaque_handle_evidence(handle: int) -> dict[str, str]:
    """Return useful handle identity evidence without persisting the capability."""
    if not isinstance(handle, int) or isinstance(handle, bool) or handle <= 0:
        raise ContractError("opaque handle must be a positive integer")
    digest = hashlib.sha256(str(handle).encode()).hexdigest()
    return {"kind": STAGE1A_HANDLE_KIND, "sha256": digest}


@dataclass(frozen=True)
class TransferObservation:
    case_id: str
    direction: TransferDirection
    payload_bytes: int
    sequence_id: int
    source_memory: MemoryKind
    destination_memory: MemoryKind
    backend: str
    transport_scope: TransportScope
    handle_kind: str
    transfer_api: str
    visibility_fence: str
    expected_checksum: str
    observed_checksum: str
    host_bounce_bytes: int
    host_source_staging_bytes: int
    host_verification_bytes: int
    fallback_used: bool
    source_filled: bool
    destination_verified: bool
    elapsed_ns: int
    transfer_chunks: int
    max_transfer_chunk_bytes: int

    def validate(self) -> None:
        if self.case_id != self.direction.case_id:
            raise ContractError("case_id does not match transfer direction")
        if self.payload_bytes not in BASE_PAYLOAD_SIZES:
            raise ContractError(f"unsupported stage-1A payload size: {self.payload_bytes}")
        if self.sequence_id < 1:
            raise ContractError("sequence_id must be positive")
        if len(self.expected_checksum) != 64 or len(self.observed_checksum) != 64:
            raise ContractError("checksums must be SHA-256 hex digests")
        if min(self.host_bounce_bytes, self.host_source_staging_bytes, self.host_verification_bytes) < 0:
            raise ContractError("host byte counters must be non-negative")
        if self.elapsed_ns < 0:
            raise ContractError("byte and elapsed counters must be non-negative")
        if self.transfer_chunks < 1 or self.max_transfer_chunk_bytes < 1:
            raise ContractError("transfer chunk evidence must be positive")

    @property
    def passed(self) -> bool:
        self.validate()
        return all(
            (
                self.source_memory is MemoryKind.DEVICE,
                self.destination_memory is MemoryKind.DEVICE,
                self.backend == STAGE1A_BACKEND,
                self.transport_scope is TransportScope.HOST_LOCAL,
                self.handle_kind == STAGE1A_HANDLE_KIND,
                self.transfer_api == STAGE1A_TRANSFER_API,
                self.visibility_fence == STAGE1A_FENCE_API,
                self.expected_checksum == self.observed_checksum,
                self.host_bounce_bytes == 0,
                self.host_source_staging_bytes == self.payload_bytes,
                self.host_verification_bytes == self.payload_bytes,
                not self.fallback_used,
                self.source_filled,
                self.destination_verified,
                self.transfer_chunks
                == (self.payload_bytes + VALIDATION_P2P_CHUNK_BYTES - 1) // VALIDATION_P2P_CHUNK_BYTES,
                self.max_transfer_chunk_bytes == min(self.payload_bytes, VALIDATION_P2P_CHUNK_BYTES),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["direction"] = self.direction.value
        result["source_memory"] = self.source_memory.value
        result["destination_memory"] = self.destination_memory.value
        result["transport_scope"] = self.transport_scope.value
        result["passed"] = self.passed
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TransferObservation:
        data = dict(value)
        data.pop("passed", None)
        try:
            data["direction"] = TransferDirection(data["direction"])
            data["source_memory"] = MemoryKind(data["source_memory"])
            data["destination_memory"] = MemoryKind(data["destination_memory"])
            data["transport_scope"] = TransportScope(data["transport_scope"])
            observation = cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid transfer observation: {exc}") from exc
        observation.validate()
        return observation


def stage1a_matrix_complete(observations: tuple[TransferObservation, ...]) -> bool:
    expected = {(direction, payload_bytes) for direction in TransferDirection for payload_bytes in BASE_PAYLOAD_SIZES}
    observed = {(item.direction, item.payload_bytes) for item in observations if item.passed}
    return len(observations) == len(expected) and observed == expected


@dataclass(frozen=True)
class RoundTripObservation:
    case_id: str
    payload_bytes: int
    sequence_id: int
    source_memory: MemoryKind
    transform_memory: MemoryKind
    destination_memory: MemoryKind
    backend: str
    transport_scope: TransportScope
    handle_kind: str
    transfer_api: str
    visibility_fence: str
    transform_api: str
    transform_value: int
    input_checksum: str
    expected_output_checksum: str
    observed_output_checksum: str
    host_bounce_bytes: int
    host_intermediate_payload_bytes: int
    host_source_staging_bytes: int
    host_final_verification_bytes: int
    fallback_used: bool
    transform_device_side: bool
    forward_transfer_chunks: int
    return_transfer_chunks: int
    max_transfer_chunk_bytes: int
    forward_elapsed_ns: int
    transform_elapsed_ns: int
    return_elapsed_ns: int
    round_trip_elapsed_ns: int
    transform_workspace_bytes: int

    def validate(self) -> None:
        if self.case_id != ROUND_TRIP_CASE_ID:
            raise ContractError(f"round-trip case_id must be {ROUND_TRIP_CASE_ID}")
        if self.payload_bytes not in BASE_PAYLOAD_SIZES:
            raise ContractError(f"unsupported stage-1A payload size: {self.payload_bytes}")
        if self.sequence_id < 1:
            raise ContractError("sequence_id must be positive")
        if self.transform_value != self.sequence_id & 0xFF:
            raise ContractError("transform value must equal LOW8(sequence_id)")
        checksums = (self.input_checksum, self.expected_output_checksum, self.observed_output_checksum)
        if any(len(value) != 64 for value in checksums):
            raise ContractError("round-trip checksums must be SHA-256 hex digests")
        byte_counters = (
            self.host_bounce_bytes,
            self.host_intermediate_payload_bytes,
            self.host_source_staging_bytes,
            self.host_final_verification_bytes,
            self.transform_workspace_bytes,
        )
        if min(byte_counters) < 0:
            raise ContractError("round-trip byte counters must be non-negative")
        elapsed = (
            self.forward_elapsed_ns,
            self.transform_elapsed_ns,
            self.return_elapsed_ns,
            self.round_trip_elapsed_ns,
        )
        if min(elapsed) < 0:
            raise ContractError("round-trip elapsed counters must be non-negative")

    @property
    def passed(self) -> bool:
        self.validate()
        chunks = (self.payload_bytes + VALIDATION_P2P_CHUNK_BYTES - 1) // VALIDATION_P2P_CHUNK_BYTES
        return all(
            (
                self.source_memory is MemoryKind.DEVICE,
                self.transform_memory is MemoryKind.DEVICE,
                self.destination_memory is MemoryKind.DEVICE,
                self.backend == STAGE1A_BACKEND,
                self.transport_scope is TransportScope.HOST_LOCAL,
                self.handle_kind == STAGE1A_HANDLE_KIND,
                self.transfer_api == STAGE1A_TRANSFER_API,
                self.visibility_fence == STAGE1A_FENCE_API,
                self.transform_api == ROUND_TRIP_TRANSFORM_API,
                self.expected_output_checksum == self.observed_output_checksum,
                self.host_bounce_bytes == 0,
                self.host_intermediate_payload_bytes == 0,
                self.host_source_staging_bytes == self.payload_bytes,
                self.host_final_verification_bytes == self.payload_bytes,
                not self.fallback_used,
                self.transform_device_side,
                self.forward_transfer_chunks == chunks,
                self.return_transfer_chunks == chunks,
                self.max_transfer_chunk_bytes == min(self.payload_bytes, VALIDATION_P2P_CHUNK_BYTES),
                self.round_trip_elapsed_ns
                >= self.forward_elapsed_ns + self.transform_elapsed_ns + self.return_elapsed_ns,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["source_memory"] = self.source_memory.value
        result["transform_memory"] = self.transform_memory.value
        result["destination_memory"] = self.destination_memory.value
        result["transport_scope"] = self.transport_scope.value
        result["passed"] = self.passed
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RoundTripObservation:
        data = dict(value)
        data.pop("passed", None)
        try:
            data["source_memory"] = MemoryKind(data["source_memory"])
            data["transform_memory"] = MemoryKind(data["transform_memory"])
            data["destination_memory"] = MemoryKind(data["destination_memory"])
            data["transport_scope"] = TransportScope(data["transport_scope"])
            observation = cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid round-trip observation: {exc}") from exc
        observation.validate()
        return observation


def stage1a_round_trip_complete(observations: tuple[RoundTripObservation, ...]) -> bool:
    expected = set(BASE_PAYLOAD_SIZES)
    observed = {item.payload_bytes for item in observations if item.passed}
    return len(observations) == len(expected) and observed == expected
