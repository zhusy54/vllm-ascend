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
    fallback_used: bool
    source_filled: bool
    destination_verified: bool
    elapsed_ns: int

    def validate(self) -> None:
        if self.case_id != self.direction.case_id:
            raise ContractError("case_id does not match transfer direction")
        if self.payload_bytes not in BASE_PAYLOAD_SIZES:
            raise ContractError(f"unsupported stage-1A payload size: {self.payload_bytes}")
        if self.sequence_id < 1:
            raise ContractError("sequence_id must be positive")
        if len(self.expected_checksum) != 64 or len(self.observed_checksum) != 64:
            raise ContractError("checksums must be SHA-256 hex digests")
        if self.host_bounce_bytes < 0 or self.elapsed_ns < 0:
            raise ContractError("byte and elapsed counters must be non-negative")

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
                not self.fallback_used,
                self.source_filled,
                self.destination_verified,
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


def stage1a_matrix_complete(observations: tuple[TransferObservation, ...]) -> bool:
    expected = {(direction, payload_bytes) for direction in TransferDirection for payload_bytes in BASE_PAYLOAD_SIZES}
    observed = {(item.direction, item.payload_bytes) for item in observations if item.passed}
    return len(observations) == len(expected) and observed == expected
