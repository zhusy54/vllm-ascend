# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.pypto_wse_validation.contracts import ContractError, TransportScope
from tools.pypto_wse_validation.stage1a_contracts import (
    BASE_PAYLOAD_SIZES,
    STAGE1A_BACKEND,
    STAGE1A_FENCE_API,
    STAGE1A_HANDLE_KIND,
    STAGE1A_TRANSFER_API,
    MemoryKind,
    TransferDirection,
    TransferObservation,
    deterministic_payload,
    opaque_handle_evidence,
    payload_checksum,
    stage1a_matrix_complete,
)


def _observation(direction: TransferDirection, size: int) -> TransferObservation:
    checksum = payload_checksum(deterministic_payload("run-1", 1, size, size))
    return TransferObservation(
        case_id=direction.case_id,
        direction=direction,
        payload_bytes=size,
        sequence_id=size,
        source_memory=MemoryKind.DEVICE,
        destination_memory=MemoryKind.DEVICE,
        backend=STAGE1A_BACKEND,
        transport_scope=TransportScope.HOST_LOCAL,
        handle_kind=STAGE1A_HANDLE_KIND,
        transfer_api=STAGE1A_TRANSFER_API,
        visibility_fence=STAGE1A_FENCE_API,
        expected_checksum=checksum,
        observed_checksum=checksum,
        host_bounce_bytes=0,
        fallback_used=False,
        source_filled=True,
        destination_verified=True,
        elapsed_ns=1,
    )


def test_payload_is_deterministic_and_context_sensitive():
    first = deterministic_payload("run-1", 2, 3, 4096)
    assert first == deterministic_payload("run-1", 2, 3, 4096)
    assert first != deterministic_payload("run-1", 2, 4, 4096)
    assert len(first) == 4096


@pytest.mark.parametrize("field,value", [("run_id", ""), ("generation", 0), ("sequence_id", 0), ("size", 0)])
def test_payload_rejects_invalid_identity(field, value):
    args = {"run_id": "run-1", "generation": 1, "sequence_id": 1, "size": 64}
    args[field] = value
    with pytest.raises(ContractError):
        deterministic_payload(**args)


def test_handle_evidence_is_redacted_and_stable():
    evidence = opaque_handle_evidence(12345)
    assert evidence == opaque_handle_evidence(12345)
    assert evidence["kind"] == STAGE1A_HANDLE_KIND
    assert evidence["sha256"] != "12345"
    assert set(evidence) == {"kind", "sha256"}


@pytest.mark.parametrize(
    "change",
    [
        {"source_memory": MemoryKind.HOST},
        {"destination_memory": MemoryKind.HOST},
        {"backend": "host_tcp"},
        {"transport_scope": TransportScope.SIMULATION},
        {"transfer_api": "aclrtMemcpy:ACL_MEMCPY_HOST_TO_DEVICE"},
        {"host_bounce_bytes": 64},
        {"fallback_used": True},
        {"observed_checksum": "0" * 64},
        {"source_filled": False},
        {"destination_verified": False},
    ],
)
def test_observation_fails_closed_when_c1_proof_is_missing(change):
    assert not replace(_observation(TransferDirection.NPU_TO_WSE_SURROGATE, 64), **change).passed


def test_stage1a_matrix_requires_both_directions_and_all_base_sizes():
    complete = tuple(_observation(direction, size) for direction in TransferDirection for size in BASE_PAYLOAD_SIZES)
    assert stage1a_matrix_complete(complete)
    assert not stage1a_matrix_complete(complete[:-1])
    assert not stage1a_matrix_complete(complete + (complete[0],))
