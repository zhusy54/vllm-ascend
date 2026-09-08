# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.pypto_wse_validation.contracts import ContractError, TransportScope
from tools.pypto_wse_validation.stage1a_contracts import (
    BASE_PAYLOAD_SIZES,
    ROUND_TRIP_CASE_ID,
    ROUND_TRIP_TRANSFORM_API,
    STAGE1A_BACKEND,
    STAGE1A_FENCE_API,
    STAGE1A_HANDLE_KIND,
    STAGE1A_TRANSFER_API,
    VALIDATION_P2P_CHUNK_BYTES,
    MemoryKind,
    RoundTripObservation,
    deterministic_payload,
    payload_checksum,
    stage1a_round_trip_complete,
    transform_payload,
)


def _round_trip(size: int) -> RoundTripObservation:
    sequence_id = 8 + BASE_PAYLOAD_SIZES.index(size) + 1
    payload = deterministic_payload("run-1", 1, sequence_id, size)
    transformed = transform_payload(payload, sequence_id)
    chunks = (size + VALIDATION_P2P_CHUNK_BYTES - 1) // VALIDATION_P2P_CHUNK_BYTES
    return RoundTripObservation(
        case_id=ROUND_TRIP_CASE_ID,
        payload_bytes=size,
        sequence_id=sequence_id,
        source_memory=MemoryKind.DEVICE,
        transform_memory=MemoryKind.DEVICE,
        destination_memory=MemoryKind.DEVICE,
        backend=STAGE1A_BACKEND,
        transport_scope=TransportScope.HOST_LOCAL,
        handle_kind=STAGE1A_HANDLE_KIND,
        transfer_api=STAGE1A_TRANSFER_API,
        visibility_fence=STAGE1A_FENCE_API,
        transform_api=ROUND_TRIP_TRANSFORM_API,
        transform_value=sequence_id & 0xFF,
        input_checksum=payload_checksum(payload),
        expected_output_checksum=payload_checksum(transformed),
        observed_output_checksum=payload_checksum(transformed),
        host_bounce_bytes=0,
        host_intermediate_payload_bytes=0,
        host_source_staging_bytes=size,
        host_final_verification_bytes=size,
        fallback_used=False,
        transform_device_side=True,
        forward_transfer_chunks=chunks,
        return_transfer_chunks=chunks,
        max_transfer_chunk_bytes=min(size, VALIDATION_P2P_CHUNK_BYTES),
        forward_elapsed_ns=10,
        transform_elapsed_ns=20,
        return_elapsed_ns=10,
        round_trip_elapsed_ns=50,
        transform_workspace_bytes=0,
    )


def test_transform_payload_uses_sequence_low8():
    assert transform_payload(bytes((0x00, 0x55, 0xFF)), 0x101) == bytes((0x01, 0x54, 0xFE))
    with pytest.raises(ContractError):
        transform_payload(b"x", 0)


@pytest.mark.parametrize(
    "change",
    [
        {"transform_memory": MemoryKind.HOST},
        {"transform_api": "host_xor"},
        {"host_bounce_bytes": 64},
        {"host_intermediate_payload_bytes": 64},
        {"transform_device_side": False},
        {"observed_output_checksum": "0" * 64},
        {"forward_transfer_chunks": 2},
        {"return_transfer_chunks": 2},
        {"round_trip_elapsed_ns": 1},
    ],
)
def test_round_trip_fails_closed_when_device_path_proof_is_missing(change):
    assert not replace(_round_trip(64), **change).passed


def test_round_trip_matrix_requires_every_base_size():
    complete = tuple(_round_trip(size) for size in BASE_PAYLOAD_SIZES)
    assert stage1a_round_trip_complete(complete)
    assert RoundTripObservation.from_dict(complete[0].to_dict()) == complete[0]
    assert not stage1a_round_trip_complete(complete[:-1])
    assert not stage1a_round_trip_complete(complete + (complete[0],))
