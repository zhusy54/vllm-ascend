# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.pypto_wse_validation.contracts import (
    COMPLETION_ENTRY_BYTES,
    EXPECTED_LAYOUT_HASH,
    MAX_PAYLOAD_BYTES,
    PROTOCOL_VERSION,
    SLOT_COUNT,
    SUBMISSION_ENTRY_BYTES,
    AccessFlag,
    CommunicationManifest,
    ContractError,
    DeviceKind,
    EndpointRole,
    PatternKind,
    PublishEvent,
    RemoteMemoryHandle,
    SlotState,
    TestCompletion,
    TestDescriptor,
    TransportScope,
    validate_manifest_pair,
    validate_publish_order,
    validate_slot_transition,
)


def _handle(owner: str, name: str, generation: int = 7) -> RemoteMemoryHandle:
    return RemoteMemoryHandle(
        owner_endpoint_id=owner,
        buffer_id=name,
        generation=generation,
        size_bytes=MAX_PAYLOAD_BYTES,
        alignment=4096,
        access_flags=int(AccessFlag.READ | AccessFlag.WRITE),
        handle_kind="STAGE0_OPAQUE",
        opaque_handle=f"opaque-{owner}-{name}",
    )


def _manifests() -> tuple[CommunicationManifest, CommunicationManifest]:
    common = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": "stage0-test",
        "generation": 7,
        "device_kind": DeviceKind.ASCEND_NPU,
        "transport_kind": "npu-surrogate-bootstrap",
        "transport_scope": TransportScope.SIMULATION,
        "capabilities": ("device_init", "tcp_bootstrap"),
        "layout_hash": EXPECTED_LAYOUT_HASH,
        "slot_count": SLOT_COUNT,
        "slot_bytes": MAX_PAYLOAD_BYTES,
        "max_inflight": SLOT_COUNT,
    }
    attention = CommunicationManifest(
        endpoint_id="attention",
        role=EndpointRole.ATTENTION,
        output_window=_handle("attention", "output"),
        completion_queue=_handle("attention", "completion"),
        **common,
    )
    surrogate = CommunicationManifest(
        endpoint_id="wse-surrogate",
        role=EndpointRole.WSE_SURROGATE,
        input_window=_handle("wse-surrogate", "input"),
        submission_queue=_handle("wse-surrogate", "submission"),
        **common,
    )
    return attention, surrogate


def test_manifest_pair_round_trip_is_canonical() -> None:
    attention, surrogate = _manifests()
    validate_manifest_pair(attention, surrogate)

    decoded = CommunicationManifest.from_dict(attention.to_dict())
    assert decoded == attention
    assert decoded.to_json() == attention.to_json()
    assert len(EXPECTED_LAYOUT_HASH) == 64


@pytest.mark.parametrize(
    ("side", "field", "value", "message"),
    [
        ("attention", "protocol_version", 2, "unsupported protocol_version"),
        ("attention", "generation", 8, "generation does not match manifest"),
        ("attention", "layout_hash", "0" * 64, "layout_hash"),
        ("attention", "transport_scope", TransportScope.HOST_LOCAL, "SIMULATION"),
        ("attention", "role", EndpointRole.WSE_SURROGATE, "must export"),
        ("surrogate", "run_id", "other-run", "run_id mismatch"),
    ],
)
def test_manifest_pair_rejects_contract_drift(side: str, field: str, value: object, message: str) -> None:
    attention, surrogate = _manifests()
    if side == "attention":
        attention = replace(attention, **{field: value})
    else:
        surrogate = replace(surrogate, **{field: value})
    with pytest.raises(ContractError, match=message):
        validate_manifest_pair(attention, surrogate)


def test_manifest_rejects_wrong_owner_and_read_only_peer_window() -> None:
    attention, _ = _manifests()
    wrong_owner = replace(attention.output_window, owner_endpoint_id="peer")
    with pytest.raises(ContractError, match="owner does not match"):
        replace(attention, output_window=wrong_owner).validate()

    read_only = replace(attention.output_window, access_flags=int(AccessFlag.READ))
    with pytest.raises(ContractError, match="peer writes"):
        replace(attention, output_window=read_only).validate()


def test_descriptor_and_completion_have_frozen_wire_sizes() -> None:
    descriptor = TestDescriptor(
        protocol_version=PROTOCOL_VERSION,
        flags=0,
        generation=7,
        sequence_id=1,
        input_slot=0,
        output_slot=1,
        payload_bytes=4096,
        pattern_kind=int(PatternKind.HASH_XOR_SEQUENCE_LOW8),
        expected_checksum=123,
    )
    completion = TestCompletion(
        protocol_version=PROTOCOL_VERSION,
        status=0,
        generation=7,
        sequence_id=1,
        output_slot=1,
        output_bytes=4096,
        output_checksum=456,
    )

    assert len(descriptor.to_bytes()) == SUBMISSION_ENTRY_BYTES
    assert TestDescriptor.from_bytes(descriptor.to_bytes()) == descriptor
    assert len(completion.to_bytes()) == COMPLETION_ENTRY_BYTES
    assert TestCompletion.from_bytes(completion.to_bytes()) == completion


def test_wire_records_reject_bounds_and_truncation() -> None:
    descriptor = TestDescriptor(
        protocol_version=PROTOCOL_VERSION,
        flags=0,
        generation=1,
        sequence_id=1,
        input_slot=SLOT_COUNT,
        output_slot=0,
        payload_bytes=1,
        pattern_kind=int(PatternKind.HASH_XOR_SEQUENCE_LOW8),
        expected_checksum=0,
    )
    with pytest.raises(ContractError, match="input_slot"):
        descriptor.to_bytes()
    with pytest.raises(ContractError, match="descriptor must be"):
        TestDescriptor.from_bytes(b"short")

    completion = TestCompletion(PROTOCOL_VERSION, 0, 1, 1, 0, 0, 0)
    with pytest.raises(ContractError, match="successful completion"):
        completion.to_bytes()


def test_slot_state_machine_accepts_only_the_frozen_cycle() -> None:
    states = list(SlotState)
    for current, next_state in zip(states, (*states[1:], SlotState.FREE)):
        validate_slot_transition(current, next_state)

    with pytest.raises(ContractError, match="invalid slot transition"):
        validate_slot_transition(SlotState.RESERVED, SlotState.COMMAND_PUBLISHED)


def test_publish_order_requires_both_visibility_fences() -> None:
    valid = list(PublishEvent)
    validate_publish_order(valid)

    without_input_fence = [event for event in valid if event is not PublishEvent.INPUT_FENCE]
    with pytest.raises(ContractError, match="invalid publish order"):
        validate_publish_order(without_input_fence)

    reordered = valid.copy()
    reordered[-2:] = reversed(reordered[-2:])
    with pytest.raises(ContractError, match="invalid publish order"):
        validate_publish_order(reordered)
