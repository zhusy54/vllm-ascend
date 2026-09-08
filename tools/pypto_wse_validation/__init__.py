# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Validation helpers for the PyPTO WSE NPU-surrogate prototype."""

from tools.pypto_wse_validation.contracts import (
    EXPECTED_LAYOUT_HASH,
    MAX_PAYLOAD_BYTES,
    PROTOCOL_VERSION,
    SLOT_COUNT,
    AccessFlag,
    CommunicationManifest,
    ContractError,
    DeviceKind,
    EndpointRole,
    RemoteMemoryHandle,
    SlotState,
    TestCompletion,
    TestDescriptor,
    TransportScope,
    validate_manifest_pair,
    validate_publish_order,
    validate_slot_transition,
)

__all__ = [
    "EXPECTED_LAYOUT_HASH",
    "MAX_PAYLOAD_BYTES",
    "PROTOCOL_VERSION",
    "SLOT_COUNT",
    "AccessFlag",
    "CommunicationManifest",
    "ContractError",
    "DeviceKind",
    "EndpointRole",
    "RemoteMemoryHandle",
    "SlotState",
    "TestCompletion",
    "TestDescriptor",
    "TransportScope",
    "validate_manifest_pair",
    "validate_publish_order",
    "validate_slot_transition",
]
