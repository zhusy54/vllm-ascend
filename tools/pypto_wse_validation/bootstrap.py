# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Stage-0 control-plane framing and contract-only manifest helpers."""

from __future__ import annotations

import json
import socket
import struct
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tools.pypto_wse_validation.contracts import (
    COMPLETION_ENTRY_BYTES,
    EXPECTED_LAYOUT_HASH,
    MAX_PAYLOAD_BYTES,
    PROTOCOL_VERSION,
    SLOT_COUNT,
    SUBMISSION_ENTRY_BYTES,
    AccessFlag,
    CommunicationManifest,
    DeviceKind,
    EndpointRole,
    RemoteMemoryHandle,
    TransportScope,
)

MAX_CONTROL_FRAME_BYTES = 64 * 1024
ALLOWED_MESSAGE_TYPES = frozenset({"HELLO", "READY", "HEALTH", "STOP", "ERROR"})
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "completion",
        "descriptor",
        "output",
        "payload",
        "submit_task",
        "task",
        "tensor",
    }
)
_FRAME_LENGTH = struct.Struct("!I")


class BootstrapError(RuntimeError):
    """Raised when the stage-0 bootstrap protocol is violated."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_PAYLOAD_KEYS:
                found.add(normalized)
            found.update(_forbidden_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_forbidden_keys(child))
    return found


def validate_control_message(
    message: Mapping[str, Any],
    *,
    run_id: str,
    generation: int,
    expected_type: str | None = None,
    expected_role: EndpointRole | None = None,
) -> None:
    message_type = message.get("type")
    if message_type not in ALLOWED_MESSAGE_TYPES:
        raise BootstrapError(f"control message type is not allowed: {message_type!r}")
    if expected_type is not None and message_type != expected_type:
        raise BootstrapError(f"expected {expected_type}, received {message_type!r}")
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise BootstrapError("control message protocol_version mismatch")
    if message.get("run_id") != run_id:
        raise BootstrapError("control message run_id mismatch")
    if message.get("generation") != generation:
        raise BootstrapError("control message generation mismatch")
    if expected_role is not None and message.get("role") != expected_role.value:
        raise BootstrapError(f"expected peer role {expected_role.value}, received {message.get('role')!r}")
    forbidden = _forbidden_keys(message)
    if forbidden:
        raise BootstrapError(f"control message contains data-plane keys: {sorted(forbidden)}")


def make_control_message(
    message_type: str,
    *,
    run_id: str,
    generation: int,
    role: EndpointRole,
    fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "generation": generation,
        "protocol_version": PROTOCOL_VERSION,
        "role": role.value,
        "run_id": run_id,
        "type": message_type,
    }
    if fields:
        overlap = set(message).intersection(fields)
        if overlap:
            raise BootstrapError(f"reserved control fields cannot be replaced: {sorted(overlap)}")
        message.update(fields)
    validate_control_message(message, run_id=run_id, generation=generation)
    return message


def _receive_exact(connection: socket.socket, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise BootstrapError("control connection closed before a complete frame was received")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass
class ControlChannel:
    connection: socket.socket
    sent_messages: Counter[str] = field(default_factory=Counter)
    received_messages: Counter[str] = field(default_factory=Counter)
    sent_bytes: int = 0
    received_bytes: int = 0

    def send(self, message: Mapping[str, Any]) -> None:
        payload = _canonical_json(message)
        if len(payload) > MAX_CONTROL_FRAME_BYTES:
            raise BootstrapError(f"control frame exceeds {MAX_CONTROL_FRAME_BYTES} bytes")
        self.connection.sendall(_FRAME_LENGTH.pack(len(payload)) + payload)
        self.sent_messages[str(message["type"])] += 1
        self.sent_bytes += len(payload)

    def receive(self) -> dict[str, Any]:
        (length,) = _FRAME_LENGTH.unpack(_receive_exact(self.connection, _FRAME_LENGTH.size))
        if length == 0 or length > MAX_CONTROL_FRAME_BYTES:
            raise BootstrapError(f"invalid control frame length: {length}")
        payload = _receive_exact(self.connection, length)
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BootstrapError("control frame is not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise BootstrapError("control frame must contain a JSON object")
        self.received_messages[str(decoded.get("type", "INVALID"))] += 1
        self.received_bytes += length
        return decoded

    def evidence(self) -> dict[str, Any]:
        return {
            "received_bytes": self.received_bytes,
            "received_messages": dict(sorted(self.received_messages.items())),
            "sent_bytes": self.sent_bytes,
            "sent_messages": dict(sorted(self.sent_messages.items())),
        }


def _contract_handle(endpoint_id: str, buffer_id: str, generation: int, size_bytes: int) -> RemoteMemoryHandle:
    return RemoteMemoryHandle(
        owner_endpoint_id=endpoint_id,
        buffer_id=buffer_id,
        generation=generation,
        size_bytes=size_bytes,
        alignment=4096,
        access_flags=int(AccessFlag.READ | AccessFlag.WRITE),
        handle_kind="CONTRACT_ONLY",
        opaque_handle="NOT_BOUND_STAGE0",
    )


def make_contract_manifest(
    *,
    run_id: str,
    generation: int,
    role: EndpointRole,
    device_id: int,
) -> dict[str, Any]:
    endpoint_id = role.value.lower().replace("_", "-")
    common: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "endpoint_id": endpoint_id,
        "generation": generation,
        "role": role,
        "device_kind": DeviceKind.ASCEND_NPU,
        "transport_kind": "stage0-control-only",
        "transport_scope": TransportScope.SIMULATION,
        "capabilities": ("independent_device_runtime_init", "tcp_loopback_control"),
        "layout_hash": EXPECTED_LAYOUT_HASH,
        "slot_count": SLOT_COUNT,
        "slot_bytes": MAX_PAYLOAD_BYTES,
        "max_inflight": SLOT_COUNT,
    }
    if role is EndpointRole.ATTENTION:
        manifest = CommunicationManifest(
            output_window=_contract_handle(endpoint_id, "output-window", generation, MAX_PAYLOAD_BYTES),
            completion_queue=_contract_handle(
                endpoint_id,
                "completion-queue",
                generation,
                COMPLETION_ENTRY_BYTES * SLOT_COUNT,
            ),
            **common,
        )
    else:
        manifest = CommunicationManifest(
            input_window=_contract_handle(endpoint_id, "input-window", generation, MAX_PAYLOAD_BYTES),
            submission_queue=_contract_handle(
                endpoint_id,
                "submission-queue",
                generation,
                SUBMISSION_ENTRY_BYTES * SLOT_COUNT,
            ),
            **common,
        )
    return {
        "device_id": device_id,
        "evidence_status": "CONTRACT_ONLY",
        "manifest": manifest.to_dict(),
        "runtime_bound": False,
    }
