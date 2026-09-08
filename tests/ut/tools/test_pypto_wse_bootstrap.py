# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import json
import socket
import struct
from pathlib import Path

import pytest

from tools.pypto_wse_validation.bootstrap import (
    BootstrapError,
    ControlChannel,
    make_contract_manifest,
    make_control_message,
    validate_control_message,
)
from tools.pypto_wse_validation.contracts import (
    CommunicationManifest,
    EndpointRole,
    validate_manifest_pair,
)
from tools.pypto_wse_validation.launch_local import main


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_control_channel_round_trip_tracks_only_control_bytes() -> None:
    left, right = socket.socketpair()
    try:
        sender = ControlChannel(left)
        receiver = ControlChannel(right)
        message = make_control_message(
            "READY",
            run_id="run-1",
            generation=1,
            role=EndpointRole.ATTENTION,
            fields={"runtime_initialized": True},
        )
        sender.send(message)
        received = receiver.receive()
        validate_control_message(
            received,
            run_id="run-1",
            generation=1,
            expected_type="READY",
            expected_role=EndpointRole.ATTENTION,
        )
        assert received == message
        assert sender.sent_messages == {"READY": 1}
        assert sender.sent_bytes == receiver.received_bytes
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("forbidden", ("payload", "tensor", "descriptor", "completion", "task"))
def test_control_message_rejects_data_plane_fields(forbidden: str) -> None:
    with pytest.raises(BootstrapError, match="data-plane keys"):
        make_control_message(
            "HEALTH",
            run_id="run-1",
            generation=1,
            role=EndpointRole.ATTENTION,
            fields={"nested": {forbidden: "forbidden"}},
        )


def test_control_channel_rejects_oversized_declared_frame() -> None:
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!I", 64 * 1024 + 1))
        with pytest.raises(BootstrapError, match="invalid control frame length"):
            ControlChannel(right).receive()
    finally:
        left.close()
        right.close()


def test_contract_only_manifests_are_valid_but_not_runtime_evidence() -> None:
    attention = make_contract_manifest(run_id="run-1", generation=1, role=EndpointRole.ATTENTION, device_id=0)
    surrogate = make_contract_manifest(
        run_id="run-1",
        generation=1,
        role=EndpointRole.WSE_SURROGATE,
        device_id=1,
    )
    validate_manifest_pair(
        CommunicationManifest.from_dict(attention["manifest"]),
        CommunicationManifest.from_dict(surrogate["manifest"]),
    )
    assert attention["evidence_status"] == "CONTRACT_ONLY"
    assert attention["runtime_bound"] is False
    assert attention["manifest"]["output_window"]["opaque_handle"] == "NOT_BOUND_STAGE0"


@pytest.mark.parametrize("start_order", ("attention-first", "wse-first"))
def test_launcher_uses_independent_processes_and_both_start_orders(tmp_path: Path, start_order: str) -> None:
    artifact_dir = tmp_path / start_order
    assert (
        main(
            [
                "run",
                "--devices",
                "0,1",
                "--artifact-dir",
                str(artifact_dir),
                "--runtime-backend",
                "noop",
                "--start-order",
                start_order,
                "--start-delay",
                "0.05",
                "--timeout",
                "5",
            ]
        )
        == 0
    )
    result = _read(artifact_dir / "result.json")
    attention = _read(artifact_dir / "attention_endpoint.json")
    surrogate = _read(artifact_dir / "wse_surrogate_endpoint.json")
    assert result["success"] is True
    assert result["capability_level"] == "C0"
    assert result["start_order"] == start_order
    assert result["control_plane"]["message_counts"] == {"HEALTH": 2, "HELLO": 2, "READY": 2, "STOP": 2}
    assert result["forbidden_hot_path_activity"] == {
        "host_completion_messages": 0,
        "host_payload_bytes": 0,
        "host_task_messages": 0,
    }
    assert attention["pid"] != surrogate["pid"]
    assert attention["device_id"] != surrogate["device_id"]
    assert attention["working_directory"] != surrogate["working_directory"]
    assert (artifact_dir / attention["working_directory"]).is_dir()
    assert (artifact_dir / surrogate["working_directory"]).is_dir()
    assert attention["runtime_close_attempts"] == [
        {"attempt": "1", "outcome": "CLOSED"},
        {"attempt": "2", "outcome": "CLOSED"},
    ]
    assert _read(artifact_dir / "transport.json")["data_plane"] == "NOT_EXERCISED"


def test_launcher_reports_endpoint_init_failure_without_hanging(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "failed"
    assert (
        main(
            [
                "run",
                "--devices",
                "0,1",
                "--artifact-dir",
                str(artifact_dir),
                "--runtime-backend",
                "noop",
                "--start-order",
                "wse-first",
                "--start-delay",
                "0",
                "--timeout",
                "0.5",
                "--fail-role",
                "WSE_SURROGATE",
            ]
        )
        == 1
    )
    result = _read(artifact_dir / "result.json")
    assert result["success"] is False
    assert result["capability_level"] == "NONE"
    assert result["endpoints"]["WSE_SURROGATE"]["success"] is False
