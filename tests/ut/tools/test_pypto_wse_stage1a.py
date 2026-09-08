# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import json
import socket
import threading
from types import SimpleNamespace

import pytest

from tools.pypto_wse_validation.bootstrap import ControlChannel
from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1a import (
    Stage1AError,
    _aggregate_result,
    _message,
    _Protocol,
    execute_transfer_matrix,
)
from tools.pypto_wse_validation.stage1a_contracts import BASE_PAYLOAD_SIZES, TransferDirection


class _Window:
    def __init__(self, address: int):
        self.address = address


class _FakeRuntime:
    def __init__(self, memory):
        self.memory = memory

    def copy_host_to_device(self, destination, payload):
        self.memory[destination][: len(payload)] = payload

    def copy_device_to_host(self, source, size):
        return bytes(self.memory[source][:size])

    def copy_device_to_device(self, destination, source, size):
        self.memory[destination][:size] = self.memory[source][:size]


def test_control_message_rejects_payload_data():
    with pytest.raises(Stage1AError, match="payload data"):
        _message(
            "TRANSFER_COMPLETE",
            run_id="run-1",
            generation=1,
            role=EndpointRole.ATTENTION,
            fields={"nested": {"payload": "forbidden"}},
        )


def test_two_endpoint_protocol_executes_full_bidirectional_matrix():
    left, right = socket.socketpair()
    memory = {1: bytearray(max(BASE_PAYLOAD_SIZES)), 2: bytearray(max(BASE_PAYLOAD_SIZES))}
    results = {}
    errors = []

    def run(role, connection, local_address, peer_address):
        try:
            protocol = _Protocol(ControlChannel(connection), role=role, run_id="run-1", generation=1)
            results[role] = execute_transfer_matrix(
                protocol,
                runtime=_FakeRuntime(memory),
                local_window=_Window(local_address),
                peer_window=_Window(peer_address),
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    attention = threading.Thread(target=run, args=(EndpointRole.ATTENTION, left, 1, 2))
    surrogate = threading.Thread(target=run, args=(EndpointRole.WSE_SURROGATE, right, 2, 1))
    attention.start()
    surrogate.start()
    attention.join(timeout=5)
    surrogate.join(timeout=5)
    left.close()
    right.close()

    assert not attention.is_alive()
    assert not surrogate.is_alive()
    assert not errors
    assert {item.direction for item in results[EndpointRole.ATTENTION]} == {TransferDirection.NPU_TO_WSE_SURROGATE}
    assert {item.direction for item in results[EndpointRole.WSE_SURROGATE]} == {TransferDirection.WSE_SURROGATE_TO_NPU}
    assert all(item.passed for observations in results.values() for item in observations)


def test_aggregate_result_claims_only_surrogate_c1(tmp_path):
    left, right = socket.socketpair()
    memory = {1: bytearray(max(BASE_PAYLOAD_SIZES)), 2: bytearray(max(BASE_PAYLOAD_SIZES))}
    observations = {}

    def run(role, connection, local_address, peer_address):
        protocol = _Protocol(ControlChannel(connection), role=role, run_id="run-1", generation=1)
        observations[role] = execute_transfer_matrix(
            protocol,
            runtime=_FakeRuntime(memory),
            local_window=_Window(local_address),
            peer_window=_Window(peer_address),
        )

    threads = [
        threading.Thread(target=run, args=(EndpointRole.ATTENTION, left, 1, 2)),
        threading.Thread(target=run, args=(EndpointRole.WSE_SURROGATE, right, 2, 1)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    left.close()
    right.close()

    processes = {}
    for index, role in enumerate(EndpointRole, start=1):
        artifact = {
            "control": {"sent_bytes": 1, "sent_messages": {"TRANSFER_COMPLETE": 4, "VERIFIED": 4}},
            "observations": [item.to_dict() for item in observations[role]],
            "success": True,
        }
        (tmp_path / f"{role.value.lower()}_stage1a.json").write_text(json.dumps(artifact), encoding="utf-8")
        (tmp_path / f"{role.value.lower()}_stage1a.log").write_text("", encoding="utf-8")
        processes[role] = SimpleNamespace(returncode=0, pid=index)
    args = SimpleNamespace(
        artifact_dir=tmp_path,
        devices=(0, 1),
        generation=1,
        run_id="run-1",
        start_order="attention-first",
    )
    result = _aggregate_result(args, processes)
    assert result["success"] is True
    assert result["capability_level"] == "C1"
    assert result["claim_scope"] == "NPU_SURROGATE_ONLY"
    assert result["npu_wse_capability_level"] == "NOT_ESTABLISHED"
    assert result["evidence_status"] == "SIMULATION"
    assert result["host_bounce_bytes"] == 0
    assert result["host_source_staging_bytes"] > 0
    assert result["host_verification_bytes"] == result["host_source_staging_bytes"]
    assert result["data_results"]["NPU_TO_WSE_SURROGATE"]["passed"] == len(BASE_PAYLOAD_SIZES)
