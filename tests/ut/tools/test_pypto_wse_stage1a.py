# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import json
import socket
import threading
from types import SimpleNamespace

import pytest

from tools.pypto_wse_validation.acl_vmm import DeviceTransformEvidence
from tools.pypto_wse_validation.bootstrap import ControlChannel
from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1a import (
    Stage1AError,
    _aggregate_result,
    _message,
    _Protocol,
    execute_round_trip_matrix,
    execute_transfer_matrix,
)
from tools.pypto_wse_validation.stage1a_contracts import (
    BASE_PAYLOAD_SIZES,
    ROUND_TRIP_TRANSFORM_API,
    TransferDirection,
)


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
        return (size + 64 * 1024 - 1) // (64 * 1024)


class _FakeTransform:
    def __init__(self, memory):
        self.memory = memory

    def apply(self, address, size, value):
        self.memory[address][:size] = bytes(item ^ value for item in self.memory[address][:size])
        return DeviceTransformEvidence(api=ROUND_TRIP_TRANSFORM_API, elapsed_ns=1, workspace_bytes=0)


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


def test_two_endpoint_protocol_executes_device_round_trip_matrix():
    left, right = socket.socketpair()
    memory = {1: bytearray(max(BASE_PAYLOAD_SIZES)), 2: bytearray(max(BASE_PAYLOAD_SIZES))}
    results = {}
    errors = []

    def run(role, connection, local_address, peer_address):
        try:
            protocol = _Protocol(ControlChannel(connection), role=role, run_id="run-1", generation=1)
            results[role] = execute_round_trip_matrix(
                protocol,
                runtime=_FakeRuntime(memory),
                local_window=_Window(local_address),
                peer_window=_Window(peer_address),
                transform=_FakeTransform(memory) if role is EndpointRole.WSE_SURROGATE else None,
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
    assert len(results[EndpointRole.ATTENTION]) == len(BASE_PAYLOAD_SIZES)
    assert results[EndpointRole.WSE_SURROGATE] == ()
    assert all(item.passed for item in results[EndpointRole.ATTENTION])
    assert all(item.host_intermediate_payload_bytes == 0 for item in results[EndpointRole.ATTENTION])


def test_aggregate_result_claims_only_surrogate_c1(tmp_path):
    left, right = socket.socketpair()
    memory = {1: bytearray(max(BASE_PAYLOAD_SIZES)), 2: bytearray(max(BASE_PAYLOAD_SIZES))}
    observations = {}
    round_trips = {}

    def run(role, connection, local_address, peer_address):
        protocol = _Protocol(ControlChannel(connection), role=role, run_id="run-1", generation=1)
        runtime = _FakeRuntime(memory)
        observations[role] = execute_transfer_matrix(
            protocol,
            runtime=runtime,
            local_window=_Window(local_address),
            peer_window=_Window(peer_address),
        )
        round_trips[role] = execute_round_trip_matrix(
            protocol,
            runtime=runtime,
            local_window=_Window(local_address),
            peer_window=_Window(peer_address),
            transform=_FakeTransform(memory) if role is EndpointRole.WSE_SURROGATE else None,
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
            "cleanup": {"imported_window": "CLOSED", "owned_window": "CLOSED", "runtime": "CLOSED"},
            "control": {"sent_bytes": 1, "sent_messages": {"TRANSFER_COMPLETE": 4, "VERIFIED": 4}},
            "observations": [item.to_dict() for item in observations[role]],
            "round_trips": [item.to_dict() for item in round_trips[role]],
            "success": True,
        }
        (tmp_path / f"{role.value.lower()}_stage1a.json").write_text(json.dumps(artifact), encoding="utf-8")
        (tmp_path / f"{role.value.lower()}_stage1a.log").write_text("", encoding="utf-8")
        device_logs = tmp_path / f"{role.value.lower()}_device_logs"
        device_logs.mkdir()
        (device_logs / "driver.log").write_text(
            "Enable P2P\nMEM_DEV_SMALL_P2P_HBM current_alloced_size=0\n",
            encoding="utf-8",
        )
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
    assert result["resource_cleanup"] == "VERIFIED"
    assert result["host_source_staging_bytes"] > 0
    assert result["host_verification_bytes"] == result["host_source_staging_bytes"]
    assert result["data_results"]["NPU_TO_WSE_SURROGATE"]["passed"] == len(BASE_PAYLOAD_SIZES)
    assert result["data_results"]["T03"]["passed"] == len(BASE_PAYLOAD_SIZES)
    assert result["host_intermediate_payload_bytes"] == 0
