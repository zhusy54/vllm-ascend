# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import ctypes
import json
from dataclasses import replace
from types import SimpleNamespace

from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1b import _aggregate_result, _t06_kernel_arguments
from tools.pypto_wse_validation.stage1b_contracts import (
    T06_CONTROL_OFFSET,
    T06_DRIVER_KERNEL,
    T06_MAX_INFLIGHT,
    T06_PAYLOAD_BYTES,
    T06_SEQUENCE_COUNT,
    T06_SERVICE_KERNEL,
    T06DeviceLoopReport,
)


def _driver_report() -> T06DeviceLoopReport:
    return T06DeviceLoopReport(
        processed=T06_SEQUENCE_COUNT,
        validation_errors=0,
        sequence_errors=0,
        generation_errors=0,
        checksum_errors=0,
        marker_errors=0,
        timeouts=0,
        elapsed_cycles=10,
        slot0_processed=T06_SEQUENCE_COUNT // 2,
        slot1_processed=T06_SEQUENCE_COUNT // 2,
        credits_acquired=T06_SEQUENCE_COUNT,
        terminal_tasks=T06_SEQUENCE_COUNT,
        credits_returned=T06_SEQUENCE_COUNT,
        max_inflight=T06_MAX_INFLIGHT,
        slot_overwrite_errors=0,
        no_credit_events=1,
        pending_requests=1,
        submission_retry_spins=0,
        service_pause_observed=1,
        service_resume_observed=1,
        progress_after_resume=T06_SEQUENCE_COUNT,
        peak_queue_depth=T06_MAX_INFLIGHT,
        submissions=T06_SEQUENCE_COUNT,
        completions=T06_SEQUENCE_COUNT,
    )


def test_t06_kernel_arguments_match_backpressure_layout():
    arguments = _t06_kernel_arguments(EndpointRole.ATTENTION, 0x100000, 0x500000, 7)
    assert ctypes.sizeof(arguments) == 48
    assert arguments.local_payload == 0x100000
    assert arguments.local_control == 0x100000 + T06_CONTROL_OFFSET
    assert arguments.remote_payload == 0x500000
    assert arguments.remote_control == 0x500000 + T06_CONTROL_OFFSET
    assert arguments.generation == 7
    assert arguments.payload_words == T06_PAYLOAD_BYTES // 8
    assert arguments.sequence_count == T06_SEQUENCE_COUNT


def test_aggregate_result_validates_t06_backpressure_without_claiming_c2(tmp_path):
    processes = {}
    reports = {
        EndpointRole.ATTENTION: _driver_report(),
        EndpointRole.WSE_SURROGATE: replace(_driver_report(), credits_acquired=0, credits_returned=0),
    }
    for device_id, role in enumerate(EndpointRole):
        kernel = T06_DRIVER_KERNEL if role is EndpointRole.ATTENTION else T06_SERVICE_KERNEL
        artifact = {
            "cleanup": {
                "device_kernel": "CLOSED",
                "imported_window": "CLOSED",
                "owned_window": "CLOSED",
                "runtime": "CLOSED",
            },
            "device_loop": {
                "binary_sha256": "a" * 64,
                "elapsed_ns": 100,
                "hot_path": {
                    "completion_messages": 0,
                    "control_bytes": 0,
                    "control_messages": 0,
                    "payload_bytes": 0,
                    "task_messages": 0,
                },
                "kernel": kernel,
                "launches": 1,
                "report": reports[role].to_dict(),
            },
            "host_bounce_bytes": 0,
            "success": True,
        }
        (tmp_path / f"{role.value.lower()}_stage1b.json").write_text(json.dumps(artifact), encoding="utf-8")
        (tmp_path / f"{role.value.lower()}_stage1b.log").write_text("", encoding="utf-8")
        device_logs = tmp_path / f"{role.value.lower()}_device_logs"
        device_logs.mkdir()
        (device_logs / "driver.log").write_text(
            "Enable P2P\nMEM_DEV_SMALL_P2P_HBM current_alloced_size=0\n",
            encoding="utf-8",
        )
        processes[role] = SimpleNamespace(returncode=0, pid=device_id + 1)

    args = SimpleNamespace(
        artifact_dir=tmp_path,
        case_id="T06",
        devices=(0, 1),
        generation=1,
        run_id="run-1",
        start_order="attention-first",
    )
    result = _aggregate_result(args, processes)
    assert result["success"] is True
    assert result["stage1b_progress"] == "T06_PASS"
    assert result["data_results"]["T06"] == {
        "attempts": T06_SEQUENCE_COUNT + 1,
        "bytes": T06_SEQUENCE_COUNT * T06_PAYLOAD_BYTES * 2,
        "passed": T06_SEQUENCE_COUNT,
        "status": "PASS",
    }
    observation = result["observation"]
    assert observation["third_request_outcome"] == "NO_CREDIT"
    assert observation["submission_retry_spins"] == 0
    assert observation["credits_returned"] == T06_SEQUENCE_COUNT
    assert observation["progress_after_resume"] == T06_SEQUENCE_COUNT
    assert result["host_hot_path"] == {
        "completion_messages": 0,
        "control_messages": 0,
        "payload_bytes": 0,
        "task_messages": 0,
    }
    assert result["capability_level"] == "C1"
    assert result["c2_status"] == "NOT_ESTABLISHED"
