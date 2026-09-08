# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import ctypes
import json
from dataclasses import replace
from types import SimpleNamespace

from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1b import _aggregate_result, _t05_kernel_arguments
from tools.pypto_wse_validation.stage1b_contracts import (
    T05_CONTROL_OFFSET,
    T05_DRIVER_KERNEL,
    T05_MAX_INFLIGHT,
    T05_PAYLOAD_BYTES,
    T05_SEQUENCE_COUNT,
    T05_SERVICE_KERNEL,
    T05DeviceLoopReport,
)


def _driver_report() -> T05DeviceLoopReport:
    return T05DeviceLoopReport(
        processed=T05_SEQUENCE_COUNT,
        validation_errors=0,
        sequence_errors=0,
        generation_errors=0,
        checksum_errors=0,
        marker_errors=0,
        timeouts=0,
        elapsed_cycles=10,
        slot0_processed=T05_SEQUENCE_COUNT // 2,
        slot1_processed=T05_SEQUENCE_COUNT // 2,
        credits_acquired=T05_SEQUENCE_COUNT,
        terminal_tasks=T05_SEQUENCE_COUNT,
        credits_returned=T05_SEQUENCE_COUNT,
        max_inflight=T05_MAX_INFLIGHT,
        out_of_order_completions=T05_SEQUENCE_COUNT // 2,
        slot_overwrite_errors=0,
    )


def test_t05_kernel_arguments_match_dual_slot_device_layout():
    arguments = _t05_kernel_arguments(EndpointRole.ATTENTION, 0x100000, 0x500000, 7)
    assert ctypes.sizeof(arguments) == 56
    assert arguments.local_payload == 0x100000
    assert arguments.local_control == 0x100000 + T05_CONTROL_OFFSET
    assert arguments.remote_payload == 0x500000
    assert arguments.remote_control == 0x500000 + T05_CONTROL_OFFSET
    assert arguments.generation == 7
    assert arguments.slot0_words == T05_PAYLOAD_BYTES[0] // 8
    assert arguments.slot1_words == T05_PAYLOAD_BYTES[1] // 8
    assert arguments.sequence_count == T05_SEQUENCE_COUNT


def test_aggregate_result_validates_t05_pipeline_without_claiming_c2(tmp_path):
    processes = {}
    reports = {
        EndpointRole.ATTENTION: _driver_report(),
        EndpointRole.WSE_SURROGATE: replace(_driver_report(), credits_acquired=0, credits_returned=0),
    }
    for device_id, role in enumerate(EndpointRole):
        kernel = T05_DRIVER_KERNEL if role is EndpointRole.ATTENTION else T05_SERVICE_KERNEL
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
        case_id="T05",
        devices=(0, 1),
        generation=1,
        run_id="run-1",
        start_order="attention-first",
    )
    result = _aggregate_result(args, processes)
    assert result["success"] is True
    assert result["stage1b_progress"] == "T05_PASS"
    assert result["data_results"]["T05"] == {
        "attempts": T05_SEQUENCE_COUNT,
        "bytes": (T05_SEQUENCE_COUNT // 2) * sum(T05_PAYLOAD_BYTES) * 2,
        "passed": T05_SEQUENCE_COUNT,
        "status": "PASS",
    }
    assert result["observation"]["credits_returned"] == T05_SEQUENCE_COUNT
    assert result["observation"]["out_of_order_completions"] == T05_SEQUENCE_COUNT // 2
    assert result["host_hot_path"] == {
        "completion_messages": 0,
        "control_messages": 0,
        "payload_bytes": 0,
        "task_messages": 0,
    }
    assert result["capability_level"] == "C1"
    assert result["c2_status"] == "NOT_ESTABLISHED"
