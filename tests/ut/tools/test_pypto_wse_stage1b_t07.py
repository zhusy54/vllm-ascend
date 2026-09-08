# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import ctypes
import json
from dataclasses import replace
from types import SimpleNamespace

from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1b import _aggregate_result, _t07_kernel_arguments
from tools.pypto_wse_validation.stage1b_contracts import (
    T07_CONTROL_OFFSET,
    T07_DRIVER_KERNEL,
    T07_MAX_INFLIGHT,
    T07_PAYLOAD_BYTES,
    T07_SEQUENCE_COUNT,
    T07_SERVICE_KERNEL,
    T07DeviceLoopReport,
)


def _driver_report() -> T07DeviceLoopReport:
    return T07DeviceLoopReport(
        processed=T07_SEQUENCE_COUNT,
        validation_errors=0,
        sequence_errors=0,
        generation_errors=0,
        checksum_errors=0,
        head_marker_errors=0,
        tail_marker_errors=0,
        stale_payload_errors=0,
        incomplete_payload_errors=0,
        premature_completion_errors=0,
        timeouts=0,
        elapsed_cycles=10,
        slot0_processed=T07_SEQUENCE_COUNT // 2,
        slot1_processed=T07_SEQUENCE_COUNT // 2,
        input_publish_fences=T07_SEQUENCE_COUNT,
        input_visibility_checks=0,
        output_publish_fences=0,
        immediate_completion_checks=T07_SEQUENCE_COUNT,
        post_completion_delay_cycles=0,
        submissions=T07_SEQUENCE_COUNT,
        completions=T07_SEQUENCE_COUNT,
        slot_overwrite_errors=0,
        payload_words_validated=T07_SEQUENCE_COUNT * (T07_PAYLOAD_BYTES // 8),
        unique_tail_markers_validated=T07_SEQUENCE_COUNT,
        max_inflight=T07_MAX_INFLIGHT,
        reserved=0,
    )


def _service_report() -> T07DeviceLoopReport:
    return replace(
        _driver_report(),
        input_publish_fences=0,
        input_visibility_checks=T07_SEQUENCE_COUNT,
        output_publish_fences=T07_SEQUENCE_COUNT,
        immediate_completion_checks=0,
    )


def test_t07_kernel_arguments_match_large_payload_layout():
    arguments = _t07_kernel_arguments(EndpointRole.ATTENTION, 0x100000, 0x500000, 7)
    assert ctypes.sizeof(arguments) == 48
    assert arguments.local_payload == 0x100000
    assert arguments.local_control == 0x100000 + T07_CONTROL_OFFSET
    assert arguments.remote_payload == 0x500000
    assert arguments.remote_control == 0x500000 + T07_CONTROL_OFFSET
    assert arguments.generation == 7
    assert arguments.payload_words == T07_PAYLOAD_BYTES // 8
    assert arguments.sequence_count == T07_SEQUENCE_COUNT


def test_aggregate_result_validates_t07_visibility_without_claiming_c2(tmp_path):
    processes = {}
    reports = {
        EndpointRole.ATTENTION: _driver_report(),
        EndpointRole.WSE_SURROGATE: _service_report(),
    }
    for device_id, role in enumerate(EndpointRole):
        kernel = T07_DRIVER_KERNEL if role is EndpointRole.ATTENTION else T07_SERVICE_KERNEL
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
        case_id="T07",
        devices=(0, 1),
        generation=1,
        run_id="run-1",
        start_order="attention-first",
    )
    result = _aggregate_result(args, processes)
    assert result["success"] is True
    assert result["stage1b_progress"] == "T07_PASS"
    assert result["data_results"]["T07"] == {
        "attempts": T07_SEQUENCE_COUNT,
        "bytes": T07_SEQUENCE_COUNT * T07_PAYLOAD_BYTES * 2,
        "passed": T07_SEQUENCE_COUNT,
        "status": "PASS",
    }
    observation = result["observation"]
    assert observation["unique_tail_markers_validated"] == T07_SEQUENCE_COUNT
    assert observation["immediate_completion_checks"] == T07_SEQUENCE_COUNT
    assert observation["post_completion_delay_cycles"] == 0
    assert observation["stale_payload_errors"] == 0
    assert result["host_hot_path"] == {
        "completion_messages": 0,
        "control_messages": 0,
        "payload_bytes": 0,
        "task_messages": 0,
    }
    assert result["capability_level"] == "C1"
    assert result["c2_status"] == "NOT_ESTABLISHED"
