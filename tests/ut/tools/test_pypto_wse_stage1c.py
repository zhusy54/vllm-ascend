# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import ctypes
import json
from types import SimpleNamespace

import pytest

from tests.ut.tools.test_pypto_wse_stage1c_contracts import _fault_report, _t09_report, _t12_report
from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1c import _aggregate_result, _kernel_arguments
from tools.pypto_wse_validation.stage1c_contracts import (
    T09_CONTROL_OFFSET,
    T09_DRIVER_KERNEL,
    T09_SERVICE_KERNEL,
    T09_TOTAL_COUNT,
    T10_CONTROL_OFFSET,
    T10_DRIVER_KERNEL,
    T10_SERVICE_KERNEL,
    T10_TIMEOUT_CYCLES,
    T12_ACCEPTED_COUNT,
    T12_CLOSE_ORDER,
    T12_CONTROL_OFFSET,
    T12_DRIVER_KERNEL,
    T12_SERVICE_KERNEL,
)


@pytest.mark.parametrize(
    ("case_id", "control_offset", "expected_size"),
    (("T09", T09_CONTROL_OFFSET, 48), ("T10", T10_CONTROL_OFFSET, 56), ("T12", T12_CONTROL_OFFSET, 40)),
)
def test_kernel_arguments_match_stage1c_layout(case_id, control_offset, expected_size):
    arguments = _kernel_arguments(case_id, 0x100000, 0x400000, 3)
    assert ctypes.sizeof(arguments) == expected_size
    assert arguments.local_payload == 0x100000
    assert arguments.local_control == 0x100000 + control_offset
    assert arguments.remote_payload == 0x400000
    assert arguments.remote_control == 0x400000 + control_offset
    assert arguments.generation == 3
    if case_id == "T09":
        assert arguments.total_count == T09_TOTAL_COUNT
    elif case_id == "T10":
        assert arguments.timeout_cycles == T10_TIMEOUT_CYCLES
        assert arguments.mode == 10


def _write_artifacts(tmp_path, case_id: str):
    kernels = {
        "T09": (T09_DRIVER_KERNEL, T09_SERVICE_KERNEL),
        "T10": (T10_DRIVER_KERNEL, T10_SERVICE_KERNEL),
        "T12": (T12_DRIVER_KERNEL, T12_SERVICE_KERNEL),
    }[case_id]
    processes = {}
    for index, role in enumerate(EndpointRole):
        driver = role is EndpointRole.ATTENTION
        if case_id == "T09":
            report = _t09_report(driver)
        elif case_id == "T10":
            report = _fault_report(service=not driver, mode=10)
        else:
            report = _t12_report(driver)
        artifact = {
            "cleanup": {
                "device_kernel": "CLOSED",
                "imported_window": "CLOSED",
                "owned_window": "CLOSED",
                "runtime": "CLOSED",
            },
            "device_loop": {
                "binary_sha256": ("a" if driver else "b") * 64,
                "hot_path": {
                    "completion_messages": 0,
                    "control_bytes": 0,
                    "control_messages": 0,
                    "payload_bytes": 0,
                    "task_messages": 0,
                },
                "kernel": kernels[0 if driver else 1],
                "launches": 1,
                "report": report.to_dict(),
            },
            "host_bounce_bytes": 0,
            "lifecycle": {
                "close_order": list(T12_CLOSE_ORDER) if case_id == "T12" else [],
                "duplicate_close_attempts": 4 if case_id == "T12" else 0,
                "idempotent_close_successes": 4 if case_id == "T12" else 0,
                "duplicate_close_rejections": 0,
            },
            "metrics": {"host_memory_growth_bytes": 0, "host_cpu_utilization_pct": 1.0},
            "success": True,
        }
        (tmp_path / f"{role.value.lower()}_stage1c.json").write_text(json.dumps(artifact), encoding="utf-8")
        (tmp_path / f"{role.value.lower()}_stage1c.log").write_text("", encoding="utf-8")
        log_dir = tmp_path / f"{role.value.lower()}_device_logs"
        log_dir.mkdir()
        (log_dir / "driver.log").write_text(
            "Enable P2P\nMEM_DEV_SMALL_P2P_HBM current_alloced_size=0\n",
            encoding="utf-8",
        )
        processes[role] = SimpleNamespace(returncode=0, pid=index + 1)
    return processes


@pytest.mark.parametrize(("case_id", "passed"), (("T09", 10100), ("T10", 1), ("T12", T12_ACCEPTED_COUNT)))
def test_aggregate_result_accepts_complete_stage1c_evidence(tmp_path, case_id, passed):
    processes = _write_artifacts(tmp_path, case_id)
    args = SimpleNamespace(
        artifact_dir=tmp_path,
        case_id=case_id,
        devices=(0, 1),
        generation=1,
        run_id="run-1",
        start_order="attention-first",
    )
    result = _aggregate_result(args, processes)
    assert result["success"] is True
    assert result["stage1c_progress"] == f"{case_id}_PASS"
    assert result["data_results"][case_id]["passed"] == passed
    assert result["capability_level"] == "C1"
    assert result["c2_status"] == "NOT_ESTABLISHED"
