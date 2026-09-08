# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import ctypes
import json
from types import SimpleNamespace

from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1b import _aggregate_result, _kernel_arguments
from tools.pypto_wse_validation.stage1b_contracts import (
    T04_CONTROL_OFFSET,
    T04_DRIVER_KERNEL,
    T04_PAYLOAD_BYTES,
    T04_SEQUENCE_COUNT,
    T04_SERVICE_KERNEL,
    DeviceLoopReport,
)


def _report() -> dict[str, int]:
    return DeviceLoopReport(
        processed=T04_SEQUENCE_COUNT,
        validation_errors=0,
        sequence_errors=0,
        generation_errors=0,
        checksum_errors=0,
        marker_errors=0,
        timeouts=0,
        elapsed_cycles=10,
    ).to_dict()


def test_kernel_arguments_match_one_slot_device_layout():
    arguments = _kernel_arguments(EndpointRole.ATTENTION, 0x100000, 0x400000, 7)
    assert ctypes.sizeof(arguments) == 48
    assert arguments.local_payload == 0x100000
    assert arguments.local_control == 0x100000 + T04_CONTROL_OFFSET
    assert arguments.remote_payload == 0x400000
    assert arguments.remote_control == 0x400000 + T04_CONTROL_OFFSET
    assert arguments.generation == 7
    assert arguments.payload_words == T04_PAYLOAD_BYTES // 8
    assert arguments.sequence_count == T04_SEQUENCE_COUNT


def test_aggregate_result_preserves_c1_boundary_after_t04(tmp_path):
    processes = {}
    for device_id, role in enumerate(EndpointRole):
        kernel = T04_DRIVER_KERNEL if role is EndpointRole.ATTENTION else T04_SERVICE_KERNEL
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
                "report": _report(),
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
        devices=(0, 1),
        generation=1,
        run_id="run-1",
        start_order="attention-first",
    )
    result = _aggregate_result(args, processes)
    assert result["success"] is True
    assert result["stage1b_progress"] == "T04_PASS"
    assert result["data_results"]["T04"]["passed"] == T04_SEQUENCE_COUNT
    assert result["host_hot_path"] == {
        "completion_messages": 0,
        "control_messages": 0,
        "payload_bytes": 0,
        "task_messages": 0,
    }
    assert result["capability_level"] == "C1"
    assert result["c2_status"] == "NOT_ESTABLISHED"
    assert result["npu_wse_capability_level"] == "NOT_ESTABLISHED"
