# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import ctypes
import json
from types import SimpleNamespace

from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1a import _message
from tools.pypto_wse_validation.stage1b_contracts import (
    T08_BASELINE_SEQUENCE_COUNT,
    T08_CONTROL_OFFSET,
    T08_DRIVER_KERNEL,
    T08_NEXT_SEQUENCE_COUNT,
    T08_PAYLOAD_BYTES,
    T08_SERVICE_KERNEL,
    T08DeviceLoopReport,
)
from tools.pypto_wse_validation.stage1b_t08 import _aggregate_result, _kernel_arguments


def _report(*, driver: bool, mode: int) -> dict[str, int]:
    sequences = T08_BASELINE_SEQUENCE_COUNT if mode == 0 else T08_NEXT_SEQUENCE_COUNT
    return T08DeviceLoopReport(
        processed=sequences,
        validation_errors=0,
        sequence_errors=0,
        generation_errors=0,
        checksum_errors=0,
        marker_errors=0,
        timeouts=0,
        elapsed_cycles=10,
        submissions=sequences,
        completions=sequences,
        stale_descriptor_injections=int(driver and mode == 1),
        stale_descriptor_rejections=int(not driver and mode == 1),
        stale_completion_injections=int(not driver and mode == 1),
        stale_completion_rejections=int(driver and mode == 1),
        old_completion_credit_releases=0,
        current_slot_preserved=int(mode == 1),
        credits_acquired=sequences if driver else 0,
        credits_returned=sequences if driver else 0,
        terminal_tasks=sequences,
        progress_after_stale=sequences if mode == 1 else 0,
        input_fences=sequences if driver else 0,
        output_fences=0 if driver else sequences,
        mode=mode,
        reserved=0,
    ).to_dict()


def test_kernel_arguments_match_t08_generation_layout():
    arguments = _kernel_arguments(
        0x100000,
        0x400000,
        generation=8,
        previous_generation=7,
        sequence_count=1,
        mode=1,
    )
    assert ctypes.sizeof(arguments) == 64
    assert arguments.local_payload == 0x100000
    assert arguments.local_control == 0x100000 + T08_CONTROL_OFFSET
    assert arguments.remote_payload == 0x400000
    assert arguments.remote_control == 0x400000 + T08_CONTROL_OFFSET
    assert arguments.generation == 8
    assert arguments.previous_generation == 7
    assert arguments.payload_words == T08_PAYLOAD_BYTES // 8
    assert arguments.sequence_count == 1
    assert arguments.mode == 1


def test_probe_result_is_an_allowed_metadata_only_message():
    message = _message(
        "PROBE_RESULT",
        run_id="run-1",
        generation=2,
        role=EndpointRole.ATTENTION,
        fields={"api": "aclrtMemImportFromShareableHandle", "rejected": True, "result_code": 145001},
    )
    assert message["type"] == "PROBE_RESULT"
    assert message["rejected"] is True


def test_aggregate_result_requires_two_clean_generations_and_stale_rejection(tmp_path):
    processes = {}
    for device_id, role in enumerate(EndpointRole):
        driver = role is EndpointRole.ATTENTION
        kernel = T08_DRIVER_KERNEL if driver else T08_SERVICE_KERNEL
        binary_hash = "a" * 64 if driver else "b" * 64
        artifact = {
            "cleanup": {
                generation: {
                    "device_kernel": "CLOSED",
                    "imported_window": "CLOSED",
                    "owned_window": "CLOSED",
                    "runtime": "CLOSED",
                }
                for generation in ("baseline", "next")
            },
            "generation_runs": [
                {
                    "binary_sha256": binary_hash,
                    "generation": 1 + mode,
                    "hot_path": {
                        "completion_messages": 0,
                        "control_bytes": 0,
                        "control_messages": 0,
                        "payload_bytes": 0,
                        "task_messages": 0,
                    },
                    "kernel": kernel,
                    "launches": 1,
                    "mode": mode,
                    "report": _report(driver=driver, mode=mode),
                }
                for mode in (0, 1)
            ],
            "handle_invalidation": {
                "api": "aclrtMemImportFromShareableHandle",
                "before_reallocate": {"rejected": True, "result_code": 145001},
                "after_reallocate": {"rejected": True, "result_code": 145001},
                "local_handle_changed": True,
                "peer_handle_changed": True,
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
    assert result["stage1b_progress"] == "T08_PASS"
    assert result["data_results"]["T08"]["passed"] == 5
    assert result["observation"]["old_handle_rejected_before_reallocate"] == 2
    assert result["observation"]["old_handle_rejected_after_reallocate"] == 2
    assert result["observation"]["old_completion_credit_releases"] == 0
    assert result["observation"]["current_slot_preserved"] is True
    assert result["capability_level"] == "C1"
    assert result["c2_status"] == "NOT_ESTABLISHED"
