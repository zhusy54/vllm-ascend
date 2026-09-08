# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import ctypes
import json
from types import SimpleNamespace

import pytest

from tests.ut.tools.test_pypto_wse_stage1c_contracts import _fault_report
from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1c_contracts import T10_CONTROL_OFFSET, T11_TIMEOUT_CYCLES
from tools.pypto_wse_validation.stage1c_t11 import _aggregate_result, _kernel_arguments


def test_t11_kernel_arguments_use_long_bounded_fault_timeout():
    arguments = _kernel_arguments(0x100000, 0x400000, 7)
    assert ctypes.sizeof(arguments) == 56
    assert arguments.local_control == 0x100000 + T10_CONTROL_OFFSET
    assert arguments.remote_control == 0x400000 + T10_CONTROL_OFFSET
    assert arguments.generation == 7
    assert arguments.timeout_cycles == T11_TIMEOUT_CYCLES
    assert arguments.mode == 11


@pytest.mark.parametrize("victim", tuple(EndpointRole))
def test_t11_aggregate_requires_expected_sigkill_survivor_cleanup_and_recovery(tmp_path, victim):
    survivor = EndpointRole.WSE_SURROGATE if victim is EndpointRole.ATTENTION else EndpointRole.ATTENTION
    artifact = {
        "cleanup": {
            "device_kernel": "CLOSED",
            "imported_window": "CLOSED",
            "owned_window": "CLOSED",
            "runtime": "CLOSED",
        },
        "device_loop": {
            "report": _fault_report(service=survivor is EndpointRole.WSE_SURROGATE, mode=11).to_dict(),
        },
        "old_peer_handle_probe": {"rejected": True, "result_code": 507899},
        "success": True,
    }
    (tmp_path / f"{survivor.value.lower()}_stage1c_t11.json").write_text(json.dumps(artifact), encoding="utf-8")
    processes = {
        victim: SimpleNamespace(returncode=-9),
        survivor: SimpleNamespace(returncode=0),
    }
    for role in EndpointRole:
        (tmp_path / f"{role.value.lower()}_stage1c_t11.log").write_text("", encoding="utf-8")
        (tmp_path / f"{role.value.lower()}_device_logs").mkdir()
    args = SimpleNamespace(
        artifact_dir=tmp_path,
        devices=(0, 1),
        generation=1,
        run_id="run-1",
        start_order="attention-first",
        timeout=60.0,
        victim_role=victim.value,
    )
    recovery = {"observation": {"validated_sequences": 100}, "success": True}
    result = _aggregate_result(
        args,
        processes,
        survivor_elapsed_seconds=20.1,
        recovery_returncode=0,
        recovery_result=recovery,
    )
    assert result["success"] is True
    assert result["stage1c_progress"] == "T11_PASS"
    assert result["observation"]["victim_role"] == victim.value
    assert result["observation"]["survivor_role"] == survivor.value
    assert result["observation"]["old_handle_rejected"] is True
    assert result["observation"]["recovery_success"] is True
