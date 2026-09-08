# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.ut.tools.test_pypto_wse_stage1a_contracts import _observation
from tools.pypto_wse_validation.collect_stage1a import collect_stage1a
from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1a_contracts import BASE_PAYLOAD_SIZES, TransferDirection


def _write_run(path: Path, start_order: str, *, raw_handle: bool = False, success: bool = True) -> None:
    path.mkdir()
    observations = {
        EndpointRole.ATTENTION: [
            _observation(TransferDirection.NPU_TO_WSE_SURROGATE, size) for size in BASE_PAYLOAD_SIZES
        ],
        EndpointRole.WSE_SURROGATE: [
            replace(
                _observation(TransferDirection.WSE_SURROGATE_TO_NPU, size),
                sequence_id=index + len(BASE_PAYLOAD_SIZES),
            )
            for index, size in enumerate(BASE_PAYLOAD_SIZES, start=1)
        ],
    }
    for device_id, role in enumerate(EndpointRole):
        manifest = {"opaque_handle": {"kind": "ACL_VMM_SHAREABLE_HANDLE", "sha256": "a" * 64}}
        if raw_handle:
            manifest["shareable_handle"] = 123
        endpoint = {
            "cleanup": {"imported_window": "CLOSED", "owned_window": "CLOSED", "runtime": "CLOSED"},
            "control": {},
            "device_id": device_id,
            "error": None,
            "manifest": manifest,
            "observations": [item.to_dict() for item in observations[role]],
            "role": role.value,
            "success": success,
        }
        (path / f"{role.value.lower()}_stage1a.json").write_text(json.dumps(endpoint), encoding="utf-8")
    result = {
        "capability_level": "C1" if success else "NONE",
        "claim_scope": "NPU_SURROGATE_ONLY",
        "data_results": {
            direction.value: {"passed": len(BASE_PAYLOAD_SIZES), "status": "PASS"} for direction in TransferDirection
        },
        "host_bounce_bytes": 0,
        "npu_wse_capability_level": "NOT_ESTABLISHED",
        "resource_cleanup": "VERIFIED",
        "start_order": start_order,
        "success": success,
    }
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")


def test_collect_requires_two_start_orders_and_preserves_claim_boundary(tmp_path):
    attention = tmp_path / "attention"
    surrogate = tmp_path / "surrogate"
    diagnostic = tmp_path / "diagnostic"
    _write_run(attention, "attention-first")
    _write_run(surrogate, "wse-first")
    _write_run(diagnostic, "attention-first", success=False)
    evidence = collect_stage1a(
        successful_runs=(("task-a", attention), ("task-b", surrogate)),
        diagnostic_runs=(("task-c", diagnostic),),
        implementation_revision="abc123",
        collected_at="2026-09-08T00:00:00+00:00",
    )
    assert evidence["conclusion"] == {
        "capability_level": "C1",
        "claim_scope": "NPU_SURROGATE_ONLY",
        "evidence_status": "SIMULATION",
        "npu_wse_capability_level": "NOT_ESTABLISHED",
        "t01": "PASS",
        "t02": "PASS",
        "t03": "NOT_RUN",
    }
    assert [item["task_id"] for item in evidence["successful_runs"]] == ["task-a", "task-b"]


def test_collect_rejects_raw_shareable_handle(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first, "attention-first", raw_handle=True)
    _write_run(second, "wse-first")
    with pytest.raises(ValueError, match="raw capability"):
        collect_stage1a(
            successful_runs=(("task-a", first), ("task-b", second)),
            diagnostic_runs=(),
            implementation_revision="abc123",
        )
