# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.ut.tools.test_pypto_wse_stage1b import _report
from tests.ut.tools.test_pypto_wse_stage1b_contracts import _observation
from tools.pypto_wse_validation.collect_stage1b import collect_stage1b
from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1b_contracts import T04_SEQUENCE_COUNT


def _write_run(path: Path, start_order: str, *, raw_handle: bool = False) -> None:
    path.mkdir()
    observation = _observation()
    for device_id, role in enumerate(EndpointRole):
        manifest = {"opaque_handle": {"kind": "ACL_VMM_SHAREABLE_HANDLE", "sha256": "a" * 64}}
        if raw_handle:
            manifest["shareable_handle"] = 123
        endpoint = {
            "cleanup": {
                "device_kernel": "CLOSED",
                "imported_window": "CLOSED",
                "owned_window": "CLOSED",
                "runtime": "CLOSED",
            },
            "control": {},
            "device_id": device_id,
            "device_loop": {
                "binary_sha256": "a" * 64,
                "elapsed_ns": 1,
                "hot_path": {},
                "kernel": observation.driver_kernel if role is EndpointRole.ATTENTION else observation.service_kernel,
                "launches": 1,
                "report": _report(),
            },
            "error": None,
            "manifest": manifest,
            "role": role.value,
            "success": True,
        }
        (path / f"{role.value.lower()}_stage1b.json").write_text(json.dumps(endpoint), encoding="utf-8")
    result = {
        "capability_level": "C1",
        "c2_status": "NOT_ESTABLISHED",
        "claim_scope": "NPU_SURROGATE_ONLY",
        "data_results": {"T04": {"passed": T04_SEQUENCE_COUNT, "status": "PASS"}},
        "host_bounce_bytes": 0,
        "host_hot_path": {"completion_messages": 0, "payload_bytes": 0, "task_messages": 0},
        "npu_wse_capability_level": "NOT_ESTABLISHED",
        "observation": observation.to_dict(),
        "resource_cleanup": "VERIFIED",
        "stage1b_progress": "T04_PASS",
        "start_order": start_order,
        "success": True,
    }
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")


def test_collect_requires_both_start_orders_and_preserves_claim_boundary(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first, "attention-first")
    _write_run(second, "wse-first")
    evidence = collect_stage1b(
        successful_runs=(("task-a", first), ("task-b", second)),
        implementation_revision="abc123",
        collected_at="2026-09-08T00:00:00+00:00",
    )
    assert evidence["conclusion"] == {
        "capability_level": "C1",
        "claim_scope": "NPU_SURROGATE_ONLY",
        "c2_status": "NOT_ESTABLISHED",
        "evidence_status": "SIMULATION",
        "npu_wse_capability_level": "NOT_ESTABLISHED",
        "t04": "PASS",
        "t05_t12": "NOT_RUN",
    }


def test_collect_rejects_raw_shareable_handle(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first, "attention-first", raw_handle=True)
    _write_run(second, "wse-first")
    with pytest.raises(ValueError, match="raw capability"):
        collect_stage1b(
            successful_runs=(("task-a", first), ("task-b", second)),
            implementation_revision="abc123",
        )
