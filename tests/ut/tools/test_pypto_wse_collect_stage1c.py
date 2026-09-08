# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ut.tools.test_pypto_wse_stage1c import _write_artifacts
from tests.ut.tools.test_pypto_wse_stage1c_contracts import _fault_report
from tools.pypto_wse_validation.collect_stage1c import PRIOR_CASES, collect_stage1c
from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1c import _aggregate_result as _aggregate_normal
from tools.pypto_wse_validation.stage1c_t11 import _aggregate_result as _aggregate_t11


def _write_prior(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for case_id in PRIOR_CASES:
        path = tmp_path / f"{case_id}.json"
        conclusion = {
            "claim_scope": "NPU_SURROGATE_ONLY",
            "evidence_status": "SIMULATION",
        }
        if case_id == "T01_T03":
            conclusion.update({"t01": "PASS", "t02": "PASS", "t03": "PASS"})
        else:
            conclusion[case_id.lower()] = "PASS"
        path.write_text(json.dumps({"conclusion": conclusion}), encoding="utf-8")
        paths[case_id] = path
    return paths


def _write_normal_run(path: Path, case_id: str, start_order: str) -> tuple[str, Path]:
    path.mkdir()
    processes = _write_artifacts(path, case_id)
    args = SimpleNamespace(
        artifact_dir=path,
        devices=(0, 1),
        generation=1,
        run_id=f"run-{case_id}",
        start_order=start_order,
        case_id=case_id,
    )
    result = _aggregate_normal(args, processes)
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return f"task-{case_id}-{start_order}", path


def _write_t11_run(path: Path, victim: EndpointRole) -> tuple[str, Path]:
    path.mkdir()
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
        "old_peer_handle_probe": {
            "opaque_handle": {"kind": "ACL_VMM_SHAREABLE_HANDLE", "sha256": "a" * 64},
            "rejected": True,
            "result_code": 507899,
        },
        "success": True,
    }
    (path / f"{survivor.value.lower()}_stage1c_t11.json").write_text(json.dumps(artifact), encoding="utf-8")
    processes = {victim: SimpleNamespace(returncode=-9), survivor: SimpleNamespace(returncode=0)}
    for role in EndpointRole:
        (path / f"{role.value.lower()}_stage1c_t11.log").write_text("", encoding="utf-8")
        (path / f"{role.value.lower()}_device_logs").mkdir()
    args = SimpleNamespace(
        artifact_dir=path,
        devices=(0, 1),
        generation=1,
        run_id="run-T11",
        start_order="wse-first" if victim is EndpointRole.ATTENTION else "attention-first",
        timeout=60.0,
        victim_role=victim.value,
    )
    result = _aggregate_t11(
        args,
        processes,
        survivor_elapsed_seconds=20.1,
        recovery_returncode=0,
        recovery_result={"observation": {"validated_sequences": 100}, "success": True},
    )
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return f"task-T11-{victim.value}", path


def test_collect_stage1c_establishes_only_host_local_surrogate_c2(tmp_path):
    runs = {
        case_id: [
            _write_normal_run(tmp_path / f"{case_id}-a", case_id, "attention-first"),
            _write_normal_run(tmp_path / f"{case_id}-w", case_id, "wse-first"),
        ]
        for case_id in ("T09", "T10", "T12")
    }
    runs["T11"] = [
        _write_t11_run(tmp_path / "T11-a", EndpointRole.ATTENTION),
        _write_t11_run(tmp_path / "T11-w", EndpointRole.WSE_SURROGATE),
    ]
    evidence = collect_stage1c(
        successful_runs=runs,
        prior_evidence=_write_prior(tmp_path),
        implementation_revision="abc123",
        collected_at="2026-09-08T00:00:00+00:00",
    )
    assert evidence["conclusion"] == {
        "capability_level": "C2",
        "claim_scope": "HOST_LOCAL_NPU_SURROGATE_ONLY",
        "c2_status": "ESTABLISHED_HOST_LOCAL_NPU_SURROGATE",
        "evidence_status": "SIMULATION",
        "npu_wse_capability_level": "NOT_ESTABLISHED",
        "t01_t12": "PASS",
        "t09": "PASS",
        "t10": "PASS",
        "t11": "PASS",
        "t12": "PASS",
    }


def test_collect_stage1c_rejects_missing_fault_direction(tmp_path):
    runs = {
        case_id: [
            _write_normal_run(tmp_path / f"{case_id}-a", case_id, "attention-first"),
            _write_normal_run(tmp_path / f"{case_id}-w", case_id, "wse-first"),
        ]
        for case_id in ("T09", "T10", "T12")
    }
    first = _write_t11_run(tmp_path / "T11-a", EndpointRole.ATTENTION)
    second = _write_t11_run(tmp_path / "T11-a2", EndpointRole.ATTENTION)
    runs["T11"] = [first, second]
    with pytest.raises(ValueError, match="both process-exit victims"):
        collect_stage1c(
            successful_runs=runs,
            prior_evidence=_write_prior(tmp_path),
            implementation_revision="abc123",
        )
