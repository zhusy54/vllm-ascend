# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Collect T01-T12 evidence and establish host-local NPU-surrogate C2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1c_contracts import (
    T09_CASE_ID,
    T10_CASE_ID,
    T11_CASE_ID,
    T12_CASE_ID,
    T09Observation,
    T10Observation,
    T11Observation,
    T12Observation,
)

SCHEMA_VERSION = 1
FORBIDDEN_ARTIFACT_KEYS = frozenset({"address", "device_addr", "shareable_handle"})
PRIOR_CASES = ("T01_T03", "T04", "T05", "T06", "T07", "T08")
STAGE1C_CASES = (T09_CASE_ID, T10_CASE_ID, T11_CASE_ID, T12_CASE_ID)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"evidence must be an object: {path}")
    return value


def _find_forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_ARTIFACT_KEYS:
                found.add(str(key).lower())
            found.update(_find_forbidden_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_find_forbidden_keys(child))
    return found


def _file_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def _validate_prior(case_id: str, path: Path) -> dict[str, Any]:
    evidence = _read_json(path)
    conclusion = evidence.get("conclusion", {})
    required = ("t01", "t02", "t03") if case_id == "T01_T03" else (case_id.lower(),)
    if any(conclusion.get(key) != "PASS" for key in required):
        raise ValueError(f"prior evidence does not prove {case_id}: {path}")
    if conclusion.get("claim_scope") != "NPU_SURROGATE_ONLY" or conclusion.get("evidence_status") != "SIMULATION":
        raise ValueError(f"prior evidence has incompatible scope: {path}")
    return {"case_id": case_id, "conclusion": conclusion, "file": _file_identity(path)}


def _observation(case_id: str, raw: object):
    if not isinstance(raw, dict):
        raise ValueError(f"{case_id} run is missing an observation")
    observation_types = {
        T09_CASE_ID: T09Observation,
        T10_CASE_ID: T10Observation,
        T11_CASE_ID: T11Observation,
        T12_CASE_ID: T12Observation,
    }
    observation = observation_types[case_id].from_dict(raw)
    if not observation.passed:
        raise ValueError(f"{case_id} observation failed its contract")
    return observation


def _sanitized_endpoint(run_dir: Path, case_id: str, role: EndpointRole) -> dict[str, Any] | None:
    if case_id == T11_CASE_ID:
        path = run_dir / f"{role.value.lower()}_stage1c_t11.json"
    else:
        path = run_dir / f"{role.value.lower()}_stage1c.json"
    if not path.is_file():
        return None
    endpoint = _read_json(path)
    sanitized = {
        key: endpoint.get(key)
        for key in (
            "case_id",
            "cleanup",
            "control",
            "device_id",
            "device_loop",
            "error",
            "generation",
            "host_bounce_bytes",
            "lifecycle",
            "manifest",
            "metrics",
            "old_peer_handle_probe",
            "role",
            "success",
        )
    }
    forbidden = _find_forbidden_keys(sanitized)
    if forbidden:
        raise ValueError(f"endpoint artifact contains raw capability fields: {sorted(forbidden)}")
    return sanitized


def _load_run(task_id: str, run_dir: Path, case_id: str) -> dict[str, Any]:
    result = _read_json(run_dir / "result.json")
    observation = _observation(case_id, result.get("observation"))
    if (
        result.get("success") is not True
        or result.get("stage1c_progress") != f"{case_id}_PASS"
        or result.get("data_results", {}).get(case_id, {}).get("status") != "PASS"
        or result.get("claim_scope") != "NPU_SURROGATE_ONLY"
        or result.get("evidence_status") != "SIMULATION"
        or result.get("npu_wse_capability_level") != "NOT_ESTABLISHED"
    ):
        raise ValueError(f"{case_id} run result is incomplete: {run_dir}")
    endpoints = {role.value: _sanitized_endpoint(run_dir, case_id, role) for role in EndpointRole}
    sanitized_result = dict(result)
    forbidden = _find_forbidden_keys(sanitized_result)
    if forbidden:
        raise ValueError(f"result contains raw capability fields: {sorted(forbidden)}")
    return {
        "endpoints": endpoints,
        "observation": observation.to_dict(),
        "result": sanitized_result,
        "task_id": task_id,
    }


def collect_stage1c(
    *,
    successful_runs: Mapping[str, Sequence[tuple[str, Path]]],
    prior_evidence: Mapping[str, Path],
    implementation_revision: str,
    collected_at: str | None = None,
) -> dict[str, Any]:
    if set(successful_runs) != set(STAGE1C_CASES):
        raise ValueError("successful runs must contain exactly T09, T10, T11, and T12")
    if set(prior_evidence) != set(PRIOR_CASES):
        raise ValueError("prior evidence must contain T01_T03 and T04 through T08")
    prior = [_validate_prior(case_id, prior_evidence[case_id]) for case_id in PRIOR_CASES]
    loaded: dict[str, list[dict[str, Any]]] = {}
    for case_id in STAGE1C_CASES:
        runs = successful_runs[case_id]
        if len(runs) != 2:
            raise ValueError(f"{case_id} requires exactly two successful runs")
        loaded[case_id] = [_load_run(task_id, run_dir, case_id) for task_id, run_dir in runs]
    for case_id in (T09_CASE_ID, T10_CASE_ID, T12_CASE_ID):
        if {run["result"].get("start_order") for run in loaded[case_id]} != {
            "attention-first",
            "wse-first",
        }:
            raise ValueError(f"{case_id} must cover both endpoint start orders")
    if {run["observation"]["victim_role"] for run in loaded[T11_CASE_ID]} != {
        EndpointRole.ATTENTION.value,
        EndpointRole.WSE_SURROGATE.value,
    }:
        raise ValueError("T11 must cover both process-exit victims")
    return {
        "collected_at": collected_at or datetime.now(UTC).isoformat(),
        "conclusion": {
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
        },
        "implementation_revision": implementation_revision,
        "prior_evidence": prior,
        "profile": "NPU_SURROGATE",
        "schema_version": SCHEMA_VERSION,
        "successful_runs": loaded,
    }


def _parse_assignment(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("assignment must use NAME=PATH")
    return name, Path(raw_path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successful-run", action="append", type=_parse_assignment, required=True)
    parser.add_argument("--prior-evidence", action="append", type=_parse_assignment, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--collected-at")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs: dict[str, list[tuple[str, Path]]] = {case_id: [] for case_id in STAGE1C_CASES}
    for name, path in args.successful_run:
        case_id, separator, task_id = name.partition(":")
        if not separator or case_id not in runs or not task_id:
            raise ValueError("successful run name must use T09:TASK_ID through T12:TASK_ID")
        runs[case_id].append((task_id, path))
    prior = dict(args.prior_evidence)
    evidence = collect_stage1c(
        successful_runs=runs,
        prior_evidence=prior,
        implementation_revision=args.implementation_revision,
        collected_at=args.collected_at,
    )
    _write_json(args.output, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
