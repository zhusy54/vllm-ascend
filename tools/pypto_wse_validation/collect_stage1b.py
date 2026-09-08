# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Collect and validate sanitized Stage 1B device-loop evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.pypto_wse_validation.contracts import EndpointRole
from tools.pypto_wse_validation.stage1b_contracts import (
    T04_CASE_ID,
    T04_SEQUENCE_COUNT,
    T05_CASE_ID,
    T05_SEQUENCE_COUNT,
    T06_CASE_ID,
    T06_SEQUENCE_COUNT,
    T07_CASE_ID,
    T07_SEQUENCE_COUNT,
    T04Observation,
    T05Observation,
    T06Observation,
    T07Observation,
)

SCHEMA_VERSION = 1
FORBIDDEN_ARTIFACT_KEYS = frozenset({"address", "device_addr", "shareable_handle"})


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Stage 1B artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Stage 1B artifact must be an object: {path}")
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


def _endpoint_evidence(run_dir: Path, role: EndpointRole) -> dict[str, Any]:
    endpoint = _read_json(run_dir / f"{role.value.lower()}_stage1b.json")
    evidence = {
        "cleanup": endpoint.get("cleanup"),
        "control": endpoint.get("control"),
        "device_id": endpoint.get("device_id"),
        "device_loop": endpoint.get("device_loop"),
        "error": endpoint.get("error"),
        "manifest": endpoint.get("manifest"),
        "role": endpoint.get("role"),
        "success": endpoint.get("success"),
    }
    forbidden = _find_forbidden_keys(evidence)
    if forbidden:
        raise ValueError(f"endpoint artifact contains raw capability fields: {sorted(forbidden)}")
    return evidence


def _validate_success(result: Mapping[str, Any], endpoints: Mapping[str, Mapping[str, Any]], case_id: str) -> None:
    observation_raw = result.get("observation")
    if case_id == T04_CASE_ID:
        observation_type = T04Observation
        sequence_count = T04_SEQUENCE_COUNT
    elif case_id == T05_CASE_ID:
        observation_type = T05Observation
        sequence_count = T05_SEQUENCE_COUNT
    elif case_id == T06_CASE_ID:
        observation_type = T06Observation
        sequence_count = T06_SEQUENCE_COUNT
    else:
        observation_type = T07Observation
        sequence_count = T07_SEQUENCE_COUNT
    if not isinstance(observation_raw, dict) or not observation_type.from_dict(observation_raw).passed:
        raise ValueError(f"successful run is missing valid {case_id} device-loop evidence")
    case_result = result.get("data_results", {}).get(case_id, {})
    if (
        result.get("success") is not True
        or result.get("stage1b_progress") != f"{case_id}_PASS"
        or case_result.get("status") != "PASS"
        or case_result.get("passed") != sequence_count
    ):
        raise ValueError(f"successful run has incomplete {case_id} results")
    if result.get("capability_level") != "C1" or result.get("c2_status") != "NOT_ESTABLISHED":
        raise ValueError(f"{case_id}-only run overclaims C2")
    if result.get("claim_scope") != "NPU_SURROGATE_ONLY":
        raise ValueError("successful run has an unsafe claim scope")
    if result.get("npu_wse_capability_level") != "NOT_ESTABLISHED":
        raise ValueError("successful run overclaims real NPU-WSE capability")
    if result.get("host_hot_path") != {
        "completion_messages": 0,
        "control_messages": 0,
        "payload_bytes": 0,
        "task_messages": 0,
    }:
        raise ValueError(f"successful {case_id} run contains Host hot-path traffic")
    if result.get("resource_cleanup") != "VERIFIED" or result.get("host_bounce_bytes") != 0:
        raise ValueError("successful run is missing cleanup or zero-bounce proof")
    for role in EndpointRole:
        endpoint = endpoints[role.value]
        if endpoint.get("success") is not True or endpoint.get("error") is not None:
            raise ValueError(f"successful run endpoint failed: {role.value}")


def _load_run(task_id: str, run_dir: Path, case_id: str) -> dict[str, Any]:
    result = _read_json(run_dir / "result.json")
    endpoints = {role.value: _endpoint_evidence(run_dir, role) for role in EndpointRole}
    _validate_success(result, endpoints, case_id)
    return {"endpoints": endpoints, "result": result, "task_id": task_id}


def collect_stage1b(
    *,
    successful_runs: Sequence[tuple[str, Path]],
    implementation_revision: str,
    case_id: str = T04_CASE_ID,
    collected_at: str | None = None,
) -> dict[str, Any]:
    if case_id not in (T04_CASE_ID, T05_CASE_ID, T06_CASE_ID, T07_CASE_ID):
        raise ValueError(f"unsupported Stage 1B case: {case_id}")
    if len(successful_runs) < 2:
        raise ValueError("at least two successful Stage 1B runs are required")
    loaded = [_load_run(task_id, run_dir, case_id) for task_id, run_dir in successful_runs]
    if {run["result"].get("start_order") for run in loaded} != {"attention-first", "wse-first"}:
        raise ValueError("successful evidence must cover both endpoint start orders")
    conclusion = {
        "capability_level": "C1",
        "claim_scope": "NPU_SURROGATE_ONLY",
        "c2_status": "NOT_ESTABLISHED",
        "evidence_status": "SIMULATION",
        "npu_wse_capability_level": "NOT_ESTABLISHED",
    }
    if case_id == T04_CASE_ID:
        conclusion.update({"t04": "PASS", "t05_t12": "NOT_RUN"})
    elif case_id == T05_CASE_ID:
        conclusion.update({"t05": "PASS", "t06_t12": "NOT_RUN", "validated_case": case_id})
    elif case_id == T06_CASE_ID:
        conclusion.update({"t06": "PASS", "t07_t12": "NOT_RUN", "validated_case": case_id})
    else:
        conclusion.update({"t07": "PASS", "t08_t12": "NOT_RUN", "validated_case": case_id})
    return {
        "collected_at": collected_at or datetime.now(UTC).isoformat(),
        "conclusion": conclusion,
        "implementation_revision": implementation_revision,
        "profile": "NPU_SURROGATE",
        "schema_version": SCHEMA_VERSION,
        "successful_runs": loaded,
    }


def _parse_run(value: str) -> tuple[str, Path]:
    task_id, separator, raw_path = value.partition("=")
    if not separator or not task_id or not raw_path:
        raise argparse.ArgumentTypeError("run must use TASK_ID=ARTIFACT_DIR")
    return task_id, Path(raw_path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successful-run", action="append", type=_parse_run, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument(
        "--case-id",
        choices=(T04_CASE_ID, T05_CASE_ID, T06_CASE_ID, T07_CASE_ID),
        default=T04_CASE_ID,
    )
    parser.add_argument("--collected-at")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence = collect_stage1b(
        successful_runs=args.successful_run,
        implementation_revision=args.implementation_revision,
        case_id=args.case_id,
        collected_at=args.collected_at,
    )
    _write_json(args.output, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
