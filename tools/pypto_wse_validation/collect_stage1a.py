# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Collect and validate sanitized stage-1A run evidence."""

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
from tools.pypto_wse_validation.stage1a_contracts import (
    BASE_PAYLOAD_SIZES,
    ROUND_TRIP_CASE_ID,
    RoundTripObservation,
    TransferObservation,
)

SCHEMA_VERSION = 1
FORBIDDEN_ARTIFACT_KEYS = frozenset({"address", "device_addr", "shareable_handle"})


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read stage-1A artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"stage-1A artifact must be an object: {path}")
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


def _endpoint_evidence(
    run_dir: Path,
    role: EndpointRole,
    *,
    validate_observations: bool,
) -> dict[str, Any]:
    endpoint = _read_json(run_dir / f"{role.value.lower()}_stage1a.json")
    evidence = {
        "cleanup": endpoint.get("cleanup"),
        "control": endpoint.get("control"),
        "device_id": endpoint.get("device_id"),
        "error": endpoint.get("error"),
        "manifest": endpoint.get("manifest"),
        "observations": endpoint.get("observations", []),
        "round_trips": endpoint.get("round_trips", []),
        "role": endpoint.get("role"),
        "success": endpoint.get("success"),
    }
    forbidden = _find_forbidden_keys(evidence)
    if forbidden:
        raise ValueError(f"endpoint artifact contains raw capability fields: {sorted(forbidden)}")
    if validate_observations:
        for raw in evidence["observations"]:
            TransferObservation.from_dict(raw)
        for raw in evidence["round_trips"]:
            RoundTripObservation.from_dict(raw)
    return evidence


def _validate_success(result: Mapping[str, Any], endpoints: Mapping[str, Mapping[str, Any]]) -> None:
    if result.get("success") is not True or result.get("capability_level") != "C1":
        raise ValueError("successful run did not establish surrogate C1")
    if result.get("claim_scope") != "NPU_SURROGATE_ONLY":
        raise ValueError("successful run has an unsafe claim scope")
    if result.get("npu_wse_capability_level") != "NOT_ESTABLISHED":
        raise ValueError("successful run overclaims real NPU-WSE capability")
    if result.get("resource_cleanup") != "VERIFIED" or result.get("host_bounce_bytes") != 0:
        raise ValueError("successful run is missing cleanup or zero-bounce proof")
    for direction in ("NPU_TO_WSE_SURROGATE", "WSE_SURROGATE_TO_NPU"):
        direction_result = result.get("data_results", {}).get(direction, {})
        if direction_result.get("status") != "PASS" or direction_result.get("passed") != len(BASE_PAYLOAD_SIZES):
            raise ValueError(f"successful run has incomplete direction evidence: {direction}")
    round_trip_result = result.get("data_results", {}).get(ROUND_TRIP_CASE_ID, {})
    if round_trip_result.get("status") != "PASS" or round_trip_result.get("passed") != len(BASE_PAYLOAD_SIZES):
        raise ValueError("successful run has incomplete T03 round-trip evidence")
    if result.get("host_intermediate_payload_bytes") != 0:
        raise ValueError("successful run used Host intermediate payload for T03")
    for role in EndpointRole:
        endpoint = endpoints[role.value]
        if endpoint.get("success") is not True or endpoint.get("error") is not None:
            raise ValueError(f"successful run endpoint failed: {role.value}")
        if endpoint.get("cleanup") != {
            "imported_window": "CLOSED",
            "owned_window": "CLOSED",
            "runtime": "CLOSED",
        }:
            raise ValueError(f"successful run endpoint cleanup failed: {role.value}")


def _load_run(task_id: str, run_dir: Path, *, require_success: bool) -> dict[str, Any]:
    result = _read_json(run_dir / "result.json")
    endpoints = {
        role.value: _endpoint_evidence(run_dir, role, validate_observations=require_success) for role in EndpointRole
    }
    if require_success:
        _validate_success(result, endpoints)
    return {"endpoints": endpoints, "result": result, "task_id": task_id}


def collect_stage1a(
    *,
    successful_runs: Sequence[tuple[str, Path]],
    diagnostic_runs: Sequence[tuple[str, Path]],
    implementation_revision: str,
    collected_at: str | None = None,
) -> dict[str, Any]:
    if len(successful_runs) < 2:
        raise ValueError("at least two successful stage-1A runs are required")
    loaded_successes = [_load_run(task_id, run_dir, require_success=True) for task_id, run_dir in successful_runs]
    start_orders = {run["result"].get("start_order") for run in loaded_successes}
    if start_orders != {"attention-first", "wse-first"}:
        raise ValueError("successful evidence must cover both endpoint start orders")
    loaded_diagnostics = [_load_run(task_id, run_dir, require_success=False) for task_id, run_dir in diagnostic_runs]
    return {
        "collected_at": collected_at or datetime.now(UTC).isoformat(),
        "conclusion": {
            "capability_level": "C1",
            "claim_scope": "NPU_SURROGATE_ONLY",
            "evidence_status": "SIMULATION",
            "npu_wse_capability_level": "NOT_ESTABLISHED",
            "t01": "PASS",
            "t02": "PASS",
            "t03": "PASS",
        },
        "diagnostic_runs": loaded_diagnostics,
        "implementation_revision": implementation_revision,
        "profile": "NPU_SURROGATE",
        "schema_version": SCHEMA_VERSION,
        "successful_runs": loaded_successes,
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
    parser.add_argument("--diagnostic-run", action="append", type=_parse_run, default=[])
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--collected-at")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence = collect_stage1a(
        successful_runs=args.successful_run,
        diagnostic_runs=args.diagnostic_run,
        implementation_revision=args.implementation_revision,
        collected_at=args.collected_at,
    )
    _write_json(args.output, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
