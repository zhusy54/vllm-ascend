# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Run V01-V06 and collect redacted proxy validation evidence."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypto_test.contracts import MAX_ELEMENTS
from pypto_test.launcher import run_generation


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    element_counts: tuple[int, ...]
    generations: tuple[int, ...] = (1,)


CASE_DEFINITIONS = {
    "V01": CaseDefinition("V01", ()),
    "V02": CaseDefinition("V02", (1024,)),
    "V03": CaseDefinition("V03", (16, 1024, 16 * 1024, MAX_ELEMENTS)),
    "V04": CaseDefinition("V04", tuple((16, 1024, 16 * 1024, 256)[index % 4] for index in range(100))),
    "V05": CaseDefinition("V05", (1024,)),
    "V06": CaseDefinition("V06", (1024,), (1, 2)),
}

DEFAULT_MATRIX = (
    ("V01", "attention-first"),
    ("V01", "wse-first"),
    ("V02", "attention-first"),
    ("V03", "attention-first"),
    ("V03", "wse-first"),
    ("V04", "attention-first"),
    ("V05", "attention-first"),
    ("V06", "attention-first"),
)


class EvidenceError(RuntimeError):
    """Raised when measured evidence does not meet a case contract."""


def validate_generation(evidence: dict[str, Any], *, expected_requests: int) -> None:
    if evidence.get("status") != "PASS":
        raise EvidenceError("generation did not report PASS")
    if len(evidence.get("executions", ())) != expected_requests:
        raise EvidenceError("request count mismatch")
    if evidence["endpoint_bundle"]["transport_scope"] != "HOST_LOCAL":
        raise EvidenceError("prototype must remain HOST_LOCAL")
    if evidence["endpoint_bundle"]["backend_kind"] != "NPU_SURROGATE":
        raise EvidenceError("prototype must use the NPU surrogate")
    if evidence["resident_kernel_launches"] != {"attention": 1, "wse_surrogate": 1}:
        raise EvidenceError("resident kernels were not launched exactly once")
    traffic = evidence["service"]["traffic"]
    if traffic["host_intermediate_bytes"] != 0:
        raise EvidenceError("Host participated in the A/B/C intermediate path")
    driver_report = evidence["service"]["driver"]["report"]
    surrogate_report = evidence["surrogate"]["backend"]["report"]
    for report_name, report in (("driver", driver_report), ("surrogate", surrogate_report)):
        if report["accepted"] != expected_requests or report["completed"] != expected_requests:
            raise EvidenceError(f"{report_name} request counters mismatch")
        error_count = sum(value for key, value in report.items() if key.endswith("errors"))
        if error_count:
            raise EvidenceError(f"{report_name} reported {error_count} validation errors")
    if driver_report["a_runs"] != expected_requests or driver_report["c_runs"] != expected_requests:
        raise EvidenceError("A/C device run counters mismatch")
    if surrogate_report["b_runs"] != expected_requests:
        raise EvidenceError("B device run counter mismatch")
    memory = evidence["bootstrap"]["memory"]
    remote_memory = evidence["surrogate"]["memory"]
    for endpoint in (memory, remote_memory):
        if endpoint["allocated_window_count"] != 1 or endpoint["mapping_count"] != 2:
            raise EvidenceError("bootstrap memory counts mismatch")
        if endpoint["live_mapping_count"] != 0:
            raise EvidenceError("bootstrap left a live mapping")
    if evidence["bootstrap"]["lease_state"] != "RELEASED":
        raise EvidenceError("bootstrap lease was not released")


def run_case(
    *,
    case_id: str,
    start_order: str,
    attention_device: int,
    surrogate_device: int,
    kernel_dir: Path,
) -> dict[str, Any]:
    definition = CASE_DEFINITIONS[case_id]
    generations = []
    for generation in definition.generations:
        evidence = run_generation(
            attention_device=attention_device,
            surrogate_device=surrogate_device,
            generation=generation,
            element_counts=list(definition.element_counts),
            start_order=start_order,
            kernel_dir=kernel_dir,
        )
        validate_generation(evidence, expected_requests=len(definition.element_counts))
        generations.append(evidence)
    if case_id == "V06":
        first, second = generations
        if first["endpoint_bundle"]["lease_id"] == second["endpoint_bundle"]["lease_id"]:
            raise EvidenceError("V06 reused a bootstrap lease")
        if first["run_id"] == second["run_id"]:
            raise EvidenceError("V06 reused a generation control session")
    return {
        "case_id": case_id,
        "generations": generations,
        "start_order": start_order,
        "status": "PASS",
    }


def collect(
    *,
    matrix: tuple[tuple[str, str], ...],
    attention_device: int,
    surrogate_device: int,
    kernel_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for case_id, start_order in matrix:
        result = run_case(
            case_id=case_id,
            start_order=start_order,
            attention_device=attention_device,
            surrogate_device=surrogate_device,
            kernel_dir=kernel_dir,
        )
        cases.append(result)
        output = artifact_dir / f"{case_id.lower()}-{start_order}.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {
        "cases": cases,
        "environment": _environment(),
        "scope": {
            "backend": "NPU_SURROGATE",
            "host_count": 1,
            "pypto_compiler_scheduler": "NOT_VALIDATED",
            "transport_scope": "HOST_LOCAL",
        },
        "status": "PASS",
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _environment() -> dict[str, Any]:
    try:
        npu_smi = subprocess.run(
            ["npu-smi", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        npu_summary = npu_smi.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        npu_summary = f"unavailable: {exc}"
    return {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "npu_smi_info": npu_summary,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-device", type=int, required=True)
    parser.add_argument("--surrogate-device", type=int, required=True)
    parser.add_argument("--kernel-dir", type=Path, default=Path(__file__).parent / "build")
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--case", choices=tuple(CASE_DEFINITIONS), action="append")
    parser.add_argument("--start-order", choices=("attention-first", "wse-first"), default="attention-first")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    matrix = tuple((case_id, args.start_order) for case_id in args.case) if args.case else DEFAULT_MATRIX
    summary = collect(
        matrix=matrix,
        attention_device=args.attention_device,
        surrogate_device=args.surrogate_device,
        kernel_dir=args.kernel_dir,
        artifact_dir=args.artifact_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
