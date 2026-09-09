# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Run V01-V06 and collect redacted proxy validation evidence.

Correct output alone is insufficient for this prototype: Host forwarding an
intermediate value could produce the same answer while violating the intended
architecture.  The collector therefore correlates three evidence groups:

* device reports prove A, B, and C ran once per accepted request;
* transfer counters prove Host touched ingress and final egress, but no
  intermediate payload;
* memory/lifecycle audits prove one Bootstrap allocation per endpoint and zero
  live mappings after release.

Detailed artifacts are useful for local debugging and remain gitignored.  Only
a small summary with no raw VMM handle or Device VA is checked into docs.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypto_test.pseudo_pypto.contracts import MAX_ELEMENTS
from pypto_test.run_proxy_service import run_proxy_service
from pypto_test.validation.validation_utils import get_input_payload, return_result


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    element_counts: tuple[int, ...]
    generations: tuple[int, ...] = (1,)


CASE_DEFINITIONS = {
    # V01: lifecycle only; V02: one 4 KiB request; V03: boundary sizes;
    # V04: repeated slot reuse; V05: orderly shutdown; V06: new generation.
    "V01": CaseDefinition("V01", ()),
    "V02": CaseDefinition("V02", (1024,)),
    "V03": CaseDefinition("V03", (16, 1024, 16 * 1024, MAX_ELEMENTS)),
    "V04": CaseDefinition("V04", tuple((16, 1024, 16 * 1024, 256)[index % 4] for index in range(100))),
    "V05": CaseDefinition("V05", (1024,)),
    "V06": CaseDefinition("V06", (1024,), (1, 2)),
}

DEFAULT_MATRIX = (
    # Both process start orders must reach READY, and both must complete the
    # full V03 size range.  Other cases need one order in this first prototype.
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
    """Fail closed unless execution, ownership, and cleanup all agree."""

    # Scope checks prevent a same-Host second-NPU run from being reported as
    # real-WSE or cross-Host evidence.
    if evidence.get("status") != "PASS":
        raise EvidenceError("generation did not report PASS")
    if evidence.get("execution_summary", {}).get("request_count") != expected_requests:
        raise EvidenceError("request count mismatch")
    if evidence["endpoint_bundle"]["transport_scope"] != "HOST_LOCAL":
        raise EvidenceError("prototype must remain HOST_LOCAL")
    if evidence["endpoint_bundle"]["backend_kind"] != "WSE":
        raise EvidenceError("prototype must expose the WSE backend contract")
    if evidence["resident_kernel_launches"] != {"attention": 1, "wse": 1}:
        raise EvidenceError("resident kernels were not launched exactly once")

    # A zero intermediate byte count is the observable Host-side assertion;
    # device reports below independently show where A/B/C actually ran.
    traffic = evidence["service"]["traffic"]
    if traffic["host_intermediate_bytes"] != 0:
        raise EvidenceError("Host participated in the A/B/C intermediate path")
    driver_report = evidence["service"]["driver"]["report"]
    wse_report = evidence["wse"]["backend"]["report"]
    for report_name, report in (("driver", driver_report), ("wse", wse_report)):
        if report["accepted"] != expected_requests or report["completed"] != expected_requests:
            raise EvidenceError(f"{report_name} request counters mismatch")
        error_count = sum(value for key, value in report.items() if key.endswith("errors"))
        if error_count:
            raise EvidenceError(f"{report_name} reported {error_count} validation errors")
    if driver_report["a_runs"] != expected_requests or driver_report["c_runs"] != expected_requests:
        raise EvidenceError("A/C device run counters mismatch")
    if wse_report["b_runs"] != expected_requests:
        raise EvidenceError("B device run counter mismatch")

    # Each endpoint has one owned mapping and one imported peer mapping during
    # execution.  Both must be gone by the time evidence is returned.
    memory = evidence["bootstrap"]["memory"]
    remote_memory = evidence["wse"]["memory"]
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
    wse_device: int,
    kernel_dir: Path,
) -> dict[str, Any]:
    """Run every generation required by one named validation case."""

    definition = CASE_DEFINITIONS[case_id]
    generations = []
    for generation in definition.generations:
        input_payloads = [
            get_input_payload(generation=generation, request_id=request_id, element_count=element_count)
            for request_id, element_count in enumerate(definition.element_counts, start=1)
        ]
        evidence = run_proxy_service(
            attention_device=attention_device,
            wse_device=wse_device,
            generation=generation,
            input_payloads=input_payloads,
            start_order=start_order,
            kernel_dir=kernel_dir,
            result_handler=return_result,
        )
        validate_generation(evidence, expected_requests=len(definition.element_counts))
        generations.append(evidence)
    if case_id == "V06":
        # V06 intentionally uses fresh processes and Bootstrap state.  Distinct
        # run/lease identities plus release evidence reject logical reuse of G.
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
    wse_device: int,
    kernel_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Execute the selected matrix and persist replay/debug artifacts."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for case_id, start_order in matrix:
        result = run_case(
            case_id=case_id,
            start_order=start_order,
            attention_device=attention_device,
            wse_device=wse_device,
            kernel_dir=kernel_dir,
        )
        cases.append(result)
        output = artifact_dir / f"{case_id.lower()}-{start_order}.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {
        "cases": cases,
        "environment": _environment(),
        "scope": {
            "backend": "WSE",
            "host_count": 1,
            "pypto_compiler_scheduler": "NOT_VALIDATED",
            "transport_scope": "HOST_LOCAL",
            "wse_device_implementation": "SECOND_NPU",
        },
        "status": "PASS",
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _environment() -> dict[str, Any]:
    """Capture environment context; NPU health text is not a pass criterion."""

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
    parser.add_argument("--wse-device", type=int, required=True)
    prototype_dir = Path(__file__).parents[1]
    parser.add_argument("--kernel-dir", type=Path, default=prototype_dir / "build")
    parser.add_argument("--artifact-dir", type=Path, default=prototype_dir / "artifacts")
    parser.add_argument("--case", choices=tuple(CASE_DEFINITIONS), action="append")
    parser.add_argument("--start-order", choices=("attention-first", "wse-first"), default="attention-first")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    matrix = tuple((case_id, args.start_order) for case_id in args.case) if args.case else DEFAULT_MATRIX
    summary = collect(
        matrix=matrix,
        attention_device=args.attention_device,
        wse_device=args.wse_device,
        kernel_dir=args.kernel_dir,
        artifact_dir=args.artifact_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
