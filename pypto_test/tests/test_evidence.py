# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Fail-closed checks for the multi-source validation evidence.

The synthetic record is not execution proof.  It isolates collector policy so
that a nonzero Host intermediate path, repeated kernel launch, leaked mapping,
or missing B execution cannot accidentally be summarized as PASS.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import pypto_test.run_proxy_service as proxy_entry
from pypto_test.pseudo_pypto.communication import ExecutionResult, checksum_u32
from pypto_test.validation.collect_evidence import EvidenceError, validate_generation
from pypto_test.validation.validation_utils import expected_abc, get_input_payload, return_result


def make_evidence():
    report = {
        "accepted": 1,
        "completed": 1,
        "generation_errors": 0,
        "request_errors": 0,
        "sequence_errors": 0,
        "validation_errors": 0,
        "checksum_errors": 0,
    }
    return {
        "bootstrap": {
            "lease_state": "RELEASED",
            "memory": {
                "allocated_window_count": 1,
                "live_mapping_count": 0,
                "mapping_count": 2,
            },
        },
        "endpoint_bundle": {"backend_kind": "WSE", "transport_scope": "HOST_LOCAL"},
        "execution_summary": {"request_count": 1},
        "resident_kernel_launches": {"attention": 1, "wse": 1},
        "service": {
            "driver": {"report": {**report, "a_runs": 1, "c_runs": 1}},
            "traffic": {"host_intermediate_bytes": 0},
        },
        "status": "PASS",
        "wse": {
            "backend": {"report": {**report, "b_runs": 1}},
            "memory": {
                "allocated_window_count": 1,
                "live_mapping_count": 0,
                "mapping_count": 2,
            },
        },
    }


def test_evidence_validator_accepts_measured_contract():
    validate_generation(make_evidence(), expected_requests=1)


def test_return_result_validates_and_prints_final_output(capsys):
    payload = get_input_payload(generation=1, request_id=1, element_count=4)
    output = expected_abc(payload)
    result = ExecutionResult(1, 1, 1, 4, output, checksum_u32(output), 1, 128, len(output))
    return_result(payload, result)
    assert '"status": "PASS"' in capsys.readouterr().out


def test_run_proxy_service_module_has_direct_cli(monkeypatch, tmp_path, capsys):
    output = tmp_path / "result.json"
    captured = {}

    def fake_run_proxy_service(**kwargs):
        captured.update(kwargs)
        return {"status": "PASS"}

    monkeypatch.setattr(proxy_entry, "run_proxy_service", fake_run_proxy_service)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proxy_service",
            "--attention-device",
            "0",
            "--wse-device",
            "1",
            "--elements",
            "16",
            "--output",
            str(output),
        ],
    )
    assert proxy_entry.main() == 0
    assert len(captured["input_payloads"]) == 1
    assert captured["kernel_dir"] == Path(proxy_entry.__file__).parent / "build"
    assert '"status": "PASS"' in output.read_text()
    assert '"status": "PASS"' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("path", "value", "match"),
    (
        (("service", "traffic", "host_intermediate_bytes"), 4, "Host participated"),
        (("resident_kernel_launches", "attention"), 2, "exactly once"),
        (("bootstrap", "memory", "live_mapping_count"), 1, "live mapping"),
        (("wse", "backend", "report", "b_runs"), 0, "B device"),
    ),
)
def test_evidence_validator_fails_closed(path, value, match):
    evidence = deepcopy(make_evidence())
    target = evidence
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(EvidenceError, match=match):
        validate_generation(evidence, expected_requests=1)
