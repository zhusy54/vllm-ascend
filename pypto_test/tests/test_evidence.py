# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Fail-closed checks for the multi-source validation evidence.

The synthetic record is not execution proof.  It isolates collector policy so
that a nonzero Host intermediate path, repeated kernel launch, leaked mapping,
or missing B execution cannot accidentally be summarized as PASS.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from pypto_test.collect_evidence import EvidenceError, validate_generation


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
        "endpoint_bundle": {"backend_kind": "NPU_SURROGATE", "transport_scope": "HOST_LOCAL"},
        "executions": [{}],
        "resident_kernel_launches": {"attention": 1, "wse_surrogate": 1},
        "service": {
            "driver": {"report": {**report, "a_runs": 1, "c_runs": 1}},
            "traffic": {"host_intermediate_bytes": 0},
        },
        "status": "PASS",
        "surrogate": {
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


@pytest.mark.parametrize(
    ("path", "value", "match"),
    (
        (("service", "traffic", "host_intermediate_bytes"), 4, "Host participated"),
        (("resident_kernel_launches", "attention"), 2, "exactly once"),
        (("bootstrap", "memory", "live_mapping_count"), 1, "live mapping"),
        (("surrogate", "backend", "report", "b_runs"), 0, "B device"),
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
