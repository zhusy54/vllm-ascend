# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Input generation and result checking used only by the validation harness.

The proxy service accepts bytes and returns an :class:`ExecutionResult`; it
does not know how a test payload is generated or what the fixed ABC oracle is.
Keeping those policies here prevents validation logic from becoming part of
the assumed PyPTO service contract.
"""

from __future__ import annotations

import json
import struct

from pypto_test.pseudo_pypto.communication import MAX_ELEMENTS, ExecutionResult


class ResultValidationError(RuntimeError):
    """Raised when the returned C output differs from the validation oracle."""


def get_input_payload(*, generation: int, request_id: int, element_count: int) -> bytes:
    """Build a reproducible uint32 payload for one validation request."""

    if generation <= 0 or request_id <= 0:
        raise ValueError("generation and request_id must be positive")
    if element_count <= 0 or element_count > MAX_ELEMENTS:
        raise ValueError(f"element_count must be in [1, {MAX_ELEMENTS}]")
    values = (
        ((generation * 0x9E3779B1) ^ (request_id * 0x85EBCA77) ^ (index * 0xC2B2AE3D)) & 0xFFFFFFFF
        for index in range(element_count)
    )
    return b"".join(struct.pack("<I", value) for value in values)


def expected_abc(payload: bytes) -> bytes:
    """Return the CPU oracle for ``C(B(A(x))) == 2*x+5``."""

    if not payload or len(payload) > MAX_ELEMENTS * 4 or len(payload) % 4:
        raise ValueError("payload must contain valid little-endian uint32 values")
    values = (((value[0] + 1) * 2 + 3) & 0xFFFFFFFF for value in struct.iter_unpack("<I", payload))
    return b"".join(struct.pack("<I", value) for value in values)


def return_result(input_payload: bytes, result: ExecutionResult) -> None:
    """Validate and print one final service result after C has completed.

    This function runs after ``service.execute`` returns.  It is an external
    oracle and reporting hook; it never reads Device memory or advances the
    A -> B -> C dependency chain.
    """

    if result.output != expected_abc(input_payload):
        raise ResultValidationError(f"request {result.request_id} output does not match 2*x+5")
    print(
        json.dumps(
            {
                "element_count": result.element_count,
                "output_checksum": result.output_checksum,
                "request_id": result.request_id,
                "sequence": result.sequence,
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
