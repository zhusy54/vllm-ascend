# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

from dataclasses import replace

import pytest

from pypto_test.contracts import (
    ABC_TASKS,
    CACHE_LINE_BYTES,
    DEFAULT_LAYOUT,
    EXPECTED_LAYOUT_HASH,
    MAX_ELEMENTS,
    NPU_WINDOW_BYTES,
    WSE_WINDOW_BYTES,
    CompletionDescriptor,
    ContractError,
    DriverReport,
    HostRequestDescriptor,
    LifecycleLine,
    PseudoProgramSpec,
    RemoteTaskDescriptor,
    ServiceReport,
    SignalLine,
    checksum_u32,
    deterministic_input,
    expected_abc,
)


def test_fixed_program_and_layout_are_stable():
    spec = PseudoProgramSpec()
    spec.validate()
    assert spec.tasks == ABC_TASKS
    assert DEFAULT_LAYOUT.layout_hash == EXPECTED_LAYOUT_HASH
    assert NPU_WINDOW_BYTES > 3 * 1024 * 1024
    assert WSE_WINDOW_BYTES > 1024 * 1024


@pytest.mark.parametrize(
    "spec",
    (
        PseudoProgramSpec(program_id="other"),
        PseudoProgramSpec(dtype="float32"),
        PseudoProgramSpec(max_elements=MAX_ELEMENTS - 1),
        PseudoProgramSpec(tasks=ABC_TASKS[:2]),
    ),
)
def test_program_rejects_non_abc_contracts(spec):
    with pytest.raises(ContractError):
        spec.validate()


def test_control_descriptors_round_trip_as_cache_lines():
    request = HostRequestDescriptor(1, 2, 3, 4, 5)
    remote = RemoteTaskDescriptor(1, 2, 3, 4, 5)
    completion = CompletionDescriptor(1, 2, 3, 0, 4, 5)
    signal = SignalLine(5)
    lifecycle = LifecycleLine(stop_requested=1, stopped=0, ready=1)
    for value, decoder in (
        (request, HostRequestDescriptor.from_bytes),
        (remote, RemoteTaskDescriptor.from_bytes),
        (completion, CompletionDescriptor.from_bytes),
        (signal, SignalLine.from_bytes),
        (lifecycle, LifecycleLine.from_bytes),
    ):
        payload = value.to_bytes()
        assert len(payload) == CACHE_LINE_BYTES
        assert decoder(payload) == value


def test_descriptor_rejects_invalid_element_count():
    with pytest.raises(ContractError, match="element_count"):
        replace(HostRequestDescriptor(1, 1, 1, 0, 1), element_count=MAX_ELEMENTS + 1).to_bytes()


def test_uint32_reference_is_deterministic_and_wraps():
    payload = deterministic_input(generation=3, request_id=7, element_count=32)
    assert payload == deterministic_input(generation=3, request_id=7, element_count=32)
    assert payload != deterministic_input(generation=3, request_id=8, element_count=32)
    assert len(expected_abc(payload)) == len(payload)
    wrap = bytes.fromhex("ffffffff")
    assert expected_abc(wrap) == bytes.fromhex("03000000")
    assert checksum_u32(wrap) == 0xFFFFFFFF


def test_device_report_wire_sizes_are_stable():
    payload = bytes(16 * 8)
    assert DriverReport.from_bytes(payload).accepted == 0
    assert ServiceReport.from_bytes(payload).completed == 0
