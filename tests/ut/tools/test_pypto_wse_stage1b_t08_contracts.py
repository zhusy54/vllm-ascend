# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.pypto_wse_validation.contracts import ContractError, TransportScope
from tools.pypto_wse_validation.stage1a_contracts import STAGE1A_BACKEND, STAGE1A_HANDLE_KIND
from tools.pypto_wse_validation.stage1b_contracts import (
    T08_BASELINE_SEQUENCE_COUNT,
    T08_DEVICE_CONTEXT,
    T08_DRIVER_KERNEL,
    T08_HANDLE_PROBE_API,
    T08_INPUT_FENCE,
    T08_MAX_INFLIGHT,
    T08_NEXT_SEQUENCE_COUNT,
    T08_OUTPUT_FENCE,
    T08_PAYLOAD_BYTES,
    T08_SERVICE_KERNEL,
    T08_SLOT_COUNT,
    T08DeviceLoopReport,
    T08Observation,
)


def _report(*, driver: bool, mode: int) -> T08DeviceLoopReport:
    sequences = T08_BASELINE_SEQUENCE_COUNT if mode == 0 else T08_NEXT_SEQUENCE_COUNT
    return T08DeviceLoopReport(
        processed=sequences,
        validation_errors=0,
        sequence_errors=0,
        generation_errors=0,
        checksum_errors=0,
        marker_errors=0,
        timeouts=0,
        elapsed_cycles=1,
        submissions=sequences,
        completions=sequences,
        stale_descriptor_injections=int(driver and mode == 1),
        stale_descriptor_rejections=int(not driver and mode == 1),
        stale_completion_injections=int(not driver and mode == 1),
        stale_completion_rejections=int(driver and mode == 1),
        old_completion_credit_releases=0,
        current_slot_preserved=int(mode == 1),
        credits_acquired=sequences if driver else 0,
        credits_returned=sequences if driver else 0,
        terminal_tasks=sequences,
        progress_after_stale=sequences if mode == 1 else 0,
        input_fences=sequences if driver else 0,
        output_fences=0 if driver else sequences,
        mode=mode,
        reserved=0,
    )


def _observation() -> T08Observation:
    return T08Observation(
        case_id="T08",
        generation=7,
        next_generation=8,
        baseline_sequence_count=T08_BASELINE_SEQUENCE_COUNT,
        next_sequence_count=T08_NEXT_SEQUENCE_COUNT,
        slot_count=T08_SLOT_COUNT,
        max_inflight=T08_MAX_INFLIGHT,
        payload_bytes=T08_PAYLOAD_BYTES,
        device_submissions=T08_BASELINE_SEQUENCE_COUNT + T08_NEXT_SEQUENCE_COUNT,
        device_completions=T08_BASELINE_SEQUENCE_COUNT + T08_NEXT_SEQUENCE_COUNT,
        validated_sequences=T08_BASELINE_SEQUENCE_COUNT + T08_NEXT_SEQUENCE_COUNT,
        stale_descriptor_injections=1,
        stale_descriptor_rejections=1,
        stale_completion_injections=1,
        stale_completion_rejections=1,
        old_completion_credit_releases=0,
        current_slot_preserved=True,
        progress_after_stale=T08_NEXT_SEQUENCE_COUNT,
        old_handle_rejected_before_reallocate=2,
        old_handle_rejected_after_reallocate=2,
        old_handle_import_successes=0,
        new_handle_collisions=0,
        handle_probe_api=T08_HANDLE_PROBE_API,
        backend=STAGE1A_BACKEND,
        transport_scope=TransportScope.HOST_LOCAL,
        handle_kind=STAGE1A_HANDLE_KIND,
        driver_kernel=T08_DRIVER_KERNEL,
        service_kernel=T08_SERVICE_KERNEL,
        driver_context=T08_DEVICE_CONTEXT,
        service_context=T08_DEVICE_CONTEXT,
        driver_launches=2,
        service_launches=2,
        input_fence=T08_INPUT_FENCE,
        output_fence=T08_OUTPUT_FENCE,
        host_hot_path_control_messages=0,
        host_hot_path_task_messages=0,
        host_hot_path_completion_messages=0,
        host_hot_path_payload_bytes=0,
        host_bounce_bytes=0,
        fallback_used=False,
        baseline_driver_report=_report(driver=True, mode=0),
        baseline_service_report=_report(driver=False, mode=0),
        next_driver_report=_report(driver=True, mode=1),
        next_service_report=_report(driver=False, mode=1),
        driver_binary_sha256="a" * 64,
        service_binary_sha256="b" * 64,
    )


def test_t08_contract_accepts_generation_and_handle_isolation():
    observation = _observation()
    assert observation.passed
    assert T08Observation.from_dict(observation.to_dict()) == observation
    report = _report(driver=True, mode=1)
    assert T08DeviceLoopReport.from_bytes(T08DeviceLoopReport._STRUCT.pack(*report.to_dict().values())) == report


@pytest.mark.parametrize(
    "change",
    [
        {"next_generation": 9},
        {"stale_descriptor_rejections": 0},
        {"stale_completion_rejections": 0},
        {"old_completion_credit_releases": 1},
        {"current_slot_preserved": False},
        {"progress_after_stale": 0},
        {"old_handle_rejected_before_reallocate": 1},
        {"old_handle_rejected_after_reallocate": 1},
        {"old_handle_import_successes": 1},
        {"new_handle_collisions": 1},
        {"next_driver_report": replace(_report(driver=True, mode=1), credits_returned=0)},
        {"next_service_report": replace(_report(driver=False, mode=1), stale_descriptor_rejections=0)},
    ],
)
def test_t08_contract_fails_closed(change):
    candidate = replace(_observation(), **change)
    if change == {"next_generation": 9}:
        with pytest.raises(ContractError):
            candidate.validate()
    else:
        assert not candidate.passed


def test_t08_report_rejects_wrong_binary_size():
    with pytest.raises(ContractError, match="192 bytes"):
        T08DeviceLoopReport.from_bytes(b"short")
