# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.pypto_wse_validation.contracts import ContractError, TransportScope
from tools.pypto_wse_validation.stage1a_contracts import STAGE1A_BACKEND, STAGE1A_HANDLE_KIND
from tools.pypto_wse_validation.stage1b_contracts import (
    T06_BACKPRESSURE_WAIT,
    T06_DEVICE_CONTEXT,
    T06_DRIVER_KERNEL,
    T06_INPUT_FENCE,
    T06_MAX_INFLIGHT,
    T06_OUTPUT_FENCE,
    T06_PAYLOAD_BYTES,
    T06_SEQUENCE_COUNT,
    T06_SERVICE_KERNEL,
    T06_SLOT_COUNT,
    T06_THIRD_REQUEST_OUTCOME,
    T06DeviceLoopReport,
    T06Observation,
)


def _driver_report() -> T06DeviceLoopReport:
    return T06DeviceLoopReport(
        processed=T06_SEQUENCE_COUNT,
        validation_errors=0,
        sequence_errors=0,
        generation_errors=0,
        checksum_errors=0,
        marker_errors=0,
        timeouts=0,
        elapsed_cycles=1,
        slot0_processed=T06_SEQUENCE_COUNT // 2,
        slot1_processed=T06_SEQUENCE_COUNT // 2,
        credits_acquired=T06_SEQUENCE_COUNT,
        terminal_tasks=T06_SEQUENCE_COUNT,
        credits_returned=T06_SEQUENCE_COUNT,
        max_inflight=T06_MAX_INFLIGHT,
        slot_overwrite_errors=0,
        no_credit_events=1,
        pending_requests=1,
        submission_retry_spins=0,
        service_pause_observed=1,
        service_resume_observed=1,
        progress_after_resume=T06_SEQUENCE_COUNT,
        peak_queue_depth=T06_MAX_INFLIGHT,
        submissions=T06_SEQUENCE_COUNT,
        completions=T06_SEQUENCE_COUNT,
    )


def _service_report() -> T06DeviceLoopReport:
    return replace(_driver_report(), credits_acquired=0, credits_returned=0)


def _observation() -> T06Observation:
    return T06Observation(
        case_id="T06",
        generation=1,
        slot_count=T06_SLOT_COUNT,
        max_inflight=T06_MAX_INFLIGHT,
        sequence_count=T06_SEQUENCE_COUNT,
        payload_bytes=T06_PAYLOAD_BYTES,
        attempted_submissions=T06_SEQUENCE_COUNT + 1,
        device_submissions=T06_SEQUENCE_COUNT,
        device_completions=T06_SEQUENCE_COUNT,
        validated_sequences=T06_SEQUENCE_COUNT,
        no_credit_events=1,
        pending_requests=1,
        terminal_tasks=T06_SEQUENCE_COUNT,
        credits_acquired=T06_SEQUENCE_COUNT,
        credits_returned=T06_SEQUENCE_COUNT,
        submission_retry_spins=0,
        service_pause_observed=True,
        service_resume_observed=True,
        progress_after_resume=T06_SEQUENCE_COUNT,
        third_request_outcome=T06_THIRD_REQUEST_OUTCOME,
        backpressure_wait=T06_BACKPRESSURE_WAIT,
        backend=STAGE1A_BACKEND,
        transport_scope=TransportScope.HOST_LOCAL,
        handle_kind=STAGE1A_HANDLE_KIND,
        driver_kernel=T06_DRIVER_KERNEL,
        service_kernel=T06_SERVICE_KERNEL,
        driver_context=T06_DEVICE_CONTEXT,
        service_context=T06_DEVICE_CONTEXT,
        driver_launches=1,
        service_launches=1,
        input_fence=T06_INPUT_FENCE,
        output_fence=T06_OUTPUT_FENCE,
        host_hot_path_control_messages=0,
        host_hot_path_task_messages=0,
        host_hot_path_completion_messages=0,
        host_hot_path_payload_bytes=0,
        host_bounce_bytes=0,
        fallback_used=False,
        driver_report=_driver_report(),
        service_report=_service_report(),
        driver_binary_sha256="a" * 64,
        service_binary_sha256="b" * 64,
    )


def test_t06_contract_accepts_no_credit_then_resumed_progress():
    observation = _observation()
    assert observation.passed
    assert T06Observation.from_dict(observation.to_dict()) == observation
    raw = T06DeviceLoopReport._STRUCT.pack(*_driver_report().to_dict().values())
    assert T06DeviceLoopReport.from_bytes(raw) == _driver_report()


@pytest.mark.parametrize(
    "change",
    [
        {"attempted_submissions": T06_SEQUENCE_COUNT},
        {"no_credit_events": 0},
        {"credits_returned": T06_SEQUENCE_COUNT - 1},
        {"submission_retry_spins": 1},
        {"service_pause_observed": False},
        {"service_resume_observed": False},
        {"progress_after_resume": 0},
        {"third_request_outcome": "OVERWROTE_SLOT"},
        {"driver_report": replace(_driver_report(), slot_overwrite_errors=1)},
        {"service_report": replace(_service_report(), no_credit_events=0)},
    ],
)
def test_t06_contract_fails_closed(change):
    assert not replace(_observation(), **change).passed


def test_t06_report_rejects_wrong_binary_size():
    with pytest.raises(ContractError, match="192 bytes"):
        T06DeviceLoopReport.from_bytes(b"short")
