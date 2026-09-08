# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.pypto_wse_validation.contracts import ContractError, TransportScope
from tools.pypto_wse_validation.stage1a_contracts import STAGE1A_BACKEND, STAGE1A_HANDLE_KIND
from tools.pypto_wse_validation.stage1b_contracts import (
    T07_COMPLETION_CHECK,
    T07_DEVICE_CONTEXT,
    T07_DRIVER_KERNEL,
    T07_INPUT_FENCE,
    T07_MAX_INFLIGHT,
    T07_OUTPUT_FENCE,
    T07_PAYLOAD_BYTES,
    T07_SEQUENCE_COUNT,
    T07_SERVICE_KERNEL,
    T07_SLOT_COUNT,
    T07DeviceLoopReport,
    T07Observation,
)


def _driver_report() -> T07DeviceLoopReport:
    return T07DeviceLoopReport(
        processed=T07_SEQUENCE_COUNT,
        validation_errors=0,
        sequence_errors=0,
        generation_errors=0,
        checksum_errors=0,
        head_marker_errors=0,
        tail_marker_errors=0,
        stale_payload_errors=0,
        incomplete_payload_errors=0,
        premature_completion_errors=0,
        timeouts=0,
        elapsed_cycles=1,
        slot0_processed=T07_SEQUENCE_COUNT // 2,
        slot1_processed=T07_SEQUENCE_COUNT // 2,
        input_publish_fences=T07_SEQUENCE_COUNT,
        input_visibility_checks=0,
        output_publish_fences=0,
        immediate_completion_checks=T07_SEQUENCE_COUNT,
        post_completion_delay_cycles=0,
        submissions=T07_SEQUENCE_COUNT,
        completions=T07_SEQUENCE_COUNT,
        slot_overwrite_errors=0,
        payload_words_validated=T07_SEQUENCE_COUNT * (T07_PAYLOAD_BYTES // 8),
        unique_tail_markers_validated=T07_SEQUENCE_COUNT,
        max_inflight=T07_MAX_INFLIGHT,
        reserved=0,
    )


def _service_report() -> T07DeviceLoopReport:
    return replace(
        _driver_report(),
        input_publish_fences=0,
        input_visibility_checks=T07_SEQUENCE_COUNT,
        output_publish_fences=T07_SEQUENCE_COUNT,
        immediate_completion_checks=0,
    )


def _observation() -> T07Observation:
    return T07Observation(
        case_id="T07",
        generation=1,
        slot_count=T07_SLOT_COUNT,
        max_inflight=T07_MAX_INFLIGHT,
        sequence_count=T07_SEQUENCE_COUNT,
        payload_bytes=T07_PAYLOAD_BYTES,
        device_submissions=T07_SEQUENCE_COUNT,
        device_completions=T07_SEQUENCE_COUNT,
        validated_sequences=T07_SEQUENCE_COUNT,
        unique_tail_markers_validated=T07_SEQUENCE_COUNT,
        immediate_completion_checks=T07_SEQUENCE_COUNT,
        post_completion_delay_cycles=0,
        stale_payload_errors=0,
        incomplete_payload_errors=0,
        premature_completion_errors=0,
        completion_check=T07_COMPLETION_CHECK,
        backend=STAGE1A_BACKEND,
        transport_scope=TransportScope.HOST_LOCAL,
        handle_kind=STAGE1A_HANDLE_KIND,
        driver_kernel=T07_DRIVER_KERNEL,
        service_kernel=T07_SERVICE_KERNEL,
        driver_context=T07_DEVICE_CONTEXT,
        service_context=T07_DEVICE_CONTEXT,
        driver_launches=1,
        service_launches=1,
        input_fence=T07_INPUT_FENCE,
        output_fence=T07_OUTPUT_FENCE,
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


def test_t07_contract_accepts_immediate_large_payload_visibility():
    observation = _observation()
    assert observation.passed
    assert T07Observation.from_dict(observation.to_dict()) == observation
    raw = T07DeviceLoopReport._STRUCT.pack(*_driver_report().to_dict().values())
    assert T07DeviceLoopReport.from_bytes(raw) == _driver_report()


@pytest.mark.parametrize(
    "change",
    [
        {"unique_tail_markers_validated": T07_SEQUENCE_COUNT - 1},
        {"immediate_completion_checks": T07_SEQUENCE_COUNT - 1},
        {"post_completion_delay_cycles": 1},
        {"stale_payload_errors": 1},
        {"incomplete_payload_errors": 1},
        {"premature_completion_errors": 1},
        {"completion_check": "SLEEP_THEN_CHECK"},
        {"host_hot_path_completion_messages": 1},
        {"driver_report": replace(_driver_report(), tail_marker_errors=1)},
        {"service_report": replace(_service_report(), input_visibility_checks=T07_SEQUENCE_COUNT - 1)},
    ],
)
def test_t07_contract_fails_closed(change):
    assert not replace(_observation(), **change).passed


def test_t07_report_rejects_wrong_binary_size():
    with pytest.raises(ContractError, match="208 bytes"):
        T07DeviceLoopReport.from_bytes(b"short")
