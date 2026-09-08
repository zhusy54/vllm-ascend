# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.pypto_wse_validation.contracts import ContractError, TransportScope
from tools.pypto_wse_validation.stage1a_contracts import STAGE1A_BACKEND, STAGE1A_HANDLE_KIND
from tools.pypto_wse_validation.stage1b_contracts import (
    T05_DEVICE_CONTEXT,
    T05_DRIVER_KERNEL,
    T05_INPUT_FENCE,
    T05_MAX_INFLIGHT,
    T05_OUTPUT_FENCE,
    T05_PAYLOAD_BYTES,
    T05_SEQUENCE_COUNT,
    T05_SERVICE_KERNEL,
    T05_SLOT_COUNT,
    T05DeviceLoopReport,
    T05Observation,
)


def _driver_report() -> T05DeviceLoopReport:
    return T05DeviceLoopReport(
        processed=T05_SEQUENCE_COUNT,
        validation_errors=0,
        sequence_errors=0,
        generation_errors=0,
        checksum_errors=0,
        marker_errors=0,
        timeouts=0,
        elapsed_cycles=1,
        slot0_processed=T05_SEQUENCE_COUNT // 2,
        slot1_processed=T05_SEQUENCE_COUNT // 2,
        credits_acquired=T05_SEQUENCE_COUNT,
        terminal_tasks=T05_SEQUENCE_COUNT,
        credits_returned=T05_SEQUENCE_COUNT,
        max_inflight=T05_MAX_INFLIGHT,
        out_of_order_completions=T05_SEQUENCE_COUNT // 2,
        slot_overwrite_errors=0,
    )


def _service_report() -> T05DeviceLoopReport:
    return replace(_driver_report(), credits_acquired=0, credits_returned=0)


def _observation() -> T05Observation:
    return T05Observation(
        case_id="T05",
        generation=1,
        slot_count=T05_SLOT_COUNT,
        max_inflight=T05_MAX_INFLIGHT,
        sequence_count=T05_SEQUENCE_COUNT,
        payload_bytes=T05_PAYLOAD_BYTES,
        backend=STAGE1A_BACKEND,
        transport_scope=TransportScope.HOST_LOCAL,
        handle_kind=STAGE1A_HANDLE_KIND,
        driver_kernel=T05_DRIVER_KERNEL,
        service_kernel=T05_SERVICE_KERNEL,
        driver_context=T05_DEVICE_CONTEXT,
        service_context=T05_DEVICE_CONTEXT,
        driver_launches=1,
        service_launches=1,
        device_submissions=T05_SEQUENCE_COUNT,
        device_completions=T05_SEQUENCE_COUNT,
        validated_sequences=T05_SEQUENCE_COUNT,
        credits_acquired=T05_SEQUENCE_COUNT,
        terminal_tasks=T05_SEQUENCE_COUNT,
        credits_returned=T05_SEQUENCE_COUNT,
        out_of_order_completions=T05_SEQUENCE_COUNT // 2,
        input_fence=T05_INPUT_FENCE,
        output_fence=T05_OUTPUT_FENCE,
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


def test_t05_contract_accepts_dual_slot_out_of_order_completion():
    observation = _observation()
    assert observation.passed
    assert T05Observation.from_dict(observation.to_dict()) == observation
    raw = T05DeviceLoopReport._STRUCT.pack(*_driver_report().to_dict().values())
    assert T05DeviceLoopReport.from_bytes(raw) == _driver_report()


@pytest.mark.parametrize(
    "change",
    [
        {"credits_returned": T05_SEQUENCE_COUNT - 1},
        {"out_of_order_completions": 0},
        {"host_hot_path_control_messages": 1},
        {"host_hot_path_payload_bytes": T05_PAYLOAD_BYTES[0]},
        {"driver_report": replace(_driver_report(), slot_overwrite_errors=1)},
        {"driver_report": replace(_driver_report(), credits_returned=T05_SEQUENCE_COUNT - 1)},
        {"service_report": replace(_service_report(), sequence_errors=1)},
    ],
)
def test_t05_contract_fails_closed(change):
    assert not replace(_observation(), **change).passed


@pytest.mark.parametrize(
    "change",
    [
        {"slot_count": 1},
        {"max_inflight": 1},
        {"payload_bytes": (T05_PAYLOAD_BYTES[0], T05_PAYLOAD_BYTES[0])},
    ],
)
def test_t05_contract_rejects_invalid_shape(change):
    with pytest.raises(ContractError):
        replace(_observation(), **change).validate()


def test_t05_report_rejects_wrong_binary_size():
    with pytest.raises(ContractError, match="128 bytes"):
        T05DeviceLoopReport.from_bytes(b"short")
