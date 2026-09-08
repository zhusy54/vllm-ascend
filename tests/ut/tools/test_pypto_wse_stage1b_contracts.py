# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.pypto_wse_validation.contracts import ContractError, TransportScope
from tools.pypto_wse_validation.stage1a_contracts import STAGE1A_BACKEND, STAGE1A_HANDLE_KIND
from tools.pypto_wse_validation.stage1b_contracts import (
    T04_DEVICE_CONTEXT,
    T04_DRIVER_KERNEL,
    T04_INPUT_FENCE,
    T04_OUTPUT_FENCE,
    T04_PAYLOAD_BYTES,
    T04_SEQUENCE_COUNT,
    T04_SERVICE_KERNEL,
    DeviceLoopReport,
    T04Observation,
)


def _report() -> DeviceLoopReport:
    return DeviceLoopReport(
        processed=T04_SEQUENCE_COUNT,
        validation_errors=0,
        sequence_errors=0,
        generation_errors=0,
        checksum_errors=0,
        marker_errors=0,
        timeouts=0,
        elapsed_cycles=1,
    )


def _observation() -> T04Observation:
    return T04Observation(
        case_id="T04",
        generation=1,
        slot=0,
        sequence_count=T04_SEQUENCE_COUNT,
        payload_bytes=T04_PAYLOAD_BYTES,
        backend=STAGE1A_BACKEND,
        transport_scope=TransportScope.HOST_LOCAL,
        handle_kind=STAGE1A_HANDLE_KIND,
        driver_kernel=T04_DRIVER_KERNEL,
        service_kernel=T04_SERVICE_KERNEL,
        driver_context=T04_DEVICE_CONTEXT,
        service_context=T04_DEVICE_CONTEXT,
        driver_launches=1,
        service_launches=1,
        device_submissions=T04_SEQUENCE_COUNT,
        device_completions=T04_SEQUENCE_COUNT,
        validated_sequences=T04_SEQUENCE_COUNT,
        input_fence=T04_INPUT_FENCE,
        output_fence=T04_OUTPUT_FENCE,
        host_hot_path_control_messages=0,
        host_hot_path_task_messages=0,
        host_hot_path_completion_messages=0,
        host_hot_path_payload_bytes=0,
        host_bounce_bytes=0,
        fallback_used=False,
        driver_report=_report(),
        service_report=_report(),
        driver_binary_sha256="a" * 64,
        service_binary_sha256="b" * 64,
    )


def test_t04_contract_accepts_complete_device_loop():
    observation = _observation()
    assert observation.passed
    assert T04Observation.from_dict(observation.to_dict()) == observation
    assert DeviceLoopReport.from_bytes(DeviceLoopReport._STRUCT.pack(*_report().to_dict().values())) == _report()


@pytest.mark.parametrize(
    "change",
    [
        {"driver_context": "HOST_LOOP"},
        {"driver_launches": T04_SEQUENCE_COUNT},
        {"device_submissions": T04_SEQUENCE_COUNT - 1},
        {"host_hot_path_control_messages": 1},
        {"host_hot_path_task_messages": 1},
        {"host_hot_path_completion_messages": 1},
        {"host_hot_path_payload_bytes": T04_PAYLOAD_BYTES},
        {"host_bounce_bytes": T04_PAYLOAD_BYTES},
        {"fallback_used": True},
        {"driver_report": replace(_report(), checksum_errors=1)},
        {"service_report": replace(_report(), timeouts=1)},
    ],
)
def test_t04_contract_fails_closed(change):
    assert not replace(_observation(), **change).passed


def test_t04_report_rejects_wrong_binary_size():
    with pytest.raises(ContractError, match="64 bytes"):
        DeviceLoopReport.from_bytes(b"short")
