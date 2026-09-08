# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Fail-closed evidence contract for Stage 1B case T04."""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from typing import Any

from tools.pypto_wse_validation.contracts import ContractError, TransportScope
from tools.pypto_wse_validation.stage1a_contracts import (
    STAGE1A_BACKEND,
    STAGE1A_HANDLE_KIND,
)

T04_CASE_ID = "T04"
T04_SEQUENCE_COUNT = 100
T04_SLOT = 0
T04_PAYLOAD_BYTES = 4 * 1024
T04_CONTROL_OFFSET = 1024 * 1024
T04_CONTROL_BYTES = 3 * 64
T04_WINDOW_BYTES = T04_CONTROL_OFFSET + T04_CONTROL_BYTES
T04_DRIVER_KERNEL = "pypto_stage1b_t04_driver_0_mix_aiv"
T04_SERVICE_KERNEL = "pypto_stage1b_t04_service_0_mix_aiv"
T04_DEVICE_CONTEXT = "AIV_DEVICE_KERNEL"
T04_INPUT_FENCE = "st_dev_payload_metadata+dsb_all+st_dev_submission"
T04_OUTPUT_FENCE = "st_dev_output_metadata+dsb_all+st_dev_completion"


@dataclass(frozen=True)
class DeviceLoopReport:
    processed: int
    validation_errors: int
    sequence_errors: int
    generation_errors: int
    checksum_errors: int
    marker_errors: int
    timeouts: int
    elapsed_cycles: int

    _STRUCT = struct.Struct("<QQQQQQQQ")

    @classmethod
    def from_bytes(cls, payload: bytes) -> DeviceLoopReport:
        if len(payload) != cls._STRUCT.size:
            raise ContractError(f"T04 device report must be {cls._STRUCT.size} bytes")
        return cls(*cls._STRUCT.unpack(payload))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class T04Observation:
    case_id: str
    generation: int
    slot: int
    sequence_count: int
    payload_bytes: int
    backend: str
    transport_scope: TransportScope
    handle_kind: str
    driver_kernel: str
    service_kernel: str
    driver_context: str
    service_context: str
    driver_launches: int
    service_launches: int
    device_submissions: int
    device_completions: int
    validated_sequences: int
    input_fence: str
    output_fence: str
    host_hot_path_control_messages: int
    host_hot_path_task_messages: int
    host_hot_path_completion_messages: int
    host_hot_path_payload_bytes: int
    host_bounce_bytes: int
    fallback_used: bool
    driver_report: DeviceLoopReport
    service_report: DeviceLoopReport
    driver_binary_sha256: str
    service_binary_sha256: str

    def validate(self) -> None:
        if self.case_id != T04_CASE_ID:
            raise ContractError(f"case_id must be {T04_CASE_ID}")
        if self.generation < 1:
            raise ContractError("generation must be positive")
        if self.slot != T04_SLOT:
            raise ContractError(f"T04 must use slot {T04_SLOT}")
        if self.sequence_count != T04_SEQUENCE_COUNT:
            raise ContractError(f"T04 must run {T04_SEQUENCE_COUNT} sequences")
        if self.payload_bytes != T04_PAYLOAD_BYTES:
            raise ContractError(f"T04 payload must be {T04_PAYLOAD_BYTES} bytes")
        counters = (
            self.driver_launches,
            self.service_launches,
            self.device_submissions,
            self.device_completions,
            self.validated_sequences,
            self.host_hot_path_control_messages,
            self.host_hot_path_task_messages,
            self.host_hot_path_completion_messages,
            self.host_hot_path_payload_bytes,
            self.host_bounce_bytes,
        )
        if min(counters) < 0:
            raise ContractError("T04 counters must be non-negative")
        if len(self.driver_binary_sha256) != 64 or len(self.service_binary_sha256) != 64:
            raise ContractError("T04 kernel hashes must be SHA-256 digests")

    @property
    def passed(self) -> bool:
        self.validate()
        reports = (self.driver_report, self.service_report)
        return all(
            (
                self.backend == STAGE1A_BACKEND,
                self.transport_scope is TransportScope.HOST_LOCAL,
                self.handle_kind == STAGE1A_HANDLE_KIND,
                self.driver_kernel == T04_DRIVER_KERNEL,
                self.service_kernel == T04_SERVICE_KERNEL,
                self.driver_context == T04_DEVICE_CONTEXT,
                self.service_context == T04_DEVICE_CONTEXT,
                self.driver_launches == 1,
                self.service_launches == 1,
                self.device_submissions == T04_SEQUENCE_COUNT,
                self.device_completions == T04_SEQUENCE_COUNT,
                self.validated_sequences == T04_SEQUENCE_COUNT,
                self.input_fence == T04_INPUT_FENCE,
                self.output_fence == T04_OUTPUT_FENCE,
                self.host_hot_path_control_messages == 0,
                self.host_hot_path_task_messages == 0,
                self.host_hot_path_completion_messages == 0,
                self.host_hot_path_payload_bytes == 0,
                self.host_bounce_bytes == 0,
                not self.fallback_used,
                all(report.processed == T04_SEQUENCE_COUNT for report in reports),
                all(
                    report.validation_errors == 0
                    and report.sequence_errors == 0
                    and report.generation_errors == 0
                    and report.checksum_errors == 0
                    and report.marker_errors == 0
                    and report.timeouts == 0
                    and report.elapsed_cycles > 0
                    for report in reports
                ),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["transport_scope"] = self.transport_scope.value
        result["passed"] = self.passed
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> T04Observation:
        data = dict(value)
        data.pop("passed", None)
        try:
            data["transport_scope"] = TransportScope(data["transport_scope"])
            data["driver_report"] = DeviceLoopReport(**data["driver_report"])
            data["service_report"] = DeviceLoopReport(**data["service_report"])
            observation = cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid T04 observation: {exc}") from exc
        observation.validate()
        return observation
