# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Fail-closed evidence contracts for Stage 1B device-loop cases."""

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

T05_CASE_ID = "T05"
T05_SEQUENCE_COUNT = 100
T05_SLOT_COUNT = 2
T05_MAX_INFLIGHT = 2
T05_PAYLOAD_BYTES = (4 * 1024, 64 * 1024)
T05_SLOT_STRIDE = 1024 * 1024
T05_CONTROL_OFFSET = T05_SLOT_COUNT * T05_SLOT_STRIDE
T05_CONTROL_BYTES = (T05_SLOT_COUNT * 2 * 64) + (2 * 64)
T05_WINDOW_BYTES = T05_CONTROL_OFFSET + T05_CONTROL_BYTES
T05_DRIVER_KERNEL = "pypto_stage1b_t05_driver_0_mix_aiv"
T05_SERVICE_KERNEL = "pypto_stage1b_t05_service_0_mix_aiv"
T05_DEVICE_CONTEXT = "AIV_DEVICE_KERNEL"
T05_INPUT_FENCE = "st_dev_payload_metadata+dsb_all+st_dev_submission"
T05_OUTPUT_FENCE = "st_dev_output_metadata+dsb_all+st_dev_completion"


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
class T05DeviceLoopReport:
    processed: int
    validation_errors: int
    sequence_errors: int
    generation_errors: int
    checksum_errors: int
    marker_errors: int
    timeouts: int
    elapsed_cycles: int
    slot0_processed: int
    slot1_processed: int
    credits_acquired: int
    terminal_tasks: int
    credits_returned: int
    max_inflight: int
    out_of_order_completions: int
    slot_overwrite_errors: int

    _STRUCT = struct.Struct("<QQQQQQQQQQQQQQQQ")

    @classmethod
    def from_bytes(cls, payload: bytes) -> T05DeviceLoopReport:
        if len(payload) != cls._STRUCT.size:
            raise ContractError(f"T05 device report must be {cls._STRUCT.size} bytes")
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


@dataclass(frozen=True)
class T05Observation:
    case_id: str
    generation: int
    slot_count: int
    max_inflight: int
    sequence_count: int
    payload_bytes: tuple[int, int]
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
    credits_acquired: int
    terminal_tasks: int
    credits_returned: int
    out_of_order_completions: int
    input_fence: str
    output_fence: str
    host_hot_path_control_messages: int
    host_hot_path_task_messages: int
    host_hot_path_completion_messages: int
    host_hot_path_payload_bytes: int
    host_bounce_bytes: int
    fallback_used: bool
    driver_report: T05DeviceLoopReport
    service_report: T05DeviceLoopReport
    driver_binary_sha256: str
    service_binary_sha256: str

    def validate(self) -> None:
        if self.case_id != T05_CASE_ID:
            raise ContractError(f"case_id must be {T05_CASE_ID}")
        if self.generation < 1:
            raise ContractError("generation must be positive")
        if self.slot_count != T05_SLOT_COUNT:
            raise ContractError(f"T05 must use {T05_SLOT_COUNT} slots")
        if self.max_inflight != T05_MAX_INFLIGHT:
            raise ContractError(f"T05 max_inflight must be {T05_MAX_INFLIGHT}")
        if self.sequence_count != T05_SEQUENCE_COUNT:
            raise ContractError(f"T05 must run {T05_SEQUENCE_COUNT} sequences")
        if tuple(self.payload_bytes) != T05_PAYLOAD_BYTES:
            raise ContractError(f"T05 payload sizes must be {T05_PAYLOAD_BYTES}")
        counters = (
            self.driver_launches,
            self.service_launches,
            self.device_submissions,
            self.device_completions,
            self.validated_sequences,
            self.credits_acquired,
            self.terminal_tasks,
            self.credits_returned,
            self.out_of_order_completions,
            self.host_hot_path_control_messages,
            self.host_hot_path_task_messages,
            self.host_hot_path_completion_messages,
            self.host_hot_path_payload_bytes,
            self.host_bounce_bytes,
        )
        if min(counters) < 0:
            raise ContractError("T05 counters must be non-negative")
        if len(self.driver_binary_sha256) != 64 or len(self.service_binary_sha256) != 64:
            raise ContractError("T05 kernel hashes must be SHA-256 digests")

    @staticmethod
    def _common_report_passed(report: T05DeviceLoopReport) -> bool:
        return all(
            (
                report.processed == T05_SEQUENCE_COUNT,
                report.validation_errors == 0,
                report.sequence_errors == 0,
                report.generation_errors == 0,
                report.checksum_errors == 0,
                report.marker_errors == 0,
                report.timeouts == 0,
                report.elapsed_cycles > 0,
                report.slot0_processed == T05_SEQUENCE_COUNT // 2,
                report.slot1_processed == T05_SEQUENCE_COUNT // 2,
                report.terminal_tasks == T05_SEQUENCE_COUNT,
                report.max_inflight == T05_MAX_INFLIGHT,
                report.out_of_order_completions == T05_SEQUENCE_COUNT // 2,
                report.slot_overwrite_errors == 0,
            )
        )

    @property
    def passed(self) -> bool:
        self.validate()
        return all(
            (
                self.backend == STAGE1A_BACKEND,
                self.transport_scope is TransportScope.HOST_LOCAL,
                self.handle_kind == STAGE1A_HANDLE_KIND,
                self.driver_kernel == T05_DRIVER_KERNEL,
                self.service_kernel == T05_SERVICE_KERNEL,
                self.driver_context == T05_DEVICE_CONTEXT,
                self.service_context == T05_DEVICE_CONTEXT,
                self.driver_launches == 1,
                self.service_launches == 1,
                self.device_submissions == T05_SEQUENCE_COUNT,
                self.device_completions == T05_SEQUENCE_COUNT,
                self.validated_sequences == T05_SEQUENCE_COUNT,
                self.credits_acquired == T05_SEQUENCE_COUNT,
                self.terminal_tasks == T05_SEQUENCE_COUNT,
                self.credits_returned == T05_SEQUENCE_COUNT,
                self.out_of_order_completions == T05_SEQUENCE_COUNT // 2,
                self.input_fence == T05_INPUT_FENCE,
                self.output_fence == T05_OUTPUT_FENCE,
                self.host_hot_path_control_messages == 0,
                self.host_hot_path_task_messages == 0,
                self.host_hot_path_completion_messages == 0,
                self.host_hot_path_payload_bytes == 0,
                self.host_bounce_bytes == 0,
                not self.fallback_used,
                self._common_report_passed(self.driver_report),
                self._common_report_passed(self.service_report),
                self.driver_report.credits_acquired == T05_SEQUENCE_COUNT,
                self.driver_report.credits_returned == T05_SEQUENCE_COUNT,
                self.service_report.credits_acquired == 0,
                self.service_report.credits_returned == 0,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["payload_bytes"] = list(self.payload_bytes)
        result["transport_scope"] = self.transport_scope.value
        result["passed"] = self.passed
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> T05Observation:
        data = dict(value)
        data.pop("passed", None)
        try:
            data["payload_bytes"] = tuple(data["payload_bytes"])
            data["transport_scope"] = TransportScope(data["transport_scope"])
            data["driver_report"] = T05DeviceLoopReport(**data["driver_report"])
            data["service_report"] = T05DeviceLoopReport(**data["service_report"])
            observation = cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid T05 observation: {exc}") from exc
        observation.validate()
        return observation
