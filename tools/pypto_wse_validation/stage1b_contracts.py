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

T06_CASE_ID = "T06"
T06_SEQUENCE_COUNT = 4
T06_SLOT_COUNT = 2
T06_MAX_INFLIGHT = 2
T06_PAYLOAD_BYTES = 4 * 1024
T06_SLOT_STRIDE = T05_SLOT_STRIDE
T06_CONTROL_OFFSET = T06_SLOT_COUNT * T06_SLOT_STRIDE
T06_CONTROL_BYTES = (T06_SLOT_COUNT * 2 * 64) + 64 + (3 * 64)
T06_WINDOW_BYTES = T06_CONTROL_OFFSET + T06_CONTROL_BYTES
T06_DRIVER_KERNEL = "pypto_stage1b_t06_driver_0_mix_aiv"
T06_SERVICE_KERNEL = "pypto_stage1b_t06_service_0_mix_aiv"
T06_DEVICE_CONTEXT = "AIV_DEVICE_KERNEL"
T06_INPUT_FENCE = T05_INPUT_FENCE
T06_OUTPUT_FENCE = T05_OUTPUT_FENCE
T06_THIRD_REQUEST_OUTCOME = "NO_CREDIT"
T06_BACKPRESSURE_WAIT = "PENDING_STATE_NO_SUBMIT_RETRY"

T07_CASE_ID = "T07"
T07_SEQUENCE_COUNT = 100
T07_SLOT_COUNT = 2
T07_MAX_INFLIGHT = 1
T07_PAYLOAD_BYTES = 1024 * 1024
T07_SLOT_STRIDE = T06_SLOT_STRIDE
T07_CONTROL_OFFSET = T07_SLOT_COUNT * T07_SLOT_STRIDE
T07_CONTROL_BYTES = (T07_SLOT_COUNT * 2 * 64) + (4 * 64)
T07_WINDOW_BYTES = T07_CONTROL_OFFSET + T07_CONTROL_BYTES
T07_DRIVER_KERNEL = "pypto_stage1b_t07_driver_0_mix_aiv"
T07_SERVICE_KERNEL = "pypto_stage1b_t07_service_0_mix_aiv"
T07_DEVICE_CONTEXT = "AIV_DEVICE_KERNEL"
T07_INPUT_FENCE = "st_dev_large_payload_markers+dsb_all+st_dev_descriptor+dsb_all+st_dev_submission"
T07_OUTPUT_FENCE = "st_dev_large_output_markers+dsb_all+st_dev_completion_metadata+dsb_all+st_dev_completion"
T07_COMPLETION_CHECK = "IMMEDIATE_AFTER_SIGNAL_NO_DELAY"

T08_CASE_ID = "T08"
T08_BASELINE_SEQUENCE_COUNT = 4
T08_NEXT_SEQUENCE_COUNT = 1
T08_SLOT_COUNT = 1
T08_MAX_INFLIGHT = 1
T08_PAYLOAD_BYTES = 4 * 1024
T08_CONTROL_OFFSET = 1024 * 1024
T08_CONTROL_BYTES = (2 * 2 * 64) + 64 + (3 * 64)
T08_WINDOW_BYTES = T08_CONTROL_OFFSET + T08_CONTROL_BYTES
T08_DRIVER_KERNEL = "pypto_stage1b_t08_driver_0_mix_aiv"
T08_SERVICE_KERNEL = "pypto_stage1b_t08_service_0_mix_aiv"
T08_DEVICE_CONTEXT = "AIV_DEVICE_KERNEL"
T08_INPUT_FENCE = "st_dev_payload_metadata+dsb_all+st_dev_submission"
T08_OUTPUT_FENCE = "st_dev_output_metadata+dsb_all+st_dev_completion"
T08_STALE_STATUS = "STALE_GENERATION"
T08_HANDLE_PROBE_API = "aclrtMemImportFromShareableHandle"


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
class T06DeviceLoopReport:
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
    slot_overwrite_errors: int
    no_credit_events: int
    pending_requests: int
    submission_retry_spins: int
    service_pause_observed: int
    service_resume_observed: int
    progress_after_resume: int
    peak_queue_depth: int
    submissions: int
    completions: int

    _STRUCT = struct.Struct("<" + ("Q" * 24))

    @classmethod
    def from_bytes(cls, payload: bytes) -> T06DeviceLoopReport:
        if len(payload) != cls._STRUCT.size:
            raise ContractError(f"T06 device report must be {cls._STRUCT.size} bytes")
        return cls(*cls._STRUCT.unpack(payload))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class T07DeviceLoopReport:
    processed: int
    validation_errors: int
    sequence_errors: int
    generation_errors: int
    checksum_errors: int
    head_marker_errors: int
    tail_marker_errors: int
    stale_payload_errors: int
    incomplete_payload_errors: int
    premature_completion_errors: int
    timeouts: int
    elapsed_cycles: int
    slot0_processed: int
    slot1_processed: int
    input_publish_fences: int
    input_visibility_checks: int
    output_publish_fences: int
    immediate_completion_checks: int
    post_completion_delay_cycles: int
    submissions: int
    completions: int
    slot_overwrite_errors: int
    payload_words_validated: int
    unique_tail_markers_validated: int
    max_inflight: int
    reserved: int

    _STRUCT = struct.Struct("<" + ("Q" * 26))

    @classmethod
    def from_bytes(cls, payload: bytes) -> T07DeviceLoopReport:
        if len(payload) != cls._STRUCT.size:
            raise ContractError(f"T07 device report must be {cls._STRUCT.size} bytes")
        return cls(*cls._STRUCT.unpack(payload))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class T08DeviceLoopReport:
    processed: int
    validation_errors: int
    sequence_errors: int
    generation_errors: int
    checksum_errors: int
    marker_errors: int
    timeouts: int
    elapsed_cycles: int
    submissions: int
    completions: int
    stale_descriptor_injections: int
    stale_descriptor_rejections: int
    stale_completion_injections: int
    stale_completion_rejections: int
    old_completion_credit_releases: int
    current_slot_preserved: int
    credits_acquired: int
    credits_returned: int
    terminal_tasks: int
    progress_after_stale: int
    input_fences: int
    output_fences: int
    mode: int
    reserved: int

    _STRUCT = struct.Struct("<" + ("Q" * 24))

    @classmethod
    def from_bytes(cls, payload: bytes) -> T08DeviceLoopReport:
        if len(payload) != cls._STRUCT.size:
            raise ContractError(f"T08 device report must be {cls._STRUCT.size} bytes")
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


@dataclass(frozen=True)
class T06Observation:
    case_id: str
    generation: int
    slot_count: int
    max_inflight: int
    sequence_count: int
    payload_bytes: int
    attempted_submissions: int
    device_submissions: int
    device_completions: int
    validated_sequences: int
    no_credit_events: int
    pending_requests: int
    terminal_tasks: int
    credits_acquired: int
    credits_returned: int
    submission_retry_spins: int
    service_pause_observed: bool
    service_resume_observed: bool
    progress_after_resume: int
    third_request_outcome: str
    backpressure_wait: str
    backend: str
    transport_scope: TransportScope
    handle_kind: str
    driver_kernel: str
    service_kernel: str
    driver_context: str
    service_context: str
    driver_launches: int
    service_launches: int
    input_fence: str
    output_fence: str
    host_hot_path_control_messages: int
    host_hot_path_task_messages: int
    host_hot_path_completion_messages: int
    host_hot_path_payload_bytes: int
    host_bounce_bytes: int
    fallback_used: bool
    driver_report: T06DeviceLoopReport
    service_report: T06DeviceLoopReport
    driver_binary_sha256: str
    service_binary_sha256: str

    def validate(self) -> None:
        if self.case_id != T06_CASE_ID:
            raise ContractError(f"case_id must be {T06_CASE_ID}")
        if self.generation < 1:
            raise ContractError("generation must be positive")
        if self.slot_count != T06_SLOT_COUNT:
            raise ContractError(f"T06 must use {T06_SLOT_COUNT} slots")
        if self.max_inflight != T06_MAX_INFLIGHT:
            raise ContractError(f"T06 max_inflight must be {T06_MAX_INFLIGHT}")
        if self.sequence_count != T06_SEQUENCE_COUNT:
            raise ContractError(f"T06 must run {T06_SEQUENCE_COUNT} sequences")
        if self.payload_bytes != T06_PAYLOAD_BYTES:
            raise ContractError(f"T06 payload must be {T06_PAYLOAD_BYTES} bytes")
        counters = (
            self.attempted_submissions,
            self.device_submissions,
            self.device_completions,
            self.validated_sequences,
            self.no_credit_events,
            self.pending_requests,
            self.terminal_tasks,
            self.credits_acquired,
            self.credits_returned,
            self.submission_retry_spins,
            self.progress_after_resume,
            self.driver_launches,
            self.service_launches,
            self.host_hot_path_control_messages,
            self.host_hot_path_task_messages,
            self.host_hot_path_completion_messages,
            self.host_hot_path_payload_bytes,
            self.host_bounce_bytes,
        )
        if min(counters) < 0:
            raise ContractError("T06 counters must be non-negative")
        if len(self.driver_binary_sha256) != 64 or len(self.service_binary_sha256) != 64:
            raise ContractError("T06 kernel hashes must be SHA-256 digests")

    @staticmethod
    def _common_report_passed(report: T06DeviceLoopReport) -> bool:
        return all(
            (
                report.processed == T06_SEQUENCE_COUNT,
                report.validation_errors == 0,
                report.sequence_errors == 0,
                report.generation_errors == 0,
                report.checksum_errors == 0,
                report.marker_errors == 0,
                report.timeouts == 0,
                report.elapsed_cycles > 0,
                report.slot0_processed == T06_SEQUENCE_COUNT // 2,
                report.slot1_processed == T06_SEQUENCE_COUNT // 2,
                report.terminal_tasks == T06_SEQUENCE_COUNT,
                report.max_inflight == T06_MAX_INFLIGHT,
                report.slot_overwrite_errors == 0,
                report.no_credit_events == 1,
                report.pending_requests == 1,
                report.submission_retry_spins == 0,
                report.service_pause_observed == 1,
                report.service_resume_observed == 1,
                report.progress_after_resume == T06_SEQUENCE_COUNT,
                report.peak_queue_depth == T06_MAX_INFLIGHT,
                report.submissions == T06_SEQUENCE_COUNT,
                report.completions == T06_SEQUENCE_COUNT,
            )
        )

    @property
    def passed(self) -> bool:
        self.validate()
        return all(
            (
                self.attempted_submissions == T06_SEQUENCE_COUNT + 1,
                self.device_submissions == T06_SEQUENCE_COUNT,
                self.device_completions == T06_SEQUENCE_COUNT,
                self.validated_sequences == T06_SEQUENCE_COUNT,
                self.no_credit_events == 1,
                self.pending_requests == 1,
                self.terminal_tasks == T06_SEQUENCE_COUNT,
                self.credits_acquired == T06_SEQUENCE_COUNT,
                self.credits_returned == T06_SEQUENCE_COUNT,
                self.submission_retry_spins == 0,
                self.service_pause_observed,
                self.service_resume_observed,
                self.progress_after_resume == T06_SEQUENCE_COUNT,
                self.third_request_outcome == T06_THIRD_REQUEST_OUTCOME,
                self.backpressure_wait == T06_BACKPRESSURE_WAIT,
                self.backend == STAGE1A_BACKEND,
                self.transport_scope is TransportScope.HOST_LOCAL,
                self.handle_kind == STAGE1A_HANDLE_KIND,
                self.driver_kernel == T06_DRIVER_KERNEL,
                self.service_kernel == T06_SERVICE_KERNEL,
                self.driver_context == T06_DEVICE_CONTEXT,
                self.service_context == T06_DEVICE_CONTEXT,
                self.driver_launches == 1,
                self.service_launches == 1,
                self.input_fence == T06_INPUT_FENCE,
                self.output_fence == T06_OUTPUT_FENCE,
                self.host_hot_path_control_messages == 0,
                self.host_hot_path_task_messages == 0,
                self.host_hot_path_completion_messages == 0,
                self.host_hot_path_payload_bytes == 0,
                self.host_bounce_bytes == 0,
                not self.fallback_used,
                self._common_report_passed(self.driver_report),
                self._common_report_passed(self.service_report),
                self.driver_report.credits_acquired == T06_SEQUENCE_COUNT,
                self.driver_report.credits_returned == T06_SEQUENCE_COUNT,
                self.service_report.credits_acquired == 0,
                self.service_report.credits_returned == 0,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["transport_scope"] = self.transport_scope.value
        result["passed"] = self.passed
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> T06Observation:
        data = dict(value)
        data.pop("passed", None)
        try:
            data["transport_scope"] = TransportScope(data["transport_scope"])
            data["driver_report"] = T06DeviceLoopReport(**data["driver_report"])
            data["service_report"] = T06DeviceLoopReport(**data["service_report"])
            observation = cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid T06 observation: {exc}") from exc
        observation.validate()
        return observation


@dataclass(frozen=True)
class T07Observation:
    case_id: str
    generation: int
    slot_count: int
    max_inflight: int
    sequence_count: int
    payload_bytes: int
    device_submissions: int
    device_completions: int
    validated_sequences: int
    unique_tail_markers_validated: int
    immediate_completion_checks: int
    post_completion_delay_cycles: int
    stale_payload_errors: int
    incomplete_payload_errors: int
    premature_completion_errors: int
    completion_check: str
    backend: str
    transport_scope: TransportScope
    handle_kind: str
    driver_kernel: str
    service_kernel: str
    driver_context: str
    service_context: str
    driver_launches: int
    service_launches: int
    input_fence: str
    output_fence: str
    host_hot_path_control_messages: int
    host_hot_path_task_messages: int
    host_hot_path_completion_messages: int
    host_hot_path_payload_bytes: int
    host_bounce_bytes: int
    fallback_used: bool
    driver_report: T07DeviceLoopReport
    service_report: T07DeviceLoopReport
    driver_binary_sha256: str
    service_binary_sha256: str

    def validate(self) -> None:
        if self.case_id != T07_CASE_ID:
            raise ContractError(f"case_id must be {T07_CASE_ID}")
        if self.generation < 1:
            raise ContractError("generation must be positive")
        if self.slot_count != T07_SLOT_COUNT:
            raise ContractError(f"T07 must use {T07_SLOT_COUNT} slots")
        if self.max_inflight != T07_MAX_INFLIGHT:
            raise ContractError(f"T07 max_inflight must be {T07_MAX_INFLIGHT}")
        if self.sequence_count != T07_SEQUENCE_COUNT:
            raise ContractError(f"T07 must run {T07_SEQUENCE_COUNT} sequences")
        if self.payload_bytes != T07_PAYLOAD_BYTES:
            raise ContractError(f"T07 payload must be {T07_PAYLOAD_BYTES} bytes")
        counters = (
            self.device_submissions,
            self.device_completions,
            self.validated_sequences,
            self.unique_tail_markers_validated,
            self.immediate_completion_checks,
            self.post_completion_delay_cycles,
            self.stale_payload_errors,
            self.incomplete_payload_errors,
            self.premature_completion_errors,
            self.driver_launches,
            self.service_launches,
            self.host_hot_path_control_messages,
            self.host_hot_path_task_messages,
            self.host_hot_path_completion_messages,
            self.host_hot_path_payload_bytes,
            self.host_bounce_bytes,
        )
        if min(counters) < 0:
            raise ContractError("T07 counters must be non-negative")
        if len(self.driver_binary_sha256) != 64 or len(self.service_binary_sha256) != 64:
            raise ContractError("T07 kernel hashes must be SHA-256 digests")

    @staticmethod
    def _common_report_passed(report: T07DeviceLoopReport) -> bool:
        return all(
            (
                report.processed == T07_SEQUENCE_COUNT,
                report.validation_errors == 0,
                report.sequence_errors == 0,
                report.generation_errors == 0,
                report.checksum_errors == 0,
                report.head_marker_errors == 0,
                report.tail_marker_errors == 0,
                report.stale_payload_errors == 0,
                report.incomplete_payload_errors == 0,
                report.premature_completion_errors == 0,
                report.timeouts == 0,
                report.elapsed_cycles > 0,
                report.slot0_processed == T07_SEQUENCE_COUNT // 2,
                report.slot1_processed == T07_SEQUENCE_COUNT // 2,
                report.post_completion_delay_cycles == 0,
                report.submissions == T07_SEQUENCE_COUNT,
                report.completions == T07_SEQUENCE_COUNT,
                report.slot_overwrite_errors == 0,
                report.payload_words_validated == T07_SEQUENCE_COUNT * (T07_PAYLOAD_BYTES // struct.calcsize("Q")),
                report.unique_tail_markers_validated == T07_SEQUENCE_COUNT,
                report.max_inflight == T07_MAX_INFLIGHT,
                report.reserved == 0,
            )
        )

    @property
    def passed(self) -> bool:
        self.validate()
        return all(
            (
                self.device_submissions == T07_SEQUENCE_COUNT,
                self.device_completions == T07_SEQUENCE_COUNT,
                self.validated_sequences == T07_SEQUENCE_COUNT,
                self.unique_tail_markers_validated == T07_SEQUENCE_COUNT,
                self.immediate_completion_checks == T07_SEQUENCE_COUNT,
                self.post_completion_delay_cycles == 0,
                self.stale_payload_errors == 0,
                self.incomplete_payload_errors == 0,
                self.premature_completion_errors == 0,
                self.completion_check == T07_COMPLETION_CHECK,
                self.backend == STAGE1A_BACKEND,
                self.transport_scope is TransportScope.HOST_LOCAL,
                self.handle_kind == STAGE1A_HANDLE_KIND,
                self.driver_kernel == T07_DRIVER_KERNEL,
                self.service_kernel == T07_SERVICE_KERNEL,
                self.driver_context == T07_DEVICE_CONTEXT,
                self.service_context == T07_DEVICE_CONTEXT,
                self.driver_launches == 1,
                self.service_launches == 1,
                self.input_fence == T07_INPUT_FENCE,
                self.output_fence == T07_OUTPUT_FENCE,
                self.host_hot_path_control_messages == 0,
                self.host_hot_path_task_messages == 0,
                self.host_hot_path_completion_messages == 0,
                self.host_hot_path_payload_bytes == 0,
                self.host_bounce_bytes == 0,
                not self.fallback_used,
                self._common_report_passed(self.driver_report),
                self._common_report_passed(self.service_report),
                self.driver_report.input_publish_fences == T07_SEQUENCE_COUNT,
                self.driver_report.input_visibility_checks == 0,
                self.driver_report.output_publish_fences == 0,
                self.driver_report.immediate_completion_checks == T07_SEQUENCE_COUNT,
                self.service_report.input_publish_fences == 0,
                self.service_report.input_visibility_checks == T07_SEQUENCE_COUNT,
                self.service_report.output_publish_fences == T07_SEQUENCE_COUNT,
                self.service_report.immediate_completion_checks == 0,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["transport_scope"] = self.transport_scope.value
        result["passed"] = self.passed
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> T07Observation:
        data = dict(value)
        data.pop("passed", None)
        try:
            data["transport_scope"] = TransportScope(data["transport_scope"])
            data["driver_report"] = T07DeviceLoopReport(**data["driver_report"])
            data["service_report"] = T07DeviceLoopReport(**data["service_report"])
            observation = cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid T07 observation: {exc}") from exc
        observation.validate()
        return observation


@dataclass(frozen=True)
class T08Observation:
    case_id: str
    generation: int
    next_generation: int
    baseline_sequence_count: int
    next_sequence_count: int
    slot_count: int
    max_inflight: int
    payload_bytes: int
    device_submissions: int
    device_completions: int
    validated_sequences: int
    stale_descriptor_injections: int
    stale_descriptor_rejections: int
    stale_completion_injections: int
    stale_completion_rejections: int
    old_completion_credit_releases: int
    current_slot_preserved: bool
    progress_after_stale: int
    old_handle_rejected_before_reallocate: int
    old_handle_rejected_after_reallocate: int
    old_handle_import_successes: int
    new_handle_collisions: int
    handle_probe_api: str
    backend: str
    transport_scope: TransportScope
    handle_kind: str
    driver_kernel: str
    service_kernel: str
    driver_context: str
    service_context: str
    driver_launches: int
    service_launches: int
    input_fence: str
    output_fence: str
    host_hot_path_control_messages: int
    host_hot_path_task_messages: int
    host_hot_path_completion_messages: int
    host_hot_path_payload_bytes: int
    host_bounce_bytes: int
    fallback_used: bool
    baseline_driver_report: T08DeviceLoopReport
    baseline_service_report: T08DeviceLoopReport
    next_driver_report: T08DeviceLoopReport
    next_service_report: T08DeviceLoopReport
    driver_binary_sha256: str
    service_binary_sha256: str

    def validate(self) -> None:
        if self.case_id != T08_CASE_ID:
            raise ContractError(f"case_id must be {T08_CASE_ID}")
        if self.generation < 1 or self.next_generation != self.generation + 1:
            raise ContractError("T08 generations must be consecutive and positive")
        if self.baseline_sequence_count != T08_BASELINE_SEQUENCE_COUNT:
            raise ContractError(f"T08 baseline must run {T08_BASELINE_SEQUENCE_COUNT} sequences")
        if self.next_sequence_count != T08_NEXT_SEQUENCE_COUNT:
            raise ContractError(f"T08 next generation must run {T08_NEXT_SEQUENCE_COUNT} sequence")
        if self.slot_count != T08_SLOT_COUNT or self.max_inflight != T08_MAX_INFLIGHT:
            raise ContractError("T08 must use one slot and one credit")
        if self.payload_bytes != T08_PAYLOAD_BYTES:
            raise ContractError(f"T08 payload must be {T08_PAYLOAD_BYTES} bytes")
        counters = (
            self.device_submissions,
            self.device_completions,
            self.validated_sequences,
            self.stale_descriptor_injections,
            self.stale_descriptor_rejections,
            self.stale_completion_injections,
            self.stale_completion_rejections,
            self.old_completion_credit_releases,
            self.progress_after_stale,
            self.old_handle_rejected_before_reallocate,
            self.old_handle_rejected_after_reallocate,
            self.old_handle_import_successes,
            self.new_handle_collisions,
            self.driver_launches,
            self.service_launches,
            self.host_hot_path_control_messages,
            self.host_hot_path_task_messages,
            self.host_hot_path_completion_messages,
            self.host_hot_path_payload_bytes,
            self.host_bounce_bytes,
        )
        if min(counters) < 0:
            raise ContractError("T08 counters must be non-negative")
        if len(self.driver_binary_sha256) != 64 or len(self.service_binary_sha256) != 64:
            raise ContractError("T08 kernel hashes must be SHA-256 digests")

    @staticmethod
    def _report_errors_are_zero(report: T08DeviceLoopReport) -> bool:
        return (
            all(
                value == 0
                for value in (
                    report.validation_errors,
                    report.sequence_errors,
                    report.generation_errors,
                    report.checksum_errors,
                    report.marker_errors,
                    report.timeouts,
                    report.old_completion_credit_releases,
                    report.reserved,
                )
            )
            and report.elapsed_cycles > 0
        )

    @classmethod
    def _baseline_report_passed(cls, report: T08DeviceLoopReport, *, driver: bool) -> bool:
        return all(
            (
                cls._report_errors_are_zero(report),
                report.processed == T08_BASELINE_SEQUENCE_COUNT,
                report.submissions == T08_BASELINE_SEQUENCE_COUNT,
                report.completions == T08_BASELINE_SEQUENCE_COUNT,
                report.stale_descriptor_injections == 0,
                report.stale_descriptor_rejections == 0,
                report.stale_completion_injections == 0,
                report.stale_completion_rejections == 0,
                report.current_slot_preserved == 0,
                report.credits_acquired == (T08_BASELINE_SEQUENCE_COUNT if driver else 0),
                report.credits_returned == (T08_BASELINE_SEQUENCE_COUNT if driver else 0),
                report.terminal_tasks == T08_BASELINE_SEQUENCE_COUNT,
                report.progress_after_stale == 0,
                report.input_fences == (T08_BASELINE_SEQUENCE_COUNT if driver else 0),
                report.output_fences == (0 if driver else T08_BASELINE_SEQUENCE_COUNT),
                report.mode == 0,
            )
        )

    @classmethod
    def _next_report_passed(cls, report: T08DeviceLoopReport, *, driver: bool) -> bool:
        return all(
            (
                cls._report_errors_are_zero(report),
                report.processed == T08_NEXT_SEQUENCE_COUNT,
                report.submissions == T08_NEXT_SEQUENCE_COUNT,
                report.completions == T08_NEXT_SEQUENCE_COUNT,
                report.stale_descriptor_injections == (1 if driver else 0),
                report.stale_descriptor_rejections == (0 if driver else 1),
                report.stale_completion_injections == (0 if driver else 1),
                report.stale_completion_rejections == (1 if driver else 0),
                report.current_slot_preserved == 1,
                report.credits_acquired == (T08_NEXT_SEQUENCE_COUNT if driver else 0),
                report.credits_returned == (T08_NEXT_SEQUENCE_COUNT if driver else 0),
                report.terminal_tasks == T08_NEXT_SEQUENCE_COUNT,
                report.progress_after_stale == T08_NEXT_SEQUENCE_COUNT,
                report.input_fences == (T08_NEXT_SEQUENCE_COUNT if driver else 0),
                report.output_fences == (0 if driver else T08_NEXT_SEQUENCE_COUNT),
                report.mode == 1,
            )
        )

    @property
    def passed(self) -> bool:
        self.validate()
        expected_sequences = T08_BASELINE_SEQUENCE_COUNT + T08_NEXT_SEQUENCE_COUNT
        return all(
            (
                self.device_submissions == expected_sequences,
                self.device_completions == expected_sequences,
                self.validated_sequences == expected_sequences,
                self.stale_descriptor_injections == 1,
                self.stale_descriptor_rejections == 1,
                self.stale_completion_injections == 1,
                self.stale_completion_rejections == 1,
                self.old_completion_credit_releases == 0,
                self.current_slot_preserved,
                self.progress_after_stale == T08_NEXT_SEQUENCE_COUNT,
                self.old_handle_rejected_before_reallocate == 2,
                self.old_handle_rejected_after_reallocate == 2,
                self.old_handle_import_successes == 0,
                self.new_handle_collisions == 0,
                self.handle_probe_api == T08_HANDLE_PROBE_API,
                self.backend == STAGE1A_BACKEND,
                self.transport_scope is TransportScope.HOST_LOCAL,
                self.handle_kind == STAGE1A_HANDLE_KIND,
                self.driver_kernel == T08_DRIVER_KERNEL,
                self.service_kernel == T08_SERVICE_KERNEL,
                self.driver_context == T08_DEVICE_CONTEXT,
                self.service_context == T08_DEVICE_CONTEXT,
                self.driver_launches == 2,
                self.service_launches == 2,
                self.input_fence == T08_INPUT_FENCE,
                self.output_fence == T08_OUTPUT_FENCE,
                self.host_hot_path_control_messages == 0,
                self.host_hot_path_task_messages == 0,
                self.host_hot_path_completion_messages == 0,
                self.host_hot_path_payload_bytes == 0,
                self.host_bounce_bytes == 0,
                not self.fallback_used,
                self._baseline_report_passed(self.baseline_driver_report, driver=True),
                self._baseline_report_passed(self.baseline_service_report, driver=False),
                self._next_report_passed(self.next_driver_report, driver=True),
                self._next_report_passed(self.next_service_report, driver=False),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["transport_scope"] = self.transport_scope.value
        result["passed"] = self.passed
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> T08Observation:
        data = dict(value)
        data.pop("passed", None)
        try:
            data["transport_scope"] = TransportScope(data["transport_scope"])
            for name in (
                "baseline_driver_report",
                "baseline_service_report",
                "next_driver_report",
                "next_service_report",
            ):
                data[name] = T08DeviceLoopReport(**data[name])
            observation = cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid T08 observation: {exc}") from exc
        observation.validate()
        return observation
