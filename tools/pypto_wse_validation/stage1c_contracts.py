# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Fail-closed evidence contracts for Stage 1C T09 through T12."""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from typing import Any

from tools.pypto_wse_validation.contracts import ContractError, EndpointRole, TransportScope
from tools.pypto_wse_validation.stage1a_contracts import STAGE1A_BACKEND, STAGE1A_HANDLE_KIND

T09_CASE_ID = "T09"
T09_WARMUP_COUNT = 100
T09_MEASURED_COUNT = 10_000
T09_TOTAL_COUNT = T09_WARMUP_COUNT + T09_MEASURED_COUNT
T09_PAYLOAD_BYTES = (64, 4 * 1024, 64 * 1024, 1024 * 1024)
T09_SLOT_COUNT = 2
T09_MAX_INFLIGHT = 2
T09_PROGRESS_INTERVAL = 100
T09_PROGRESS_CHECKPOINTS = T09_MEASURED_COUNT // T09_PROGRESS_INTERVAL
T09_CONTROL_OFFSET = T09_SLOT_COUNT * max(T09_PAYLOAD_BYTES)
T09_CONTROL_BYTES = (T09_SLOT_COUNT * 2 * 64) + (4 * 64)
T09_WINDOW_BYTES = T09_CONTROL_OFFSET + T09_CONTROL_BYTES
T09_DRIVER_KERNEL = "pypto_stage1c_t09_driver_0_mix_aiv"
T09_SERVICE_KERNEL = "pypto_stage1c_t09_service_0_mix_aiv"

T10_CASE_ID = "T10"
T10_PAYLOAD_BYTES = 4 * 1024
T10_TIMEOUT_CYCLES = 500_000_000
T10_CONTROL_OFFSET = 1024 * 1024
T10_CONTROL_BYTES = (2 * 64) + (3 * 64)
T10_WINDOW_BYTES = T10_CONTROL_OFFSET + T10_CONTROL_BYTES
T10_DRIVER_KERNEL = "pypto_stage1c_fault_driver_0_mix_aiv"
T10_SERVICE_KERNEL = "pypto_stage1c_fault_service_0_mix_aiv"

T11_CASE_ID = "T11"
T11_TIMEOUT_CYCLES = 20_000_000_000
T11_FAULT_LIMITATION = "SAME_HOST_PROCESS_EXIT_NOT_HOST_POWER_LOSS_OR_NETWORK_PARTITION"

T12_CASE_ID = "T12"
T12_ACCEPTED_COUNT = 4
T12_PAYLOAD_BYTES = 4 * 1024
T12_SLOT_COUNT = 2
T12_MAX_INFLIGHT = 2
T12_CONTROL_OFFSET = 1024 * 1024
T12_CONTROL_BYTES = (T12_SLOT_COUNT * 2 * 64) + 64 + (3 * 64)
T12_WINDOW_BYTES = T12_CONTROL_OFFSET + T12_CONTROL_BYTES
T12_DRIVER_KERNEL = "pypto_stage1c_t12_driver_0_mix_aiv"
T12_SERVICE_KERNEL = "pypto_stage1c_t12_service_0_mix_aiv"
T12_CLOSE_ORDER = (
    "STOP_NEW_SUBMISSIONS",
    "DRAIN_ACCEPTED_REQUESTS",
    "VALIDATE_CREDITS_RETURNED",
    "STOP_SERVICE",
    "RELEASE_PEER_IMPORT",
    "UNREGISTER_LOCAL_WINDOW",
    "FREE_LOCAL_MEMORY",
    "DESTROY_RUNTIME",
)

DEVICE_CONTEXT = "AIV_DEVICE_KERNEL"
INPUT_FENCE = "st_dev_payload_metadata+dsb_all+st_dev_submission"
OUTPUT_FENCE = "st_dev_output_metadata+dsb_all+st_dev_completion"


@dataclass(frozen=True)
class T09DeviceReport:
    processed: int
    warmup_processed: int
    measured_processed: int
    validation_errors: int
    sequence_errors: int
    generation_errors: int
    checksum_errors: int
    marker_errors: int
    timeouts: int
    elapsed_cycles: int
    slot0_processed: int
    slot1_processed: int
    submissions: int
    completions: int
    credits_acquired: int
    credits_returned: int
    terminal_tasks: int
    max_inflight: int
    queue_stalls: int
    progress_checkpoints: int
    progress_interval: int
    payload_64_count: int
    payload_4k_count: int
    payload_64k_count: int
    payload_1m_count: int
    payload_words_validated: int
    input_fences: int
    output_fences: int
    reserved0: int
    reserved1: int
    reserved2: int
    reserved3: int

    _STRUCT = struct.Struct("<" + ("Q" * 32))

    @classmethod
    def from_bytes(cls, payload: bytes) -> T09DeviceReport:
        if len(payload) != cls._STRUCT.size:
            raise ContractError(f"T09 device report must be {cls._STRUCT.size} bytes")
        return cls(*cls._STRUCT.unpack(payload))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class FaultDeviceReport:
    accepted: int
    timed_out: int
    successes: int
    validation_errors: int
    sequence_errors: int
    generation_errors: int
    timeouts: int
    elapsed_cycles: int
    submissions: int
    completions: int
    service_pause_observed: int
    generation_unhealthy: int
    accepting_new_requests: int
    quarantined_slots: int
    terminal_errors: int
    forged_success_completions: int
    credits_returned: int
    slot_reuse_after_timeout: int
    configured_timeout_cycles: int
    timeout_within_bound: int
    input_fences: int
    output_fences: int
    mode: int
    reserved: int

    _STRUCT = struct.Struct("<" + ("Q" * 24))

    @classmethod
    def from_bytes(cls, payload: bytes) -> FaultDeviceReport:
        if len(payload) != cls._STRUCT.size:
            raise ContractError(f"fault device report must be {cls._STRUCT.size} bytes")
        return cls(*cls._STRUCT.unpack(payload))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class T12DeviceReport:
    accepted: int
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
    credits_acquired: int
    credits_returned: int
    terminal_tasks: int
    max_inflight: int
    stop_new_submissions: int
    post_stop_attempts: int
    post_stop_rejections: int
    drain_started: int
    drain_completed: int
    service_stopped: int
    input_fences: int
    output_fences: int
    reserved: int

    _STRUCT = struct.Struct("<" + ("Q" * 24))

    @classmethod
    def from_bytes(cls, payload: bytes) -> T12DeviceReport:
        if len(payload) != cls._STRUCT.size:
            raise ContractError(f"T12 device report must be {cls._STRUCT.size} bytes")
        return cls(*cls._STRUCT.unpack(payload))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _hashes_valid(driver_hash: str, service_hash: str) -> bool:
    return len(driver_hash) == 64 and len(service_hash) == 64


def _base_transport_passed(
    *,
    backend: str,
    transport_scope: TransportScope,
    handle_kind: str,
    host_hot_path_messages: int,
    host_hot_path_payload_bytes: int,
    host_bounce_bytes: int,
    fallback_used: bool,
) -> bool:
    return all(
        (
            backend == STAGE1A_BACKEND,
            transport_scope is TransportScope.HOST_LOCAL,
            handle_kind == STAGE1A_HANDLE_KIND,
            host_hot_path_messages == 0,
            host_hot_path_payload_bytes == 0,
            host_bounce_bytes == 0,
            not fallback_used,
        )
    )


@dataclass(frozen=True)
class T09Observation:
    case_id: str
    generation: int
    backend: str
    transport_scope: TransportScope
    handle_kind: str
    warmup_count: int
    measured_count: int
    payload_bytes: tuple[int, ...]
    slot_count: int
    max_inflight: int
    progress_interval: int
    progress_checkpoints: int
    device_submissions: int
    device_completions: int
    validated_sequences: int
    device_memory_growth_bytes: int
    registration_growth: int
    host_memory_growth_bytes: int
    host_memory_growth_limit_bytes: int
    host_cpu_utilization_pct: float
    queue_stalls: int
    host_hot_path_messages: int
    host_hot_path_payload_bytes: int
    host_bounce_bytes: int
    fallback_used: bool
    driver_report: T09DeviceReport
    service_report: T09DeviceReport
    driver_binary_sha256: str
    service_binary_sha256: str

    @property
    def passed(self) -> bool:
        expected_payload_count = T09_MEASURED_COUNT // len(T09_PAYLOAD_BYTES)
        reports = (self.driver_report, self.service_report)
        return all(
            (
                self.case_id == T09_CASE_ID,
                self.generation > 0,
                self.warmup_count == T09_WARMUP_COUNT,
                self.measured_count >= T09_MEASURED_COUNT,
                self.payload_bytes == T09_PAYLOAD_BYTES,
                self.slot_count == T09_SLOT_COUNT,
                self.max_inflight == T09_MAX_INFLIGHT,
                self.progress_interval == T09_PROGRESS_INTERVAL,
                self.progress_checkpoints == T09_PROGRESS_CHECKPOINTS,
                self.device_submissions == T09_TOTAL_COUNT,
                self.device_completions == T09_TOTAL_COUNT,
                self.validated_sequences == T09_TOTAL_COUNT,
                self.device_memory_growth_bytes == 0,
                self.registration_growth == 0,
                0 <= self.host_memory_growth_bytes <= self.host_memory_growth_limit_bytes,
                0 <= self.host_cpu_utilization_pct <= 100,
                self.queue_stalls == 0,
                _base_transport_passed(
                    backend=self.backend,
                    transport_scope=self.transport_scope,
                    handle_kind=self.handle_kind,
                    host_hot_path_messages=self.host_hot_path_messages,
                    host_hot_path_payload_bytes=self.host_hot_path_payload_bytes,
                    host_bounce_bytes=self.host_bounce_bytes,
                    fallback_used=self.fallback_used,
                ),
                _hashes_valid(self.driver_binary_sha256, self.service_binary_sha256),
                all(
                    report.processed == T09_TOTAL_COUNT
                    and report.warmup_processed == T09_WARMUP_COUNT
                    and report.measured_processed == T09_MEASURED_COUNT
                    and report.validation_errors == 0
                    and report.sequence_errors == 0
                    and report.generation_errors == 0
                    and report.checksum_errors == 0
                    and report.marker_errors == 0
                    and report.timeouts == 0
                    and report.elapsed_cycles > 0
                    and report.slot0_processed == T09_TOTAL_COUNT // 2
                    and report.slot1_processed == T09_TOTAL_COUNT // 2
                    and report.submissions == T09_TOTAL_COUNT
                    and report.completions == T09_TOTAL_COUNT
                    and report.terminal_tasks == T09_TOTAL_COUNT
                    and report.max_inflight == T09_MAX_INFLIGHT
                    and report.queue_stalls == 0
                    and report.progress_checkpoints == T09_PROGRESS_CHECKPOINTS
                    and report.progress_interval == T09_PROGRESS_INTERVAL
                    and report.payload_64_count == expected_payload_count
                    and report.payload_4k_count == expected_payload_count
                    and report.payload_64k_count == expected_payload_count
                    and report.payload_1m_count == expected_payload_count
                    and report.payload_words_validated > 0
                    and report.credits_acquired == (T09_TOTAL_COUNT if report.input_fences else 0)
                    and report.credits_returned == (T09_TOTAL_COUNT if report.input_fences else 0)
                    and report.input_fences in (0, T09_TOTAL_COUNT)
                    and report.output_fences in (0, T09_TOTAL_COUNT)
                    and report.input_fences != report.output_fences
                    and report.reserved0 == report.reserved1 == report.reserved2 == report.reserved3 == 0
                    for report in reports
                ),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["transport_scope"] = self.transport_scope.value
        result["payload_bytes"] = list(self.payload_bytes)
        result["passed"] = self.passed
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> T09Observation:
        data = dict(value)
        data.pop("passed", None)
        try:
            data["transport_scope"] = TransportScope(data["transport_scope"])
            data["payload_bytes"] = tuple(data["payload_bytes"])
            data["driver_report"] = T09DeviceReport(**data["driver_report"])
            data["service_report"] = T09DeviceReport(**data["service_report"])
            return cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid T09 observation: {exc}") from exc


def _fault_report_passed(report: FaultDeviceReport, *, mode: int, service: bool) -> bool:
    return all(
        (
            report.accepted == 1,
            report.timed_out == 1,
            report.successes == 0,
            report.validation_errors == 0,
            report.sequence_errors == 0,
            report.generation_errors == 0,
            report.timeouts == 1,
            report.elapsed_cycles >= report.configured_timeout_cycles,
            report.submissions == 1,
            report.completions == 0,
            report.service_pause_observed == 1,
            report.generation_unhealthy == 1,
            report.accepting_new_requests == 0,
            report.quarantined_slots == 1,
            report.terminal_errors == 1,
            report.forged_success_completions == 0,
            report.credits_returned == 0,
            report.slot_reuse_after_timeout == 0,
            report.timeout_within_bound == 1,
            report.input_fences == (0 if service else 1),
            report.output_fences == 0,
            report.mode == mode,
            report.reserved == 0,
        )
    )


@dataclass(frozen=True)
class T10Observation:
    case_id: str
    generation: int
    configured_timeout_cycles: int
    backend: str
    transport_scope: TransportScope
    handle_kind: str
    timeout_error: str
    generation_unhealthy: bool
    accepting_new_requests: bool
    quarantined_slots: int
    forged_success_completions: int
    slot_reuse_after_timeout: int
    host_hot_path_messages: int
    host_hot_path_payload_bytes: int
    host_bounce_bytes: int
    fallback_used: bool
    driver_report: FaultDeviceReport
    service_report: FaultDeviceReport
    driver_binary_sha256: str
    service_binary_sha256: str

    @property
    def passed(self) -> bool:
        return all(
            (
                self.case_id == T10_CASE_ID,
                self.generation > 0,
                self.configured_timeout_cycles == T10_TIMEOUT_CYCLES,
                self.timeout_error == "WSE_UNRESPONSIVE_TIMEOUT",
                self.generation_unhealthy,
                not self.accepting_new_requests,
                self.quarantined_slots == 1,
                self.forged_success_completions == 0,
                self.slot_reuse_after_timeout == 0,
                _base_transport_passed(
                    backend=self.backend,
                    transport_scope=self.transport_scope,
                    handle_kind=self.handle_kind,
                    host_hot_path_messages=self.host_hot_path_messages,
                    host_hot_path_payload_bytes=self.host_hot_path_payload_bytes,
                    host_bounce_bytes=self.host_bounce_bytes,
                    fallback_used=self.fallback_used,
                ),
                _fault_report_passed(self.driver_report, mode=10, service=False),
                _fault_report_passed(self.service_report, mode=10, service=True),
                _hashes_valid(self.driver_binary_sha256, self.service_binary_sha256),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["transport_scope"] = self.transport_scope.value
        result["passed"] = self.passed
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> T10Observation:
        data = dict(value)
        data.pop("passed", None)
        try:
            data["transport_scope"] = TransportScope(data["transport_scope"])
            data["driver_report"] = FaultDeviceReport(**data["driver_report"])
            data["service_report"] = FaultDeviceReport(**data["service_report"])
            return cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid T10 observation: {exc}") from exc


@dataclass(frozen=True)
class T11Observation:
    case_id: str
    generation: int
    next_generation: int
    victim_role: EndpointRole
    victim_exit_code: int
    survivor_role: EndpointRole
    survivor_report: FaultDeviceReport
    survivor_cleanup_complete: bool
    survivor_wait_bounded: bool
    old_handle_rejected: bool
    old_handle_probe_result: int
    recovery_success: bool
    recovery_validated_sequences: int
    old_resource_reused: bool
    launcher_reported_expected_failure: bool
    limitation: str

    @property
    def passed(self) -> bool:
        return all(
            (
                self.case_id == T11_CASE_ID,
                self.generation > 0,
                self.next_generation == self.generation + 1,
                self.victim_role is not self.survivor_role,
                self.victim_exit_code < 0,
                _fault_report_passed(
                    self.survivor_report,
                    mode=11,
                    service=self.survivor_role is EndpointRole.WSE_SURROGATE,
                ),
                self.survivor_cleanup_complete,
                self.survivor_wait_bounded,
                self.old_handle_rejected,
                self.old_handle_probe_result != 0,
                self.recovery_success,
                self.recovery_validated_sequences == 100,
                not self.old_resource_reused,
                self.launcher_reported_expected_failure,
                self.limitation == T11_FAULT_LIMITATION,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["victim_role"] = self.victim_role.value
        result["survivor_role"] = self.survivor_role.value
        result["passed"] = self.passed
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> T11Observation:
        data = dict(value)
        data.pop("passed", None)
        try:
            data["victim_role"] = EndpointRole(data["victim_role"])
            data["survivor_role"] = EndpointRole(data["survivor_role"])
            data["survivor_report"] = FaultDeviceReport(**data["survivor_report"])
            return cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid T11 observation: {exc}") from exc


@dataclass(frozen=True)
class T12Observation:
    case_id: str
    generation: int
    backend: str
    transport_scope: TransportScope
    handle_kind: str
    accepted_tasks: int
    terminal_outcomes: int
    credits_acquired: int
    credits_returned: int
    early_window_releases: int
    close_order: tuple[str, ...]
    duplicate_close_attempts: int
    idempotent_close_successes: int
    duplicate_close_rejections: int
    residual_windows: int
    residual_queues: int
    residual_registrations: int
    residual_contexts: int
    host_hot_path_messages: int
    host_hot_path_payload_bytes: int
    host_bounce_bytes: int
    fallback_used: bool
    driver_report: T12DeviceReport
    service_report: T12DeviceReport
    driver_binary_sha256: str
    service_binary_sha256: str

    @property
    def passed(self) -> bool:
        reports = (self.driver_report, self.service_report)
        return all(
            (
                self.case_id == T12_CASE_ID,
                self.generation > 0,
                self.accepted_tasks == T12_ACCEPTED_COUNT,
                self.terminal_outcomes == T12_ACCEPTED_COUNT,
                self.credits_acquired == T12_ACCEPTED_COUNT,
                self.credits_returned == T12_ACCEPTED_COUNT,
                self.early_window_releases == 0,
                self.close_order == T12_CLOSE_ORDER,
                self.duplicate_close_attempts == 8,
                self.idempotent_close_successes == 8,
                self.duplicate_close_rejections == 0,
                self.residual_windows == 0,
                self.residual_queues == 0,
                self.residual_registrations == 0,
                self.residual_contexts == 0,
                _base_transport_passed(
                    backend=self.backend,
                    transport_scope=self.transport_scope,
                    handle_kind=self.handle_kind,
                    host_hot_path_messages=self.host_hot_path_messages,
                    host_hot_path_payload_bytes=self.host_hot_path_payload_bytes,
                    host_bounce_bytes=self.host_bounce_bytes,
                    fallback_used=self.fallback_used,
                ),
                all(
                    report.accepted == T12_ACCEPTED_COUNT
                    and report.processed == T12_ACCEPTED_COUNT
                    and report.validation_errors == 0
                    and report.sequence_errors == 0
                    and report.generation_errors == 0
                    and report.checksum_errors == 0
                    and report.marker_errors == 0
                    and report.timeouts == 0
                    and report.elapsed_cycles > 0
                    and report.submissions == T12_ACCEPTED_COUNT
                    and report.completions == T12_ACCEPTED_COUNT
                    and report.terminal_tasks == T12_ACCEPTED_COUNT
                    and report.max_inflight == T12_MAX_INFLIGHT
                    and report.stop_new_submissions == 1
                    and report.post_stop_attempts == 1
                    and report.post_stop_rejections == 1
                    and report.drain_started == 1
                    and report.drain_completed == 1
                    and report.service_stopped == 1
                    and report.input_fences in (0, T12_ACCEPTED_COUNT)
                    and report.output_fences in (0, T12_ACCEPTED_COUNT)
                    and report.input_fences != report.output_fences
                    and report.reserved == 0
                    for report in reports
                ),
                self.driver_report.credits_acquired == T12_ACCEPTED_COUNT,
                self.driver_report.credits_returned == T12_ACCEPTED_COUNT,
                self.service_report.credits_acquired == 0,
                self.service_report.credits_returned == 0,
                _hashes_valid(self.driver_binary_sha256, self.service_binary_sha256),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["transport_scope"] = self.transport_scope.value
        result["close_order"] = list(self.close_order)
        result["passed"] = self.passed
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> T12Observation:
        data = dict(value)
        data.pop("passed", None)
        try:
            data["transport_scope"] = TransportScope(data["transport_scope"])
            data["close_order"] = tuple(data["close_order"])
            data["driver_report"] = T12DeviceReport(**data["driver_report"])
            data["service_report"] = T12DeviceReport(**data["service_report"])
            return cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid T12 observation: {exc}") from exc
