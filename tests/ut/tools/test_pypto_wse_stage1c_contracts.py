# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

from dataclasses import replace

from tools.pypto_wse_validation.contracts import EndpointRole, TransportScope
from tools.pypto_wse_validation.stage1a_contracts import STAGE1A_BACKEND, STAGE1A_HANDLE_KIND
from tools.pypto_wse_validation.stage1c_contracts import (
    T09_MEASURED_COUNT,
    T09_PAYLOAD_BYTES,
    T09_PROGRESS_CHECKPOINTS,
    T09_PROGRESS_INTERVAL,
    T09_TOTAL_COUNT,
    T09_WARMUP_COUNT,
    T10_TIMEOUT_CYCLES,
    T11_FAULT_LIMITATION,
    T11_TIMEOUT_CYCLES,
    T12_ACCEPTED_COUNT,
    T12_CLOSE_ORDER,
    FaultDeviceReport,
    T09DeviceReport,
    T09Observation,
    T10Observation,
    T11Observation,
    T12DeviceReport,
    T12Observation,
)


def _t09_report(driver: bool) -> T09DeviceReport:
    return T09DeviceReport(
        processed=T09_TOTAL_COUNT,
        warmup_processed=T09_WARMUP_COUNT,
        measured_processed=T09_MEASURED_COUNT,
        validation_errors=0,
        sequence_errors=0,
        generation_errors=0,
        checksum_errors=0,
        marker_errors=0,
        timeouts=0,
        elapsed_cycles=1,
        slot0_processed=T09_TOTAL_COUNT // 2,
        slot1_processed=T09_TOTAL_COUNT // 2,
        submissions=T09_TOTAL_COUNT,
        completions=T09_TOTAL_COUNT,
        credits_acquired=T09_TOTAL_COUNT if driver else 0,
        credits_returned=T09_TOTAL_COUNT if driver else 0,
        terminal_tasks=T09_TOTAL_COUNT,
        max_inflight=2,
        queue_stalls=0,
        progress_checkpoints=T09_PROGRESS_CHECKPOINTS,
        progress_interval=T09_PROGRESS_INTERVAL,
        payload_64_count=2500,
        payload_4k_count=2500,
        payload_64k_count=2500,
        payload_1m_count=2500,
        payload_words_validated=1,
        input_fences=T09_TOTAL_COUNT if driver else 0,
        output_fences=0 if driver else T09_TOTAL_COUNT,
        reserved0=0,
        reserved1=0,
        reserved2=0,
        reserved3=0,
    )


def _fault_report(*, service: bool, mode: int) -> FaultDeviceReport:
    timeout = T10_TIMEOUT_CYCLES if mode == 10 else T11_TIMEOUT_CYCLES
    return FaultDeviceReport(
        accepted=1,
        timed_out=1,
        successes=0,
        validation_errors=0,
        sequence_errors=0,
        generation_errors=0,
        timeouts=1,
        elapsed_cycles=timeout + 1,
        submissions=1,
        completions=0,
        service_pause_observed=1,
        generation_unhealthy=1,
        accepting_new_requests=0,
        quarantined_slots=1,
        terminal_errors=1,
        forged_success_completions=0,
        credits_returned=0,
        slot_reuse_after_timeout=0,
        configured_timeout_cycles=timeout,
        timeout_within_bound=1,
        input_fences=0 if service else 1,
        output_fences=0,
        mode=mode,
        reserved=0,
    )


def _t12_report(driver: bool) -> T12DeviceReport:
    return T12DeviceReport(
        accepted=T12_ACCEPTED_COUNT,
        processed=T12_ACCEPTED_COUNT,
        validation_errors=0,
        sequence_errors=0,
        generation_errors=0,
        checksum_errors=0,
        marker_errors=0,
        timeouts=0,
        elapsed_cycles=1,
        submissions=T12_ACCEPTED_COUNT,
        completions=T12_ACCEPTED_COUNT,
        credits_acquired=T12_ACCEPTED_COUNT if driver else 0,
        credits_returned=T12_ACCEPTED_COUNT if driver else 0,
        terminal_tasks=T12_ACCEPTED_COUNT,
        max_inflight=2,
        stop_new_submissions=1,
        post_stop_attempts=1,
        post_stop_rejections=1,
        drain_started=1,
        drain_completed=1,
        service_stopped=1,
        input_fences=T12_ACCEPTED_COUNT if driver else 0,
        output_fences=0 if driver else T12_ACCEPTED_COUNT,
        reserved=0,
    )


def test_t09_observation_passes_round_trip_and_fails_closed():
    observation = T09Observation(
        case_id="T09",
        generation=1,
        backend=STAGE1A_BACKEND,
        transport_scope=TransportScope.HOST_LOCAL,
        handle_kind=STAGE1A_HANDLE_KIND,
        warmup_count=T09_WARMUP_COUNT,
        measured_count=T09_MEASURED_COUNT,
        payload_bytes=T09_PAYLOAD_BYTES,
        slot_count=2,
        max_inflight=2,
        progress_interval=T09_PROGRESS_INTERVAL,
        progress_checkpoints=T09_PROGRESS_CHECKPOINTS,
        device_submissions=T09_TOTAL_COUNT,
        device_completions=T09_TOTAL_COUNT,
        validated_sequences=T09_TOTAL_COUNT,
        device_memory_growth_bytes=0,
        registration_growth=0,
        host_memory_growth_bytes=0,
        host_memory_growth_limit_bytes=64 * 1024 * 1024,
        host_cpu_utilization_pct=1.0,
        queue_stalls=0,
        host_hot_path_messages=0,
        host_hot_path_payload_bytes=0,
        host_bounce_bytes=0,
        fallback_used=False,
        driver_report=_t09_report(True),
        service_report=_t09_report(False),
        driver_binary_sha256="a" * 64,
        service_binary_sha256="b" * 64,
    )
    assert observation.passed
    assert T09Observation.from_dict(observation.to_dict()).passed
    assert not replace(observation, queue_stalls=1).passed
    assert not replace(observation, host_memory_growth_bytes=65 * 1024 * 1024).passed


def test_t10_observation_requires_explicit_timeout_without_slot_release():
    observation = T10Observation(
        case_id="T10",
        generation=1,
        configured_timeout_cycles=T10_TIMEOUT_CYCLES,
        backend=STAGE1A_BACKEND,
        transport_scope=TransportScope.HOST_LOCAL,
        handle_kind=STAGE1A_HANDLE_KIND,
        timeout_error="WSE_UNRESPONSIVE_TIMEOUT",
        generation_unhealthy=True,
        accepting_new_requests=False,
        quarantined_slots=1,
        forged_success_completions=0,
        slot_reuse_after_timeout=0,
        host_hot_path_messages=0,
        host_hot_path_payload_bytes=0,
        host_bounce_bytes=0,
        fallback_used=False,
        driver_report=_fault_report(service=False, mode=10),
        service_report=_fault_report(service=True, mode=10),
        driver_binary_sha256="a" * 64,
        service_binary_sha256="b" * 64,
    )
    assert observation.passed
    assert T10Observation.from_dict(observation.to_dict()).passed
    assert not replace(observation, accepting_new_requests=True).passed
    assert not replace(observation, forged_success_completions=1).passed


def test_t11_observation_requires_sigkill_cleanup_old_handle_rejection_and_recovery():
    observation = T11Observation(
        case_id="T11",
        generation=1,
        next_generation=2,
        victim_role=EndpointRole.WSE_SURROGATE,
        victim_exit_code=-9,
        survivor_role=EndpointRole.ATTENTION,
        survivor_report=_fault_report(service=False, mode=11),
        survivor_cleanup_complete=True,
        survivor_wait_bounded=True,
        old_handle_rejected=True,
        old_handle_probe_result=507899,
        recovery_success=True,
        recovery_validated_sequences=100,
        old_resource_reused=False,
        launcher_reported_expected_failure=True,
        limitation=T11_FAULT_LIMITATION,
    )
    assert observation.passed
    assert T11Observation.from_dict(observation.to_dict()).passed
    assert not replace(observation, victim_exit_code=0).passed
    assert not replace(observation, recovery_success=False).passed


def test_t12_observation_requires_ordered_idempotent_cleanup():
    observation = T12Observation(
        case_id="T12",
        generation=1,
        backend=STAGE1A_BACKEND,
        transport_scope=TransportScope.HOST_LOCAL,
        handle_kind=STAGE1A_HANDLE_KIND,
        accepted_tasks=T12_ACCEPTED_COUNT,
        terminal_outcomes=T12_ACCEPTED_COUNT,
        credits_acquired=T12_ACCEPTED_COUNT,
        credits_returned=T12_ACCEPTED_COUNT,
        early_window_releases=0,
        close_order=T12_CLOSE_ORDER,
        duplicate_close_attempts=8,
        idempotent_close_successes=8,
        duplicate_close_rejections=0,
        residual_windows=0,
        residual_queues=0,
        residual_registrations=0,
        residual_contexts=0,
        host_hot_path_messages=0,
        host_hot_path_payload_bytes=0,
        host_bounce_bytes=0,
        fallback_used=False,
        driver_report=_t12_report(True),
        service_report=_t12_report(False),
        driver_binary_sha256="a" * 64,
        service_binary_sha256="b" * 64,
    )
    assert observation.passed
    assert T12Observation.from_dict(observation.to_dict()).passed
    assert not replace(observation, close_order=tuple(reversed(T12_CLOSE_ORDER))).passed
    assert not replace(observation, residual_contexts=1).passed
