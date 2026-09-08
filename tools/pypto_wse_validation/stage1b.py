# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Run Stage 1B device-loop cases with two one-shot resident AIV kernels."""

from __future__ import annotations

import argparse
import ctypes
import os
import random
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.pypto_wse_validation.acl_kernel import AclDeviceKernel
from tools.pypto_wse_validation.acl_vmm import AclVmmRuntime
from tools.pypto_wse_validation.bootstrap import ControlChannel
from tools.pypto_wse_validation.collect_stage0 import parse_device_list
from tools.pypto_wse_validation.contracts import EndpointRole, TransportScope
from tools.pypto_wse_validation.stage1a import (
    DEFAULT_GENERATION,
    DEFAULT_START_DELAY_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    PROJECT_ROOT,
    SCHEMA_VERSION,
    _available_port,
    _connect,
    _device_log_evidence,
    _file_evidence,
    _listen,
    _manifest,
    _parse_peer_export,
    _Protocol,
    _read_json,
    _stop_processes,
    _wait_processes,
    _write_json,
)
from tools.pypto_wse_validation.stage1a_contracts import (
    STAGE1A_BACKEND,
    STAGE1A_HANDLE_KIND,
    opaque_handle_evidence,
)
from tools.pypto_wse_validation.stage1b_contracts import (
    T04_CASE_ID,
    T04_CONTROL_OFFSET,
    T04_DEVICE_CONTEXT,
    T04_DRIVER_KERNEL,
    T04_INPUT_FENCE,
    T04_OUTPUT_FENCE,
    T04_PAYLOAD_BYTES,
    T04_SEQUENCE_COUNT,
    T04_SERVICE_KERNEL,
    T04_SLOT,
    T04_WINDOW_BYTES,
    T05_CASE_ID,
    T05_CONTROL_OFFSET,
    T05_DEVICE_CONTEXT,
    T05_DRIVER_KERNEL,
    T05_INPUT_FENCE,
    T05_MAX_INFLIGHT,
    T05_OUTPUT_FENCE,
    T05_PAYLOAD_BYTES,
    T05_SEQUENCE_COUNT,
    T05_SERVICE_KERNEL,
    T05_SLOT_COUNT,
    T05_WINDOW_BYTES,
    T06_BACKPRESSURE_WAIT,
    T06_CASE_ID,
    T06_CONTROL_OFFSET,
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
    T06_WINDOW_BYTES,
    DeviceLoopReport,
    T04Observation,
    T05DeviceLoopReport,
    T05Observation,
    T06DeviceLoopReport,
    T06Observation,
)

T04_DRIVER_BINARY = "stage1b_t04_driver.o"
T04_SERVICE_BINARY = "stage1b_t04_service.o"
T05_DRIVER_BINARY = "stage1b_t05_driver.o"
T05_SERVICE_BINARY = "stage1b_t05_service.o"
T06_DRIVER_BINARY = "stage1b_t06_driver.o"
T06_SERVICE_BINARY = "stage1b_t06_service.o"


@dataclass(frozen=True)
class _CaseSpec:
    case_id: str
    control_offset: int
    window_bytes: int
    driver_binary: str
    service_binary: str
    driver_kernel: str
    service_kernel: str
    report_offset: int
    report_bytes: int


def _case_spec(case_id: str) -> _CaseSpec:
    if case_id == T04_CASE_ID:
        return _CaseSpec(
            case_id=T04_CASE_ID,
            control_offset=T04_CONTROL_OFFSET,
            window_bytes=T04_WINDOW_BYTES,
            driver_binary=T04_DRIVER_BINARY,
            service_binary=T04_SERVICE_BINARY,
            driver_kernel=T04_DRIVER_KERNEL,
            service_kernel=T04_SERVICE_KERNEL,
            report_offset=2 * 64,
            report_bytes=DeviceLoopReport._STRUCT.size,
        )
    if case_id == T05_CASE_ID:
        return _CaseSpec(
            case_id=T05_CASE_ID,
            control_offset=T05_CONTROL_OFFSET,
            window_bytes=T05_WINDOW_BYTES,
            driver_binary=T05_DRIVER_BINARY,
            service_binary=T05_SERVICE_BINARY,
            driver_kernel=T05_DRIVER_KERNEL,
            service_kernel=T05_SERVICE_KERNEL,
            report_offset=T05_SLOT_COUNT * 2 * 64,
            report_bytes=T05DeviceLoopReport._STRUCT.size,
        )
    if case_id == T06_CASE_ID:
        return _CaseSpec(
            case_id=T06_CASE_ID,
            control_offset=T06_CONTROL_OFFSET,
            window_bytes=T06_WINDOW_BYTES,
            driver_binary=T06_DRIVER_BINARY,
            service_binary=T06_SERVICE_BINARY,
            driver_kernel=T06_DRIVER_KERNEL,
            service_kernel=T06_SERVICE_KERNEL,
            report_offset=(T06_SLOT_COUNT * 2 * 64) + 64,
            report_bytes=T06DeviceLoopReport._STRUCT.size,
        )
    raise ValueError(f"unsupported Stage 1B case: {case_id}")


class _T04KernelArguments(ctypes.Structure):
    _fields_ = [
        ("local_payload", ctypes.c_uint64),
        ("local_control", ctypes.c_uint64),
        ("remote_payload", ctypes.c_uint64),
        ("remote_control", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("payload_words", ctypes.c_uint32),
        ("sequence_count", ctypes.c_uint32),
    ]


class _T05KernelArguments(ctypes.Structure):
    _fields_ = [
        ("local_payload", ctypes.c_uint64),
        ("local_control", ctypes.c_uint64),
        ("remote_payload", ctypes.c_uint64),
        ("remote_control", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("slot0_words", ctypes.c_uint32),
        ("slot1_words", ctypes.c_uint32),
        ("sequence_count", ctypes.c_uint32),
    ]


def _kernel_arguments(role: EndpointRole, local_address: int, peer_address: int, generation: int):
    del role  # Both role-specific binaries consume the same address ordering.
    return _T04KernelArguments(
        local_payload=local_address,
        local_control=local_address + T04_CONTROL_OFFSET,
        remote_payload=peer_address,
        remote_control=peer_address + T04_CONTROL_OFFSET,
        generation=generation,
        payload_words=T04_PAYLOAD_BYTES // ctypes.sizeof(ctypes.c_uint64),
        sequence_count=T04_SEQUENCE_COUNT,
    )


def _t05_kernel_arguments(role: EndpointRole, local_address: int, peer_address: int, generation: int):
    del role  # Both role-specific binaries consume the same address ordering.
    return _T05KernelArguments(
        local_payload=local_address,
        local_control=local_address + T05_CONTROL_OFFSET,
        remote_payload=peer_address,
        remote_control=peer_address + T05_CONTROL_OFFSET,
        generation=generation,
        slot0_words=T05_PAYLOAD_BYTES[0] // ctypes.sizeof(ctypes.c_uint64),
        slot1_words=T05_PAYLOAD_BYTES[1] // ctypes.sizeof(ctypes.c_uint64),
        sequence_count=T05_SEQUENCE_COUNT,
    )


def _t06_kernel_arguments(role: EndpointRole, local_address: int, peer_address: int, generation: int):
    del role  # Both role-specific binaries consume the same address ordering.
    return _T04KernelArguments(
        local_payload=local_address,
        local_control=local_address + T06_CONTROL_OFFSET,
        remote_payload=peer_address,
        remote_control=peer_address + T06_CONTROL_OFFSET,
        generation=generation,
        payload_words=T06_PAYLOAD_BYTES // ctypes.sizeof(ctypes.c_uint64),
        sequence_count=T06_SEQUENCE_COUNT,
    )


def _kernel_arguments_for_case(case_id: str):
    if case_id == T04_CASE_ID:
        return _kernel_arguments
    if case_id == T05_CASE_ID:
        return _t05_kernel_arguments
    return _t06_kernel_arguments


def _report_type_for_case(case_id: str):
    if case_id == T04_CASE_ID:
        return DeviceLoopReport
    if case_id == T05_CASE_ID:
        return T05DeviceLoopReport
    return T06DeviceLoopReport


def _counter_delta(after: Mapping[str, Any], before: Mapping[str, Any], key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def _message_count(evidence: Mapping[str, Any]) -> int:
    return sum(int(value) for value in evidence.get("sent_messages", {}).values())


def run_endpoint(args: argparse.Namespace) -> int:
    role = EndpointRole(args.role)
    case = _case_spec(args.case_id)
    artifact_path = args.artifact_dir / f"{role.value.lower()}_stage1b.json"
    started_at = datetime.now(UTC).isoformat()
    runtime: AclVmmRuntime | None = None
    local_window: Any | None = None
    peer_window: Any | None = None
    kernel: AclDeviceKernel | None = None
    channel: ControlChannel | None = None
    report: DeviceLoopReport | T05DeviceLoopReport | T06DeviceLoopReport | None = None
    kernel_elapsed_ns = 0
    hot_path: dict[str, int] = {
        "control_bytes": 0,
        "control_messages": 0,
        "task_messages": 0,
        "completion_messages": 0,
        "payload_bytes": 0,
    }
    cleanup = {
        "device_kernel": "NOT_LOADED",
        "imported_window": "NOT_CREATED",
        "owned_window": "NOT_CREATED",
        "runtime": "NOT_INITIALIZED",
    }
    error: dict[str, str] | None = None
    sanitized_manifest: dict[str, Any] | None = None
    deadline = time.monotonic() + args.timeout
    try:
        runtime = AclVmmRuntime(args.device_id, access_device_id=args.access_device_id)
        runtime.initialize()
        cleanup["runtime"] = "OPEN"
        local_window = runtime.allocate_window(case.window_bytes)
        cleanup["owned_window"] = "OPEN"
        runtime.copy_host_to_device(local_window.address, bytes(case.window_bytes))
        local_manifest = _manifest(runtime, local_window)
        sanitized_manifest = dict(local_manifest)
        sanitized_manifest.pop("shareable_handle")
        sanitized_manifest["opaque_handle"] = opaque_handle_evidence(local_window.shareable_handle)

        connection = (
            _connect(args.host, args.port, deadline)
            if role is EndpointRole.ATTENTION
            else _listen(args.host, args.port, deadline)
        )
        with connection:
            connection.settimeout(max(0.001, deadline - time.monotonic()))
            channel = ControlChannel(connection)
            protocol = _Protocol(channel, role=role, run_id=args.run_id, generation=args.generation)
            peer_message = protocol.exchange("MANIFEST", {"window": local_manifest})
            peer_export, peer_device_id = _parse_peer_export(peer_message)
            if peer_device_id == args.device_id:
                raise ValueError("peer must use a distinct device")
            peer_window = runtime.import_window(peer_export, peer_device_id=peer_device_id)
            cleanup["imported_window"] = "OPEN"
            protocol.exchange("ATTACHED", {"peer_mapping_bytes": peer_window.mapping_bytes})

            binary_name = case.driver_binary if role is EndpointRole.ATTENTION else case.service_binary
            kernel = AclDeviceKernel(runtime, args.kernel_dir / binary_name)
            cleanup["device_kernel"] = "OPEN"
            kernel_arguments = _kernel_arguments_for_case(case.case_id)
            kernel.launch(kernel_arguments(role, local_window.address, peer_window.address, args.generation))
            protocol.exchange("READY", {"case_id": case.case_id, "device_kernel_launched": True})
            hot_path_start = channel.evidence()
            kernel_elapsed_ns = kernel.synchronize()
            hot_path_end = channel.evidence()
            hot_path["control_bytes"] = _counter_delta(hot_path_end, hot_path_start, "sent_bytes")
            hot_path["control_messages"] = _message_count(hot_path_end) - _message_count(hot_path_start)
            raw_report = runtime.copy_device_to_host(
                local_window.address + case.control_offset + case.report_offset,
                case.report_bytes,
            )
            report_type = _report_type_for_case(case.case_id)
            report = report_type.from_bytes(raw_report)
            kernel.close()
            cleanup["device_kernel"] = "CLOSED"

            protocol.exchange("DRAIN", {"device_processed": report.processed})
            peer_window.close()
            cleanup["imported_window"] = "CLOSED"
            protocol.exchange("DETACHED", {"imported_window": "CLOSED"})
            local_window.close()
            cleanup["owned_window"] = "CLOSED"
            protocol.exchange("RELEASED", {"owned_window": "CLOSED"})
    except BaseException as exc:  # noqa: BLE001
        error = {"message": str(exc), "type": type(exc).__name__}
        traceback.print_exc()
    finally:
        if kernel is not None and not kernel.closed:
            try:
                kernel.close()
                cleanup["device_kernel"] = "CLOSED"
            except BaseException as exc:  # noqa: BLE001
                cleanup["device_kernel"] = f"ERROR:{type(exc).__name__}:{exc}"
                error = error or {"message": str(exc), "type": type(exc).__name__}
        if peer_window is not None and not peer_window.closed:
            try:
                peer_window.close()
                cleanup["imported_window"] = "CLOSED"
            except BaseException as exc:  # noqa: BLE001
                cleanup["imported_window"] = f"ERROR:{type(exc).__name__}:{exc}"
                error = error or {"message": str(exc), "type": type(exc).__name__}
        if local_window is not None and not local_window.closed:
            try:
                local_window.close()
                cleanup["owned_window"] = "CLOSED"
            except BaseException as exc:  # noqa: BLE001
                cleanup["owned_window"] = f"ERROR:{type(exc).__name__}:{exc}"
                error = error or {"message": str(exc), "type": type(exc).__name__}
        if runtime is not None:
            try:
                runtime.close()
                cleanup["runtime"] = "CLOSED"
            except BaseException as exc:  # noqa: BLE001
                cleanup["runtime"] = f"ERROR:{type(exc).__name__}:{exc}"
                error = error or {"message": str(exc), "type": type(exc).__name__}
        evidence = {
            "cleanup": cleanup,
            "closed_at": datetime.now(UTC).isoformat(),
            "control": channel.evidence() if channel is not None else {},
            "device_id": args.device_id,
            "device_loop": {
                "binary_sha256": kernel.binary_sha256 if kernel is not None else None,
                "elapsed_ns": kernel_elapsed_ns,
                "hot_path": hot_path,
                "kernel": case.driver_kernel if role is EndpointRole.ATTENTION else case.service_kernel,
                "launches": 1 if kernel is not None and kernel.launched else 0,
                "report": report.to_dict() if report is not None else None,
            },
            "error": error,
            "generation": args.generation,
            "host_bounce_bytes": 0,
            "manifest": sanitized_manifest,
            "pid": os.getpid(),
            "role": role.value,
            "run_id": args.run_id,
            "schema_version": SCHEMA_VERSION,
            "started_at": started_at,
            "success": error is None,
        }
        _write_json(artifact_path, evidence)
    return 0 if error is None else 1


def _endpoint_command(role: EndpointRole, device_id: int, access_device_id: int, args: argparse.Namespace, port: int):
    return [
        sys.executable,
        "-m",
        "tools.pypto_wse_validation.stage1b",
        "endpoint",
        "--role",
        role.value,
        "--device-id",
        str(device_id),
        "--access-device-id",
        str(access_device_id),
        "--host",
        args.host,
        "--port",
        str(port),
        "--run-id",
        args.run_id,
        "--case-id",
        args.case_id,
        "--generation",
        str(args.generation),
        "--timeout",
        str(args.timeout),
        "--artifact-dir",
        str(args.artifact_dir.resolve()),
        "--kernel-dir",
        str(args.kernel_dir.resolve()),
    ]


def _report_from_artifact(
    artifact: Mapping[str, Any], case_id: str
) -> DeviceLoopReport | T05DeviceLoopReport | T06DeviceLoopReport:
    raw = artifact.get("device_loop", {}).get("report")
    if not isinstance(raw, Mapping):
        raise ValueError(f"endpoint is missing a {case_id} device report")
    report_type = _report_type_for_case(case_id)
    return report_type(**raw)


def _observation_from_artifacts(
    args: argparse.Namespace, artifacts: Mapping[EndpointRole, Mapping[str, Any]]
) -> T04Observation | T05Observation | T06Observation:
    case_id = getattr(args, "case_id", T04_CASE_ID)
    attention_loop = artifacts[EndpointRole.ATTENTION]["device_loop"]
    service_loop = artifacts[EndpointRole.WSE_SURROGATE]["device_loop"]
    driver_report = _report_from_artifact(artifacts[EndpointRole.ATTENTION], case_id)
    service_report = _report_from_artifact(artifacts[EndpointRole.WSE_SURROGATE], case_id)
    common = {
        "case_id": case_id,
        "generation": args.generation,
        "backend": STAGE1A_BACKEND,
        "transport_scope": TransportScope.HOST_LOCAL,
        "handle_kind": STAGE1A_HANDLE_KIND,
        "driver_kernel": str(attention_loop["kernel"]),
        "service_kernel": str(service_loop["kernel"]),
        "driver_launches": int(attention_loop["launches"]),
        "service_launches": int(service_loop["launches"]),
        "device_submissions": service_report.processed,
        "device_completions": driver_report.processed,
        "validated_sequences": driver_report.processed,
        "host_hot_path_control_messages": sum(
            int(artifacts[role]["device_loop"]["hot_path"]["control_messages"]) for role in EndpointRole
        ),
        "host_hot_path_task_messages": sum(
            int(artifacts[role]["device_loop"]["hot_path"]["task_messages"]) for role in EndpointRole
        ),
        "host_hot_path_completion_messages": sum(
            int(artifacts[role]["device_loop"]["hot_path"]["completion_messages"]) for role in EndpointRole
        ),
        "host_hot_path_payload_bytes": sum(
            int(artifacts[role]["device_loop"]["hot_path"]["payload_bytes"]) for role in EndpointRole
        ),
        "host_bounce_bytes": sum(int(artifacts[role].get("host_bounce_bytes", -1)) for role in EndpointRole),
        "fallback_used": False,
        "driver_report": driver_report,
        "service_report": service_report,
        "driver_binary_sha256": str(attention_loop["binary_sha256"]),
        "service_binary_sha256": str(service_loop["binary_sha256"]),
    }
    if case_id == T04_CASE_ID:
        return T04Observation(
            slot=T04_SLOT,
            sequence_count=T04_SEQUENCE_COUNT,
            payload_bytes=T04_PAYLOAD_BYTES,
            driver_context=T04_DEVICE_CONTEXT,
            service_context=T04_DEVICE_CONTEXT,
            input_fence=T04_INPUT_FENCE,
            output_fence=T04_OUTPUT_FENCE,
            **common,
        )
    if case_id == T05_CASE_ID:
        if not isinstance(driver_report, T05DeviceLoopReport) or not isinstance(service_report, T05DeviceLoopReport):
            raise ValueError("T05 artifacts contain the wrong report type")
        return T05Observation(
            slot_count=T05_SLOT_COUNT,
            max_inflight=T05_MAX_INFLIGHT,
            sequence_count=T05_SEQUENCE_COUNT,
            payload_bytes=T05_PAYLOAD_BYTES,
            driver_context=T05_DEVICE_CONTEXT,
            service_context=T05_DEVICE_CONTEXT,
            credits_acquired=driver_report.credits_acquired,
            terminal_tasks=driver_report.terminal_tasks,
            credits_returned=driver_report.credits_returned,
            out_of_order_completions=driver_report.out_of_order_completions,
            input_fence=T05_INPUT_FENCE,
            output_fence=T05_OUTPUT_FENCE,
            **common,
        )
    if not isinstance(driver_report, T06DeviceLoopReport) or not isinstance(service_report, T06DeviceLoopReport):
        raise ValueError("T06 artifacts contain the wrong report type")
    return T06Observation(
        slot_count=T06_SLOT_COUNT,
        max_inflight=T06_MAX_INFLIGHT,
        sequence_count=T06_SEQUENCE_COUNT,
        payload_bytes=T06_PAYLOAD_BYTES,
        attempted_submissions=driver_report.submissions + driver_report.no_credit_events,
        no_credit_events=driver_report.no_credit_events,
        pending_requests=driver_report.pending_requests,
        terminal_tasks=driver_report.terminal_tasks,
        credits_acquired=driver_report.credits_acquired,
        credits_returned=driver_report.credits_returned,
        submission_retry_spins=driver_report.submission_retry_spins,
        service_pause_observed=bool(driver_report.service_pause_observed and service_report.service_pause_observed),
        service_resume_observed=bool(driver_report.service_resume_observed and service_report.service_resume_observed),
        progress_after_resume=min(driver_report.progress_after_resume, service_report.progress_after_resume),
        third_request_outcome=T06_THIRD_REQUEST_OUTCOME,
        backpressure_wait=T06_BACKPRESSURE_WAIT,
        driver_context=T06_DEVICE_CONTEXT,
        service_context=T06_DEVICE_CONTEXT,
        input_fence=T06_INPUT_FENCE,
        output_fence=T06_OUTPUT_FENCE,
        **common,
    )


def _aggregate_result(args: argparse.Namespace, processes: Mapping[EndpointRole, subprocess.Popen[Any]]):
    case_id = getattr(args, "case_id", T04_CASE_ID)
    endpoints: dict[str, Any] = {}
    artifacts: dict[EndpointRole, dict[str, Any]] = {}
    all_success = True
    all_cleanup = True
    driver_proofs: list[bool] = []
    for role, process in processes.items():
        artifact_name = f"{role.value.lower()}_stage1b.json"
        artifact = _read_json(args.artifact_dir / artifact_name) or {"success": False}
        artifacts[role] = artifact
        endpoint_success = bool(artifact.get("success")) and process.returncode == 0
        all_success = all_success and endpoint_success
        cleanup = artifact.get("cleanup", {})
        cleanup_complete = cleanup == {
            "device_kernel": "CLOSED",
            "imported_window": "CLOSED",
            "owned_window": "CLOSED",
            "runtime": "CLOSED",
        }
        all_cleanup = all_cleanup and cleanup_complete
        device_logs = _device_log_evidence(args.artifact_dir / f"{role.value.lower()}_device_logs")
        driver_proofs.append(bool(device_logs["p2p_enable_observed"] and device_logs["p2p_memory_released"]))
        endpoints[role.value] = {
            "artifact": artifact_name,
            "cleanup": cleanup,
            "device_logs": device_logs,
            "exit_code": process.returncode,
            "host_log": _file_evidence(args.artifact_dir / f"{role.value.lower()}_stage1b.log"),
            "pid": process.pid,
            "success": endpoint_success,
        }

    observation: T04Observation | T05Observation | T06Observation | None = None
    with suppress(KeyError, TypeError, ValueError):
        observation = _observation_from_artifacts(args, artifacts)

    case_passed = bool(
        all_success and all_cleanup and all(driver_proofs) and observation is not None and observation.passed
    )
    if case_id == T04_CASE_ID:
        sequence_count = T04_SEQUENCE_COUNT
        transferred_bytes = T04_SEQUENCE_COUNT * T04_PAYLOAD_BYTES * 2
    elif case_id == T05_CASE_ID:
        sequence_count = T05_SEQUENCE_COUNT
        transferred_bytes = (T05_SEQUENCE_COUNT // T05_SLOT_COUNT) * sum(T05_PAYLOAD_BYTES) * 2
    else:
        sequence_count = T06_SEQUENCE_COUNT
        transferred_bytes = T06_SEQUENCE_COUNT * T06_PAYLOAD_BYTES * 2
    attempted_submissions = (
        observation.attempted_submissions if isinstance(observation, T06Observation) else sequence_count
    )
    return {
        "actual_backend": STAGE1A_BACKEND,
        "capability_level": "C1",
        "claim_scope": "NPU_SURROGATE_ONLY",
        "completed_at": datetime.now(UTC).isoformat(),
        "c2_status": "NOT_ESTABLISHED",
        "data_results": {
            case_id: {
                "attempts": attempted_submissions,
                "bytes": transferred_bytes,
                "passed": observation.validated_sequences if observation is not None else 0,
                "status": "PASS" if case_passed else "FAIL",
            }
        },
        "devices": {"ATTENTION": args.devices[0], "WSE_SURROGATE": args.devices[1]},
        "endpoints": endpoints,
        "evidence_status": "SIMULATION",
        "fallback_used": False,
        "generation": args.generation,
        "host_bounce_bytes": observation.host_bounce_bytes if observation is not None else None,
        "host_hot_path": {
            "completion_messages": observation.host_hot_path_completion_messages if observation is not None else None,
            "control_messages": observation.host_hot_path_control_messages if observation is not None else None,
            "payload_bytes": observation.host_hot_path_payload_bytes if observation is not None else None,
            "task_messages": observation.host_hot_path_task_messages if observation is not None else None,
        },
        "npu_wse_capability_level": "NOT_ESTABLISHED",
        "observation": observation.to_dict() if observation is not None else None,
        "profile": "NPU_SURROGATE",
        "resource_cleanup": "VERIFIED" if all_cleanup else "FAILED",
        "run_id": args.run_id,
        "schema_version": SCHEMA_VERSION,
        "stage1b_progress": f"{case_id}_{'PASS' if case_passed else 'FAIL'}",
        "start_order": args.start_order,
        "success": case_passed,
        "transport_scope": TransportScope.HOST_LOCAL.value,
    }


def run_launcher(args: argparse.Namespace) -> int:
    if args.devices[0] == args.devices[1]:
        raise ValueError("Stage 1B requires two distinct devices")
    case = _case_spec(args.case_id)
    for binary_name in (case.driver_binary, case.service_binary):
        if not (args.kernel_dir / binary_name).is_file():
            raise ValueError(f"missing Stage 1B kernel binary: {args.kernel_dir / binary_name}")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    port = _available_port(args.host)
    roles = [EndpointRole.ATTENTION, EndpointRole.WSE_SURROGATE]
    if args.start_order == "wse-first":
        roles.reverse()
    elif args.start_order == "random":
        random.SystemRandom().shuffle(roles)
        args.start_order = "attention-first" if roles[0] is EndpointRole.ATTENTION else "wse-first"
    devices = {EndpointRole.ATTENTION: args.devices[0], EndpointRole.WSE_SURROGATE: args.devices[1]}
    access_devices = {
        EndpointRole.ATTENTION: args.access_device_ids[0],
        EndpointRole.WSE_SURROGATE: args.access_device_ids[1],
    }
    processes: dict[EndpointRole, subprocess.Popen[Any]] = {}
    logs: list[Any] = []
    deadline = time.monotonic() + args.timeout
    try:
        for index, role in enumerate(roles):
            log = (args.artifact_dir / f"{role.value.lower()}_stage1b.log").open("w", encoding="utf-8")
            logs.append(log)
            endpoint_env = os.environ.copy()
            inherited_python_path = endpoint_env.get("PYTHONPATH")
            endpoint_env["PYTHONPATH"] = (
                f"{PROJECT_ROOT}{os.pathsep}{inherited_python_path}" if inherited_python_path else str(PROJECT_ROOT)
            )
            device_log_dir = args.artifact_dir / f"{role.value.lower()}_device_logs"
            device_log_dir.mkdir(exist_ok=True)
            endpoint_env["ASCEND_PROCESS_LOG_PATH"] = str(device_log_dir.resolve())
            process = subprocess.Popen(  # noqa: S603
                _endpoint_command(role, devices[role], access_devices[role], args, port),
                cwd=args.artifact_dir,
                env=endpoint_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes[role] = process
            if index == 0:
                time.sleep(args.start_delay)
        if not _wait_processes(processes, deadline):
            _stop_processes(processes)
        else:
            for process in processes.values():
                process.wait()
    finally:
        _stop_processes(processes)
        for log in logs:
            log.close()
    result = _aggregate_result(args, processes)
    _write_json(args.artifact_dir / "result.json", result)
    return 0 if result["success"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--devices", type=parse_device_list, required=True)
    run.add_argument("--access-device-ids", type=parse_device_list)
    run.add_argument("--artifact-dir", type=Path, required=True)
    run.add_argument("--kernel-dir", type=Path, required=True)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--run-id", default=f"stage1b-{uuid.uuid4().hex}")
    run.add_argument("--case-id", choices=(T04_CASE_ID, T05_CASE_ID, T06_CASE_ID), default=T04_CASE_ID)
    run.add_argument("--generation", type=int, default=DEFAULT_GENERATION)
    run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    run.add_argument("--start-delay", type=float, default=DEFAULT_START_DELAY_SECONDS)
    run.add_argument("--start-order", choices=("attention-first", "wse-first", "random"), default="random")

    endpoint = subparsers.add_parser("endpoint")
    endpoint.add_argument("--role", choices=tuple(role.value for role in EndpointRole), required=True)
    endpoint.add_argument("--device-id", type=int, required=True)
    endpoint.add_argument("--access-device-id", type=int, required=True)
    endpoint.add_argument("--artifact-dir", type=Path, required=True)
    endpoint.add_argument("--kernel-dir", type=Path, required=True)
    endpoint.add_argument("--host", required=True)
    endpoint.add_argument("--port", type=int, required=True)
    endpoint.add_argument("--run-id", required=True)
    endpoint.add_argument("--case-id", choices=(T04_CASE_ID, T05_CASE_ID, T06_CASE_ID), required=True)
    endpoint.add_argument("--generation", type=int, required=True)
    endpoint.add_argument("--timeout", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "endpoint":
        return run_endpoint(args)
    if args.access_device_ids is None:
        args.access_device_ids = args.devices
    if len(args.devices) != 2 or len(args.access_device_ids) != 2:
        raise ValueError("--devices and --access-device-ids must each contain exactly two IDs")
    return run_launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
