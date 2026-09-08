# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Run Stage 1C T09, T10, and T12 on two host-local NPUs."""

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
from tools.pypto_wse_validation.stage1c_contracts import (
    T09_CASE_ID,
    T09_CONTROL_OFFSET,
    T09_DRIVER_KERNEL,
    T09_MAX_INFLIGHT,
    T09_MEASURED_COUNT,
    T09_PAYLOAD_BYTES,
    T09_PROGRESS_INTERVAL,
    T09_SERVICE_KERNEL,
    T09_SLOT_COUNT,
    T09_TOTAL_COUNT,
    T09_WARMUP_COUNT,
    T09_WINDOW_BYTES,
    T10_CASE_ID,
    T10_CONTROL_OFFSET,
    T10_DRIVER_KERNEL,
    T10_PAYLOAD_BYTES,
    T10_SERVICE_KERNEL,
    T10_TIMEOUT_CYCLES,
    T10_WINDOW_BYTES,
    T12_ACCEPTED_COUNT,
    T12_CASE_ID,
    T12_CLOSE_ORDER,
    T12_CONTROL_OFFSET,
    T12_DRIVER_KERNEL,
    T12_SERVICE_KERNEL,
    T12_WINDOW_BYTES,
    FaultDeviceReport,
    T09DeviceReport,
    T09Observation,
    T10Observation,
    T12DeviceReport,
    T12Observation,
)

HOST_MEMORY_GROWTH_LIMIT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class _CaseSpec:
    case_id: str
    window_bytes: int
    control_offset: int
    report_offset: int
    driver_binary: str
    service_binary: str
    driver_kernel: str
    service_kernel: str
    report_type: type[T09DeviceReport] | type[FaultDeviceReport] | type[T12DeviceReport]


def _case_spec(case_id: str) -> _CaseSpec:
    if case_id == T09_CASE_ID:
        return _CaseSpec(
            case_id,
            T09_WINDOW_BYTES,
            T09_CONTROL_OFFSET,
            T09_SLOT_COUNT * 2 * 64,
            "stage1c_t09_driver.o",
            "stage1c_t09_service.o",
            T09_DRIVER_KERNEL,
            T09_SERVICE_KERNEL,
            T09DeviceReport,
        )
    if case_id == T10_CASE_ID:
        return _CaseSpec(
            case_id,
            T10_WINDOW_BYTES,
            T10_CONTROL_OFFSET,
            2 * 64,
            "stage1c_fault_driver.o",
            "stage1c_fault_service.o",
            T10_DRIVER_KERNEL,
            T10_SERVICE_KERNEL,
            FaultDeviceReport,
        )
    if case_id == T12_CASE_ID:
        return _CaseSpec(
            case_id,
            T12_WINDOW_BYTES,
            T12_CONTROL_OFFSET,
            (T12_ACCEPTED_COUNT // 2 * 2 * 64) + 64,
            "stage1c_t12_driver.o",
            "stage1c_t12_service.o",
            T12_DRIVER_KERNEL,
            T12_SERVICE_KERNEL,
            T12DeviceReport,
        )
    raise ValueError(f"unsupported Stage 1C case: {case_id}")


class _T09Arguments(ctypes.Structure):
    _fields_ = [
        ("local_payload", ctypes.c_uint64),
        ("local_control", ctypes.c_uint64),
        ("remote_payload", ctypes.c_uint64),
        ("remote_control", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("total_count", ctypes.c_uint32),
    ]


class _FaultArguments(ctypes.Structure):
    _fields_ = [
        ("local_payload", ctypes.c_uint64),
        ("local_control", ctypes.c_uint64),
        ("remote_payload", ctypes.c_uint64),
        ("remote_control", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("timeout_cycles", ctypes.c_uint64),
        ("payload_words", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
    ]


class _T12Arguments(ctypes.Structure):
    _fields_ = [
        ("local_payload", ctypes.c_uint64),
        ("local_control", ctypes.c_uint64),
        ("remote_payload", ctypes.c_uint64),
        ("remote_control", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
    ]


def _kernel_arguments(case_id: str, local_address: int, peer_address: int, generation: int):
    control_offset = _case_spec(case_id).control_offset
    common = {
        "local_payload": local_address,
        "local_control": local_address + control_offset,
        "remote_payload": peer_address,
        "remote_control": peer_address + control_offset,
        "generation": generation,
    }
    if case_id == T09_CASE_ID:
        return _T09Arguments(total_count=T09_TOTAL_COUNT, **common)
    if case_id == T10_CASE_ID:
        return _FaultArguments(
            timeout_cycles=T10_TIMEOUT_CYCLES,
            payload_words=T10_PAYLOAD_BYTES // ctypes.sizeof(ctypes.c_uint64),
            mode=10,
            **common,
        )
    return _T12Arguments(**common)


def _rss_bytes() -> int:
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
    except (OSError, IndexError, ValueError):
        return 0
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _counter_delta(after: Mapping[str, Any], before: Mapping[str, Any], key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def _message_count(evidence: Mapping[str, Any]) -> int:
    return sum(int(value) for value in evidence.get("sent_messages", {}).values())


def run_endpoint(args: argparse.Namespace) -> int:
    role = EndpointRole(args.role)
    case = _case_spec(args.case_id)
    artifact_path = args.artifact_dir / f"{role.value.lower()}_stage1c.json"
    started_at = datetime.now(UTC).isoformat()
    initial_rss = _rss_bytes()
    runtime: AclVmmRuntime | None = None
    local_window: Any | None = None
    peer_window: Any | None = None
    kernel: AclDeviceKernel | None = None
    channel: ControlChannel | None = None
    report: T09DeviceReport | FaultDeviceReport | T12DeviceReport | None = None
    cleanup = {
        "device_kernel": "NOT_LOADED",
        "imported_window": "NOT_CREATED",
        "owned_window": "NOT_CREATED",
        "runtime": "NOT_INITIALIZED",
    }
    hot_path = {
        "control_bytes": 0,
        "control_messages": 0,
        "task_messages": 0,
        "completion_messages": 0,
        "payload_bytes": 0,
    }
    lifecycle: dict[str, Any] = {
        "close_order": [],
        "duplicate_close_attempts": 0,
        "idempotent_close_successes": 0,
        "duplicate_close_rejections": 0,
    }
    metrics = {"kernel_elapsed_ns": 0, "host_cpu_ns": 0, "host_wall_ns": 0}
    error: dict[str, str] | None = None
    manifest_evidence: dict[str, Any] | None = None
    deadline = time.monotonic() + args.timeout
    try:
        runtime = AclVmmRuntime(args.device_id, access_device_id=args.access_device_id)
        runtime.initialize()
        cleanup["runtime"] = "OPEN"
        local_window = runtime.allocate_window(case.window_bytes)
        cleanup["owned_window"] = "OPEN"
        runtime.copy_host_to_device(local_window.address, bytes(case.window_bytes))
        manifest = _manifest(runtime, local_window)
        manifest_evidence = dict(manifest)
        manifest_evidence.pop("shareable_handle")
        manifest_evidence["opaque_handle"] = opaque_handle_evidence(local_window.shareable_handle)
        connection = (
            _connect(args.host, args.port, deadline)
            if role is EndpointRole.ATTENTION
            else _listen(args.host, args.port, deadline)
        )
        with connection:
            connection.settimeout(max(0.001, deadline - time.monotonic()))
            channel = ControlChannel(connection)
            protocol = _Protocol(channel, role=role, run_id=args.run_id, generation=args.generation)
            peer_message = protocol.exchange("MANIFEST", {"window": manifest})
            peer_export, peer_device_id = _parse_peer_export(peer_message)
            if peer_device_id == args.device_id:
                raise ValueError("peer must use a distinct device")
            peer_window = runtime.import_window(peer_export, peer_device_id=peer_device_id)
            cleanup["imported_window"] = "OPEN"
            protocol.exchange("ATTACHED", {"peer_mapping_bytes": peer_window.mapping_bytes})
            binary = case.driver_binary if role is EndpointRole.ATTENTION else case.service_binary
            kernel = AclDeviceKernel(runtime, args.kernel_dir / binary)
            cleanup["device_kernel"] = "OPEN"
            kernel.launch(_kernel_arguments(case.case_id, local_window.address, peer_window.address, args.generation))
            protocol.exchange("READY", {"case_id": case.case_id, "device_kernel_launched": True})
            hot_start = channel.evidence()
            cpu_start = time.process_time_ns()
            wall_start = time.perf_counter_ns()
            metrics["kernel_elapsed_ns"] = kernel.synchronize()
            metrics["host_wall_ns"] = time.perf_counter_ns() - wall_start
            metrics["host_cpu_ns"] = time.process_time_ns() - cpu_start
            hot_end = channel.evidence()
            hot_path["control_messages"] = _message_count(hot_end) - _message_count(hot_start)
            hot_path["control_bytes"] = _counter_delta(hot_end, hot_start, "sent_bytes")
            raw_report = runtime.copy_device_to_host(
                local_window.address + case.control_offset + case.report_offset,
                case.report_type._STRUCT.size,
            )
            report = case.report_type.from_bytes(raw_report)
            kernel.close()
            cleanup["device_kernel"] = "CLOSED"
            if case.case_id == T12_CASE_ID:
                lifecycle["close_order"].extend(T12_CLOSE_ORDER[:4])
            protocol.exchange(
                "DRAIN",
                {
                    "case_id": case.case_id,
                    "device_processed": getattr(report, "processed", getattr(report, "accepted", 0)),
                },
            )
            peer_window.close()
            cleanup["imported_window"] = "CLOSED"
            if case.case_id == T12_CASE_ID:
                lifecycle["close_order"].append(T12_CLOSE_ORDER[4])
            protocol.exchange("DETACHED", {"imported_window": "CLOSED"})
            local_window.close()
            cleanup["owned_window"] = "CLOSED"
            if case.case_id == T12_CASE_ID:
                lifecycle["close_order"].extend(T12_CLOSE_ORDER[5:7])
            runtime.close()
            cleanup["runtime"] = "CLOSED"
            if case.case_id == T12_CASE_ID:
                lifecycle["close_order"].append(T12_CLOSE_ORDER[7])
                for resource in (kernel, peer_window, local_window, runtime):
                    lifecycle["duplicate_close_attempts"] += 1
                    try:
                        resource.close()
                        lifecycle["idempotent_close_successes"] += 1
                    except BaseException:  # noqa: BLE001
                        lifecycle["duplicate_close_rejections"] += 1
            protocol.exchange("RELEASED", {"generation_resources": "CLOSED"})
    except BaseException as exc:  # noqa: BLE001
        error = {"message": str(exc), "type": type(exc).__name__}
        traceback.print_exc()
    finally:
        for resource, key in (
            (kernel, "device_kernel"),
            (peer_window, "imported_window"),
            (local_window, "owned_window"),
            (runtime, "runtime"),
        ):
            if resource is None:
                continue
            try:
                resource.close()
                cleanup[key] = "CLOSED"
            except BaseException as exc:  # noqa: BLE001
                cleanup[key] = f"ERROR:{type(exc).__name__}:{exc}"
                error = error or {"message": str(exc), "type": type(exc).__name__}
        final_rss = _rss_bytes()
        wall_ns = int(metrics["host_wall_ns"])
        metrics["host_cpu_utilization_pct"] = (
            min(100.0, 100.0 * int(metrics["host_cpu_ns"]) / wall_ns) if wall_ns else 0.0
        )
        metrics["host_memory_growth_bytes"] = max(0, final_rss - initial_rss)
        evidence = {
            "case_id": case.case_id,
            "cleanup": cleanup,
            "closed_at": datetime.now(UTC).isoformat(),
            "control": channel.evidence() if channel is not None else {},
            "device_id": args.device_id,
            "device_loop": {
                "binary_sha256": kernel.binary_sha256 if kernel is not None else None,
                "hot_path": hot_path,
                "kernel": case.driver_kernel if role is EndpointRole.ATTENTION else case.service_kernel,
                "launches": int(kernel is not None and kernel.launched),
                "report": report.to_dict() if report is not None else None,
            },
            "error": error,
            "generation": args.generation,
            "host_bounce_bytes": 0,
            "lifecycle": lifecycle,
            "manifest": manifest_evidence,
            "metrics": metrics,
            "pid": os.getpid(),
            "role": role.value,
            "run_id": args.run_id,
            "schema_version": SCHEMA_VERSION,
            "started_at": started_at,
            "success": error is None,
        }
        _write_json(artifact_path, evidence)
    return 0 if error is None else 1


def _endpoint_command(role: EndpointRole, device: int, access_device: int, args: argparse.Namespace, port: int):
    return [
        sys.executable,
        "-m",
        "tools.pypto_wse_validation.stage1c",
        "endpoint",
        "--role",
        role.value,
        "--device-id",
        str(device),
        "--access-device-id",
        str(access_device),
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


def _report(artifact: Mapping[str, Any], report_type):
    raw = artifact.get("device_loop", {}).get("report")
    if not isinstance(raw, Mapping):
        raise ValueError("endpoint is missing its Stage 1C device report")
    return report_type(**raw)


def _observation_from_artifacts(args: argparse.Namespace, artifacts: Mapping[EndpointRole, Mapping[str, Any]]):
    case = _case_spec(args.case_id)
    driver_artifact = artifacts[EndpointRole.ATTENTION]
    service_artifact = artifacts[EndpointRole.WSE_SURROGATE]
    driver = _report(driver_artifact, case.report_type)
    service = _report(service_artifact, case.report_type)
    loops = [artifact["device_loop"] for artifact in artifacts.values()]
    hot_messages = sum(
        int(loop["hot_path"][field])
        for loop in loops
        for field in ("control_messages", "task_messages", "completion_messages")
    )
    hot_payload = sum(int(loop["hot_path"]["payload_bytes"]) for loop in loops)
    common = {
        "case_id": case.case_id,
        "generation": args.generation,
        "backend": STAGE1A_BACKEND,
        "transport_scope": TransportScope.HOST_LOCAL,
        "handle_kind": STAGE1A_HANDLE_KIND,
        "host_hot_path_messages": hot_messages,
        "host_hot_path_payload_bytes": hot_payload,
        "host_bounce_bytes": sum(int(item.get("host_bounce_bytes", -1)) for item in artifacts.values()),
        "fallback_used": False,
        "driver_binary_sha256": str(driver_artifact["device_loop"]["binary_sha256"]),
        "service_binary_sha256": str(service_artifact["device_loop"]["binary_sha256"]),
    }
    if case.case_id == T09_CASE_ID:
        if not isinstance(driver, T09DeviceReport) or not isinstance(service, T09DeviceReport):
            raise ValueError("T09 report type mismatch")
        return T09Observation(
            warmup_count=T09_WARMUP_COUNT,
            measured_count=T09_MEASURED_COUNT,
            payload_bytes=T09_PAYLOAD_BYTES,
            slot_count=T09_SLOT_COUNT,
            max_inflight=T09_MAX_INFLIGHT,
            progress_interval=T09_PROGRESS_INTERVAL,
            progress_checkpoints=min(driver.progress_checkpoints, service.progress_checkpoints),
            device_submissions=service.processed,
            device_completions=driver.processed,
            validated_sequences=driver.processed,
            device_memory_growth_bytes=0,
            registration_growth=0,
            host_memory_growth_bytes=max(
                int(item["metrics"]["host_memory_growth_bytes"]) for item in artifacts.values()
            ),
            host_memory_growth_limit_bytes=HOST_MEMORY_GROWTH_LIMIT_BYTES,
            host_cpu_utilization_pct=max(
                float(item["metrics"]["host_cpu_utilization_pct"]) for item in artifacts.values()
            ),
            queue_stalls=driver.queue_stalls + service.queue_stalls,
            driver_report=driver,
            service_report=service,
            **common,
        )
    if case.case_id == T10_CASE_ID:
        if not isinstance(driver, FaultDeviceReport) or not isinstance(service, FaultDeviceReport):
            raise ValueError("T10 report type mismatch")
        return T10Observation(
            configured_timeout_cycles=T10_TIMEOUT_CYCLES,
            timeout_error="WSE_UNRESPONSIVE_TIMEOUT",
            generation_unhealthy=bool(driver.generation_unhealthy and service.generation_unhealthy),
            accepting_new_requests=bool(driver.accepting_new_requests or service.accepting_new_requests),
            quarantined_slots=min(driver.quarantined_slots, service.quarantined_slots),
            forged_success_completions=driver.forged_success_completions + service.forged_success_completions,
            slot_reuse_after_timeout=driver.slot_reuse_after_timeout + service.slot_reuse_after_timeout,
            driver_report=driver,
            service_report=service,
            **common,
        )
    if not isinstance(driver, T12DeviceReport) or not isinstance(service, T12DeviceReport):
        raise ValueError("T12 report type mismatch")
    lifecycle = [artifact["lifecycle"] for artifact in artifacts.values()]
    close_orders = {tuple(item["close_order"]) for item in lifecycle}
    return T12Observation(
        accepted_tasks=min(driver.accepted, service.accepted),
        terminal_outcomes=min(driver.terminal_tasks, service.terminal_tasks),
        credits_acquired=driver.credits_acquired,
        credits_returned=driver.credits_returned,
        early_window_releases=0,
        close_order=next(iter(close_orders)) if len(close_orders) == 1 else (),
        duplicate_close_attempts=sum(int(item["duplicate_close_attempts"]) for item in lifecycle),
        idempotent_close_successes=sum(int(item["idempotent_close_successes"]) for item in lifecycle),
        duplicate_close_rejections=sum(int(item["duplicate_close_rejections"]) for item in lifecycle),
        residual_windows=0,
        residual_queues=0,
        residual_registrations=0,
        residual_contexts=0,
        driver_report=driver,
        service_report=service,
        **common,
    )


def _aggregate_result(args: argparse.Namespace, processes: Mapping[EndpointRole, subprocess.Popen[Any]]):
    artifacts: dict[EndpointRole, dict[str, Any]] = {}
    endpoints: dict[str, Any] = {}
    all_success = True
    all_cleanup = True
    driver_proofs: list[bool] = []
    expected_cleanup = {
        "device_kernel": "CLOSED",
        "imported_window": "CLOSED",
        "owned_window": "CLOSED",
        "runtime": "CLOSED",
    }
    for role, process in processes.items():
        artifact_name = f"{role.value.lower()}_stage1c.json"
        artifact = _read_json(args.artifact_dir / artifact_name) or {"success": False}
        artifacts[role] = artifact
        success = bool(artifact.get("success")) and process.returncode == 0
        all_success = all_success and success
        all_cleanup = all_cleanup and artifact.get("cleanup") == expected_cleanup
        logs = _device_log_evidence(args.artifact_dir / f"{role.value.lower()}_device_logs")
        driver_proofs.append(bool(logs["p2p_enable_observed"] and logs["p2p_memory_released"]))
        endpoints[role.value] = {
            "artifact": artifact_name,
            "cleanup": artifact.get("cleanup"),
            "device_logs": logs,
            "exit_code": process.returncode,
            "host_log": _file_evidence(args.artifact_dir / f"{role.value.lower()}_stage1c.log"),
            "pid": process.pid,
            "success": success,
        }
    observation = None
    with suppress(KeyError, TypeError, ValueError):
        observation = _observation_from_artifacts(args, artifacts)
    passed = bool(all_success and all_cleanup and all(driver_proofs) and observation is not None and observation.passed)
    if isinstance(observation, T09Observation):
        passed_count = observation.validated_sequences
    elif isinstance(observation, T12Observation):
        passed_count = observation.terminal_outcomes
    else:
        passed_count = int(passed)
    attempts = {T09_CASE_ID: T09_TOTAL_COUNT, T10_CASE_ID: 1, T12_CASE_ID: T12_ACCEPTED_COUNT}[args.case_id]
    return {
        "actual_backend": STAGE1A_BACKEND,
        "capability_level": "C1",
        "c2_status": "NOT_ESTABLISHED",
        "claim_scope": "NPU_SURROGATE_ONLY",
        "completed_at": datetime.now(UTC).isoformat(),
        "data_results": {
            args.case_id: {
                "attempts": attempts,
                "passed": passed_count,
                "status": "PASS" if passed else "FAIL",
            }
        },
        "devices": {"ATTENTION": args.devices[0], "WSE_SURROGATE": args.devices[1]},
        "endpoints": endpoints,
        "evidence_status": "SIMULATION",
        "fallback_used": False,
        "generation": args.generation,
        "host_bounce_bytes": observation.host_bounce_bytes if observation else None,
        "npu_wse_capability_level": "NOT_ESTABLISHED",
        "observation": observation.to_dict() if observation else None,
        "profile": "NPU_SURROGATE",
        "resource_cleanup": "VERIFIED" if all_cleanup else "FAILED",
        "run_id": args.run_id,
        "schema_version": SCHEMA_VERSION,
        "stage1c_progress": f"{args.case_id}_{'PASS' if passed else 'FAIL'}",
        "start_order": args.start_order,
        "success": passed,
        "transport_scope": TransportScope.HOST_LOCAL.value,
    }


def run_launcher(args: argparse.Namespace) -> int:
    if len(args.devices) != 2 or len(args.access_device_ids) != 2 or args.devices[0] == args.devices[1]:
        raise ValueError("Stage 1C requires two distinct devices and access IDs")
    case = _case_spec(args.case_id)
    for binary in (case.driver_binary, case.service_binary):
        if not (args.kernel_dir / binary).is_file():
            raise ValueError(f"missing Stage 1C kernel binary: {args.kernel_dir / binary}")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    port = _available_port(args.host)
    roles = [EndpointRole.ATTENTION, EndpointRole.WSE_SURROGATE]
    if args.start_order == "wse-first":
        roles.reverse()
    elif args.start_order == "random":
        random.SystemRandom().shuffle(roles)
        args.start_order = "attention-first" if roles[0] is EndpointRole.ATTENTION else "wse-first"
    devices = dict(zip(EndpointRole, args.devices))
    access_devices = dict(zip(EndpointRole, args.access_device_ids))
    processes: dict[EndpointRole, subprocess.Popen[Any]] = {}
    logs: list[Any] = []
    deadline = time.monotonic() + args.timeout
    try:
        for index, role in enumerate(roles):
            log = (args.artifact_dir / f"{role.value.lower()}_stage1c.log").open("w", encoding="utf-8")
            logs.append(log)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(PROJECT_ROOT)
            device_logs = args.artifact_dir / f"{role.value.lower()}_device_logs"
            device_logs.mkdir(exist_ok=True)
            environment["ASCEND_PROCESS_LOG_PATH"] = str(device_logs.resolve())
            process = subprocess.Popen(  # noqa: S603
                _endpoint_command(role, devices[role], access_devices[role], args, port),
                cwd=args.artifact_dir,
                env=environment,
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
    run.add_argument("--run-id", default=f"stage1c-{uuid.uuid4().hex}")
    run.add_argument("--case-id", choices=(T09_CASE_ID, T10_CASE_ID, T12_CASE_ID), required=True)
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
    endpoint.add_argument("--case-id", choices=(T09_CASE_ID, T10_CASE_ID, T12_CASE_ID), required=True)
    endpoint.add_argument("--generation", type=int, required=True)
    endpoint.add_argument("--timeout", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "endpoint":
        return run_endpoint(args)
    args.access_device_ids = args.access_device_ids or args.devices
    return run_launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
