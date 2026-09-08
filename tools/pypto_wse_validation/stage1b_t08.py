# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Run Stage 1B T08 generation-isolation validation on two NPUs."""

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.pypto_wse_validation.acl_kernel import AclDeviceKernel
from tools.pypto_wse_validation.acl_vmm import AclVmmRuntime, StaleImportProbe
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
    T08_BASELINE_SEQUENCE_COUNT,
    T08_CASE_ID,
    T08_CONTROL_OFFSET,
    T08_DEVICE_CONTEXT,
    T08_DRIVER_KERNEL,
    T08_HANDLE_PROBE_API,
    T08_INPUT_FENCE,
    T08_MAX_INFLIGHT,
    T08_NEXT_SEQUENCE_COUNT,
    T08_OUTPUT_FENCE,
    T08_PAYLOAD_BYTES,
    T08_SERVICE_KERNEL,
    T08_SLOT_COUNT,
    T08_WINDOW_BYTES,
    T08DeviceLoopReport,
    T08Observation,
)

T08_DRIVER_BINARY = "stage1b_t08_driver.o"
T08_SERVICE_BINARY = "stage1b_t08_service.o"
T08_REPORT_OFFSET = (2 * 2 * 64) + 64


class _T08KernelArguments(ctypes.Structure):
    _fields_ = [
        ("local_payload", ctypes.c_uint64),
        ("local_control", ctypes.c_uint64),
        ("remote_payload", ctypes.c_uint64),
        ("remote_control", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("previous_generation", ctypes.c_uint64),
        ("payload_words", ctypes.c_uint32),
        ("sequence_count", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
    ]


def _kernel_arguments(
    local_address: int,
    peer_address: int,
    *,
    generation: int,
    previous_generation: int,
    sequence_count: int,
    mode: int,
) -> _T08KernelArguments:
    return _T08KernelArguments(
        local_payload=local_address,
        local_control=local_address + T08_CONTROL_OFFSET,
        remote_payload=peer_address,
        remote_control=peer_address + T08_CONTROL_OFFSET,
        generation=generation,
        previous_generation=previous_generation,
        payload_words=T08_PAYLOAD_BYTES // ctypes.sizeof(ctypes.c_uint64),
        sequence_count=sequence_count,
        mode=mode,
    )


def _counter_delta(after: Mapping[str, Any], before: Mapping[str, Any], key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def _message_count(evidence: Mapping[str, Any]) -> int:
    return sum(int(value) for value in evidence.get("sent_messages", {}).values())


def _probe_evidence(probe: StaleImportProbe) -> dict[str, Any]:
    return {
        "api": probe.api,
        "rejected": probe.rejected,
        "result_code": probe.result_code,
    }


def _empty_cleanup() -> dict[str, str]:
    return {
        "device_kernel": "NOT_LOADED",
        "imported_window": "NOT_CREATED",
        "owned_window": "NOT_CREATED",
        "runtime": "NOT_INITIALIZED",
    }


def _close_generation(state: dict[str, Any], cleanup: dict[str, str]) -> None:
    failures: list[BaseException] = []
    for key, cleanup_key in (
        ("kernel", "device_kernel"),
        ("peer_window", "imported_window"),
        ("local_window", "owned_window"),
        ("runtime", "runtime"),
    ):
        resource = state.get(key)
        if resource is None:
            continue
        closed = bool(getattr(resource, "closed", False))
        if not closed:
            try:
                resource.close()
            except BaseException as exc:  # noqa: BLE001
                cleanup[cleanup_key] = f"ERROR:{type(exc).__name__}:{exc}"
                failures.append(exc)
                continue
        cleanup[cleanup_key] = "CLOSED"
    if failures:
        raise failures[0]


def _execute_generation(
    *,
    role: EndpointRole,
    args: argparse.Namespace,
    protocol: _Protocol,
    channel: ControlChannel,
    state: dict[str, Any],
    cleanup: dict[str, str],
    generation: int,
    previous_generation: int,
    sequence_count: int,
    mode: int,
) -> dict[str, Any]:
    runtime = state["runtime"]
    local_window = state["local_window"]
    peer_window = state["peer_window"]
    binary_name = T08_DRIVER_BINARY if role is EndpointRole.ATTENTION else T08_SERVICE_BINARY
    kernel_name = T08_DRIVER_KERNEL if role is EndpointRole.ATTENTION else T08_SERVICE_KERNEL
    kernel = AclDeviceKernel(runtime, args.kernel_dir / binary_name)
    state["kernel"] = kernel
    cleanup["device_kernel"] = "OPEN"
    kernel.launch(
        _kernel_arguments(
            local_window.address,
            peer_window.address,
            generation=generation,
            previous_generation=previous_generation,
            sequence_count=sequence_count,
            mode=mode,
        )
    )
    protocol.exchange(
        "READY",
        {"case_id": T08_CASE_ID, "device_kernel_launched": True, "mode": mode},
    )
    hot_path_start = channel.evidence()
    elapsed_ns = kernel.synchronize()
    hot_path_end = channel.evidence()
    hot_path = {
        "control_bytes": _counter_delta(hot_path_end, hot_path_start, "sent_bytes"),
        "control_messages": _message_count(hot_path_end) - _message_count(hot_path_start),
        "task_messages": 0,
        "completion_messages": 0,
        "payload_bytes": 0,
    }
    raw_report = runtime.copy_device_to_host(
        local_window.address + T08_CONTROL_OFFSET + T08_REPORT_OFFSET,
        T08DeviceLoopReport._STRUCT.size,
    )
    report = T08DeviceLoopReport.from_bytes(raw_report)
    binary_sha256 = kernel.binary_sha256
    kernel.close()
    cleanup["device_kernel"] = "CLOSED"
    protocol.exchange(
        "DRAIN",
        {"case_id": T08_CASE_ID, "device_processed": report.processed, "mode": mode},
    )
    return {
        "binary_sha256": binary_sha256,
        "elapsed_ns": elapsed_ns,
        "generation": generation,
        "hot_path": hot_path,
        "kernel": kernel_name,
        "launches": 1,
        "mode": mode,
        "report": report.to_dict(),
    }


def _sanitized_manifest(local_manifest: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = dict(local_manifest)
    shareable_handle = int(sanitized.pop("shareable_handle"))
    sanitized["opaque_handle"] = opaque_handle_evidence(shareable_handle)
    return sanitized


def run_endpoint(args: argparse.Namespace) -> int:
    role = EndpointRole(args.role)
    artifact_path = args.artifact_dir / f"{role.value.lower()}_stage1b.json"
    started_at = datetime.now(UTC).isoformat()
    deadline = time.monotonic() + args.timeout
    channel: ControlChannel | None = None
    states = {"baseline": {}, "next": {}}
    cleanup = {"baseline": _empty_cleanup(), "next": _empty_cleanup()}
    generation_runs: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    handle_invalidation: dict[str, Any] = {"api": T08_HANDLE_PROBE_API}
    error: dict[str, str] | None = None
    try:
        connection = (
            _connect(args.host, args.port, deadline)
            if role is EndpointRole.ATTENTION
            else _listen(args.host, args.port, deadline)
        )
        with connection:
            connection.settimeout(max(0.001, deadline - time.monotonic()))
            channel = ControlChannel(connection)

            baseline = states["baseline"]
            baseline_runtime = AclVmmRuntime(args.device_id, access_device_id=args.access_device_id)
            baseline["runtime"] = baseline_runtime
            baseline_runtime.initialize()
            cleanup["baseline"]["runtime"] = "OPEN"
            baseline_local = baseline_runtime.allocate_window(T08_WINDOW_BYTES)
            baseline["local_window"] = baseline_local
            cleanup["baseline"]["owned_window"] = "OPEN"
            baseline_runtime.copy_host_to_device(baseline_local.address, bytes(T08_WINDOW_BYTES))
            baseline_manifest = _manifest(baseline_runtime, baseline_local)
            manifests.append({"generation": args.generation, **_sanitized_manifest(baseline_manifest)})
            old_local_handle = int(baseline_local.shareable_handle)

            baseline_protocol = _Protocol(
                channel,
                role=role,
                run_id=args.run_id,
                generation=args.generation,
            )
            peer_message = baseline_protocol.exchange("MANIFEST", {"window": baseline_manifest})
            old_peer_export, peer_device_id = _parse_peer_export(peer_message)
            if peer_device_id == args.device_id:
                raise ValueError("peer must use a distinct device")
            baseline_peer = baseline_runtime.import_window(old_peer_export, peer_device_id=peer_device_id)
            baseline["peer_window"] = baseline_peer
            cleanup["baseline"]["imported_window"] = "OPEN"
            baseline_protocol.exchange("ATTACHED", {"peer_mapping_bytes": baseline_peer.mapping_bytes})
            generation_runs.append(
                _execute_generation(
                    role=role,
                    args=args,
                    protocol=baseline_protocol,
                    channel=channel,
                    state=baseline,
                    cleanup=cleanup["baseline"],
                    generation=args.generation,
                    previous_generation=args.generation - 1,
                    sequence_count=T08_BASELINE_SEQUENCE_COUNT,
                    mode=0,
                )
            )
            baseline_peer.close()
            cleanup["baseline"]["imported_window"] = "CLOSED"
            baseline_protocol.exchange("DETACHED", {"imported_window": "CLOSED"})
            baseline_local.close()
            cleanup["baseline"]["owned_window"] = "CLOSED"
            baseline_runtime.close()
            cleanup["baseline"]["runtime"] = "CLOSED"
            baseline_protocol.exchange("RELEASED", {"generation_resources": "CLOSED"})

            next_generation = args.generation + 1
            next_protocol = _Protocol(
                channel,
                role=role,
                run_id=args.run_id,
                generation=next_generation,
            )
            next_state = states["next"]
            next_runtime = AclVmmRuntime(args.device_id, access_device_id=args.access_device_id)
            next_state["runtime"] = next_runtime
            next_runtime.initialize()
            cleanup["next"]["runtime"] = "OPEN"
            before_probe = next_runtime.probe_stale_import(old_peer_export)
            handle_invalidation["before_reallocate"] = _probe_evidence(before_probe)
            next_protocol.exchange(
                "PROBE_RESULT",
                {"phase": "BEFORE_REALLOCATE", **_probe_evidence(before_probe)},
            )

            next_local = next_runtime.allocate_window(T08_WINDOW_BYTES)
            next_state["local_window"] = next_local
            cleanup["next"]["owned_window"] = "OPEN"
            next_runtime.copy_host_to_device(next_local.address, bytes(T08_WINDOW_BYTES))
            next_manifest = _manifest(next_runtime, next_local)
            manifests.append({"generation": next_generation, **_sanitized_manifest(next_manifest)})
            peer_message = next_protocol.exchange("MANIFEST", {"window": next_manifest})
            next_peer_export, next_peer_device_id = _parse_peer_export(peer_message)
            if next_peer_device_id != peer_device_id:
                raise ValueError("peer device changed across generations")

            old_local_opaque = opaque_handle_evidence(old_local_handle)
            next_local_opaque = opaque_handle_evidence(next_local.shareable_handle)
            old_peer_opaque = opaque_handle_evidence(old_peer_export.shareable_handle)
            next_peer_opaque = opaque_handle_evidence(next_peer_export.shareable_handle)
            handle_invalidation["local_handle_changed"] = old_local_opaque != next_local_opaque
            handle_invalidation["peer_handle_changed"] = old_peer_opaque != next_peer_opaque
            handle_invalidation["old_local_opaque_handle"] = old_local_opaque
            handle_invalidation["new_local_opaque_handle"] = next_local_opaque
            handle_invalidation["old_peer_opaque_handle"] = old_peer_opaque
            handle_invalidation["new_peer_opaque_handle"] = next_peer_opaque

            after_probe = next_runtime.probe_stale_import(old_peer_export)
            handle_invalidation["after_reallocate"] = _probe_evidence(after_probe)
            next_protocol.exchange(
                "PROBE_RESULT",
                {"phase": "AFTER_REALLOCATE", **_probe_evidence(after_probe)},
            )
            next_peer = next_runtime.import_window(next_peer_export, peer_device_id=next_peer_device_id)
            next_state["peer_window"] = next_peer
            cleanup["next"]["imported_window"] = "OPEN"
            next_protocol.exchange("ATTACHED", {"peer_mapping_bytes": next_peer.mapping_bytes})
            generation_runs.append(
                _execute_generation(
                    role=role,
                    args=args,
                    protocol=next_protocol,
                    channel=channel,
                    state=next_state,
                    cleanup=cleanup["next"],
                    generation=next_generation,
                    previous_generation=args.generation,
                    sequence_count=T08_NEXT_SEQUENCE_COUNT,
                    mode=1,
                )
            )
            next_peer.close()
            cleanup["next"]["imported_window"] = "CLOSED"
            next_protocol.exchange("DETACHED", {"imported_window": "CLOSED"})
            next_local.close()
            cleanup["next"]["owned_window"] = "CLOSED"
            next_runtime.close()
            cleanup["next"]["runtime"] = "CLOSED"
            next_protocol.exchange("RELEASED", {"generation_resources": "CLOSED"})
    except BaseException as exc:  # noqa: BLE001
        error = {"message": str(exc), "type": type(exc).__name__}
        traceback.print_exc()
    finally:
        for name in ("next", "baseline"):
            try:
                _close_generation(states[name], cleanup[name])
            except BaseException as exc:  # noqa: BLE001
                error = error or {"message": str(exc), "type": type(exc).__name__}
        evidence = {
            "cleanup": cleanup,
            "closed_at": datetime.now(UTC).isoformat(),
            "control": channel.evidence() if channel is not None else {},
            "device_id": args.device_id,
            "error": error,
            "generation": args.generation,
            "generation_runs": generation_runs,
            "handle_invalidation": handle_invalidation,
            "host_bounce_bytes": 0,
            "manifests": manifests,
            "pid": os.getpid(),
            "role": role.value,
            "run_id": args.run_id,
            "schema_version": SCHEMA_VERSION,
            "started_at": started_at,
            "success": error is None,
        }
        _write_json(artifact_path, evidence)
    return 0 if error is None else 1


def _endpoint_command(
    role: EndpointRole,
    device_id: int,
    access_device_id: int,
    args: argparse.Namespace,
    port: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.pypto_wse_validation.stage1b_t08",
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
        "--generation",
        str(args.generation),
        "--timeout",
        str(args.timeout),
        "--artifact-dir",
        str(args.artifact_dir.resolve()),
        "--kernel-dir",
        str(args.kernel_dir.resolve()),
    ]


def _generation_run(artifact: Mapping[str, Any], mode: int) -> Mapping[str, Any]:
    matches = [run for run in artifact.get("generation_runs", []) if int(run.get("mode", -1)) == mode]
    if len(matches) != 1:
        raise ValueError(f"endpoint must contain exactly one T08 mode {mode} run")
    return matches[0]


def _report(run: Mapping[str, Any]) -> T08DeviceLoopReport:
    raw = run.get("report")
    if not isinstance(raw, Mapping):
        raise ValueError("endpoint is missing a T08 device report")
    return T08DeviceLoopReport(**raw)


def _observation_from_artifacts(
    args: argparse.Namespace,
    artifacts: Mapping[EndpointRole, Mapping[str, Any]],
) -> T08Observation:
    attention = artifacts[EndpointRole.ATTENTION]
    service = artifacts[EndpointRole.WSE_SURROGATE]
    baseline_driver_run = _generation_run(attention, 0)
    baseline_service_run = _generation_run(service, 0)
    next_driver_run = _generation_run(attention, 1)
    next_service_run = _generation_run(service, 1)
    baseline_driver = _report(baseline_driver_run)
    baseline_service = _report(baseline_service_run)
    next_driver = _report(next_driver_run)
    next_service = _report(next_service_run)
    invalidation = [artifact["handle_invalidation"] for artifact in artifacts.values()]
    runs = [baseline_driver_run, baseline_service_run, next_driver_run, next_service_run]
    old_handle_rejected_before = sum(bool(item["before_reallocate"]["rejected"]) for item in invalidation)
    old_handle_rejected_after = sum(bool(item["after_reallocate"]["rejected"]) for item in invalidation)
    return T08Observation(
        case_id=T08_CASE_ID,
        generation=args.generation,
        next_generation=args.generation + 1,
        baseline_sequence_count=T08_BASELINE_SEQUENCE_COUNT,
        next_sequence_count=T08_NEXT_SEQUENCE_COUNT,
        slot_count=T08_SLOT_COUNT,
        max_inflight=T08_MAX_INFLIGHT,
        payload_bytes=T08_PAYLOAD_BYTES,
        device_submissions=baseline_service.processed + next_service.processed,
        device_completions=baseline_driver.processed + next_driver.processed,
        validated_sequences=baseline_driver.processed + next_driver.processed,
        stale_descriptor_injections=next_driver.stale_descriptor_injections,
        stale_descriptor_rejections=next_service.stale_descriptor_rejections,
        stale_completion_injections=next_service.stale_completion_injections,
        stale_completion_rejections=next_driver.stale_completion_rejections,
        old_completion_credit_releases=(
            next_driver.old_completion_credit_releases + next_service.old_completion_credit_releases
        ),
        current_slot_preserved=bool(next_driver.current_slot_preserved and next_service.current_slot_preserved),
        progress_after_stale=min(next_driver.progress_after_stale, next_service.progress_after_stale),
        old_handle_rejected_before_reallocate=old_handle_rejected_before,
        old_handle_rejected_after_reallocate=old_handle_rejected_after,
        old_handle_import_successes=4 - old_handle_rejected_before - old_handle_rejected_after,
        new_handle_collisions=sum(
            not bool(item[field]) for item in invalidation for field in ("local_handle_changed", "peer_handle_changed")
        ),
        handle_probe_api=T08_HANDLE_PROBE_API,
        backend=STAGE1A_BACKEND,
        transport_scope=TransportScope.HOST_LOCAL,
        handle_kind=STAGE1A_HANDLE_KIND,
        driver_kernel=str(baseline_driver_run["kernel"]),
        service_kernel=str(baseline_service_run["kernel"]),
        driver_context=T08_DEVICE_CONTEXT,
        service_context=T08_DEVICE_CONTEXT,
        driver_launches=int(baseline_driver_run["launches"]) + int(next_driver_run["launches"]),
        service_launches=int(baseline_service_run["launches"]) + int(next_service_run["launches"]),
        input_fence=T08_INPUT_FENCE,
        output_fence=T08_OUTPUT_FENCE,
        host_hot_path_control_messages=sum(int(run["hot_path"]["control_messages"]) for run in runs),
        host_hot_path_task_messages=sum(int(run["hot_path"]["task_messages"]) for run in runs),
        host_hot_path_completion_messages=sum(int(run["hot_path"]["completion_messages"]) for run in runs),
        host_hot_path_payload_bytes=sum(int(run["hot_path"]["payload_bytes"]) for run in runs),
        host_bounce_bytes=sum(int(artifact.get("host_bounce_bytes", -1)) for artifact in artifacts.values()),
        fallback_used=False,
        baseline_driver_report=baseline_driver,
        baseline_service_report=baseline_service,
        next_driver_report=next_driver,
        next_service_report=next_service,
        driver_binary_sha256=str(baseline_driver_run["binary_sha256"]),
        service_binary_sha256=str(baseline_service_run["binary_sha256"]),
    )


def _aggregate_result(
    args: argparse.Namespace,
    processes: Mapping[EndpointRole, subprocess.Popen[Any]],
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    artifacts: dict[EndpointRole, dict[str, Any]] = {}
    all_success = True
    all_cleanup = True
    driver_proofs: list[bool] = []
    expected_cleanup = {"baseline": _empty_cleanup(), "next": _empty_cleanup()}
    for generation_cleanup in expected_cleanup.values():
        generation_cleanup.update(
            {
                "device_kernel": "CLOSED",
                "imported_window": "CLOSED",
                "owned_window": "CLOSED",
                "runtime": "CLOSED",
            }
        )
    for role, process in processes.items():
        artifact_name = f"{role.value.lower()}_stage1b.json"
        artifact = _read_json(args.artifact_dir / artifact_name) or {"success": False}
        artifacts[role] = artifact
        endpoint_success = bool(artifact.get("success")) and process.returncode == 0
        all_success = all_success and endpoint_success
        endpoint_cleanup = artifact.get("cleanup", {})
        all_cleanup = all_cleanup and endpoint_cleanup == expected_cleanup
        device_logs = _device_log_evidence(args.artifact_dir / f"{role.value.lower()}_device_logs")
        driver_proofs.append(bool(device_logs["p2p_enable_observed"] and device_logs["p2p_memory_released"]))
        endpoints[role.value] = {
            "artifact": artifact_name,
            "cleanup": endpoint_cleanup,
            "device_logs": device_logs,
            "exit_code": process.returncode,
            "host_log": _file_evidence(args.artifact_dir / f"{role.value.lower()}_stage1b.log"),
            "pid": process.pid,
            "success": endpoint_success,
        }

    observation: T08Observation | None = None
    binaries_consistent = False
    with suppress(KeyError, TypeError, ValueError):
        observation = _observation_from_artifacts(args, artifacts)
        binaries_consistent = all(
            _generation_run(artifacts[role], 0)["binary_sha256"] == _generation_run(artifacts[role], 1)["binary_sha256"]
            for role in EndpointRole
        )
    case_passed = bool(
        all_success
        and all_cleanup
        and all(driver_proofs)
        and binaries_consistent
        and observation is not None
        and observation.passed
    )
    total_sequences = T08_BASELINE_SEQUENCE_COUNT + T08_NEXT_SEQUENCE_COUNT
    return {
        "actual_backend": STAGE1A_BACKEND,
        "capability_level": "C1",
        "claim_scope": "NPU_SURROGATE_ONLY",
        "completed_at": datetime.now(UTC).isoformat(),
        "c2_status": "NOT_ESTABLISHED",
        "data_results": {
            T08_CASE_ID: {
                "attempts": total_sequences,
                "bytes": total_sequences * T08_PAYLOAD_BYTES * 2,
                "passed": observation.validated_sequences if observation is not None else 0,
                "status": "PASS" if case_passed else "FAIL",
            }
        },
        "devices": {"ATTENTION": args.devices[0], "WSE_SURROGATE": args.devices[1]},
        "endpoints": endpoints,
        "evidence_status": "SIMULATION",
        "fallback_used": False,
        "generation": args.generation,
        "next_generation": args.generation + 1,
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
        "stage1b_progress": f"{T08_CASE_ID}_{'PASS' if case_passed else 'FAIL'}",
        "start_order": args.start_order,
        "success": case_passed,
        "transport_scope": TransportScope.HOST_LOCAL.value,
    }


def run_launcher(args: argparse.Namespace) -> int:
    if args.devices[0] == args.devices[1]:
        raise ValueError("Stage 1B T08 requires two distinct devices")
    for binary_name in (T08_DRIVER_BINARY, T08_SERVICE_BINARY):
        if not (args.kernel_dir / binary_name).is_file():
            raise ValueError(f"missing Stage 1B T08 kernel binary: {args.kernel_dir / binary_name}")
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
    run.add_argument("--run-id", default=f"stage1b-t08-{uuid.uuid4().hex}")
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
    endpoint.add_argument("--generation", type=int, required=True)
    endpoint.add_argument("--timeout", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "endpoint":
        return run_endpoint(args)
    args.access_device_ids = args.access_device_ids or args.devices
    if len(args.devices) != 2 or len(args.access_device_ids) != 2:
        raise ValueError("--devices and --access-device-ids must each contain exactly two IDs")
    return run_launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
