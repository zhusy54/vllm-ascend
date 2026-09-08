# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Run Stage 1B T04 with two one-shot device-resident AIV loops."""

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
    DeviceLoopReport,
    T04Observation,
)

DRIVER_BINARY = "stage1b_t04_driver.o"
SERVICE_BINARY = "stage1b_t04_service.o"
REPORT_OFFSET = 2 * 64
REPORT_BYTES = 64


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


def _counter_delta(after: Mapping[str, Any], before: Mapping[str, Any], key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def _message_count(evidence: Mapping[str, Any]) -> int:
    return sum(int(value) for value in evidence.get("sent_messages", {}).values())


def run_endpoint(args: argparse.Namespace) -> int:
    role = EndpointRole(args.role)
    artifact_path = args.artifact_dir / f"{role.value.lower()}_stage1b.json"
    started_at = datetime.now(UTC).isoformat()
    runtime: AclVmmRuntime | None = None
    local_window: Any | None = None
    peer_window: Any | None = None
    kernel: AclDeviceKernel | None = None
    channel: ControlChannel | None = None
    report: DeviceLoopReport | None = None
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
        local_window = runtime.allocate_window(T04_WINDOW_BYTES)
        cleanup["owned_window"] = "OPEN"
        runtime.copy_host_to_device(local_window.address, bytes(T04_WINDOW_BYTES))
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

            binary_name = DRIVER_BINARY if role is EndpointRole.ATTENTION else SERVICE_BINARY
            kernel = AclDeviceKernel(runtime, args.kernel_dir / binary_name)
            cleanup["device_kernel"] = "OPEN"
            kernel.launch(_kernel_arguments(role, local_window.address, peer_window.address, args.generation))
            protocol.exchange("READY", {"case_id": T04_CASE_ID, "device_kernel_launched": True})
            hot_path_start = channel.evidence()
            kernel_elapsed_ns = kernel.synchronize()
            hot_path_end = channel.evidence()
            hot_path["control_bytes"] = _counter_delta(hot_path_end, hot_path_start, "sent_bytes")
            hot_path["control_messages"] = _message_count(hot_path_end) - _message_count(hot_path_start)
            raw_report = runtime.copy_device_to_host(
                local_window.address + T04_CONTROL_OFFSET + REPORT_OFFSET,
                REPORT_BYTES,
            )
            report = DeviceLoopReport.from_bytes(raw_report)
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
                "kernel": T04_DRIVER_KERNEL if role is EndpointRole.ATTENTION else T04_SERVICE_KERNEL,
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
        "--generation",
        str(args.generation),
        "--timeout",
        str(args.timeout),
        "--artifact-dir",
        str(args.artifact_dir.resolve()),
        "--kernel-dir",
        str(args.kernel_dir.resolve()),
    ]


def _report_from_artifact(artifact: Mapping[str, Any]) -> DeviceLoopReport:
    raw = artifact.get("device_loop", {}).get("report")
    if not isinstance(raw, Mapping):
        raise ValueError("endpoint is missing a T04 device report")
    return DeviceLoopReport(**raw)


def _aggregate_result(args: argparse.Namespace, processes: Mapping[EndpointRole, subprocess.Popen[Any]]):
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

    observation: T04Observation | None = None
    try:
        attention_loop = artifacts[EndpointRole.ATTENTION]["device_loop"]
        service_loop = artifacts[EndpointRole.WSE_SURROGATE]["device_loop"]
        driver_report = _report_from_artifact(artifacts[EndpointRole.ATTENTION])
        service_report = _report_from_artifact(artifacts[EndpointRole.WSE_SURROGATE])
        observation = T04Observation(
            case_id=T04_CASE_ID,
            generation=args.generation,
            slot=T04_SLOT,
            sequence_count=T04_SEQUENCE_COUNT,
            payload_bytes=T04_PAYLOAD_BYTES,
            backend=STAGE1A_BACKEND,
            transport_scope=TransportScope.HOST_LOCAL,
            handle_kind=STAGE1A_HANDLE_KIND,
            driver_kernel=str(attention_loop["kernel"]),
            service_kernel=str(service_loop["kernel"]),
            driver_context=T04_DEVICE_CONTEXT,
            service_context=T04_DEVICE_CONTEXT,
            driver_launches=int(attention_loop["launches"]),
            service_launches=int(service_loop["launches"]),
            device_submissions=service_report.processed,
            device_completions=driver_report.processed,
            validated_sequences=driver_report.processed,
            input_fence=T04_INPUT_FENCE,
            output_fence=T04_OUTPUT_FENCE,
            host_hot_path_control_messages=sum(
                int(artifacts[role]["device_loop"]["hot_path"]["control_messages"]) for role in EndpointRole
            ),
            host_hot_path_task_messages=sum(
                int(artifacts[role]["device_loop"]["hot_path"]["task_messages"]) for role in EndpointRole
            ),
            host_hot_path_completion_messages=sum(
                int(artifacts[role]["device_loop"]["hot_path"]["completion_messages"]) for role in EndpointRole
            ),
            host_hot_path_payload_bytes=sum(
                int(artifacts[role]["device_loop"]["hot_path"]["payload_bytes"]) for role in EndpointRole
            ),
            host_bounce_bytes=sum(int(artifacts[role].get("host_bounce_bytes", -1)) for role in EndpointRole),
            fallback_used=False,
            driver_report=driver_report,
            service_report=service_report,
            driver_binary_sha256=str(attention_loop["binary_sha256"]),
            service_binary_sha256=str(service_loop["binary_sha256"]),
        )
    except (KeyError, TypeError, ValueError):
        pass

    t04_passed = bool(
        all_success and all_cleanup and all(driver_proofs) and observation is not None and observation.passed
    )
    return {
        "actual_backend": STAGE1A_BACKEND,
        "capability_level": "C1",
        "claim_scope": "NPU_SURROGATE_ONLY",
        "completed_at": datetime.now(UTC).isoformat(),
        "c2_status": "NOT_ESTABLISHED",
        "data_results": {
            T04_CASE_ID: {
                "attempts": T04_SEQUENCE_COUNT,
                "bytes": T04_SEQUENCE_COUNT * T04_PAYLOAD_BYTES * 2,
                "passed": observation.validated_sequences if observation is not None else 0,
                "status": "PASS" if t04_passed else "FAIL",
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
        "stage1b_progress": "T04_PASS" if t04_passed else "T04_FAIL",
        "start_order": args.start_order,
        "success": t04_passed,
        "transport_scope": TransportScope.HOST_LOCAL.value,
    }


def run_launcher(args: argparse.Namespace) -> int:
    if args.devices[0] == args.devices[1]:
        raise ValueError("Stage 1B requires two distinct devices")
    for binary_name in (DRIVER_BINARY, SERVICE_BINARY):
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
    if args.access_device_ids is None:
        args.access_device_ids = args.devices
    if len(args.devices) != 2 or len(args.access_device_ids) != 2:
        raise ValueError("--devices and --access-device-ids must each contain exactly two IDs")
    return run_launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
