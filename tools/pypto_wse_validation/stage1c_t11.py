# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Run Stage 1C T11 process-exit and next-generation recovery validation."""

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
    _write_json,
)
from tools.pypto_wse_validation.stage1a_contracts import opaque_handle_evidence
from tools.pypto_wse_validation.stage1c import _FaultArguments
from tools.pypto_wse_validation.stage1c_contracts import (
    T10_CONTROL_OFFSET,
    T10_PAYLOAD_BYTES,
    T10_WINDOW_BYTES,
    T11_CASE_ID,
    T11_FAULT_LIMITATION,
    T11_TIMEOUT_CYCLES,
    FaultDeviceReport,
    T11Observation,
)

FAULT_DRIVER_BINARY = "stage1c_fault_driver.o"
FAULT_SERVICE_BINARY = "stage1c_fault_service.o"
FAULT_DRIVER_KERNEL = "pypto_stage1c_fault_driver_0_mix_aiv"
FAULT_SERVICE_KERNEL = "pypto_stage1c_fault_service_0_mix_aiv"
FAULT_REPORT_OFFSET = 2 * 64


def _ready_path(artifact_dir: Path, role: EndpointRole) -> Path:
    return artifact_dir / f"{role.value.lower()}_t11_ready.json"


def _kernel_arguments(local_address: int, peer_address: int, generation: int) -> _FaultArguments:
    return _FaultArguments(
        local_payload=local_address,
        local_control=local_address + T10_CONTROL_OFFSET,
        remote_payload=peer_address,
        remote_control=peer_address + T10_CONTROL_OFFSET,
        generation=generation,
        timeout_cycles=T11_TIMEOUT_CYCLES,
        payload_words=T10_PAYLOAD_BYTES // ctypes.sizeof(ctypes.c_uint64),
        mode=11,
    )


def run_endpoint(args: argparse.Namespace) -> int:
    role = EndpointRole(args.role)
    artifact_path = args.artifact_dir / f"{role.value.lower()}_stage1c_t11.json"
    runtime: AclVmmRuntime | None = None
    local_window: Any | None = None
    peer_window: Any | None = None
    kernel: AclDeviceKernel | None = None
    channel: ControlChannel | None = None
    report: FaultDeviceReport | None = None
    probe_evidence: dict[str, Any] | None = None
    cleanup = {
        "device_kernel": "NOT_LOADED",
        "imported_window": "NOT_CREATED",
        "owned_window": "NOT_CREATED",
        "runtime": "NOT_INITIALIZED",
    }
    error: dict[str, str] | None = None
    deadline = time.monotonic() + args.timeout
    started_at = datetime.now(UTC).isoformat()
    try:
        runtime = AclVmmRuntime(args.device_id, access_device_id=args.access_device_id)
        runtime.initialize()
        cleanup["runtime"] = "OPEN"
        local_window = runtime.allocate_window(T10_WINDOW_BYTES)
        cleanup["owned_window"] = "OPEN"
        runtime.copy_host_to_device(local_window.address, bytes(T10_WINDOW_BYTES))
        local_manifest = _manifest(runtime, local_window)
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
            binary = FAULT_DRIVER_BINARY if role is EndpointRole.ATTENTION else FAULT_SERVICE_BINARY
            kernel_name = FAULT_DRIVER_KERNEL if role is EndpointRole.ATTENTION else FAULT_SERVICE_KERNEL
            kernel = AclDeviceKernel(runtime, args.kernel_dir / binary)
            cleanup["device_kernel"] = "OPEN"
            kernel.launch(_kernel_arguments(local_window.address, peer_window.address, args.generation))
            protocol.exchange("READY", {"case_id": T11_CASE_ID, "device_kernel_launched": True})
            _write_json(
                _ready_path(args.artifact_dir, role),
                {"generation": args.generation, "kernel": kernel_name, "role": role.value},
            )
            elapsed_ns = kernel.synchronize()
            raw_report = runtime.copy_device_to_host(
                local_window.address + T10_CONTROL_OFFSET + FAULT_REPORT_OFFSET,
                FaultDeviceReport._STRUCT.size,
            )
            report = FaultDeviceReport.from_bytes(raw_report)
            kernel.close()
            cleanup["device_kernel"] = "CLOSED"
            peer_window.close()
            cleanup["imported_window"] = "CLOSED"
            probe = runtime.probe_stale_import(peer_export)
            probe_evidence = {
                "api": probe.api,
                "opaque_handle": opaque_handle_evidence(peer_export.shareable_handle),
                "rejected": probe.rejected,
                "result_code": probe.result_code,
            }
            local_window.close()
            cleanup["owned_window"] = "CLOSED"
            runtime.close()
            cleanup["runtime"] = "CLOSED"
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
        _write_json(
            artifact_path,
            {
                "case_id": T11_CASE_ID,
                "cleanup": cleanup,
                "closed_at": datetime.now(UTC).isoformat(),
                "control": channel.evidence() if channel is not None else {},
                "device_id": args.device_id,
                "device_loop": {
                    "binary_sha256": kernel.binary_sha256 if kernel is not None else None,
                    "elapsed_ns": elapsed_ns if "elapsed_ns" in locals() else 0,
                    "kernel": kernel_name if "kernel_name" in locals() else None,
                    "launches": int(kernel is not None and kernel.launched),
                    "report": report.to_dict() if report is not None else None,
                },
                "error": error,
                "generation": args.generation,
                "host_bounce_bytes": 0,
                "old_peer_handle_probe": probe_evidence,
                "pid": os.getpid(),
                "role": role.value,
                "run_id": args.run_id,
                "schema_version": SCHEMA_VERSION,
                "started_at": started_at,
                "success": error is None and probe_evidence is not None and probe_evidence["rejected"] is True,
            },
        )
    return 0 if error is None and probe_evidence is not None and probe_evidence["rejected"] is True else 1


def _endpoint_command(role: EndpointRole, device: int, access_device: int, args: argparse.Namespace, port: int):
    return [
        sys.executable,
        "-m",
        "tools.pypto_wse_validation.stage1c_t11",
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
        "--generation",
        str(args.generation),
        "--timeout",
        str(args.timeout),
        "--artifact-dir",
        str(args.artifact_dir.resolve()),
        "--kernel-dir",
        str(args.kernel_dir.resolve()),
    ]


def _wait_ready(args: argparse.Namespace, processes: Mapping[EndpointRole, subprocess.Popen[Any]]) -> bool:
    deadline = time.monotonic() + args.ready_timeout
    while time.monotonic() < deadline:
        if all(_ready_path(args.artifact_dir, role).is_file() for role in EndpointRole):
            return True
        if any(process.poll() is not None for process in processes.values()):
            return False
        time.sleep(0.02)
    return False


def _run_recovery(args: argparse.Namespace) -> tuple[int, dict[str, Any] | None]:
    recovery_dir = args.artifact_dir / "recovery"
    command = [
        sys.executable,
        "-m",
        "tools.pypto_wse_validation.stage1b",
        "run",
        "--devices",
        ",".join(str(device) for device in args.devices),
        "--access-device-ids",
        ",".join(str(device) for device in args.access_device_ids),
        "--artifact-dir",
        str(recovery_dir.resolve()),
        "--kernel-dir",
        str(args.kernel_dir.resolve()),
        "--case-id",
        "T04",
        "--generation",
        str(args.generation + 1),
        "--timeout",
        str(args.timeout),
        "--start-order",
        args.start_order,
    ]
    with (args.artifact_dir / "recovery.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=args.artifact_dir,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
    return completed.returncode, _read_json(recovery_dir / "result.json")


def _aggregate_result(
    args: argparse.Namespace,
    processes: Mapping[EndpointRole, subprocess.Popen[Any]],
    *,
    survivor_elapsed_seconds: float,
    recovery_returncode: int,
    recovery_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    victim = EndpointRole(args.victim_role)
    survivor = EndpointRole.WSE_SURROGATE if victim is EndpointRole.ATTENTION else EndpointRole.ATTENTION
    survivor_artifact = _read_json(args.artifact_dir / f"{survivor.value.lower()}_stage1c_t11.json") or {}
    raw_report = survivor_artifact.get("device_loop", {}).get("report")
    observation = None
    with suppress(KeyError, TypeError, ValueError):
        report = FaultDeviceReport(**raw_report)
        probe = survivor_artifact["old_peer_handle_probe"]
        recovery_observation = recovery_result["observation"]
        observation = T11Observation(
            case_id=T11_CASE_ID,
            generation=args.generation,
            next_generation=args.generation + 1,
            victim_role=victim,
            victim_exit_code=int(processes[victim].returncode),
            survivor_role=survivor,
            survivor_report=report,
            survivor_cleanup_complete=survivor_artifact.get("cleanup")
            == {
                "device_kernel": "CLOSED",
                "imported_window": "CLOSED",
                "owned_window": "CLOSED",
                "runtime": "CLOSED",
            },
            survivor_wait_bounded=survivor_elapsed_seconds < args.timeout,
            old_handle_rejected=probe["rejected"] is True,
            old_handle_probe_result=int(probe["result_code"]),
            recovery_success=recovery_returncode == 0 and recovery_result["success"] is True,
            recovery_validated_sequences=int(recovery_observation["validated_sequences"]),
            old_resource_reused=False,
            launcher_reported_expected_failure=int(processes[victim].returncode) < 0,
            limitation=T11_FAULT_LIMITATION,
        )
    passed = bool(observation is not None and observation.passed and survivor_artifact.get("success") is True)
    endpoints = {}
    for role, process in processes.items():
        logs = _device_log_evidence(args.artifact_dir / f"{role.value.lower()}_device_logs")
        endpoints[role.value] = {
            "device_logs": logs,
            "exit_code": process.returncode,
            "expected_process_exit": role is victim,
            "host_log": _file_evidence(args.artifact_dir / f"{role.value.lower()}_stage1c_t11.log"),
            "success": (role is victim and process.returncode < 0)
            or (role is survivor and survivor_artifact.get("success") is True),
        }
    return {
        "actual_backend": "CANN_ACL_VMM_P2P",
        "capability_level": "C1",
        "c2_status": "NOT_ESTABLISHED",
        "claim_scope": "NPU_SURROGATE_ONLY",
        "completed_at": datetime.now(UTC).isoformat(),
        "data_results": {T11_CASE_ID: {"attempts": 1, "passed": int(passed), "status": "PASS" if passed else "FAIL"}},
        "devices": {"ATTENTION": args.devices[0], "WSE_SURROGATE": args.devices[1]},
        "endpoints": endpoints,
        "evidence_status": "SIMULATION",
        "fault_injection": {"signal": "SIGKILL", "victim_role": victim.value},
        "generation": args.generation,
        "next_generation": args.generation + 1,
        "npu_wse_capability_level": "NOT_ESTABLISHED",
        "observation": observation.to_dict() if observation is not None else None,
        "profile": "NPU_SURROGATE",
        "recovery": {
            "artifact": "recovery/result.json",
            "host_log": _file_evidence(args.artifact_dir / "recovery.log"),
            "return_code": recovery_returncode,
            "success": bool(recovery_result and recovery_result.get("success")),
        },
        "run_id": args.run_id,
        "schema_version": SCHEMA_VERSION,
        "stage1c_progress": f"{T11_CASE_ID}_{'PASS' if passed else 'FAIL'}",
        "start_order": args.start_order,
        "success": passed,
        "transport_scope": TransportScope.HOST_LOCAL.value,
    }


def run_launcher(args: argparse.Namespace) -> int:
    if len(args.devices) != 2 or len(args.access_device_ids) != 2 or args.devices[0] == args.devices[1]:
        raise ValueError("T11 requires two distinct devices and access IDs")
    for binary in (FAULT_DRIVER_BINARY, FAULT_SERVICE_BINARY, "stage1b_t04_driver.o", "stage1b_t04_service.o"):
        if not (args.kernel_dir / binary).is_file():
            raise ValueError(f"missing T11 or recovery kernel binary: {args.kernel_dir / binary}")
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
    victim = EndpointRole(args.victim_role)
    survivor = EndpointRole.WSE_SURROGATE if victim is EndpointRole.ATTENTION else EndpointRole.ATTENTION
    survivor_started = 0.0
    survivor_elapsed = args.timeout
    try:
        for index, role in enumerate(roles):
            log = (args.artifact_dir / f"{role.value.lower()}_stage1c_t11.log").open("w", encoding="utf-8")
            logs.append(log)
            environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
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
        if not _wait_ready(args, processes):
            raise RuntimeError("T11 endpoints did not both reach READY")
        time.sleep(args.fault_delay)
        survivor_started = time.monotonic()
        processes[victim].kill()
        processes[victim].wait(timeout=5)
        processes[survivor].wait(timeout=args.timeout)
        survivor_elapsed = time.monotonic() - survivor_started
    finally:
        _stop_processes(processes)
        for log in logs:
            log.close()
    recovery_returncode, recovery_result = _run_recovery(args)
    result = _aggregate_result(
        args,
        processes,
        survivor_elapsed_seconds=survivor_elapsed,
        recovery_returncode=recovery_returncode,
        recovery_result=recovery_result,
    )
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
    run.add_argument("--run-id", default=f"stage1c-t11-{uuid.uuid4().hex}")
    run.add_argument("--generation", type=int, default=DEFAULT_GENERATION)
    run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    run.add_argument("--ready-timeout", type=float, default=30.0)
    run.add_argument("--fault-delay", type=float, default=0.5)
    run.add_argument("--start-delay", type=float, default=DEFAULT_START_DELAY_SECONDS)
    run.add_argument("--start-order", choices=("attention-first", "wse-first", "random"), default="random")
    run.add_argument("--victim-role", choices=tuple(role.value for role in EndpointRole), required=True)
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
    return run_launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
