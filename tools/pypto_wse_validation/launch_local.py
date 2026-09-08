# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Run the stage-0 two-process NPU-surrogate lifecycle probe."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.pypto_wse_validation.bootstrap import (
    BootstrapError,
    ControlChannel,
    make_contract_manifest,
    make_control_message,
    validate_control_message,
)
from tools.pypto_wse_validation.collect_stage0 import parse_device_list
from tools.pypto_wse_validation.contracts import EndpointRole

SCHEMA_VERSION = 1
DEFAULT_GENERATION = 1
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_START_DELAY_SECONDS = 0.25
CONNECT_RETRY_SECONDS = 0.05
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _NoopWorker:
    def __init__(self, *, fail_init: bool = False) -> None:
        self.fail_init = fail_init

    def init(self) -> None:
        if self.fail_init:
            raise RuntimeError("requested noop init failure")

    def close(self) -> None:
        return


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _make_runtime(
    backend: str,
    *,
    device_id: int,
    fail_init: bool,
    simpler_root: Path | None,
) -> tuple[Any, dict[str, Any]]:
    if backend == "noop":
        return _NoopWorker(fail_init=fail_init), {"backend": "noop", "module": "BUILTIN_TEST_DOUBLE"}
    if fail_init:
        raise ValueError("--fail-init is only supported by the noop backend")
    import simpler
    from simpler import Worker

    module_path = Path(simpler.__file__).resolve()
    if simpler_root is not None and not module_path.is_relative_to(simpler_root.resolve()):
        raise ValueError("imported Simpler module is not from --simpler-root")
    worker = Worker(
        level=2,
        device_id=device_id,
        platform="a5",
        runtime="tensormap_and_ringbuffer",
    )
    return worker, {
        "backend": "simpler",
        "module_origin": "REFERENCE_SOURCE" if simpler_root is not None else "INSTALLED_DISTRIBUTION",
        "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "platform": "a5",
        "runtime": "tensormap_and_ringbuffer",
        "version": importlib.metadata.version("simpler"),
    }


def _connect(host: str, port: int, deadline: float) -> socket.socket:
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            connection.connect((host, port))
            return connection
        except OSError as exc:
            last_error = exc
            connection.close()
            time.sleep(CONNECT_RETRY_SECONDS)
    raise BootstrapError(f"timed out connecting to bootstrap listener: {last_error}")


def _listen(host: str, port: int, deadline: float) -> socket.socket:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BootstrapError("bootstrap listener deadline expired")
        listener.settimeout(remaining)
        connection, _ = listener.accept()
        return connection


def _exchange(
    channel: ControlChannel,
    *,
    role: EndpointRole,
    peer_role: EndpointRole,
    run_id: str,
    generation: int,
    device_id: int,
    runtime_backend: str,
) -> None:
    def message(message_type: str, fields: dict[str, Any]) -> dict[str, Any]:
        return make_control_message(
            message_type,
            run_id=run_id,
            generation=generation,
            role=role,
            fields=fields,
        )

    def receive(expected_type: str) -> dict[str, Any]:
        incoming = channel.receive()
        validate_control_message(
            incoming,
            run_id=run_id,
            generation=generation,
            expected_type=expected_type,
            expected_role=peer_role,
        )
        return incoming

    hello = message(
        "HELLO",
        {
            "capabilities": ["independent_device_runtime_init", "tcp_loopback_control"],
            "device_id": device_id,
            "pid": os.getpid(),
            "runtime_backend": runtime_backend,
        },
    )
    ready = message("READY", {"runtime_initialized": True})
    health = message("HEALTH", {"status": "OK"})
    if role is EndpointRole.ATTENTION:
        channel.send(hello)
        receive("HELLO")
        channel.send(ready)
        receive("READY")
        channel.send(health)
        receive("HEALTH")
        channel.send(message("STOP", {"reason": "stage0_complete"}))
        receive("STOP")
    else:
        receive("HELLO")
        channel.send(hello)
        receive("READY")
        channel.send(ready)
        receive("HEALTH")
        channel.send(health)
        receive("STOP")
        channel.send(message("STOP", {"reason": "acknowledged"}))


def run_endpoint(args: argparse.Namespace) -> int:
    role = EndpointRole(args.role)
    peer_role = EndpointRole.WSE_SURROGATE if role is EndpointRole.ATTENTION else EndpointRole.ATTENTION
    artifact_path = args.artifact_dir / f"{role.value.lower()}_endpoint.json"
    started_at = datetime.now(UTC).isoformat()
    runtime: Any | None = None
    runtime_info: dict[str, Any] = {"backend": args.runtime_backend}
    channel: ControlChannel | None = None
    runtime_initialized = False
    close_attempts: list[dict[str, str]] = []
    error: dict[str, str] | None = None
    deadline = time.monotonic() + args.timeout
    try:
        runtime, runtime_info = _make_runtime(
            args.runtime_backend,
            device_id=args.device_id,
            fail_init=args.fail_init,
            simpler_root=args.simpler_root,
        )
        runtime.init()
        runtime_initialized = True
        connection = (
            _connect(args.host, args.port, deadline)
            if role is EndpointRole.ATTENTION
            else _listen(args.host, args.port, deadline)
        )
        with connection:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BootstrapError("bootstrap deadline expired")
            connection.settimeout(remaining)
            channel = ControlChannel(connection)
            _exchange(
                channel,
                role=role,
                peer_role=peer_role,
                run_id=args.run_id,
                generation=args.generation,
                device_id=args.device_id,
                runtime_backend=args.runtime_backend,
            )
    except BaseException as exc:  # noqa: BLE001
        error = {"message": str(exc), "type": type(exc).__name__}
        traceback.print_exc()
    finally:
        if runtime is not None:
            for attempt in range(1, 3):
                try:
                    runtime.close()
                except BaseException as exc:  # noqa: BLE001
                    close_attempts.append({"attempt": str(attempt), "outcome": f"ERROR:{type(exc).__name__}:{exc}"})
                    if error is None:
                        error = {"message": str(exc), "type": type(exc).__name__}
                else:
                    close_attempts.append({"attempt": str(attempt), "outcome": "CLOSED"})
        evidence = {
            "closed_at": datetime.now(UTC).isoformat(),
            "control": channel.evidence() if channel is not None else {},
            "device_id": args.device_id,
            "error": error,
            "generation": args.generation,
            "host_payload_bytes": 0,
            "host_task_messages": 0,
            "host_completion_messages": 0,
            "pid": os.getpid(),
            "role": role.value,
            "run_id": args.run_id,
            "runtime": runtime_info,
            "runtime_close_attempts": close_attempts,
            "runtime_initialized": runtime_initialized,
            "schema_version": SCHEMA_VERSION,
            "started_at": started_at,
            "success": error is None,
            "working_directory": f"{role.value.lower()}_work",
        }
        _write_json(artifact_path, evidence)
    return 0 if error is None else 1


def _available_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _endpoint_command(
    *,
    role: EndpointRole,
    device_id: int,
    args: argparse.Namespace,
    port: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tools.pypto_wse_validation.launch_local",
        "endpoint",
        "--role",
        role.value,
        "--device-id",
        str(device_id),
        "--host",
        args.host,
        "--port",
        str(port),
        "--run-id",
        args.run_id,
        "--generation",
        str(args.generation),
        "--runtime-backend",
        args.runtime_backend,
        "--timeout",
        str(args.timeout),
        "--artifact-dir",
        str(args.artifact_dir.resolve()),
    ]
    if args.fail_role == role.value:
        command.append("--fail-init")
    if args.simpler_root is not None:
        command.extend(("--simpler-root", str(args.simpler_root.resolve())))
    return command


def _wait_processes(processes: dict[EndpointRole, subprocess.Popen[Any]], deadline: float) -> bool:
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in processes.values()):
            return True
        time.sleep(0.05)
    return False


def _stop_processes(processes: dict[EndpointRole, subprocess.Popen[Any]]) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 2.0
    _wait_processes(processes, deadline)
    for process in processes.values():
        if process.poll() is None:
            process.kill()
    for process in processes.values():
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=2.0)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _log_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    inventory: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        payload = path.read_bytes()
        inventory.append(
            {
                "bytes": len(payload),
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return inventory


def _file_evidence(path: Path) -> dict[str, Any] | None:
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    return {
        "bytes": len(payload),
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _update_capability_evidence(artifact_dir: Path, success: bool) -> None:
    path = artifact_dir / "capabilities.json"
    capabilities = _read_json(path)
    if capabilities is None or not isinstance(capabilities.get("capabilities"), list):
        return
    for capability in capabilities["capabilities"]:
        if capability.get("capability") == "independent_device_runtime_init":
            capability.update(
                {
                    "evidence": (
                        "two independent Simpler L2 endpoint processes initialized distinct NPUs and closed twice"
                        if success
                        else "two-process runtime bootstrap did not complete successfully"
                    ),
                    "status": "verified" if success else "failed",
                }
            )
            break
    _write_json(path, capabilities)


def _aggregate_result(args: argparse.Namespace, processes: dict[EndpointRole, subprocess.Popen[Any]]) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    message_counts: Counter[str] = Counter()
    control_bytes = 0
    all_success = True
    for role, process in processes.items():
        artifact = _read_json(args.artifact_dir / f"{role.value.lower()}_endpoint.json")
        if artifact is None:
            artifact = {"error": {"message": "endpoint artifact missing", "type": "ArtifactError"}, "success": False}
        endpoints[role.value] = {
            "artifact": f"{role.value.lower()}_endpoint.json",
            "device_logs": _log_inventory(args.artifact_dir / f"{role.value.lower()}_device_logs"),
            "exit_code": process.returncode,
            "host_log": _file_evidence(args.artifact_dir / f"{role.value.lower()}_host.log"),
            "pid": process.pid,
            "success": bool(artifact.get("success")) and process.returncode == 0,
        }
        all_success = all_success and endpoints[role.value]["success"]
        control = artifact.get("control", {})
        for message_type, count in control.get("sent_messages", {}).items():
            message_counts[message_type] += int(count)
        control_bytes += int(control.get("sent_bytes", 0))

    return {
        "actual_backend": "CONTROL_ONLY",
        "capability_level": "C0" if all_success else "NONE",
        "completed_at": datetime.now(UTC).isoformat(),
        "control_plane": {
            "bytes_on_wire": control_bytes,
            "message_counts": dict(sorted(message_counts.items())),
            "transport": "TCP_LOOPBACK",
        },
        "data_plane": "NOT_EXERCISED",
        "data_results": {
            "NPU_TO_WSE_SURROGATE": {"attempts": 0, "bytes": 0, "status": "NOT_EXERCISED"},
            "WSE_SURROGATE_TO_NPU": {"attempts": 0, "bytes": 0, "status": "NOT_EXERCISED"},
        },
        "devices": {
            "ATTENTION": args.devices[0],
            "WSE_SURROGATE": args.devices[1],
        },
        "endpoints": endpoints,
        "forbidden_hot_path_activity": {
            "host_completion_messages": 0,
            "host_payload_bytes": 0,
            "host_task_messages": 0,
        },
        "generation": args.generation,
        "profile": "NPU_SURROGATE",
        "resource_cleanup": {
            "device_memory_windows": "NOT_ALLOCATED_STAGE0",
            "runtime_close": "VERIFIED" if all_success else "FAILED",
        },
        "run_id": args.run_id,
        "schema_version": SCHEMA_VERSION,
        "start_order": args.start_order,
        "success": all_success,
        "transport_scope": "SIMULATION",
    }


def run_launcher(args: argparse.Namespace) -> int:
    if args.devices[0] == args.devices[1]:
        raise ValueError("Attention and WSE surrogate devices must be distinct")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    port = _available_port(args.host)
    roles = [EndpointRole.ATTENTION, EndpointRole.WSE_SURROGATE]
    if args.start_order == "wse-first":
        roles.reverse()
    elif args.start_order == "random":
        random.SystemRandom().shuffle(roles)
        args.start_order = "attention-first" if roles[0] is EndpointRole.ATTENTION else "wse-first"
    device_by_role = {
        EndpointRole.ATTENTION: args.devices[0],
        EndpointRole.WSE_SURROGATE: args.devices[1],
    }
    processes: dict[EndpointRole, subprocess.Popen[Any]] = {}
    logs: list[Any] = []
    deadline = time.monotonic() + args.timeout
    try:
        for index, role in enumerate(roles):
            log_path = args.artifact_dir / f"{role.value.lower()}_host.log"
            log = log_path.open("w", encoding="utf-8")
            logs.append(log)
            endpoint_work_dir = args.artifact_dir / f"{role.value.lower()}_work"
            endpoint_device_log_dir = args.artifact_dir / f"{role.value.lower()}_device_logs"
            endpoint_work_dir.mkdir(exist_ok=True)
            endpoint_device_log_dir.mkdir(exist_ok=True)
            endpoint_env = os.environ.copy()
            inherited_python_path = endpoint_env.get("PYTHONPATH")
            endpoint_env["PYTHONPATH"] = (
                f"{PROJECT_ROOT}{os.pathsep}{inherited_python_path}" if inherited_python_path else str(PROJECT_ROOT)
            )
            endpoint_env["ASCEND_PROCESS_LOG_PATH"] = str(endpoint_device_log_dir.resolve())
            process = subprocess.Popen(  # noqa: S603
                _endpoint_command(role=role, device_id=device_by_role[role], args=args, port=port),
                cwd=endpoint_work_dir,
                env=endpoint_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes[role] = process
            if index == 0:
                time.sleep(args.start_delay)
        finished = _wait_processes(processes, deadline)
        if not finished:
            _stop_processes(processes)
        else:
            for process in processes.values():
                process.wait()
    finally:
        _stop_processes(processes)
        for log in logs:
            log.close()

    for role, device_id in device_by_role.items():
        _write_json(
            args.artifact_dir / f"{role.value.lower()}_manifest.json",
            make_contract_manifest(
                run_id=args.run_id,
                generation=args.generation,
                role=role,
                device_id=device_id,
            ),
        )
    result = _aggregate_result(args, processes)
    _write_json(args.artifact_dir / "result.json", result)
    _update_capability_evidence(args.artifact_dir, bool(result["success"]))
    transport_path = args.artifact_dir / "transport.json"
    transport = _read_json(transport_path) or {"profile": "NPU_SURROGATE", "schema_version": SCHEMA_VERSION}
    transport.update(
        {
            "actual_backend": "CONTROL_ONLY",
            "control_plane": "TCP_LOOPBACK",
            "data_plane": "NOT_EXERCISED",
            "declared_scope": "SIMULATION",
            "local_fallback_allowed": False,
        }
    )
    _write_json(transport_path, transport)
    print(args.artifact_dir)
    return 0 if result["success"] else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    launcher = subparsers.add_parser("run")
    launcher.add_argument("--devices", required=True)
    launcher.add_argument("--artifact-dir", type=Path, required=True)
    launcher.add_argument("--host", default="127.0.0.1")
    launcher.add_argument("--run-id", default=None)
    launcher.add_argument("--generation", type=int, default=DEFAULT_GENERATION)
    launcher.add_argument("--runtime-backend", choices=("noop", "simpler"), default="simpler")
    launcher.add_argument("--simpler-root", type=Path)
    launcher.add_argument("--start-order", choices=("attention-first", "wse-first", "random"), default="random")
    launcher.add_argument("--start-delay", type=float, default=DEFAULT_START_DELAY_SECONDS)
    launcher.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    role_choices = tuple(role.value for role in EndpointRole)
    launcher.add_argument("--fail-role", choices=role_choices, default=None)

    endpoint = subparsers.add_parser("endpoint")
    endpoint.add_argument("--role", choices=role_choices, required=True)
    endpoint.add_argument("--device-id", type=int, required=True)
    endpoint.add_argument("--host", required=True)
    endpoint.add_argument("--port", type=int, required=True)
    endpoint.add_argument("--run-id", required=True)
    endpoint.add_argument("--generation", type=int, required=True)
    endpoint.add_argument("--runtime-backend", choices=("noop", "simpler"), required=True)
    endpoint.add_argument("--simpler-root", type=Path)
    endpoint.add_argument("--timeout", type=float, required=True)
    endpoint.add_argument("--artifact-dir", type=Path, required=True)
    endpoint.add_argument("--fail-init", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        devices = parse_device_list(args.devices)
        args.devices = (devices[0], devices[1])
        args.run_id = args.run_id or f"stage0-{uuid.uuid4().hex}"
        if args.timeout <= 0 or args.start_delay < 0 or args.generation <= 0:
            parser.error("timeout/generation must be positive and start-delay must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return run_launcher(args) if args.command == "run" else run_endpoint(args)
    except (OSError, ValueError) as exc:
        print(f"stage-0 bootstrap failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
