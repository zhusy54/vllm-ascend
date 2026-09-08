# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Run stage-1A T01/T02 over host-local ACL VMM P2P."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.pypto_wse_validation.acl_vmm import AclVmmRuntime, VmmExport
from tools.pypto_wse_validation.bootstrap import BootstrapError, ControlChannel
from tools.pypto_wse_validation.collect_stage0 import parse_device_list
from tools.pypto_wse_validation.contracts import PROTOCOL_VERSION, EndpointRole, TransportScope
from tools.pypto_wse_validation.stage1a_contracts import (
    BASE_PAYLOAD_SIZES,
    STAGE1A_BACKEND,
    STAGE1A_FENCE_API,
    STAGE1A_HANDLE_KIND,
    STAGE1A_TRANSFER_API,
    MemoryKind,
    TransferDirection,
    TransferObservation,
    deterministic_payload,
    opaque_handle_evidence,
    payload_checksum,
    stage1a_matrix_complete,
)

SCHEMA_VERSION = 1
DEFAULT_GENERATION = 1
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_START_DELAY_SECONDS = 0.25
CONNECT_RETRY_SECONDS = 0.05
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_MESSAGE_TYPES = frozenset(
    {
        "MANIFEST",
        "ATTACHED",
        "READY",
        "TRANSFER_COMPLETE",
        "VERIFIED",
        "DRAIN",
        "DETACHED",
        "RELEASED",
    }
)
FORBIDDEN_DATA_KEYS = frozenset({"payload", "tensor", "output", "host_buffer"})


class Stage1AError(RuntimeError):
    """Raised when the stage-1A control or transfer protocol is violated."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in FORBIDDEN_DATA_KEYS or _contains_forbidden_key(child) for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _message(
    message_type: str,
    *,
    run_id: str,
    generation: int,
    role: EndpointRole,
    fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if message_type not in ALLOWED_MESSAGE_TYPES:
        raise Stage1AError(f"unsupported stage-1A message type: {message_type}")
    message: dict[str, Any] = {
        "generation": generation,
        "protocol_version": PROTOCOL_VERSION,
        "role": role.value,
        "run_id": run_id,
        "type": message_type,
    }
    if fields:
        overlap = set(message).intersection(fields)
        if overlap:
            raise Stage1AError(f"reserved control fields cannot be replaced: {sorted(overlap)}")
        message.update(fields)
    if _contains_forbidden_key(message):
        raise Stage1AError("stage-1A control message contains payload data")
    return message


def _validate_message(
    message: Mapping[str, Any],
    *,
    expected_type: str,
    expected_role: EndpointRole,
    run_id: str,
    generation: int,
) -> None:
    if message.get("type") != expected_type:
        raise Stage1AError(f"expected {expected_type}, received {message.get('type')!r}")
    if message.get("type") not in ALLOWED_MESSAGE_TYPES:
        raise Stage1AError("received unsupported control message")
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise Stage1AError("protocol_version mismatch")
    if message.get("role") != expected_role.value:
        raise Stage1AError("peer role mismatch")
    if message.get("run_id") != run_id or message.get("generation") != generation:
        raise Stage1AError("run identity mismatch")
    if _contains_forbidden_key(message):
        raise Stage1AError("received payload data on the control plane")


class _Protocol:
    def __init__(
        self,
        channel: ControlChannel,
        *,
        role: EndpointRole,
        run_id: str,
        generation: int,
    ) -> None:
        self.channel = channel
        self.role = role
        self.peer_role = EndpointRole.WSE_SURROGATE if role is EndpointRole.ATTENTION else EndpointRole.ATTENTION
        self.run_id = run_id
        self.generation = generation

    def send(self, message_type: str, fields: Mapping[str, Any] | None = None) -> None:
        self.channel.send(
            _message(
                message_type,
                run_id=self.run_id,
                generation=self.generation,
                role=self.role,
                fields=fields,
            )
        )

    def receive(self, expected_type: str) -> dict[str, Any]:
        message = self.channel.receive()
        _validate_message(
            message,
            expected_type=expected_type,
            expected_role=self.peer_role,
            run_id=self.run_id,
            generation=self.generation,
        )
        return message

    def exchange(self, message_type: str, fields: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.role is EndpointRole.ATTENTION:
            self.send(message_type, fields)
            return self.receive(message_type)
        message = self.receive(message_type)
        self.send(message_type, fields)
        return message


def _manifest(runtime: AclVmmRuntime, window: Any) -> dict[str, Any]:
    return {
        "backend": STAGE1A_BACKEND,
        "device_id": runtime.device_id,
        "handle_kind": STAGE1A_HANDLE_KIND,
        "logical_bytes": window.logical_bytes,
        "mapping_bytes": window.mapping_bytes,
        "shareable_handle": window.shareable_handle,
        "transport_scope": TransportScope.HOST_LOCAL.value,
    }


def _parse_peer_export(message: Mapping[str, Any]) -> tuple[VmmExport, int]:
    manifest = message.get("window")
    if not isinstance(manifest, Mapping):
        raise Stage1AError("MANIFEST is missing a window descriptor")
    if manifest.get("backend") != STAGE1A_BACKEND:
        raise Stage1AError("peer backend mismatch")
    if manifest.get("handle_kind") != STAGE1A_HANDLE_KIND:
        raise Stage1AError("peer handle kind mismatch")
    if manifest.get("transport_scope") != TransportScope.HOST_LOCAL.value:
        raise Stage1AError("peer transport scope mismatch")
    try:
        device_id = int(manifest["device_id"])
        logical_bytes = int(manifest["logical_bytes"])
        mapping_bytes = int(manifest["mapping_bytes"])
        shareable_handle = int(manifest["shareable_handle"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage1AError(f"invalid peer window descriptor: {exc}") from exc
    if device_id < 0 or logical_bytes < max(BASE_PAYLOAD_SIZES) or mapping_bytes < logical_bytes:
        raise Stage1AError("peer window dimensions are invalid")
    if shareable_handle <= 0:
        raise Stage1AError("peer shareable handle is invalid")
    return VmmExport(device_id, mapping_bytes, shareable_handle), device_id


def _transfer_observation(
    *,
    direction: TransferDirection,
    size: int,
    sequence_id: int,
    expected_checksum: str,
    observed_checksum: str,
    verified: bool,
    elapsed_ns: int,
) -> TransferObservation:
    return TransferObservation(
        case_id=direction.case_id,
        direction=direction,
        payload_bytes=size,
        sequence_id=sequence_id,
        source_memory=MemoryKind.DEVICE,
        destination_memory=MemoryKind.DEVICE,
        backend=STAGE1A_BACKEND,
        transport_scope=TransportScope.HOST_LOCAL,
        handle_kind=STAGE1A_HANDLE_KIND,
        transfer_api=STAGE1A_TRANSFER_API,
        visibility_fence=STAGE1A_FENCE_API,
        expected_checksum=expected_checksum,
        observed_checksum=observed_checksum,
        host_bounce_bytes=0,
        host_source_staging_bytes=size,
        host_verification_bytes=size,
        fallback_used=False,
        source_filled=True,
        destination_verified=verified,
        elapsed_ns=elapsed_ns,
    )


def execute_transfer_matrix(
    protocol: _Protocol,
    *,
    runtime: Any,
    local_window: Any,
    peer_window: Any,
) -> tuple[TransferObservation, ...]:
    """Execute this endpoint's half of T01/T02 and return initiated observations."""
    observations: list[TransferObservation] = []
    sequence_id = 0
    for direction in TransferDirection:
        initiator_role = (
            EndpointRole.ATTENTION
            if direction is TransferDirection.NPU_TO_WSE_SURROGATE
            else EndpointRole.WSE_SURROGATE
        )
        for size in BASE_PAYLOAD_SIZES:
            sequence_id += 1
            expected = deterministic_payload(protocol.run_id, protocol.generation, sequence_id, size)
            expected_checksum = payload_checksum(expected)
            if protocol.role is initiator_role:
                runtime.copy_host_to_device(local_window.address, expected)
                started_ns = time.perf_counter_ns()
                runtime.copy_device_to_device(peer_window.address, local_window.address, size)
                elapsed_ns = time.perf_counter_ns() - started_ns
                protocol.send(
                    "TRANSFER_COMPLETE",
                    {
                        "case_id": direction.case_id,
                        "direction": direction.value,
                        "elapsed_ns": elapsed_ns,
                        "expected_checksum": expected_checksum,
                        "payload_bytes": size,
                        "sequence_id": sequence_id,
                    },
                )
                verified = protocol.receive("VERIFIED")
                if verified.get("sequence_id") != sequence_id or verified.get("case_id") != direction.case_id:
                    raise Stage1AError("verification identity mismatch")
                observed_checksum = str(verified.get("observed_checksum", ""))
                destination_verified = verified.get("destination_verified") is True
                observation = _transfer_observation(
                    direction=direction,
                    size=size,
                    sequence_id=sequence_id,
                    expected_checksum=expected_checksum,
                    observed_checksum=observed_checksum,
                    verified=destination_verified,
                    elapsed_ns=elapsed_ns,
                )
                observations.append(observation)
                if not observation.passed:
                    raise Stage1AError(f"{direction.case_id} failed for {size} bytes")
            else:
                completed = protocol.receive("TRANSFER_COMPLETE")
                expected_fields = {
                    "case_id": direction.case_id,
                    "direction": direction.value,
                    "payload_bytes": size,
                    "sequence_id": sequence_id,
                    "expected_checksum": expected_checksum,
                }
                if any(completed.get(key) != value for key, value in expected_fields.items()):
                    raise Stage1AError("transfer metadata mismatch")
                observed = runtime.copy_device_to_host(local_window.address, size)
                observed_checksum = payload_checksum(observed)
                protocol.send(
                    "VERIFIED",
                    {
                        "case_id": direction.case_id,
                        "destination_verified": observed == expected,
                        "observed_checksum": observed_checksum,
                        "payload_bytes": size,
                        "sequence_id": sequence_id,
                    },
                )
    return tuple(observations)


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
    raise BootstrapError(f"timed out connecting to stage-1A listener: {last_error}")


def _listen(host: str, port: int, deadline: float) -> socket.socket:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        listener.settimeout(max(0.001, deadline - time.monotonic()))
        connection, _ = listener.accept()
        return connection


def run_endpoint(args: argparse.Namespace) -> int:
    role = EndpointRole(args.role)
    artifact_path = args.artifact_dir / f"{role.value.lower()}_stage1a.json"
    started_at = datetime.now(UTC).isoformat()
    runtime: AclVmmRuntime | None = None
    local_window: Any | None = None
    peer_window: Any | None = None
    channel: ControlChannel | None = None
    observations: tuple[TransferObservation, ...] = ()
    sanitized_manifest: dict[str, Any] | None = None
    cleanup: dict[str, str] = {
        "imported_window": "NOT_CREATED",
        "owned_window": "NOT_CREATED",
        "runtime": "NOT_INITIALIZED",
    }
    error: dict[str, str] | None = None
    deadline = time.monotonic() + args.timeout
    try:
        runtime = AclVmmRuntime(args.device_id, access_device_id=args.access_device_id)
        runtime.initialize()
        cleanup["runtime"] = "OPEN"
        local_window = runtime.allocate_window(max(BASE_PAYLOAD_SIZES))
        cleanup["owned_window"] = "OPEN"
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
                raise Stage1AError("peer must use a distinct device")
            peer_window = runtime.import_window(peer_export, peer_device_id=peer_device_id)
            cleanup["imported_window"] = "OPEN"
            protocol.exchange("ATTACHED", {"peer_mapping_bytes": peer_window.mapping_bytes})
            protocol.exchange("READY", {"probe": "T01_T02"})
            observations = execute_transfer_matrix(
                protocol,
                runtime=runtime,
                local_window=local_window,
                peer_window=peer_window,
            )
            protocol.exchange("DRAIN", {"inflight": 0})
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
            "error": error,
            "generation": args.generation,
            "host_bounce_bytes": 0,
            "manifest": sanitized_manifest,
            "observations": [item.to_dict() for item in observations],
            "pid": os.getpid(),
            "role": role.value,
            "run_id": args.run_id,
            "schema_version": SCHEMA_VERSION,
            "started_at": started_at,
            "success": error is None,
        }
        _write_json(artifact_path, evidence)
    return 0 if error is None else 1


def _available_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _endpoint_command(role: EndpointRole, device_id: int, access_device_id: int, args: argparse.Namespace, port: int):
    return [
        sys.executable,
        "-m",
        "tools.pypto_wse_validation.stage1a",
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
    ]


def _wait_processes(processes: Mapping[EndpointRole, subprocess.Popen[Any]], deadline: float) -> bool:
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in processes.values()):
            return True
        time.sleep(0.05)
    return False


def _stop_processes(processes: Mapping[EndpointRole, subprocess.Popen[Any]]) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    _wait_processes(processes, time.monotonic() + 2.0)
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


def _file_evidence(path: Path) -> dict[str, Any] | None:
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    return {"bytes": len(payload), "path": path.name, "sha256": hashlib.sha256(payload).hexdigest()}


def _aggregate_result(args: argparse.Namespace, processes: Mapping[EndpointRole, subprocess.Popen[Any]]):
    endpoints: dict[str, Any] = {}
    observations: list[TransferObservation] = []
    message_counts: Counter[str] = Counter()
    control_bytes = 0
    all_success = True
    for role, process in processes.items():
        artifact_name = f"{role.value.lower()}_stage1a.json"
        artifact = _read_json(args.artifact_dir / artifact_name)
        if artifact is None:
            artifact = {"success": False}
        endpoint_success = bool(artifact.get("success")) and process.returncode == 0
        all_success = all_success and endpoint_success
        endpoints[role.value] = {
            "artifact": artifact_name,
            "exit_code": process.returncode,
            "host_log": _file_evidence(args.artifact_dir / f"{role.value.lower()}_stage1a.log"),
            "pid": process.pid,
            "success": endpoint_success,
        }
        control = artifact.get("control", {})
        control_bytes += int(control.get("sent_bytes", 0))
        for message_type, count in control.get("sent_messages", {}).items():
            message_counts[message_type] += int(count)
        for raw in artifact.get("observations", []):
            observations.append(TransferObservation.from_dict(raw))
    matrix_complete = all_success and stage1a_matrix_complete(tuple(observations))
    direction_results: dict[str, Any] = {}
    for direction in TransferDirection:
        selected = [item for item in observations if item.direction is direction]
        direction_results[direction.value] = {
            "attempts": len(selected),
            "bytes": sum(item.payload_bytes for item in selected),
            "passed": sum(item.passed for item in selected),
            "status": "PASS"
            if len(selected) == len(BASE_PAYLOAD_SIZES) and all(item.passed for item in selected)
            else "FAIL",
        }
    return {
        "actual_backend": STAGE1A_BACKEND,
        "capability_level": "C1" if matrix_complete else "NONE",
        "claim_scope": "NPU_SURROGATE_ONLY",
        "completed_at": datetime.now(UTC).isoformat(),
        "control_plane": {
            "bytes_on_wire": control_bytes,
            "message_counts": dict(sorted(message_counts.items())),
            "transport": "TCP_LOOPBACK",
        },
        "data_results": direction_results,
        "devices": {"ATTENTION": args.devices[0], "WSE_SURROGATE": args.devices[1]},
        "endpoints": endpoints,
        "evidence_status": "SIMULATION",
        "fallback_used": False,
        "generation": args.generation,
        "host_bounce_bytes": 0,
        "host_source_staging_bytes": sum(item.host_source_staging_bytes for item in observations),
        "host_verification_bytes": sum(item.host_verification_bytes for item in observations),
        "npu_wse_capability_level": "NOT_ESTABLISHED",
        "profile": "NPU_SURROGATE",
        "run_id": args.run_id,
        "schema_version": SCHEMA_VERSION,
        "start_order": args.start_order,
        "success": matrix_complete,
        "transfer_api": STAGE1A_TRANSFER_API,
        "transport_identity_evidence": [
            "aclrtDeviceEnablePeerAccess returned success",
            "ACL VMM shareable handle exported and imported",
            "aclrtMemcpy used ACL_MEMCPY_DEVICE_TO_DEVICE",
        ],
        "transport_scope": TransportScope.HOST_LOCAL.value,
    }


def run_launcher(args: argparse.Namespace) -> int:
    if args.devices[0] == args.devices[1]:
        raise ValueError("stage-1A requires two distinct devices")
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
            log = (args.artifact_dir / f"{role.value.lower()}_stage1a.log").open("w", encoding="utf-8")
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
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--run-id", default=f"stage1a-{uuid.uuid4().hex}")
    run.add_argument("--generation", type=int, default=DEFAULT_GENERATION)
    run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    run.add_argument("--start-order", choices=("attention-first", "wse-first", "random"), default="random")
    run.add_argument("--start-delay", type=float, default=DEFAULT_START_DELAY_SECONDS)

    endpoint = subparsers.add_parser("endpoint")
    endpoint.add_argument("--role", choices=tuple(role.value for role in EndpointRole), required=True)
    endpoint.add_argument("--device-id", type=int, required=True)
    endpoint.add_argument("--access-device-id", type=int, required=True)
    endpoint.add_argument("--host", required=True)
    endpoint.add_argument("--port", type=int, required=True)
    endpoint.add_argument("--run-id", required=True)
    endpoint.add_argument("--generation", type=int, required=True)
    endpoint.add_argument("--timeout", type=float, required=True)
    endpoint.add_argument("--artifact-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        args.access_device_ids = args.access_device_ids or args.devices
        return run_launcher(args)
    return run_endpoint(args)


if __name__ == "__main__":
    raise SystemExit(main())
