# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Run stage-1A T01/T02/T03 over host-local ACL VMM P2P."""

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

from tools.pypto_wse_validation.acl_vmm import AclnnXorTransform, AclVmmRuntime, VmmExport
from tools.pypto_wse_validation.bootstrap import BootstrapError, ControlChannel
from tools.pypto_wse_validation.collect_stage0 import parse_device_list
from tools.pypto_wse_validation.contracts import PROTOCOL_VERSION, EndpointRole, TransportScope
from tools.pypto_wse_validation.stage1a_contracts import (
    BASE_PAYLOAD_SIZES,
    ROUND_TRIP_CASE_ID,
    ROUND_TRIP_TRANSFORM_API,
    STAGE1A_BACKEND,
    STAGE1A_FENCE_API,
    STAGE1A_HANDLE_KIND,
    STAGE1A_TRANSFER_API,
    VALIDATION_P2P_CHUNK_BYTES,
    MemoryKind,
    RoundTripObservation,
    TransferDirection,
    TransferObservation,
    deterministic_payload,
    opaque_handle_evidence,
    payload_checksum,
    stage1a_matrix_complete,
    stage1a_round_trip_complete,
    transform_payload,
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
        "ROUND_TRIP_FORWARD",
        "ROUND_TRIP_RETURN",
        "ROUND_TRIP_VERIFIED",
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
    transfer_chunks: int,
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
        transfer_chunks=transfer_chunks,
        max_transfer_chunk_bytes=min(size, VALIDATION_P2P_CHUNK_BYTES),
    )


def execute_transfer_matrix(
    protocol: _Protocol,
    *,
    runtime: Any,
    local_window: Any,
    peer_window: Any,
    observation_sink: list[TransferObservation] | None = None,
) -> tuple[TransferObservation, ...]:
    """Execute this endpoint's half of T01/T02 and return initiated observations."""
    observations = observation_sink if observation_sink is not None else []
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
                transfer_chunks = runtime.copy_device_to_device(peer_window.address, local_window.address, size)
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
                        "transfer_chunks": transfer_chunks,
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
                    transfer_chunks=transfer_chunks,
                )
                observations.append(observation)
                if not observation.passed:
                    raise Stage1AError(
                        f"{direction.case_id} failed for {size} bytes: "
                        f"expected_sha256={expected_checksum[:12]}, observed_sha256={observed_checksum[:12]}"
                    )
            else:
                completed = protocol.receive("TRANSFER_COMPLETE")
                expected_fields = {
                    "case_id": direction.case_id,
                    "direction": direction.value,
                    "payload_bytes": size,
                    "sequence_id": sequence_id,
                    "expected_checksum": expected_checksum,
                    "transfer_chunks": (size + VALIDATION_P2P_CHUNK_BYTES - 1) // VALIDATION_P2P_CHUNK_BYTES,
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


def execute_round_trip_matrix(
    protocol: _Protocol,
    *,
    runtime: Any,
    local_window: Any,
    peer_window: Any,
    transform: Any | None,
    observation_sink: list[RoundTripObservation] | None = None,
) -> tuple[RoundTripObservation, ...]:
    """Execute T03 with a device-side transform and no Host intermediate payload."""
    observations = observation_sink if observation_sink is not None else []
    for index, size in enumerate(BASE_PAYLOAD_SIZES, start=1):
        sequence_id = len(TransferDirection) * len(BASE_PAYLOAD_SIZES) + index
        payload = deterministic_payload(protocol.run_id, protocol.generation, sequence_id, size)
        expected_output = transform_payload(payload, sequence_id)
        input_checksum = payload_checksum(payload)
        expected_output_checksum = payload_checksum(expected_output)
        expected_chunks = (size + VALIDATION_P2P_CHUNK_BYTES - 1) // VALIDATION_P2P_CHUNK_BYTES
        if protocol.role is EndpointRole.ATTENTION:
            runtime.copy_host_to_device(local_window.address, payload)
            round_trip_started_ns = time.perf_counter_ns()
            forward_started_ns = time.perf_counter_ns()
            forward_chunks = runtime.copy_device_to_device(peer_window.address, local_window.address, size)
            forward_elapsed_ns = time.perf_counter_ns() - forward_started_ns
            protocol.send(
                "ROUND_TRIP_FORWARD",
                {
                    "case_id": ROUND_TRIP_CASE_ID,
                    "forward_elapsed_ns": forward_elapsed_ns,
                    "forward_transfer_chunks": forward_chunks,
                    "input_checksum": input_checksum,
                    "payload_bytes": size,
                    "sequence_id": sequence_id,
                    "transform_value": sequence_id & 0xFF,
                },
            )
            returned = protocol.receive("ROUND_TRIP_RETURN")
            expected_fields = {
                "case_id": ROUND_TRIP_CASE_ID,
                "payload_bytes": size,
                "sequence_id": sequence_id,
                "transform_api": ROUND_TRIP_TRANSFORM_API,
                "transform_value": sequence_id & 0xFF,
                "return_transfer_chunks": expected_chunks,
            }
            if any(returned.get(key) != value for key, value in expected_fields.items()):
                raise Stage1AError("round-trip return metadata mismatch")
            observed_output = runtime.copy_device_to_host(local_window.address, size)
            observed_output_checksum = payload_checksum(observed_output)
            round_trip_elapsed_ns = time.perf_counter_ns() - round_trip_started_ns
            observation = RoundTripObservation(
                case_id=ROUND_TRIP_CASE_ID,
                payload_bytes=size,
                sequence_id=sequence_id,
                source_memory=MemoryKind.DEVICE,
                transform_memory=MemoryKind.DEVICE,
                destination_memory=MemoryKind.DEVICE,
                backend=STAGE1A_BACKEND,
                transport_scope=TransportScope.HOST_LOCAL,
                handle_kind=STAGE1A_HANDLE_KIND,
                transfer_api=STAGE1A_TRANSFER_API,
                visibility_fence=STAGE1A_FENCE_API,
                transform_api=str(returned["transform_api"]),
                transform_value=sequence_id & 0xFF,
                input_checksum=input_checksum,
                expected_output_checksum=expected_output_checksum,
                observed_output_checksum=observed_output_checksum,
                host_bounce_bytes=0,
                host_intermediate_payload_bytes=0,
                host_source_staging_bytes=size,
                host_final_verification_bytes=size,
                fallback_used=False,
                transform_device_side=True,
                forward_transfer_chunks=forward_chunks,
                return_transfer_chunks=int(returned["return_transfer_chunks"]),
                max_transfer_chunk_bytes=min(size, VALIDATION_P2P_CHUNK_BYTES),
                forward_elapsed_ns=forward_elapsed_ns,
                transform_elapsed_ns=int(returned["transform_elapsed_ns"]),
                return_elapsed_ns=int(returned["return_elapsed_ns"]),
                round_trip_elapsed_ns=round_trip_elapsed_ns,
                transform_workspace_bytes=int(returned["transform_workspace_bytes"]),
            )
            observations.append(observation)
            protocol.send(
                "ROUND_TRIP_VERIFIED",
                {
                    "case_id": ROUND_TRIP_CASE_ID,
                    "destination_verified": observed_output == expected_output,
                    "observed_output_checksum": observed_output_checksum,
                    "payload_bytes": size,
                    "sequence_id": sequence_id,
                },
            )
            if not observation.passed or observed_output != expected_output:
                raise Stage1AError(
                    f"T03 failed for {size} bytes: expected_sha256={expected_output_checksum[:12]}, "
                    f"observed_sha256={observed_output_checksum[:12]}"
                )
        else:
            forwarded = protocol.receive("ROUND_TRIP_FORWARD")
            expected_fields = {
                "case_id": ROUND_TRIP_CASE_ID,
                "forward_transfer_chunks": expected_chunks,
                "input_checksum": input_checksum,
                "payload_bytes": size,
                "sequence_id": sequence_id,
                "transform_value": sequence_id & 0xFF,
            }
            if any(forwarded.get(key) != value for key, value in expected_fields.items()):
                raise Stage1AError("round-trip forward metadata mismatch")
            if transform is None:
                raise Stage1AError("WSE surrogate requires a device transform")
            transform_evidence = transform.apply(local_window.address, size, sequence_id & 0xFF)
            return_started_ns = time.perf_counter_ns()
            return_chunks = runtime.copy_device_to_device(peer_window.address, local_window.address, size)
            return_elapsed_ns = time.perf_counter_ns() - return_started_ns
            protocol.send(
                "ROUND_TRIP_RETURN",
                {
                    "case_id": ROUND_TRIP_CASE_ID,
                    "payload_bytes": size,
                    "return_elapsed_ns": return_elapsed_ns,
                    "return_transfer_chunks": return_chunks,
                    "sequence_id": sequence_id,
                    "transform_api": transform_evidence.api,
                    "transform_elapsed_ns": transform_evidence.elapsed_ns,
                    "transform_value": sequence_id & 0xFF,
                    "transform_workspace_bytes": transform_evidence.workspace_bytes,
                },
            )
            verified = protocol.receive("ROUND_TRIP_VERIFIED")
            expected_verification = {
                "case_id": ROUND_TRIP_CASE_ID,
                "destination_verified": True,
                "observed_output_checksum": expected_output_checksum,
                "payload_bytes": size,
                "sequence_id": sequence_id,
            }
            if any(verified.get(key) != value for key, value in expected_verification.items()):
                raise Stage1AError("round-trip verification failed")
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
    observations: list[TransferObservation] = []
    round_trips: list[RoundTripObservation] = []
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
            protocol.exchange("READY", {"probe": "T01_T02_T03"})
            execute_transfer_matrix(
                protocol,
                runtime=runtime,
                local_window=local_window,
                peer_window=peer_window,
                observation_sink=observations,
            )
            transform = AclnnXorTransform(runtime) if role is EndpointRole.WSE_SURROGATE else None
            execute_round_trip_matrix(
                protocol,
                runtime=runtime,
                local_window=local_window,
                peer_window=peer_window,
                transform=transform,
                observation_sink=round_trips,
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
            "round_trips": [item.to_dict() for item in round_trips],
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


def _device_log_evidence(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    p2p_enable_observed = False
    p2p_memory_released = False
    if root.is_dir():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            payload = path.read_bytes()
            files.append(
                {
                    "bytes": len(payload),
                    "path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            p2p_enable_observed = p2p_enable_observed or b"Enable P2P" in payload
            p2p_memory_released = p2p_memory_released or (
                b"P2P_HBM" in payload and b"current_alloced_size=0" in payload
            )
    return {
        "files": files,
        "p2p_enable_observed": p2p_enable_observed,
        "p2p_memory_released": p2p_memory_released,
    }


def _aggregate_result(args: argparse.Namespace, processes: Mapping[EndpointRole, subprocess.Popen[Any]]):
    endpoints: dict[str, Any] = {}
    observations: list[TransferObservation] = []
    round_trips: list[RoundTripObservation] = []
    message_counts: Counter[str] = Counter()
    control_bytes = 0
    all_success = True
    all_cleanup = True
    driver_proofs: list[bool] = []
    for role, process in processes.items():
        artifact_name = f"{role.value.lower()}_stage1a.json"
        artifact = _read_json(args.artifact_dir / artifact_name)
        if artifact is None:
            artifact = {"success": False}
        endpoint_success = bool(artifact.get("success")) and process.returncode == 0
        all_success = all_success and endpoint_success
        cleanup = artifact.get("cleanup", {})
        cleanup_complete = cleanup == {
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
        for raw in artifact.get("round_trips", []):
            round_trips.append(RoundTripObservation.from_dict(raw))
    matrix_complete = (
        all_success
        and all_cleanup
        and all(driver_proofs)
        and stage1a_matrix_complete(tuple(observations))
        and stage1a_round_trip_complete(tuple(round_trips))
    )
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
    direction_results[ROUND_TRIP_CASE_ID] = {
        "attempts": len(round_trips),
        "bytes": sum(item.payload_bytes for item in round_trips),
        "passed": sum(item.passed for item in round_trips),
        "status": (
            "PASS"
            if len(round_trips) == len(BASE_PAYLOAD_SIZES) and all(item.passed for item in round_trips)
            else "FAIL"
        ),
        "wire_bytes": 2 * sum(item.payload_bytes for item in round_trips),
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
        "host_final_verification_bytes": sum(item.host_final_verification_bytes for item in round_trips),
        "host_intermediate_payload_bytes": sum(item.host_intermediate_payload_bytes for item in round_trips),
        "host_source_staging_bytes": sum(item.host_source_staging_bytes for item in observations)
        + sum(item.host_source_staging_bytes for item in round_trips),
        "host_verification_bytes": sum(item.host_verification_bytes for item in observations)
        + sum(item.host_final_verification_bytes for item in round_trips),
        "npu_wse_capability_level": "NOT_ESTABLISHED",
        "profile": "NPU_SURROGATE",
        "resource_cleanup": "VERIFIED" if all_cleanup else "FAILED",
        "run_id": args.run_id,
        "schema_version": SCHEMA_VERSION,
        "start_order": args.start_order,
        "success": matrix_complete,
        "transfer_api": STAGE1A_TRANSFER_API,
        "max_transfer_chunk_bytes": VALIDATION_P2P_CHUNK_BYTES,
        "transport_identity_evidence": (
            [
                "aclrtDeviceEnablePeerAccess returned success",
                "ACL VMM shareable handle exported and imported",
                "aclrtMemcpy used ACL_MEMCPY_DEVICE_TO_DEVICE",
                f"{ROUND_TRIP_TRANSFORM_API} transformed WSE-surrogate Device Memory in place",
                "driver logs contain Enable P2P and released P2P_HBM counters",
            ]
            if matrix_complete
            else []
        ),
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
