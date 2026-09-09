# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Host-control RPC abstraction and the host-local prototype provider.

The pseudo-PyPTO layer never imports this module.  The implementation uses a
spawned Python process and a TCP socket today, but Bootstrap sees only RPC
operations.  A future cross-Host launcher can replace this provider without
changing the service or the Device data-plane ABI.
"""

from __future__ import annotations

import json
import multiprocessing
import queue
import socket
import struct
import time
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from pypto_test.pseudo_pypto.communication import PROTOCOL_VERSION, EndpointRole

MAX_CONTROL_FRAME_BYTES = 64 * 1024
DEFAULT_CONTROL_HOST = "127.0.0.1"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 120.0
_FRAME_LENGTH = struct.Struct("!I")
_FORBIDDEN_CONTROL_KEYS = frozenset({"input", "output", "payload", "tensor", "token"})


class HostControlRpcError(RuntimeError):
    """Raised when the Host-control transport or RPC state is invalid."""


@runtime_checkable
class HostControlRpc(Protocol):
    """Attention-side control capability used by Bootstrap and lifecycle code."""

    def launch_worker(
        self,
        worker: Callable[..., dict[str, Any]],
        *,
        worker_config: dict[str, Any],
        start_order: str,
    ) -> None: ...

    def receive_event(self, event: str) -> dict[str, Any]: ...

    def call(self, method: str, **fields: Any) -> dict[str, Any]: ...

    def close_worker(self) -> dict[str, Any]: ...

    def abort(self) -> None: ...

    def evidence(self) -> dict[str, Any]: ...


@runtime_checkable
class HostControlRpcServer(Protocol):
    """Worker-side counterpart passed to the WSE Host handler."""

    def emit(self, event: str, **fields: Any) -> None: ...

    def receive_call(self) -> tuple[str, dict[str, Any]]: ...

    def reply(self, method: str, **fields: Any) -> None: ...

    def evidence(self) -> dict[str, Any]: ...


class _FramedJsonChannel:
    """Versioned JSON framing used only inside the RPC provider."""

    def __init__(
        self,
        connection: socket.socket,
        *,
        run_id: str,
        generation: int,
        local_role: EndpointRole,
        peer_role: EndpointRole,
    ) -> None:
        self._connection = connection
        self._run_id = run_id
        self._generation = generation
        self._local_role = local_role
        self._peer_role = peer_role
        self._sent_messages: Counter[str] = Counter()
        self._received_messages: Counter[str] = Counter()
        self._sent_bytes = 0
        self._received_bytes = 0

    def send(self, kind: str, name: str, **fields: Any) -> None:
        forbidden = _find_forbidden_keys(fields)
        if forbidden:
            raise HostControlRpcError(f"control RPC contains data-plane keys: {sorted(forbidden)}")
        message = {
            "generation": self._generation,
            "kind": kind,
            "name": name,
            "protocol_version": PROTOCOL_VERSION,
            "role": self._local_role.value,
            "run_id": self._run_id,
            "body": fields,
        }
        encoded = json.dumps(message, separators=(",", ":"), sort_keys=True).encode()
        if len(encoded) > MAX_CONTROL_FRAME_BYTES:
            raise HostControlRpcError("control RPC frame is too large")
        self._connection.sendall(_FRAME_LENGTH.pack(len(encoded)) + encoded)
        self._sent_messages[name] += 1
        self._sent_bytes += len(encoded)

    def receive(self, *, kind: str, name: str | None = None) -> tuple[str, dict[str, Any]]:
        (size,) = _FRAME_LENGTH.unpack(_receive_exact(self._connection, _FRAME_LENGTH.size))
        if size <= 0 or size > MAX_CONTROL_FRAME_BYTES:
            raise HostControlRpcError("invalid control RPC frame size")
        encoded = _receive_exact(self._connection, size)
        try:
            message = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HostControlRpcError("invalid control RPC JSON") from exc
        expected = {
            "generation": self._generation,
            "kind": kind,
            "protocol_version": PROTOCOL_VERSION,
            "role": self._peer_role.value,
            "run_id": self._run_id,
        }
        if not isinstance(message, dict):
            raise HostControlRpcError("control RPC frame must be an object")
        for key, value in expected.items():
            if message.get(key) != value:
                raise HostControlRpcError(f"control RPC {key} mismatch")
        received_name = message.get("name")
        if not isinstance(received_name, str) or (name is not None and received_name != name):
            raise HostControlRpcError(f"control RPC name mismatch: expected {name}, received {received_name}")
        body = message.get("body")
        if not isinstance(body, dict):
            raise HostControlRpcError("control RPC body must be an object")
        forbidden = _find_forbidden_keys(body)
        if forbidden:
            raise HostControlRpcError(f"control RPC contains data-plane keys: {sorted(forbidden)}")
        self._received_messages[received_name] += 1
        self._received_bytes += size
        return received_name, body

    def evidence(self) -> dict[str, Any]:
        return {
            "received_bytes": self._received_bytes,
            "received_messages": dict(sorted(self._received_messages.items())),
            "sent_bytes": self._sent_bytes,
            "sent_messages": dict(sorted(self._sent_messages.items())),
        }

    def close(self) -> None:
        self._connection.close()


class SocketHostControlRpcServer:
    """Worker-side RPC endpoint; request tensors are rejected by framing."""

    def __init__(self, channel: _FramedJsonChannel) -> None:
        self._channel = channel

    def emit(self, event: str, **fields: Any) -> None:
        self._channel.send("event", event, **fields)

    def receive_call(self) -> tuple[str, dict[str, Any]]:
        return self._channel.receive(kind="call")

    def reply(self, method: str, **fields: Any) -> None:
        self._channel.send("reply", method, **fields)

    def evidence(self) -> dict[str, Any]:
        return self._channel.evidence()


class MultiprocessingSocketRpc:
    """Host-local RPC provider backed by ``spawn`` plus one TCP connection."""

    def __init__(self, *, run_id: str, generation: int) -> None:
        self._run_id = run_id
        self._generation = generation
        self._channel: _FramedJsonChannel | None = None
        self._process: Any | None = None
        self._result_queue: Any | None = None
        self._result: dict[str, Any] | None = None

    def launch_worker(
        self,
        worker: Callable[..., dict[str, Any]],
        *,
        worker_config: dict[str, Any],
        start_order: str,
    ) -> None:
        if self._process is not None:
            raise HostControlRpcError("RPC worker is already launched")
        context = multiprocessing.get_context("spawn")
        ready_event = context.Event()
        result_queue = context.Queue()
        listener, port, worker_listens = _prepare_connection(start_order)
        process = context.Process(
            target=_worker_entry,
            kwargs={
                "generation": self._generation,
                "host": DEFAULT_CONTROL_HOST,
                "listens": worker_listens,
                "port": port,
                "ready_event": ready_event,
                "result_queue": result_queue,
                "run_id": self._run_id,
                "worker": worker,
                "worker_config": worker_config,
            },
            name=f"pypto-wse-host-g{self._generation}",
        )
        process.start()
        connection = None
        try:
            if listener is not None:
                connection, _ = listener.accept()
                listener.close()
                listener = None
                connection.settimeout(DEFAULT_CONNECT_TIMEOUT_SECONDS)
            else:
                if not ready_event.wait(DEFAULT_CONNECT_TIMEOUT_SECONDS):
                    raise HostControlRpcError("WSE Host RPC listener did not become ready")
                connection = _connect(DEFAULT_CONTROL_HOST, port)
            self._channel = _FramedJsonChannel(
                connection,
                run_id=self._run_id,
                generation=self._generation,
                local_role=EndpointRole.ATTENTION,
                peer_role=EndpointRole.WSE,
            )
            self._process = process
            self._result_queue = result_queue
        except BaseException:
            if connection is not None:
                connection.close()
            if listener is not None:
                listener.close()
            if process.is_alive():
                process.terminate()
                process.join(10)
            raise

    def receive_event(self, event: str) -> dict[str, Any]:
        return self._require_channel().receive(kind="event", name=event)[1]

    def call(self, method: str, **fields: Any) -> dict[str, Any]:
        channel = self._require_channel()
        channel.send("call", method, **fields)
        return channel.receive(kind="reply", name=method)[1]

    def close_worker(self) -> dict[str, Any]:
        if self._result is not None:
            return dict(self._result)
        if self._process is None or self._result_queue is None:
            raise HostControlRpcError("RPC worker is not launched")
        self._process.join(DEFAULT_CONNECT_TIMEOUT_SECONDS)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(10)
            raise HostControlRpcError("WSE Host RPC worker did not exit")
        try:
            result = self._result_queue.get(timeout=5)
        except queue.Empty as exc:
            raise HostControlRpcError("WSE Host RPC worker returned no evidence") from exc
        if self._process.exitcode != 0 or result.get("status") != "PASS":
            raise HostControlRpcError(f"WSE Host RPC worker failed: {result}")
        self._require_channel().close()
        self._result = result
        return dict(result)

    def abort(self) -> None:
        if self._channel is not None:
            self._channel.close()
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(10)

    def evidence(self) -> dict[str, Any]:
        return self._channel.evidence() if self._channel is not None else {}

    def _require_channel(self) -> _FramedJsonChannel:
        if self._channel is None:
            raise HostControlRpcError("RPC worker is not connected")
        return self._channel


def _worker_entry(
    *,
    worker: Callable[..., dict[str, Any]],
    worker_config: dict[str, Any],
    host: str,
    port: int,
    listens: bool,
    generation: int,
    run_id: str,
    ready_event: Any,
    result_queue: Any,
) -> None:
    """Transport-owned child entry; the worker receives only an RPC server."""

    listener = None
    connection = None
    try:
        if listens:
            listener = _open_listener(host, port)
            ready_event.set()
            connection, _ = listener.accept()
        else:
            connection = _connect(host, port)
            ready_event.set()
        channel = _FramedJsonChannel(
            connection,
            run_id=run_id,
            generation=generation,
            local_role=EndpointRole.WSE,
            peer_role=EndpointRole.ATTENTION,
        )
        details = worker(server=SocketHostControlRpcServer(channel), config=worker_config)
        result_queue.put({**details, "status": "PASS"})
    except BaseException as exc:
        result_queue.put({"error": f"{type(exc).__name__}: {exc}", "status": "FAIL"})
        raise
    finally:
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()


def _prepare_connection(start_order: str) -> tuple[socket.socket | None, int, bool]:
    if start_order == "attention-first":
        listener = _open_listener(DEFAULT_CONTROL_HOST, 0)
        return listener, int(listener.getsockname()[1]), False
    if start_order == "wse-first":
        reservation = _open_listener(DEFAULT_CONTROL_HOST, 0)
        port = int(reservation.getsockname()[1])
        reservation.close()
        return None, port, True
    raise ValueError("start_order must be attention-first or wse-first")


def _open_listener(host: str, port: int) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(1)
    listener.settimeout(DEFAULT_CONNECT_TIMEOUT_SECONDS)
    return listener


def _connect(host: str, port: int) -> socket.socket:
    deadline = time.monotonic() + DEFAULT_CONNECT_TIMEOUT_SECONDS
    while True:
        try:
            connection = socket.create_connection((host, port), timeout=5.0)
            connection.settimeout(DEFAULT_CONNECT_TIMEOUT_SECONDS)
            return connection
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise HostControlRpcError("control RPC connection closed during frame receive")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _find_forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_CONTROL_KEYS:
                found.add(normalized)
            found.update(_find_forbidden_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_find_forbidden_keys(child))
    return found
