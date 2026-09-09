# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Two-process launcher for the host-local NPU/WSE-surrogate prototype."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import queue
import socket
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from pypto_test.backend import NpuSurrogateBackend
from pypto_test.bootstrap import (
    BootstrapManager,
    NpuDeviceMemoryManager,
    ProxyControlChannel,
    SurrogateDeviceMemoryManager,
    accept_surrogate_bootstrap,
)
from pypto_test.contracts import EndpointRole, deterministic_input, expected_abc
from pypto_test.service import PseudoPyptoDistributedService

DEFAULT_CONTROL_HOST = "127.0.0.1"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 120.0


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


def _surrogate_process(
    *,
    host: str,
    port: int,
    listens: bool,
    device_id: int,
    generation: int,
    run_id: str,
    kernel_binary: str,
    ready_event: Any,
    result_queue: Any,
) -> None:
    listener = None
    connection = None
    memory_manager = None
    backend = None
    try:
        if listens:
            listener = _open_listener(host, port)
            ready_event.set()
            connection, _ = listener.accept()
            connection.settimeout(DEFAULT_CONNECT_TIMEOUT_SECONDS)
        else:
            connection = _connect(host, port)
            ready_event.set()
        channel = ProxyControlChannel(connection, run_id=run_id, generation=generation)
        memory_manager = SurrogateDeviceMemoryManager(
            endpoint_id="wse-surrogate",
            device_id=device_id,
            generation=generation,
        )
        resources = accept_surrogate_bootstrap(channel=channel, memory_manager=memory_manager)
        message = channel.receive("START", EndpointRole.ATTENTION)
        del message
        backend = NpuSurrogateBackend(
            execution_port=resources.execution_port,
            local_window=resources.local_window,
            npu_peer_window=resources.npu_peer_window,
            kernel_binary=Path(kernel_binary),
        )
        ready = backend.initialize()
        channel.send("READY", EndpointRole.WSE_SURROGATE, details=ready)
        drain_evidence: dict[str, Any] | None = None
        while True:
            message = channel.receive_any(EndpointRole.ATTENTION)
            message_type = message["type"]
            if message_type == "HEALTH":
                channel.send("HEALTH_REPLY", EndpointRole.WSE_SURROGATE, details=backend.health())
            elif message_type == "DRAIN":
                drain_evidence = backend.drain()
                channel.send("DRAINED", EndpointRole.WSE_SURROGATE, details=drain_evidence)
            elif message_type == "CLOSE":
                backend.close()
                channel.send("CLOSED", EndpointRole.WSE_SURROGATE)
            elif message_type == "RELEASE":
                memory_manager.release()
                channel.send("RELEASED", EndpointRole.WSE_SURROGATE)
                result_queue.put(
                    {
                        "backend": drain_evidence,
                        "control": channel.evidence(),
                        "memory": memory_manager.evidence(),
                        "status": "PASS",
                    }
                )
                break
            else:
                raise RuntimeError(f"unexpected surrogate control message: {message_type}")
    except BaseException as exc:
        result_queue.put({"error": f"{type(exc).__name__}: {exc}", "status": "FAIL"})
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException:
                pass
        if memory_manager is not None:
            try:
                memory_manager.release()
            except BaseException:
                pass
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()


def _connected_surrogate(
    *,
    start_order: str,
    device_id: int,
    generation: int,
    run_id: str,
    kernel_binary: Path,
) -> tuple[socket.socket, multiprocessing.Process, Any]:
    context = multiprocessing.get_context("spawn")
    ready_event = context.Event()
    result_queue = context.Queue()
    if start_order == "attention-first":
        listener = _open_listener(DEFAULT_CONTROL_HOST, 0)
        port = listener.getsockname()[1]
        process = context.Process(
            target=_surrogate_process,
            kwargs={
                "device_id": device_id,
                "generation": generation,
                "host": DEFAULT_CONTROL_HOST,
                "kernel_binary": str(kernel_binary),
                "listens": False,
                "port": port,
                "ready_event": ready_event,
                "result_queue": result_queue,
                "run_id": run_id,
            },
            name=f"pypto-wse-surrogate-g{generation}",
        )
        process.start()
        connection, _ = listener.accept()
        listener.close()
        connection.settimeout(DEFAULT_CONNECT_TIMEOUT_SECONDS)
    elif start_order == "wse-first":
        reservation = _open_listener(DEFAULT_CONTROL_HOST, 0)
        port = reservation.getsockname()[1]
        reservation.close()
        process = context.Process(
            target=_surrogate_process,
            kwargs={
                "device_id": device_id,
                "generation": generation,
                "host": DEFAULT_CONTROL_HOST,
                "kernel_binary": str(kernel_binary),
                "listens": True,
                "port": port,
                "ready_event": ready_event,
                "result_queue": result_queue,
                "run_id": run_id,
            },
            name=f"pypto-wse-surrogate-g{generation}",
        )
        process.start()
        if not ready_event.wait(DEFAULT_CONNECT_TIMEOUT_SECONDS):
            raise TimeoutError("surrogate listener did not become ready")
        connection = _connect(DEFAULT_CONTROL_HOST, port)
    else:
        raise ValueError("start_order must be attention-first or wse-first")
    return connection, process, result_queue


def run_generation(
    *,
    attention_device: int,
    surrogate_device: int,
    generation: int,
    element_counts: list[int],
    start_order: str,
    kernel_dir: Path,
) -> dict[str, Any]:
    run_id = f"proxy-{uuid4().hex}"
    connection, process, result_queue = _connected_surrogate(
        start_order=start_order,
        device_id=surrogate_device,
        generation=generation,
        run_id=run_id,
        kernel_binary=kernel_dir / "b_service.o",
    )
    channel = ProxyControlChannel(connection, run_id=run_id, generation=generation)
    memory_manager = NpuDeviceMemoryManager(
        endpoint_id="attention",
        device_id=attention_device,
        generation=generation,
    )
    bootstrap = BootstrapManager(
        channel=channel,
        memory_manager=memory_manager,
        run_id=run_id,
        generation=generation,
    )
    service = None
    try:
        bundle = bootstrap.prepare()
        service = PseudoPyptoDistributedService(bundle, driver_binary=kernel_dir / "abc_driver.o")
        initialization = service.initialize()
        executions = []
        for request_id, element_count in enumerate(element_counts, start=1):
            payload = deterministic_input(
                generation=generation,
                request_id=request_id,
                element_count=element_count,
            )
            result = service.execute(payload)
            if result.output != expected_abc(payload):
                raise RuntimeError(f"request {request_id} output does not match 2*x+5")
            executions.append(
                {
                    "element_count": result.element_count,
                    "final_control_d2h_bytes": result.final_control_d2h_bytes,
                    "final_payload_d2h_bytes": result.final_payload_d2h_bytes,
                    "final_signal_poll_reads": result.final_signal_poll_reads,
                    "output_checksum": result.output_checksum,
                    "request_id": result.request_id,
                    "sequence": result.sequence,
                }
            )
        health = service.health()
        drain = service.drain()
        service.close()
        service_evidence = service.evidence()
        bootstrap.release()
        process.join(DEFAULT_CONNECT_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(10)
            raise TimeoutError("surrogate process did not exit")
        try:
            surrogate_evidence = result_queue.get(timeout=5)
        except queue.Empty as exc:
            raise RuntimeError("surrogate process returned no evidence") from exc
        if process.exitcode != 0 or surrogate_evidence.get("status") != "PASS":
            raise RuntimeError(f"surrogate process failed: {surrogate_evidence}")
        return {
            "bootstrap": bootstrap.evidence(),
            "devices": {"attention": attention_device, "wse_surrogate": surrogate_device},
            "executions": executions,
            "generation": generation,
            "health": health,
            "initialization": initialization,
            "run_id": run_id,
            "service": service_evidence,
            "start_order": start_order,
            "status": "PASS",
            "surrogate": surrogate_evidence,
            "teardown": drain,
        }
    finally:
        connection.close()
        if process.is_alive():
            process.terminate()
            process.join(10)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-device", type=int, required=True)
    parser.add_argument("--surrogate-device", type=int, required=True)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--elements", type=int, nargs="*", default=[1024])
    parser.add_argument("--start-order", choices=("attention-first", "wse-first"), default="attention-first")
    parser.add_argument("--kernel-dir", type=Path, default=Path(__file__).parent / "build")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    evidence = run_generation(
        attention_device=args.attention_device,
        surrogate_device=args.surrogate_device,
        generation=args.generation,
        element_counts=args.elements,
        start_order=args.start_order,
        kernel_dir=args.kernel_dir,
    )
    encoded = json.dumps(evidence, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
