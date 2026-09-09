# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Example entry point for the host-local NPU/WSE proxy service.

``run_proxy_service`` models the NPU Host service entry.  It asks Bootstrap to
create the second Host process and data plane, then interacts only with
PseudoPyptoDistributedService.  A complete normal generation is:

    launch WSE Host -> launch NPU Host -> build communication -> initialize kernels
      -> execute synchronous requests -> health -> drain -> close
      -> Bootstrap release -> collect redacted evidence

The child ``_wse_host_process`` contains the WSE Host control loop.  It
starts and stops the B backend but does not receive per-request commands; the
resident B kernel observes those directly in Device memory.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from pypto_test.infrastructure.bootstrap import (
    BootstrapManager,
    ProxyControlChannel,
    WseDeviceMemoryManager,
    accept_wse_communication,
    connect_control_endpoint,
    open_control_listener,
)
from pypto_test.pseudo_pypto.backend import WseBackend
from pypto_test.pseudo_pypto.contracts import EndpointRole, ExecutionResult
from pypto_test.pseudo_pypto.service import PseudoPyptoDistributedService
from pypto_test.validation.validation_utils import get_input_payload, return_result


def _wse_host_process(
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
    """Run the WSE Host bootstrap and service-lifecycle control endpoint."""

    listener = None
    connection = None
    memory_manager = None
    backend = None
    try:
        # The two branches vary process start order only.  Once connected, they
        # use the same manifest, attach, and service lifecycle protocol.
        if listens:
            listener = open_control_listener(host, port)
            ready_event.set()
            connection, _ = listener.accept()
        else:
            connection = connect_control_endpoint(host, port)
            ready_event.set()
        channel = ProxyControlChannel(connection, run_id=run_id, generation=generation)
        memory_manager = WseDeviceMemoryManager(
            endpoint_id="wse",
            device_id=device_id,
            generation=generation,
        )
        # Stage 1: initialize the WSE-side Device runtime without allocating or
        # attaching communication memory, then report Host readiness.
        memory_manager.initialize_runtime()
        channel.send("WSE_HOST_READY", EndpointRole.WSE)

        # Stage 3: wait until the NPU Host is also ready, then build the WSE
        # half of the communication data plane in this process-local Context.
        resources = accept_wse_communication(channel=channel, memory_manager=memory_manager)

        # START is the last Host control action before B becomes a resident
        # Device service.  No execute/task/payload message exists in this loop.
        message = channel.receive("START", EndpointRole.ATTENTION)
        del message
        backend = WseBackend(
            execution_port=resources.execution_port,
            local_window=resources.local_window,
            npu_peer_window=resources.npu_peer_window,
            kernel_binary=Path(kernel_binary),
        )
        ready = backend.initialize()
        channel.send("READY", EndpointRole.WSE, details=ready)
        drain_evidence: dict[str, Any] | None = None
        # Generation control state machine:
        #   START/READY -> HEALTH* -> DRAIN/DRAINED -> CLOSE/CLOSED
        #   -> RELEASE/RELEASED
        # DRAIN stops execution; RELEASE later destroys Bootstrap-owned memory.
        while True:
            message = channel.receive_any(EndpointRole.ATTENTION)
            message_type = message["type"]
            if message_type == "HEALTH":
                channel.send("HEALTH_REPLY", EndpointRole.WSE, details=backend.health())
            elif message_type == "DRAIN":
                drain_evidence = backend.drain()
                channel.send("DRAINED", EndpointRole.WSE, details=drain_evidence)
            elif message_type == "CLOSE":
                backend.close()
                channel.send("CLOSED", EndpointRole.WSE)
            elif message_type == "RELEASE":
                # Backend close was acknowledged before this branch, so no
                # Device stream can still access either VMM mapping.
                memory_manager.release()
                channel.send("RELEASED", EndpointRole.WSE)
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
                raise RuntimeError(f"unexpected WSE control message: {message_type}")
    except BaseException as exc:
        result_queue.put({"error": f"{type(exc).__name__}: {exc}", "status": "FAIL"})
        raise
    finally:
        if backend is not None:
            with suppress(BaseException):
                backend.close()
        if memory_manager is not None:
            with suppress(BaseException):
                memory_manager.release()
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()


def run_proxy_service(
    *,
    attention_device: int,
    wse_device: int,
    generation: int,
    input_payloads: list[bytes],
    start_order: str,
    kernel_dir: Path,
    result_handler: Callable[[bytes, ExecutionResult], None] | None = None,
) -> dict[str, Any]:
    """Execute one complete proxy-service lifecycle and return measured evidence.

    This function represents the upper software layer: after Bootstrap returns
    a bundle it uses only the five proxy APIs.  It never reaches into the
    WSE Host process, transport implementation, or communication allocator.
    """

    run_id = f"proxy-{uuid4().hex}"
    # Bootstrap owns both Host initialization and all communication resources.
    # ``spawn`` ensures the WSE Host begins without an inherited ACL Context.
    with BootstrapManager(
        attention_device=attention_device,
        wse_device=wse_device,
        generation=generation,
        run_id=run_id,
    ) as bootstrap:
        bootstrap.launch_wse_host(
            endpoint_target=_wse_host_process,
            kernel_binary=kernel_dir / "b_service.o",
            start_order=start_order,
        )
        bootstrap.launch_npu_host()
        bundle = bootstrap.build_communication()
        service = PseudoPyptoDistributedService(bundle, driver_binary=kernel_dir / "abc_driver.o")
        initialization = service.initialize()
        request_count = 0
        input_bytes = 0
        output_bytes = 0
        for input_payload in input_payloads:
            result = service.execute(input_payload)
            if result_handler is not None:
                # Result interpretation belongs to the caller.  The core run
                # path only returns C output and never imports a test oracle.
                result_handler(input_payload, result)
            request_count += 1
            input_bytes += len(input_payload)
            output_bytes += len(result.output)
        health = service.health()
        drain = service.drain()
        service.close()
        service_evidence = service.evidence()
        # Physical communication resources outlive service.close and are
        # released only after its lease is QUIESCED.
        wse_evidence = bootstrap.release()
        return {
            "bootstrap": bootstrap.evidence(),
            "devices": {"attention": attention_device, "wse": wse_device},
            "endpoint_bundle": {
                "backend_kind": bundle.backend_kind,
                "generation": bundle.generation,
                "layout_hash": bundle.layout.layout_hash,
                "lease_id": bundle.lease.lease_id,
                "max_inflight": bundle.layout.max_inflight,
                "transport_kind": bundle.transport_kind,
                "transport_scope": bundle.transport_scope,
            },
            "execution_summary": {
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "request_count": request_count,
            },
            "generation": generation,
            "health": health,
            "initialization": initialization,
            "run_id": run_id,
            "resident_kernel_launches": {"attention": 1, "wse": 1},
            "service": service_evidence,
            "start_order": start_order,
            "status": "PASS",
            "wse": wse_evidence,
            "teardown": drain,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-device", type=int, required=True)
    parser.add_argument("--wse-device", type=int, required=True)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--elements", type=int, nargs="*", default=[1024])
    parser.add_argument("--start-order", choices=("attention-first", "wse-first"), default="attention-first")
    parser.add_argument("--kernel-dir", type=Path, default=Path(__file__).parent / "build")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run one proxy-service instance without invoking the validation matrix."""

    args = _parse_args()
    input_payloads = [
        get_input_payload(generation=args.generation, request_id=request_id, element_count=element_count)
        for request_id, element_count in enumerate(args.elements, start=1)
    ]
    result = run_proxy_service(
        attention_device=args.attention_device,
        wse_device=args.wse_device,
        generation=args.generation,
        input_payloads=input_payloads,
        start_order=args.start_order,
        kernel_dir=args.kernel_dir,
        result_handler=return_result,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
