# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Two-process launcher for the host-local NPU/WSE-surrogate prototype.

``run_generation`` models the NPU Host service entry.  It asks Bootstrap to
create the second Host process and data plane, then interacts only with
PseudoPyptoDistributedService.  A complete normal generation is:

    launch processes -> prepare communication -> initialize kernels
      -> execute synchronous requests -> health -> drain -> close
      -> Bootstrap release -> collect redacted evidence

The child ``_surrogate_process`` contains the remote Host control loop.  It
starts and stops the B backend but does not receive per-request commands; the
resident B kernel observes those directly in Device memory.
"""

from __future__ import annotations

import argparse
import json
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from pypto_test.backend import NpuSurrogateBackend
from pypto_test.bootstrap import (
    BootstrapManager,
    ProxyControlChannel,
    SurrogateDeviceMemoryManager,
    accept_surrogate_bootstrap,
    connect_control_endpoint,
    open_control_listener,
)
from pypto_test.contracts import EndpointRole, deterministic_input, expected_abc
from pypto_test.service import PseudoPyptoDistributedService


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
    """Run the surrogate Host's bootstrap and generation control endpoint."""

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
        memory_manager = SurrogateDeviceMemoryManager(
            endpoint_id="wse-surrogate",
            device_id=device_id,
            generation=generation,
        )
        # All surrogate ACL allocation/import calls occur locally in this child
        # process.  Attention never invokes its Device Context across processes.
        resources = accept_surrogate_bootstrap(channel=channel, memory_manager=memory_manager)

        # START is the last Host control action before B becomes a resident
        # Device service.  No execute/task/payload message exists in this loop.
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
        # Generation control state machine:
        #   START/READY -> HEALTH* -> DRAIN/DRAINED -> CLOSE/CLOSED
        #   -> RELEASE/RELEASED
        # DRAIN stops execution; RELEASE later destroys Bootstrap-owned memory.
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
                # Backend close was acknowledged before this branch, so no
                # Device stream can still access either VMM mapping.
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
            with suppress(BaseException):
                backend.close()
        if memory_manager is not None:
            with suppress(BaseException):
                memory_manager.release()
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()


def run_generation(
    *,
    attention_device: int,
    surrogate_device: int,
    generation: int,
    element_counts: list[int],
    start_order: str,
    kernel_dir: Path,
) -> dict[str, Any]:
    """Execute one complete service generation and return measured evidence.

    This function represents the upper software layer: after Bootstrap returns
    a bundle it uses only the five proxy APIs.  It never reaches into the
    surrogate process, transport implementation, or communication allocator.
    """

    run_id = f"proxy-{uuid4().hex}"
    # Bootstrap owns process construction and connection lifetime.  ``spawn``
    # ensures the surrogate begins without an inherited ACL Device Context.
    bootstrap = BootstrapManager.launch_surrogate(
        endpoint_target=_surrogate_process,
        start_order=start_order,
        attention_device=attention_device,
        surrogate_device=surrogate_device,
        generation=generation,
        run_id=run_id,
        kernel_binary=kernel_dir / "b_service.o",
    )
    service = None
    try:
        # Communication must be fully attached before either resident kernel is
        # launched, because their arguments include imported peer addresses.
        bundle = bootstrap.prepare()
        service = PseudoPyptoDistributedService(bundle, driver_binary=kernel_dir / "abc_driver.o")
        initialization = service.initialize()
        executions = []
        for request_id, element_count in enumerate(element_counts, start=1):
            # Different generation/request seeds make stale-slot reuse visible.
            payload = deterministic_input(
                generation=generation,
                request_id=request_id,
                element_count=element_count,
            )
            result = service.execute(payload)
            # The CPU expression is an oracle checked only after final result
            # return; it does not participate in Device task progression.
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
        # Physical communication resources outlive service.close and are
        # released only after its lease is QUIESCED.
        surrogate_evidence = bootstrap.release()
        return {
            "bootstrap": bootstrap.evidence(),
            "devices": {"attention": attention_device, "wse_surrogate": surrogate_device},
            "endpoint_bundle": {
                "backend_kind": bundle.backend_kind,
                "generation": bundle.generation,
                "layout_hash": bundle.layout.layout_hash,
                "lease_id": bundle.lease.lease_id,
                "max_inflight": bundle.layout.max_inflight,
                "transport_kind": bundle.transport_kind,
                "transport_scope": bundle.transport_scope,
            },
            "executions": executions,
            "generation": generation,
            "health": health,
            "initialization": initialization,
            "run_id": run_id,
            "resident_kernel_launches": {"attention": 1, "wse_surrogate": 1},
            "service": service_evidence,
            "start_order": start_order,
            "status": "PASS",
            "surrogate": surrogate_evidence,
            "teardown": drain,
        }
    finally:
        bootstrap.abort()


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
