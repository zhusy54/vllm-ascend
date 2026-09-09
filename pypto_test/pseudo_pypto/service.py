# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Synchronous pseudo-PyPTO distributed service for fixed A -> B -> C."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pypto_test.pseudo_pypto.api import (
    DrainRequest,
    DrainResponse,
    ExecuteRequest,
    ExecuteResponse,
    HealthStatus,
    InitializeRequest,
    InitializeResponse,
)
from pypto_test.pseudo_pypto.backend import ExecutionRuntime, NpuExecutionBackend
from pypto_test.pseudo_pypto.communication import EndpointBundle, ServiceError, ServiceState


class PseudoPyptoDistributedService:
    """Upper-layer service boundary; no external allocator API is exposed.

    ``initialize`` performs generation setup, ``execute`` accepts one
    synchronous request, and ``health/drain/close`` manage the service
    lifecycle.  A future vLLM adapter can call this object without learning
    how Host RPC or VMM provisioning is implemented.
    """

    def __init__(
        self,
        bundle: EndpointBundle,
        *,
        driver_binary: Path,
        execution_runtime: ExecutionRuntime | None = None,
    ) -> None:
        bundle.validate()
        self._bundle = bundle
        self._driver_binary = driver_binary
        self._execution_runtime = execution_runtime
        self._state = ServiceState.NEW
        self._npu_backend: NpuExecutionBackend | None = None
        self._driver_drain: dict[str, Any] | None = None
        self._remote_drain: dict[str, Any] | None = None

    @property
    def state(self) -> ServiceState:
        return self._state

    def initialize(self, request: InitializeRequest | None = None) -> InitializeResponse:
        """Initialize in five explicit, reviewable orchestration steps."""

        if self._state is not ServiceState.NEW:
            raise ServiceError("initialize is only valid in NEW state")
        self._state = ServiceState.INITIALIZING
        initialization = request if request is not None else InitializeRequest()
        self._validate_initialization_context(initialization)
        self._construct_npu_execution_backend()
        remote_ready = self._start_wse_execution_service()
        driver_ready = self._start_npu_execution_driver()
        return self._publish_ready(initialization, remote_ready, driver_ready)

    def _validate_initialization_context(self, request: InitializeRequest) -> None:
        request.program.validate()
        self._bundle.validate()

    def _construct_npu_execution_backend(self) -> None:
        self._npu_backend = NpuExecutionBackend(
            binding=self._bundle.npu_communication,
            kernel_binary=self._driver_binary,
            runtime=self._execution_runtime,
        )

    def _start_wse_execution_service(self) -> dict[str, Any]:
        # Generation-level control only; no request tensor travels over RPC.
        return self._bundle.wse_control.start()

    def _start_npu_execution_driver(self) -> dict[str, Any]:
        if self._npu_backend is None:
            raise ServiceError("NPU execution backend was not constructed")
        return self._npu_backend.initialize()

    def _publish_ready(
        self,
        request: InitializeRequest,
        remote_ready: dict[str, Any],
        driver_ready: dict[str, Any],
    ) -> InitializeResponse:
        self._state = ServiceState.READY
        return InitializeResponse(
            generation=self._bundle.generation,
            state=self._state,
            driver_binary_sha256=str(driver_ready["binary_sha256"]),
            driver_ready_poll_reads=int(driver_ready["device_ready_poll_reads"]),
            program=request.program.to_dict(),
            remote=remote_ready,
            lifecycle=dict(driver_ready["lifecycle"]),
        )

    def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        if self._state is not ServiceState.READY:
            raise ServiceError(f"execute is not valid in {self._state.value} state")
        if self._npu_backend is None:
            raise ServiceError("NPU execution backend is not initialized")
        self._state = ServiceState.EXECUTING
        try:
            return ExecuteResponse(self._npu_backend.execute(request.payload))
        finally:
            # The prototype admits one synchronous request at a time.
            self._state = ServiceState.READY

    def health(self) -> HealthStatus:
        if self._npu_backend is None:
            raise ServiceError("service is not initialized")
        local = self._npu_backend.health()
        remote = self._bundle.wse_control.health()
        return HealthStatus(
            generation=self._bundle.generation,
            state=self._state,
            remote={"local": local, "wse": remote},
        )

    def drain(self, request: DrainRequest | None = None) -> DrainResponse:
        del request
        if self._state is ServiceState.DRAINED:
            return self._drain_response()
        if self._state is not ServiceState.READY or self._npu_backend is None:
            raise ServiceError(f"drain is not valid in {self._state.value} state")
        self._state = ServiceState.DRAINING
        self._driver_drain = self._npu_backend.drain()
        self._remote_drain = self._bundle.wse_control.drain()
        self._state = ServiceState.DRAINED
        return self._drain_response()

    def close(self) -> None:
        if self._state is ServiceState.CLOSED:
            return
        if self._state is ServiceState.READY:
            self.drain()
        if self._state is not ServiceState.DRAINED or self._npu_backend is None:
            raise ServiceError(f"close is not valid in {self._state.value} state")
        self._bundle.wse_control.close()
        self._npu_backend.close()
        self._bundle.lease.quiesce()
        self._state = ServiceState.CLOSED

    def evidence(self) -> dict[str, Any]:
        backend_evidence = self._npu_backend.evidence() if self._npu_backend is not None else {}
        return {
            **backend_evidence,
            "generation": self._bundle.generation,
            "remote": self._remote_drain,
            "state": self._state.value,
        }

    def _drain_response(self) -> DrainResponse:
        return DrainResponse(self._state, self._driver_drain, self._remote_drain)
