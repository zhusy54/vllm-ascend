# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Stable upper-layer API assumed for the pseudo-PyPTO service.

These request/response types are intentionally independent of vLLM.  A future
``DistributedExecutor`` adapter should translate vLLM calls into this API;
neither vLLM nor an external allocator is exposed through the service surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

from pypto_test.pseudo_pypto.communication import ExecutionResult, PseudoProgramSpec, ServiceState


@dataclass(frozen=True)
class InitializeRequest:
    program: PseudoProgramSpec = PseudoProgramSpec()


@dataclass(frozen=True)
class InitializeResponse:
    generation: int
    state: ServiceState
    driver_binary_sha256: str
    driver_ready_poll_reads: int
    program: dict[str, Any]
    remote: dict[str, Any]
    lifecycle: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state"] = self.state.value
        return result


@dataclass(frozen=True)
class ExecuteRequest:
    payload: bytes


@dataclass(frozen=True)
class ExecuteResponse:
    result: ExecutionResult


@dataclass(frozen=True)
class HealthStatus:
    generation: int
    state: ServiceState
    remote: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"generation": self.generation, "remote": self.remote, "state": self.state.value}


@dataclass(frozen=True)
class DrainRequest:
    """Explicit lifecycle request reserved for future drain policy fields."""


@dataclass(frozen=True)
class DrainResponse:
    state: ServiceState
    driver: dict[str, Any] | None
    remote: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {"driver": self.driver, "remote": self.remote, "state": self.state.value}


@runtime_checkable
class PyptoDistributedService(Protocol):
    """Final service boundary consumed by an upper framework adapter."""

    def initialize(self, request: InitializeRequest | None = None) -> InitializeResponse: ...

    def execute(self, request: ExecuteRequest) -> ExecuteResponse: ...

    def health(self) -> HealthStatus: ...

    def drain(self, request: DrainRequest | None = None) -> DrainResponse: ...

    def close(self) -> None: ...
