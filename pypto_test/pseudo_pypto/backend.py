# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Pseudo-PyPTO NPU/WSE execution backends and local AscendCL runtime.

External software initializes ACL and injects process-local cross-Device
mappings first.  This module owns PyPTO-local buffers, H2D/D2H, resident
kernel loading, and the fixed Device communication protocol.
"""

from __future__ import annotations

import ctypes
import hashlib
import time
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Protocol

from pypto_test.pseudo_pypto.communication import (
    CACHE_LINE_BYTES,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_ELEMENTS,
    NPU_LOCAL_CONTROL_OFFSET,
    NPU_LOCAL_FINAL_OUTPUT_OFFSET,
    NPU_LOCAL_HOST_REQUEST_DESC_OFFSET,
    NPU_LOCAL_HOST_REQUEST_SIGNAL_OFFSET,
    NPU_LOCAL_HOST_RESULT_DESC_OFFSET,
    NPU_LOCAL_HOST_RESULT_SIGNAL_OFFSET,
    NPU_LOCAL_INPUT_OFFSET,
    NPU_LOCAL_LIFECYCLE_OFFSET,
    NPU_LOCAL_REPORT_OFFSET,
    NPU_LOCAL_WINDOW_BYTES,
    WSE_LOCAL_LIFECYCLE_OFFSET,
    WSE_LOCAL_REPORT_OFFSET,
    WSE_LOCAL_WINDOW_BYTES,
    CompletionDescriptor,
    DriverReport,
    ExecutionResult,
    HostRequestDescriptor,
    LifecycleLine,
    NpuCommunicationBinding,
    NpuDeviceCommunication,
    ServiceError,
    ServiceReport,
    SignalLine,
    WseCommunicationBinding,
    WseDeviceCommunication,
    checksum_u32,
)

ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_RT_BINARY_LOAD_OPT_MAGIC = 2
ACL_RT_BINARY_MAGIC_ELF_AICORE = 0x43554245


class BackendError(RuntimeError):
    """Raised when a pseudo-PyPTO Device backend cannot make progress."""


class LocalDeviceBuffer(Protocol):
    address: int
    size: int

    def close(self) -> None: ...


class KernelSession(Protocol):
    binary_sha256: str

    def synchronize(self) -> int: ...

    def close(self) -> None: ...


class ExecutionRuntime(Protocol):
    """PyPTO-owned local memory, copy, and kernel-launch capability."""

    def allocate_local(self, size: int) -> LocalDeviceBuffer: ...

    def copy_host_to_device(self, destination: int, payload: bytes) -> None: ...

    def copy_device_to_host(self, source: int, size: int) -> bytes: ...

    def launch_kernel(self, binary_path: Path, arguments: ctypes.Structure) -> KernelSession: ...


class _AclrtBinaryLoadOptionValue(ctypes.Union):
    _fields_ = [
        ("is_lazy_load", ctypes.c_uint32),
        ("magic", ctypes.c_uint32),
        ("cpu_kernel_mode", ctypes.c_int32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class _AclrtBinaryLoadOption(ctypes.Structure):
    _fields_ = [("option_type", ctypes.c_int), ("value", _AclrtBinaryLoadOptionValue)]


class _AclrtBinaryLoadOptions(ctypes.Structure):
    _fields_ = [("options", ctypes.POINTER(_AclrtBinaryLoadOption)), ("count", ctypes.c_size_t)]


class AclLocalDeviceBuffer:
    """PyPTO-owned ordinary Device allocation; it is never exported."""

    def __init__(self, runtime: AclExecutionRuntime, address: int, size: int) -> None:
        self._runtime = runtime
        self.address = address
        self.size = size
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._runtime._check("aclrtFree", self._runtime._library.aclrtFree(self.address))
        self._closed = True


class AclExecutionRuntime:
    """Direct PyPTO AscendCL calls after external ACL bootstrap.

    The class deliberately has no ACL initialization/device-selection, VMM,
    handle, VA-map, peer-access, or finalize operation.
    """

    def __init__(self, *, library: Any | None = None) -> None:
        try:
            self._library = library if library is not None else ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise BackendError(f"unable to load libascendcl.so: {exc}") from exc
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        signatures: dict[str, tuple[list[Any], Any]] = {
            "aclrtMalloc": ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int], ctypes.c_int),
            "aclrtFree": ([ctypes.c_void_p], ctypes.c_int),
            "aclrtMemcpy": (
                [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
                ctypes.c_int,
            ),
            "aclrtBinaryLoadFromData": (
                [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.POINTER(_AclrtBinaryLoadOptions),
                    ctypes.POINTER(ctypes.c_void_p),
                ],
                ctypes.c_int,
            ),
            "aclrtBinaryGetFunctionByEntry": (
                [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)],
                ctypes.c_int,
            ),
            "aclrtBinaryUnLoad": ([ctypes.c_void_p], ctypes.c_int),
            "aclrtCreateStream": ([ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            "aclrtDestroyStream": ([ctypes.c_void_p], ctypes.c_int),
            "aclrtSynchronizeStream": ([ctypes.c_void_p], ctypes.c_int),
            "aclrtLaunchKernelWithHostArgs": (
                [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                ],
                ctypes.c_int,
            ),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(self._library, name)
            function.argtypes = argtypes
            function.restype = restype

    @staticmethod
    def _check(operation: str, result: int) -> None:
        if result != ACL_SUCCESS:
            raise BackendError(f"{operation} failed with code {result}")

    def allocate_local(self, size: int) -> AclLocalDeviceBuffer:
        if size <= 0:
            raise ValueError("local Device allocation size must be positive")
        address = ctypes.c_void_p()
        self._check("aclrtMalloc", self._library.aclrtMalloc(ctypes.byref(address), size, ACL_MEM_MALLOC_HUGE_FIRST))
        if not address.value:
            raise BackendError("aclrtMalloc returned a null address")
        return AclLocalDeviceBuffer(self, int(address.value), size)

    def copy_host_to_device(self, destination: int, payload: bytes) -> None:
        if destination <= 0 or not payload:
            raise ValueError("H2D destination and payload must be non-empty")
        source = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        self._check(
            "aclrtMemcpy H2D",
            self._library.aclrtMemcpy(destination, len(payload), source, len(payload), ACL_MEMCPY_HOST_TO_DEVICE),
        )

    def copy_device_to_host(self, source: int, size: int) -> bytes:
        if source <= 0 or size <= 0:
            raise ValueError("D2H source and size must be positive")
        destination = (ctypes.c_ubyte * size)()
        self._check(
            "aclrtMemcpy D2H",
            self._library.aclrtMemcpy(destination, size, source, size, ACL_MEMCPY_DEVICE_TO_HOST),
        )
        return bytes(destination)

    def launch_kernel(self, binary_path: Path, arguments: ctypes.Structure) -> AclDeviceKernel:
        kernel = AclDeviceKernel(self, binary_path)
        kernel.launch(arguments)
        return kernel


class AclDeviceKernel:
    """PyPTO Host wrapper for one resident AIV kernel and stream."""

    def __init__(self, runtime: AclExecutionRuntime, binary_path: Path) -> None:
        try:
            binary = binary_path.read_bytes()
        except OSError as exc:
            raise BackendError(f"cannot read AIV binary {binary_path}: {exc}") from exc
        if not binary:
            raise BackendError(f"AIV binary is empty: {binary_path}")
        self._runtime = runtime
        self.binary_sha256 = hashlib.sha256(binary).hexdigest()
        self._buffer = ctypes.create_string_buffer(binary)
        self._binary_handle = ctypes.c_void_p()
        self._function_handle = ctypes.c_void_p()
        self._stream = ctypes.c_void_p()
        self._started_ns = 0
        self._launched = False
        self._synchronized = False
        self._closed = False
        self._load()

    def _load(self) -> None:
        option = _AclrtBinaryLoadOption(
            option_type=ACL_RT_BINARY_LOAD_OPT_MAGIC,
            value=_AclrtBinaryLoadOptionValue(magic=ACL_RT_BINARY_MAGIC_ELF_AICORE),
        )
        options = _AclrtBinaryLoadOptions(options=ctypes.pointer(option), count=1)
        try:
            self._runtime._check(
                "aclrtBinaryLoadFromData",
                self._runtime._library.aclrtBinaryLoadFromData(
                    ctypes.addressof(self._buffer),
                    len(self._buffer) - 1,
                    ctypes.byref(options),
                    ctypes.byref(self._binary_handle),
                ),
            )
            self._runtime._check(
                "aclrtBinaryGetFunctionByEntry",
                self._runtime._library.aclrtBinaryGetFunctionByEntry(
                    self._binary_handle,
                    0,
                    ctypes.byref(self._function_handle),
                ),
            )
            self._runtime._check(
                "aclrtCreateStream",
                self._runtime._library.aclrtCreateStream(ctypes.byref(self._stream)),
            )
            if not self._binary_handle.value or not self._function_handle.value or not self._stream.value:
                raise BackendError("AscendCL returned a null kernel resource")
        except BaseException:
            self.close()
            raise

    def launch(self, arguments: ctypes.Structure) -> None:
        if self._closed or self._launched:
            raise BackendError("resident kernel cannot launch in its current state")
        self._started_ns = perf_counter_ns()
        self._runtime._check(
            "aclrtLaunchKernelWithHostArgs",
            self._runtime._library.aclrtLaunchKernelWithHostArgs(
                self._function_handle,
                1,
                self._stream,
                None,
                ctypes.byref(arguments),
                ctypes.sizeof(arguments),
                None,
                0,
            ),
        )
        self._launched = True

    def synchronize(self) -> int:
        if not self._launched or self._synchronized:
            raise BackendError("resident kernel cannot synchronize in its current state")
        self._runtime._check("aclrtSynchronizeStream", self._runtime._library.aclrtSynchronizeStream(self._stream))
        self._synchronized = True
        return perf_counter_ns() - self._started_ns

    def close(self) -> None:
        if self._closed:
            return
        failures: list[str] = []
        if self._stream.value:
            result = self._runtime._library.aclrtDestroyStream(self._stream)
            if result != ACL_SUCCESS:
                failures.append(f"aclrtDestroyStream failed with code {result}")
            self._stream = ctypes.c_void_p()
        if self._binary_handle.value:
            result = self._runtime._library.aclrtBinaryUnLoad(self._binary_handle)
            if result != ACL_SUCCESS:
                failures.append(f"aclrtBinaryUnLoad failed with code {result}")
            self._binary_handle = ctypes.c_void_p()
        self._closed = True
        if failures:
            raise BackendError("; ".join(failures))


class DriverKernelArguments(ctypes.Structure):
    """Fixed NPU A/C driver ABI with explicit local/shared addresses."""

    _fields_ = [
        ("local_input", ctypes.c_void_p),
        ("local_b_output", ctypes.c_void_p),
        ("local_final_output", ctypes.c_void_p),
        ("host_request_signal", ctypes.c_void_p),
        ("host_request_descriptor", ctypes.c_void_p),
        ("host_result_signal", ctypes.c_void_p),
        ("host_result_descriptor", ctypes.c_void_p),
        ("b_completion_signal", ctypes.c_void_p),
        ("b_completion_descriptor", ctypes.c_void_p),
        ("lifecycle", ctypes.c_void_p),
        ("report", ctypes.c_void_p),
        ("remote_b_input", ctypes.c_void_p),
        ("remote_submission_signal", ctypes.c_void_p),
        ("remote_submission_descriptor", ctypes.c_void_p),
        ("generation", ctypes.c_uint64),
        ("max_elements", ctypes.c_uint64),
    ]


class BServiceKernelArguments(ctypes.Structure):
    """Fixed WSE B-service ABI with explicit local/shared addresses."""

    _fields_ = [
        ("local_b_input", ctypes.c_void_p),
        ("submission_signal", ctypes.c_void_p),
        ("submission_descriptor", ctypes.c_void_p),
        ("lifecycle", ctypes.c_void_p),
        ("report", ctypes.c_void_p),
        ("remote_b_output", ctypes.c_void_p),
        ("remote_completion_signal", ctypes.c_void_p),
        ("remote_completion_descriptor", ctypes.c_void_p),
        ("generation", ctypes.c_uint64),
        ("max_elements", ctypes.c_uint64),
    ]


class NpuHostIo:
    """PyPTO-owned NPU ingress, final completion polling, and egress."""

    def __init__(self, runtime: ExecutionRuntime, local_base: int) -> None:
        self._runtime = runtime
        self._local_base = local_base
        self.traffic = {
            "final_control_d2h_bytes": 0,
            "final_payload_d2h_bytes": 0,
            "host_intermediate_bytes": 0,
            "input_h2d_bytes": 0,
            "request_control_h2d_bytes": 0,
        }

    def publish_request(self, payload: bytes, descriptor: HostRequestDescriptor) -> None:
        self._runtime.copy_host_to_device(self._local_base + NPU_LOCAL_INPUT_OFFSET, payload)
        self._runtime.copy_host_to_device(
            self._local_base + NPU_LOCAL_HOST_REQUEST_DESC_OFFSET,
            descriptor.to_bytes(),
        )
        self._runtime.copy_host_to_device(
            self._local_base + NPU_LOCAL_HOST_REQUEST_SIGNAL_OFFSET,
            SignalLine(descriptor.sequence).to_bytes(),
        )
        self.traffic["input_h2d_bytes"] += len(payload)
        self.traffic["request_control_h2d_bytes"] += 2 * CACHE_LINE_BYTES

    def read_final_signal(self) -> SignalLine:
        self.traffic["final_control_d2h_bytes"] += CACHE_LINE_BYTES
        return SignalLine.from_bytes(
            self._runtime.copy_device_to_host(
                self._local_base + NPU_LOCAL_HOST_RESULT_SIGNAL_OFFSET,
                CACHE_LINE_BYTES,
            )
        )

    def read_final_descriptor(self) -> CompletionDescriptor:
        self.traffic["final_control_d2h_bytes"] += CACHE_LINE_BYTES
        return CompletionDescriptor.from_bytes(
            self._runtime.copy_device_to_host(
                self._local_base + NPU_LOCAL_HOST_RESULT_DESC_OFFSET,
                CACHE_LINE_BYTES,
            )
        )

    def read_final_output(self, size: int) -> bytes:
        output = self._runtime.copy_device_to_host(self._local_base + NPU_LOCAL_FINAL_OUTPUT_OFFSET, size)
        self.traffic["final_payload_d2h_bytes"] += len(output)
        return output


class NpuExecutionBackend:
    """PyPTO NPU backend: owns local IO, A/C resident driver, and requests."""

    def __init__(
        self,
        *,
        binding: NpuCommunicationBinding,
        kernel_binary: Path,
        runtime: ExecutionRuntime | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        binding.validate()
        self.binding = binding
        self._kernel_binary = kernel_binary
        self._runtime = runtime if runtime is not None else AclExecutionRuntime()
        self._timeout_seconds = timeout_seconds
        self._local: LocalDeviceBuffer | None = None
        self._io: NpuHostIo | None = None
        self._kernel: KernelSession | None = None
        self._sequence = 0
        self._request_id = 0
        self._drain_result: dict[str, Any] | None = None

    def initialize(self) -> dict[str, Any]:
        if self._local is not None:
            raise BackendError("NPU backend is already initialized")
        local = self._runtime.allocate_local(NPU_LOCAL_WINDOW_BYTES)
        self._local = local
        self._io = NpuHostIo(self._runtime, local.address)
        communication = NpuDeviceCommunication.from_binding(self.binding)
        self._runtime.copy_host_to_device(local.address + NPU_LOCAL_CONTROL_OFFSET, bytes(8 * CACHE_LINE_BYTES))
        self._runtime.copy_host_to_device(
            communication.local_completion_signal,
            bytes(3 * CACHE_LINE_BYTES),
        )
        self._kernel = self._runtime.launch_kernel(
            self._kernel_binary,
            DriverKernelArguments(
                local.address + NPU_LOCAL_INPUT_OFFSET,
                communication.local_b_output,
                local.address + NPU_LOCAL_FINAL_OUTPUT_OFFSET,
                local.address + NPU_LOCAL_HOST_REQUEST_SIGNAL_OFFSET,
                local.address + NPU_LOCAL_HOST_REQUEST_DESC_OFFSET,
                local.address + NPU_LOCAL_HOST_RESULT_SIGNAL_OFFSET,
                local.address + NPU_LOCAL_HOST_RESULT_DESC_OFFSET,
                communication.local_completion_signal,
                communication.local_completion_descriptor,
                local.address + NPU_LOCAL_LIFECYCLE_OFFSET,
                local.address + NPU_LOCAL_REPORT_OFFSET,
                communication.remote_b_input,
                communication.remote_submission_signal,
                communication.remote_submission_descriptor,
                self.binding.generation,
                MAX_ELEMENTS,
            ),
        )
        lifecycle, polls = self._wait_for_lifecycle("ready")
        return {
            "binary_sha256": self._kernel.binary_sha256,
            "device_ready_poll_reads": polls,
            "lifecycle": lifecycle.__dict__,
        }

    def execute(self, payload: bytes) -> ExecutionResult:
        input_checksum = checksum_u32(payload)
        if self._io is None:
            raise BackendError("NPU backend is not initialized")
        self._sequence += 1
        self._request_id += 1
        descriptor = HostRequestDescriptor(
            self.binding.generation,
            self._request_id,
            len(payload) // 4,
            input_checksum,
            self._sequence,
        )
        self._io.publish_request(payload, descriptor)
        completion, poll_reads = self._wait_for_final_completion(self._sequence)
        if (
            completion.generation != self.binding.generation
            or completion.request_id != self._request_id
            or completion.sequence != self._sequence
            or completion.element_count != len(payload) // 4
            or completion.status != 0
        ):
            raise ServiceError("final completion identity or status mismatch")
        output = self._io.read_final_output(len(payload))
        if checksum_u32(output) != completion.output_checksum:
            raise ServiceError("final output checksum mismatch")
        return ExecutionResult(
            self.binding.generation,
            self._request_id,
            self._sequence,
            len(payload) // 4,
            output,
            completion.output_checksum,
            poll_reads,
            (poll_reads + 1) * CACHE_LINE_BYTES,
            len(output),
        )

    def health(self) -> dict[str, Any]:
        lifecycle = self._read_lifecycle()
        return {"ready": lifecycle.ready == 1 and lifecycle.stopped == 0}

    def drain(self) -> dict[str, Any]:
        if self._drain_result is not None:
            return dict(self._drain_result)
        if self._local is None or self._kernel is None:
            raise BackendError("NPU backend is not initialized")
        self._runtime.copy_host_to_device(
            self._local.address + NPU_LOCAL_LIFECYCLE_OFFSET,
            LifecycleLine(stop_requested=1).to_bytes(),
        )
        lifecycle, polls = self._wait_for_lifecycle("stopped")
        elapsed_ns = self._kernel.synchronize()
        report = DriverReport.from_bytes(
            self._runtime.copy_device_to_host(self._local.address + NPU_LOCAL_REPORT_OFFSET, 2 * CACHE_LINE_BYTES)
        )
        self._drain_result = {
            "device_stopped_poll_reads": polls,
            "kernel_elapsed_ns": elapsed_ns,
            "lifecycle": lifecycle.__dict__,
            "report": report.to_dict(),
        }
        return dict(self._drain_result)

    def close(self) -> None:
        if self._kernel is not None:
            if self._drain_result is None:
                self.drain()
            self._kernel.close()
            self._kernel = None
        if self._local is not None:
            self._local.close()
            self._local = None

    def evidence(self) -> dict[str, Any]:
        return {
            "driver": self._drain_result,
            "traffic": dict(self._io.traffic) if self._io is not None else None,
        }

    def _read_lifecycle(self) -> LifecycleLine:
        if self._local is None:
            raise BackendError("NPU backend has no local memory")
        return LifecycleLine.from_bytes(
            self._runtime.copy_device_to_host(
                self._local.address + NPU_LOCAL_LIFECYCLE_OFFSET,
                CACHE_LINE_BYTES,
            )
        )

    def _wait_for_final_completion(self, sequence: int) -> tuple[CompletionDescriptor, int]:
        if self._io is None:
            raise BackendError("NPU backend is not initialized")
        deadline = time.monotonic() + self._timeout_seconds
        poll_reads = 0
        while True:
            signal = self._io.read_final_signal()
            poll_reads += 1
            if signal.sequence == sequence:
                return self._io.read_final_descriptor(), poll_reads
            if signal.sequence > sequence:
                raise ServiceError("final completion sequence advanced beyond the request")
            if time.monotonic() >= deadline:
                raise ServiceError("timed out waiting for final Device completion")
            time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)

    def _wait_for_lifecycle(self, field: str) -> tuple[LifecycleLine, int]:
        deadline = time.monotonic() + self._timeout_seconds
        polls = 0
        while True:
            lifecycle = self._read_lifecycle()
            polls += 1
            if getattr(lifecycle, field) == 1:
                return lifecycle, polls
            if time.monotonic() >= deadline:
                raise BackendError(f"timed out waiting for NPU backend {field}")
            time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)


class WseExecutionBackend(Protocol):
    backend_kind: str

    def initialize(self) -> dict[str, Any]: ...

    def health(self) -> dict[str, Any]: ...

    def drain(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class WseBackend:
    """PyPTO WSE backend; this prototype runs B on a second NPU."""

    backend_kind = "WSE"

    def __init__(
        self,
        *,
        binding: WseCommunicationBinding,
        kernel_binary: Path,
        runtime: ExecutionRuntime | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        binding.validate()
        self.binding = binding
        self._kernel_binary = kernel_binary
        self._runtime = runtime if runtime is not None else AclExecutionRuntime()
        self._timeout_seconds = timeout_seconds
        self._local: LocalDeviceBuffer | None = None
        self._kernel: KernelSession | None = None
        self._drain_result: dict[str, Any] | None = None

    def initialize(self) -> dict[str, Any]:
        if self._local is not None:
            raise BackendError("WSE backend is already initialized")
        local = self._runtime.allocate_local(WSE_LOCAL_WINDOW_BYTES)
        self._local = local
        communication = WseDeviceCommunication.from_binding(self.binding)
        self._runtime.copy_host_to_device(local.address, bytes(WSE_LOCAL_WINDOW_BYTES))
        self._runtime.copy_host_to_device(
            communication.local_submission_signal,
            bytes(3 * CACHE_LINE_BYTES),
        )
        self._kernel = self._runtime.launch_kernel(
            self._kernel_binary,
            BServiceKernelArguments(
                communication.local_b_input,
                communication.local_submission_signal,
                communication.local_submission_descriptor,
                local.address + WSE_LOCAL_LIFECYCLE_OFFSET,
                local.address + WSE_LOCAL_REPORT_OFFSET,
                communication.remote_b_output,
                communication.remote_completion_signal,
                communication.remote_completion_descriptor,
                self.binding.generation,
                MAX_ELEMENTS,
            ),
        )
        lifecycle, polls = self._wait_for_lifecycle("ready")
        return {
            "backend_kind": self.backend_kind,
            "binary_sha256": self._kernel.binary_sha256,
            "device_ready_poll_reads": polls,
            "lifecycle": lifecycle.__dict__,
        }

    def health(self) -> dict[str, Any]:
        lifecycle = self._read_lifecycle()
        return {"backend_kind": self.backend_kind, "ready": lifecycle.ready == 1 and lifecycle.stopped == 0}

    def drain(self) -> dict[str, Any]:
        if self._drain_result is not None:
            return dict(self._drain_result)
        if self._local is None or self._kernel is None:
            raise BackendError("WSE backend is not initialized")
        self._runtime.copy_host_to_device(
            self._local.address + WSE_LOCAL_LIFECYCLE_OFFSET,
            LifecycleLine(stop_requested=1).to_bytes(),
        )
        lifecycle, polls = self._wait_for_lifecycle("stopped")
        elapsed_ns = self._kernel.synchronize()
        report = ServiceReport.from_bytes(
            self._runtime.copy_device_to_host(self._local.address + WSE_LOCAL_REPORT_OFFSET, 2 * CACHE_LINE_BYTES)
        )
        self._drain_result = {
            "device_stopped_poll_reads": polls,
            "kernel_elapsed_ns": elapsed_ns,
            "lifecycle": lifecycle.__dict__,
            "report": report.to_dict(),
        }
        return dict(self._drain_result)

    def close(self) -> None:
        if self._kernel is not None:
            if self._drain_result is None:
                self.drain()
            self._kernel.close()
            self._kernel = None
        if self._local is not None:
            self._local.close()
            self._local = None

    def _read_lifecycle(self) -> LifecycleLine:
        if self._local is None:
            raise BackendError("WSE backend has no local memory")
        return LifecycleLine.from_bytes(
            self._runtime.copy_device_to_host(
                self._local.address + WSE_LOCAL_LIFECYCLE_OFFSET,
                CACHE_LINE_BYTES,
            )
        )

    def _wait_for_lifecycle(self, field: str) -> tuple[LifecycleLine, int]:
        deadline = time.monotonic() + self._timeout_seconds
        polls = 0
        while True:
            lifecycle = self._read_lifecycle()
            polls += 1
            if getattr(lifecycle, field) == 1:
                return lifecycle, polls
            if time.monotonic() >= deadline:
                raise BackendError(f"timed out waiting for WSE backend {field}")
            time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)


class RealWseBackend:
    """Reserved PyPTO execution boundary for a future physical WSE runtime."""

    backend_kind = "REAL_WSE_UNIMPLEMENTED"

    def initialize(self) -> dict[str, Any]:
        raise NotImplementedError("real WSE backend is not part of this prototype")

    def health(self) -> dict[str, Any]:
        raise NotImplementedError("real WSE backend is not part of this prototype")

    def drain(self) -> dict[str, Any]:
        raise NotImplementedError("real WSE backend is not part of this prototype")

    def close(self) -> None:
        raise NotImplementedError("real WSE backend is not part of this prototype")
