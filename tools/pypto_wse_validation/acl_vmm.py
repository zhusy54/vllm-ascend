# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Minimal AscendCL VMM/P2P binding used by the stage-1A validation probe."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

ACL_SUCCESS = 0
ACL_MEM_HANDLE_TYPE_NONE = 0
ACL_MEM_ALLOCATION_TYPE_PINNED = 0
ACL_HBM_MEM_NORMAL = 5
ACL_MEM_LOCATION_TYPE_DEVICE = 1
ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM = 0
ACL_RT_MEM_ACCESS_FLAGS_READWRITE = 0x3
ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION = 0x1
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
VALIDATION_P2P_CHUNK_BYTES = 64 * 1024
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_UINT8 = 4
ACL_FORMAT_ND = 2


class AclError(RuntimeError):
    """Raised when an AscendCL operation fails."""


class _AclMemLocation(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("type", ctypes.c_int)]


class _AclPhysicalMemProp(ctypes.Structure):
    _fields_ = [
        ("handle_type", ctypes.c_int),
        ("allocation_type", ctypes.c_int),
        ("mem_attr", ctypes.c_int),
        ("location", _AclMemLocation),
        ("reserve", ctypes.c_uint64),
    ]


class _AclMemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_int),
        ("location", _AclMemLocation),
        ("reserved", ctypes.c_uint8 * 12),
    ]


def _align_up(value: int, alignment: int) -> int:
    if value <= 0 or alignment <= 0:
        raise ValueError("value and alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def _load_acl_library() -> Any:
    try:
        return ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise AclError(f"unable to load libascendcl.so: {exc}") from exc


@dataclass(frozen=True)
class VmmExport:
    device_id: int
    mapping_bytes: int
    shareable_handle: int


@dataclass(frozen=True)
class DeviceTransformEvidence:
    api: str
    elapsed_ns: int
    workspace_bytes: int


@dataclass(frozen=True)
class StaleImportProbe:
    api: str
    rejected: bool
    result_code: int


class AclVmmRuntime:
    """Own one process-local ACL context and its VMM operations."""

    def __init__(self, device_id: int, *, access_device_id: int | None = None, library: Any | None = None) -> None:
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        self.device_id = device_id
        self.access_device_id = device_id if access_device_id is None else access_device_id
        if self.access_device_id < 0:
            raise ValueError("access_device_id must be non-negative")
        self._library = library if library is not None else _load_acl_library()
        self._initialized = False
        self._closed = False
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        signatures: dict[str, tuple[list[Any], Any]] = {
            "aclInit": ([ctypes.c_char_p], ctypes.c_int),
            "aclFinalize": ([], ctypes.c_int),
            "aclrtSetDevice": ([ctypes.c_int], ctypes.c_int),
            "aclrtResetDevice": ([ctypes.c_int], ctypes.c_int),
            "aclrtSynchronizeDevice": ([], ctypes.c_int),
            "aclrtDeviceEnablePeerAccess": ([ctypes.c_int, ctypes.c_uint32], ctypes.c_int),
            "aclrtMemGetAllocationGranularity": (
                [ctypes.POINTER(_AclPhysicalMemProp), ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)],
                ctypes.c_int,
            ),
            "aclrtMallocPhysical": (
                [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_size_t,
                    ctypes.POINTER(_AclPhysicalMemProp),
                    ctypes.c_uint64,
                ],
                ctypes.c_int,
            ),
            "aclrtFreePhysical": ([ctypes.c_void_p], ctypes.c_int),
            "aclrtReserveMemAddress": (
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint64],
                ctypes.c_int,
            ),
            "aclrtReleaseMemAddress": ([ctypes.c_void_p], ctypes.c_int),
            "aclrtMapMem": (
                [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint64],
                ctypes.c_int,
            ),
            "aclrtUnmapMem": ([ctypes.c_void_p], ctypes.c_int),
            "aclrtMemSetAccess": (
                [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(_AclMemAccessDesc), ctypes.c_size_t],
                ctypes.c_int,
            ),
            "aclrtMemExportToShareableHandle": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)],
                ctypes.c_int,
            ),
            "aclrtMemImportFromShareableHandle": (
                [ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)],
                ctypes.c_int,
            ),
            "aclrtMemcpy": (
                [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
                ctypes.c_int,
            ),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(self._library, name)
            function.argtypes = argtypes
            function.restype = restype

    def _check(self, operation: str, result: int) -> None:
        if result != ACL_SUCCESS:
            raise AclError(f"{operation} failed with code {result}")

    def initialize(self) -> None:
        if self._closed:
            raise AclError("ACL runtime is closed")
        if self._initialized:
            return
        self._check("aclInit", self._library.aclInit(None))
        try:
            self._check("aclrtSetDevice", self._library.aclrtSetDevice(self.device_id))
        except BaseException:
            self._library.aclFinalize()
            raise
        self._initialized = True

    def _require_initialized(self) -> None:
        if not self._initialized or self._closed:
            raise AclError("ACL runtime is not initialized")

    def _physical_properties(self) -> _AclPhysicalMemProp:
        return _AclPhysicalMemProp(
            handle_type=ACL_MEM_HANDLE_TYPE_NONE,
            allocation_type=ACL_MEM_ALLOCATION_TYPE_PINNED,
            mem_attr=ACL_HBM_MEM_NORMAL,
            location=_AclMemLocation(id=self.device_id, type=ACL_MEM_LOCATION_TYPE_DEVICE),
            reserve=0,
        )

    def _access_descriptor(self) -> _AclMemAccessDesc:
        return _AclMemAccessDesc(
            flags=ACL_RT_MEM_ACCESS_FLAGS_READWRITE,
            location=_AclMemLocation(id=self.access_device_id, type=ACL_MEM_LOCATION_TYPE_DEVICE),
        )

    def allocation_granularity(self) -> int:
        self._require_initialized()
        granularity = ctypes.c_size_t()
        properties = self._physical_properties()
        self._check(
            "aclrtMemGetAllocationGranularity",
            self._library.aclrtMemGetAllocationGranularity(
                ctypes.byref(properties),
                ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM,
                ctypes.byref(granularity),
            ),
        )
        if granularity.value == 0:
            raise AclError("aclrtMemGetAllocationGranularity returned zero")
        return int(granularity.value)

    def allocate_window(self, logical_bytes: int) -> OwnedVmmWindow:
        self._require_initialized()
        return OwnedVmmWindow.create(self, logical_bytes)

    def import_window(self, exported: VmmExport, *, peer_device_id: int) -> ImportedVmmWindow:
        self._require_initialized()
        return ImportedVmmWindow.create(self, exported, peer_device_id=peer_device_id)

    def probe_stale_import(self, exported: VmmExport) -> StaleImportProbe:
        """Probe an old shareable handle without enabling peer access or mapping it."""
        self._require_initialized()
        if exported.mapping_bytes <= 0 or exported.shareable_handle <= 0:
            raise ValueError("import descriptor must contain positive values")
        imported_handle = ctypes.c_void_p()
        result = self._library.aclrtMemImportFromShareableHandle(
            exported.shareable_handle,
            self.device_id,
            ctypes.byref(imported_handle),
        )
        if result != ACL_SUCCESS:
            return StaleImportProbe(
                api="aclrtMemImportFromShareableHandle",
                rejected=True,
                result_code=int(result),
            )
        if imported_handle.value:
            self._check(
                "aclrtFreePhysical stale import probe",
                self._library.aclrtFreePhysical(imported_handle),
            )
        return StaleImportProbe(
            api="aclrtMemImportFromShareableHandle",
            rejected=False,
            result_code=ACL_SUCCESS,
        )

    def _reserve_and_map(self, handle: int, mapping_bytes: int) -> int:
        address = ctypes.c_void_p()
        self._check(
            "aclrtReserveMemAddress",
            self._library.aclrtReserveMemAddress(ctypes.byref(address), mapping_bytes, 0, None, 0),
        )
        try:
            self._check("aclrtMapMem", self._library.aclrtMapMem(address, mapping_bytes, 0, handle, 0))
            descriptor = self._access_descriptor()
            self._check(
                "aclrtMemSetAccess",
                self._library.aclrtMemSetAccess(address, mapping_bytes, ctypes.byref(descriptor), 1),
            )
        except BaseException:
            self._library.aclrtUnmapMem(address)
            self._library.aclrtReleaseMemAddress(address)
            raise
        if address.value is None:
            raise AclError("aclrtReserveMemAddress returned a null address")
        return int(address.value)

    def copy_host_to_device(self, destination: int, payload: bytes) -> None:
        self._require_initialized()
        if destination <= 0 or not payload:
            raise ValueError("destination and payload must be non-empty")
        source = ctypes.create_string_buffer(payload)
        self._check(
            "aclrtMemcpy H2D",
            self._library.aclrtMemcpy(
                destination, len(payload), ctypes.addressof(source), len(payload), ACL_MEMCPY_HOST_TO_DEVICE
            ),
        )

    def copy_device_to_host(self, source: int, size: int) -> bytes:
        self._require_initialized()
        if source <= 0 or size <= 0:
            raise ValueError("source and size must be positive")
        destination = ctypes.create_string_buffer(size)
        self._check(
            "aclrtMemcpy D2H",
            self._library.aclrtMemcpy(ctypes.addressof(destination), size, source, size, ACL_MEMCPY_DEVICE_TO_HOST),
        )
        return destination.raw

    def copy_device_to_device(self, destination: int, source: int, size: int) -> int:
        self._require_initialized()
        if destination <= 0 or source <= 0 or size <= 0:
            raise ValueError("source, destination, and size must be positive")
        copy_calls = 0
        for offset in range(0, size, VALIDATION_P2P_CHUNK_BYTES):
            chunk_bytes = min(VALIDATION_P2P_CHUNK_BYTES, size - offset)
            self._check(
                "aclrtMemcpy D2D",
                self._library.aclrtMemcpy(
                    destination + offset,
                    chunk_bytes,
                    source + offset,
                    chunk_bytes,
                    ACL_MEMCPY_DEVICE_TO_DEVICE,
                ),
            )
            copy_calls += 1
        self._check("aclrtSynchronizeDevice", self._library.aclrtSynchronizeDevice())
        return copy_calls

    def close(self) -> None:
        if self._closed:
            return
        failures: list[str] = []
        if self._initialized:
            reset_result = self._library.aclrtResetDevice(self.device_id)
            if reset_result != ACL_SUCCESS:
                failures.append(f"aclrtResetDevice failed with code {reset_result}")
            finalize_result = self._library.aclFinalize()
            if finalize_result != ACL_SUCCESS:
                failures.append(f"aclFinalize failed with code {finalize_result}")
        self._initialized = False
        self._closed = True
        if failures:
            raise AclError("; ".join(failures))


class _MappedWindow:
    def __init__(self, runtime: AclVmmRuntime, address: int, handle: int, mapping_bytes: int) -> None:
        self.runtime = runtime
        self.address = address
        self.handle = handle
        self.mapping_bytes = mapping_bytes
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        failures: list[str] = []
        for operation, argument in (
            ("aclrtUnmapMem", self.address),
            ("aclrtReleaseMemAddress", self.address),
            ("aclrtFreePhysical", self.handle),
        ):
            result = getattr(self.runtime._library, operation)(argument)
            if result != ACL_SUCCESS:
                failures.append(f"{operation} failed with code {result}")
        self.closed = True
        if failures:
            raise AclError("; ".join(failures))


class OwnedVmmWindow(_MappedWindow):
    def __init__(
        self,
        runtime: AclVmmRuntime,
        address: int,
        handle: int,
        mapping_bytes: int,
        logical_bytes: int,
        shareable_handle: int,
    ) -> None:
        super().__init__(runtime, address, handle, mapping_bytes)
        self.logical_bytes = logical_bytes
        self.shareable_handle = shareable_handle

    @classmethod
    def create(cls, runtime: AclVmmRuntime, logical_bytes: int) -> OwnedVmmWindow:
        if logical_bytes <= 0:
            raise ValueError("logical_bytes must be positive")
        mapping_bytes = _align_up(logical_bytes, runtime.allocation_granularity())
        handle = ctypes.c_void_p()
        properties = runtime._physical_properties()
        runtime._check(
            "aclrtMallocPhysical",
            runtime._library.aclrtMallocPhysical(ctypes.byref(handle), mapping_bytes, ctypes.byref(properties), 0),
        )
        if handle.value is None:
            raise AclError("aclrtMallocPhysical returned a null handle")
        native_handle = int(handle.value)
        try:
            address = runtime._reserve_and_map(native_handle, mapping_bytes)
            shareable = ctypes.c_uint64()
            runtime._check(
                "aclrtMemExportToShareableHandle",
                runtime._library.aclrtMemExportToShareableHandle(
                    native_handle,
                    ACL_MEM_HANDLE_TYPE_NONE,
                    ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION,
                    ctypes.byref(shareable),
                ),
            )
        except BaseException:
            if "address" in locals():
                runtime._library.aclrtUnmapMem(address)
                runtime._library.aclrtReleaseMemAddress(address)
            runtime._library.aclrtFreePhysical(native_handle)
            raise
        if shareable.value == 0:
            runtime._library.aclrtUnmapMem(address)
            runtime._library.aclrtReleaseMemAddress(address)
            runtime._library.aclrtFreePhysical(native_handle)
            raise AclError("aclrtMemExportToShareableHandle returned zero")
        return cls(runtime, address, native_handle, mapping_bytes, logical_bytes, int(shareable.value))

    @property
    def export(self) -> VmmExport:
        return VmmExport(
            device_id=self.runtime.device_id,
            mapping_bytes=self.mapping_bytes,
            shareable_handle=self.shareable_handle,
        )


class ImportedVmmWindow(_MappedWindow):
    @classmethod
    def create(
        cls,
        runtime: AclVmmRuntime,
        exported: VmmExport,
        *,
        peer_device_id: int,
    ) -> ImportedVmmWindow:
        if exported.mapping_bytes <= 0 or exported.shareable_handle <= 0:
            raise ValueError("import descriptor must contain positive values")
        runtime._check(
            "aclrtDeviceEnablePeerAccess",
            runtime._library.aclrtDeviceEnablePeerAccess(peer_device_id, 0),
        )
        handle = ctypes.c_void_p()
        runtime._check(
            "aclrtMemImportFromShareableHandle",
            runtime._library.aclrtMemImportFromShareableHandle(
                exported.shareable_handle,
                runtime.device_id,
                ctypes.byref(handle),
            ),
        )
        if handle.value is None:
            raise AclError("aclrtMemImportFromShareableHandle returned a null handle")
        native_handle = int(handle.value)
        try:
            address = runtime._reserve_and_map(native_handle, exported.mapping_bytes)
        except BaseException:
            runtime._library.aclrtFreePhysical(native_handle)
            raise
        return cls(runtime, address, native_handle, exported.mapping_bytes)


class AclnnXorTransform:
    """Launch an in-place UINT8 XOR directly on a local VMM window."""

    API = "aclnnInplaceBitwiseXorScalar"

    def __init__(self, runtime: AclVmmRuntime, *, op_library: Any | None = None) -> None:
        runtime._require_initialized()
        self.runtime = runtime
        self._op_library = op_library if op_library is not None else self._load_op_library()
        self._configure_signatures()

    @staticmethod
    def _load_op_library() -> Any:
        try:
            return ctypes.CDLL("libopapi.so", mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise AclError(f"unable to load libopapi.so: {exc}") from exc

    def _configure_signatures(self) -> None:
        acl = self.runtime._library
        acl.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        acl.aclrtCreateStream.restype = ctypes.c_int
        acl.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
        acl.aclrtDestroyStream.restype = ctypes.c_int
        acl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
        acl.aclrtSynchronizeStream.restype = ctypes.c_int
        acl.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
        acl.aclrtMalloc.restype = ctypes.c_int
        acl.aclrtFree.argtypes = [ctypes.c_void_p]
        acl.aclrtFree.restype = ctypes.c_int

        op = self._op_library
        op.aclCreateTensor.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        op.aclCreateTensor.restype = ctypes.c_void_p
        op.aclCreateScalar.argtypes = [ctypes.c_void_p, ctypes.c_int]
        op.aclCreateScalar.restype = ctypes.c_void_p
        op.aclDestroyTensor.argtypes = [ctypes.c_void_p]
        op.aclDestroyTensor.restype = ctypes.c_int
        op.aclDestroyScalar.argtypes = [ctypes.c_void_p]
        op.aclDestroyScalar.restype = ctypes.c_int
        op.aclnnInplaceBitwiseXorScalarGetWorkspaceSize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        op.aclnnInplaceBitwiseXorScalarGetWorkspaceSize.restype = ctypes.c_int
        op.aclnnInplaceBitwiseXorScalar.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        op.aclnnInplaceBitwiseXorScalar.restype = ctypes.c_int

    def apply(self, address: int, size: int, value: int) -> DeviceTransformEvidence:
        self.runtime._require_initialized()
        if address <= 0 or size <= 0:
            raise ValueError("transform address and size must be positive")
        if not 0 <= value <= 0xFF:
            raise ValueError("UINT8 XOR value must be in [0, 255]")

        shape = (ctypes.c_int64 * 1)(size)
        stride = (ctypes.c_int64 * 1)(1)
        tensor = self._op_library.aclCreateTensor(
            shape,
            1,
            ACL_UINT8,
            stride,
            0,
            ACL_FORMAT_ND,
            shape,
            1,
            address,
        )
        if not tensor:
            raise AclError("aclCreateTensor returned null")
        scalar_value = ctypes.c_uint8(value)
        scalar = self._op_library.aclCreateScalar(ctypes.byref(scalar_value), ACL_UINT8)
        if not scalar:
            self._op_library.aclDestroyTensor(tensor)
            raise AclError("aclCreateScalar returned null")

        stream = ctypes.c_void_p()
        workspace = ctypes.c_void_p()
        workspace_bytes = ctypes.c_uint64()
        executor = ctypes.c_void_p()
        failures: list[str] = []
        started_ns = 0
        elapsed_ns = 0
        try:
            self.runtime._check(
                "aclnnInplaceBitwiseXorScalarGetWorkspaceSize",
                self._op_library.aclnnInplaceBitwiseXorScalarGetWorkspaceSize(
                    tensor,
                    scalar,
                    ctypes.byref(workspace_bytes),
                    ctypes.byref(executor),
                ),
            )
            if not executor.value:
                raise AclError("aclnnInplaceBitwiseXorScalar returned a null executor")
            self.runtime._check("aclrtCreateStream", self.runtime._library.aclrtCreateStream(ctypes.byref(stream)))
            if workspace_bytes.value:
                self.runtime._check(
                    "aclrtMalloc workspace",
                    self.runtime._library.aclrtMalloc(
                        ctypes.byref(workspace),
                        workspace_bytes.value,
                        ACL_MEM_MALLOC_HUGE_FIRST,
                    ),
                )
            started_ns = perf_counter_ns()
            self.runtime._check(
                self.API,
                self._op_library.aclnnInplaceBitwiseXorScalar(
                    workspace,
                    workspace_bytes.value,
                    executor,
                    stream,
                ),
            )
            self.runtime._check(
                "aclrtSynchronizeStream",
                self.runtime._library.aclrtSynchronizeStream(stream),
            )
            elapsed_ns = perf_counter_ns() - started_ns
        finally:
            if workspace.value:
                result = self.runtime._library.aclrtFree(workspace)
                if result != ACL_SUCCESS:
                    failures.append(f"aclrtFree workspace failed with code {result}")
            if stream.value:
                result = self.runtime._library.aclrtDestroyStream(stream)
                if result != ACL_SUCCESS:
                    failures.append(f"aclrtDestroyStream failed with code {result}")
            result = self._op_library.aclDestroyScalar(scalar)
            if result != ACL_SUCCESS:
                failures.append(f"aclDestroyScalar failed with code {result}")
            result = self._op_library.aclDestroyTensor(tensor)
            if result != ACL_SUCCESS:
                failures.append(f"aclDestroyTensor failed with code {result}")
            if failures:
                raise AclError("; ".join(failures))
        return DeviceTransformEvidence(
            api=self.API,
            elapsed_ns=elapsed_ns,
            workspace_bytes=int(workspace_bytes.value),
        )
