# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Minimal direct AIV binary loader used by the Stage 1B validation probe."""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from tools.pypto_wse_validation.acl_vmm import ACL_SUCCESS, AclError, AclVmmRuntime

ACL_RT_BINARY_LOAD_OPT_MAGIC = 2
ACL_RT_BINARY_MAGIC_ELF_AICORE = 0x43554245


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


class AclDeviceKernel:
    """Own a loaded AICore binary, one function entry, and one stream."""

    def __init__(self, runtime: AclVmmRuntime, binary_path: Path) -> None:
        runtime._require_initialized()
        try:
            binary = binary_path.read_bytes()
        except OSError as exc:
            raise AclError(f"cannot read AIV binary {binary_path}: {exc}") from exc
        if not binary:
            raise AclError(f"AIV binary is empty: {binary_path}")

        self.runtime = runtime
        self.binary_path = binary_path
        self.binary_sha256 = hashlib.sha256(binary).hexdigest()
        self._buffer = ctypes.create_string_buffer(binary)
        self._binary_handle = ctypes.c_void_p()
        self._function_handle = ctypes.c_void_p()
        self._stream = ctypes.c_void_p()
        self._launched = False
        self._synchronized = False
        self._closed = False
        self._started_ns = 0
        self._configure_signatures(runtime._library)
        self._load()

    @staticmethod
    def _configure_signatures(acl: Any) -> None:
        acl.aclrtBinaryLoadFromData.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(_AclrtBinaryLoadOptions),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        acl.aclrtBinaryLoadFromData.restype = ctypes.c_int
        acl.aclrtBinaryGetFunctionByEntry.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)]
        acl.aclrtBinaryGetFunctionByEntry.restype = ctypes.c_int
        acl.aclrtBinaryUnLoad.argtypes = [ctypes.c_void_p]
        acl.aclrtBinaryUnLoad.restype = ctypes.c_int
        acl.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        acl.aclrtCreateStream.restype = ctypes.c_int
        acl.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
        acl.aclrtDestroyStream.restype = ctypes.c_int
        acl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
        acl.aclrtSynchronizeStream.restype = ctypes.c_int
        acl.aclrtLaunchKernelWithHostArgs.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        acl.aclrtLaunchKernelWithHostArgs.restype = ctypes.c_int

    def _load(self) -> None:
        option = _AclrtBinaryLoadOption(
            option_type=ACL_RT_BINARY_LOAD_OPT_MAGIC,
            value=_AclrtBinaryLoadOptionValue(magic=ACL_RT_BINARY_MAGIC_ELF_AICORE),
        )
        options = _AclrtBinaryLoadOptions(options=ctypes.pointer(option), count=1)
        try:
            self.runtime._check(
                "aclrtBinaryLoadFromData",
                self.runtime._library.aclrtBinaryLoadFromData(
                    ctypes.addressof(self._buffer),
                    len(self._buffer) - 1,
                    ctypes.byref(options),
                    ctypes.byref(self._binary_handle),
                ),
            )
            if not self._binary_handle.value:
                raise AclError("aclrtBinaryLoadFromData returned a null handle")
            self.runtime._check(
                "aclrtBinaryGetFunctionByEntry",
                self.runtime._library.aclrtBinaryGetFunctionByEntry(
                    self._binary_handle,
                    0,
                    ctypes.byref(self._function_handle),
                ),
            )
            if not self._function_handle.value:
                raise AclError("aclrtBinaryGetFunctionByEntry returned a null handle")
            self.runtime._check(
                "aclrtCreateStream",
                self.runtime._library.aclrtCreateStream(ctypes.byref(self._stream)),
            )
            if not self._stream.value:
                raise AclError("aclrtCreateStream returned a null stream")
        except BaseException:
            self.close()
            raise

    def launch(self, arguments: ctypes.Structure) -> None:
        if self._closed:
            raise AclError("AIV kernel is closed")
        if self._launched:
            raise AclError("AIV kernel can only be launched once")
        self._started_ns = perf_counter_ns()
        self.runtime._check(
            "aclrtLaunchKernelWithHostArgs",
            self.runtime._library.aclrtLaunchKernelWithHostArgs(
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

    @property
    def launched(self) -> bool:
        return self._launched

    @property
    def closed(self) -> bool:
        return self._closed

    def synchronize(self) -> int:
        if not self._launched:
            raise AclError("AIV kernel has not been launched")
        if self._synchronized:
            raise AclError("AIV kernel has already been synchronized")
        self.runtime._check(
            "aclrtSynchronizeStream",
            self.runtime._library.aclrtSynchronizeStream(self._stream),
        )
        self._synchronized = True
        return perf_counter_ns() - self._started_ns

    def close(self) -> None:
        if self._closed:
            return
        failures: list[str] = []
        if self._stream.value:
            result = self.runtime._library.aclrtDestroyStream(self._stream)
            if result != ACL_SUCCESS:
                failures.append(f"aclrtDestroyStream failed with code {result}")
            self._stream = ctypes.c_void_p()
        if self._binary_handle.value:
            result = self.runtime._library.aclrtBinaryUnLoad(self._binary_handle)
            if result != ACL_SUCCESS:
                failures.append(f"aclrtBinaryUnLoad failed with code {result}")
            self._binary_handle = ctypes.c_void_p()
        self._closed = True
        if failures:
            raise AclError("; ".join(failures))
