# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import ctypes

import pytest

from tools.pypto_wse_validation.acl_vmm import (
    ACL_MEMCPY_DEVICE_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    AclError,
    AclVmmRuntime,
    VmmExport,
    _AclMemAccessDesc,
    _AclMemLocation,
    _AclPhysicalMemProp,
    _align_up,
)


class _FakeFunction:
    def __init__(self, name, owner):
        self.name = name
        self.owner = owner
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.owner.calls.append((self.name, args))
        if self.name == "aclrtMemGetAllocationGranularity":
            ctypes.cast(args[2], ctypes.POINTER(ctypes.c_size_t))[0] = 2 * 1024 * 1024
        elif self.name == "aclrtMallocPhysical":
            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = 0x1000
        elif self.name == "aclrtReserveMemAddress":
            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = self.owner.next_address
            self.owner.next_address += 0x10000000
        elif self.name == "aclrtMemExportToShareableHandle":
            ctypes.cast(args[3], ctypes.POINTER(ctypes.c_uint64))[0] = 0xABC
        elif self.name == "aclrtMemImportFromShareableHandle":
            ctypes.cast(args[2], ctypes.POINTER(ctypes.c_void_p))[0] = 0x2000
        return self.owner.results.get(self.name, 0)


class _FakeAcl:
    _FUNCTIONS = (
        "aclInit",
        "aclFinalize",
        "aclrtSetDevice",
        "aclrtResetDevice",
        "aclrtSynchronizeDevice",
        "aclrtDeviceEnablePeerAccess",
        "aclrtMemGetAllocationGranularity",
        "aclrtMallocPhysical",
        "aclrtFreePhysical",
        "aclrtReserveMemAddress",
        "aclrtReleaseMemAddress",
        "aclrtMapMem",
        "aclrtUnmapMem",
        "aclrtMemSetAccess",
        "aclrtMemExportToShareableHandle",
        "aclrtMemImportFromShareableHandle",
        "aclrtMemcpy",
    )

    def __init__(self):
        self.calls = []
        self.results = {}
        self.next_address = 0x3000
        for name in self._FUNCTIONS:
            setattr(self, name, _FakeFunction(name, self))


def _runtime() -> tuple[AclVmmRuntime, _FakeAcl]:
    library = _FakeAcl()
    runtime = AclVmmRuntime(2, access_device_id=6, library=library)
    runtime.initialize()
    return runtime, library


def test_acl_struct_layout_matches_cann_abi():
    assert ctypes.sizeof(_AclMemLocation) == 8
    assert ctypes.sizeof(_AclPhysicalMemProp) == 32
    assert ctypes.sizeof(_AclMemAccessDesc) == 24


def test_alignment_rejects_invalid_values():
    assert _align_up(1, 64) == 64
    assert _align_up(65, 64) == 128
    with pytest.raises(ValueError):
        _align_up(0, 64)


def test_allocate_export_and_import_use_vmm_and_peer_access():
    runtime, library = _runtime()
    owner = runtime.allocate_window(1024 * 1024)
    peer = runtime.import_window(owner.export, peer_device_id=3)
    try:
        assert owner.mapping_bytes == 2 * 1024 * 1024
        assert owner.export == VmmExport(device_id=2, mapping_bytes=2 * 1024 * 1024, shareable_handle=0xABC)
        assert peer.address != owner.address
        access_calls = [args for name, args in library.calls if name == "aclrtMemSetAccess"]
        descriptor = ctypes.cast(access_calls[0][2], ctypes.POINTER(_AclMemAccessDesc))[0]
        assert descriptor.location.id == 6
        assert any(name == "aclrtDeviceEnablePeerAccess" and args[0] == 3 for name, args in library.calls)
    finally:
        peer.close()
        owner.close()
        runtime.close()


def test_copy_methods_use_exact_direction_and_explicit_sync():
    runtime, library = _runtime()
    runtime.copy_host_to_device(0x1000, b"abcd")
    runtime.copy_device_to_host(0x1000, 4)
    runtime.copy_device_to_device(0x2000, 0x1000, 4)
    kinds = [args[4] for name, args in library.calls if name == "aclrtMemcpy"]
    assert kinds == [ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST, ACL_MEMCPY_DEVICE_TO_DEVICE]
    assert any(name == "aclrtSynchronizeDevice" for name, _ in library.calls)
    runtime.close()


def test_runtime_close_is_idempotent_and_surfaces_cleanup_failure():
    runtime, library = _runtime()
    library.results["aclrtResetDevice"] = 17
    with pytest.raises(AclError, match="aclrtResetDevice failed with code 17"):
        runtime.close()
    runtime.close()
    assert sum(name == "aclFinalize" for name, _ in library.calls) == 1
