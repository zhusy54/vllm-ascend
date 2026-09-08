# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import ctypes

import pytest

from tests.ut.tools.test_pypto_wse_acl_vmm import _FakeAcl, _FakeFunction
from tools.pypto_wse_validation.acl_kernel import AclDeviceKernel
from tools.pypto_wse_validation.acl_vmm import AclError, AclVmmRuntime


class _Arguments(ctypes.Structure):
    _fields_ = [("address", ctypes.c_uint64)]


def _runtime(tmp_path):
    library = _FakeAcl()
    for name in (
        "aclrtBinaryLoadFromData",
        "aclrtBinaryGetFunctionByEntry",
        "aclrtBinaryUnLoad",
        "aclrtLaunchKernelWithHostArgs",
    ):
        setattr(library, name, _FakeFunction(name, library))
    runtime = AclVmmRuntime(0, library=library)
    runtime.initialize()
    binary_path = tmp_path / "kernel.o"
    binary_path.write_bytes(b"AIV binary")
    return runtime, library, binary_path


def test_device_kernel_load_launch_sync_and_close(tmp_path):
    runtime, library, binary_path = _runtime(tmp_path)
    kernel = AclDeviceKernel(runtime, binary_path)
    kernel.launch(_Arguments(address=0x1234))
    assert kernel.synchronize() >= 0
    kernel.close()
    assert [
        name
        for name, _ in library.calls
        if name
        in {
            "aclrtBinaryLoadFromData",
            "aclrtBinaryGetFunctionByEntry",
            "aclrtCreateStream",
            "aclrtLaunchKernelWithHostArgs",
            "aclrtSynchronizeStream",
            "aclrtDestroyStream",
            "aclrtBinaryUnLoad",
        }
    ] == [
        "aclrtBinaryLoadFromData",
        "aclrtBinaryGetFunctionByEntry",
        "aclrtCreateStream",
        "aclrtLaunchKernelWithHostArgs",
        "aclrtSynchronizeStream",
        "aclrtDestroyStream",
        "aclrtBinaryUnLoad",
    ]
    runtime.close()


def test_device_kernel_rejects_second_launch(tmp_path):
    runtime, _, binary_path = _runtime(tmp_path)
    kernel = AclDeviceKernel(runtime, binary_path)
    kernel.launch(_Arguments(address=0x1234))
    with pytest.raises(AclError, match="only be launched once"):
        kernel.launch(_Arguments(address=0x1234))
    kernel.synchronize()
    kernel.close()
    runtime.close()
