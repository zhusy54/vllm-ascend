# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Architectural tests for the prototype's intended dependency direction."""

from __future__ import annotations

import ast
from pathlib import Path

from pypto_test.infrastructure.memory import AscendVmmMemoryProvider
from pypto_test.pseudo_pypto.backend import AclExecutionRuntime
from pypto_test.pseudo_pypto.communication import (
    NPU_SHARED_WINDOW_BYTES,
    WSE_SHARED_WINDOW_BYTES,
    NpuCommunicationBinding,
    NpuDeviceCommunication,
    WseCommunicationBinding,
    WseDeviceCommunication,
)


def test_pseudo_pypto_does_not_import_external_or_validation_packages():
    package = Path(__file__).parents[1] / "pseudo_pypto"
    forbidden = ("pypto_test.infrastructure", "pypto_test.validation", "tools.pypto_wse_validation")
    for source_path in package.glob("*.py"):
        tree = ast.parse(source_path.read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(name.startswith(forbidden) for name in imported), source_path


def test_external_provider_and_pypto_runtime_have_disjoint_capabilities():
    provider_methods = set(dir(AscendVmmMemoryProvider))
    pypto_methods = set(dir(AclExecutionRuntime))
    assert {"initialize_host", "allocate_shared_window", "attach_peer", "binding", "release"} <= provider_methods
    assert {"allocate_local", "copy_host_to_device", "copy_device_to_host", "launch_kernel"} <= pypto_methods
    assert not {"copy_host_to_device", "copy_device_to_host", "launch_kernel"} & provider_methods
    assert not {"initialize_host", "allocate_shared_window", "attach_peer", "release"} & pypto_methods


def test_device_communication_resolves_fixed_process_local_abi():
    npu = NpuDeviceCommunication.from_binding(
        NpuCommunicationBinding(1, 1_000_000, NPU_SHARED_WINDOW_BYTES, 3_000_000, WSE_SHARED_WINDOW_BYTES)
    )
    wse = WseDeviceCommunication.from_binding(
        WseCommunicationBinding(1, 3_000_000, WSE_SHARED_WINDOW_BYTES, 1_000_000, NPU_SHARED_WINDOW_BYTES)
    )
    assert npu.remote_b_input == wse.local_b_input
    assert npu.remote_submission_signal == wse.local_submission_signal
    assert wse.remote_b_output == npu.local_b_output
    assert wse.remote_completion_signal == npu.local_completion_signal
