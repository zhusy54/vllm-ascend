# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.pypto_wse_validation.collect_stage0 import (
    CommandResult,
    collect_stage0,
    parse_device_list,
    parse_npu_smi,
)

NPU_SMI_OUTPUT = """\
+------------------------------+
| npu-smi 25.7.rc1.6 Version: 25.7.rc1.6 |
+==============================+
| 0 | Ascend950PR | OK | 0 |
| 1 | Ascend950PR | OK | 0 |
+------------------------------+
"""


def _make_simpler_tree(root: Path) -> None:
    files = {
        "python/simpler/worker.py": "def remote_malloc(): pass\ndef remote_export(): pass\ndef remote_import(): pass\n",
        "python/simpler/remote_l3_protocol.py": 'HOST_TCP_TRANSPORT_PROFILE = "host_tcp"\n',
        "src/a5/platform/onboard/host/comm_hccl.cpp": (
            "aclrtMemExportToShareableHandle();\naclrtMemImportFromShareableHandle();\n"
        ),
        "src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/urma/urma_completion_kernel.h": (
            "#if defined(PTO_URMA_SUPPORTED)\n#endif\n"
        ),
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _runner(*, dirty: bool = False, npu_output: str = NPU_SMI_OUTPUT):
    def run(argv) -> CommandResult:
        command = tuple(str(item) for item in argv)
        if command == ("npu-smi", "info"):
            return CommandResult(command, 0, npu_output, "")
        if command[-2:] == ("rev-parse", "HEAD"):
            return CommandResult(command, 0, "407438ef677b9a5787c4d645e8cae4f7d4919d4e\n", "")
        if "status" in command:
            return CommandResult(command, 0, " M python/simpler/worker.py\n" if dirty else "", "")
        raise AssertionError(f"unexpected command: {command}")

    return run


def _versions(name: str) -> str:
    return {"pypto": "0.2.1", "simpler": "0.1.0"}[name]


def test_parse_device_list_accepts_ranges_and_rejects_invalid_values() -> None:
    assert parse_device_list("0,1") == (0, 1)
    assert parse_device_list("4-5") == (4, 5)
    for invalid in ("0", "0,0", "2-1", "0,a", "0,,1"):
        with pytest.raises(ValueError):
            parse_device_list(invalid)


def test_parse_npu_smi_extracts_device_identity_and_health() -> None:
    assert parse_npu_smi(NPU_SMI_OUTPUT) == (
        {"device_id": 0, "name": "Ascend950PR", "health": "OK"},
        {"device_id": 1, "name": "Ascend950PR", "health": "OK"},
    )


def test_collect_stage0_writes_sanitized_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    simpler_root = tmp_path / "private" / "simpler"
    artifact_dir = tmp_path / "artifacts"
    _make_simpler_tree(simpler_root)
    cann = tmp_path / "cann-9.2.0"
    cann.mkdir()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann))

    artifacts = collect_stage0(
        devices=(0, 1),
        simpler_root=simpler_root,
        artifact_dir=artifact_dir,
        runner=_runner(),
        version_lookup=_versions,
        collected_at="2026-09-08T00:00:00+00:00",
    )

    environment = artifacts["environment.json"]
    assert environment["ready_for_bootstrap"] is True
    assert environment["hardware"]["requested_devices_healthy"] is True
    assert environment["reference"]["root"] == "<SIMPLER_ROOT>"
    assert artifacts["transport.json"]["declared_scope"] == "SIMULATION"
    assert artifacts["transport.json"]["actual_backend"] == "NOT_EXERCISED"
    capability_by_name = {item["capability"]: item for item in artifacts["capabilities.json"]["capabilities"]}
    assert capability_by_name["host_tcp_device_memory"]["status"] == "unsupported"
    assert capability_by_name["a5_urma"]["status"] == "unsupported"

    serialized = ""
    for name in artifacts:
        persisted = json.loads((artifact_dir / name).read_text())
        assert persisted == artifacts[name]
        serialized += (artifact_dir / name).read_text()
    assert str(simpler_root) not in serialized


def test_collect_stage0_fails_readiness_for_dirty_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    simpler_root = tmp_path / "simpler"
    _make_simpler_tree(simpler_root)
    cann = tmp_path / "cann-9.2.0"
    cann.mkdir()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann))

    artifacts = collect_stage0(
        devices=(0, 1),
        simpler_root=simpler_root,
        artifact_dir=tmp_path / "artifacts",
        runner=_runner(dirty=True),
        version_lookup=_versions,
    )
    assert artifacts["environment.json"]["ready_for_bootstrap"] is False
    assert artifacts["environment.json"]["reference"]["code_paths_clean"] is False


def test_collect_stage0_fails_readiness_for_missing_or_unhealthy_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simpler_root = tmp_path / "simpler"
    _make_simpler_tree(simpler_root)
    cann = tmp_path / "cann-9.2.0"
    cann.mkdir()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann))
    unhealthy = NPU_SMI_OUTPUT.replace("| 1 | Ascend950PR | OK", "| 1 | Ascend950PR | Warning")

    artifacts = collect_stage0(
        devices=(0, 1),
        simpler_root=simpler_root,
        artifact_dir=tmp_path / "artifacts",
        runner=_runner(npu_output=unhealthy),
        version_lookup=_versions,
    )
    assert artifacts["environment.json"]["ready_for_bootstrap"] is False
    assert artifacts["environment.json"]["hardware"]["requested_devices_healthy"] is False
