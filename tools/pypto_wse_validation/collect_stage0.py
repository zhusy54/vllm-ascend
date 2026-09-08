# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Collect stage-0 environment and transport capability evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_SIMPLER_ROOT = Path("../simpler-mix-spmd-sync-start")
NPU_ROW = re.compile(r"^\|\s*(?P<device>\d+)\s*\|\s*(?P<name>Ascend\S+)\s*\|\s*(?P<health>\S+)\s*\|")
DRIVER_VERSION = re.compile(r"npu-smi\s+(?P<version>\S+)")
CANN_VERSION = re.compile(r"cann-(?P<version>[0-9][A-Za-z0-9_.-]*)$")


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str]], CommandResult]
VersionLookup = Callable[[str], str]


def run_command(argv: Sequence[str]) -> CommandResult:
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True)
    except OSError as exc:
        return CommandResult(tuple(argv), 127, "", f"{type(exc).__name__}: {exc}")
    return CommandResult(tuple(argv), result.returncode, result.stdout, result.stderr)


def parse_device_list(value: str) -> tuple[int, ...]:
    devices: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("device list contains an empty entry")
        if "-" in part:
            bounds = part.split("-", maxsplit=1)
            if len(bounds) != 2 or not all(item.isdigit() for item in bounds):
                raise ValueError(f"invalid device range {part!r}")
            start, end = (int(item) for item in bounds)
            if start > end:
                raise ValueError(f"device range start exceeds end: {part!r}")
            devices.extend(range(start, end + 1))
        elif part.isdigit():
            devices.append(int(part))
        else:
            raise ValueError(f"invalid device id {part!r}")
    if len(devices) != len(set(devices)):
        raise ValueError("device ids must be unique")
    if len(devices) != 2:
        raise ValueError(f"NPU surrogate profile requires exactly 2 devices, got {len(devices)}")
    return tuple(devices)


def parse_npu_smi(output: str) -> tuple[dict[str, Any], ...]:
    devices: list[dict[str, Any]] = []
    for line in output.splitlines():
        match = NPU_ROW.match(line)
        if match is None:
            continue
        devices.append(
            {
                "device_id": int(match.group("device")),
                "name": match.group("name"),
                "health": match.group("health"),
            }
        )
    return tuple(devices)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _command_evidence(result: CommandResult) -> dict[str, Any]:
    combined = result.stdout + result.stderr
    return {
        "available": result.returncode != 127,
        "returncode": result.returncode,
        "output_sha256": _sha256_text(combined),
    }


def _version_or_unknown(name: str, lookup: VersionLookup) -> str:
    try:
        return lookup(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _source_signals(simpler_root: Path) -> dict[str, bool]:
    worker = _read_text(simpler_root / "python/simpler/worker.py")
    remote_protocol = _read_text(simpler_root / "python/simpler/remote_l3_protocol.py")
    comm_vmm = _read_text(simpler_root / "src/a5/platform/onboard/host/comm_hccl.cpp")
    urma = _read_text(
        simpler_root / "src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/urma/urma_completion_kernel.h"
    )
    source_tree = "\n".join((worker, remote_protocol, comm_vmm, urma))
    return {
        "remote_buffer_api_present": all(
            token in worker for token in ("def remote_malloc", "def remote_export", "def remote_import")
        ),
        "host_tcp_profile_present": 'HOST_TCP_TRANSPORT_PROFILE = "host_tcp"' in remote_protocol,
        "a5_vmm_export_import_present": all(
            token in comm_vmm for token in ("aclrtMemExportToShareableHandle", "aclrtMemImportFromShareableHandle")
        ),
        "urma_code_present": "PTO_URMA_SUPPORTED" in urma,
        "urma_capability_defined": "#define PTO_URMA_SUPPORTED" in source_tree,
    }


def _capability_matrix(signals: dict[str, bool]) -> list[dict[str, str]]:
    vmm_note = (
        "A5 VMM export/import implementation found; runtime execution is still required"
        if signals["a5_vmm_export_import_present"]
        else "A5 VMM export/import implementation was not found"
    )
    urma_status = "unknown" if signals["urma_capability_defined"] else "unsupported"
    urma_note = (
        "PTO_URMA_SUPPORTED is defined; runtime execution is still required"
        if signals["urma_capability_defined"]
        else "URMA code is gated by PTO_URMA_SUPPORTED with no source definition"
    )
    return [
        {
            "capability": "device_visibility",
            "status": "verified",
            "evidence": "requested devices are checked against successful npu-smi output",
        },
        {
            "capability": "independent_device_runtime_init",
            "status": "unknown",
            "evidence": "requires the two-process bootstrap probe",
        },
        {
            "capability": "device_memory_allocate_free",
            "status": "unknown",
            "evidence": "implementation presence is not runtime evidence",
        },
        {
            "capability": "memory_register_unregister",
            "status": "unknown",
            "evidence": "no stage-0 runtime call has been executed",
        },
        {
            "capability": "export_import_remote_handle",
            "status": "unknown",
            "evidence": vmm_note,
        },
        {
            "capability": "host_tcp_device_memory",
            "status": "unsupported",
            "evidence": "host_tcp remote buffers use host-side session storage",
        },
        {
            "capability": "a5_urma",
            "status": urma_status,
            "evidence": urma_note,
        },
        {
            "capability": "npu_to_surrogate_data",
            "status": "unknown",
            "evidence": "deferred to stage 1A",
        },
        {
            "capability": "surrogate_to_npu_data",
            "status": "unknown",
            "evidence": "deferred to stage 1A",
        },
        {
            "capability": "device_side_submit_completion",
            "status": "unknown",
            "evidence": "deferred to stage 1B and requires device trace",
        },
        {
            "capability": "visibility_fence_flush",
            "status": "unknown",
            "evidence": "exact API and scope are not yet mapped",
        },
        {
            "capability": "backend_identity_counters",
            "status": "unknown",
            "evidence": "query or trace source is not yet mapped",
        },
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def collect_stage0(
    *,
    devices: tuple[int, int],
    simpler_root: Path,
    artifact_dir: Path,
    runner: CommandRunner = run_command,
    version_lookup: VersionLookup = importlib.metadata.version,
    collected_at: str | None = None,
) -> dict[str, dict[str, Any]]:
    if devices[0] == devices[1]:
        raise ValueError("Attention and WSE surrogate devices must be distinct")
    resolved_root = simpler_root.resolve()
    npu_result = runner(("npu-smi", "info"))
    visible_devices = parse_npu_smi(npu_result.stdout) if npu_result.returncode == 0 else ()
    visible_by_id = {item["device_id"]: item for item in visible_devices}
    requested = [visible_by_id.get(device_id) for device_id in devices]
    requested_healthy = all(item is not None and item["health"] == "OK" for item in requested)

    git_revision = runner(("git", "-C", str(resolved_root), "rev-parse", "HEAD"))
    git_status = runner(
        (
            "git",
            "-C",
            str(resolved_root),
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "python",
            "src",
            "simpler_setup",
            "CMakeLists.txt",
            "pyproject.toml",
        )
    )
    source_clean = git_status.returncode == 0 and not git_status.stdout.strip()
    source_revision = git_revision.stdout.strip() if git_revision.returncode == 0 else "UNKNOWN"

    cann_root = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    try:
        resolved_cann = cann_root.resolve(strict=True)
        cann_version_match = CANN_VERSION.search(resolved_cann.name)
        cann_version = cann_version_match.group("version") if cann_version_match else "UNKNOWN"
    except OSError:
        resolved_cann = cann_root
        cann_version = "NOT_FOUND"

    driver_match = DRIVER_VERSION.search(npu_result.stdout)
    signals = _source_signals(resolved_root)
    capability_matrix = _capability_matrix(signals)
    collected = collected_at or datetime.now(UTC).isoformat()
    environment = {
        "schema_version": SCHEMA_VERSION,
        "collected_at": collected,
        "profile": "NPU_SURROGATE",
        "host": {
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "hardware": {
            "requested_devices": [
                {"role": "ATTENTION", "device_id": devices[0]},
                {"role": "WSE_SURROGATE", "device_id": devices[1]},
            ],
            "visible_devices": list(visible_devices),
            "requested_devices_healthy": requested_healthy,
        },
        "software": {
            "driver": driver_match.group("version") if driver_match else "UNKNOWN",
            "cann": cann_version,
            "pypto": _version_or_unknown("pypto", version_lookup),
            "simpler": _version_or_unknown("simpler", version_lookup),
        },
        "reference": {
            "root": "<SIMPLER_ROOT>",
            "revision": source_revision,
            "code_paths_clean": source_clean,
        },
        "commands": {
            "npu_smi": _command_evidence(npu_result),
            "simpler_revision": _command_evidence(git_revision),
            "simpler_code_status": _command_evidence(git_status),
        },
        "ready_for_bootstrap": bool(
            npu_result.returncode == 0
            and requested_healthy
            and source_revision != "UNKNOWN"
            and source_clean
            and cann_version not in {"UNKNOWN", "NOT_FOUND"}
        ),
    }
    transport = {
        "schema_version": SCHEMA_VERSION,
        "collected_at": collected,
        "profile": "NPU_SURROGATE",
        "declared_scope": "SIMULATION",
        "control_plane": "TCP_LOOPBACK_NOT_YET_EXERCISED",
        "data_plane": "NOT_EXERCISED",
        "local_fallback_allowed": False,
        "actual_backend": "NOT_EXERCISED",
        "source_signals": signals,
    }
    capabilities = {
        "schema_version": SCHEMA_VERSION,
        "collected_at": collected,
        "profile": "NPU_SURROGATE",
        "capabilities": capability_matrix,
    }
    artifacts = {
        "environment.json": environment,
        "transport.json": transport,
        "capabilities.json": capabilities,
    }
    for name, value in artifacts.items():
        _write_json(artifact_dir / name, value)
    return artifacts


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True, help="two device ids, for example 0,1 or 0-1")
    parser.add_argument("--simpler-root", type=Path, default=DEFAULT_SIMPLER_ROOT)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        devices = parse_device_list(args.devices)
        artifacts = collect_stage0(
            devices=(devices[0], devices[1]),
            simpler_root=args.simpler_root,
            artifact_dir=args.artifact_dir,
        )
    except (OSError, ValueError) as exc:
        print(f"stage-0 collection failed: {exc}", file=sys.stderr)
        return 2
    environment = artifacts["environment.json"]
    print(args.artifact_dir)
    return 0 if environment["ready_for_bootstrap"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
