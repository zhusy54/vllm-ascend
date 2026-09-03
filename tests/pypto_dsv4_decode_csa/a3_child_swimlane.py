# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Capture the current decode-CSA child-task swimlane on an A2/A3 device.

Borrowed-device L1 deliberately disables Simpler DFX.  This diagnostic reuses
the exact compiled TRB callable and static B4/S8/C8191 argument contract in a
fresh L2 execution, where chip-swimlane collection is supported.  Its absolute
duration is not an L1 performance result; the artifact is for child placement,
dependency, and scheduler-gap analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.pypto_dsv4_decode_csa.fixtures import build_zero_fixture
from vllm_ascend.ops._pypto_dsv4_csa import DecodeCSAProgramSpec

TRB_RUNTIME = "tensormap_and_ringbuffer"
DEFAULT_START_POSITION = 8191


def capture_child_swimlane(*, device: int, start_position: int) -> Path:
    """Run one full chip-swimlane capture and return its DFX directory."""
    import torch
    import torch_npu
    from pypto.runtime import RunConfig

    if device < 0:
        raise ValueError("device must be non-negative")
    if start_position < 0:
        raise ValueError("start_position must be non-negative")

    torch_npu.npu.set_device(device)
    spec = DecodeCSAProgramSpec(batch=4)
    fixture = build_zero_fixture(
        spec,
        # The L2 diagnostic runner owns device memory and accepts host tensors;
        # it copies them to ``device`` and copies writable arguments back during
        # finalize.  The production L1 path continues to consume NPU tensors.
        device=torch.device("cpu"),
        runtime=TRB_RUNTIME,
        start_positions=(start_position,) * spec.batch,
    )
    compile_config = RunConfig(
        platform="a2a3",
        device_id=device,
        runtime=TRB_RUNTIME,
    )
    compiled = fixture.program.compile(config=compile_config)
    ordered_arguments = tuple(fixture.arguments[name] for name in compiled.param_names)

    # PyPTO's onboard L2 runner performs the dependency and clean timing passes
    # separately, then invokes Simpler's converter to emit a Perfetto JSON.
    dfx_config = RunConfig(
        platform="a2a3",
        device_id=device,
        runtime=TRB_RUNTIME,
        enable_l2_swimlane=True,
    )
    compiled(*ordered_arguments, config=dfx_config)

    if int(torch.count_nonzero(fixture.output)) != 0:
        raise AssertionError("zero fixture produced a non-zero output")

    dfx_dir = compiled.output_dir / "dfx_outputs"
    required = (
        dfx_dir / "chip_swimlane_records.json",
        dfx_dir / "deps.json",
    )
    missing = tuple(path.name for path in required if not path.is_file())
    merged = tuple(dfx_dir.glob("merged_swimlane_*.json"))
    if missing or len(merged) != 1:
        raise RuntimeError(
            f"incomplete child-swimlane output: missing={missing}, merged_count={len(merged)}, directory={dfx_dir}"
        )
    return dfx_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--start-position", type=int, default=DEFAULT_START_POSITION)
    args = parser.parse_args()
    output = capture_child_swimlane(
        device=args.device,
        start_position=args.start_position,
    )
    print(f"child swimlane directory: {output}")


if __name__ == "__main__":
    main()
