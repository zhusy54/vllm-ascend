#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Two-rank probe: device-side fused allreduce vs the publish+barrier path.

    python -m torch.distributed.run --standalone --nproc_per_node=2 \\
        examples/offline_pypto_qwen3_allreduce_fused.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

INDUCTOR_ROOT = "/mnt/workspace/inductor"


def _sanitize_sys_path() -> None:
    sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry) != INDUCTOR_ROOT]


def main() -> int:
    _sanitize_sys_path()
    os.environ.setdefault("PYPTO_LIB_ROOT", str(Path(INDUCTOR_ROOT) / "pypto-lib"))
    os.environ.setdefault("PTO_PLATFORM", "a2a3")
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"

    import torch
    import torch.distributed as dist

    from vllm_ascend.models.pypto_qwen3_adapter import HIDDEN, PyptoChipSession
    from vllm_ascend.models.pypto_qwen3_tp import COMM_MARKER, build_gloo_shmem_comm

    dist.init_process_group(backend="gloo")
    rank = int(dist.get_rank())
    world = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world != 2:
        print(f"need world_size=2, got {world}", file=sys.stderr)
        return 2
    if dist.get_backend() == "hccl":
        print("HCCL process group is forbidden on this path", file=sys.stderr)
        return 2
    torch.npu.set_device(local_rank)
    device = f"npu:{local_rank}"
    session = PyptoChipSession(device_id=local_rank)
    comm = build_gloo_shmem_comm(
        rank=rank,
        world_size=world,
        device=device,
        group=dist.group.WORLD,
        session=session,
    )
    iterations = int(os.environ.get("PYPTO_QWEN3_ALLREDUCE_ITERS", "160"))
    if iterations < 1:
        raise ValueError(f"PYPTO_QWEN3_ALLREDUCE_ITERS must be positive, got {iterations}")
    max_diff = 0.0
    for step in range(iterations):
        live = step % 3 + 1
        base = torch.arange(live * HIDDEN, dtype=torch.float32, device=device).reshape(live, HIDDEN)
        local = base + float(rank + 1 + step)
        fused = comm.allreduce_sum_fused(local, boundary=f"probe_fused_{step}")
        expected = base * 2.0 + float(3 + 2 * step)
        max_diff = max(max_diff, float((fused - expected).abs().max().item()))
        if step == 0:
            ref = comm.allreduce_sum(local, boundary="probe_host")
            max_diff = max(max_diff, float((fused - ref).abs().max().item()))
    print(
        f"PYPTO_QWEN3_FUSED_AR rank={rank} iterations={iterations} "
        f"credits={comm.signal_credit} maxdiff={max_diff}",
        flush=True,
    )
    print(COMM_MARKER, "fused_allreduce_probe", flush=True)
    ok = max_diff < 1e-3 and comm.signal_credit == iterations * 2
    global_ok = torch.tensor(int(ok), dtype=torch.int32)
    dist.all_reduce(global_ok, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            "PYPTO_QWEN3_FUSED_ALLREDUCE_OK"
            if int(global_ok.item()) == 1
            else "PYPTO_QWEN3_FUSED_ALLREDUCE_FAIL",
            flush=True,
        )
    dist.destroy_process_group()
    return 0 if int(global_ok.item()) == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
