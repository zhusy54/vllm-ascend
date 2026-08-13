#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Two-rank Megatron TP generate: Gloo + torch_npu SHMEM + pypto collectives.

Launch::

    source /mnt/workspace/inductor/shmem/install/set_env.sh
    python -m torch.distributed.run --standalone --nproc_per_node=2 \\
        examples/offline_pypto_qwen3_14b_tp2.py

Does not use vLLM EngineCore (this container's /dev/shm is 64MiB).
Each rank keeps its Megatron shard (``wq`` last-dim 2560) and runs the
TP host. pypto allreduce is used after ``o_proj`` and ``down_proj``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL_PATH = os.environ.get("QWEN3_14B_PATH", "/mnt/workspace/inductor/models/Qwen3-14B")
INDUCTOR_ROOT = "/mnt/workspace/inductor"


def _sanitize_sys_path() -> None:
    cleaned = []
    for entry in sys.path:
        if os.path.abspath(entry) == INDUCTOR_ROOT:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


def main() -> int:
    _sanitize_sys_path()
    os.environ.setdefault("PYPTO_LIB_ROOT", str(Path(INDUCTOR_ROOT) / "pypto-lib"))
    os.environ.setdefault("PTO_PLATFORM", "a2a3")
    os.environ.setdefault("QWEN3_PA_BLOCK_DIM", "20")
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from vllm_ascend.models.pypto_qwen3_adapter import (
        PyptoChipSession,
        build_rope_tables,
        load_hf_state_from_dir,
        pack_official_weights,
        pin_weight_bundle_dtypes,
    )
    from vllm_ascend.models.pypto_qwen3_tp import (
        COMM_MARKER,
        build_gloo_shmem_comm,
        describe_tp_compute_args,
        select_compute_bundle,
    )
    from vllm_ascend.models.pypto_qwen3_tp_runner import PyptoTpRunner

    if not Path(MODEL_PATH).exists():
        print(f"model path missing: {MODEL_PATH}", file=sys.stderr)
        return 2

    dist.init_process_group(backend="gloo")
    rank = int(dist.get_rank())
    world = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world != 2:
        print(f"need world_size=2, got {world}", file=sys.stderr)
        return 2
    print(f"PYPTO_QWEN3_TP2 backend={dist.get_backend()} rank={rank}/{world}", flush=True)
    if dist.get_backend() == "hccl":
        print("HCCL process group is forbidden on this path", file=sys.stderr)
        return 2

    torch.npu.set_device(local_rank)
    device = f"npu:{local_rank}"

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "1+1等于几？只回答数字。"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = torch.tensor(tok.encode(prompt, add_special_tokens=False), dtype=torch.int32)
    ntok = int(input_ids.numel())

    bundle = pin_weight_bundle_dtypes(pack_official_weights(load_hf_state_from_dir(MODEL_PATH).items()))
    shard = select_compute_bundle(bundle, rank, world)
    del bundle
    print(
        f"PYPTO_QWEN3_TP_SHARD rank={rank}/{world} "
        f"wq={tuple(shard.wq.shape)} wo={tuple(shard.wo.shape)} "
        f"w_down={tuple(shard.w_down.shape)}",
        flush=True,
    )
    print(f"PYPTO_QWEN3_ARGS {describe_tp_compute_args(shard)}", flush=True)
    if int(shard.wq.shape[-1]) != 2560:
        print(f"expected wq last-dim 2560, got {tuple(shard.wq.shape)}", file=sys.stderr)
        dist.destroy_process_group()
        return 2
    shard = type(shard)(**{name: tensor.to(device) for name, tensor in shard.__dict__.items()})
    rope_cos, rope_sin = build_rope_tables()
    rope_cos = rope_cos.to(device)
    rope_sin = rope_sin.to(device)

    session = PyptoChipSession(device_id=local_rank)
    comm = build_gloo_shmem_comm(
        rank=rank,
        world_size=world,
        device=device,
        group=dist.group.WORLD,
        session=session,
    )
    runner = PyptoTpRunner(shard, comm, session, rope_cos, rope_sin)
    runner.compile()

    logits = runner.prefill(input_ids.to(device))
    next_id = int(logits[0].argmax().item())
    ids = [next_id]
    max_tokens = int(os.environ.get("PYPTO_MAX_TOKENS", "32"))
    seq = ntok + 1
    for _ in range(max_tokens - 1):
        step_ids = torch.tensor([ids[-1]], dtype=torch.int32, device=device)
        logits = runner.decode(step_ids, seq)
        nxt = int(logits[0].argmax().item())
        ids.append(nxt)
        seq += 1
        if nxt == tok.eos_token_id:
            break
    text = tok.decode(ids, skip_special_tokens=True).strip()
    if rank == 0:
        print("OUTPUT:", text, flush=True)
        if "2" not in text:
            print("greedy text does not contain 2", file=sys.stderr)
            dist.destroy_process_group()
            return 1
        print("PYPTO_QWEN3_14B_TP2_GENERATE_OK", flush=True)
        print(COMM_MARKER, "generate_done", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
