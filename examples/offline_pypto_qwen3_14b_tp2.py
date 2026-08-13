#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Two-rank TP generate: Gloo + torch_npu SHMEM + pypto allreduce.

Launch::

    source /mnt/workspace/inductor/shmem/install/set_env.sh
    torchrun --standalone --nproc_per_node=2 \\
        examples/offline_pypto_qwen3_14b_tp2.py

Does not use vLLM EngineCore (this container's /dev/shm is 64MiB).
Uses the shipped adapter + TP shard/gather + pypto collective.
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
        HEAD_DIM,
        HIDDEN,
        NUM_LAYERS,
        PADDED_VOCAB,
        PAGE_SIZE,
        REAL_VOCAB,
        SAMPLED_IDS_PAD,
        STAGE_DECODE,
        STAGE_PREFILL,
        PyptoChipSession,
        build_decode_kernel_args,
        build_prefill_kernel_args,
        build_rope_tables,
        compact_vllm_kv_for_contract,
        ensure_pypto_lib_on_path,
        invoke_pypto_kernel,
        load_hf_state_from_dir,
        pack_official_weights,
        pin_weight_bundle_dtypes,
        scatter_contract_kv_to_vllm,
        slice_real_vocab_logits,
    )
    from vllm_ascend.models.pypto_qwen3_tp import (
        COMM_MARKER,
        build_gloo_shmem_comm,
        gloo_gather_contract_bundle,
        shard_contract_bundle,
    )

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

    os.environ.setdefault("QWEN3_PA_BLOCK_DIM", "20")
    ensure_pypto_lib_on_path()
    from contract.registry import get_contract
    from pypto.backend import BackendType, set_backend_type

    set_backend_type(BackendType.Ascend910B)
    kernels = get_contract("qwen3", "14b").load_kernels().functions

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
    shard = shard_contract_bundle(bundle, rank, world)
    print(
        f"PYPTO_QWEN3_TP_SHARD rank={rank}/{world} "
        f"wq={tuple(shard.wq.shape)} wo={tuple(shard.wo.shape)}",
        flush=True,
    )
    bundle = gloo_gather_contract_bundle(shard, rank=rank, world=world, group=dist.group.WORLD)
    bundle = type(bundle)(**{name: tensor.to(device) for name, tensor in bundle.__dict__.items()})
    rope_cos, rope_sin = build_rope_tables()
    rope_cos = rope_cos.to(device)
    rope_sin = rope_sin.to(device)

    session = PyptoChipSession(device_id=local_rank)
    comm = build_gloo_shmem_comm(
        rank=rank,
        world_size=world,
        device=device,
        cols=HIDDEN,
        group=dist.group.WORLD,
        session=session,
    )

    n_pages = 1
    layer_kvs = [
        (
            torch.zeros(n_pages, PAGE_SIZE, 8, HEAD_DIM, dtype=torch.bfloat16, device=device),
            torch.zeros(n_pages, PAGE_SIZE, 8, HEAD_DIM, dtype=torch.bfloat16, device=device),
        )
        for _ in range(NUM_LAYERS)
    ]

    def _step(stage: str, token_ids: torch.Tensor, seq: int) -> torch.Tensor:
        seq_lens = torch.tensor([seq], dtype=torch.int32, device=device)
        query_start_loc = torch.tensor([0, int(token_ids.numel())], dtype=torch.int32, device=device)
        block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
        if stage == STAGE_PREFILL:
            slot_mapping = torch.arange(int(token_ids.numel()), dtype=torch.int32, device=device)
        else:
            slot_mapping = torch.tensor([seq - 1], dtype=torch.int32, device=device)
        chunk_lens = torch.tensor([int(token_ids.numel())], dtype=torch.int32, device=device)
        k_cache, v_cache, compact_table, compact_slots, phys = compact_vllm_kv_for_contract(
            layer_kvs, block_table, slot_mapping, seq_lens, chunk_lens=chunk_lens
        )
        logits = torch.zeros((1, PADDED_VOCAB), dtype=torch.float32, device=device)
        print(f"PYPTO_QWEN3_STAGE {stage}", flush=True)
        if stage == STAGE_PREFILL:
            args = build_prefill_kernel_args(
                input_ids=token_ids.to(device),
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                block_table=compact_table,
                slot_mapping=compact_slots,
                k_cache=k_cache,
                v_cache=v_cache,
                weights=bundle,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                logits=logits,
            )
            invoke_pypto_kernel(kernels["prefill_fwd"], args, session=session)
        else:
            sampled_out = torch.zeros((1, SAMPLED_IDS_PAD), dtype=torch.int32, device=device)
            next_hidden = torch.zeros((1, HIDDEN), dtype=torch.bfloat16, device=device)
            args = build_decode_kernel_args(
                token_ids=token_ids.to(device),
                seq_lens=seq_lens,
                block_table=compact_table,
                slot_mapping=compact_slots,
                k_cache=k_cache,
                v_cache=v_cache,
                weights=bundle,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                logits=logits,
                sampled_ids_out=sampled_out,
                next_hidden=next_hidden,
            )
            invoke_pypto_kernel(kernels["decode_fwd"], args, session=session)
        scatter_contract_kv_to_vllm(k_cache, v_cache, layer_kvs, phys)
        real = slice_real_vocab_logits(logits)
        reduced = comm.allreduce_sum(real[0, :HIDDEN])
        real = real.clone()
        real[0, :HIDDEN] = reduced.to(dtype=real.dtype) / world
        print(COMM_MARKER, f"stage={stage} rank={rank}", flush=True)
        return real

    logits = _step(STAGE_PREFILL, input_ids.to(device), ntok)
    next_id = int(logits[0].argmax().item())
    ids = [next_id]
    max_tokens = int(os.environ.get("PYPTO_MAX_TOKENS", "32"))
    seq = ntok + 1
    for _ in range(max_tokens - 1):
        step_ids = torch.tensor([ids[-1]], dtype=torch.int32, device=device)
        logits = _step(STAGE_DECODE, step_ids, seq)
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
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
