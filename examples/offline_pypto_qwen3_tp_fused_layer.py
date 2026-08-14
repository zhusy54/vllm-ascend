#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Two-rank P1: one fused layer vs PyptoTpRunner hidden.

    python -m torch.distributed.run --standalone --nproc_per_node=2 \\
        examples/offline_pypto_qwen3_tp_fused_layer.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

INDUCTOR_ROOT = "/mnt/workspace/inductor"


def _sanitize_sys_path() -> None:
    sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry) != INDUCTOR_ROOT]


def _one_layer_shard(rank: int, world: int, device: str):
    import torch

    from vllm_ascend.models.pypto_qwen3_adapter import PyptoQwen3WeightBundle
    from vllm_ascend.models.pypto_qwen3_tp import select_compute_bundle

    hidden = 5120
    kv = 1024
    inter = 17408
    head = 128
    vocab = 256
    g = torch.Generator(device="cpu")
    g.manual_seed(0)

    def randn(*shape: int, dtype=torch.bfloat16) -> torch.Tensor:
        # Keep the synthetic layer in a model-like numeric range.  Unit-variance
        # 5K-wide projections amplify harmless BF16 rounding into six digits.
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.02).to(dtype=dtype)

    full = PyptoQwen3WeightBundle(
        input_rms_weight=torch.ones((1, hidden), dtype=torch.float32),
        wq=randn(hidden, hidden),
        wk=randn(hidden, kv),
        wv=randn(hidden, kv),
        q_norm_weight=torch.ones((1, head), dtype=torch.float32),
        k_norm_weight=torch.ones((1, head), dtype=torch.float32),
        wo=randn(hidden, hidden),
        w_gate=randn(hidden, inter),
        w_up=randn(hidden, inter),
        w_down=randn(inter, hidden),
        post_rms_weight=torch.ones((1, hidden), dtype=torch.float32),
        final_norm_weight=torch.ones((1, hidden), dtype=torch.float32),
        padded_lm_head_weight=randn(vocab, hidden),
        padded_embed_weight=randn(vocab, hidden),
    )
    shard = select_compute_bundle(full, rank, world)
    return type(shard)(**{name: tensor.to(device) for name, tensor in shard.__dict__.items()})


def _error_stats(actual, expected) -> str:
    """Compact scale-aware diagnostics for BF16 pipeline comparisons."""
    import torch

    delta = actual.float() - expected.float()
    rmse = torch.sqrt(torch.mean(delta.square()))
    ref_rms = torch.sqrt(torch.mean(expected.float().square()))
    rel_rmse = rmse / torch.clamp(ref_rms, min=1.0e-12)
    return (
        f"max={float(delta.abs().max().item()):.8g} "
        f"rmse={float(rmse.item()):.8g} "
        f"rel_rmse={float(rel_rmse.item()):.8g} "
        f"ref_absmax={float(expected.float().abs().max().item()):.8g}"
    )


def _within_error(actual, expected, *, max_abs: float, max_rel_rmse: float) -> bool:
    """Fail closed on non-finite values and bound both peak and global error."""
    import torch

    actual_f = actual.float()
    expected_f = expected.float()
    if not bool(torch.isfinite(actual_f).all().item()):
        return False
    if not bool(torch.isfinite(expected_f).all().item()):
        return False
    delta = actual_f - expected_f
    rmse = torch.sqrt(torch.mean(delta.square()))
    ref_rms = torch.sqrt(torch.mean(expected_f.square()))
    rel_rmse = rmse / torch.clamp(ref_rms, min=1.0e-12)
    return bool(delta.abs().max().item() <= max_abs and rel_rmse.item() <= max_rel_rmse)


def main() -> int:
    _sanitize_sys_path()
    os.environ.setdefault("PYPTO_LIB_ROOT", str(Path(INDUCTOR_ROOT) / "pypto-lib"))
    os.environ.setdefault("PTO_PLATFORM", "a2a3")
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"

    import torch
    import torch.distributed as dist

    from vllm_ascend.models.pypto_qwen3_adapter import PyptoChipSession, build_rope_tables, wrap_torch_npu_ptr
    from vllm_ascend.models.pypto_qwen3_tp import _dispatch_chip, build_gloo_shmem_comm
    from vllm_ascend.models.pypto_qwen3_tp_fused import make_decode_mask, make_prefill_mask
    from vllm_ascend.models.pypto_qwen3_tp_fused_layer import compile_fused_layer_chip
    from vllm_ascend.models.pypto_qwen3_tp_fused_ops import (
        CACHE_ROWS,
        HIDDEN,
        KV_HIDDEN_TP,
        MAX_CACHE,
        TOK,
        TP_WORLD,
    )
    from vllm_ascend.models.pypto_qwen3_tp_runner import PyptoTpRunner

    dist.init_process_group(backend="gloo")
    rank = int(dist.get_rank())
    world = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world != 2 or dist.get_backend() == "hccl":
        print("need gloo world_size=2", file=sys.stderr)
        return 2
    torch.npu.set_device(local_rank)
    device = f"npu:{local_rank}"
    shard = _one_layer_shard(rank, world, device)
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

    torch.manual_seed(1)
    n_tok = 4
    hidden = torch.randn((n_tok, HIDDEN), dtype=torch.bfloat16, device=device)
    ref = runner._forward_layers(hidden.clone(), pos0=0, causal=True)

    compiled = compile_fused_layer_chip(decode=False, device_id=local_rank)
    hidden_pad = hidden.new_zeros((TOK, HIDDEN))
    hidden_pad[:n_tok].copy_(hidden)
    hidden_out = torch.zeros_like(hidden_pad)
    k_cache = torch.zeros((CACHE_ROWS, KV_HIDDEN_TP), dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros_like(k_cache)
    mask = make_prefill_mask(n_tok, device)
    prefill_credit_base = comm.reserve_signal_credits(4)
    meta = torch.tensor([n_tok, 0, prefill_credit_base], dtype=torch.int32, device=device)
    data_dt = comm.window.as_device_tensor("data_buf", (TOK, HIDDEN), torch.float32)
    signal_dt = comm.window.as_device_tensor("signal", (TP_WORLD, 1), torch.int32)
    ctx = int(comm.window.device_ctx_ptr)

    def arg(tensor: torch.Tensor):
        owned = tensor.detach().contiguous()
        return wrap_torch_npu_ptr(owned) if owned.device.type != "cpu" else owned

    owners = [
        hidden_pad,
        shard.input_rms_weight,
        shard.wq,
        shard.wk,
        shard.wv,
        shard.q_norm_weight,
        shard.k_norm_weight,
        shard.wo,
        shard.post_rms_weight,
        shard.w_gate,
        shard.w_up,
        shard.w_down,
        rope_cos,
        rope_sin,
        mask,
        k_cache,
        v_cache,
        hidden_out,
        meta,
    ]
    session._live_args = owners
    session.launch_tag = "tp_fused_layer_prefill"
    _dispatch_chip(
        session,
        compiled,
        *(arg(t) for t in owners[:15]),
        arg(k_cache),
        arg(v_cache),
        data_dt,
        signal_dt,
        arg(hidden_out),
        arg(meta),
        ctx,
        ctx,
    )
    fused = hidden_out[:n_tok]
    diff = float((fused.float() - ref.float()).abs().max().item())
    ref_k = runner.k_cache[0].reshape(MAX_CACHE, KV_HIDDEN_TP)
    ref_v = runner.v_cache[0].reshape(MAX_CACHE, KV_HIDDEN_TP)
    prefill_k_diff = float((k_cache[:n_tok].float() - ref_k[:n_tok].float()).abs().max().item())
    prefill_v_diff = float((v_cache[:n_tok].float() - ref_v[:n_tok].float()).abs().max().item())
    print(
        f"PYPTO_QWEN3_FUSED_LAYER rank={rank} prefill_maxdiff={diff} "
        f"prefill_k_maxdiff={prefill_k_diff} prefill_v_maxdiff={prefill_v_diff}",
        flush=True,
    )
    print(
        f"PYPTO_QWEN3_FUSED_LAYER_STATS rank={rank} prefill[{_error_stats(fused, ref)}] "
        f"k[{_error_stats(k_cache[:n_tok], ref_k[:n_tok])}] "
        f"v[{_error_stats(v_cache[:n_tok], ref_v[:n_tok])}]",
        flush=True,
    )

    step = torch.randn((1, HIDDEN), dtype=torch.bfloat16, device=device)
    ref_dec = runner._forward_layers(step.clone(), pos0=n_tok, causal=False)
    compiled_d = compile_fused_layer_chip(decode=True, device_id=local_rank)
    dec_in = step.new_zeros((TOK, HIDDEN))
    dec_in[:1].copy_(step)
    dec_out = torch.zeros_like(dec_in)
    dec_mask = make_decode_mask(n_tok + 1, device)
    decode_credit_base = comm.reserve_signal_credits(4)
    meta_d = torch.tensor([1, n_tok, decode_credit_base], dtype=torch.int32, device=device)
    session.launch_tag = "tp_fused_layer_decode"
    owners_d = [
        dec_in,
        shard.input_rms_weight,
        shard.wq,
        shard.wk,
        shard.wv,
        shard.q_norm_weight,
        shard.k_norm_weight,
        shard.wo,
        shard.post_rms_weight,
        shard.w_gate,
        shard.w_up,
        shard.w_down,
        rope_cos,
        rope_sin,
        dec_mask,
        k_cache,
        v_cache,
        dec_out,
        meta_d,
    ]
    session._live_args = owners_d
    _dispatch_chip(
        session,
        compiled_d,
        *(arg(t) for t in owners_d[:15]),
        arg(k_cache),
        arg(v_cache),
        data_dt,
        signal_dt,
        arg(dec_out),
        arg(meta_d),
        ctx,
        ctx,
    )
    diff_d = float((dec_out[:1].float() - ref_dec.float()).abs().max().item())
    decode_k_diff = float((k_cache[: n_tok + 1].float() - ref_k[: n_tok + 1].float()).abs().max().item())
    decode_v_diff = float((v_cache[: n_tok + 1].float() - ref_v[: n_tok + 1].float()).abs().max().item())
    print(
        f"PYPTO_QWEN3_FUSED_LAYER rank={rank} decode_maxdiff={diff_d} "
        f"decode_k_maxdiff={decode_k_diff} decode_v_maxdiff={decode_v_diff} "
        f"credits={comm.signal_credit}",
        flush=True,
    )
    print(
        f"PYPTO_QWEN3_FUSED_LAYER_STATS rank={rank} decode[{_error_stats(dec_out[:1], ref_dec)}] "
        f"k[{_error_stats(k_cache[: n_tok + 1], ref_k[: n_tok + 1])}] "
        f"v[{_error_stats(v_cache[: n_tok + 1], ref_v[: n_tok + 1])}]",
        flush=True,
    )
    ok = (
        _within_error(fused, ref, max_abs=0.25, max_rel_rmse=0.015)
        and _within_error(dec_out[:1], ref_dec, max_abs=0.25, max_rel_rmse=0.015)
        and _within_error(
            k_cache[: n_tok + 1],
            ref_k[: n_tok + 1],
            max_abs=0.03125,
            max_rel_rmse=0.005,
        )
        and _within_error(
            v_cache[: n_tok + 1],
            ref_v[: n_tok + 1],
            max_abs=0.0625,
            max_rel_rmse=0.005,
        )
        and comm.signal_credit == 8
    )
    global_ok = torch.tensor(int(ok), dtype=torch.int32)
    dist.all_reduce(global_ok, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            "PYPTO_QWEN3_FUSED_LAYER_OK" if int(global_ok.item()) == 1 else "PYPTO_QWEN3_FUSED_LAYER_FAIL",
            flush=True,
        )
    dist.destroy_process_group()
    return 0 if int(global_ok.item()) == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
