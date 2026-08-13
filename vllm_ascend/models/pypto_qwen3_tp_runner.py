#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Megatron TP=2 generate: pypto GEMMs on shards + pypto allreduce at o/down."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from vllm_ascend.models.pypto_qwen3_adapter import (
    HEAD_DIM,
    HIDDEN,
    NUM_HEADS,
    NUM_KV_HEADS,
    PADDED_VOCAB,
    REAL_VOCAB,
    invoke_pypto_kernel,
    slice_real_vocab_logits,
)
from vllm_ascend.models.pypto_qwen3_tp import (
    COMM_MARKER,
    STAGE_TP_DECODE,
    STAGE_TP_PREFILL,
    TP_BOUNDARY_DOWN_PROJ,
    TP_BOUNDARY_O_PROJ,
    TP_TOK_PAD,
    TP_WORLD,
    _bundle_num_layers,
    describe_tp_compute_args,
    layer_stacked_view,
)

EPS = 1e-6
ATTN_SCALE = HEAD_DIM**-0.5
HEADS_TP = NUM_HEADS // TP_WORLD
KV_HEADS_TP = NUM_KV_HEADS // TP_WORLD
HIDDEN_TP = HIDDEN // TP_WORLD
KV_HIDDEN_TP = KV_HEADS_TP * HEAD_DIM
MAX_CACHE = 128


def rmsnorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Qwen3 RMSNorm: no mean-center, weight is ``[H]`` or ``[1, H]``."""
    xf = hidden.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    scale = torch.rsqrt(var + eps)
    return (xf * scale * weight.float().reshape(1, -1)).to(dtype=hidden.dtype)


def qk_rmsnorm(heads: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Per-head RMSNorm. ``heads`` is ``[T, n_heads, D]``."""
    xf = heads.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(var + eps) * weight.float().reshape(1, 1, -1)


def apply_neox_rope(heads: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE matching ``build_rope_tables`` (cos/sin are ``[T, D]``)."""
    half = heads.shape[-1] // 2
    x1 = heads[..., :half]
    x2 = heads[..., half:]
    cos_lo = cos[..., :half].unsqueeze(1)
    cos_hi = cos[..., half:].unsqueeze(1)
    sin_lo = sin[..., :half].unsqueeze(1)
    sin_hi = sin[..., half:].unsqueeze(1)
    rot_lo = x1 * cos_lo - x2 * sin_lo
    rot_hi = x2 * cos_hi + x1 * sin_hi
    return torch.cat((rot_lo, rot_hi), dim=-1)


def local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """GQA attention on this rank's heads. Returns ``[Tq, Hq, D]``."""
    n_q = int(query.shape[1])
    n_kv = int(key.shape[1])
    if n_q % n_kv != 0:
        raise ValueError(f"Q heads {n_q} not divisible by KV heads {n_kv}")
    repeat = n_q // n_kv
    key_exp = key.repeat_interleave(repeat, dim=1)
    value_exp = value.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query.float(), key_exp.float()) * ATTN_SCALE
    if causal:
        t_q = int(query.shape[0])
        t_k = int(key.shape[0])
        mask = torch.ones((t_q, t_k), device=query.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    ctx = torch.einsum("hqk,khd->qhd", probs, value_exp.float())
    return ctx.to(dtype=query.dtype)


class PyptoTpRunner:
    """One rank of TP=2: sharded pypto GEMMs + pypto allreduce after o/down."""

    def __init__(
        self,
        shard: Any,
        comm: Any,
        session: Any,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        *,
        max_cache: int = MAX_CACHE,
    ) -> None:
        self.shard = shard
        self.comm = comm
        self.session = session
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin
        self.num_layers = _bundle_num_layers(shard)
        self.max_cache = int(max_cache)
        device = shard.wq.device
        self.k_cache = torch.zeros(
            (self.num_layers, self.max_cache, KV_HEADS_TP, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self._kernels: dict[str, Any] | None = None
        self._logged_args = False

    def compile(self) -> None:
        from vllm_ascend.models.pypto_qwen3_tp_kernels import TP_GEMM_KERNELS, TOK

        if self._kernels is not None:
            return
        if TOK != TP_TOK_PAD:
            raise ValueError(f"kernel TOK {TOK} != TP_TOK_PAD {TP_TOK_PAD}")
        compiled: dict[str, Any] = {}
        device = self.shard.wq.device
        for name, (kernel, k_dim, n_dim) in TP_GEMM_KERNELS.items():
            sample = (
                torch.zeros((TOK, k_dim), dtype=torch.bfloat16, device=device),
                torch.zeros((k_dim, n_dim), dtype=torch.bfloat16, device=device),
                torch.zeros((TOK, n_dim), dtype=torch.float32, device=device),
            )
            print(f"PYPTO_QWEN3_TP_COMPILE {name} k={k_dim} n={n_dim}", flush=True)
            self.session.compile(kernel, sample)
            compiled[name] = kernel
        self._kernels = compiled
        if not self._logged_args:
            print(f"PYPTO_QWEN3_ARGS {describe_tp_compute_args(self.shard)}", flush=True)
            self._logged_args = True

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        print(f"PYPTO_QWEN3_STAGE {STAGE_TP_PREFILL}", flush=True)
        hidden = F.embedding(input_ids.reshape(-1).to(device=self.shard.padded_embed_weight.device), self.shard.padded_embed_weight)
        hidden = self._forward_layers(hidden, pos0=0, causal=True)
        return self._logits(hidden[-1:])

    def decode(self, token_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        print(f"PYPTO_QWEN3_STAGE {STAGE_TP_DECODE}", flush=True)
        hidden = F.embedding(
            token_ids.reshape(-1).to(device=self.shard.padded_embed_weight.device),
            self.shard.padded_embed_weight,
        )
        hidden = self._forward_layers(hidden, pos0=seq_len - hidden.shape[0], causal=False)
        return self._logits(hidden[-1:])

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = rmsnorm(hidden, self.shard.final_norm_weight)
        logits = F.linear(normed.float(), self.shard.padded_lm_head_weight.float())
        if logits.shape[-1] < PADDED_VOCAB:
            raise ValueError(f"lm_head width {logits.shape[-1]} < padded vocab {PADDED_VOCAB}")
        return slice_real_vocab_logits(logits)

    def _forward_layers(self, hidden: torch.Tensor, *, pos0: int, causal: bool) -> torch.Tensor:
        n_tok = int(hidden.shape[0])
        if pos0 < 0:
            raise ValueError(f"pos0 must be >= 0, got {pos0}")
        if pos0 + n_tok > self.max_cache:
            raise ValueError(f"cache overflow pos0={pos0} n_tok={n_tok} max={self.max_cache}")
        self.compile()
        positions = torch.arange(pos0, pos0 + n_tok, device=hidden.device)
        cos = self.rope_cos.index_select(0, positions)
        sin = self.rope_sin.index_select(0, positions)
        for layer in range(self.num_layers):
            residual = hidden
            xn = rmsnorm(hidden, layer_stacked_view(self.shard.input_rms_weight, layer, self.num_layers))
            q = self._gemm("q", xn, layer_stacked_view(self.shard.wq, layer, self.num_layers))
            k = self._gemm("kv", xn, layer_stacked_view(self.shard.wk, layer, self.num_layers))
            v = self._gemm("kv", xn, layer_stacked_view(self.shard.wv, layer, self.num_layers))
            q = q.view(n_tok, HEADS_TP, HEAD_DIM)
            k = k.view(n_tok, KV_HEADS_TP, HEAD_DIM)
            v = v.view(n_tok, KV_HEADS_TP, HEAD_DIM)
            q = apply_neox_rope(
                qk_rmsnorm(q, layer_stacked_view(self.shard.q_norm_weight, layer, self.num_layers)),
                cos,
                sin,
            ).to(dtype=hidden.dtype)
            k = apply_neox_rope(
                qk_rmsnorm(k, layer_stacked_view(self.shard.k_norm_weight, layer, self.num_layers)),
                cos,
                sin,
            ).to(dtype=hidden.dtype)
            self.k_cache[layer, pos0 : pos0 + n_tok].copy_(k)
            self.v_cache[layer, pos0 : pos0 + n_tok].copy_(v.to(dtype=hidden.dtype))
            ctx_end = pos0 + n_tok
            attn = local_attention(
                q,
                self.k_cache[layer, :ctx_end],
                self.v_cache[layer, :ctx_end],
                causal=causal,
            )
            o_partial = self._gemm(
                "o",
                attn.reshape(n_tok, HIDDEN_TP),
                layer_stacked_view(self.shard.wo, layer, self.num_layers),
            )
            hidden = residual + self.comm.allreduce_sum(
                o_partial, boundary=f"{TP_BOUNDARY_O_PROJ}:{layer}"
            ).to(dtype=residual.dtype)
            residual = hidden
            xn = rmsnorm(hidden, layer_stacked_view(self.shard.post_rms_weight, layer, self.num_layers))
            gate = self._gemm("gate_up", xn, layer_stacked_view(self.shard.w_gate, layer, self.num_layers))
            up = self._gemm("gate_up", xn, layer_stacked_view(self.shard.w_up, layer, self.num_layers))
            down_partial = self._gemm(
                "down",
                F.silu(gate) * up,
                layer_stacked_view(self.shard.w_down, layer, self.num_layers),
            )
            hidden = residual + self.comm.allreduce_sum(
                down_partial, boundary=f"{TP_BOUNDARY_DOWN_PROJ}:{layer}"
            ).to(dtype=residual.dtype)
        print(COMM_MARKER, f"layers={self.num_layers} tokens={n_tok} pos0={pos0}", flush=True)
        return hidden

    def _gemm(self, name: str, left: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        from vllm_ascend.models.pypto_qwen3_tp_kernels import TOK

        if self._kernels is None:
            raise RuntimeError("PyptoTpRunner.compile has not run")
        kernel = self._kernels[name]
        n_tok = int(left.shape[0])
        if n_tok > TOK:
            raise ValueError(f"{name} token count {n_tok} exceeds TOK={TOK}")
        if left.dtype != torch.bfloat16:
            left = left.to(dtype=torch.bfloat16)
        if not left.is_contiguous():
            left = left.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        left_pad = left.new_zeros((TOK, left.shape[1]))
        left_pad[:n_tok].copy_(left)
        out_pad = torch.zeros((TOK, weight.shape[1]), dtype=torch.float32, device=left.device)
        invoke_pypto_kernel(kernel, (left_pad, weight, out_pad), session=self.session)
        return out_pad[:n_tok]


def check_gemm_matches_torch(session: Any, kernel: Any, left: torch.Tensor, weight: torch.Tensor) -> float:
    """Return max-abs error of one pypto GEMM vs float32 torch.matmul."""
    from vllm_ascend.models.pypto_qwen3_tp_kernels import TOK

    n_tok = int(left.shape[0])
    left_pad = left.new_zeros((TOK, left.shape[1]))
    left_pad[:n_tok] = left
    out_pad = torch.zeros((TOK, weight.shape[1]), dtype=torch.float32, device=left.device)
    invoke_pypto_kernel(kernel, (left_pad, weight, out_pad), session=session)
    ref = torch.matmul(left.float(), weight.float())
    return float((out_pad[:n_tok] - ref).abs().max().item())
