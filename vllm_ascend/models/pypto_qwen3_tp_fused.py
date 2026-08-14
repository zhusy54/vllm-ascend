#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""40-layer Megatron TP=2 prefill/decode as one ChipWorker.run each."""

from __future__ import annotations

import os
from typing import Any

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from vllm_ascend.models.pypto_qwen3_adapter import (
    PADDED_VOCAB,
    REAL_VOCAB,
    slice_real_vocab_logits,
    wrap_torch_npu_ptr,
)
from vllm_ascend.models.pypto_qwen3_tp import (
    COMM_MARKER,
    STAGE_TP_DECODE,
    STAGE_TP_PREFILL,
    _bundle_num_layers,
    _dispatch_chip,
    describe_tp_compute_args,
    layer_stacked_view,
    tp_compile_output_dir,
)
from vllm_ascend.models.pypto_qwen3_tp_fused_layer import (
    ATTN_ROWS,
    tp_allreduce_step,
    tp_attention_cast_step,
    tp_decode_attention_context_step,
    tp_decode_attention_prepare_step,
    tp_decode_attention_scores_step,
    tp_decode_attention_softmax_step,
    tp_ffn_down_step,
    tp_ffn_gate_up_step,
    tp_ffn_residual_rms_step,
    tp_ffn_swiglu_step,
    tp_hidden_rms_step,
    tp_o_gemm_step,
    tp_prefill_attention_context_step,
    tp_prefill_attention_prepare_step,
    tp_prefill_attention_scores_step,
    tp_prefill_attention_softmax_step,
    tp_qkv_gemm_step,
    tp_qkv_post_step,
    tp_residual_tail,
)
from vllm_ascend.models.pypto_qwen3_tp_fused_ops import (
    CACHE_ROWS,
    HEAD_DIM,
    HIDDEN,
    KV_HIDDEN_TP,
    MAX_CACHE,
    MAX_SEQ,
    NUM_LAYERS,
    TOK,
    TP_WORLD,
    VOCAB,
    rms_hidden_step,
    tp_embed_step,
    tp_lm_head_step,
)
from vllm_ascend.models.pypto_qwen3_tp_kernels import HIDDEN_TP, INTER_TP

CREDITS_PER_LAYER = 4
NEG_INF = -1.0e9
RING_DEP_POOL = int(os.environ.get("PTO2_RING_DEP_POOL", "262144"))


@pl.jit
def tp_prefill_fwd(
    input_ids: pl.Tensor[[TOK], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    input_rms: pl.Tensor[[NUM_LAYERS, HIDDEN], pl.FP32],
    wq: pl.Tensor[[NUM_LAYERS * HIDDEN, HIDDEN_TP], pl.BF16],
    wk: pl.Tensor[[NUM_LAYERS * HIDDEN, KV_HIDDEN_TP], pl.BF16],
    wv: pl.Tensor[[NUM_LAYERS * HIDDEN, KV_HIDDEN_TP], pl.BF16],
    q_norm: pl.Tensor[[NUM_LAYERS, HEAD_DIM], pl.FP32],
    k_norm: pl.Tensor[[NUM_LAYERS, HEAD_DIM], pl.FP32],
    wo: pl.Tensor[[NUM_LAYERS * HIDDEN_TP, HIDDEN], pl.BF16],
    post_rms: pl.Tensor[[NUM_LAYERS, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[NUM_LAYERS * HIDDEN, INTER_TP], pl.BF16],
    w_up: pl.Tensor[[NUM_LAYERS * HIDDEN, INTER_TP], pl.BF16],
    w_down: pl.Tensor[[NUM_LAYERS * INTER_TP, HIDDEN], pl.BF16],
    final_norm: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    attn_mask: pl.Tensor[[TOK, TOK], pl.FP32],
    k_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    v_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[TOK, HIDDEN], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[TP_WORLD, 1], pl.INT32]],
    logits: pl.InOut[pl.Tensor[[TOK, VOCAB], pl.FP32]],
    meta: pl.Tensor[[3], pl.INT32],
):
    n_tok = pl.cast(pl.tensor.read(meta, [0]), pl.INDEX)
    pos0 = pl.cast(pl.tensor.read(meta, [1]), pl.INDEX)
    credit_base = pl.cast(pl.tensor.read(meta, [2]), pl.INDEX)
    hidden = pl.create_tensor([TOK, HIDDEN], dtype=pl.BF16, init_value=0)
    hidden = tp_embed_step(input_ids, embed_weight, hidden, n_tok)
    for layer_idx in pl.range(NUM_LAYERS):
        xn_attn = pl.create_tensor([TOK, HIDDEN], dtype=pl.BF16)
        xn_ffn = pl.create_tensor([TOK, HIDDEN], dtype=pl.BF16)
        q_proj_fp = pl.create_tensor([TOK, HIDDEN_TP], dtype=pl.FP32)
        k_proj_fp = pl.create_tensor([TOK, KV_HIDDEN_TP], dtype=pl.FP32)
        v_proj_fp = pl.create_tensor([TOK, KV_HIDDEN_TP], dtype=pl.FP32)
        q_attn_fp = pl.create_tensor([TOK, HIDDEN_TP], dtype=pl.FP32)
        k_attn_fp = pl.create_tensor([TOK, KV_HIDDEN_TP], dtype=pl.FP32)
        v_attn_fp = pl.create_tensor([TOK, KV_HIDDEN_TP], dtype=pl.FP32)
        context_fp = pl.create_tensor([TOK, HIDDEN_TP], dtype=pl.FP32)
        q_rope_bf = pl.create_tensor([TOK, HIDDEN_TP], dtype=pl.BF16)
        k_rope_bf = pl.create_tensor([TOK, KV_HIDDEN_TP], dtype=pl.BF16)
        v_proj_bf = pl.create_tensor([TOK, KV_HIDDEN_TP], dtype=pl.BF16)
        context_bf = pl.create_tensor([TOK, HIDDEN_TP], dtype=pl.BF16)
        attn_scores = pl.create_tensor([ATTN_ROWS, TOK], dtype=pl.FP32)
        attn_probs = pl.create_tensor([ATTN_ROWS, TOK], dtype=pl.FP32)
        o_partial = pl.create_tensor([TOK, HIDDEN], dtype=pl.FP32)
        o_reduced = pl.create_tensor([TOK, HIDDEN], dtype=pl.FP32)
        attn_residual = pl.create_tensor([TOK, HIDDEN], dtype=pl.BF16)
        gate = pl.create_tensor([TOK, INTER_TP], dtype=pl.FP32)
        up = pl.create_tensor([TOK, INTER_TP], dtype=pl.FP32)
        act = pl.create_tensor([TOK, INTER_TP], dtype=pl.BF16)
        down_partial = pl.create_tensor([TOK, HIDDEN], dtype=pl.FP32)
        down_reduced = pl.create_tensor([TOK, HIDDEN], dtype=pl.FP32)
        hidden_next = pl.create_tensor([TOK, HIDDEN], dtype=pl.BF16)

        input_rms_layer = pl.slice(input_rms, [1, HIDDEN], [layer_idx, 0])
        wq_layer = pl.slice(wq, [HIDDEN, HIDDEN_TP], [layer_idx * HIDDEN, 0])
        wk_layer = pl.slice(wk, [HIDDEN, KV_HIDDEN_TP], [layer_idx * HIDDEN, 0])
        wv_layer = pl.slice(wv, [HIDDEN, KV_HIDDEN_TP], [layer_idx * HIDDEN, 0])
        q_norm_layer = pl.slice(q_norm, [1, HEAD_DIM], [layer_idx, 0])
        k_norm_layer = pl.slice(k_norm, [1, HEAD_DIM], [layer_idx, 0])
        wo_layer = pl.slice(wo, [HIDDEN_TP, HIDDEN], [layer_idx * HIDDEN_TP, 0])
        post_rms_layer = pl.slice(post_rms, [1, HIDDEN], [layer_idx, 0])
        w_gate_layer = pl.slice(
            w_gate,
            [HIDDEN, INTER_TP],
            [layer_idx * HIDDEN, 0],
        )
        w_up_layer = pl.slice(w_up, [HIDDEN, INTER_TP], [layer_idx * HIDDEN, 0])
        w_down_layer = pl.slice(
            w_down,
            [INTER_TP, HIDDEN],
            [layer_idx * INTER_TP, 0],
        )

        xn_attn = tp_hidden_rms_step(hidden, input_rms_layer, xn_attn)
        q_proj_fp, k_proj_fp, v_proj_fp = tp_qkv_gemm_step(
            xn_attn,
            wq_layer,
            wk_layer,
            wv_layer,
            q_proj_fp,
            k_proj_fp,
            v_proj_fp,
        )
        k_cache, v_cache, q_rope_bf, k_rope_bf, v_proj_bf = tp_qkv_post_step(
            q_proj_fp,
            k_proj_fp,
            v_proj_fp,
            q_norm_layer,
            k_norm_layer,
            rope_cos,
            rope_sin,
            k_cache,
            v_cache,
            q_rope_bf,
            k_rope_bf,
            v_proj_bf,
            layer_idx,
            pos0,
            n_tok,
        )
        q_attn_fp, k_attn_fp, v_attn_fp = tp_prefill_attention_prepare_step(
            q_rope_bf,
            k_rope_bf,
            v_proj_bf,
            q_attn_fp,
            k_attn_fp,
            v_attn_fp,
        )
        attn_scores = tp_prefill_attention_scores_step(q_attn_fp, k_attn_fp, attn_scores)
        attn_probs = tp_prefill_attention_softmax_step(
            attn_scores,
            attn_mask,
            attn_probs,
        )
        context_fp = tp_prefill_attention_context_step(attn_probs, v_attn_fp, context_fp)
        context_bf = tp_attention_cast_step(context_fp, context_bf)
        o_partial = tp_o_gemm_step(context_bf, wo_layer, o_partial)
        credit = credit_base + layer_idx * CREDITS_PER_LAYER
        pub_o = credit + 1
        cons_o = credit + 2
        pub_d = credit + 3
        cons_d = credit + 4
        data, signal, o_reduced = tp_allreduce_step(
            o_partial,
            data,
            signal,
            o_reduced,
            pub_o,
            cons_o,
        )
        attn_residual, xn_ffn = tp_ffn_residual_rms_step(
            hidden,
            o_reduced,
            post_rms_layer,
            attn_residual,
            xn_ffn,
        )
        gate, up = tp_ffn_gate_up_step(
            xn_ffn,
            w_gate_layer,
            w_up_layer,
            gate,
            up,
        )
        act = tp_ffn_swiglu_step(gate, up, act)
        down_partial = tp_ffn_down_step(act, w_down_layer, down_partial)
        data, signal, down_reduced = tp_allreduce_step(
            down_partial,
            data,
            signal,
            down_reduced,
            pub_d,
            cons_d,
        )
        hidden = tp_residual_tail(attn_residual, down_reduced, hidden_next)
    normed = pl.create_tensor([TOK, HIDDEN], dtype=pl.BF16)
    normed = rms_hidden_step(hidden, final_norm, normed)
    logits = tp_lm_head_step(normed, lm_head, logits)
    return logits


@pl.jit
def tp_decode_fwd(
    input_ids: pl.Tensor[[TOK], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    input_rms: pl.Tensor[[NUM_LAYERS, HIDDEN], pl.FP32],
    wq: pl.Tensor[[NUM_LAYERS * HIDDEN, HIDDEN_TP], pl.BF16],
    wk: pl.Tensor[[NUM_LAYERS * HIDDEN, KV_HIDDEN_TP], pl.BF16],
    wv: pl.Tensor[[NUM_LAYERS * HIDDEN, KV_HIDDEN_TP], pl.BF16],
    q_norm: pl.Tensor[[NUM_LAYERS, HEAD_DIM], pl.FP32],
    k_norm: pl.Tensor[[NUM_LAYERS, HEAD_DIM], pl.FP32],
    wo: pl.Tensor[[NUM_LAYERS * HIDDEN_TP, HIDDEN], pl.BF16],
    post_rms: pl.Tensor[[NUM_LAYERS, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[NUM_LAYERS * HIDDEN, INTER_TP], pl.BF16],
    w_up: pl.Tensor[[NUM_LAYERS * HIDDEN, INTER_TP], pl.BF16],
    w_down: pl.Tensor[[NUM_LAYERS * INTER_TP, HIDDEN], pl.BF16],
    final_norm: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    attn_mask: pl.Tensor[[TOK, MAX_CACHE], pl.FP32],
    k_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    v_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[TOK, HIDDEN], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[TP_WORLD, 1], pl.INT32]],
    logits: pl.InOut[pl.Tensor[[TOK, VOCAB], pl.FP32]],
    meta: pl.Tensor[[3], pl.INT32],
):
    n_tok = pl.cast(pl.tensor.read(meta, [0]), pl.INDEX)
    pos0 = pl.cast(pl.tensor.read(meta, [1]), pl.INDEX)
    credit_base = pl.cast(pl.tensor.read(meta, [2]), pl.INDEX)
    hidden = pl.create_tensor([TOK, HIDDEN], dtype=pl.BF16, init_value=0)
    hidden = tp_embed_step(input_ids, embed_weight, hidden, n_tok)
    for layer_idx in pl.range(NUM_LAYERS):
        xn_attn = pl.create_tensor([TOK, HIDDEN], dtype=pl.BF16)
        xn_ffn = pl.create_tensor([TOK, HIDDEN], dtype=pl.BF16)
        q_proj_fp = pl.create_tensor([TOK, HIDDEN_TP], dtype=pl.FP32)
        k_proj_fp = pl.create_tensor([TOK, KV_HIDDEN_TP], dtype=pl.FP32)
        v_proj_fp = pl.create_tensor([TOK, KV_HIDDEN_TP], dtype=pl.FP32)
        q_attn_fp = pl.create_tensor([TOK, HIDDEN_TP], dtype=pl.FP32)
        context_fp = pl.create_tensor([TOK, HIDDEN_TP], dtype=pl.FP32)
        q_rope_bf = pl.create_tensor([TOK, HIDDEN_TP], dtype=pl.BF16)
        k_rope_bf = pl.create_tensor([TOK, KV_HIDDEN_TP], dtype=pl.BF16)
        v_proj_bf = pl.create_tensor([TOK, KV_HIDDEN_TP], dtype=pl.BF16)
        context_bf = pl.create_tensor([TOK, HIDDEN_TP], dtype=pl.BF16)
        k_ctx_fp = pl.create_tensor([MAX_CACHE, KV_HIDDEN_TP], dtype=pl.FP32)
        v_ctx_fp = pl.create_tensor([MAX_CACHE, KV_HIDDEN_TP], dtype=pl.FP32)
        attn_scores = pl.create_tensor([ATTN_ROWS, MAX_CACHE], dtype=pl.FP32)
        attn_probs = pl.create_tensor([ATTN_ROWS, MAX_CACHE], dtype=pl.FP32)
        o_partial = pl.create_tensor([TOK, HIDDEN], dtype=pl.FP32)
        o_reduced = pl.create_tensor([TOK, HIDDEN], dtype=pl.FP32)
        attn_residual = pl.create_tensor([TOK, HIDDEN], dtype=pl.BF16)
        gate = pl.create_tensor([TOK, INTER_TP], dtype=pl.FP32)
        up = pl.create_tensor([TOK, INTER_TP], dtype=pl.FP32)
        act = pl.create_tensor([TOK, INTER_TP], dtype=pl.BF16)
        down_partial = pl.create_tensor([TOK, HIDDEN], dtype=pl.FP32)
        down_reduced = pl.create_tensor([TOK, HIDDEN], dtype=pl.FP32)
        hidden_next = pl.create_tensor([TOK, HIDDEN], dtype=pl.BF16)

        input_rms_layer = pl.slice(input_rms, [1, HIDDEN], [layer_idx, 0])
        wq_layer = pl.slice(wq, [HIDDEN, HIDDEN_TP], [layer_idx * HIDDEN, 0])
        wk_layer = pl.slice(wk, [HIDDEN, KV_HIDDEN_TP], [layer_idx * HIDDEN, 0])
        wv_layer = pl.slice(wv, [HIDDEN, KV_HIDDEN_TP], [layer_idx * HIDDEN, 0])
        q_norm_layer = pl.slice(q_norm, [1, HEAD_DIM], [layer_idx, 0])
        k_norm_layer = pl.slice(k_norm, [1, HEAD_DIM], [layer_idx, 0])
        wo_layer = pl.slice(wo, [HIDDEN_TP, HIDDEN], [layer_idx * HIDDEN_TP, 0])
        post_rms_layer = pl.slice(post_rms, [1, HIDDEN], [layer_idx, 0])
        w_gate_layer = pl.slice(
            w_gate,
            [HIDDEN, INTER_TP],
            [layer_idx * HIDDEN, 0],
        )
        w_up_layer = pl.slice(w_up, [HIDDEN, INTER_TP], [layer_idx * HIDDEN, 0])
        w_down_layer = pl.slice(
            w_down,
            [INTER_TP, HIDDEN],
            [layer_idx * INTER_TP, 0],
        )

        xn_attn = tp_hidden_rms_step(hidden, input_rms_layer, xn_attn)
        q_proj_fp, k_proj_fp, v_proj_fp = tp_qkv_gemm_step(
            xn_attn,
            wq_layer,
            wk_layer,
            wv_layer,
            q_proj_fp,
            k_proj_fp,
            v_proj_fp,
        )
        k_cache, v_cache, q_rope_bf, k_rope_bf, v_proj_bf = tp_qkv_post_step(
            q_proj_fp,
            k_proj_fp,
            v_proj_fp,
            q_norm_layer,
            k_norm_layer,
            rope_cos,
            rope_sin,
            k_cache,
            v_cache,
            q_rope_bf,
            k_rope_bf,
            v_proj_bf,
            layer_idx,
            pos0,
            n_tok,
        )
        k_layer = pl.slice(
            k_cache,
            [MAX_CACHE, KV_HIDDEN_TP],
            [layer_idx * MAX_CACHE, 0],
        )
        v_layer = pl.slice(
            v_cache,
            [MAX_CACHE, KV_HIDDEN_TP],
            [layer_idx * MAX_CACHE, 0],
        )
        q_attn_fp, k_ctx_fp, v_ctx_fp = tp_decode_attention_prepare_step(
            q_rope_bf,
            k_layer,
            v_layer,
            q_attn_fp,
            k_ctx_fp,
            v_ctx_fp,
        )
        attn_scores = tp_decode_attention_scores_step(
            q_attn_fp,
            k_ctx_fp,
            attn_scores,
        )
        attn_probs = tp_decode_attention_softmax_step(
            attn_scores,
            attn_mask,
            attn_probs,
        )
        context_fp = tp_decode_attention_context_step(attn_probs, v_ctx_fp, context_fp)
        context_bf = tp_attention_cast_step(context_fp, context_bf)
        o_partial = tp_o_gemm_step(context_bf, wo_layer, o_partial)
        credit = credit_base + layer_idx * CREDITS_PER_LAYER
        pub_o = credit + 1
        cons_o = credit + 2
        pub_d = credit + 3
        cons_d = credit + 4
        data, signal, o_reduced = tp_allreduce_step(
            o_partial,
            data,
            signal,
            o_reduced,
            pub_o,
            cons_o,
        )
        attn_residual, xn_ffn = tp_ffn_residual_rms_step(
            hidden,
            o_reduced,
            post_rms_layer,
            attn_residual,
            xn_ffn,
        )
        gate, up = tp_ffn_gate_up_step(
            xn_ffn,
            w_gate_layer,
            w_up_layer,
            gate,
            up,
        )
        act = tp_ffn_swiglu_step(gate, up, act)
        down_partial = tp_ffn_down_step(act, w_down_layer, down_partial)
        data, signal, down_reduced = tp_allreduce_step(
            down_partial,
            data,
            signal,
            down_reduced,
            pub_d,
            cons_d,
        )
        hidden = tp_residual_tail(attn_residual, down_reduced, hidden_next)
    normed = pl.create_tensor([TOK, HIDDEN], dtype=pl.BF16)
    normed = rms_hidden_step(hidden, final_norm, normed)
    logits = tp_lm_head_step(normed, lm_head, logits)
    return logits


def make_prefill_mask(n_tok: int, device: torch.device | str) -> torch.Tensor:
    """Causal + pad mask for a TOK-padded prefill window."""
    mask = torch.zeros((TOK, TOK), dtype=torch.float32, device=device)
    causal = torch.triu(torch.ones((TOK, TOK), dtype=torch.bool, device=device), diagonal=1)
    mask = mask.masked_fill(causal, NEG_INF)
    if n_tok < TOK:
        mask[:, n_tok:] = NEG_INF
    return mask.contiguous()


def make_decode_mask(seq_len: int, device: torch.device | str) -> torch.Tensor:
    """Only query row 0 attends to cache[:seq_len]."""
    mask = torch.full((TOK, MAX_CACHE), NEG_INF, dtype=torch.float32, device=device)
    live = max(int(seq_len), 0)
    if live > MAX_CACHE:
        raise ValueError(f"seq_len {seq_len} exceeds MAX_CACHE {MAX_CACHE}")
    if live > 0:
        mask[0, :live] = 0
    return mask.contiguous()


def _pad_ids(token_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    flat = token_ids.reshape(-1).to(dtype=torch.int32)
    n_tok = int(flat.numel())
    if n_tok > TOK:
        raise ValueError(f"token count {n_tok} exceeds TOK={TOK}")
    padded = flat.new_zeros((TOK,))
    if n_tok:
        padded[:n_tok] = flat
    return padded.contiguous(), n_tok


def _compile_cfg(stage: str, device_id: int = 0):
    from pypto.runtime import RunConfig

    return RunConfig(
        platform=os.environ.get("PTO_PLATFORM", "a2a3"),
        device_id=int(device_id),
        ring_dep_pool=RING_DEP_POOL,
        save_kernels_dir=tp_compile_output_dir(stage),
    )


class PyptoTpFusedHost:
    """One rank of fused TP=2: embed through lm_head in one chip per stage."""

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
        if max_cache != MAX_CACHE:
            raise ValueError(f"fused host cache is compiled for {MAX_CACHE}, got {max_cache}")
        self.shard = shard
        self.comm = comm
        self.session = session
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin
        self.num_layers = _bundle_num_layers(shard)
        if self.num_layers != NUM_LAYERS:
            raise ValueError(f"fused host expects {NUM_LAYERS} layers, got {self.num_layers}")
        device = shard.wq.device
        self.k_cache = torch.zeros(
            (CACHE_ROWS, KV_HIDDEN_TP),
            dtype=torch.bfloat16,
            device=device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self._prefill = None
        self._decode = None
        self._logged_args = False
        session.config.ring_dep_pool = RING_DEP_POOL

    def compile(self, *, decode: bool | None = None) -> None:
        """Compile only the requested stage, or both stages when ``decode`` is omitted."""
        device_id = int(getattr(self.session.config, "device_id", 0))
        if decode is not True and self._prefill is None:
            print("PYPTO_QWEN3_TP_COMPILE tp_prefill_fwd", flush=True)
            cfg = _compile_cfg("tp_prefill_fwd", device_id)
            self._prefill = tp_prefill_fwd.compile(*self._sample_args(decode=False), config=cfg)
        if decode is not False and self._decode is None:
            print("PYPTO_QWEN3_TP_COMPILE tp_decode_fwd", flush=True)
            cfg = _compile_cfg("tp_decode_fwd", device_id)
            self._decode = tp_decode_fwd.compile(*self._sample_args(decode=True), config=cfg)
        if not self._logged_args:
            print(f"PYPTO_QWEN3_ARGS {describe_tp_compute_args(self.shard)}", flush=True)
            self._logged_args = True

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        print(f"PYPTO_QWEN3_STAGE {STAGE_TP_PREFILL}", flush=True)
        self.compile(decode=False)
        ids, n_tok = _pad_ids(input_ids.to(device=self.shard.padded_embed_weight.device))
        mask = make_prefill_mask(n_tok, ids.device)
        logits = self._run_stage(
            self._prefill,
            ids,
            mask,
            n_tok=n_tok,
            pos0=0,
            tag="tp_prefill_fwd",
        )
        print(COMM_MARKER, f"fused layers={self.num_layers} tokens={n_tok} pos0=0", flush=True)
        return slice_real_vocab_logits(logits[n_tok - 1 : n_tok])

    def decode(self, token_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        print(f"PYPTO_QWEN3_STAGE {STAGE_TP_DECODE}", flush=True)
        self.compile(decode=True)
        ids, n_tok = _pad_ids(token_ids.to(device=self.shard.padded_embed_weight.device))
        if n_tok != 1:
            raise ValueError(f"fused decode expects 1 token, got {n_tok}")
        pos0 = int(seq_len) - 1
        if pos0 < 0:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        mask = make_decode_mask(seq_len, ids.device)
        logits = self._run_stage(
            self._decode,
            ids,
            mask,
            n_tok=1,
            pos0=pos0,
            tag="tp_decode_fwd",
        )
        print(COMM_MARKER, f"fused layers={self.num_layers} tokens=1 pos0={pos0}", flush=True)
        return slice_real_vocab_logits(logits[0:1])

    def _sample_args(self, *, decode: bool) -> tuple[torch.Tensor, ...]:
        shard = self.shard
        meta_device = torch.device("meta")

        def host(tensor: torch.Tensor) -> torch.Tensor:
            return torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device=meta_device)

        mask = (
            torch.empty((TOK, MAX_CACHE), dtype=torch.float32, device=meta_device)
            if decode
            else torch.empty((TOK, TOK), dtype=torch.float32, device=meta_device)
        )
        return (
            torch.empty((TOK,), dtype=torch.int32, device=meta_device),
            host(shard.padded_embed_weight),
            host(shard.input_rms_weight),
            host(shard.wq),
            host(shard.wk),
            host(shard.wv),
            host(shard.q_norm_weight),
            host(shard.k_norm_weight),
            host(shard.wo),
            host(shard.post_rms_weight),
            host(shard.w_gate),
            host(shard.w_up),
            host(shard.w_down),
            host(shard.final_norm_weight),
            host(shard.padded_lm_head_weight),
            host(self.rope_cos),
            host(self.rope_sin),
            mask,
            torch.empty((CACHE_ROWS, KV_HIDDEN_TP), dtype=torch.bfloat16, device=meta_device),
            torch.empty((CACHE_ROWS, KV_HIDDEN_TP), dtype=torch.bfloat16, device=meta_device),
            torch.empty((TOK, HIDDEN), dtype=torch.float32, device=meta_device),
            torch.empty((TP_WORLD, 1), dtype=torch.int32, device=meta_device),
            torch.empty((TOK, VOCAB), dtype=torch.float32, device=meta_device),
            torch.empty((3,), dtype=torch.int32, device=meta_device),
        )

    def _run_stage(
        self,
        compiled: Any,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        *,
        n_tok: int,
        pos0: int,
        tag: str,
    ) -> torch.Tensor:
        shard = self.shard
        device = input_ids.device
        logits = torch.zeros((TOK, VOCAB), dtype=torch.float32, device=device)
        credit_base = self.comm.reserve_signal_credits(NUM_LAYERS * CREDITS_PER_LAYER)
        meta = torch.tensor([n_tok, pos0, credit_base], dtype=torch.int32, device=device)
        data_dt = self.comm.window.as_device_tensor("data_buf", (TOK, HIDDEN), torch.float32)
        signal_dt = self.comm.window.as_device_tensor("signal", (TP_WORLD, 1), torch.int32)
        ctx = int(self.comm.window.device_ctx_ptr)

        def arg(tensor: torch.Tensor):
            owned = tensor.detach()
            if not owned.is_contiguous():
                owned = owned.contiguous()
            return wrap_torch_npu_ptr(owned) if owned.device.type != "cpu" else owned

        owners = [
            input_ids,
            shard.padded_embed_weight,
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
            shard.final_norm_weight,
            shard.padded_lm_head_weight,
            self.rope_cos,
            self.rope_sin,
            attn_mask,
            self.k_cache,
            self.v_cache,
            logits,
            meta,
        ]
        self.session._live_args = owners
        prev = self.session.launch_tag
        self.session.launch_tag = tag
        try:
            _dispatch_chip(
                self.session,
                compiled,
                *(arg(tensor) for tensor in owners[:18]),
                arg(self.k_cache),
                arg(self.v_cache),
                data_dt,
                signal_dt,
                arg(logits),
                arg(meta),
                ctx,
                ctx,
            )
        finally:
            self.session.launch_tag = prev
        if logits.shape[-1] < PADDED_VOCAB or REAL_VOCAB > PADDED_VOCAB:
            raise ValueError(f"lm_head width {logits.shape[-1]} incompatible with vocab")
        return logits


def slice_layer_bundle(bundle: Any, layer: int, num_layers: int | None = None) -> Any:
    """Return a one-layer view of a stacked contract bundle."""
    from vllm_ascend.models.pypto_qwen3_adapter import PyptoQwen3WeightBundle

    layers = int(num_layers if num_layers is not None else _bundle_num_layers(bundle))
    fields: dict[str, torch.Tensor] = {}
    shared = ("final_norm_weight", "padded_lm_head_weight", "padded_embed_weight")
    for name, tensor in bundle.__dict__.items():
        if name in shared:
            fields[name] = tensor
        else:
            fields[name] = layer_stacked_view(tensor, layer, layers)
    return PyptoQwen3WeightBundle(**fields)
