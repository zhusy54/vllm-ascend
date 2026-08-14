#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""One Megatron TP=2 transformer layer as one ordered ChipWorker graph."""

from __future__ import annotations

import os

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto.runtime import RunConfig

from vllm_ascend.models.pypto_qwen3_tp import tp_compile_output_dir
from vllm_ascend.models.pypto_qwen3_tp_fused_ops import (
    ATTN_SCALE,
    CACHE_ROWS,
    HEAD_DIM,
    HEADS_TP,
    HIDDEN,
    KV_HEADS_TP,
    KV_HIDDEN_TP,
    MAX_CACHE,
    MAX_SEQ,
    Q_PER_KV,
    TOK,
    TP_WORLD,
    k_norm_rope,
    q_norm_rope,
    residual_add,
    rms_hidden,
    swiglu,
    tp_allreduce,
    write_kv,
)
from vllm_ascend.models.pypto_qwen3_tp_kernels import (
    HIDDEN_TP,
    INTER_TP,
    gemm_h_htp_step,
    gemm_h_inter_step,
    gemm_h_kv_step,
    gemm_htp_h_step,
    gemm_inter_h_step,
)

ATTN_ROWS = HEADS_TP * TOK
CACHE_TILES = MAX_CACHE // TOK


@pl.jit.inline
def _cast_htp_to_bf16(
    src: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    dst: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.BF16]],
):
    for h in pl.range(HEADS_TP):
        h0 = h * HEAD_DIM
        dst = pl.assemble(
            dst,
            pl.cast(pl.slice(src, [TOK, HEAD_DIM], [0, h0]), target_type=pl.BF16),
            [0, h0],
        )
    return dst


@pl.jit.inline
def _cast_htp_to_fp32(
    src: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    dst: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    for h in pl.range(HEADS_TP):
        h0 = h * HEAD_DIM
        dst = pl.assemble(
            dst,
            pl.cast(pl.slice(src, [TOK, HEAD_DIM], [0, h0]), target_type=pl.FP32),
            [0, h0],
        )
    return dst


@pl.jit.inline
def _cast_kv_to_bf16(
    src: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    dst: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16]],
):
    for h in pl.range(KV_HEADS_TP):
        h0 = h * HEAD_DIM
        dst = pl.assemble(
            dst,
            pl.cast(pl.slice(src, [TOK, HEAD_DIM], [0, h0]), target_type=pl.BF16),
            [0, h0],
        )
    return dst


@pl.jit.inline
def _cast_kv_to_fp32(
    src: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    dst: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    for h in pl.range(KV_HEADS_TP):
        h0 = h * HEAD_DIM
        dst = pl.assemble(
            dst,
            pl.cast(pl.slice(src, [TOK, HEAD_DIM], [0, h0]), target_type=pl.FP32),
            [0, h0],
        )
    return dst


@pl.jit.incore
def tp_hidden_rms_step(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    xn: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    """Pure AIV: normalize hidden before the QKV cube task."""
    return rms_hidden(hidden, weight, xn)


@pl.jit.incore
def tp_qkv_gemm_step(
    xn: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    wq: pl.Tensor[[HIDDEN, HIDDEN_TP], pl.BF16],
    wk: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    wv: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    q_fp: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
    k_fp: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
    v_fp: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    """Pure AIC: compute all three projections after RMS has completed."""
    q_fp = gemm_h_htp_step(xn, wq, q_fp)
    k_fp = gemm_h_kv_step(xn, wk, k_fp)
    v_fp = gemm_h_kv_step(xn, wv, v_fp)
    return q_fp, k_fp, v_fp


@pl.jit.incore
def tp_qkv_post_step(
    q_fp: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    k_fp: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    v_fp: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    q_norm: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    k_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    v_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    q_bf: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.BF16]],
    k_bf: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16]],
    v_bf: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16]],
    layer_idx: pl.Scalar[pl.INDEX],
    pos0: pl.Scalar[pl.INDEX],
    n_tok: pl.Scalar[pl.INDEX],
):
    """Pure AIV: QK norm/RoPE, V cast, and cache writeback."""
    cos = pl.slice(rope_cos, [TOK, HEAD_DIM], [pos0, 0])
    sin = pl.slice(rope_sin, [TOK, HEAD_DIM], [pos0, 0])
    q_bf = q_norm_rope(q_fp, q_norm, cos, sin, q_bf)
    k_bf = k_norm_rope(k_fp, k_norm, cos, sin, k_bf)
    v_bf = _cast_kv_to_bf16(v_fp, v_bf)
    row0 = layer_idx * MAX_CACHE + pos0
    k_cache, v_cache = write_kv(k_bf, v_bf, k_cache, v_cache, row0, n_tok)
    return k_cache, v_cache, q_bf, k_bf, v_bf


@pl.jit.incore
def tp_prefill_attention_prepare_step(
    q_bf: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    k_bf: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    v_bf: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    q_fp: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
    k_fp: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
    v_fp: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    """Pure AIV: materialize BF16 QKV as FP32 GM inputs for attention."""
    q_fp = _cast_htp_to_fp32(q_bf, q_fp)
    k_fp = _cast_kv_to_fp32(k_bf, k_fp)
    v_fp = _cast_kv_to_fp32(v_bf, v_fp)
    return q_fp, k_fp, v_fp


@pl.jit.incore
def tp_prefill_attention_scores_step(
    q_fp: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    k_fp: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    scores: pl.InOut[pl.Tensor[[ATTN_ROWS, TOK], pl.FP32]],
):
    """Pure AIC: one raw QK^T score matrix per local query head."""
    for h in pl.range(HEADS_TP):
        kv_h = h // Q_PER_KV
        raw = pl.matmul(
            pl.slice(q_fp, [TOK, HEAD_DIM], [0, h * HEAD_DIM]),
            pl.slice(k_fp, [TOK, HEAD_DIM], [0, kv_h * HEAD_DIM]),
            out_dtype=pl.FP32,
            b_trans=True,
        )
        scores = pl.assemble(scores, raw, [h * TOK, 0])
    return scores


@pl.jit.incore
def tp_prefill_attention_softmax_step(
    scores: pl.Tensor[[ATTN_ROWS, TOK], pl.FP32],
    mask: pl.Tensor[[TOK, TOK], pl.FP32],
    probs: pl.InOut[pl.Tensor[[ATTN_ROWS, TOK], pl.FP32]],
):
    """Pure AIV: scale, mask, and softmax each head's scores."""
    for h in pl.range(HEADS_TP):
        score_h = pl.mul(pl.slice(scores, [TOK, TOK], [h * TOK, 0]), ATTN_SCALE)
        masked = pl.add(score_h, mask)
        shifted = pl.exp(pl.row_expand_sub(masked, pl.row_max(masked)))
        prob_h = pl.row_expand_div(shifted, pl.row_sum(shifted))
        probs = pl.assemble(probs, prob_h, [h * TOK, 0])
    return probs


@pl.jit.incore
def tp_prefill_attention_context_step(
    probs: pl.Tensor[[ATTN_ROWS, TOK], pl.FP32],
    v_fp: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    context: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    """Pure AIC: multiply softmax probabilities by V."""
    for h in pl.range(HEADS_TP):
        kv_h = h // Q_PER_KV
        ctx = pl.matmul(
            pl.slice(probs, [TOK, TOK], [h * TOK, 0]),
            pl.slice(v_fp, [TOK, HEAD_DIM], [0, kv_h * HEAD_DIM]),
            out_dtype=pl.FP32,
        )
        context = pl.assemble(context, ctx, [0, h * HEAD_DIM])
    return context


@pl.jit.incore
def tp_decode_attention_prepare_step(
    q_bf: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    k_cache: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.BF16],
    v_cache: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.BF16],
    q_fp: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
    k_ctx_fp: pl.InOut[pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.FP32]],
    v_ctx_fp: pl.InOut[pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.FP32]],
):
    """Pure AIV: cast decode Q and the selected layer cache to FP32."""
    q_fp = _cast_htp_to_fp32(q_bf, q_fp)
    for sb in pl.range(CACHE_TILES):
        s0 = sb * TOK
        for h in pl.range(KV_HEADS_TP):
            h0 = h * HEAD_DIM
            k_ctx_fp = pl.assemble(
                k_ctx_fp,
                pl.cast(
                    pl.slice(k_cache, [TOK, HEAD_DIM], [s0, h0]),
                    target_type=pl.FP32,
                ),
                [s0, h0],
            )
            v_ctx_fp = pl.assemble(
                v_ctx_fp,
                pl.cast(
                    pl.slice(v_cache, [TOK, HEAD_DIM], [s0, h0]),
                    target_type=pl.FP32,
                ),
                [s0, h0],
            )
    return q_fp, k_ctx_fp, v_ctx_fp


@pl.jit.incore
def tp_decode_attention_scores_step(
    q_fp: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    k_ctx_fp: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.FP32],
    scores: pl.InOut[pl.Tensor[[ATTN_ROWS, MAX_CACHE], pl.FP32]],
):
    """Pure AIC: build decode QK^T scores in TOK-wide cache tiles."""
    for h in pl.range(HEADS_TP):
        kv_h = h // Q_PER_KV
        for sb in pl.range(CACHE_TILES):
            s0 = sb * TOK
            raw = pl.matmul(
                pl.slice(q_fp, [TOK, HEAD_DIM], [0, h * HEAD_DIM]),
                pl.slice(k_ctx_fp, [TOK, HEAD_DIM], [s0, kv_h * HEAD_DIM]),
                out_dtype=pl.FP32,
                b_trans=True,
            )
            scores = pl.assemble(scores, raw, [h * TOK, s0])
    return scores


@pl.jit.incore
def tp_decode_attention_softmax_step(
    scores: pl.Tensor[[ATTN_ROWS, MAX_CACHE], pl.FP32],
    mask: pl.Tensor[[TOK, MAX_CACHE], pl.FP32],
    probs: pl.InOut[pl.Tensor[[ATTN_ROWS, MAX_CACHE], pl.FP32]],
):
    """Pure AIV: full-cache masked softmax, matching the eager reference."""
    for h in pl.range(HEADS_TP):
        score_h = pl.mul(
            pl.slice(scores, [TOK, MAX_CACHE], [h * TOK, 0]),
            ATTN_SCALE,
        )
        masked = pl.add(score_h, mask)
        shifted = pl.exp(pl.row_expand_sub(masked, pl.row_max(masked)))
        prob_h = pl.row_expand_div(shifted, pl.row_sum(shifted))
        probs = pl.assemble(probs, prob_h, [h * TOK, 0])
    return probs


@pl.jit.incore
def tp_decode_attention_context_step(
    probs: pl.Tensor[[ATTN_ROWS, MAX_CACHE], pl.FP32],
    v_ctx_fp: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.FP32],
    context: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    """Pure AIC: accumulate the four cache tiles into each head context."""
    for h in pl.range(HEADS_TP):
        kv_h = h // Q_PER_KV
        ctx = pl.matmul(
            pl.slice(probs, [TOK, TOK], [h * TOK, 0]),
            pl.slice(v_ctx_fp, [TOK, HEAD_DIM], [0, kv_h * HEAD_DIM]),
            out_dtype=pl.FP32,
        )
        for sb in pl.range(1, CACHE_TILES):
            s0 = sb * TOK
            ctx = pl.matmul_acc(
                ctx,
                pl.slice(probs, [TOK, TOK], [h * TOK, s0]),
                pl.slice(v_ctx_fp, [TOK, HEAD_DIM], [s0, kv_h * HEAD_DIM]),
            )
        context = pl.assemble(context, ctx, [0, h * HEAD_DIM])
    return context


@pl.jit.incore
def tp_attention_cast_step(
    context: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    q_bf: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.BF16]],
):
    """Pure AIV: quantize attention context before the row-parallel O GEMM."""
    return _cast_htp_to_bf16(context, q_bf)


@pl.jit.incore
def tp_o_gemm_step(
    context: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    wo: pl.Tensor[[HIDDEN_TP, HIDDEN], pl.BF16],
    partial: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    """Pure AIC: row-parallel O projection."""
    return gemm_htp_h_step(context, wo, partial)


@pl.jit.incore
def tp_ffn_residual_rms_step(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    reduced: pl.Tensor[[TOK, HIDDEN], pl.FP32],
    post_rms: pl.Tensor[[1, HIDDEN], pl.FP32],
    hidden_out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
    xn: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    """Pure AIV: attention residual followed by post-attention RMSNorm."""
    hidden_out = residual_add(hidden, reduced, hidden_out)
    xn = rms_hidden(hidden_out, post_rms, xn)
    return hidden_out, xn


@pl.jit.incore
def tp_ffn_gate_up_step(
    xn: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    gate: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
    up: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
):
    """Pure AIC: gate and up projections."""
    gate = gemm_h_inter_step(xn, w_gate, gate)
    up = gemm_h_inter_step(xn, w_up, up)
    return gate, up


@pl.jit.incore
def tp_ffn_swiglu_step(
    gate: pl.Tensor[[TOK, INTER_TP], pl.FP32],
    up: pl.Tensor[[TOK, INTER_TP], pl.FP32],
    act: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.BF16]],
):
    """Pure AIV: SiLU(gate) * up and BF16 conversion."""
    return swiglu(gate, up, act)


@pl.jit.incore
def tp_ffn_down_step(
    act: pl.Tensor[[TOK, INTER_TP], pl.BF16],
    w_down: pl.Tensor[[INTER_TP, HIDDEN], pl.BF16],
    partial: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    """Pure AIC: row-parallel down projection."""
    return gemm_inter_h_step(act, w_down, partial)


@pl.jit.incore
def tp_residual_tail(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    reduced: pl.Tensor[[TOK, HIDDEN], pl.FP32],
    hidden_out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    """Pure AIV: finish the FFN residual."""
    return residual_add(hidden, reduced, hidden_out)


@pl.jit.incore
def tp_allreduce_step(
    partial: pl.Tensor[[TOK, HIDDEN], pl.FP32],
    data: pl.InOut[pld.DistributedTensor[[TOK, HIDDEN], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[TP_WORLD, 1], pl.INT32]],
    reduced: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
    expected_publish: pl.Scalar[pl.INDEX],
    expected_consume: pl.Scalar[pl.INDEX],
):
    """Pure AIV: one allreduce boundary with publish and consume credits."""
    return tp_allreduce(
        partial,
        data,
        signal,
        reduced,
        expected_publish,
        expected_consume,
    )


@pl.jit
def tp_fused_layer_prefill_chip(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    input_rms: pl.Tensor[[1, HIDDEN], pl.FP32],
    wq: pl.Tensor[[HIDDEN, HIDDEN_TP], pl.BF16],
    wk: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    wv: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    q_norm: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    wo: pl.Tensor[[HIDDEN_TP, HIDDEN], pl.BF16],
    post_rms: pl.Tensor[[1, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    w_down: pl.Tensor[[INTER_TP, HIDDEN], pl.BF16],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    attn_mask: pl.Tensor[[TOK, TOK], pl.FP32],
    k_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    v_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[TOK, HIDDEN], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[TP_WORLD, 1], pl.INT32]],
    hidden_out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
    meta: pl.Tensor[[3], pl.INT32],
):
    n_tok = pl.cast(pl.tensor.read(meta, [0]), pl.INDEX)
    pos0 = pl.cast(pl.tensor.read(meta, [1]), pl.INDEX)
    credit_base = pl.cast(pl.tensor.read(meta, [2]), pl.INDEX)
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

    xn_attn = tp_hidden_rms_step(hidden, input_rms, xn_attn)
    q_proj_fp, k_proj_fp, v_proj_fp = tp_qkv_gemm_step(
        xn_attn,
        wq,
        wk,
        wv,
        q_proj_fp,
        k_proj_fp,
        v_proj_fp,
    )
    k_cache, v_cache, q_rope_bf, k_rope_bf, v_proj_bf = tp_qkv_post_step(
        q_proj_fp,
        k_proj_fp,
        v_proj_fp,
        q_norm,
        k_norm,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        q_rope_bf,
        k_rope_bf,
        v_proj_bf,
        0,
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
    attn_probs = tp_prefill_attention_softmax_step(attn_scores, attn_mask, attn_probs)
    context_fp = tp_prefill_attention_context_step(attn_probs, v_attn_fp, context_fp)
    context_bf = tp_attention_cast_step(context_fp, context_bf)
    o_partial = tp_o_gemm_step(context_bf, wo, o_partial)

    pub_o = credit_base + 1
    cons_o = credit_base + 2
    pub_d = credit_base + 3
    cons_d = credit_base + 4
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
        post_rms,
        attn_residual,
        xn_ffn,
    )
    gate, up = tp_ffn_gate_up_step(xn_ffn, w_gate, w_up, gate, up)
    act = tp_ffn_swiglu_step(gate, up, act)
    down_partial = tp_ffn_down_step(act, w_down, down_partial)
    data, signal, down_reduced = tp_allreduce_step(
        down_partial,
        data,
        signal,
        down_reduced,
        pub_d,
        cons_d,
    )
    hidden_out = tp_residual_tail(attn_residual, down_reduced, hidden_out)
    return hidden_out


@pl.jit
def tp_fused_layer_decode_chip(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    input_rms: pl.Tensor[[1, HIDDEN], pl.FP32],
    wq: pl.Tensor[[HIDDEN, HIDDEN_TP], pl.BF16],
    wk: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    wv: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    q_norm: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    wo: pl.Tensor[[HIDDEN_TP, HIDDEN], pl.BF16],
    post_rms: pl.Tensor[[1, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    w_down: pl.Tensor[[INTER_TP, HIDDEN], pl.BF16],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    attn_mask: pl.Tensor[[TOK, MAX_CACHE], pl.FP32],
    k_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    v_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    data: pl.InOut[pld.DistributedTensor[[TOK, HIDDEN], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[TP_WORLD, 1], pl.INT32]],
    hidden_out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
    meta: pl.Tensor[[3], pl.INT32],
):
    n_tok = pl.cast(pl.tensor.read(meta, [0]), pl.INDEX)
    pos0 = pl.cast(pl.tensor.read(meta, [1]), pl.INDEX)
    credit_base = pl.cast(pl.tensor.read(meta, [2]), pl.INDEX)
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

    xn_attn = tp_hidden_rms_step(hidden, input_rms, xn_attn)
    q_proj_fp, k_proj_fp, v_proj_fp = tp_qkv_gemm_step(
        xn_attn,
        wq,
        wk,
        wv,
        q_proj_fp,
        k_proj_fp,
        v_proj_fp,
    )
    k_cache, v_cache, q_rope_bf, k_rope_bf, v_proj_bf = tp_qkv_post_step(
        q_proj_fp,
        k_proj_fp,
        v_proj_fp,
        q_norm,
        k_norm,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        q_rope_bf,
        k_rope_bf,
        v_proj_bf,
        0,
        pos0,
        n_tok,
    )
    k_layer = pl.slice(k_cache, [MAX_CACHE, KV_HIDDEN_TP], [0, 0])
    v_layer = pl.slice(v_cache, [MAX_CACHE, KV_HIDDEN_TP], [0, 0])
    q_attn_fp, k_ctx_fp, v_ctx_fp = tp_decode_attention_prepare_step(
        q_rope_bf,
        k_layer,
        v_layer,
        q_attn_fp,
        k_ctx_fp,
        v_ctx_fp,
    )
    attn_scores = tp_decode_attention_scores_step(q_attn_fp, k_ctx_fp, attn_scores)
    attn_probs = tp_decode_attention_softmax_step(attn_scores, attn_mask, attn_probs)
    context_fp = tp_decode_attention_context_step(attn_probs, v_ctx_fp, context_fp)
    context_bf = tp_attention_cast_step(context_fp, context_bf)
    o_partial = tp_o_gemm_step(context_bf, wo, o_partial)

    pub_o = credit_base + 1
    cons_o = credit_base + 2
    pub_d = credit_base + 3
    cons_d = credit_base + 4
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
        post_rms,
        attn_residual,
        xn_ffn,
    )
    gate, up = tp_ffn_gate_up_step(xn_ffn, w_gate, w_up, gate, up)
    act = tp_ffn_swiglu_step(gate, up, act)
    down_partial = tp_ffn_down_step(act, w_down, down_partial)
    data, signal, down_reduced = tp_allreduce_step(
        down_partial,
        data,
        signal,
        down_reduced,
        pub_d,
        cons_d,
    )
    hidden_out = tp_residual_tail(attn_residual, down_reduced, hidden_out)
    return hidden_out


def compile_fused_layer_chip(decode: bool = False, device_id: int = 0):
    """Compile the one-layer prefill or decode chip on CPU samples."""
    cfg = RunConfig(
        platform=os.environ.get("PTO_PLATFORM", "a2a3"),
        device_id=int(device_id),
        ring_dep_pool=int(os.environ.get("PTO2_RING_DEP_POOL", "262144")),
        save_kernels_dir=tp_compile_output_dir(
            "tp_fused_layer_decode_chip" if decode else "tp_fused_layer_prefill_chip"
        ),
    )
    mask_cols = MAX_CACHE if decode else TOK
    sample = (
        torch.zeros((TOK, HIDDEN), dtype=torch.bfloat16),
        torch.zeros((1, HIDDEN), dtype=torch.float32),
        torch.zeros((HIDDEN, HIDDEN_TP), dtype=torch.bfloat16),
        torch.zeros((HIDDEN, KV_HIDDEN_TP), dtype=torch.bfloat16),
        torch.zeros((HIDDEN, KV_HIDDEN_TP), dtype=torch.bfloat16),
        torch.zeros((1, HEAD_DIM), dtype=torch.float32),
        torch.zeros((1, HEAD_DIM), dtype=torch.float32),
        torch.zeros((HIDDEN_TP, HIDDEN), dtype=torch.bfloat16),
        torch.zeros((1, HIDDEN), dtype=torch.float32),
        torch.zeros((HIDDEN, INTER_TP), dtype=torch.bfloat16),
        torch.zeros((HIDDEN, INTER_TP), dtype=torch.bfloat16),
        torch.zeros((INTER_TP, HIDDEN), dtype=torch.bfloat16),
        torch.zeros((MAX_SEQ, HEAD_DIM), dtype=torch.float32),
        torch.zeros((MAX_SEQ, HEAD_DIM), dtype=torch.float32),
        torch.zeros((TOK, mask_cols), dtype=torch.float32),
        torch.zeros((CACHE_ROWS, KV_HIDDEN_TP), dtype=torch.bfloat16),
        torch.zeros((CACHE_ROWS, KV_HIDDEN_TP), dtype=torch.bfloat16),
        torch.zeros((TOK, HIDDEN), dtype=torch.float32),
        torch.zeros((TP_WORLD, 1), dtype=torch.int32),
        torch.zeros((TOK, HIDDEN), dtype=torch.bfloat16),
        torch.zeros((3,), dtype=torch.int32),
    )
    chip = tp_fused_layer_decode_chip if decode else tp_fused_layer_prefill_chip
    return chip.compile(*sample, config=cfg)
