#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Inlined TP=2 vector / comm / embed / lm_head steps for the fused host."""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from vllm_ascend.models.pypto_qwen3_tp_kernels import (
    HIDDEN,
    HIDDEN_TP,
    INTER_TP,
    K_CHUNK,
    KV_HIDDEN_TP,
    N_CHUNK,
    TOK,
)

HEAD_DIM = 128
HALF_DIM = 64
HEADS_TP = 20
KV_HEADS_TP = 4
Q_PER_KV = 5
NUM_LAYERS = 40
MAX_CACHE = 128
CACHE_ROWS = NUM_LAYERS * MAX_CACHE
MAX_SEQ = 4096
VOCAB = 152064
EPS = 1e-6
HIDDEN_INV = 1.0 / float(HIDDEN)
HEAD_DIM_INV = 1.0 / float(HEAD_DIM)
ATTN_SCALE = HEAD_DIM**-0.5
RMS_CHUNK = 128
SILU_CHUNK = 128
TP_WORLD = 2
NEG_INF = -1.0e9


@pl.jit.inline
def rms_hidden(
    x: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    gamma: pl.Tensor[[1, HIDDEN], pl.FP32],
    y: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    sq_sum = pl.full([1, TOK], dtype=pl.FP32, value=0.0)
    for hb in pl.range(HIDDEN // RMS_CHUNK):
        h0 = hb * RMS_CHUNK
        tile = pl.cast(pl.slice(x, [TOK, RMS_CHUNK], [0, h0]), target_type=pl.FP32)
        sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(tile, tile)), [1, TOK]))
    inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)), [TOK, 1])
    for hb in pl.range(HIDDEN // RMS_CHUNK):
        h0 = hb * RMS_CHUNK
        tile = pl.cast(pl.slice(x, [TOK, RMS_CHUNK], [0, h0]), target_type=pl.FP32)
        weight = pl.slice(gamma, [1, RMS_CHUNK], [0, h0])
        normed = pl.col_expand_mul(pl.row_expand_mul(tile, inv), weight)
        y = pl.assemble(y, pl.cast(normed, target_type=pl.BF16), [0, h0])
    return y


@pl.jit.inline
def q_norm_rope(
    heads: pl.Tensor[[TOK, HIDDEN_TP], pl.FP32],
    gamma: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[TOK, HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[TOK, HEAD_DIM], pl.FP32],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.BF16]],
):
    for h in pl.range(HEADS_TP):
        h0 = h * HEAD_DIM
        qh = pl.slice(heads, [TOK, HEAD_DIM], [0, h0])
        sq = pl.reshape(pl.row_sum(pl.mul(qh, qh)), [1, TOK])
        inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq, HEAD_DIM_INV), EPS)), [TOK, 1])
        normed = pl.col_expand_mul(pl.row_expand_mul(qh, inv), gamma)
        lo = pl.slice(normed, [TOK, HALF_DIM], [0, 0])
        hi = pl.slice(normed, [TOK, HALF_DIM], [0, HALF_DIM])
        cos_lo = pl.slice(cos, [TOK, HALF_DIM], [0, 0])
        cos_hi = pl.slice(cos, [TOK, HALF_DIM], [0, HALF_DIM])
        sin_lo = pl.slice(sin, [TOK, HALF_DIM], [0, 0])
        sin_hi = pl.slice(sin, [TOK, HALF_DIM], [0, HALF_DIM])
        rot_lo = pl.sub(pl.mul(lo, cos_lo), pl.mul(hi, sin_lo))
        rot_hi = pl.add(pl.mul(hi, cos_hi), pl.mul(lo, sin_hi))
        out = pl.assemble(out, pl.cast(rot_lo, target_type=pl.BF16), [0, h0])
        out = pl.assemble(out, pl.cast(rot_hi, target_type=pl.BF16), [0, h0 + HALF_DIM])
    return out


@pl.jit.inline
def k_norm_rope(
    heads: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32],
    gamma: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[TOK, HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[TOK, HEAD_DIM], pl.FP32],
    out: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16]],
):
    for h in pl.range(KV_HEADS_TP):
        h0 = h * HEAD_DIM
        kh = pl.slice(heads, [TOK, HEAD_DIM], [0, h0])
        sq = pl.reshape(pl.row_sum(pl.mul(kh, kh)), [1, TOK])
        inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq, HEAD_DIM_INV), EPS)), [TOK, 1])
        normed = pl.col_expand_mul(pl.row_expand_mul(kh, inv), gamma)
        lo = pl.slice(normed, [TOK, HALF_DIM], [0, 0])
        hi = pl.slice(normed, [TOK, HALF_DIM], [0, HALF_DIM])
        cos_lo = pl.slice(cos, [TOK, HALF_DIM], [0, 0])
        cos_hi = pl.slice(cos, [TOK, HALF_DIM], [0, HALF_DIM])
        sin_lo = pl.slice(sin, [TOK, HALF_DIM], [0, 0])
        sin_hi = pl.slice(sin, [TOK, HALF_DIM], [0, HALF_DIM])
        rot_lo = pl.sub(pl.mul(lo, cos_lo), pl.mul(hi, sin_lo))
        rot_hi = pl.add(pl.mul(hi, cos_hi), pl.mul(lo, sin_hi))
        out = pl.assemble(out, pl.cast(rot_lo, target_type=pl.BF16), [0, h0])
        out = pl.assemble(out, pl.cast(rot_hi, target_type=pl.BF16), [0, h0 + HALF_DIM])
    return out


@pl.jit.inline
def prefill_gqa(
    query: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    key: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    value: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    mask: pl.Tensor[[TOK, TOK], pl.FP32],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    for h in pl.range(HEADS_TP):
        kv_h = h // Q_PER_KV
        qh = pl.cast(pl.slice(query, [TOK, HEAD_DIM], [0, h * HEAD_DIM]), target_type=pl.FP32)
        kh = pl.cast(pl.slice(key, [TOK, HEAD_DIM], [0, kv_h * HEAD_DIM]), target_type=pl.FP32)
        vh = pl.cast(pl.slice(value, [TOK, HEAD_DIM], [0, kv_h * HEAD_DIM]), target_type=pl.FP32)
        scores = pl.mul(pl.matmul(qh, kh, out_dtype=pl.FP32, b_trans=True), ATTN_SCALE)
        scores = pl.add(scores, mask)
        shifted = pl.exp(pl.row_expand_sub(scores, pl.row_max(scores)))
        probs = pl.row_expand_div(shifted, pl.row_sum(shifted))
        ctx = pl.matmul(probs, vh, out_dtype=pl.FP32)
        out = pl.assemble(out, ctx, [0, h * HEAD_DIM])
    return out


@pl.jit.incore
def decode_gqa(
    query: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    key: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.BF16],
    value: pl.Tensor[[MAX_CACHE, KV_HIDDEN_TP], pl.BF16],
    mask: pl.Tensor[[TOK, MAX_CACHE], pl.FP32],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    seq_tile = TOK
    for h in pl.range(HEADS_TP):
        kv_h = h // Q_PER_KV
        qh = pl.cast(pl.slice(query, [TOK, HEAD_DIM], [0, h * HEAD_DIM]), target_type=pl.FP32)
        # PTOAS requires a row-major FP32 tile row to be at least 32 bytes.
        # Reductions turn aligned [TOK, 8] seeds into the [TOK, 1]
        # column-vector state required by online softmax.
        m_i = pl.row_max(pl.full([TOK, 8], dtype=pl.FP32, value=-1.0e30))
        l_i = pl.row_sum(pl.full([TOK, 8], dtype=pl.FP32, value=0.0))
        o_i = pl.full([TOK, HEAD_DIM], dtype=pl.FP32, value=0.0)
        for sb in pl.range(MAX_CACHE // seq_tile):
            s0 = sb * seq_tile
            kh = pl.cast(
                pl.slice(key, [seq_tile, HEAD_DIM], [s0, kv_h * HEAD_DIM]),
                target_type=pl.FP32,
            )
            vh = pl.cast(
                pl.slice(value, [seq_tile, HEAD_DIM], [s0, kv_h * HEAD_DIM]),
                target_type=pl.FP32,
            )
            scores = pl.mul(pl.matmul(qh, kh, out_dtype=pl.FP32, b_trans=True), ATTN_SCALE)
            scores = pl.add(scores, pl.slice(mask, [TOK, seq_tile], [0, s0]))
            m_blk = pl.reshape(pl.row_max(scores), [TOK, 1])
            m_new = pl.maximum(m_i, m_blk)
            alpha = pl.exp(pl.sub(m_i, m_new))
            probs = pl.exp(pl.row_expand_sub(scores, m_new))
            l_i = pl.add(pl.mul(l_i, alpha), pl.reshape(pl.row_sum(probs), [TOK, 1]))
            o_i = pl.add(pl.row_expand_mul(o_i, alpha), pl.matmul(probs, vh, out_dtype=pl.FP32))
            m_i = m_new
        o_i = pl.row_expand_div(o_i, l_i)
        out = pl.assemble(out, o_i, [0, h * HEAD_DIM])
    return out


@pl.jit.inline
def swiglu(
    gate: pl.Tensor[[TOK, INTER_TP], pl.FP32],
    up: pl.Tensor[[TOK, INTER_TP], pl.FP32],
    out: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.BF16]],
):
    for nb in pl.range(INTER_TP // SILU_CHUNK):
        n0 = nb * SILU_CHUNK
        gate_t = pl.slice(gate, [TOK, SILU_CHUNK], [0, n0])
        up_t = pl.slice(up, [TOK, SILU_CHUNK], [0, n0])
        sigmoid = pl.recip(pl.add(pl.exp(pl.mul(gate_t, -1.0)), 1.0))
        act = pl.mul(pl.mul(gate_t, sigmoid), up_t)
        out = pl.assemble(out, pl.cast(act, target_type=pl.BF16), [0, n0])
    return out


@pl.jit.inline
def residual_add(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    delta: pl.Tensor[[TOK, HIDDEN], pl.FP32],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    for hb in pl.range(HIDDEN // RMS_CHUNK):
        h0 = hb * RMS_CHUNK
        hid = pl.cast(pl.slice(hidden, [TOK, RMS_CHUNK], [0, h0]), target_type=pl.FP32)
        add = pl.add(hid, pl.slice(delta, [TOK, RMS_CHUNK], [0, h0]))
        out = pl.assemble(out, pl.cast(add, target_type=pl.BF16), [0, h0])
    return out


@pl.jit.inline
def write_kv(
    key: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    value: pl.Tensor[[TOK, KV_HIDDEN_TP], pl.BF16],
    k_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    v_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_HIDDEN_TP], pl.BF16]],
    row0: pl.Scalar[pl.INDEX],
    n_tok: pl.Scalar[pl.INDEX],
):
    for t in pl.range(TOK):
        if t < n_tok:
            k_cache = pl.assemble(k_cache, pl.slice(key, [1, KV_HIDDEN_TP], [t, 0]), [row0 + t, 0])
            v_cache = pl.assemble(v_cache, pl.slice(value, [1, KV_HIDDEN_TP], [t, 0]), [row0 + t, 0])
    return k_cache, v_cache


@pl.jit.inline
def tp_allreduce(
    partial: pl.Tensor[[TOK, HIDDEN], pl.FP32],
    data: pl.InOut[pld.DistributedTensor[[TOK, HIDDEN], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[TP_WORLD, 1], pl.INT32]],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
    expected_pub: pl.Scalar[pl.INDEX],
    expected_cons: pl.Scalar[pl.INDEX],
):
    for t in pl.range(TOK):
        data = pl.store(pl.load(partial, [t, 0], [1, HIDDEN]), [t, 0], data)
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)
    peer = (my_rank + 1) % nranks
    pld.system.notify(signal, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    pld.system.wait(signal=signal, offsets=[peer, 0], expected=expected_pub, cmp=pld.WaitCmp.Ge)
    for t in pl.range(TOK):
        local = pl.load(data, [t, 0], [1, HIDDEN])
        remote = pld.tile.remote_load(data, peer=peer, offsets=[t, 0], shape=[1, HIDDEN])
        out = pl.store(pl.add(local, remote), [t, 0], out)
    pld.system.notify(signal, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    pld.system.wait(signal=signal, offsets=[peer, 0], expected=expected_cons, cmp=pld.WaitCmp.Ge)
    return data, signal, out


@pl.jit.inline
def tp_embed(
    input_ids: pl.Tensor[[TOK], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    hidden: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
    n_tok: pl.Scalar[pl.INDEX],
):
    for t in pl.range(TOK):
        if t < n_tok:
            token_row = pl.cast(pl.tensor.read(input_ids, [t]), pl.INDEX)
            for k0 in pl.range(0, HIDDEN, K_CHUNK):
                hidden = pl.assemble(
                    hidden,
                    pl.slice(embed_weight, [1, K_CHUNK], [token_row, k0]),
                    [t, k0],
                )
    return hidden


@pl.jit.inline
def tp_lm_head(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    logits: pl.InOut[pl.Tensor[[TOK, VOCAB], pl.FP32]],
):
    for nb in pl.range(VOCAB // N_CHUNK):
        n0 = nb * N_CHUNK
        acc = pl.matmul(
            pl.slice(hidden, [TOK, K_CHUNK], [0, 0]),
            pl.slice(weight, [N_CHUNK, K_CHUNK], [n0, 0]),
            out_dtype=pl.FP32,
            b_trans=True,
        )
        for kb in pl.range(1, HIDDEN // K_CHUNK):
            k0 = kb * K_CHUNK
            acc = pl.matmul_acc(
                acc,
                pl.slice(hidden, [TOK, K_CHUNK], [0, k0]),
                pl.slice(weight, [N_CHUNK, K_CHUNK], [n0, k0]),
                b_trans=True,
            )
        logits = pl.assemble(logits, acc, [0, n0])
    return logits


@pl.jit.incore
def tp_embed_step(
    input_ids: pl.Tensor[[TOK], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    hidden: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
    n_tok: pl.Scalar[pl.INDEX],
):
    return tp_embed(input_ids, embed_weight, hidden, n_tok)


@pl.jit.incore
def rms_hidden_step(
    x: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    gamma: pl.Tensor[[1, HIDDEN], pl.FP32],
    y: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.BF16]],
):
    return rms_hidden(x, gamma, y)


@pl.jit.incore
def tp_lm_head_step(
    hidden: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    logits: pl.InOut[pl.Tensor[[TOK, VOCAB], pl.FP32]],
):
    return tp_lm_head(hidden, weight, logits)
