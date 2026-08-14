#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""TP=2 tiled GEMMs for Megatron-sharded Qwen3-14B contract weights.

One chip is ``[TOK, K] @ [K, N]`` with K=256 / N=128 tiles (512B K-axis,
L0B 64KiB). The token axis is the matmul M so cube fractal M stays 32.
"""

from __future__ import annotations

import pypto.language as pl

HIDDEN = 5120
HIDDEN_TP = 2560
KV_HIDDEN_TP = 512
INTER_TP = 8704
K_CHUNK = 256
N_CHUNK = 128
TOK = 32
HIDDEN_K_BLOCKS = HIDDEN // K_CHUNK
HIDDEN_N_BLOCKS = HIDDEN // N_CHUNK
HIDDEN_TP_K_BLOCKS = HIDDEN_TP // K_CHUNK
HIDDEN_TP_N_BLOCKS = HIDDEN_TP // N_CHUNK
KV_N_BLOCKS = KV_HIDDEN_TP // N_CHUNK
INTER_K_BLOCKS = INTER_TP // K_CHUNK
INTER_N_BLOCKS = INTER_TP // N_CHUNK


@pl.jit.inline
def gemm_h_htp_step(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, HIDDEN_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    for nb in pl.range(HIDDEN_TP_N_BLOCKS):
        n0 = nb * N_CHUNK
        acc = pl.matmul(
            pl.slice(left, [TOK, K_CHUNK], [0, 0]),
            pl.slice(weight, [K_CHUNK, N_CHUNK], [0, n0]),
            out_dtype=pl.FP32,
        )
        for kb in pl.range(1, HIDDEN_K_BLOCKS):
            k0 = kb * K_CHUNK
            acc = pl.matmul_acc(
                acc,
                pl.slice(left, [TOK, K_CHUNK], [0, k0]),
                pl.slice(weight, [K_CHUNK, N_CHUNK], [k0, n0]),
            )
        out = pl.assemble(out, acc, [0, n0])
    return out


@pl.jit.incore
def gemm_h_htp_incore(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, HIDDEN_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    return gemm_h_htp_step(left, weight, out)


@pl.jit
def gemm_h_htp_chip(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, HIDDEN_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN_TP], pl.FP32]],
):
    return gemm_h_htp_incore(left, weight, out)


@pl.jit.inline
def gemm_h_kv_step(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    for nb in pl.range(KV_N_BLOCKS):
        n0 = nb * N_CHUNK
        acc = pl.matmul(
            pl.slice(left, [TOK, K_CHUNK], [0, 0]),
            pl.slice(weight, [K_CHUNK, N_CHUNK], [0, n0]),
            out_dtype=pl.FP32,
        )
        for kb in pl.range(1, HIDDEN_K_BLOCKS):
            k0 = kb * K_CHUNK
            acc = pl.matmul_acc(
                acc,
                pl.slice(left, [TOK, K_CHUNK], [0, k0]),
                pl.slice(weight, [K_CHUNK, N_CHUNK], [k0, n0]),
            )
        out = pl.assemble(out, acc, [0, n0])
    return out


@pl.jit.incore
def gemm_h_kv_incore(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    return gemm_h_kv_step(left, weight, out)


@pl.jit
def gemm_h_kv_chip(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, KV_HIDDEN_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, KV_HIDDEN_TP], pl.FP32]],
):
    return gemm_h_kv_incore(left, weight, out)


@pl.jit.inline
def gemm_htp_h_step(
    left: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    weight: pl.Tensor[[HIDDEN_TP, HIDDEN], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    for nb in pl.range(HIDDEN_N_BLOCKS):
        n0 = nb * N_CHUNK
        acc = pl.matmul(
            pl.slice(left, [TOK, K_CHUNK], [0, 0]),
            pl.slice(weight, [K_CHUNK, N_CHUNK], [0, n0]),
            out_dtype=pl.FP32,
        )
        for kb in pl.range(1, HIDDEN_TP_K_BLOCKS):
            k0 = kb * K_CHUNK
            acc = pl.matmul_acc(
                acc,
                pl.slice(left, [TOK, K_CHUNK], [0, k0]),
                pl.slice(weight, [K_CHUNK, N_CHUNK], [k0, n0]),
            )
        out = pl.assemble(out, acc, [0, n0])
    return out


@pl.jit.incore
def gemm_htp_h_incore(
    left: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    weight: pl.Tensor[[HIDDEN_TP, HIDDEN], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    return gemm_htp_h_step(left, weight, out)


@pl.jit
def gemm_htp_h_chip(
    left: pl.Tensor[[TOK, HIDDEN_TP], pl.BF16],
    weight: pl.Tensor[[HIDDEN_TP, HIDDEN], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    return gemm_htp_h_incore(left, weight, out)


@pl.jit.inline
def gemm_h_inter_step(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
):
    for nb in pl.range(INTER_N_BLOCKS):
        n0 = nb * N_CHUNK
        acc = pl.matmul(
            pl.slice(left, [TOK, K_CHUNK], [0, 0]),
            pl.slice(weight, [K_CHUNK, N_CHUNK], [0, n0]),
            out_dtype=pl.FP32,
        )
        for kb in pl.range(1, HIDDEN_K_BLOCKS):
            k0 = kb * K_CHUNK
            acc = pl.matmul_acc(
                acc,
                pl.slice(left, [TOK, K_CHUNK], [0, k0]),
                pl.slice(weight, [K_CHUNK, N_CHUNK], [k0, n0]),
            )
        out = pl.assemble(out, acc, [0, n0])
    return out


@pl.jit.incore
def gemm_h_inter_incore(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
):
    return gemm_h_inter_step(left, weight, out)


@pl.jit
def gemm_h_inter_chip(
    left: pl.Tensor[[TOK, HIDDEN], pl.BF16],
    weight: pl.Tensor[[HIDDEN, INTER_TP], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, INTER_TP], pl.FP32]],
):
    return gemm_h_inter_incore(left, weight, out)


@pl.jit.inline
def gemm_inter_h_step(
    left: pl.Tensor[[TOK, INTER_TP], pl.BF16],
    weight: pl.Tensor[[INTER_TP, HIDDEN], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    for nb in pl.range(HIDDEN_N_BLOCKS):
        n0 = nb * N_CHUNK
        acc = pl.matmul(
            pl.slice(left, [TOK, K_CHUNK], [0, 0]),
            pl.slice(weight, [K_CHUNK, N_CHUNK], [0, n0]),
            out_dtype=pl.FP32,
        )
        for kb in pl.range(1, INTER_K_BLOCKS):
            k0 = kb * K_CHUNK
            acc = pl.matmul_acc(
                acc,
                pl.slice(left, [TOK, K_CHUNK], [0, k0]),
                pl.slice(weight, [K_CHUNK, N_CHUNK], [k0, n0]),
            )
        out = pl.assemble(out, acc, [0, n0])
    return out


@pl.jit.incore
def gemm_inter_h_incore(
    left: pl.Tensor[[TOK, INTER_TP], pl.BF16],
    weight: pl.Tensor[[INTER_TP, HIDDEN], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    return gemm_inter_h_step(left, weight, out)


@pl.jit
def gemm_inter_h_chip(
    left: pl.Tensor[[TOK, INTER_TP], pl.BF16],
    weight: pl.Tensor[[INTER_TP, HIDDEN], pl.BF16],
    out: pl.InOut[pl.Tensor[[TOK, HIDDEN], pl.FP32]],
):
    return gemm_inter_h_incore(left, weight, out)


TP_GEMM_KERNELS = {
    "q": (gemm_h_htp_chip, HIDDEN, HIDDEN_TP),
    "kv": (gemm_h_kv_chip, HIDDEN, KV_HIDDEN_TP),
    "o": (gemm_htp_h_chip, HIDDEN_TP, HIDDEN),
    "gate_up": (gemm_h_inter_chip, HIDDEN, INTER_TP),
    "down": (gemm_inter_h_chip, INTER_TP, HIDDEN),
}
