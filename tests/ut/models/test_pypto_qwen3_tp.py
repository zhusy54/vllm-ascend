#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""CPU tests for the shipped PyPTO Qwen3-14B TP=2 adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from vllm_ascend.models import pypto_qwen3_adapter as adapter
from vllm_ascend.models import pypto_qwen3_tp as tp
from vllm_ascend.models.pypto_qwen3_tp import (
    TP_BOUNDARY_DOWN_PROJ,
    TP_BOUNDARY_O_PROJ,
    describe_tp_compute_args,
    gather_contract_bundle,
    layer_stacked_view,
    select_compute_bundle,
    shard_contract_bundle,
    shard_first_dim,
    shard_last_dim,
    shard_stacked_row_parallel,
    tp_allreduce_slot_plan,
)


def _hf_state(
    *,
    num_layers: int,
    hidden: int,
    kv_hidden: int,
    intermediate: int,
    head_dim: int,
    vocab: int,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
    }
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}."
        state[prefix + "input_layernorm.weight"] = torch.ones(hidden)
        state[prefix + "self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        state[prefix + "self_attn.k_proj.weight"] = torch.randn(kv_hidden, hidden)
        state[prefix + "self_attn.v_proj.weight"] = torch.randn(kv_hidden, hidden)
        state[prefix + "self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        state[prefix + "self_attn.q_norm.weight"] = torch.ones(head_dim)
        state[prefix + "self_attn.k_norm.weight"] = torch.ones(head_dim)
        state[prefix + "post_attention_layernorm.weight"] = torch.ones(hidden)
        state[prefix + "mlp.gate_proj.weight"] = torch.randn(intermediate, hidden)
        state[prefix + "mlp.up_proj.weight"] = torch.randn(intermediate, hidden)
        state[prefix + "mlp.down_proj.weight"] = torch.randn(hidden, intermediate)
    return state


def test_shard_last_and_first_dim_round_trip_from_live_tensor() -> None:
    tensor = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    parts = [shard_last_dim(tensor, rank, 2) for rank in range(2)]
    torch.testing.assert_close(torch.cat(parts, dim=-1), tensor)
    tall = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    rows = [shard_first_dim(tall, rank, 2) for rank in range(2)]
    torch.testing.assert_close(torch.cat(rows, dim=0), tall)


def test_shard_contract_bundle_round_trips_packed_weights() -> None:
    state = _hf_state(
        num_layers=2,
        hidden=8,
        kv_hidden=4,
        intermediate=16,
        head_dim=4,
        vocab=5,
    )
    bundle = adapter.pack_official_weights(state.items(), padded_vocab=8, num_layers=2)
    shards = [shard_contract_bundle(bundle, rank, 2) for rank in range(2)]
    assert shards[0].wq.shape[-1] == bundle.wq.shape[-1] // 2
    assert shards[0].wo.shape[0] == bundle.wo.shape[0] // 2
    restored = gather_contract_bundle(shards)
    torch.testing.assert_close(restored.wq, bundle.wq)
    torch.testing.assert_close(restored.wo, bundle.wo)
    torch.testing.assert_close(restored.w_down, bundle.w_down)
    torch.testing.assert_close(restored.padded_embed_weight, bundle.padded_embed_weight)


def test_shard_stacked_row_parallel_is_per_layer_not_layer_range() -> None:
    # Two layers, 8 K-rows each. Layer 0 is 1s, layer 1 is 2s.
    layer0 = torch.ones(8, 4)
    layer1 = torch.full((8, 4), 2.0)
    stacked = torch.cat((layer0, layer1), dim=0)
    rank0 = shard_stacked_row_parallel(stacked, 0, 2, rows_per_layer=8)
    rank1 = shard_stacked_row_parallel(stacked, 1, 2, rows_per_layer=8)
    assert tuple(rank0.shape) == (8, 4)
    torch.testing.assert_close(rank0[:4], torch.ones(4, 4))
    torch.testing.assert_close(rank0[4:], torch.full((4, 4), 2.0))
    torch.testing.assert_close(rank1[:4], torch.ones(4, 4))
    torch.testing.assert_close(rank1[4:], torch.full((4, 4), 2.0))
    # Naive dim0 split would give rank 0 only layer 0 (all 1s).
    naive0 = shard_first_dim(stacked, 0, 2)
    assert float(naive0.max().item()) == 1.0
    assert float(rank0.max().item()) == 2.0


def test_select_compute_bundle_keeps_megatron_shards() -> None:
    state = _hf_state(
        num_layers=2,
        hidden=8,
        kv_hidden=4,
        intermediate=16,
        head_dim=4,
        vocab=5,
    )
    bundle = adapter.pack_official_weights(state.items(), padded_vocab=8, num_layers=2)
    compute = select_compute_bundle(bundle, rank=0, world=2)
    assert compute.wq.shape[-1] == bundle.wq.shape[-1] // 2
    assert compute.wo.shape[0] == bundle.wo.shape[0] // 2
    assert compute.w_down.shape[0] == bundle.w_down.shape[0] // 2
    # Compute bundle is the shard, not a reconstructed full-width copy.
    assert compute.wq.shape != bundle.wq.shape
    line = describe_tp_compute_args(compute)
    assert f"wq={tuple(int(d) for d in compute.wq.shape)}" in line
    assert compute.wq.shape[-1] == bundle.wq.shape[-1] // 2
    # Inverse gather still rebuilds the packed full tensors (tests only).
    other = select_compute_bundle(bundle, rank=1, world=2)
    restored = gather_contract_bundle([compute, other])
    torch.testing.assert_close(restored.wq, bundle.wq)
    torch.testing.assert_close(restored.wo, bundle.wo)
    wq0 = layer_stacked_view(compute.wq, 0, 2)
    wq1 = layer_stacked_view(compute.wq, 1, 2)
    assert wq0.shape[-1] == bundle.wq.shape[-1] // 2
    assert wq0.shape[0] == bundle.wq.shape[0] // 2
    assert wq1.shape == wq0.shape
    assert TP_BOUNDARY_O_PROJ == "o_proj"
    assert TP_BOUNDARY_DOWN_PROJ == "down_proj"


def test_shipped_tp2_entry_keeps_shards_and_boundary_collectives() -> None:
    root = Path(__file__).resolve().parents[3]
    example = (root / "examples" / "offline_pypto_qwen3_14b_tp2.py").read_text()
    model = (root / "vllm_ascend" / "models" / "pypto_qwen3.py").read_text()
    runner = (root / "vllm_ascend" / "models" / "pypto_qwen3_tp_runner.py").read_text()
    assert "select_compute_bundle" in example
    assert "gloo_gather_contract_bundle" not in example
    assert "real[0, :HIDDEN]" not in example
    assert "gloo_gather_contract_bundle" not in model
    assert "real[0, :HIDDEN]" not in model
    assert "TP_BOUNDARY_O_PROJ" in runner
    assert "TP_BOUNDARY_DOWN_PROJ" in runner


def test_tp_allreduce_slot_plan_matches_shmem_carve() -> None:
    from pypto.runtime.shmem_gloo import COMM_CONTEXT_SIZE, align_up, carve_window_layout

    names, nbytes = tp_allreduce_slot_plan(adapter.HIDDEN, torch.float32)
    assert names == ("data_buf", "out_buf")
    assert nbytes[0] == adapter.HIDDEN * 4
    offsets, window_bytes = carve_window_layout((COMM_CONTEXT_SIZE, *nbytes))
    assert offsets[0] == 0
    assert window_bytes == align_up(COMM_CONTEXT_SIZE) + align_up(nbytes[0]) + align_up(nbytes[1])
    assert tp.COMM_MARKER.startswith("PYPTO_QWEN3_COMM gloo+shmem+pypto")
