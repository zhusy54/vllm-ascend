#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""CPU tests for the shipped PyPTO Qwen3-14B vLLM adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models import pypto_qwen3_adapter as adapter


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


def test_is_pypto_qwen3_architecture_reads_hf_config() -> None:
    pypto = SimpleNamespace(hf_config=SimpleNamespace(architectures=["PyptoQwen3ForCausalLM"]))
    vanilla = SimpleNamespace(hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]))
    empty = SimpleNamespace(hf_config=SimpleNamespace(architectures=[]))
    assert adapter.is_pypto_qwen3_architecture(pypto) is True
    assert adapter.is_pypto_qwen3_architecture(vanilla) is False
    assert adapter.is_pypto_qwen3_architecture(empty) is False


def test_load_hf_state_from_dir_reads_safetensors_files(tmp_path) -> None:
    from safetensors.torch import save_file

    first = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    second = torch.ones(2, 2)
    save_file({"model.embed_tokens.weight": first}, tmp_path / "a.safetensors")
    save_file({"model.norm.weight": second}, tmp_path / "b.safetensors")

    state = adapter.load_hf_state_from_dir(tmp_path)

    assert set(state) == {"model.embed_tokens.weight", "model.norm.weight"}
    torch.testing.assert_close(state["model.embed_tokens.weight"], first)
    torch.testing.assert_close(state["model.norm.weight"], second)


def test_collect_hf_state_dict_owns_cpu_copies() -> None:
    staging = torch.ones(2, 3)
    state = adapter.collect_hf_state_dict([("w", staging)])
    staging.zero_()

    assert state["w"].device.type == "cpu"
    assert float(state["w"].sum().item()) == 6.0


def test_flatten_block_table_round_trips_vllm_layout() -> None:
    batch, max_blocks = 3, 5
    block_table = torch.arange(batch * max_blocks, dtype=torch.int64).reshape(batch, max_blocks)
    flat = adapter.flatten_block_table(block_table)

    assert flat.dtype == torch.int32
    assert tuple(flat.shape) == (batch * max_blocks,)
    assert adapter.block_table_stride(block_table, batch) == max_blocks
    torch.testing.assert_close(flat.reshape(batch, max_blocks), block_table.to(torch.int32))


def test_normalize_slot_mapping_keeps_page_offset_formula() -> None:
    page_size = adapter.PAGE_SIZE
    pages = torch.tensor([0, 3, 7], dtype=torch.int64)
    offsets = torch.tensor([0, 4, page_size - 1], dtype=torch.int64)
    slot_mapping = pages * page_size + offsets

    normalized = adapter.normalize_slot_mapping(slot_mapping)

    assert normalized.dtype == torch.int32
    torch.testing.assert_close(normalized // page_size, pages.to(torch.int32))
    torch.testing.assert_close(normalized % page_size, offsets.to(torch.int32))


def test_prefill_chunk_meta_matches_query_start_loc() -> None:
    query_start_loc = torch.tensor([0, 4, 4, 11], dtype=torch.int64)
    chunk_lens, chunk_offsets = adapter.prefill_chunk_meta(query_start_loc)

    torch.testing.assert_close(chunk_offsets, query_start_loc[:-1].to(torch.int32))
    torch.testing.assert_close(
        chunk_lens,
        (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int32),
    )
    assert int(chunk_lens.sum().item()) == int(query_start_loc[-1].item())


def test_split_kv_pair_matches_ascend_allocate_layout() -> None:
    num_pages = 3
    key = torch.randn(num_pages, adapter.PAGE_SIZE, adapter.NUM_KV_HEADS, adapter.HEAD_DIM)
    value = torch.randn_like(key)
    flat_key, flat_value = adapter.vllm_layer_kv_views((key, value))
    assert tuple(flat_key.shape) == (
        num_pages * adapter.PAGE_SIZE * adapter.NUM_KV_HEADS,
        adapter.HEAD_DIM,
    )
    marker = 4.5
    flat_key[0, 0] = marker
    assert float(key[0, 0, 0, 0].item()) == marker
    torch.testing.assert_close(flat_value.reshape_as(value), value)


def test_shared_vllm_kv_view_writes_land_in_vllm_pages() -> None:
    num_layers, num_pages = 3, 2
    stacked, layers = adapter.allocate_shared_vllm_kv(
        num_layers=num_layers,
        num_pages=num_pages,
        dtype=torch.float32,
        device="cpu",
    )
    key, value, shared = adapter.stack_vllm_kv_as_contract(layers)
    assert shared
    marker = 3.25
    key[0, 0] = marker
    value[-1, -1] = marker

    viewed_key, viewed_value = adapter.contract_kv_from_stacked(stacked)
    assert key.data_ptr() == viewed_key.data_ptr()
    assert float(layers[0][0, 0, 0, 0, 0].item()) == marker
    assert float(viewed_value[-1, -1].item()) == marker
    assert tuple(key.shape) == (
        num_layers * num_pages * adapter.PAGE_SIZE * adapter.NUM_KV_HEADS,
        adapter.HEAD_DIM,
    )


def test_stack_kv_pairs_does_not_require_5d() -> None:
    layers = [
        (
            torch.zeros(2, adapter.PAGE_SIZE, adapter.NUM_KV_HEADS, adapter.HEAD_DIM),
            torch.zeros(2, adapter.PAGE_SIZE, adapter.NUM_KV_HEADS, adapter.HEAD_DIM),
        )
        for _ in range(2)
    ]
    key, value, shared = adapter.stack_vllm_kv_as_contract(layers)
    assert shared is False
    assert key.shape[0] == 2 * 2 * adapter.PAGE_SIZE * adapter.NUM_KV_HEADS
    assert value.shape == key.shape


def test_compact_kv_only_copies_referenced_pages() -> None:
    page_size = adapter.PAGE_SIZE
    num_pages, kv_heads, head_dim = 4, 2, 4
    layers = []
    for layer_idx in range(2):
        key = torch.arange(num_pages * page_size * kv_heads * head_dim, dtype=torch.float32)
        key = key.reshape(num_pages, page_size, kv_heads, head_dim) + layer_idx * 1000
        value = key + 0.5
        layers.append((key.clone(), value.clone()))
    block_table = torch.tensor([[2, 0, 0, 0]], dtype=torch.int32)
    slot_mapping = torch.tensor([2 * page_size + 3], dtype=torch.int32)
    seq_lens = torch.tensor([page_size + 3], dtype=torch.int32)

    key, value, compact_table, compact_slots, phys = adapter.compact_vllm_kv_for_contract(
        layers, block_table, slot_mapping, seq_lens, page_size=page_size
    )
    assert phys.tolist() == [0, 2]
    assert compact_table.reshape(-1)[:2].tolist() == [1, 0]
    assert int(compact_slots[0].item()) == page_size + 3
    rows_per_page = page_size * kv_heads
    # Compact page 1 is physical page 2 of layer 0.
    key[rows_per_page : 2 * rows_per_page] = 7
    adapter.scatter_contract_kv_to_vllm(key, value, layers, phys, page_size=page_size)
    assert float(layers[0][0][2, 0, 0, 0].item()) == 7.0
    assert float(layers[0][0][1, 0, 0, 0].item()) != 7.0
    # Last page only has 3 live tokens; the unused tail must stay zero.
    tail = key[3 * kv_heads : rows_per_page]
    assert float(tail.abs().sum().item()) == 0.0


def test_compact_kv_drops_nan_page_tails() -> None:
    page_size, kv_heads, head_dim = adapter.PAGE_SIZE, 2, 4
    key = torch.zeros(1, page_size, kv_heads, head_dim)
    value = torch.zeros_like(key)
    key[:, 22:] = float("nan")
    value[:, 22:] = float("nan")
    key[:, :22] = 1.25
    value[:, :22] = 0.5
    layers = [(key, value)]
    block_table = torch.tensor([[0]], dtype=torch.int32)
    slot_mapping = torch.arange(22, dtype=torch.int32)
    seq_lens = torch.tensor([22], dtype=torch.int32)

    packed_k, packed_v, _, _, _ = adapter.compact_vllm_kv_for_contract(
        layers, block_table, slot_mapping, seq_lens, page_size=page_size
    )
    rows_per_page = page_size * kv_heads
    live = packed_k[: 22 * kv_heads]
    tail = packed_k[22 * kv_heads : rows_per_page]
    assert bool(torch.isfinite(packed_k).all())
    assert bool(torch.isfinite(packed_v).all())
    assert float(live[0, 0].item()) == 1.25
    assert float(tail.abs().sum().item()) == 0.0


def test_compact_skips_current_chunk_slots() -> None:
    page_size, kv_heads, head_dim = adapter.PAGE_SIZE, 2, 4
    key = torch.full((1, page_size, kv_heads, head_dim), float("nan"))
    value = torch.full_like(key, float("nan"))
    layers = [(key, value)]
    block_table = torch.tensor([[0]], dtype=torch.int32)
    slot_mapping = torch.arange(22, dtype=torch.int32)
    seq_lens = torch.tensor([22], dtype=torch.int32)
    chunk_lens = torch.tensor([22], dtype=torch.int32)

    packed_k, packed_v, _, _, _ = adapter.compact_vllm_kv_for_contract(
        layers,
        block_table,
        slot_mapping,
        seq_lens,
        page_size=page_size,
        chunk_lens=chunk_lens,
    )
    assert bool(torch.isfinite(packed_k).all())
    assert bool(torch.isfinite(packed_v).all())
    assert float(packed_k.abs().sum().item()) == 0.0


def test_copy_back_restores_separate_vllm_layers() -> None:
    layers = [
        torch.zeros(2, 2, adapter.PAGE_SIZE, adapter.NUM_KV_HEADS, adapter.HEAD_DIM)
        for _ in range(2)
    ]
    key, value, shared = adapter.stack_vllm_kv_as_contract(layers)
    assert shared is False
    key.fill_(1.0)
    value.fill_(2.0)
    adapter.copy_contract_kv_to_vllm(key, value, layers)

    for layer_kv in layers:
        layer_key, layer_value = adapter.vllm_layer_kv_views(layer_kv)
        assert torch.equal(layer_key, torch.ones_like(layer_key))
        assert torch.equal(layer_value, torch.full_like(layer_value, 2.0))


def test_pack_official_weights_uses_prepare_weights_shapes() -> None:
    num_layers, hidden, head_dim = 2, 8, 4
    kv_hidden, intermediate, vocab, padded = 4, 16, 5, 8
    state = _hf_state(
        num_layers=num_layers,
        hidden=hidden,
        kv_hidden=kv_hidden,
        intermediate=intermediate,
        head_dim=head_dim,
        vocab=vocab,
    )
    bundle = adapter.pack_official_weights(
        state.items(),
        padded_vocab=padded,
        num_layers=num_layers,
    )

    assert tuple(bundle.input_rms_weight.shape) == (num_layers, hidden)
    assert tuple(bundle.wq.shape) == (num_layers * hidden, hidden)
    assert tuple(bundle.wk.shape) == (num_layers * hidden, kv_hidden)
    assert tuple(bundle.w_gate.shape) == (num_layers * hidden, intermediate)
    assert tuple(bundle.w_down.shape) == (num_layers * intermediate, hidden)
    assert tuple(bundle.padded_lm_head_weight.shape) == (padded, hidden)
    assert tuple(bundle.padded_embed_weight.shape) == (padded, hidden)
    assert bundle.padded_lm_head_weight.dtype == torch.bfloat16
    # Padding rows reuse the first LM-head row so padded logits stay finite.
    torch.testing.assert_close(
        bundle.padded_lm_head_weight[vocab:],
        bundle.padded_lm_head_weight[:1].expand(padded - vocab, -1),
    )
    # Kernel layout transposes HF [out, in] weights.
    torch.testing.assert_close(
        bundle.wq[:hidden],
        state["model.layers.0.self_attn.q_proj.weight"].transpose(0, 1).to(torch.bfloat16),
    )


def test_build_prefill_and_decode_args_follow_live_inputs() -> None:
    batch, tokens, max_blocks = 2, 6, 3
    input_ids = torch.arange(tokens, dtype=torch.int64)
    seq_lens = torch.tensor([4, 2], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 4, 6], dtype=torch.int64)
    block_table = torch.arange(batch * max_blocks, dtype=torch.int32).reshape(batch, max_blocks)
    slot_mapping = torch.arange(tokens, dtype=torch.int64) + 10
    k_cache = torch.zeros(4, adapter.HEAD_DIM, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    weights = PyptoTinyWeights(hidden=4, vocab=8)
    rope_cos, rope_sin = adapter.build_rope_tables(max_seq=16, head_dim=adapter.HEAD_DIM)
    logits = torch.zeros(batch, weights.bundle.padded_lm_head_weight.shape[0])

    prefill_args = adapter.build_prefill_kernel_args(
        input_ids=input_ids,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        block_table=block_table,
        slot_mapping=slot_mapping,
        k_cache=k_cache,
        v_cache=v_cache,
        weights=weights.bundle,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        logits=logits,
    )
    chunk_lens, chunk_offsets = adapter.prefill_chunk_meta(query_start_loc)
    assert len(prefill_args) == 25
    assert prefill_args[0].dtype == torch.int32
    torch.testing.assert_close(prefill_args[0], input_ids.to(torch.int32))
    torch.testing.assert_close(prefill_args[1], seq_lens.to(torch.int32))
    torch.testing.assert_close(prefill_args[2], chunk_lens)
    torch.testing.assert_close(prefill_args[3], chunk_offsets)
    torch.testing.assert_close(prefill_args[12], adapter.flatten_block_table(block_table))
    torch.testing.assert_close(prefill_args[13], adapter.normalize_slot_mapping(slot_mapping))
    assert prefill_args[14] is k_cache and prefill_args[15] is v_cache
    assert prefill_args[-1] is logits
    assert prefill_args[-2] is weights.bundle.padded_embed_weight

    token_ids = torch.tensor([9, 11], dtype=torch.int64)
    decode_slot = torch.tensor([16, 17], dtype=torch.int64)
    sampled_out = torch.zeros(batch, adapter.SAMPLED_IDS_PAD, dtype=torch.int32)
    next_hidden = torch.zeros(batch, 4, dtype=torch.bfloat16)
    decode_args = adapter.build_decode_kernel_args(
        token_ids=token_ids,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=decode_slot,
        k_cache=k_cache,
        v_cache=v_cache,
        weights=weights.bundle,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        logits=logits,
        sampled_ids_out=sampled_out,
        next_hidden=next_hidden,
    )
    assert len(decode_args) == 25
    torch.testing.assert_close(decode_args[6], seq_lens.to(torch.int32))
    torch.testing.assert_close(decode_args[7], adapter.flatten_block_table(block_table))
    torch.testing.assert_close(decode_args[8], adapter.normalize_slot_mapping(decode_slot))
    packed = adapter.pack_sampled_ids(token_ids)
    torch.testing.assert_close(decode_args[-3], packed)
    assert decode_args[-2] is sampled_out
    assert decode_args[-1] is next_hidden
    sliced = adapter.slice_real_vocab_logits(torch.randn(1, adapter.PADDED_VOCAB))
    assert sliced.shape[-1] == adapter.REAL_VOCAB


class PyptoTinyWeights:
    def __init__(self, hidden: int, vocab: int) -> None:
        self.bundle = adapter.PyptoQwen3WeightBundle(
            input_rms_weight=torch.ones(1, hidden),
            wq=torch.zeros(hidden, hidden, dtype=torch.bfloat16),
            wk=torch.zeros(hidden, hidden, dtype=torch.bfloat16),
            wv=torch.zeros(hidden, hidden, dtype=torch.bfloat16),
            q_norm_weight=torch.ones(1, adapter.HEAD_DIM),
            k_norm_weight=torch.ones(1, adapter.HEAD_DIM),
            wo=torch.zeros(hidden, hidden, dtype=torch.bfloat16),
            w_gate=torch.zeros(hidden, hidden, dtype=torch.bfloat16),
            w_up=torch.zeros(hidden, hidden, dtype=torch.bfloat16),
            w_down=torch.zeros(hidden, hidden, dtype=torch.bfloat16),
            post_rms_weight=torch.ones(1, hidden),
            final_norm_weight=torch.ones(1, hidden),
            padded_lm_head_weight=torch.zeros(vocab, hidden, dtype=torch.bfloat16),
            padded_embed_weight=torch.zeros(vocab, hidden, dtype=torch.bfloat16),
        )


def test_build_runtime_model_from_hf_rejects_missing_layer() -> None:
    state = _hf_state(num_layers=1, hidden=4, kv_hidden=2, intermediate=8, head_dim=2, vocab=3)
    del state["model.layers.0.self_attn.q_proj.weight"]
    with pytest.raises(KeyError, match="self_attn.q_proj.weight"):
        adapter.build_runtime_model_from_hf(state, num_layers=1)


def test_runtime_model_fields_come_from_the_input_state() -> None:
    state = _hf_state(num_layers=1, hidden=4, kv_hidden=2, intermediate=8, head_dim=2, vocab=3)
    model = adapter.build_runtime_model_from_hf(state, num_layers=1)
    assert isinstance(model, SimpleNamespace)
    assert model.embed_tokens is state["model.embed_tokens.weight"]
    assert model.layers[0].wq is state["model.layers.0.self_attn.q_proj.weight"]
    assert model.final_norm_weight is state["model.norm.weight"]


def test_pin_weight_bundle_dtypes_follows_contract() -> None:
    dirty = adapter.PyptoQwen3WeightBundle(
        input_rms_weight=torch.ones(1, 4, dtype=torch.bfloat16),
        wq=torch.zeros(4, 4, dtype=torch.float32),
        wk=torch.zeros(4, 2, dtype=torch.float32),
        wv=torch.zeros(4, 2, dtype=torch.float32),
        q_norm_weight=torch.ones(1, adapter.HEAD_DIM, dtype=torch.bfloat16),
        k_norm_weight=torch.ones(1, adapter.HEAD_DIM, dtype=torch.bfloat16),
        wo=torch.zeros(4, 4, dtype=torch.float32),
        w_gate=torch.zeros(4, 8, dtype=torch.float32),
        w_up=torch.zeros(4, 8, dtype=torch.float32),
        w_down=torch.zeros(8, 4, dtype=torch.float32),
        post_rms_weight=torch.ones(1, 4, dtype=torch.bfloat16),
        final_norm_weight=torch.ones(1, 4, dtype=torch.bfloat16),
        padded_lm_head_weight=torch.zeros(8, 4, dtype=torch.float32),
        padded_embed_weight=torch.zeros(8, 4, dtype=torch.float32),
    )
    pinned = adapter.pin_weight_bundle_dtypes(dirty)
    assert pinned.input_rms_weight.dtype == torch.float32
    assert pinned.q_norm_weight.dtype == torch.float32
    assert pinned.final_norm_weight.dtype == torch.float32
    assert pinned.wq.dtype == torch.bfloat16
    assert pinned.padded_embed_weight.dtype == torch.bfloat16
    torch.testing.assert_close(pinned.input_rms_weight, torch.ones(1, 4))


def test_materialize_keeps_seq_lens_as_int32_batch_vector() -> None:
    weights = torch.zeros(2, 2)
    seq_lens = torch.tensor([22], dtype=torch.int32)
    chunk_lens = torch.tensor([22], dtype=torch.int32)
    live = adapter.materialize_npu_args(
        (weights, seq_lens, chunk_lens),
        device=torch.device("cpu"),
    )
    assert live[1].dtype == torch.int32
    assert tuple(live[1].shape) == (1,)
    assert live[2].dtype == torch.int32
    assert tuple(live[2].shape) == (1,)
    torch.testing.assert_close(live[1], seq_lens)
    assert live[1].is_contiguous()


def test_wrap_torch_npu_ptr_requires_contiguous_owner() -> None:
    cpu = torch.arange(4, dtype=torch.int32)
    wrapped = adapter.wrap_torch_npu_ptr(cpu)
    assert int(wrapped.data_ptr()) == int(cpu.data_ptr())
    with pytest.raises(ValueError, match="contiguous"):
        adapter.wrap_torch_npu_ptr(cpu.as_strided((2,), (2,)))


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="torch_npu required to move index tensors onto device pointers",
)
def test_materialize_moves_cpu_seq_lens_onto_npu_with_weights() -> None:
    weights = torch.zeros(2, 2, device="npu", dtype=torch.bfloat16)
    seq_lens = torch.tensor([22], dtype=torch.int32)
    chunk_lens = torch.tensor([22], dtype=torch.int32)
    live = adapter.materialize_npu_args((weights, seq_lens, chunk_lens))
    assert live[0].device.type == "npu"
    assert live[1].device.type == "npu"
    assert live[2].device.type == "npu"
    assert live[1].dtype == torch.int32
    assert tuple(live[1].shape) == (1,)
    assert int(live[1].item()) == 22
    wrapped = adapter.wrap_torch_npu_ptr(live[1])
    # DeviceTensor.data_ptr is the raw address, not a method.
    assert int(wrapped.data_ptr) == int(live[1].data_ptr())
    assert wrapped.shape == tuple(int(dim) for dim in live[1].shape)
    assert wrapped.dtype == live[1].dtype
