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
"""Map vLLM V1 Qwen3-14B tensors onto the pypto-lib contract ABI.

KV pages stay allocated and indexed by vllm-ascend. This module only
reshapes / packs those tensors for ``qwen3_14b.prefill_fwd`` and
``qwen3_14b.decode_fwd``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

# Contract ABI constants (must match pypto-lib/models/qwen3_14b/constants.py).
PAGE_SIZE = 128
NUM_LAYERS = 40
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = 5120
INTERMEDIATE = 17408
REAL_VOCAB = 151936
PADDED_VOCAB = 152064
SAMPLED_IDS_PAD = 8
MAX_SEQ = 4096
ROPE_THETA = 1_000_000.0
STAGE_PREFILL = "qwen3_14b.prefill_fwd"
STAGE_DECODE = "qwen3_14b.decode_fwd"

_HF_LAYER_SUFFIXES = (
    ("input_layernorm.weight", "input_rms_weight"),
    ("self_attn.q_proj.weight", "wq"),
    ("self_attn.k_proj.weight", "wk"),
    ("self_attn.v_proj.weight", "wv"),
    ("self_attn.q_norm.weight", "q_norm_weight"),
    ("self_attn.k_norm.weight", "k_norm_weight"),
    ("self_attn.o_proj.weight", "wo"),
    ("post_attention_layernorm.weight", "post_rms_weight"),
    ("mlp.gate_proj.weight", "w_gate"),
    ("mlp.up_proj.weight", "w_up"),
    ("mlp.down_proj.weight", "w_down"),
)


@dataclass(frozen=True)
class PyptoQwen3WeightBundle:
    """Kernel-ready stacked weights plus padded embed / LM head."""

    input_rms_weight: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    wo: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor
    post_rms_weight: torch.Tensor
    final_norm_weight: torch.Tensor
    padded_lm_head_weight: torch.Tensor
    padded_embed_weight: torch.Tensor


def default_pypto_lib_root() -> Path:
    env = os.environ.get("PYPTO_LIB_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "pypto-lib"


def ensure_pypto_lib_on_path(root: Path | None = None) -> Path:
    resolved = (root or default_pypto_lib_root()).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"pypto-lib root not found: {resolved}")
    path = str(resolved)
    if path not in sys.path:
        sys.path.insert(0, path)
    return resolved


def collect_hf_state_dict(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Materialize a HF-style name -> tensor map from a vLLM weight iterator."""
    state: dict[str, torch.Tensor] = {}
    for name, tensor in weights:
        state[name] = tensor
    if not state:
        raise ValueError("HF weight iterator was empty")
    return state


def build_runtime_model_from_hf(
    state: dict[str, torch.Tensor],
    *,
    num_layers: int = NUM_LAYERS,
) -> SimpleNamespace:
    """Group HF Qwen3 tensors into the namespace ``prepare_qwen3_weights`` expects."""
    missing: list[str] = []
    layers: list[SimpleNamespace] = []
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}."
        fields: dict[str, torch.Tensor] = {}
        for hf_suffix, attr in _HF_LAYER_SUFFIXES:
            key = prefix + hf_suffix
            if key not in state:
                missing.append(key)
                continue
            fields[attr] = state[key]
        layers.append(SimpleNamespace(**fields))

    embed_key = "model.embed_tokens.weight"
    norm_key = "model.norm.weight"
    if embed_key not in state:
        missing.append(embed_key)
    if norm_key not in state:
        missing.append(norm_key)
    if missing:
        raise KeyError("missing HF tensors required by the Qwen3-14B contract: " + ", ".join(missing))

    lm_head = state.get("lm_head.weight", state[embed_key])
    return SimpleNamespace(
        embed_tokens=state[embed_key],
        lm_head=lm_head,
        final_norm_weight=state[norm_key],
        layers=tuple(layers),
    )


def pack_official_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    padded_vocab: int = PADDED_VOCAB,
    pypto_lib_root: Path | None = None,
    tensor_exporter: Any | None = None,
    num_layers: int = NUM_LAYERS,
) -> PyptoQwen3WeightBundle:
    """Load official HF shards into the contract layout via ``prepare_qwen3_weights``."""
    state = collect_hf_state_dict(weights)
    runtime_model = build_runtime_model_from_hf(state, num_layers=num_layers)
    prepare = _load_prepare_qwen3_weights(pypto_lib_root)
    exporter = tensor_exporter or (lambda tensor: tensor.contiguous())
    prepared = prepare(
        runtime_model,
        exporter,
        padded_vocab=padded_vocab,
        release_layers=False,
    )
    decode = prepared.decode_weights
    return PyptoQwen3WeightBundle(
        input_rms_weight=decode["decode_input_rms_weight"],
        wq=decode["decode_wq"],
        wk=decode["decode_wk"],
        wv=decode["decode_wv"],
        q_norm_weight=decode["decode_q_norm_weight"],
        k_norm_weight=decode["decode_k_norm_weight"],
        wo=decode["decode_wo"],
        w_gate=decode["decode_w_gate"],
        w_up=decode["decode_w_up"],
        w_down=decode["decode_w_down"],
        post_rms_weight=decode["decode_post_rms_weight"],
        final_norm_weight=prepared.final_norm_weight,
        padded_lm_head_weight=prepared.padded_lm_head_weight,
        padded_embed_weight=prepared.padded_embed_weight,
    )


def flatten_block_table(block_table: torch.Tensor) -> torch.Tensor:
    """Flatten vLLM ``[batch, max_blocks]`` (or already-flat) to contract int32."""
    table = block_table.to(dtype=torch.int32).contiguous()
    if table.ndim == 1:
        return table
    if table.ndim != 2:
        raise ValueError(f"block_table must be rank 1 or 2, got shape {tuple(table.shape)}")
    return table.reshape(-1).contiguous()


def block_table_stride(block_table: torch.Tensor, batch: int) -> int:
    flat = flatten_block_table(block_table)
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if flat.numel() % batch != 0:
        raise ValueError(
            f"flat block_table length {flat.numel()} is not divisible by batch {batch}"
        )
    return flat.numel() // batch


def normalize_slot_mapping(slot_mapping: torch.Tensor) -> torch.Tensor:
    """Contract slot_mapping is int32; values stay ``page * PAGE_SIZE + offset``."""
    return slot_mapping.to(dtype=torch.int32).reshape(-1).contiguous()


def prefill_chunk_meta(
    query_start_loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``chunk_lens`` / ``chunk_offsets`` from vLLM packed ``query_start_loc``."""
    qsl = query_start_loc.to(dtype=torch.int32).reshape(-1)
    if qsl.numel() < 2:
        raise ValueError("query_start_loc must contain at least two entries")
    chunk_lens = (qsl[1:] - qsl[:-1]).contiguous()
    chunk_offsets = qsl[:-1].contiguous()
    return chunk_lens, chunk_offsets


def pack_sampled_ids(
    token_ids: torch.Tensor,
    *,
    sampled_ids_pad: int = SAMPLED_IDS_PAD,
) -> torch.Tensor:
    """Pack per-row token ids into the decode ``[batch, SAMPLED_IDS_PAD]`` buffer."""
    ids = token_ids.to(dtype=torch.int32).reshape(-1)
    packed = token_ids.new_zeros((ids.numel(), sampled_ids_pad), dtype=torch.int32)
    packed[:, 0] = ids
    return packed


def build_rope_tables(
    *,
    max_seq: int = MAX_SEQ,
    head_dim: int = HEAD_DIM,
    theta: float = ROPE_THETA,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Neox-style half-half cos/sin tables used by the Qwen3-14B kernels."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    positions = torch.arange(max_seq, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def split_vllm_layer_kv(layer_kv: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-layer ``(k, v)`` each ``(pages, page_size, kv_heads, dim)``.

    vllm-ascend binds either a stacked ``(2, P, S, H, D)`` tensor or a
    ``(k, v)`` pair of 4-D pages (the allocate path used on this machine).
    """
    if isinstance(layer_kv, (list, tuple)):
        if (
            len(layer_kv) == 2
            and all(isinstance(part, torch.Tensor) and part.ndim == 4 for part in layer_kv)
        ):
            return layer_kv[0], layer_kv[1]
        if len(layer_kv) >= 1:
            return split_vllm_layer_kv(layer_kv[0])
    if not isinstance(layer_kv, torch.Tensor):
        raise ValueError(f"unrecognized vLLM layer KV type: {type(layer_kv)!r}")
    if layer_kv.ndim == 5 and layer_kv.shape[0] == 2:
        return layer_kv[0], layer_kv[1]
    raise ValueError(
        "expected vLLM layer KV as (2, pages, page_size, kv_heads, dim) or "
        f"(k, v) 4-D pages, got shape {tuple(layer_kv.shape)}"
    )


def flatten_paged_kv(paged: torch.Tensor) -> torch.Tensor:
    """View ``(pages, page_size, kv_heads, dim)`` as contract ``(P*S*H, D)``."""
    if paged.ndim != 4:
        raise ValueError(f"paged KV must be rank 4, got {tuple(paged.shape)}")
    num_pages, page_size, num_kv_heads, head_dim = paged.shape
    if page_size != PAGE_SIZE:
        raise ValueError(f"vLLM page_size must be {PAGE_SIZE}, got {page_size}")
    return paged.reshape(num_pages * page_size * num_kv_heads, head_dim)


def vllm_layer_kv_views(layer_kv: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """View one vLLM layer cache as contract K/V rows."""
    key, value = split_vllm_layer_kv(layer_kv)
    return flatten_paged_kv(key), flatten_paged_kv(value)


def allocate_shared_vllm_kv(
    *,
    num_layers: int,
    num_pages: int,
    page_size: int = PAGE_SIZE,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One stacked allocation plus per-layer ``(2, P, S, H, D)`` views.

    The stacked buffer is what the fused host reads. Layer views are what
    vLLM's block manager indexes. They share storage.
    """
    stacked = torch.zeros(
        (2, num_layers, num_pages, page_size, num_kv_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    layers = [stacked[:, layer] for layer in range(num_layers)]
    return stacked, layers


def contract_kv_from_stacked(stacked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape ``[2, L, P, S, H, D]`` to contract ``[L*P*S*H, D]`` views."""
    if stacked.ndim != 6 or stacked.shape[0] != 2:
        raise ValueError(f"stacked KV must be (2, L, P, S, H, D), got {tuple(stacked.shape)}")
    num_layers, num_pages, page_size, num_kv_heads, head_dim = stacked.shape[1:]
    rows = num_layers * num_pages * page_size * num_kv_heads
    return stacked[0].reshape(rows, head_dim), stacked[1].reshape(rows, head_dim)


def referenced_page_ids(
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int = PAGE_SIZE,
) -> torch.Tensor:
    """Physical page ids touched by this step (vLLM BlockManager ids)."""
    table = block_table.to(dtype=torch.int64)
    seq = seq_lens.to(device=table.device, dtype=torch.int64)
    if table.ndim == 1:
        batch = int(seq.numel())
        if table.numel() % batch != 0:
            raise ValueError("flat block_table is not divisible by batch")
        table = table.reshape(batch, -1)
    blocks_needed = (seq.clamp(min=1) + page_size - 1) // page_size
    mask = torch.arange(table.shape[1], device=table.device).unsqueeze(0) < blocks_needed.unsqueeze(1)
    pages = table.masked_select(mask)
    slot_pages = slot_mapping.to(device=table.device, dtype=torch.int64).reshape(-1) // page_size
    pages = torch.cat((pages, slot_pages), dim=0)
    pages = pages[pages >= 0]
    unique_pages = torch.unique(pages, sorted=True)
    if unique_pages.numel() == 0:
        raise ValueError("no referenced KV pages")
    return unique_pages.to(dtype=torch.int64)


def compact_vllm_kv_for_contract(
    layer_kvs: Sequence[Any],
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int = PAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy only referenced vLLM pages into a layer-major contract buffer.

    Returns ``(k, v, compact_block_table, compact_slot_mapping, phys_pages)``.
    ``phys_pages[compact_id]`` is the original vLLM page id so the caller can
    scatter writes back. This does not allocate a second block pool.
    """
    phys_pages = referenced_page_ids(
        block_table, slot_mapping, seq_lens, page_size=page_size
    )
    page_to_compact = torch.full(
        (int(phys_pages.max().item()) + 1,),
        -1,
        dtype=torch.int64,
        device=phys_pages.device,
    )
    page_to_compact[phys_pages] = torch.arange(phys_pages.numel(), device=phys_pages.device)

    table = block_table.to(dtype=torch.int64)
    seq = seq_lens.to(device=table.device, dtype=torch.int64)
    if table.ndim == 1:
        batch = int(seq.numel())
        table = table.reshape(batch, -1)
    blocks_needed = (seq.clamp(min=1) + page_size - 1) // page_size
    valid = torch.arange(table.shape[1], device=table.device).unsqueeze(0) < blocks_needed.unsqueeze(1)
    compact_table = torch.zeros_like(table)
    compact_table[valid] = page_to_compact[table[valid]]
    slots = slot_mapping.to(device=phys_pages.device, dtype=torch.int64).reshape(-1)
    compact_slots = page_to_compact[slots // page_size]
    compact_slots = compact_slots * page_size + (slots % page_size)

    first_k, first_v = split_vllm_layer_kv(layer_kvs[0])
    head_dim = first_k.shape[-1]
    num_kv_heads = first_k.shape[-2]
    rows_per_page = page_size * num_kv_heads
    num_layers = len(layer_kvs)
    num_pages = int(phys_pages.numel())
    contract_rows = num_layers * num_pages * rows_per_page
    key = first_k.new_empty((contract_rows, head_dim))
    value = first_v.new_empty((contract_rows, head_dim))
    for layer_idx, layer_kv in enumerate(layer_kvs):
        layer_key, layer_value = split_vllm_layer_kv(layer_kv)
        dst0 = layer_idx * num_pages * rows_per_page
        gathered_k = layer_key.index_select(0, phys_pages).reshape(num_pages * rows_per_page, head_dim)
        gathered_v = layer_value.index_select(0, phys_pages).reshape(num_pages * rows_per_page, head_dim)
        key[dst0 : dst0 + gathered_k.shape[0]].copy_(gathered_k)
        value[dst0 : dst0 + gathered_v.shape[0]].copy_(gathered_v)
    return (
        key,
        value,
        compact_table.to(dtype=torch.int32).reshape(-1).contiguous(),
        compact_slots.to(dtype=torch.int32).contiguous(),
        phys_pages,
    )


def scatter_contract_kv_to_vllm(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_kvs: Sequence[Any],
    phys_pages: torch.Tensor,
    *,
    page_size: int = PAGE_SIZE,
) -> None:
    """Write compact contract pages back into the original vLLM page ids."""
    num_layers = len(layer_kvs)
    num_pages = int(phys_pages.numel())
    first_k, _ = split_vllm_layer_kv(layer_kvs[0])
    rows_per_page = page_size * first_k.shape[-2]
    head_dim = first_k.shape[-1]
    expected = num_layers * num_pages * rows_per_page
    if key.shape[0] != expected or value.shape[0] != expected:
        raise ValueError(f"contract KV rows {tuple(key.shape)} != {expected}")
    for layer_idx, layer_kv in enumerate(layer_kvs):
        layer_key, layer_value = split_vllm_layer_kv(layer_kv)
        src0 = layer_idx * num_pages * rows_per_page
        packed_k = key[src0 : src0 + num_pages * rows_per_page].reshape(
            num_pages, page_size, first_k.shape[-2], head_dim
        )
        packed_v = value[src0 : src0 + num_pages * rows_per_page].reshape(
            num_pages, page_size, first_k.shape[-2], head_dim
        )
        layer_key[phys_pages] = packed_k
        layer_value[phys_pages] = packed_v


def stack_vllm_kv_as_contract(
    layer_kvs: Sequence[Any],
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Stack per-layer vLLM KV into contract ``(k, v, shared_storage)``.

    ``shared_storage`` is True when the result aliases ``layer_kvs`` (writes
    land in the vLLM pages). False means the caller must copy back after the
    kernel mutates the stacked tensors.
    """
    if not layer_kvs:
        raise ValueError("layer_kvs is empty")
    stacked = _try_recover_stacked(layer_kvs)
    if stacked is not None:
        key, value = contract_kv_from_stacked(stacked)
        return key, value, True
    keys = [vllm_layer_kv_views(layer_kv)[0] for layer_kv in layer_kvs]
    values = [vllm_layer_kv_views(layer_kv)[1] for layer_kv in layer_kvs]
    return torch.cat(keys, dim=0), torch.cat(values, dim=0), False


def copy_contract_kv_to_vllm(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_kvs: Sequence[torch.Tensor],
) -> None:
    """Write stacked contract K/V back into per-layer vLLM pages."""
    offset = 0
    for layer_kv in layer_kvs:
        layer_key, layer_value = vllm_layer_kv_views(layer_kv)
        rows = layer_key.shape[0]
        layer_key.copy_(key[offset : offset + rows])
        layer_value.copy_(value[offset : offset + rows])
        offset += rows
    if offset != key.shape[0] or offset != value.shape[0]:
        raise ValueError(
            f"contract KV rows {key.shape[0]}/{value.shape[0]} != sum of layer rows {offset}"
        )


def build_prefill_kernel_args(
    *,
    input_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    weights: PyptoQwen3WeightBundle,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Args in actual ``prefill_fwd`` order (input_ids + embed, not the stale host)."""
    chunk_lens, chunk_offsets = prefill_chunk_meta(query_start_loc)
    return (
        input_ids.to(dtype=torch.int32).reshape(-1).contiguous(),
        seq_lens.to(dtype=torch.int32).reshape(-1).contiguous(),
        chunk_lens,
        chunk_offsets,
        weights.input_rms_weight,
        weights.wq,
        weights.wk,
        weights.wv,
        weights.q_norm_weight,
        weights.k_norm_weight,
        rope_cos,
        rope_sin,
        flatten_block_table(block_table),
        normalize_slot_mapping(slot_mapping),
        k_cache,
        v_cache,
        weights.wo,
        weights.post_rms_weight,
        weights.w_gate,
        weights.w_up,
        weights.w_down,
        weights.final_norm_weight,
        weights.padded_lm_head_weight,
        weights.padded_embed_weight,
        logits,
    )


def build_decode_kernel_args(
    *,
    token_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    weights: PyptoQwen3WeightBundle,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    logits: torch.Tensor,
    sampled_ids_out: torch.Tensor,
    next_hidden: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Args in actual ``decode_fwd`` order."""
    sampled_ids_in = pack_sampled_ids(token_ids)
    return (
        weights.input_rms_weight,
        weights.wq,
        weights.wk,
        weights.wv,
        weights.q_norm_weight,
        weights.k_norm_weight,
        seq_lens.to(dtype=torch.int32).reshape(-1).contiguous(),
        flatten_block_table(block_table),
        normalize_slot_mapping(slot_mapping),
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        weights.wo,
        weights.w_gate,
        weights.w_up,
        weights.w_down,
        weights.post_rms_weight,
        weights.final_norm_weight,
        weights.padded_lm_head_weight,
        logits,
        weights.padded_embed_weight,
        sampled_ids_in,
        sampled_ids_out,
        next_hidden,
    )


def invoke_pypto_kernel(kernel: Any, args: Sequence[torch.Tensor]) -> Any:
    """Specialize from torch tensors, then execute via DeviceTensor views."""
    compiled = kernel.compile(*args)
    return compiled(*wrap_tensors_for_pypto(args))


def wrap_tensors_for_pypto(tensors: Sequence[torch.Tensor]) -> tuple[Any, ...]:
    """Pack NPU tensors as DeviceTensor so pypto skips H2D/D2H copies.

    The pypto L2 runner rejects ``torch.Tensor`` on NPU (``expected CPU``).
    A ``DeviceTensor`` around the same ``data_ptr`` keeps the buffer
    caller-managed. CPU tensors are passed through.
    """
    from pypto.runtime.device_tensor import DeviceTensor

    wrapped: list[Any] = []
    for tensor in tensors:
        contig = tensor.contiguous()
        if contig.device.type == "cpu":
            wrapped.append(contig)
            continue
        wrapped.append(
            DeviceTensor(
                int(contig.data_ptr()),
                tuple(int(dim) for dim in contig.shape),
                contig.dtype,
            )
        )
    return tuple(wrapped)


def slice_real_vocab_logits(
    logits: torch.Tensor,
    *,
    real_vocab: int = REAL_VOCAB,
) -> torch.Tensor:
    """Drop LM-head padding before handing logits back to the vLLM sampler."""
    if logits.shape[-1] < real_vocab:
        raise ValueError(f"logits last dim {logits.shape[-1]} < real_vocab {real_vocab}")
    return logits[..., :real_vocab]


def _load_prepare_qwen3_weights(pypto_lib_root: Path | None) -> Any:
    root = ensure_pypto_lib_on_path(pypto_lib_root)
    variant_dir = root / "models" / "qwen3_14b"
    module_path = variant_dir / "weights.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location("_vllm_ascend_qwen3_14b_weights", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(variant_dir))
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(variant_dir))
        except ValueError:
            pass
    return module.prepare_qwen3_weights


def _try_recover_stacked(layer_kvs: Sequence[Any]) -> torch.Tensor | None:
    first = layer_kvs[0]
    if not isinstance(first, torch.Tensor):
        return None
    if first.ndim != 5 or first.shape[0] != 2 or first.storage_offset() != 0:
        return None
    num_layers = len(layer_kvs)
    storage_ptr = first.untyped_storage().data_ptr()
    for layer_kv in layer_kvs:
        if layer_kv.untyped_storage().data_ptr() != storage_ptr:
            return None
        if tuple(layer_kv.shape) != tuple(first.shape) or layer_kv.stride() != first.stride():
            return None
    if num_layers == 1:
        layer_stride = first.numel() // 2
    else:
        layer_stride = layer_kvs[1].storage_offset() - first.storage_offset()
        if layer_stride <= 0:
            return None
    expected = (2, num_layers, *first.shape[1:])
    strides = (
        first.stride(0),
        layer_stride,
        first.stride(1),
        first.stride(2),
        first.stride(3),
        first.stride(4),
    )
    try:
        viewed = torch.as_strided(first, size=expected, stride=strides)
    except RuntimeError:
        return None
    if viewed[:, 0].data_ptr() != first.data_ptr():
        return None
    if num_layers > 1 and viewed[:, 1].data_ptr() != layer_kvs[1].data_ptr():
        return None
    return viewed
