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
"""Single-card Qwen3-14B path that runs pypto-lib fused prefill/decode.

vllm-ascend still owns the paged KV allocator (``block_table`` /
``slot_mapping`` / per-layer ``(2, pages, 128, 8, 128)`` buffers). This
module only dispatches those tensors into ``qwen3_14b.prefill_fwd`` and
``qwen3_14b.decode_fwd``. Select it with
``hf_overrides={"architectures": ["PyptoQwen3ForCausalLM"]}``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.encoder_only_attention import Attention
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.pypto_qwen3_adapter import (
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    MAX_SEQ,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    PADDED_VOCAB,
    PAGE_SIZE,
    REAL_VOCAB,
    SAMPLED_IDS_PAD,
    STAGE_DECODE,
    STAGE_PREFILL,
    PyptoQwen3WeightBundle,
    build_decode_kernel_args,
    build_prefill_kernel_args,
    build_rope_tables,
    collect_hf_state_dict,
    compact_vllm_kv_for_contract,
    ensure_pypto_lib_on_path,
    load_hf_state_from_dir,
    pack_official_weights,
    pin_weight_bundle_dtypes,
    prefill_chunk_meta,
    scatter_contract_kv_to_vllm,
    slice_real_vocab_logits,
    invoke_pypto_kernel,
    PyptoChipSession,
)
from vllm_ascend.models.pypto_qwen3_tp import (
    COMM_MARKER,
    build_gloo_shmem_comm,
    gloo_gather_contract_bundle,
    shard_contract_bundle,
    tp_rank_and_world,
)


def _tp_cpu_group():
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group().cpu_group
    if group is None:
        import torch.distributed as dist

        group = dist.group.WORLD
    return group

logger = init_logger(__name__)


class PyptoQwen3ForCausalLM(nn.Module):
    """Fused pypto Qwen3-14B; vanilla ``Qwen3ForCausalLM`` stays the default."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        self.config = config
        self.vllm_config = vllm_config
        self._validate_single_card_shape(vllm_config)

        scale = HEAD_DIM**-0.5
        self.attn_layers = nn.ModuleList(
            [
                Attention(
                    NUM_HEADS,
                    HEAD_DIM,
                    scale,
                    num_kv_heads=NUM_KV_HEADS,
                    cache_config=cache_config,
                    prefix=f"{prefix}model.layers.{layer_idx}.self_attn.attn",
                )
                for layer_idx in range(NUM_LAYERS)
            ]
        )
        # Keep RoPE on CPU float32. Registering these as buffers lets
        # ``LLM(dtype=bfloat16)`` cast them; the fused host expects fp32 tables.
        rope_cos, rope_sin = build_rope_tables(max_seq=MAX_SEQ, head_dim=HEAD_DIM)
        self.rope_cos = rope_cos.cpu().float().contiguous()
        self.rope_sin = rope_sin.cpu().float().contiguous()
        self._bundle: PyptoQwen3WeightBundle | None = None
        self._last_logits: torch.Tensor | None = None
        self._kernels: dict[str, object] | None = None
        self._chip: PyptoChipSession | None = None
        self._tp_comm = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        consumed = collect_hf_state_dict(weights)
        model_path = getattr(self.vllm_config.model_config, "model", None)
        if model_path and Path(model_path).is_dir():
            state = load_hf_state_from_dir(model_path)
        else:
            state = consumed
        bundle = pin_weight_bundle_dtypes(
            pack_official_weights(state.items(), padded_vocab=PADDED_VOCAB)
        )
        tp_rank, tp_world = tp_rank_and_world()
        if tp_world > 1:
            shard = shard_contract_bundle(bundle, tp_rank, tp_world)
            print(
                f"PYPTO_QWEN3_TP_SHARD rank={tp_rank}/{tp_world} "
                f"wq={tuple(shard.wq.shape)} wo={tuple(shard.wo.shape)} "
                f"w_down={tuple(shard.w_down.shape)}",
                flush=True,
            )
            group = _tp_cpu_group()
            bundle = gloo_gather_contract_bundle(
                shard, rank=tp_rank, world=tp_world, group=group
            )
        # Keep contract weights as a Python object, not nn buffers, so
        # ``LLM(dtype=bfloat16)`` cannot recast RMS / RoPE to bf16.
        if torch.npu.is_available():
            bundle = PyptoQwen3WeightBundle(
                **{name: tensor.to(device="npu") for name, tensor in bundle.__dict__.items()}
            )
            self.rope_cos = self.rope_cos.float().contiguous().to(device="npu")
            self.rope_sin = self.rope_sin.float().contiguous().to(device="npu")
            # Create ChipWorker before vLLM sizes the KV pool. The fused host's
            # workspace is aclrtMalloc'd at init; if that happens after the
            #  KV reservation the two heaps fight and prefill writes NaNs.
            # Replay of the same args outside vLLM is finite (argmax 17).
            if self._chip is None:
                torch.npu.empty_cache()
                self._chip = PyptoChipSession()
                free, total = torch.npu.mem_get_info()
                print(
                    f"PYPTO_QWEN3_CHIP_INIT after_weights "
                    f"free={free/1024**3:.2f}GiB total={total/1024**3:.2f}GiB",
                    flush=True,
                )
        self._bundle = bundle
        weight_bytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in bundle.__dict__.values())
        print(
            f"PYPTO_QWEN3_WEIGHTS src={model_path} "
            f"dev={next(iter(bundle.__dict__.values())).device} "
            f"bytes={weight_bytes} wq={tuple(bundle.wq.shape)}/{bundle.wq.dtype} "
            f"wq_amax={float(bundle.wq.float().abs().max())} "
            f"wq_head={bundle.wq.reshape(-1)[:4].float().cpu().tolist()} "
            f"rms={bundle.input_rms_weight.dtype} rope={self.rope_cos.dtype}",
            flush=True,
        )
        logger.info(
            "Packed official Qwen3-14B weights on %s (%s bytes) wq_absmax=%s",
            next(iter(bundle.__dict__.values())).device,
            weight_bytes,
            float(bundle.wq.float().abs().max()),
        )
        return set(consumed) | set(state)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._bundle is None:
            raise RuntimeError("PyptoQwen3ForCausalLM.load_weights has not run")
        # vLLM only needs this for runner detection / dummy embeds. The fused
        # host gathers embeddings itself. Do the lookup on CPU so weights stay put.
        embed = self._bundle.padded_embed_weight
        ids = input_ids.to(device=embed.device)
        return torch.nn.functional.embedding(ids, embed).to(
            device=input_ids.device, dtype=torch.bfloat16
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions, intermediate_tensors, inputs_embeds
        if input_ids is None:
            raise RuntimeError("PyptoQwen3ForCausalLM requires input_ids (no inputs_embeds path)")
        if self._bundle is None:
            raise RuntimeError("PyptoQwen3ForCausalLM.load_weights has not run")

        dummy = input_ids.new_zeros((input_ids.shape[0], HIDDEN), dtype=torch.bfloat16)
        try:
            metadata = _unwrap_attn_metadata(get_forward_context().attn_metadata)
            layer_kvs = self._collect_layer_kvs()
        except RuntimeError:
            # profile_run / dummy_run happens before KV pages are bound.
            self._last_logits = dummy.new_zeros((1, REAL_VOCAB), dtype=torch.float32)
            return dummy

        num_tokens = int(getattr(metadata, "num_actual_tokens", input_ids.shape[0]))
        token_ids = input_ids[:num_tokens].reshape(-1)
        seq_lens = metadata.seq_lens
        if seq_lens is None:
            raise RuntimeError("attn metadata is missing seq_lens")
        seq_lens = seq_lens.to(dtype=torch.int32).reshape(-1)
        slot_mapping = metadata.slot_mapping[:num_tokens]
        block_table = metadata.block_tables
        if block_table is None:
            block_table = getattr(metadata, "block_table", None)
        if block_table is None:
            raise RuntimeError("attn metadata is missing block_tables")
        num_prefills = int(getattr(metadata, "num_prefills", 0) or 0)
        num_decodes = int(getattr(metadata, "num_decodes", 0) or 0)
        num_reqs = num_prefills + num_decodes
        if num_reqs > 0:
            seq_lens = seq_lens[:num_reqs]
            if block_table.ndim == 2:
                block_table = block_table[:num_reqs]
                # Drop BlockManager padding. A padded table of length 8 with a
                # 1-page compact cache makes the kernel think max_blocks=8.
                need = int(((int(seq_lens.max().item()) + PAGE_SIZE - 1) // PAGE_SIZE))
                need = max(need, 1)
                if block_table.shape[1] > need:
                    block_table = block_table[:, :need]
        batch = int(seq_lens.shape[0])
        query_start_loc = metadata.query_start_loc
        if query_start_loc is None:
            query_start_loc = token_ids.new_tensor([0, num_tokens], dtype=torch.int32)
        else:
            query_start_loc = query_start_loc[: batch + 1]
        chunk_lens, _chunk_offsets = prefill_chunk_meta(query_start_loc)

        k_cache, v_cache, compact_table, compact_slots, phys_pages = compact_vllm_kv_for_contract(
            layer_kvs,
            block_table,
            slot_mapping,
            seq_lens,
            chunk_lens=chunk_lens,
        )
        seq_lens = seq_lens.to(dtype=torch.int32).contiguous()
        token_ids = token_ids.to(dtype=torch.int32).contiguous()
        query_start_loc = query_start_loc.to(dtype=torch.int32).contiguous()
        compact_table = compact_table.contiguous()
        compact_slots = compact_slots.contiguous()
        logits = token_ids.new_zeros((batch, PADDED_VOCAB), dtype=torch.float32)
        print(
            f"PYPTO_QWEN3_COMPACT batch={batch} ntok={num_tokens} seq={seq_lens.tolist()} "
            f"chunk={chunk_lens.detach().cpu().tolist()} table={compact_table.tolist()[:8]} "
            f"slots={compact_slots[:8].tolist()} ids={token_ids[:8].tolist()} "
            f"kabs0={float(k_cache.float().abs().max())} "
            f"dev={k_cache.device} wq_dev={self._bundle.wq.device}",
            flush=True,
        )
        kernels = self._ensure_kernels()
        chip = None if os.environ.get("PYPTO_QWEN3_CPU_INVOKE") == "1" else self._ensure_chip()

        if num_prefills > 0 and num_decodes == 0:
            stage = STAGE_PREFILL
            args = build_prefill_kernel_args(
                input_ids=token_ids,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                block_table=compact_table,
                slot_mapping=compact_slots,
                k_cache=k_cache,
                v_cache=v_cache,
                weights=self._bundle,
                rope_cos=self.rope_cos,
                rope_sin=self.rope_sin,
                logits=logits,
            )
            print(f"PYPTO_QWEN3_STAGE {stage}", flush=True)
            invoke_pypto_kernel(kernels["prefill_fwd"], args, session=chip)
        elif num_decodes > 0 and num_prefills == 0:
            stage = STAGE_DECODE
            sampled_ids_out = token_ids.new_zeros((batch, SAMPLED_IDS_PAD), dtype=torch.int32)
            next_hidden = token_ids.new_zeros((batch, HIDDEN), dtype=torch.bfloat16)
            args = build_decode_kernel_args(
                token_ids=token_ids,
                seq_lens=seq_lens,
                block_table=compact_table,
                slot_mapping=compact_slots,
                k_cache=k_cache,
                v_cache=v_cache,
                weights=self._bundle,
                rope_cos=self.rope_cos,
                rope_sin=self.rope_sin,
                logits=logits,
                sampled_ids_out=sampled_ids_out,
                next_hidden=next_hidden,
            )
            print(f"PYPTO_QWEN3_STAGE {stage}", flush=True)
            invoke_pypto_kernel(kernels["decode_fwd"], args, session=chip)
        else:
            # Profile / dummy batches: do not pretend a fused host ran.
            self._last_logits = logits[:, :REAL_VOCAB].contiguous()
            return dummy

        scatter_contract_kv_to_vllm(k_cache, v_cache, layer_kvs, phys_pages)

        real = slice_real_vocab_logits(logits).contiguous()
        tp_rank, tp_world = tp_rank_and_world()
        if tp_world > 1:
            comm = self._ensure_tp_comm(real.device)
            reduced = comm.allreduce_sum(real[0, :HIDDEN])
            real = real.clone()
            real[0, :HIDDEN] = reduced.to(dtype=real.dtype) / tp_world
            print(COMM_MARKER, f"stage={stage} rank={tp_rank}", flush=True)
        self._last_logits = real
        topv, topi = torch.topk(real[0], k=min(4, real.shape[-1]))
        print(
            f"PYPTO_QWEN3_LOGITS stage={stage} batch={batch} ntok={num_tokens} "
            f"seq={seq_lens.tolist()} kabs={float(k_cache.float().abs().max())} "
            f"argmax={int(real[0].argmax())} top_ids={topi.tolist()} "
            f"top_val={[float(v) for v in topv]}",
            flush=True,
        )
        return input_ids.new_zeros((input_ids.shape[0], HIDDEN), dtype=torch.bfloat16)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if self._last_logits is None:
            return None
        logits = self._last_logits.to(device=hidden_states.device, dtype=torch.float32)
        if logits.shape[0] != hidden_states.shape[0]:
            if logits.shape[0] == 1:
                logits = logits.expand(hidden_states.shape[0], -1).contiguous()
            else:
                logits = logits[: hidden_states.shape[0]].contiguous()
        return logits

    def _collect_layer_kvs(self) -> list[torch.Tensor]:
        layer_kvs: list[torch.Tensor] = []
        for attn in self.attn_layers:
            cache = attn.kv_cache
            if cache is None or (isinstance(cache, (list, tuple)) and not cache):
                raise RuntimeError("vLLM has not bound KV cache onto the pypto Attention layers")
            layer_kvs.append(cache)
        return layer_kvs

    def _ensure_tp_comm(self, device: torch.device):
        if self._tp_comm is not None:
            return self._tp_comm
        rank, world = tp_rank_and_world()
        group = _tp_cpu_group()
        self._tp_comm = build_gloo_shmem_comm(
            rank=rank,
            world_size=world,
            device=str(device),
            cols=HIDDEN,
            group=group,
            session=self._ensure_chip(),
        )
        return self._tp_comm

    def _ensure_chip(self) -> PyptoChipSession:
        if self._chip is None:
            if torch.npu.is_available():
                torch.npu.empty_cache()
            self._chip = PyptoChipSession()
        return self._chip

    def _ensure_kernels(self) -> dict[str, object]:
        if self._kernels is not None:
            return self._kernels
        # decode_fwd attention is pl.spmd(DEFAULT_BLOCK_DIM, sync_start=True).
        # Must be set before load_kernels() imports paged_attention_cce.
        os.environ.setdefault("QWEN3_PA_BLOCK_DIM", "20")
        ensure_pypto_lib_on_path()
        from pypto.backend import BackendType, set_backend_type
        from contract.registry import get_contract

        set_backend_type(BackendType.Ascend910B)
        contract = get_contract("qwen3", "14b")
        loaded = contract.load_kernels()
        self._kernels = {
            "prefill_fwd": loaded.functions["prefill_fwd"],
            "decode_fwd": loaded.functions["decode_fwd"],
        }
        logger.info("Loaded pypto-lib Qwen3-14B kernels %s / %s", STAGE_PREFILL, STAGE_DECODE)
        return self._kernels

    @staticmethod
    def _validate_single_card_shape(vllm_config: VllmConfig) -> None:
        config = vllm_config.model_config.hf_config
        block_size = int(vllm_config.cache_config.block_size)
        max_model_len = int(vllm_config.model_config.max_model_len)
        tp = int(vllm_config.parallel_config.tensor_parallel_size)
        if tp not in (1, 2):
            raise ValueError(f"PyptoQwen3ForCausalLM supports tp=1 or tp=2, got tp={tp}")
        if block_size != PAGE_SIZE:
            raise ValueError(f"PyptoQwen3ForCausalLM requires block_size={PAGE_SIZE}, got {block_size}")
        if max_model_len > MAX_SEQ:
            raise ValueError(f"max_model_len {max_model_len} exceeds kernel MAX_SEQ {MAX_SEQ}")
        expected = {
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": NUM_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
        }
        actual = {key: getattr(config, key, None) for key in expected}
        if actual != expected:
            raise ValueError(f"checkpoint shape is not Qwen3-14B: {actual} != {expected}")


def _unwrap_attn_metadata(raw: object) -> object:
    if raw is None:
        raise RuntimeError("forward context has no attn_metadata")
    if hasattr(raw, "slot_mapping") and (
        hasattr(raw, "block_tables") or hasattr(raw, "block_table")
    ):
        return raw
    if isinstance(raw, dict):
        for value in raw.values():
            try:
                return _unwrap_attn_metadata(value)
            except RuntimeError:
                continue
    if isinstance(raw, (list, tuple)):
        for value in raw:
            try:
                return _unwrap_attn_metadata(value)
            except RuntimeError:
                continue
    raise RuntimeError(f"cannot find AscendMetadata in attn_metadata type {type(raw)!r}")
