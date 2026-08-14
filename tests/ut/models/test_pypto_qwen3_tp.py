#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""CPU tests for the shipped PyPTO Qwen3-14B TP=2 adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
    fused = (root / "vllm_ascend" / "models" / "pypto_qwen3_tp_fused.py").read_text()
    assert "select_compute_bundle" in example
    assert "gloo_gather_contract_bundle" not in example
    assert "real[0, :HIDDEN]" not in example
    assert "PyptoTpFusedHost" in example
    assert "PYPTO_QWEN3_TP_FUSED" in example
    assert 'os.environ.get("PYPTO_QWEN3_TP_FUSED", "1") == "1"' in example
    assert "用两句话介绍北京。" in example
    assert "PYPTO_MIN_OUTPUT_TOKENS" in example
    assert "PYPTO_EXPECT_TEXT" in example
    assert "dist.ReduceOp.MIN" in example
    assert "def _select_rank0_token" in example
    assert "dist.broadcast(selected, src=0)" in example
    assert "rank_logits_match" in example
    assert "gloo_gather_contract_bundle" not in model
    assert "real[0, :HIDDEN]" not in model
    assert "PYPTO_QWEN3_EXPERIMENTAL_TP_ENGINE" in model
    assert "requires exactly one request" in model
    assert 'os.environ.get("PYPTO_QWEN3_TP_FUSED", "1") == "1"' in model
    assert "TP_BOUNDARY_O_PROJ" in runner
    assert "TP_BOUNDARY_DOWN_PROJ" in runner
    assert "launch_tag" in runner
    assert "def tp_prefill_fwd" in fused
    assert "def tp_decode_fwd" in fused
    assert "dist.barrier" not in fused
    assert fused.count("init_value=0") >= 2
    assert 'torch.device("meta")' in fused
    assert "reserve_signal_credits" in fused
    assert "tp_compile_output_dir(stage)" in fused
    tp_src = (root / "vllm_ascend" / "models" / "pypto_qwen3_tp.py").read_text()
    adapter_src = (root / "vllm_ascend" / "models" / "pypto_qwen3_adapter.py").read_text()
    assert "build_call_config(session.config" in tp_src
    assert "def capture_dfx" in adapter_src
    fused_ops = (root / "vllm_ascend" / "models" / "pypto_qwen3_tp_fused_ops.py").read_text()
    fused_layer = (root / "vllm_ascend" / "models" / "pypto_qwen3_tp_fused_layer.py").read_text()
    assert "pld.system.notify" in fused_ops
    assert "def tp_allreduce" in fused_ops
    assert "def tp_allreduce_step" in fused_layer
    assert "def tp_o_down_boundaries" not in fused_layer
    # Cross-engine GM producer/consumer chains must be separate orchestration
    # tasks.  Recombining these into one incore function races AIV and AIC.
    for task in (
        "tp_hidden_rms_step",
        "tp_qkv_gemm_step",
        "tp_qkv_post_step",
        "tp_prefill_attention_scores_step",
        "tp_prefill_attention_softmax_step",
        "tp_prefill_attention_context_step",
        "tp_decode_attention_scores_step",
        "tp_decode_attention_softmax_step",
        "tp_decode_attention_context_step",
        "tp_attention_cast_step",
        "tp_o_gemm_step",
        "tp_ffn_residual_rms_step",
        "tp_ffn_gate_up_step",
        "tp_ffn_swiglu_step",
        "tp_ffn_down_step",
    ):
        assert f"def {task}" in fused_layer
        assert f"{task}(" in fused
    for unsafe_group in (
        "tp_prefill_layer",
        "tp_decode_front",
        "tp_decode_back",
        "tp_ffn_down",
    ):
        assert f"def {unsafe_group}(" not in fused_layer
        assert f"{unsafe_group}(" not in fused
    assert fused_layer.count("tp_allreduce_step(") >= 5
    assert "tp_allreduce_o(" not in fused_layer
    assert "tp_allreduce_down(" not in fused_layer
    assert "tp_allreduce_o(" not in fused
    assert "tp_allreduce_down(" not in fused
    assert fused.count("tp_allreduce_step(") == 4

    profile = (
        root / "tests" / "pypto_qwen3_profiles" / "collect_qwen3_tp2_profiles.py"
    ).read_text()
    assert "EXPECTED_WHOLE_GRAPH_TASKS = 1 + 40 * 16 + 2" in profile
    assert 'Counter({"aiv": 402, "aic": 241})' in profile
    assert "_assert_torch_whole_graph_steps(prof_dir, active)" in profile
    assert "select_rank0_token" in profile
    assert "assert_rank_logits_match" in profile


def test_tp_allreduce_slot_plan_matches_shmem_carve() -> None:
    from pypto.runtime.shmem_gloo import COMM_CONTEXT_SIZE, align_up, carve_window_layout

    names, nbytes = tp_allreduce_slot_plan(adapter.HIDDEN, torch.float32)
    assert names == ("data_buf", "signal")
    assert nbytes[0] == adapter.HIDDEN * 4
    assert nbytes[1] == 2 * 4
    offsets, window_bytes = carve_window_layout((COMM_CONTEXT_SIZE, *nbytes))
    assert offsets[0] == 0
    assert window_bytes == align_up(COMM_CONTEXT_SIZE) + align_up(nbytes[0]) + align_up(nbytes[1])
    assert tp.COMM_MARKER.startswith("PYPTO_QWEN3_COMM gloo+shmem+pypto")
    text = Path(tp.__file__).read_text()
    assert "pld.system.notify" in text
    assert "pld.system.wait" in text
    assert text.count("pld.system.notify") >= 2
    assert text.count("pld.system.wait") >= 2
    assert text.count("pl.RUNTIME") >= 2
    assert "def allreduce_sum_fused" in text
    assert "_zero_window_slot(window, \"signal\"" in text
    assert "rank{rank}_pid{os.getpid()}" in text
    assert text.count("torch.npu.synchronize()") >= 3
    assert "dist.barrier" not in text.split("def allreduce_sum_fused")[1].split("def build_gloo_shmem_comm")[0]


def test_fused_allreduce_credits_are_monotonic_and_guard_int32() -> None:
    comm = object.__new__(tp.PyptoGlooShmemComm)
    comm.signal_credit = 0
    assert comm.reserve_signal_credits(2) == 0
    assert comm.reserve_signal_credits(4) == 2
    assert comm.signal_credit == 6

    comm.signal_credit = torch.iinfo(torch.int32).max - 1
    try:
        comm.reserve_signal_credits(2)
    except OverflowError as exc:
        assert "overflow INT32" in str(exc)
    else:
        raise AssertionError("expected INT32 signal credit overflow")


def test_fused_allreduce_dispatches_credits_before_materialized_contexts(monkeypatch) -> None:
    dispatched: list[tuple[object, ...]] = []
    window = SimpleNamespace(
        device_ctx_ptr=77,
        as_device_tensor=lambda *_args: SimpleNamespace(),
    )
    session = SimpleNamespace(launch_tag=None)
    comm = tp.PyptoGlooShmemComm(
        rank=0,
        world_size=2,
        window=window,
        session=session,
        publish=None,
        allreduce=None,
        fused=object(),
        cols=adapter.HIDDEN,
        rows=tp.TP_TOK_PAD,
        group=None,
    )
    monkeypatch.setattr(tp, "_dispatch_chip", lambda *args: dispatched.append(args))

    result = comm.allreduce_sum_fused(torch.ones((1, adapter.HIDDEN)))

    assert result.shape == (1, adapter.HIDDEN)
    assert len(dispatched) == 1
    assert dispatched[0][-4:] == (1, 2, 77, 77)


def test_zero_window_slot_does_not_clobber_neighboring_shmem_slots() -> None:
    storage = torch.full((64,), 0xA5, dtype=torch.uint8)
    window = SimpleNamespace(tensor=storage, offsets={"data_buf": 0, "signal": 32})
    tp._zero_window_slot(window, "signal", 8)
    assert torch.count_nonzero(storage[32:40]) == 0
    assert torch.all(storage[:32] == 0xA5)
    assert torch.all(storage[40:] == 0xA5)


def test_tp_probes_keep_monotonic_credits_and_global_verdicts() -> None:
    root = Path(__file__).resolve().parents[3]
    allreduce_probe = (root / "examples" / "offline_pypto_qwen3_allreduce_fused.py").read_text()
    layer_probe = (root / "examples" / "offline_pypto_qwen3_tp_fused_layer.py").read_text()
    assert "dist.all_reduce(global_ok, op=dist.ReduceOp.MIN)" in allreduce_probe
    assert "dist.all_reduce(global_ok, op=dist.ReduceOp.MIN)" in layer_probe
    assert layer_probe.count("reserve_signal_credits(4)") == 2
    assert 'offsets["signal"]' not in layer_probe
