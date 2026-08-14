#!/usr/bin/env python3
"""Collect TP=1 (fused host) Qwen3-14B swimlanes and a torch_npu profile.

Uses the published fused hosts (``qwen3_14b.prefill_fwd`` / ``decode_fwd``).
Writes under ``tp1/swimlane`` and ``tp1/torch``.

    python collect_qwen3_profiles.py --mode swimlane
    python collect_qwen3_profiles.py --mode torch
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

MODEL_PATH = os.environ.get("QWEN3_14B_PATH", "/mnt/workspace/inductor/models/Qwen3-14B")
INDUCTOR_ROOT = "/mnt/workspace/inductor"
HERE = Path(__file__).resolve().parent
OUT = HERE / "tp1"
SWIMLANE_DIR = OUT / "swimlane"
TORCH_DIR = OUT / "torch"


def _sanitize_sys_path() -> None:
    cleaned = [entry for entry in sys.path if os.path.abspath(entry) != INDUCTOR_ROOT]
    sys.path[:] = cleaned


def _prepare_env() -> None:
    os.environ.setdefault("PYPTO_LIB_ROOT", str(Path(INDUCTOR_ROOT) / "pypto-lib"))
    os.environ.setdefault("PTO_PLATFORM", "a2a3")
    os.environ.setdefault("QWEN3_PA_BLOCK_DIM", "20")
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"


def _load_runtime():
    import torch
    from transformers import AutoTokenizer

    from vllm_ascend.models.pypto_qwen3_adapter import (
        HEAD_DIM,
        HIDDEN,
        NUM_LAYERS,
        PADDED_VOCAB,
        PAGE_SIZE,
        SAMPLED_IDS_PAD,
        STAGE_DECODE,
        STAGE_PREFILL,
        PyptoChipSession,
        build_decode_kernel_args,
        build_prefill_kernel_args,
        build_rope_tables,
        compact_vllm_kv_for_contract,
        ensure_pypto_lib_on_path,
        invoke_pypto_kernel,
        load_hf_state_from_dir,
        pack_official_weights,
        pin_weight_bundle_dtypes,
        scatter_contract_kv_to_vllm,
        slice_real_vocab_logits,
    )

    return torch, AutoTokenizer, {
        "HEAD_DIM": HEAD_DIM,
        "HIDDEN": HIDDEN,
        "NUM_LAYERS": NUM_LAYERS,
        "PADDED_VOCAB": PADDED_VOCAB,
        "PAGE_SIZE": PAGE_SIZE,
        "SAMPLED_IDS_PAD": SAMPLED_IDS_PAD,
        "STAGE_DECODE": STAGE_DECODE,
        "STAGE_PREFILL": STAGE_PREFILL,
        "PyptoChipSession": PyptoChipSession,
        "build_decode_kernel_args": build_decode_kernel_args,
        "build_prefill_kernel_args": build_prefill_kernel_args,
        "build_rope_tables": build_rope_tables,
        "compact_vllm_kv_for_contract": compact_vllm_kv_for_contract,
        "ensure_pypto_lib_on_path": ensure_pypto_lib_on_path,
        "invoke_pypto_kernel": invoke_pypto_kernel,
        "load_hf_state_from_dir": load_hf_state_from_dir,
        "pack_official_weights": pack_official_weights,
        "pin_weight_bundle_dtypes": pin_weight_bundle_dtypes,
        "scatter_contract_kv_to_vllm": scatter_contract_kv_to_vllm,
        "slice_real_vocab_logits": slice_real_vocab_logits,
    }


def _make_session(device_id: int, *, swimlane: bool, dep_gen: bool = False) -> object:
    from vllm_ascend.models.pypto_qwen3_adapter import PyptoChipSession

    session = PyptoChipSession(device_id=device_id)
    session.config.device_id = device_id
    session.config.enable_l2_swimlane = bool(swimlane)
    session.config.enable_dep_gen = bool(dep_gen)
    # decode_fwd fan-in overflows the default spill pool when DFX is on.
    session.config.ring_dep_pool = 1 << 18
    return session


def _build_model(device: str, A: dict):
    A["ensure_pypto_lib_on_path"]()
    from contract.registry import get_contract
    from pypto.backend import BackendType, set_backend_type

    set_backend_type(BackendType.Ascend910B)
    kernels = get_contract("qwen3", "14b").load_kernels().functions
    tok = A["AutoTokenizer"].from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "1+1等于几？只回答数字。"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = A["torch"].tensor(tok.encode(prompt, add_special_tokens=False), dtype=A["torch"].int32)
    bundle = A["pin_weight_bundle_dtypes"](
        A["pack_official_weights"](A["load_hf_state_from_dir"](MODEL_PATH).items())
    )
    bundle = type(bundle)(**{name: tensor.to(device) for name, tensor in bundle.__dict__.items()})
    rope_cos, rope_sin = A["build_rope_tables"]()
    rope_cos = rope_cos.to(device)
    rope_sin = rope_sin.to(device)
    layer_kvs = [
        (
            A["torch"].zeros(1, A["PAGE_SIZE"], 8, A["HEAD_DIM"], dtype=A["torch"].bfloat16, device=device),
            A["torch"].zeros(1, A["PAGE_SIZE"], 8, A["HEAD_DIM"], dtype=A["torch"].bfloat16, device=device),
        )
        for _ in range(A["NUM_LAYERS"])
    ]
    return tok, input_ids, bundle, rope_cos, rope_sin, layer_kvs, kernels


def _step(stage: str, token_ids, seq: int, *, device, A, bundle, rope_cos, rope_sin, layer_kvs, kernels, session):
    torch = A["torch"]
    seq_lens = torch.tensor([seq], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, int(token_ids.numel())], dtype=torch.int32, device=device)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    if stage == A["STAGE_PREFILL"]:
        slot_mapping = torch.arange(int(token_ids.numel()), dtype=torch.int32, device=device)
    else:
        slot_mapping = torch.tensor([seq - 1], dtype=torch.int32, device=device)
    chunk_lens = torch.tensor([int(token_ids.numel())], dtype=torch.int32, device=device)
    k_cache, v_cache, compact_table, compact_slots, phys = A["compact_vllm_kv_for_contract"](
        layer_kvs, block_table, slot_mapping, seq_lens, chunk_lens=chunk_lens
    )
    logits = torch.zeros((1, A["PADDED_VOCAB"]), dtype=torch.float32, device=device)
    print(f"PYPTO_QWEN3_STAGE {stage}", flush=True)
    if stage == A["STAGE_PREFILL"]:
        args = A["build_prefill_kernel_args"](
            input_ids=token_ids.to(device),
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_table=compact_table,
            slot_mapping=compact_slots,
            k_cache=k_cache,
            v_cache=v_cache,
            weights=bundle,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            logits=logits,
        )
        A["invoke_pypto_kernel"](kernels["prefill_fwd"], args, session=session)
    else:
        sampled_out = torch.zeros((1, A["SAMPLED_IDS_PAD"]), dtype=torch.int32, device=device)
        next_hidden = torch.zeros((1, A["HIDDEN"]), dtype=torch.bfloat16, device=device)
        args = A["build_decode_kernel_args"](
            token_ids=token_ids.to(device),
            seq_lens=seq_lens,
            block_table=compact_table,
            slot_mapping=compact_slots,
            k_cache=k_cache,
            v_cache=v_cache,
            weights=bundle,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            logits=logits,
            sampled_ids_out=sampled_out,
            next_hidden=next_hidden,
        )
        A["invoke_pypto_kernel"](kernels["decode_fwd"], args, session=session)
    A["scatter_contract_kv_to_vllm"](k_cache, v_cache, layer_kvs, phys)
    return A["slice_real_vocab_logits"](logits)


def _convert_swimlanes(out_dir: Path) -> list[Path]:
    converted: list[Path] = []
    for records in out_dir.rglob("l2_swimlane_records.json"):
        dest = out_dir / f"merged_swimlane_{records.parent.parent.name}.json"
        cmd = [
            sys.executable,
            "-m",
            "simpler_setup.tools.swimlane_converter",
            str(records),
            "-o",
            str(dest),
        ]
        print("CONVERT", " ".join(cmd), flush=True)
        import subprocess

        subprocess.check_call(cmd)
        converted.append(dest)
        print(f"SWIMLANE_JSON {dest}", flush=True)
    return converted


def _plot_swimlane_png(records_path: Path, png_path: Path, title: str) -> None:
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from simpler_setup.tools.swimlane_converter import read_perf_data

    data = read_perf_data(str(records_path))
    tasks = data.get("aicore_tasks") or data.get("joined_tasks") or []
    if not tasks and isinstance(data, dict):
        # Fallback: raw file
        raw = json.loads(records_path.read_text())
        tasks = raw.get("aicore_tasks") or []
    if not tasks:
        print(f"no aicore tasks in {records_path}", flush=True)
        return
    # Normalize to dicts with start/end/core
    rows = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        start = task.get("start_time_us", task.get("start_us"))
        end = task.get("end_time_us", task.get("end_us"))
        if start is None or end is None:
            continue
        core = f"{task.get('core_type', 'core')}{task.get('core_id', '')}"
        name = str(task.get("func_name") or task.get("name") or task.get("func_id") or "task")
        rows.append((core, float(start), float(end), name))
    if not rows:
        print(f"no timed rows in {records_path}", flush=True)
        return
    t0 = min(r[1] for r in rows)
    cores = sorted({r[0] for r in rows})
    fig_h = max(4.0, 0.22 * len(cores) + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    colors = plt.cm.tab20.colors
    for idx, core in enumerate(cores):
        segs = [(r[1] - t0, r[2] - r[1]) for r in rows if r[0] == core]
        ax.broken_barh(segs, (idx - 0.4, 0.8), facecolors=colors[idx % len(colors)])
    ax.set_yticks(range(len(cores)))
    ax.set_yticklabels(cores, fontsize=7)
    ax.set_xlabel("time (us from first AICore start)")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"SWIMLANE_PNG {png_path} tasks={len(rows)} cores={len(cores)}", flush=True)


def _create_torch_profiler(save_path: Path, active: int):
    import torch_npu

    save_path.mkdir(parents=True, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU,
        ],
        with_stack=True,
        record_shapes=False,
        profile_memory=False,
        experimental_config=experimental_config,
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=0, active=active, repeat=1, skip_first=0
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(save_path)),
    )


def _plot_torch_png(prof_root: Path, png_path: Path) -> None:
    import csv
    from collections import defaultdict

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = next(prof_root.rglob("kernel_details.csv"), None)
    if csv_path is None:
        print("kernel_details.csv missing", flush=True)
        return
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        rows = list(reader)
    print(f"kernel_details {csv_path} rows={len(rows)} cols={len(cols)}", flush=True)
    dur_key = next((k for k in ("Duration(us)", "Duration(µs)", "dur") if k in cols), None)
    name_key = next((k for k in ("Name", "Op Name", "OP Type") if k in cols), None)
    if not dur_key or not name_key:
        print(f"unexpected columns: {cols[:12]}", flush=True)
        return
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        try:
            totals[row[name_key]] += float(row[dur_key])
        except (TypeError, ValueError):
            continue
    top = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [item[0][:60] for item in reversed(top)]
    vals = [item[1] / 1000.0 for item in reversed(top)]
    ax.barh(names, vals)
    ax.set_xlabel("sum Duration (ms)")
    ax.set_title("torch_npu profiler — top kernels")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"TORCH_PNG {png_path}", flush=True)


def _snapshot_latest_dfx(label: str) -> Path | None:
    records = sorted(Path.cwd().rglob("l2_swimlane_records.json"), key=lambda p: p.stat().st_mtime)
    if not records:
        print(f"no l2_swimlane_records.json after {label}", flush=True)
        return None
    src = records[-1].parent
    dest = SWIMLANE_DIR / f"{label}_records"
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dest / item.name
        if item.is_file():
            shutil.copy2(item, target)
    print(f"SNAPSHOT {label} <- {src} -> {dest}", flush=True)
    return dest


def run_swimlane(device_id: int, stages: str) -> int:
    torch, AutoTokenizer, A = _load_runtime()
    A["AutoTokenizer"] = AutoTokenizer
    A["torch"] = torch
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    want_prefill = stages in ("prefill", "both")
    want_decode = stages in ("decode", "both")

    # Prefill DFX and decode DFX must not share a ChipWorker config flip mid-run
    # in a way that reuses a mixed save dir. Compile/run each stage separately.
    session = _make_session(device_id, swimlane=False, dep_gen=False)
    tok, input_ids, bundle, rope_cos, rope_sin, layer_kvs, kernels = _build_model(device, A)
    ntok = int(input_ids.numel())
    ok = True

    if want_prefill:
        session.config.enable_l2_swimlane = True
        session.config.enable_dep_gen = True
        _step(
            A["STAGE_PREFILL"], input_ids, ntok,
            device=device, A=A, bundle=bundle, rope_cos=rope_cos, rope_sin=rope_sin,
            layer_kvs=layer_kvs, kernels=kernels, session=session,
        )
        snap = _snapshot_latest_dfx("prefill")
        ok = ok and snap is not None
        session.config.enable_l2_swimlane = False
        session.config.enable_dep_gen = False
    else:
        _step(
            A["STAGE_PREFILL"], input_ids, ntok,
            device=device, A=A, bundle=bundle, rope_cos=rope_cos, rope_sin=rope_sin,
            layer_kvs=layer_kvs, kernels=kernels, session=session,
        )

    if want_decode:
        next_id = 17  # overwritten after a real prefill
        # Use last prefill argmax if we just ran it with logits still unknown;
        # re-run a silent greedy id from a tiny CPU-less path: call decode after
        # the prefill above. We do not have logits if prefill was DFX-only copy.
        # Recompute next token without DFX if needed.
        session.config.enable_l2_swimlane = False
        session.config.enable_dep_gen = False
        # Fresh KV + prefill without DFX so decode sees a valid cache.
        layer_kvs = [
            (
                torch.zeros(1, A["PAGE_SIZE"], 8, A["HEAD_DIM"], dtype=torch.bfloat16, device=device),
                torch.zeros(1, A["PAGE_SIZE"], 8, A["HEAD_DIM"], dtype=torch.bfloat16, device=device),
            )
            for _ in range(A["NUM_LAYERS"])
        ]
        logits = _step(
            A["STAGE_PREFILL"], input_ids, ntok,
            device=device, A=A, bundle=bundle, rope_cos=rope_cos, rope_sin=rope_sin,
            layer_kvs=layer_kvs, kernels=kernels, session=session,
        )
        next_id = int(logits[0].argmax().item())
        session.config.enable_l2_swimlane = True
        session.config.enable_dep_gen = False
        _step(
            A["STAGE_DECODE"], torch.tensor([next_id], dtype=torch.int32, device=device), ntok + 1,
            device=device, A=A, bundle=bundle, rope_cos=rope_cos, rope_sin=rope_sin,
            layer_kvs=layer_kvs, kernels=kernels, session=session,
        )
        snap = _snapshot_latest_dfx("decode")
        ok = ok and snap is not None

    for label in (("prefill",) if want_prefill else ()) + (("decode",) if want_decode else ()):
        snap = SWIMLANE_DIR / f"{label}_records"
        merged = list(snap.glob("merged_swimlane*.json"))
        records = snap / "l2_swimlane_records.json"
        if not merged and records.exists():
            _convert_swimlanes(snap)
            merged = list(snap.glob("merged_swimlane*.json"))
        if merged:
            shutil.copy2(merged[-1], SWIMLANE_DIR / f"{label}.json")
        if records.exists():
            _plot_swimlane_png(records, SWIMLANE_DIR / f"{label}.png", f"PyPTO TP1 L2 {label}")
        print(f"SWIMLANE_{label.upper()} dir={snap} merged={bool(merged)}", flush=True)
    return 0 if ok else 1


def run_torch(device_id: int, decode_steps: int) -> int:
    torch, AutoTokenizer, A = _load_runtime()
    A["AutoTokenizer"] = AutoTokenizer
    A["torch"] = torch
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    session = _make_session(device_id, swimlane=False)
    tok, input_ids, bundle, rope_cos, rope_sin, layer_kvs, kernels = _build_model(device, A)
    ntok = int(input_ids.numel())

    def generate(n_decode: int) -> str:
        logits = _step(
            A["STAGE_PREFILL"], input_ids, ntok,
            device=device, A=A, bundle=bundle, rope_cos=rope_cos, rope_sin=rope_sin,
            layer_kvs=layer_kvs, kernels=kernels, session=session,
        )
        ids = [int(logits[0].argmax().item())]
        seq = ntok + 1
        for _ in range(n_decode):
            logits = _step(
                A["STAGE_DECODE"],
                torch.tensor([ids[-1]], dtype=torch.int32, device=device),
                seq,
                device=device, A=A, bundle=bundle, rope_cos=rope_cos, rope_sin=rope_sin,
                layer_kvs=layer_kvs, kernels=kernels, session=session,
            )
            ids.append(int(logits[0].argmax().item()))
            seq += 1
        return tok.decode(ids, skip_special_tokens=True).strip()

    print("TORCH_WARMUP compile+one generate outside profiler", flush=True)
    warm = generate(1)
    print(f"TORCH_WARMUP_TEXT {warm!r}", flush=True)
    # Fresh KV for the profiled generate.
    layer_kvs = [
        (
            torch.zeros(1, A["PAGE_SIZE"], 8, A["HEAD_DIM"], dtype=torch.bfloat16, device=device),
            torch.zeros(1, A["PAGE_SIZE"], 8, A["HEAD_DIM"], dtype=torch.bfloat16, device=device),
        )
        for _ in range(A["NUM_LAYERS"])
    ]
    prof_dir = Path("/tmp/qwen3_torch_npu_prof")
    if prof_dir.exists():
        shutil.rmtree(prof_dir)
    active = 1 + decode_steps
    with _create_torch_profiler(prof_dir, active=active) as prof:
        logits = _step(
            A["STAGE_PREFILL"], input_ids, ntok,
            device=device, A=A, bundle=bundle, rope_cos=rope_cos, rope_sin=rope_sin,
            layer_kvs=layer_kvs, kernels=kernels, session=session,
        )
        torch.npu.synchronize()
        prof.step()
        nxt = int(logits[0].argmax().item())
        ids = [nxt]
        seq = ntok + 1
        for _ in range(decode_steps):
            logits = _step(
                A["STAGE_DECODE"],
                torch.tensor([ids[-1]], dtype=torch.int32, device=device),
                seq,
                device=device, A=A, bundle=bundle, rope_cos=rope_cos, rope_sin=rope_sin,
                layer_kvs=layer_kvs, kernels=kernels, session=session,
            )
            torch.npu.synchronize()
            prof.step()
            ids.append(int(logits[0].argmax().item()))
            seq += 1
        prof.step()
    text = tok.decode(ids, skip_special_tokens=True).strip()
    print(f"TORCH_OUTPUT {text!r}", flush=True)
    dest = TORCH_DIR / "prof"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(prof_dir, dest)
    _plot_torch_png(dest, TORCH_DIR / "top_kernels.png")
    print(f"TORCH_PROF_DIR {dest}", flush=True)
    return 0


def main() -> int:
    _sanitize_sys_path()
    _prepare_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("swimlane", "torch"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--stages", choices=("prefill", "decode", "both"), default="both")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SWIMLANE_DIR.mkdir(parents=True, exist_ok=True)
    TORCH_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PTO2_RING_DEP_POOL", "262144")
    if args.mode == "swimlane":
        return run_swimlane(args.device, args.stages)
    return run_torch(args.device, args.decode_steps)


if __name__ == "__main__":
    raise SystemExit(main())
