#!/usr/bin/env python3
"""Collect TP=2 fused whole-model PyPTO swimlanes or a torch_npu profile.

The fused host launches the complete 40-layer graph exactly once for prefill
and once for decode. ``--mode`` is intentionally exclusive: L2 DFX and the
torch profiler must run in separate torchrun processes.

Launch::

    python -m torch.distributed.run --standalone --nproc_per_node=2 \\
        tests/pypto_qwen3_profiles/collect_qwen3_tp2_profiles.py \\
        --mode swimlane

    python -m torch.distributed.run --standalone --nproc_per_node=2 \\
        tests/pypto_qwen3_profiles/collect_qwen3_tp2_profiles.py \\
        --mode torch

Writes under ``tp2/swimlane`` and ``tp2/torch``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

MODEL_PATH = os.environ.get("QWEN3_14B_PATH", "/mnt/workspace/inductor/models/Qwen3-14B")
INDUCTOR_ROOT = "/mnt/workspace/inductor"
HERE = Path(__file__).resolve().parent
OUT = HERE / "tp2"
SWIMLANE_DIR = OUT / "swimlane"
TORCH_DIR = OUT / "torch"
SCRATCH = Path(
    os.environ.get(
        "PYPTO_TP2_PROF_SCRATCH",
        "/tmp/pypto_qwen3_tp2_profiles",
    )
)
EXPECTED_WHOLE_GRAPH_TASKS = 1 + 40 * 16 + 2
EXPECTED_WHOLE_GRAPH_CORE_TASKS = Counter({"aiv": 402, "aic": 241})


def _sanitize_sys_path() -> None:
    sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry) != INDUCTOR_ROOT]


def _prepare_env() -> None:
    os.environ.setdefault("PYPTO_LIB_ROOT", str(Path(INDUCTOR_ROOT) / "pypto-lib"))
    os.environ.setdefault("PTO_PLATFORM", "a2a3")
    os.environ.setdefault("QWEN3_PA_BLOCK_DIM", "20")
    os.environ.setdefault("PTO2_RING_DEP_POOL", "262144")
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"


def _skip_per_launch_convert() -> None:
    import pypto.runtime.runner as runtime_runner

    runtime_runner._generate_swimlane = lambda *args, **kwargs: None


def _load_records(path: Path) -> dict:
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data, _end = json.JSONDecoder().raw_decode(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def _validate_aicore_tasks(records: dict, source: Path) -> list[dict]:
    """Validate raw L2 records and normalize every AICore task."""
    try:
        level = int(records.get("l2_swimlane_level"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} has invalid l2_swimlane_level") from exc
    if level not in (1, 2, 3, 4):
        raise ValueError(f"{source} has unsupported l2_swimlane_level={level}")
    metadata = records.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{source} has no metadata object")
    try:
        frequency = int(metadata.get("clock_freq_hz"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} has invalid metadata.clock_freq_hz") from exc
    if frequency <= 0:
        raise ValueError(f"{source} has non-positive metadata.clock_freq_hz={frequency}")
    raw_tasks = records.get("aicore_tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError(f"{source} has no aicore_tasks")
    core_types = metadata.get("core_types") or []
    if not isinstance(core_types, list):
        raise ValueError(f"{source} metadata.core_types is not a list")
    try:
        num_cores = int(metadata.get("num_cores"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} has invalid metadata.num_cores") from exc
    if num_cores <= 0 or len(core_types) != num_cores:
        raise ValueError(
            f"{source} has inconsistent core metadata: num_cores={num_cores}, core_types={len(core_types)}"
        )

    tasks: list[dict] = []
    for index, row in enumerate(raw_tasks):
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            raise ValueError(f"{source} aicore_tasks[{index}] has fewer than 5 columns")
        try:
            core_id = int(row[0])
            task_token = int(row[1])
            registered_task_id = int(row[2])
            start_cycles = int(row[3])
            end_cycles = int(row[4])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source} aicore_tasks[{index}] contains a non-integer field") from exc
        if not 0 <= core_id < num_cores:
            raise ValueError(f"{source} aicore_tasks[{index}] has out-of-range core_id={core_id}")
        if start_cycles <= 0 or end_cycles <= start_cycles:
            raise ValueError(f"{source} aicore_tasks[{index}] has invalid interval [{start_cycles}, {end_cycles}]")
        core_type = str(core_types[core_id])
        if core_type not in _CORE_COLORS:
            raise ValueError(f"{source} aicore_tasks[{index}] has unsupported core type {core_type!r}")
        tasks.append(
            {
                "core_id": core_id,
                "core_type": core_type,
                "task_token": task_token,
                "ring_id": (task_token >> 32) & 0xFFFFFFFF,
                "local_task_id": task_token & 0xFFFFFFFF,
                "registered_task_id": registered_task_id,
                "start_cycles": start_cycles,
                "end_cycles": end_cycles,
            }
        )
    return tasks


def _task_timing(records: dict, source: Path) -> tuple[list[dict], float, int]:
    tasks = _validate_aicore_tasks(records, source)
    frequency = int(records["metadata"]["clock_freq_hz"])
    base_cycles = min(int(task["start_cycles"]) for task in tasks)
    scale_us = 1_000_000.0 / float(frequency)
    for task in tasks:
        task["start_us"] = (int(task["start_cycles"]) - base_cycles) * scale_us
        task["duration_us"] = (int(task["end_cycles"]) - int(task["start_cycles"])) * scale_us
    return tasks, scale_us, base_cycles


_CORE_COLORS = {
    "aic": "#E45756",
    "aiv": "#4C78A8",
    "unknown": "#9D755D",
}


def _plot_task_gantt(records: dict, source: Path, png_path: Path, title: str) -> None:
    """Render every raw AICore record on its physical-core lane."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    tasks, _scale_us, _base_cycles = _task_timing(records, source)
    cores = sorted({int(task["core_id"]) for task in tasks})
    by_core: dict[int, list[dict]] = {core: [] for core in cores}
    for task in tasks:
        by_core[int(task["core_id"])].append(task)
    wall_us = max(float(task["start_us"]) + float(task["duration_us"]) for task in tasks)
    fig_h = max(8.0, 0.24 * len(cores) + 2.5)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    seen_types: set[str] = set()
    for lane, core in enumerate(cores):
        for task in sorted(by_core[core], key=lambda item: float(item["start_us"])):
            core_type = str(task["core_type"])
            seen_types.add(core_type)
            ax.broken_barh(
                [
                    (
                        float(task["start_us"]) / 1000.0,
                        max(float(task["duration_us"]) / 1000.0, 0.0001),
                    )
                ],
                (lane - 0.4, 0.8),
                facecolors=_CORE_COLORS.get(core_type, _CORE_COLORS["unknown"]),
                edgecolors="#222222",
                linewidth=0.15,
            )
    ax.set_yticks(range(len(cores)))
    ax.set_yticklabels([f"core {core}" for core in cores], fontsize=7)
    ax.set_xlabel("device time from first AICore task (ms)")
    ax.set_ylabel("physical AICore")
    ax.set_title(f"{title}\n{len(tasks)} raw tasks on {len(cores)} cores; wall={wall_us / 1000.0:.3f} ms")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        handles=[
            Patch(color=_CORE_COLORS.get(kind, _CORE_COLORS["unknown"]), label=kind) for kind in sorted(seen_types)
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise RuntimeError(f"failed to write task-level swimlane PNG {png_path}")
    print(
        f"SWIMLANE_PNG {png_path} tasks={len(tasks)} cores={len(cores)} wall_ms={wall_us / 1000.0:.3f}",
        flush=True,
    )


def _write_task_chrome_trace(records: dict, source: Path, dest: Path, title: str) -> None:
    tasks, _scale_us, _base_cycles = _task_timing(records, source)
    events: list[dict] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": 4,
            "tid": 0,
            "args": {"name": title},
        }
    ]
    for core in sorted({int(task["core_id"]) for task in tasks}):
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 4,
                "tid": core,
                "args": {"name": f"AICore {core}"},
            }
        )
    for task in tasks:
        events.append(
            {
                "name": f"r{task['ring_id']}:t{task['local_task_id']}",
                "cat": str(task["core_type"]),
                "ph": "X",
                "ts": float(task["start_us"]),
                "dur": max(float(task["duration_us"]), 0.001),
                "pid": 4,
                "tid": int(task["core_id"]),
                "args": {
                    "task_token": str(task["task_token"]),
                    "registered_task_id": int(task["registered_task_id"]),
                    "start_cycles": int(task["start_cycles"]),
                    "end_cycles": int(task["end_cycles"]),
                },
            }
        )
    dest.write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ms"}))
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"failed to write task-level Chrome trace {dest}")
    print(f"CHROME_TRACE {dest} tasks={len(tasks)}", flush=True)


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
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=active, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(save_path)),
    )


def _require_torch_profile_files(prof_root: Path) -> tuple[list[Path], list[Path]]:
    csv_paths = [path for path in prof_root.rglob("kernel_details.csv") if path.stat().st_size > 0]
    trace_paths = [path for path in prof_root.rglob("trace_view.json") if path.stat().st_size > 0]
    if not csv_paths:
        raise RuntimeError(f"torch profiler produced no non-empty kernel_details.csv under {prof_root}")
    if not trace_paths:
        raise RuntimeError(f"torch profiler produced no non-empty trace_view.json under {prof_root}")
    return csv_paths, trace_paths


def _assert_torch_whole_graph_steps(prof_root: Path, active: int) -> None:
    import csv

    csv_paths, _trace_paths = _require_torch_profile_files(prof_root)
    graph_steps: list[int] = []
    for csv_path in csv_paths:
        with csv_path.open() as handle:
            for row in csv.DictReader(handle):
                if row.get("Name") != "aicore_kernel_0":
                    continue
                try:
                    graph_steps.append(int(row["Step Id"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"torch profiler whole-graph row has invalid Step Id in {csv_path}"
                    ) from exc
    expected_steps = list(range(active))
    if sorted(graph_steps) != expected_steps:
        raise RuntimeError(
            "torch profiler did not capture every fused whole-graph call: "
            f"expected aicore_kernel_0 steps {expected_steps}, got {sorted(graph_steps)}"
        )


def _plot_torch_png(prof_root: Path, png_path: Path) -> None:
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_paths, trace_paths = _require_torch_profile_files(prof_root)
    csv_path = csv_paths[0]
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        rows = list(reader)
    print(f"kernel_details {csv_path} rows={len(rows)} cols={len(cols)}", flush=True)
    if not rows:
        raise RuntimeError(f"torch profiler kernel table is empty: {csv_path}")
    dur_key = next((k for k in ("Duration(us)", "Duration(µs)", "dur") if k in cols), None)
    name_key = next((k for k in ("Name", "Op Name", "OP Type") if k in cols), None)
    if not dur_key or not name_key:
        raise RuntimeError(f"unexpected kernel_details.csv columns: {cols[:12]}")
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        try:
            name = str(row[name_key]).strip()
            duration = float(row[dur_key])
        except (KeyError, TypeError, ValueError):
            continue
        if name and duration >= 0:
            totals[name] += duration
    top = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:20]
    if not top:
        raise RuntimeError(f"torch profiler kernel table has no valid duration rows: {csv_path}")
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [item[0][:60] for item in reversed(top)]
    vals = [item[1] / 1000.0 for item in reversed(top)]
    ax.barh(names, vals)
    ax.set_xlabel("sum Duration (ms)")
    ax.set_title("TP2 torch_npu profiler — top kernels")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise RuntimeError(f"failed to write torch profiler PNG {png_path}")
    print(
        f"TORCH_PNG {png_path} kernels={len(totals)} traces={len(trace_paths)}",
        flush=True,
    )


def _set_swimlane(session, enabled: bool, capture_dir: Path | None) -> None:
    session.config.enable_l2_swimlane = bool(enabled)
    session.config.enable_dep_gen = False
    session.config.ring_dep_pool = 1 << 18
    session.swimlane_capture_dir = capture_dir
    session.swimlane_seq = 0
    session.launch_tag = ""


def _reset_cache(host) -> None:
    host.k_cache.zero_()
    host.v_cache.zero_()


def _assert_single_fused_launch(launch_dir: Path, label: str, rank: int) -> tuple[Path, dict]:
    """Prove that one stage produced one valid, non-empty ChipWorker.run record."""
    records = sorted(launch_dir.glob("*/l2_swimlane_records.json"))
    if len(records) != 1:
        raise RuntimeError(
            f"fused {label} rank {rank} expected exactly one ChipWorker.run, "
            f"captured {len(records)} records under {launch_dir}"
        )
    record_path = records[0]
    meta_path = record_path.parent / "meta.json"
    if not meta_path.is_file():
        raise RuntimeError(f"fused {label} rank {rank} capture has no {meta_path}")
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"fused {label} rank {rank} has invalid capture metadata {meta_path}") from exc
    expected_tag = f"tp_{label}_fwd"
    if meta.get("tag") != expected_tag:
        raise RuntimeError(f"fused {label} rank {rank} expected tag {expected_tag!r}, got {meta.get('tag')!r}")
    try:
        raw_records = _load_records(record_path)
        tasks = _validate_aicore_tasks(raw_records, record_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"fused {label} rank {rank} has invalid L2 records {record_path}: {exc}") from exc
    core_tasks = Counter(str(task["core_type"]) for task in tasks)
    if len(tasks) != EXPECTED_WHOLE_GRAPH_TASKS or core_tasks != EXPECTED_WHOLE_GRAPH_CORE_TASKS:
        raise RuntimeError(
            f"fused {label} rank {rank} is not the complete 40-layer graph: "
            f"expected {EXPECTED_WHOLE_GRAPH_TASKS} tasks "
            f"{dict(EXPECTED_WHOLE_GRAPH_CORE_TASKS)}, got {len(tasks)} {dict(core_tasks)}"
        )
    print(
        f"PYPTO_QWEN3_PROFILE_STAGE {expected_tag} ChipWorker.run=1 aicore_tasks={len(tasks)} rank={rank}",
        flush=True,
    )
    return record_path, raw_records


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _clear_swimlane_outputs() -> None:
    for label in ("prefill", "decode"):
        _remove_path(SWIMLANE_DIR / f"{label}_records")
        _remove_path(SWIMLANE_DIR / f"{label}.json")
        _remove_path(SWIMLANE_DIR / f"{label}.png")
        _remove_path(SWIMLANE_DIR / f"{label}_trace.json")


def _clear_torch_outputs() -> None:
    _remove_path(TORCH_DIR / "prof")
    _remove_path(TORCH_DIR / "top_kernels.png")


def _finish_stage(record_path: Path, records: dict, label: str, rank: int) -> None:
    if rank != 0:
        return
    dest_dir = SWIMLANE_DIR / f"{label}_records"
    _remove_path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    clean_json = json.dumps(records)
    raw_dest = dest_dir / "l2_swimlane_records.json"
    raw_dest.write_text(clean_json)
    stage_json = SWIMLANE_DIR / f"{label}.json"
    stage_json.write_text(clean_json)
    if not raw_dest.is_file() or raw_dest.stat().st_size == 0:
        raise RuntimeError(f"failed to preserve L2 records at {raw_dest}")
    if not stage_json.is_file() or stage_json.stat().st_size == 0:
        raise RuntimeError(f"failed to preserve L2 records at {stage_json}")
    _plot_task_gantt(
        records,
        record_path,
        SWIMLANE_DIR / f"{label}.png",
        f"PyPTO TP2 fused 40-layer L2 {label} (one ChipWorker.run)",
    )
    _write_task_chrome_trace(
        records,
        record_path,
        SWIMLANE_DIR / f"{label}_trace.json",
        f"PyPTO TP2 fused {label}",
    )
    tasks = _validate_aicore_tasks(records, record_path)
    print(
        f"SWIMLANE_{label.upper()} ChipWorker.run=1 aicore_tasks={len(tasks)}",
        flush=True,
    )


def main() -> int:
    _sanitize_sys_path()
    _prepare_env()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("swimlane", "torch"),
        required=True,
        help="collect L2 swimlanes or torch profile; run the other mode in a new process",
    )
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--skip-warmup-generate", action="store_true")
    args = parser.parse_args()
    if args.decode_steps < 0:
        parser.error("--decode-steps must be non-negative")

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from vllm_ascend.models.pypto_qwen3_adapter import (
        PyptoChipSession,
        build_rope_tables,
        load_hf_state_from_dir,
        pack_official_weights,
        pin_weight_bundle_dtypes,
    )
    from vllm_ascend.models.pypto_qwen3_tp import (
        COMM_MARKER,
        build_gloo_shmem_comm,
        describe_tp_compute_args,
        select_compute_bundle,
    )
    from vllm_ascend.models.pypto_qwen3_tp_fused import PyptoTpFusedHost

    if args.mode == "swimlane":
        _skip_per_launch_convert()
    dist.init_process_group(backend="gloo")
    rank = int(dist.get_rank())
    world = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world != 2:
        print(f"need world_size=2, got {world}", file=sys.stderr)
        return 2
    if dist.get_backend() == "hccl":
        print("HCCL process group is forbidden on this path", file=sys.stderr)
        return 2
    torch.npu.set_device(local_rank)
    device = f"npu:{local_rank}"
    if rank == 0:
        OUT.mkdir(parents=True, exist_ok=True)
        SWIMLANE_DIR.mkdir(parents=True, exist_ok=True)
        TORCH_DIR.mkdir(parents=True, exist_ok=True)
        if args.mode == "swimlane":
            _clear_swimlane_outputs()
        else:
            _clear_torch_outputs()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "1+1等于几？只回答数字。"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = torch.tensor(tok.encode(prompt, add_special_tokens=False), dtype=torch.int32)
    ntok = int(input_ids.numel())
    bundle = pin_weight_bundle_dtypes(pack_official_weights(load_hf_state_from_dir(MODEL_PATH).items()))
    shard = select_compute_bundle(bundle, rank, world)
    del bundle
    print(f"PYPTO_QWEN3_ARGS {describe_tp_compute_args(shard)}", flush=True)
    if int(shard.wq.shape[-1]) != 2560:
        print(f"expected wq last-dim 2560, got {tuple(shard.wq.shape)}", file=sys.stderr)
        dist.destroy_process_group()
        return 2
    shard = type(shard)(**{name: tensor.to(device) for name, tensor in shard.__dict__.items()})
    rope_cos, rope_sin = build_rope_tables()
    session = PyptoChipSession(device_id=local_rank)
    session.config.device_id = local_rank
    session.config.ring_dep_pool = 1 << 18
    comm = build_gloo_shmem_comm(
        rank=rank,
        world_size=world,
        device=device,
        group=dist.group.WORLD,
        session=session,
    )
    host = PyptoTpFusedHost(
        shard,
        comm,
        session,
        rope_cos.to(device),
        rope_sin.to(device),
    )
    print(f"PYPTO_QWEN3_TP_PATH fused profile_mode={args.mode}", flush=True)
    print("TP2_FUSED_COMPILE start", flush=True)
    host.compile()
    print("TP2_FUSED_COMPILE done", flush=True)

    rank_logits_match = True

    def select_rank0_token(logits: torch.Tensor) -> int:
        nonlocal rank_logits_match
        local_token = int(logits[0].argmax().item())
        selected = torch.zeros((), dtype=torch.int32)
        if rank == 0:
            selected.fill_(local_token)
        dist.broadcast(selected, src=0)
        token = int(selected.item())
        rank_logits_match = rank_logits_match and local_token == token
        return token

    def assert_rank_logits_match() -> None:
        match = torch.tensor(int(rank_logits_match), dtype=torch.int32)
        dist.all_reduce(match, op=dist.ReduceOp.MIN)
        if int(match.item()) != 1:
            raise RuntimeError("TP ranks produced different argmax tokens")

    def greedy_decode(n_decode: int) -> list[int]:
        logits = host.prefill(input_ids.to(device))
        ids = [select_rank0_token(logits)]
        seq = ntok + 1
        for _ in range(max(n_decode, 0)):
            step_ids = torch.tensor([ids[-1]], dtype=torch.int32, device=device)
            logits = host.decode(step_ids, seq)
            ids.append(select_rank0_token(logits))
            seq += 1
        return ids

    if not args.skip_warmup_generate:
        print("TP2_WARMUP generate without DFX", flush=True)
        warm_ids = greedy_decode(1)
        if rank == 0:
            print(f"TP2_WARMUP_TEXT {tok.decode(warm_ids, skip_special_tokens=True).strip()!r}", flush=True)
        _reset_cache(host)

    if args.mode == "swimlane":
        prefill_dir = SCRATCH / f"rank{rank}_prefill"
        decode_dir = SCRATCH / f"rank{rank}_decode"
        if prefill_dir.exists():
            shutil.rmtree(prefill_dir)
        prefill_dir.mkdir(parents=True)
        if decode_dir.exists():
            shutil.rmtree(decode_dir)
        decode_dir.mkdir(parents=True)

        print("TP2_FUSED_SWIMLANE prefill", flush=True)
        _set_swimlane(session, True, prefill_dir)
        logits = host.prefill(input_ids.to(device))
        next_id = select_rank0_token(logits)
        _set_swimlane(session, False, None)
        prefill_record_path, prefill_records = _assert_single_fused_launch(
            prefill_dir,
            "prefill",
            rank,
        )
        dist.barrier()
        _finish_stage(prefill_record_path, prefill_records, "prefill", rank)
        dist.barrier()

        print("TP2_FUSED_SWIMLANE decode", flush=True)
        _set_swimlane(session, True, decode_dir)
        decode_logits = host.decode(
            torch.tensor([next_id], dtype=torch.int32, device=device),
            ntok + 1,
        )
        decode_id = select_rank0_token(decode_logits)
        _set_swimlane(session, False, None)
        decode_record_path, decode_records = _assert_single_fused_launch(
            decode_dir,
            "decode",
            rank,
        )
        dist.barrier()
        _finish_stage(decode_record_path, decode_records, "decode", rank)
        assert_rank_logits_match()
        if rank == 0:
            text = tok.decode(
                [next_id, decode_id],
                skip_special_tokens=True,
            ).strip()
            print(f"TP2_SWIMLANE_TEXT {text!r}", flush=True)
            print(COMM_MARKER, "swimlane_done", flush=True)
        _reset_cache(host)
        dist.barrier()
        dist.destroy_process_group()
        return 0

    prof_dir = SCRATCH / f"rank{rank}_torch"
    if prof_dir.exists():
        shutil.rmtree(prof_dir)
    active = 1 + args.decode_steps
    print(f"TP2_TORCH profile active={active} with_stack", flush=True)
    with _create_torch_profiler(prof_dir, active=active) as prof:
        logits = host.prefill(input_ids.to(device))
        torch.npu.synchronize()
        prof.step()
        ids = [select_rank0_token(logits)]
        seq = ntok + 1
        for _ in range(args.decode_steps):
            logits = host.decode(torch.tensor([ids[-1]], dtype=torch.int32, device=device), seq)
            torch.npu.synchronize()
            prof.step()
            ids.append(select_rank0_token(logits))
            seq += 1
        prof.step()
    assert_rank_logits_match()
    rank_csvs, rank_traces = _require_torch_profile_files(prof_dir)
    _assert_torch_whole_graph_steps(prof_dir, active)
    print(
        f"TORCH_PROFILE_ARTIFACTS rank={rank} kernel_tables={len(rank_csvs)} traces={len(rank_traces)}",
        flush=True,
    )
    dist.barrier()
    if rank == 0:
        dest = TORCH_DIR / "prof"
        _remove_path(dest)
        shutil.copytree(prof_dir, dest)
        _plot_torch_png(dest, TORCH_DIR / "top_kernels.png")
        print(f"TORCH_OUTPUT {tok.decode(ids, skip_special_tokens=True).strip()!r}", flush=True)
        print(f"TORCH_PROF_DIR {dest}", flush=True)
        print(COMM_MARKER, "torch_profile_done", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
