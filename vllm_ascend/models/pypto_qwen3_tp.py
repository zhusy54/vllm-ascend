#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""TP=2 helpers: Megatron shard of contract weights and Gloo+SHMEM pypto allreduce.

Each rank keeps its column/row shard on device. Compute uses those shards
(last-dim 2560 for ``wq``). pypto allreduce over torch_npu symmetric memory
reduces the full-width partials after ``o_proj`` and ``down_proj``. Gloo is
the host process group — not HCCL/HCCP.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm_ascend.models.pypto_qwen3_adapter import (
    HIDDEN,
    PyptoChipSession,
    PyptoQwen3WeightBundle,
    wrap_torch_npu_ptr,
)

COMM_MARKER = "PYPTO_QWEN3_COMM gloo+shmem+pypto_allreduce"
TP_WORLD = 2
TP_TOK_PAD = 32
TP_SIGNAL_ROWS = TP_WORLD
TP_SIGNAL_COLS = 1
STAGE_TP_PREFILL = "qwen3_14b.tp_prefill_fwd"
STAGE_TP_DECODE = "qwen3_14b.tp_decode_fwd"
TP_BOUNDARY_O_PROJ = "o_proj"
TP_BOUNDARY_DOWN_PROJ = "down_proj"

# Column-parallel contract tensors: split the output (last) dim.
_COLUMN_FIELDS = ("wq", "wk", "wv", "w_gate", "w_up")
# Row-parallel: split each layer's reduced (K) dim, not the stacked layer axis.
_ROW_FIELDS = ("wo", "w_down")


def shard_last_dim(tensor: torch.Tensor, rank: int, world: int = TP_WORLD) -> torch.Tensor:
    """Take this rank's contiguous slice of the last dimension."""
    if world < 1:
        raise ValueError(f"world must be >= 1, got {world}")
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} out of range for world={world}")
    size = int(tensor.shape[-1])
    if size % world != 0:
        raise ValueError(f"last dim {size} is not divisible by world {world}")
    width = size // world
    return tensor.narrow(-1, rank * width, width).contiguous()


def shard_first_dim(tensor: torch.Tensor, rank: int, world: int = TP_WORLD) -> torch.Tensor:
    """Take this rank's contiguous slice of dim 0 (not for stacked layer weights)."""
    if world < 1:
        raise ValueError(f"world must be >= 1, got {world}")
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} out of range for world={world}")
    size = int(tensor.shape[0])
    if size % world != 0:
        raise ValueError(f"dim0 {size} is not divisible by world {world}")
    width = size // world
    return tensor.narrow(0, rank * width, width).contiguous()


def shard_stacked_row_parallel(
    tensor: torch.Tensor,
    rank: int,
    world: int,
    *,
    rows_per_layer: int,
) -> torch.Tensor:
    """Split each layer's K dim. ``tensor`` is ``[L * rows_per_layer, out]``."""
    if world < 1:
        raise ValueError(f"world must be >= 1, got {world}")
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} out of range for world={world}")
    if rows_per_layer < 1 or rows_per_layer % world != 0:
        raise ValueError(f"rows_per_layer {rows_per_layer} not divisible by world {world}")
    if tensor.ndim != 2:
        raise ValueError(f"expected rank-2 stacked weight, got {tuple(tensor.shape)}")
    if tensor.shape[0] % rows_per_layer != 0:
        raise ValueError(f"stacked dim0 {tensor.shape[0]} is not a multiple of rows_per_layer {rows_per_layer}")
    num_layers = int(tensor.shape[0]) // rows_per_layer
    layered = tensor.reshape(num_layers, rows_per_layer, tensor.shape[-1])
    width = rows_per_layer // world
    return layered[:, rank * width : (rank + 1) * width].reshape(num_layers * width, tensor.shape[-1]).contiguous()


def _bundle_num_layers(bundle: PyptoQwen3WeightBundle) -> int:
    """``input_rms_weight`` is stacked ``[L, H]`` by ``prepare_qwen3_weights``."""
    return int(bundle.input_rms_weight.shape[0])


def select_compute_bundle(
    bundle: PyptoQwen3WeightBundle,
    rank: int,
    world: int = TP_WORLD,
) -> PyptoQwen3WeightBundle:
    """Return the on-device Megatron shard used for compute (no gather)."""
    if world == 1:
        return bundle
    return shard_contract_bundle(bundle, rank, world)


def describe_tp_compute_args(bundle: PyptoQwen3WeightBundle) -> str:
    """``PYPTO_QWEN3_ARGS`` payload naming the sharded linear layouts."""
    return (
        f"wq={tuple(int(d) for d in bundle.wq.shape)} "
        f"wk={tuple(int(d) for d in bundle.wk.shape)} "
        f"wv={tuple(int(d) for d in bundle.wv.shape)} "
        f"wo={tuple(int(d) for d in bundle.wo.shape)} "
        f"w_gate={tuple(int(d) for d in bundle.w_gate.shape)} "
        f"w_up={tuple(int(d) for d in bundle.w_up.shape)} "
        f"w_down={tuple(int(d) for d in bundle.w_down.shape)}"
    )


def layer_stacked_view(tensor: torch.Tensor, layer: int, num_layers: int) -> torch.Tensor:
    """Slice one layer out of a dim-0 stacked contract tensor."""
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if not 0 <= layer < num_layers:
        raise ValueError(f"layer {layer} out of range for num_layers={num_layers}")
    if tensor.shape[0] % num_layers != 0:
        raise ValueError(f"stacked dim0 {tensor.shape[0]} is not divisible by num_layers {num_layers}")
    rows = int(tensor.shape[0]) // num_layers
    return tensor.narrow(0, layer * rows, rows)


def shard_contract_bundle(
    bundle: PyptoQwen3WeightBundle,
    rank: int,
    world: int = TP_WORLD,
) -> PyptoQwen3WeightBundle:
    """Megatron-style split of a packed Qwen3-14B contract bundle."""
    num_layers = _bundle_num_layers(bundle)
    fields: dict[str, torch.Tensor] = {}
    for name, tensor in bundle.__dict__.items():
        if name in _COLUMN_FIELDS:
            fields[name] = shard_last_dim(tensor, rank, world)
        elif name in _ROW_FIELDS:
            rows_per_layer = int(tensor.shape[0]) // num_layers
            fields[name] = shard_stacked_row_parallel(tensor, rank, world, rows_per_layer=rows_per_layer)
        else:
            fields[name] = tensor.contiguous()
    return PyptoQwen3WeightBundle(**fields)


def gloo_gather_contract_bundle(
    shard: PyptoQwen3WeightBundle,
    *,
    rank: int,
    world: int,
    group: Any,
) -> PyptoQwen3WeightBundle:
    """All-gather TP shards over a Gloo group and concat to a full bundle."""
    import torch.distributed as dist

    del rank
    gathered_by_name: dict[str, list[torch.Tensor]] = {}
    for name, tensor in shard.__dict__.items():
        cpu = tensor.detach().contiguous().cpu()
        bufs = [torch.empty_like(cpu) for _ in range(world)]
        dist.all_gather(bufs, cpu, group=group)
        gathered_by_name[name] = bufs
    shards = [
        PyptoQwen3WeightBundle(**{name: gathered_by_name[name][src] for name in shard.__dict__}) for src in range(world)
    ]
    return gather_contract_bundle(shards)


def gather_contract_bundle(
    shards: list[PyptoQwen3WeightBundle],
) -> PyptoQwen3WeightBundle:
    """Inverse of :func:`shard_contract_bundle` for ``world == len(shards)``."""
    if not shards:
        raise ValueError("shards is empty")
    world = len(shards)
    fields: dict[str, torch.Tensor] = {}
    for name in shards[0].__dict__:
        pieces = [getattr(shard, name) for shard in shards]
        if name in _COLUMN_FIELDS:
            fields[name] = torch.cat(pieces, dim=-1).contiguous()
        elif name in _ROW_FIELDS:
            num_layers = _bundle_num_layers(shards[0])
            width = int(pieces[0].shape[0]) // num_layers
            rows_per_layer = width * world
            layered = [part.reshape(num_layers, width, part.shape[-1]) for part in pieces]
            fields[name] = (
                torch.cat(layered, dim=1).reshape(num_layers * rows_per_layer, pieces[0].shape[-1]).contiguous()
            )
        else:
            fields[name] = pieces[0].contiguous()
    del world
    return PyptoQwen3WeightBundle(**fields)


def tp_allreduce_slot_plan(
    cols: int,
    dtype: torch.dtype,
    rows: int = 1,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Named SHMEM slots for a 2-rank pypto allreduce of ``[rows, cols]``."""
    if cols < 1:
        raise ValueError(f"cols must be >= 1, got {cols}")
    if rows < 1:
        raise ValueError(f"rows must be >= 1, got {rows}")
    elem = torch.empty((), dtype=dtype).element_size()
    payload = rows * cols * elem
    signal = TP_SIGNAL_ROWS * TP_SIGNAL_COLS * 4
    return ("data_buf", "signal"), (payload, signal)


def tp_compile_output_dir(program: str) -> str:
    """Give each torchrun rank/process its own PyPTO compiler output directory."""
    base = Path(os.environ.get("PYPTO_QWEN3_BUILD_DIR", "build_output"))
    rank = os.environ.get("LOCAL_RANK", "0")
    return str(base / f"{program}_rank{rank}_pid{os.getpid()}")


def _compile_allreduce_kernels(
    cols: int,
    rows: int = 1,
    platform: str = "a2a3",
) -> tuple[Any, Any]:
    import pypto.language as pl
    import pypto.language.distributed as pld
    from pypto.runtime import RunConfig

    if cols != HIDDEN:
        raise ValueError(f"pypto TP allreduce is compiled for HIDDEN={HIDDEN}, got cols={cols}")
    if rows != TP_TOK_PAD:
        raise ValueError(f"pypto TP allreduce is compiled for rows={TP_TOK_PAD}, got rows={rows}")

    @pl.jit.incore
    def publish_step(
        inp: pl.Tensor[[TP_TOK_PAD, HIDDEN], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[TP_TOK_PAD, HIDDEN], pl.FP32]],
    ):
        for t in pl.range(TP_TOK_PAD):
            data = pl.store(pl.load(inp, [t, 0], [1, HIDDEN]), [t, 0], data)
        return data

    @pl.jit
    def publish_chip(
        inp: pl.Tensor[[TP_TOK_PAD, HIDDEN], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[TP_TOK_PAD, HIDDEN], pl.FP32]],
    ):
        return publish_step(inp, data)

    @pl.jit.incore
    def allreduce_step(
        data: pld.DistributedTensor[[TP_TOK_PAD, HIDDEN], pl.FP32],
        out: pl.InOut[pl.Tensor[[TP_TOK_PAD, HIDDEN], pl.FP32]],
    ):
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        peer = (my_rank + 1) % nranks
        for t in pl.range(TP_TOK_PAD):
            local = pl.load(data, [t, 0], [1, HIDDEN])
            remote = pld.tile.remote_load(data, peer=peer, offsets=[t, 0], shape=[1, HIDDEN])
            out = pl.store(pl.add(local, remote), [t, 0], out)
        return out

    @pl.jit
    def allreduce_chip(
        data: pld.DistributedTensor[[TP_TOK_PAD, HIDDEN], pl.FP32],
        out: pl.InOut[pl.Tensor[[TP_TOK_PAD, HIDDEN], pl.FP32]],
    ):
        return allreduce_step(data, out)

    meta = torch.empty((TP_TOK_PAD, HIDDEN), dtype=torch.float32)
    publish_cfg = RunConfig(
        platform=platform,
        device_id=0,
        save_kernels_dir=tp_compile_output_dir("tp_publish_chip"),
    )
    allreduce_cfg = RunConfig(
        platform=platform,
        device_id=0,
        save_kernels_dir=tp_compile_output_dir("tp_allreduce_chip"),
    )
    return (
        publish_chip.compile(meta, meta, config=publish_cfg),
        allreduce_chip.compile(meta, meta, config=allreduce_cfg),
    )


def _compile_fused_allreduce(platform: str = "a2a3") -> Any:
    """One chip: store local partial, device-side allreduce, write the sum."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from pypto.runtime import RunConfig

    @pl.jit.incore
    def fused_allreduce_step(
        inp: pl.Tensor[[TP_TOK_PAD, HIDDEN], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[TP_TOK_PAD, HIDDEN], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[TP_WORLD, 1], pl.INT32]],
        out: pl.InOut[pl.Tensor[[TP_TOK_PAD, HIDDEN], pl.FP32]],
        expected_publish: pl.Scalar[pl.INDEX],
        expected_consume: pl.Scalar[pl.INDEX],
    ):
        for t in pl.range(TP_TOK_PAD):
            data = pl.store(pl.load(inp, [t, 0], [1, HIDDEN]), [t, 0], data)
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        peer = (my_rank + 1) % nranks
        pld.system.notify(
            signal,
            peer=peer,
            offsets=[my_rank, 0],
            value=1,
            op=pld.NotifyOp.AtomicAdd,
        )
        pld.system.wait(
            signal=signal,
            offsets=[peer, 0],
            expected=expected_publish,
            cmp=pld.WaitCmp.Ge,
        )
        for t in pl.range(TP_TOK_PAD):
            local = pl.load(data, [t, 0], [1, HIDDEN])
            remote = pld.tile.remote_load(data, peer=peer, offsets=[t, 0], shape=[1, HIDDEN])
            out = pl.store(pl.add(local, remote), [t, 0], out)
        pld.system.notify(
            signal,
            peer=peer,
            offsets=[my_rank, 0],
            value=1,
            op=pld.NotifyOp.AtomicAdd,
        )
        pld.system.wait(
            signal=signal,
            offsets=[peer, 0],
            expected=expected_consume,
            cmp=pld.WaitCmp.Ge,
        )
        return out

    def fused_allreduce_chip(
        inp: pl.Tensor[[TP_TOK_PAD, HIDDEN], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[TP_TOK_PAD, HIDDEN], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[TP_WORLD, 1], pl.INT32]],
        out: pl.InOut[pl.Tensor[[TP_TOK_PAD, HIDDEN], pl.FP32]],
        expected_publish: pl.Scalar[pl.INDEX],
        expected_consume: pl.Scalar[pl.INDEX],
    ):
        return fused_allreduce_step(
            inp,
            data,
            signal,
            out,
            expected_publish,
            expected_consume,
        )

    # This module deliberately imports PyPTO lazily.  With postponed Python
    # annotations, ``pl``/``pld`` therefore are not in the function globals
    # that signature-mode compilation consults.  Resolve the public signature
    # before decorating so the two credits can remain true runtime scalars.
    fused_allreduce_chip.__annotations__ = {
        "inp": pl.Tensor[[TP_TOK_PAD, HIDDEN], pl.FP32],
        "data": pl.InOut[pld.DistributedTensor[[TP_TOK_PAD, HIDDEN], pl.FP32]],
        "signal": pl.InOut[pld.DistributedTensor[[TP_WORLD, 1], pl.INT32]],
        "out": pl.InOut[pl.Tensor[[TP_TOK_PAD, HIDDEN], pl.FP32]],
        "expected_publish": pl.Scalar[pl.INDEX],
        "expected_consume": pl.Scalar[pl.INDEX],
    }
    fused_allreduce_chip = pl.jit(fused_allreduce_chip)

    cfg = RunConfig(
        platform=platform,
        device_id=0,
        save_kernels_dir=tp_compile_output_dir("tp_fused_allreduce_chip"),
    )
    return fused_allreduce_chip.compile(
        expected_publish=pl.RUNTIME,
        expected_consume=pl.RUNTIME,
        config=cfg,
    )


def _dispatch_chip(session: PyptoChipSession, compiled: Any, *values: Any) -> None:
    from pypto.runtime.device_tensor import DeviceTensor
    from pypto.runtime.task_interface import ChipStorageTaskArgs, device_tensor_to_tensor, make_tensor_arg

    args = ChipStorageTaskArgs()
    tensors = [val for val in values if not isinstance(val, int)]
    scalars = [val for val in values if isinstance(val, int)]
    for val in tensors:
        if isinstance(val, DeviceTensor):
            args.add_tensor(device_tensor_to_tensor(val))
        else:
            host = val.detach().contiguous()
            if host.device.type != "cpu":
                host = host.cpu()
            args.add_tensor(make_tensor_arg(host))
    for val in scalars:
        args.add_scalar(int(val))
    dfx_dir = None
    if session.config.any_dfx_enabled():
        dfx_dir = Path(compiled.output_dir) / "dfx_outputs"
        dfx_dir.mkdir(parents=True, exist_ok=True)
    cfg = compiled.build_call_config(session.config, dfx_dir=dfx_dir)
    # torch_npu and ChipWorker own different execution streams.  Finish all
    # pointer-backed input writes before handing ownership to the compiled
    # graph, then make its output visible before Python reuses those buffers.
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    session.worker._run_chip(compiled.chip_callable, args, cfg)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    session.capture_dfx(compiled)


@dataclass
class PyptoGlooShmemComm:
    """One Gloo world + SHMEM window + compiled pypto allreduce."""

    rank: int
    world_size: int
    window: Any
    session: PyptoChipSession
    publish: Any
    allreduce: Any
    fused: Any
    cols: int
    rows: int
    group: Any
    signal_credit: int = 0

    def reserve_signal_credits(self, count: int) -> int:
        """Reserve monotonically increasing INT32 signal credits and return the old base."""
        if count < 1:
            raise ValueError(f"signal credit count must be positive, got {count}")
        next_credit = self.signal_credit + count
        if next_credit > torch.iinfo(torch.int32).max:
            raise OverflowError(f"signal credit would overflow INT32: base={self.signal_credit}, count={count}")
        base = self.signal_credit
        self.signal_credit = next_credit
        return base

    def allreduce_sum(self, tensor: torch.Tensor, *, boundary: str = "hidden") -> torch.Tensor:
        """Sum-allreduce of ``[..., HIDDEN]`` via pypto. Pads the token axis to ``rows``."""
        import torch.distributed as dist

        if tensor.shape[-1] != self.cols:
            raise ValueError(f"allreduce width {tensor.shape[-1]} != compiled cols {self.cols}")
        live = tensor.detach().float().reshape(-1, self.cols).contiguous()
        n_rows = int(live.shape[0])
        if n_rows > self.rows:
            chunks = [
                self.allreduce_sum(live[start : start + self.rows], boundary=boundary)
                for start in range(0, n_rows, self.rows)
            ]
            return torch.cat(chunks, dim=0).reshape(tensor.shape).to(device=tensor.device, dtype=tensor.dtype)
        payload = torch.zeros((self.rows, self.cols), dtype=torch.float32, device=live.device)
        payload[:n_rows].copy_(live)
        data_dt = self.window.as_device_tensor("data_buf", (self.rows, self.cols), torch.float32)
        out_torch = torch.zeros((self.rows, self.cols), dtype=torch.float32, device=payload.device)
        prev_tag = self.session.launch_tag
        safe = boundary.replace(":", "_")
        payload_arg = wrap_torch_npu_ptr(payload) if payload.device.type != "cpu" else payload
        try:
            self.session.launch_tag = f"{safe}.publish"
            _dispatch_chip(
                self.session,
                self.publish,
                payload_arg,
                data_dt,
                int(self.window.device_ctx_ptr),
            )
            dist.barrier(group=self.group)
            self.session.launch_tag = f"{safe}.allreduce"
            _dispatch_chip(
                self.session,
                self.allreduce,
                data_dt,
                wrap_torch_npu_ptr(out_torch) if out_torch.device.type != "cpu" else out_torch,
                int(self.window.device_ctx_ptr),
            )
        finally:
            self.session.launch_tag = prev_tag
        dist.barrier(group=self.group)
        print(
            COMM_MARKER,
            f"boundary={boundary} rank={self.rank} rows={n_rows} cols={self.cols}",
            flush=True,
        )
        result = out_torch[:n_rows].to(device=tensor.device, dtype=tensor.dtype)
        return result.reshape(tensor.shape)

    def allreduce_sum_fused(self, tensor: torch.Tensor, *, boundary: str = "hidden") -> torch.Tensor:
        """Sum-allreduce in one ChipWorker.run. No host barrier between publish and reduce."""
        if tensor.shape[-1] != self.cols:
            raise ValueError(f"allreduce width {tensor.shape[-1]} != compiled cols {self.cols}")
        live = tensor.detach().float().reshape(-1, self.cols).contiguous()
        n_rows = int(live.shape[0])
        if n_rows > self.rows:
            chunks = [
                self.allreduce_sum_fused(live[start : start + self.rows], boundary=boundary)
                for start in range(0, n_rows, self.rows)
            ]
            return torch.cat(chunks, dim=0).reshape(tensor.shape).to(device=tensor.device, dtype=tensor.dtype)
        payload = torch.zeros((self.rows, self.cols), dtype=torch.float32, device=live.device)
        payload[:n_rows].copy_(live)
        data_dt = self.window.as_device_tensor("data_buf", (self.rows, self.cols), torch.float32)
        signal_dt = self.window.as_device_tensor("signal", (TP_WORLD, 1), torch.int32)
        out_torch = torch.zeros((self.rows, self.cols), dtype=torch.float32, device=payload.device)
        prev_tag = self.session.launch_tag
        self.session.launch_tag = f"{boundary.replace(':', '_')}.fused_allreduce"
        ctx = int(self.window.device_ctx_ptr)
        credit_base = self.reserve_signal_credits(2)
        try:
            _dispatch_chip(
                self.session,
                self.fused,
                wrap_torch_npu_ptr(payload) if payload.device.type != "cpu" else payload,
                data_dt,
                signal_dt,
                wrap_torch_npu_ptr(out_torch) if out_torch.device.type != "cpu" else out_torch,
                credit_base + 1,
                credit_base + 2,
                ctx,
                ctx,
            )
        finally:
            self.session.launch_tag = prev_tag
        print(
            COMM_MARKER,
            f"fused boundary={boundary} rank={self.rank} rows={n_rows} cols={self.cols}",
            flush=True,
        )
        result = out_torch[:n_rows].to(device=tensor.device, dtype=tensor.dtype)
        return result.reshape(tensor.shape)


def _zero_window_slot(window: Any, slot: str, nbytes: int) -> None:
    """Zero exactly one carved SHMEM slot before device-side credit waits begin."""
    if nbytes < 1:
        raise ValueError(f"slot nbytes must be positive, got {nbytes}")
    if slot not in window.offsets:
        raise KeyError(f"unknown SHMEM slot {slot!r}; have {sorted(window.offsets)}")
    offset = int(window.offsets[slot])
    window.tensor[offset : offset + nbytes].zero_()


def build_gloo_shmem_comm(
    *,
    rank: int,
    world_size: int,
    device: str,
    cols: int = HIDDEN,
    rows: int = TP_TOK_PAD,
    group: Any | None = None,
    session: PyptoChipSession | None = None,
) -> PyptoGlooShmemComm:
    """Rendezvous a SHMEM window on a Gloo group and compile the allreduce host."""
    import torch.distributed as dist
    from pypto.runtime.shmem_gloo import acquire_gloo_shmem_window

    if world_size != TP_WORLD:
        raise ValueError(f"pypto TP allreduce is 2-rank only, got world_size={world_size}")
    if group is None:
        group = dist.group.WORLD
    group_name = group.group_name
    names, nbytes = tp_allreduce_slot_plan(cols, torch.float32, rows=rows)
    window = acquire_gloo_shmem_window(
        rank=rank,
        world_size=world_size,
        device=device,
        slot_names=names,
        slot_nbytes=nbytes,
        group_name=group_name,
    )
    signal_nbytes = nbytes[names.index("signal")]
    _zero_window_slot(window, "signal", signal_nbytes)
    torch.npu.synchronize()
    dist.barrier(group=group)
    chip = session or PyptoChipSession()
    platform = os.environ.get("PTO_PLATFORM", "a2a3")
    publish, allreduce = _compile_allreduce_kernels(cols, rows=rows, platform=platform)
    fused = _compile_fused_allreduce(platform=platform)
    return PyptoGlooShmemComm(
        rank=rank,
        world_size=world_size,
        window=window,
        session=chip,
        publish=publish,
        allreduce=allreduce,
        fused=fused,
        cols=cols,
        rows=rows,
        group=group,
    )


def tp_rank_and_world() -> tuple[int, int]:
    """Read vLLM TP rank/world; fall back to single rank if PG is down."""
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return int(get_tensor_model_parallel_rank()), int(get_tensor_model_parallel_world_size())
    except Exception:
        return 0, 1
