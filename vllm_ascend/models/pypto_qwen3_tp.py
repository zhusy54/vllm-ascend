#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""TP=2 helpers: contract weight split and Gloo+SHMEM pypto allreduce.

The published Qwen3-14B fused hosts are full-width. Each rank keeps a
Megatron-style shard; ranks reconstruct a full compute bundle, then run
the existing hosts. TP reductions on the generate path go through the
published pypto two-rank allreduce over torch_npu symmetric memory, with
Gloo as the host process group — not HCCL/HCCP.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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

# Column-parallel contract tensors: split the output (last) dim.
_COLUMN_FIELDS = ("wq", "wk", "wv", "w_gate", "w_up")
# Row-parallel contract tensors: split the reduced (first) dim.
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
    """Take this rank's contiguous slice of dim 0."""
    if world < 1:
        raise ValueError(f"world must be >= 1, got {world}")
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} out of range for world={world}")
    size = int(tensor.shape[0])
    if size % world != 0:
        raise ValueError(f"dim0 {size} is not divisible by world {world}")
    width = size // world
    return tensor.narrow(0, rank * width, width).contiguous()


def shard_contract_bundle(
    bundle: PyptoQwen3WeightBundle,
    rank: int,
    world: int = TP_WORLD,
) -> PyptoQwen3WeightBundle:
    """Megatron-style split of a packed Qwen3-14B contract bundle."""
    fields: dict[str, torch.Tensor] = {}
    for name, tensor in bundle.__dict__.items():
        if name in _COLUMN_FIELDS:
            fields[name] = shard_last_dim(tensor, rank, world)
        elif name in _ROW_FIELDS:
            fields[name] = shard_first_dim(tensor, rank, world)
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
        PyptoQwen3WeightBundle(**{name: gathered_by_name[name][src] for name in shard.__dict__})
        for src in range(world)
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
            fields[name] = torch.cat(pieces, dim=0).contiguous()
        else:
            fields[name] = pieces[0].contiguous()
    del world
    return PyptoQwen3WeightBundle(**fields)


def tp_allreduce_slot_plan(cols: int, dtype: torch.dtype) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Named SHMEM slots for a 2-rank pypto allreduce of ``[1, cols]``."""
    if cols < 1:
        raise ValueError(f"cols must be >= 1, got {cols}")
    elem = torch.empty((), dtype=dtype).element_size()
    payload = cols * elem
    return ("data_buf", "out_buf"), (payload, payload)


def _compile_allreduce_kernels(cols: int, platform: str = "a2a3") -> tuple[Any, Any]:
    import pypto.language as pl
    import pypto.language.distributed as pld
    from pypto.runtime import RunConfig

    if cols != HIDDEN:
        raise ValueError(f"pypto TP allreduce is compiled for HIDDEN={HIDDEN}, got cols={cols}")

    @pl.jit.incore
    def publish_step(
        inp: pl.Tensor[[1, HIDDEN], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, HIDDEN], pl.FP32]],
    ):
        return pl.store(pl.load(inp, [0, 0], [1, HIDDEN]), [0, 0], data)

    @pl.jit
    def publish_chip(
        inp: pl.Tensor[[1, HIDDEN], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, HIDDEN], pl.FP32]],
    ):
        return publish_step(inp, data)

    @pl.jit.incore
    def allreduce_step(
        data: pld.DistributedTensor[[1, HIDDEN], pl.FP32],
        out: pl.Out[pl.Tensor[[1, HIDDEN], pl.FP32]],
    ):
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        peer = (my_rank + 1) % nranks
        local = pl.load(data, [0, 0], [1, HIDDEN])
        remote = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, HIDDEN])
        return pl.store(pl.add(local, remote), [0, 0], out)

    @pl.jit
    def allreduce_chip(
        data: pld.DistributedTensor[[1, HIDDEN], pl.FP32],
        out: pl.Out[pl.Tensor[[1, HIDDEN], pl.FP32]],
    ):
        return allreduce_step(data, out)

    meta = torch.empty((1, HIDDEN), dtype=torch.float32)
    cfg = RunConfig(platform=platform, device_id=0)
    return publish_chip.compile(meta, meta, config=cfg), allreduce_chip.compile(meta, meta, config=cfg)


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
    session.worker._run_chip(compiled.chip_callable, args, compiled.build_call_config())


@dataclass
class PyptoGlooShmemComm:
    """One Gloo world + SHMEM window + compiled pypto allreduce."""

    rank: int
    world_size: int
    window: Any
    session: PyptoChipSession
    publish: Any
    allreduce: Any
    cols: int
    group: Any

    def allreduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """In-place-style sum-allreduce of a rank-1/2 fp32 payload, via pypto."""
        import torch.distributed as dist

        flat = tensor.detach().float().reshape(1, -1).contiguous()
        if int(flat.shape[-1]) != self.cols:
            raise ValueError(f"allreduce width {flat.shape[-1]} != compiled cols {self.cols}")
        payload = torch.empty((1, self.cols), dtype=torch.float32, device=flat.device)
        payload.copy_(flat)
        data_dt = self.window.as_device_tensor("data_buf", (1, self.cols), torch.float32)
        out_dt = self.window.as_device_tensor("out_buf", (1, self.cols), torch.float32)
        out_torch = torch.zeros((1, self.cols), dtype=torch.float32, device=payload.device)
        _dispatch_chip(self.session, self.publish, payload, data_dt, int(self.window.device_ctx_ptr))
        dist.barrier(group=self.group)
        _dispatch_chip(
            self.session,
            self.allreduce,
            data_dt,
            wrap_torch_npu_ptr(out_torch) if out_torch.device.type != "cpu" else out_torch,
            int(self.window.device_ctx_ptr),
        )
        dist.barrier(group=self.group)
        print(COMM_MARKER, f"rank={self.rank} cols={self.cols}", flush=True)
        result = out_torch.to(device=tensor.device, dtype=tensor.dtype)
        return result.reshape(tensor.shape)


def build_gloo_shmem_comm(
    *,
    rank: int,
    world_size: int,
    device: str,
    cols: int = HIDDEN,
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
    names, nbytes = tp_allreduce_slot_plan(cols, torch.float32)
    window = acquire_gloo_shmem_window(
        rank=rank,
        world_size=world_size,
        device=device,
        slot_names=names,
        slot_nbytes=nbytes,
        group_name=group_name,
    )
    chip = session or PyptoChipSession()
    platform = os.environ.get("PTO_PLATFORM", "a2a3")
    publish, allreduce = _compile_allreduce_kernels(cols, platform=platform)
    return PyptoGlooShmemComm(
        rank=rank,
        world_size=world_size,
        window=window,
        session=chip,
        publish=publish,
        allreduce=allreduce,
        cols=cols,
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
