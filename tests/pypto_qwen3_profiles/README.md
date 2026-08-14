# Qwen3-14B PyPTO profile 采集

TP=2 采集器使用 40 层整网 fused host。prefill 调用
`tp_prefill_fwd` 一次，decode 调用 `tp_decode_fwd` 一次；每个 stage 的
L2 原始记录必须恰好只有一份，而且其中必须包含整图的 643 个
`aicore_tasks`（AIV 402、AIC 241），否则脚本会直接失败。rank 0 和 rank 1 都执行此校验，
最终可视化产物保存 rank 0 的记录。

泳道与 torch profiler 必须使用两个独立进程生命周期，不能在同一次
torchrun 中串行采集：

```bash
source /mnt/workspace/inductor/env.sh
source /mnt/workspace/inductor/shmem/install/set_env.sh
cd /mnt/workspace/inductor/vllm-ascend

.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/pypto_qwen3_profiles/collect_qwen3_tp2_profiles.py --mode swimlane

# 上一条 torchrun 完全退出后，再启动新进程：
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/pypto_qwen3_profiles/collect_qwen3_tp2_profiles.py --mode torch \
  --decode-steps 4
```

泳道模式会输出以下验收标记（rank 0/1 各一行）：

```text
PYPTO_QWEN3_PROFILE_STAGE tp_prefill_fwd ChipWorker.run=1 aicore_tasks=<N> rank=0
PYPTO_QWEN3_PROFILE_STAGE tp_decode_fwd ChipWorker.run=1 aicore_tasks=<N> rank=0
```

实际 prefill 标记也会带 `aicore_tasks=<N>`。PNG 使用上下两级视图：上方保留
raw records 中每条 AICore task 的真实物理 core 与整图 wall-clock 时间轴；主图把
40 层逐行展开，每一行从该层起点重新计时并共用同一毫秒刻度，因此 640 个层内
task 的实测时长都可辨认，未用最小显示宽度夸大短 task。下方等宽色块只说明每层
16 个 task 的顺序与 AIC/AIV 类型，不表示时长。Perfetto trace 仍保留可交互的原始
物理 core 时间线。torch 模式要求两个 rank 都同时生成非空的
`kernel_details.csv` 和 `trace_view.json`，并确认 `aicore_kernel_0` 整图调用数等于
1 次 prefill 加配置的 decode 次数；任一缺失都会失败，不会沿用旧产物。
两个模式都会广播 rank 0 的 greedy token 作为下一步输入，同时硬校验
两 rank 本地 argmax 完全一致。

产物仍写到：

- `tp2/swimlane/{prefill,decode}.{json,png}`
- `tp2/swimlane/{prefill,decode}_trace.json`
- `tp2/torch/prof/`
- `tp2/torch/top_kernels.png`

`tp1/`、`scratch/` 等历史或临时生成物由 `.gitignore` 排除；本次经过校验的
`tp2/` 泳道、Perfetto trace 和 torch profiler 产物保留在目录中，随代码一起
交付。重新采集会先清理对应 stage 的旧产物，避免新旧数据混用。
原始 torch profiler 包含采集机的 hostname、PID 和绝对路径等运行时元数据；
本次按要求一并保留，若对外公开需先确认是否要脱敏。

`collect_qwen3_profiles.py` 是 TP=1 fused host 的独立采集器。

## 2026-08-14 真机结果

- L2 prefill：rank0/rank1 均为 `ChipWorker.run=1`、`aicore_tasks=643`；
  rank0 任务级泳道 wall time 328.098 ms。
- L2 decode：rank0/rank1 均为 `ChipWorker.run=1`、`aicore_tasks=643`；
  rank0 任务级泳道 wall time 329.974 ms。
- 独立 torch profiler：1 次 prefill + 4 次 decode；两个 rank 均产生非空
  `kernel_details.csv` 和 `trace_view.json`，rank0 汇总表 47 行并成功生成
  `top_kernels.png`。
- 两个 torchrun 均 `EXIT=0`。上述数字来自当前机器的一次真机采集，不应当作
  跨机器性能基线；结构性验收项是每阶段单次 capture 和 643 个整图 task。
