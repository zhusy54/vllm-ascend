# DeepSeek V4 Decode CSA PyPTO L1 验证指南

本目录是 DeepSeek V4 Flash decode CSA 的公共测试支持目录。正式 kernel、
adapter 和 production dispatch 位于
`vllm_ascend/ops/_pypto_dsv4_csa/`；本目录只保存 fixture、Host contract、
A3 runner、结果 schema 和验证说明。`pypto-lib` 只作为历史算法参考，运行时
不会导入它。

本文档描述当前工作树能够真实执行的命令，并刻意区分 Host 证据、A3 功能
证据和仍未完成的计划项。它不是完整 DeepSeek V4 模型部署说明。

## 1. 当前固定环境

本轮记录的仓库基线是：

| 仓库 | commit |
| --- | --- |
| vLLM-Ascend | `e7cb166290dfcbf2f997aa67f01b323be643fe0e` |
| vLLM | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| PyPTO | `9cece0b730a96fe1a52c2637537132f524ffe1ea` |
| simpler | `b6f905f63277597bd2547d672fd9d57b6013fca9` |
| pypto-lib（只读参考） | `0073f4228811eae687f9049b417be803c75c49e1` |

目标 Python 环境是
`/mnt/workspace/inductor/vllm-ascend/.venv`，Torch/torch_npu 主版本为
2.12。A3 测试固定使用逻辑 `device0`；命令行统一写成 `--device 0`。

进入环境：

```bash
cd /mnt/workspace/inductor/vllm-ascend
source .venv/bin/activate
unset PTOAS_ROOT
export PATH=/mnt/workspace/inductor/pto/PTOAS/build-v0.57-llvm21-cann9.2-clean/tools/ptoas:$PATH
export LD_LIBRARY_PATH=/mnt/workspace/inductor/toolchains/gcc15/lib:${LD_LIBRARY_PATH:-}
```

不要把系统 `libstdc++` 替换掉。上述 `LD_LIBRARY_PATH` 只作用于当前测试
shell/子进程。

## 2. 只读 Host 环境检查

在任何 A3 命令之前运行：

```bash
python -m tests.pypto_dsv4_decode_csa.check_environment --device 0
```

检查器只使用标准库完成以下只读操作：

- 读取 Python 版本、venv prefix 和 distribution metadata；
- 用 `find_spec()` 定位模块但不导入 `torch`、`torch_npu`、PyPTO、simpler
  或 vLLM；
- 用 `git -C <repo> rev-parse HEAD` 读取五个仓库的 commit；
- 检查固定 PTOAS 是否可执行、`PATH` 是否选中它、`PTOAS_ROOT` 是否已清空；
- 检查 GCC 15 runtime 是否位于当前 `LD_LIBRARY_PATH`；
- 只记录本轮约定的逻辑 device 为 0。

检查器不会调用 `npu-smi`、不会导入 `torch_npu`、不会打开 NPU context、
不会编译 kernel，也不会写文件或修改系统。输出是 versioned JSON；任一必需项
失败时进程退出码为 1。`npu_probe_performed=false` 只表示检查器没有碰 NPU，
不表示设备健康或空闲。

## 3. Host UT

定向运行环境检查与交付文档契约：

```bash
python -m pytest -q \
  tests/pypto_dsv4_decode_csa/test_environment_check.py
```

运行全部 CSA Host suite：

```bash
python -m pytest -q tests/pypto_dsv4_decode_csa
python -m ruff check \
  vllm_ascend/ops/_pypto_dsv4_csa \
  tests/pypto_dsv4_decode_csa
python -m ruff format --check \
  vllm_ascend/ops/_pypto_dsv4_csa \
  tests/pypto_dsv4_decode_csa
```

Host UT 不初始化 NPU，因此不能代替任何 A3 数值、ACLGraph 或性能验收。

## 4. TRB/HBG 必须使用 fresh process

TRB 与 HBG 的 binary、context、function handle 和 captured graph owner 有
process-pinned 生命周期。两种 runtime 的正式结果必须分别来自独立的新 Python 进程；
不能在一个 Python interpreter 中先跑 TRB、`close()` 后再切换 HBG。

下面每一条 `python -m ...` 都是一个独立进程。不要把 runner import 到同一个
长驻 Python 脚本里循环切换 `runtime`。

### 4.1 结构与 page-stride zero smoke

TRB：

```bash
python -m tests.pypto_dsv4_decode_csa.subprocess_runner \
  --runtime tensormap_and_ringbuffer \
  --device 0 \
  --batch 4 \
  --artifact-dir /tmp/dsv4-csa-trb-artifacts \
  --result-json /tmp/dsv4-csa-trb-result.json
```

退出后再启动全新的 HBG 进程：

```bash
python -m tests.pypto_dsv4_decode_csa.subprocess_runner \
  --runtime host_build_graph \
  --device 0 \
  --batch 4 \
  --artifact-dir /tmp/dsv4-csa-hbg-artifacts \
  --result-json /tmp/dsv4-csa-hbg-result.json
```

这条 smoke 使用正式尺寸但零 projection weight，只证明 ABI、page stride、
caller-stream、预分配 output 和 ACLGraph replay；不能证明非零算法精度。

### 4.2 当前 44-slot 非零 eager/native 对比

当前最强的连续 state 证据来自 B4/S8、TP1、ratio-4。TRB 四步命令：

```bash
python -m tests.pypto_dsv4_decode_csa.a3_native_compare \
  --runtime tensormap_and_ringbuffer \
  --device 0 \
  --batch 4 \
  --warmups 0 \
  --iterations 1 \
  --start-position 0 \
  --correctness-steps 4 \
  --master-port 29659
```

该命令中的 correctness steps 是连续普通 launch，中间不隐式同步。增加
`--sync-between-correctness-steps` 只用于诊断，不能作为默认正确性证据。

128-step 固定 request advance 长链命令：

```bash
python -m tests.pypto_dsv4_decode_csa.a3_native_compare \
  --runtime tensormap_and_ringbuffer \
  --device 0 \
  --batch 4 \
  --warmups 0 \
  --iterations 1 \
  --start-position 0 \
  --correctness-steps 128 \
  --master-port 29675
```

该路径保留 raw INT8 indexer K 的 bit-exact 报告，并用独立的
INT8-K/FP16-scale 联合量化契约判定稀疏量化边界。不会通过提高全局
`atol/rtol` 掩盖 raw 差异。这条命令不含 request retire/block reuse。

### 4.3 当前 44-slot 单 custom-op ACLGraph 对比

TRB：

```bash
python -m tests.pypto_dsv4_decode_csa.a3_native_aclgraph_compare \
  --runtime tensormap_and_ringbuffer \
  --device 0 \
  --batch 4 \
  --replay-values 0.5 -1.0 1.75 \
  --warmups 3 \
  --iterations 10 \
  --master-port 29661
```

退出后，使用不同端口启动全新的 HBG 进程：

```bash
python -m tests.pypto_dsv4_decode_csa.a3_native_aclgraph_compare \
  --runtime host_build_graph \
  --device 0 \
  --batch 4 \
  --replay-values 0.5 -1.0 1.75 \
  --warmups 3 \
  --iterations 10 \
  --master-port 29663
```

两张图各只捕获一个 production `torch.ops.vllm.dsa_forward`。runner 会比较
output 和六类 mutable state，并验证 warmup/capture 地址变化、双 state world
隔离和 captured binding owner。

### 4.4 四层非零 mixed-backend eager 对比

四层 runner 为 NNNN 端到端 reference、same-input native local oracle 和候选
路由分别持有独立 cache world。TRB 示例：

```bash
python -m tests.pypto_dsv4_decode_csa.a3_native_capsule_compare \
  --runtime tensormap_and_ringbuffer \
  --device 0 \
  --batch 4 \
  --num-layers 4 \
  --candidate-topology PPPP \
  --master-port 29701
```

`--candidate-topology` 还可分别取 `NPNP`、`PNPN`。HBG 必须在前一进程退出后
以 `--runtime host_build_graph` 和不同 `--master-port` 启动。当前已保存的
TRB/HBG 三拓扑结果逐层比较 output 与六类 state，并全部通过。

这里必须区分证据边界：当前命令仍报告
`deterministic_bridge_only=true`，层壳尚不是正式 HC-pre/RMSNorm/HC-post；
它是 nonzero 多层 DSA/mixed-backend 证据，不是完整 fidelity Capsule。

### 4.5 四层单图 callable/address 生命周期 smoke

以下 runner 捕获一张包含四个 production custom-op node 与 Torch bridge 的
ACLGraph，并支持同 callable 多地址 patch 和 distinct callable：

```bash
python -m tests.pypto_dsv4_decode_csa.a3_capsule_smoke \
  --runtime tensormap_and_ringbuffer \
  --device 0 \
  --batch 4 \
  --num-layers 4 \
  --topology PP_same \
  --replay-values 0.25 -0.5 1.0
```

`--topology` 可取 `PPPP`、`PP_same` 或 `PP_distinct`。每个 runtime/topology
都必须是 fresh process。该 runner 使用 zero projection weight，只证明 graph
node 顺序、callable identity、地址快照、owner 与 replay；不能替代上一节的
nonzero 数值证据。

### 4.6 Stateful 多 bucket ACLGraph trace

`a3_stateful_aclgraph_compare` 把 request admit/advance/retire/compact/block reuse
trace 接到非零 ACLGraph replay。B4/B8/B12/B16 是四个不同的静态
specialization，因此 runner 明确使用“一 bucket 一 graph”，四张 graph 共享
一个 PyPTO runtime owner 和一组六状态 cache；它不是、也不宣称是一张动态
shape graph。默认 32-step 会多次更新并复用每张 graph 的固定地址 metadata、
hidden 和 output buffer：

```bash
python -m tests.pypto_dsv4_decode_csa.a3_stateful_aclgraph_compare \
  --runtime tensormap_and_ringbuffer \
  --device 0 \
  --steps 32 \
  --master-port 29743
```

TRB 进程完全退出后，HBG 必须使用新进程和新端口：

```bash
python -m tests.pypto_dsv4_decode_csa.a3_stateful_aclgraph_compare \
  --runtime host_build_graph \
  --device 0 \
  --steps 32 \
  --master-port 29745
```

所有 trace payload 和 graph buffer 都在 capture 前分配并搬到 device。每步只在
对应 bucket stream 上依次入队新 block 初始化、metadata/hidden 的 D2D copy、
output canary fill 和 replay；随后由测试 caller 在 graph 外同步并检查 output、
六状态、step-local write-set、block-0/page-padding canary、partial-bucket padded
output 及地址稳定性。capture metadata 对象里的 Python `actual` 是 capture 时
校验快照；后续 replay 的 active row 只由固定地址 `kv_seq_lens` 和 block table
内容决定。Host UT 只证明这些 owner/routing/CLI 合同，不能替代上述 fresh-process
A3 命令的 TRB/HBG 结果。

## 5. 证据优先级与历史结果

当前 public L1 ABI 是 44 slots：40 个 tensor 加 4 个 runtime scalar，唯一
纯输出为 index 39。过程记录第 33 节的 B4/B8/B12/B16 和多 graph 结果来自
旧 45-slot zero program；后续删除 `swa_slot_mapping`，又增加了真实
`writeback TaskId -> qk_pv` 依赖并删除 `kv_touch`。

因此所有 45-slot 结果都只算历史结构证据，不能继续代表当前源码。其后已经用
最终 44-slot + writeback 依赖重新完成 B4/B8/B12/B16 的 TRB/HBG 四 graph
同时存活、交替 replay、局部 destroy 后 survivor replay。该新结果覆盖旧
45-slot 的多 bucket/multi-graph 生命周期结论，但使用 zero projection weight，
不能单独证明 B8/B12/B16 非零算法精度。后续已用当前源码在
fresh process 中补齐 B8/B12/B16 的 TRB/HBG 非零 eager native/PyPTO 对比。
两类证据需要合并解读：zero-weight graph 矩阵证明多图生命周期，
nonzero eager 矩阵证明四 bucket 的算法与六状态精度。

## 6. 当前可以与不能得出的结论

当前可以证明：

- B4/S8、TP1、ratio-4 的 production custom-op 路径能够调用 PyPTO L1；
- 当前 44-slot TRB/HBG 单 custom-op 均有非零 ACLGraph replay 与六状态对比记录；
- warmup 和 capture 可以使用不同 stream/地址，taskQueue 路径不会 silent
  fallback；
- 真实 A3 page-strided cache 与 FP16 indexer scale 能直接通过当前 ABI；
- TRB 四个连续 decode step 的 output 和六状态可以与 native 对齐。
- 最终 44-slot + writeback 依赖的 B4/B8/B12/B16，已经分别在 fresh-process
  TRB/HBG 中完成四 graph 同存、交替 replay、局部 destroy 和 survivor replay；
  这是 zero-weight 生命周期证据。
- partial bucket 已经分别在 fresh-process TRB/HBG 中通过严格 block-0/page-padding
  canary、padded output 清零和 graph-stable metadata address 检查；这是
  zero-weight active-row guard/写隔离证据。
- B8/B12/B16 已分别在 fresh-process TRB/HBG 中完成当前 44-slot 源码的
  nonzero eager native/PyPTO 对比，output 和六类 mutable state 全部通过；
  它们不替代非零 multi-graph ACLGraph 证据。
- 上述最终 44-slot 功能矩阵已固化在
  `results/20260903_device0_final44_functional_matrix.json`。
- B4 TRB 已完成 128 个连续普通 launch，中间无 sync；output 与非量化
  state 通过，indexer K 保留 4/131072 个 1-bin raw 差异，scale bit-exact，
  联合量化契约通过，首个差异映射到 decode step 92。
- 当前 44-slot 源码的 B4/B8/B12/B16 四张 nonzero stateful ACLGraph 已在
  fresh-process TRB/HBG 中各完成 32-step 重复 replay；partial bucket、地址
  稳定、request admit/retire/compact/block reuse、output、六类 state 与 canary
  检查均通过。
- 当所有 row 的可见压缩位置数都不超过 512 时，TRB 能由 device 上的
  `need_index_score` gate 跳过 6 个 score-only task；该判定不需要 Host
  读取 metadata，ACLGraph replay 时仍依据当次 device buffer 内容决定。
- 当前 TRB ACLGraph 已有 3 个 fresh process 的正式同卡稳态 ABBA 结果；
  详细口径与数据见下文。

当前不能证明：

- 完整 DeepSeek V4 模型、vLLM Engine、TP/EP/HCCL/MoE/server 已接入；
- 128-step request-churn/endurance trace 已在 A3 上正确；已通过的
  stateful multi-bucket 验收为 32-step，另一条 128-step 证据是固定 request
  连续 advance，两者都不能替代 128-step churn/reuse 验收；
- 完整 DecoderLayer 的四层 HC-pre/RMSNorm/HC-post 与第二段 HC/FFN/MoE
  已完成；当前已通过的四层 `PPPP/NPNP/PNPN` 仍是 production HC
  attention-half 加人工 bridge，不得外推为完整模型证据；
- destroy 后 recapture 的 nonzero multi-graph 数值验收已完成；当前已通过
  的 survivor replay 是 zero-weight 生命周期证据，不能替代这一项；
- 同一 device 并发 replay 安全；
- A5 或 simulator 可用。

### 6.1 正式稳态 benchmark 的 taskQueue 模式

普通 eager 的生产 OpAPI V2 路径必须在一个全新进程中使用：

```bash
TASK_QUEUE_ENABLE=2 python -m tests.pypto_dsv4_decode_csa.a3_single_op_benchmark \
  --runtime tensormap_and_ringbuffer \
  --mode eager \
  --device 0 \
  --batch 4 \
  --start-position 8191 \
  --warmups 20 \
  --samples 100
```

torch_npu 2.12 的 ACLGraph capture 路径必须另起全新进程并使用：

```bash
TASK_QUEUE_ENABLE=1 python -m tests.pypto_dsv4_decode_csa.a3_single_op_benchmark \
  --runtime tensormap_and_ringbuffer \
  --mode aclgraph \
  --device 0 \
  --batch 4 \
  --start-position 8191 \
  --warmups 20 \
  --samples 100
```

runner 会在导入 `torch/torch_npu`、占用 runtime owner 或访问 device0 之前
核对该环境变量；缺失或错配直接 fail-fast，不会在进程内偷偷改值。`=2`
不能用于本环境的 ACLGraph capture，`=1` 的普通 eager 数字也不能冒充
本轮要求的 OpAPI V2 dequeue 性能证据。TRB/HBG、eager/ACLGraph 均应分别
使用 fresh process；诊断 profiler 沿用所属 mode 的同一个 taskQueue 值。

### 6.2 历史性能阶段（保留作为优化证据）

HBG 曾从历史默认 16384-task window 收缩到本 CSA 在 device0 证明
容量足够的最小 2 次幂 window 128，当时同卡 ABBA 的 steady-state device
p50 从约 320 ms 降至 `6.423 ms`；native production 为 `0.633 ms`。约
49.7 倍改善证明旧瓶颈主要是超大 resident package 的 replay
restore/cache maintenance。这是引入 predicate 快路之前的历史定位证据，
不是当前 HBG 的正式性能验收结果。

TRB 历史基线已有 native production-overlap、同卡 ABBA、100 样本
p50/p90/p99 与 Host/device 拆栏。固定 24 个 qk_pv block 时 p50 为
`0.821 ms`；改为 runtime 实际 20 个 AIC 后为 `0.800 ms`，同轮 native
为 `0.645 ms`，当时仍慢约 24%。Host replay 双方都约 19 us。这些数字
保留用于说明 runtime 核数修正带来的收益，但已被下文的成熟上下文
三轮 fresh-process 结果取代，不再代表当前代码的最终性能。

### 6.3 当前 TRB ACLGraph 正式稳态结果

当前正式性能口径为 B4/S8、TP1、ratio-4、start-position 8191、device0、
TRB ACLGraph。这是会执行完整 index score 路径的成熟上下文，不依赖短上下文
`visible_len <= 512` 快路。
每轮都启动一个全新 Python process，在同卡、同 caller stream 上对 native
production（保持 multistream overlap）和 PyPTO 执行 ABBA 配对采样。每个
process 使用 20 次 benchmark warmup 和 100 个正式 sample，每 20 次 enqueue
由 caller 批量同步；p50/p90/p99 只从正式 sample 计算，不混入 warmup。
三轮的 device span 如下：

| fresh process | native p50/p90/p99 | PyPTO TRB p50/p90/p99 | paired D（native - PyPTO） |
| --- | ---: | ---: | ---: |
| run 1 | `727.460 / 734.704 / 737.949 us` | `705.870 / 718.912 / 730.880 us` | `+21.590 / +15.792 / +7.068 us` |
| run 2 | `724.700 / 732.510 / 736.001 us` | `711.400 / 724.110 / 732.607 us` | `+13.300 / +8.400 / +3.394 us` |
| run 3 | `712.550 / 718.186 / 724.581 us` | `700.280 / 709.086 / 717.236 us` | `+12.270 / +9.100 / +7.345 us` |
| 三轮中位配对差 | — | — | `+13.300 / +9.100 / +7.068 us` |

`D` 先在每个 fresh process 内按同一分位做 `native - PyPTO`，再对
3 个 process 的 `D` 取中位数；它不是将两组跨进程原始样本混合，
也不是“两个跨进程中位数再相减”。正值表示 PyPTO 更快。三个 fresh process
在三个分位点上全部为正；三轮中位配对差分别为
`+13.300/+9.100/+7.068 us`，满足 `D50 > 0`、`D90 >= 0`、`D99 >= 0`
的预先门槛，因此当前结论是“TRB ACLGraph 的 B4/S8/C8191 固定 binding
稳态验收通过”。这个结论不应外推为 B8/B12/B16 性能、动态 binding、HBG
性能或完整模型吞吐均已超过 native。

最终收益来自三项可独立审计的冷/热路径改造组合：`wo_b` 在 weight prepare
阶段重排成 `[G, D/N, K/BK, N, BK]` tile-major 物理顺序，使 PB 的
`[256, 256]` cube weight tile 使用连续 GM stride；projection group 按
`[6, 7, 4, 5, 2, 3, 0, 1]` 提交以适配实测的 sibling LIFO-like 调度和
merge `3→2→1→0` ready 顺序；indexer 将 `idx_qr_proj_dequant` 与 `qr_rope`
融合成一个 AIV child，删除约 1 MiB FP32 `qr_proj` GM 中间态和一个 task
barrier。前两项不改数学归约顺序，第三项保留 native BF16 rounding 边界。
代价是每 layer 冷准备并由 owner 额外 pin 一份约 64 MiB 的 tile-major `wo_b`；
这不进入稳态样本，但属于产品显存预算，不能忽略。

计时明确不包含首次 JIT/program 编译、PTOAS/codegen、binary 注册、物化与加载、
runtime/context/owner/callable 创建和 prepare、weight pack/preparation、ordinary eager、
ACLGraph 及正式 benchmark 的全部 warmup、ACLGraph capture/build、最终采样地址上的
首次 ordinary invocation/replay、一次性 structure/binding-cache 与 event/runtime-handle
初始化，以及 correctness golden 生成、执行、对比和 validation-only copy。上述工作必须
全部结束并由 caller quiesce 后才能进入 `SamplingPhase.SAMPLE`。计时包含 production
validation、tensor address/scalar patch、taskQueue producer enqueue 与 consumer dequeue/launch、
AICPU/AICore scheduler、kernel execution，以及 workload 反复触发的 binding-cache
replacement/miss。如果真实常态调用因 tensor 地址或 scalar
变化反复产生 address/scalar patch 或 cache miss，这些都必须留在稳态口径内，
不得追认为“首次成本”剔除。本轮 ACLGraph 采样固定 captured tensor/scalar
binding，因此没有额外伪造每次 replay 的地址变化；后续若用变地址/变
scalar workload 验收，其 patch/cache-miss 必须原样计入。

因此本节的 `device_span` 只证明固定 captured binding。动态地址/scalar 场景如果
存在 start event 之前的 Host patch，正式主指标改用整个稳态 enqueue batch 从首次
backend-specific per-call 工作到最终 caller-stream quiesce 的 critical-path 除以调用
数；Host enqueue 与 device span 仍单独报告，但不能相加。

当前 runner 尚未产出包含最终 caller-stream quiesce 的 eager
`steady_enqueue_batch_critical_path`；因此现有 eager 输出只用于归因诊断，不能据此形成正式
性能 PASS。当前 fixed-binding ACLGraph 的同 stream `device_span` 不受这个缺口影响。

Native 与 PyPTO 同轮采集的一套完整诊断 profiler 产物已汇聚到
`results/20260903_device0_native_vs_pypto_diagnostic_profile/`。其中
`operator_comparison_analysis.md` 给出算子级热点、双流 overlap 和 AICPU/AICore
envelope 的完整差距闭合，
`native_trace_view.json` 与 `pypto_trace_view.json` 是从完整 ABBA trace 无损拆出的两份
Perfetto JSON，`profiler_output/trace_view.json` 保留完整 timeline，
`profiler_output/kernel_details.csv` 用于 kernel 级拆分，目录内 README 记录测试口径和
原始位置。该 profile 产生于最终三项优化之前，只能作为历史瓶颈归因证据；最终正式
性能结论仍以 `results/20260903_device0_trb_steady_state_performance.json` 为准。

PyPTO 内部每个 child task 按物理 AIC/AIV 核展开的真正细粒度泳道位于
`results/20260903_device1_pypto_child_swimlane/pypto_child_task_swimlane.json`，可直接拖入
Perfetto。它包含全部 60 个 AICore 上的 834 个 child task，以及 Scheduler 和
Orchestrator 视图。由于 borrowed-device L1 禁止 DFX，该文件由同一最终 TRB callable
在独立 L2 diagnostic run 中采集，只用于 task 排布、依赖和 scheduler-gap 分析，不作为
L1 绝对性能结论。

### 6.4 短上下文 device gate 与 top-k 语义边界

`need_index_score` 是 PyPTO program 内部的单元素 device tensor，不是新的
public ABI 参数。它由本来就必须执行的 `csa_rope_step` 产生，其
TaskId 作为 predicate dependency 传给下列 6 个 score-only task：

1. `idx_qr_proj_matmul`；
2. `idx_qr_dequant_rope`（dequant、native BF16 rounding 与 RoPE 融合）；
3. `qr_hadamard_quant_mixed`（Hadamard matmul、native rounding、amax 与
   int8 quant 合并为一个 mixed child）；
4. `weights_proj`；
5. `weights_proj_reduce`；
6. `score`。

只有当所有 request 的最大 `visible_len <= 512` 时，这 6 个 task 才整体
跳过；任意 request 越界就保守地执行完整 score 路径。
`indexer_compressor` 仍必须执行，因为它负责本 decode step 的 index cache/state
更新，不属于 score-only 工作。边界是可见压缩位置数而不是某个
Host 上的解码步常量：`visible_len == 512` 走快路，`visible_len == 513`
恢复完整 score。在 ratio-4 且 cache length 不构成更小上限时，单 token
position 2047–2050 的可见长度仍为 512，position 2051 才变为 513；
对 S8 bucket，start position 2043 的最后一行仍全部走快路，2044 则进入
完整 score 路径。

短路径的 direct top-k 先将输出填为 `-1`，再写入升序的完整可见索引
`[0, visible_len)`（加上现有 offset）。当可见数不超过 top-k 容量时，
所有可见位置本来就必须被选中，因此它与 native 是
**selection-set 等价**，但不保证 native score sort 产生的 **raw top-k 顺序等价**。
当前 top-k 是单算子内部 scratch，后续 attention 对 K/V 和对应 score 使用同一
排列；正式正确性门禁因此是最终 output 与六类 mutable state 对比，
不应把 raw top-k 顺序相等作为契约。浮点累加次序不同仍可能产生容差
范围内的尾数差异，不应将 selection-set 等价误写为 bit-exact 输出。

### 6.5 HBG 当前只作功能回归

当前 Simpler HBG record path 尚未把 predicate 编码为可 replay 的 HBG node；
遇到 predicate 时会走 ordinary fallback，因此不能用它衡量本轮“在图中
跳过 6 个 score-only task”的稳态收益。基于这一实现边界，当前正式
性能验收固定为 **TRB ACLGraph**；HBG 只保留 eager/ACLGraph 数值、状态、
owner 和 replay 生命周期的功能回归。在 HBG 实现原生 predicate record/replay 前，
任何 HBG 性能数字都不能与 TRB 快路直接合并或用于当前胜负结论。

## 7. 生命周期与清理

- warmup、capture、replay 结束后的同步由测试 caller 显式完成；L1 op 内部
  不做 stream/device synchronize。
- graph 必须先外部 quiesce，再 reset/destroy，最后才能 close backend。
- capture snapshot 当前按 context 粒度保留，销毁单 graph 不会精确释放一条
  binding；这是不查询 graph handle 的安全边界。
- 六类 cache 的 storage 变化要求显式 reinstall layer owner；普通 input、output
  和 metadata 地址可以重新 bind，但不能把它误解为 captured graph replay 时
  任意替换 storage 地址。
- 不在一个进程中切换 TRB/HBG，不依赖 close 释放 process-pinned binary。

## 8. 交付前仍需关闭

除上述 A3 缺口外，private kernel primitive 的 CANN Open Software License 与
vLLM-Ascend Apache-2.0 根许可证之间的 NOTICE/provenance 处理尚未关闭。在维护者
确认许可证方案前，这个 private implementation 不能作为可发布代码宣称完成。

完整设计、失败过程和逐步证据见：

- `tests/DeepSeekV4_Decode_CSA_PyPTO_L1开发计划.md`；
- `tests/DeepSeekV4_Decode_CSA_PyPTO_L1实现过程记录.md`。
