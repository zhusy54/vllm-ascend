# PyPTO Qwen3-14B 接入 vllm-ascend 调试过程

本文记录单卡跑通过程中的设计决策、踩坑和验证结果。代码改动只落在 `vllm-ascend`；`pypto-lib` 当库调用。

## 已确认范围（2026-08-13）

| 项 | 选择 |
| --- | --- |
| 精度 | 官方 Qwen3-14B BF16（`/mnt/workspace/inductor/models/Qwen3-14B`） |
| NPU | 物理/逻辑 device 1（`ASCEND_RT_VISIBLE_DEVICES=1`） |
| max_model_len | 1024（契约上限 4096） |
| 入口 | 离线 `vllm.LLM.generate` |
| vanilla 路径 | 保留为默认 |
| 打开方式 | 独立模型类名 `PyptoQwen3ForCausalLM` |
| 代码位置 | 只改 vllm-ascend |
| 采样 | 只 greedy / temperature=0 |
| 验证 max_tokens | 32 |
| host 编译 | 首次 launch 现编 |
| KV | **内存管理留在 vllm-ascend**（BlockManager / `block_table` / `slot_mapping`）；pypto 只做计算 |

## KV 处理方案

目标：不另起一套 pypto-serving 的独立 paged pool。

1. 模型里挂 40 个标准 `Attention`，让 V1 按昇腾默认形状分配  
   `(2, num_blocks, 128, 8, 128)`，即 `[K|V, page, token, kv_head, dim]`（BSND，page=128）。
2. `block_table` 仍是 `[B, max_blocks]`，适配层 flatten 成契约的 `[B * stride]`。
3. `slot_mapping` 公式与契约一致：`page * 128 + offset`，原样转 int32。
4. 契约 fused host 要一层叠好的  
   `k/v: [num_layers * num_pages * 128 * 8, 128]`（layer-major）。  
   - 优先：把 vLLM 每层 cache 看成上述大张量的 view（同 storage，kernel 写回无需 memcpy）。  
   - 回退：各层是独立分配时，`stack` 进契约张量，kernel 后再 `copy_` 回 vLLM 各层。分页 ID 始终是 vLLM 的。
5. 禁止自己 `malloc` 第二套 page 池、禁止改写 BlockManager。

## 契约注意点

- `contract.py` 里 `qwen3_prefill_host` 仍以 `hidden_states` 开头且没有 `embed_weight`。
- 实际 `prefill_fwd` 签名是 `input_ids` 开头，末尾带 `embed_weight`（device 侧 gather）。
- 适配层调用 **`load_kernels()` 得到的 `prefill_fwd` / `decode_fwd`**，不走过期 host wrapper。
- `decode_fwd` 内联 greedy；跑通时仍把 logits 交给 vLLM sampler（temperature=0），两边应对齐。

## 阶段日志

### 2026-08-13 阶段 3 — 第一次 launch 失败

- `LLM(...)` 在 `ModelConfig` 校验时报：`This model does not support --runner generate`。
- 原因：vLLM 用 `is_vllm_model` 认生成模型，要求类上有 `embed_input_ids`。缺了就被当成非 generate。
- 处理：给 `PyptoQwen3ForCausalLM` 补上 `embed_input_ids`（查 embedding 表；真正计算仍在 pypto host 里做）。

### 2026-08-13 阶段 4 — profile_run 无 attn_metadata

- 架构已解析为 `PyptoQwen3ForCausalLM`，8 shard 权重读完。
- `determine_available_memory` → `profile_run` 在 KV bind 之前 dummy forward，`attn_metadata` 为 None。
- 同时 `Loading model weights took 0.0040 GB`：打包后的契约权重当时还在 CPU，显存画像会偏小。
- 处理：`load_weights` 后立刻 `.to(npu)`；没有 metadata / KV 时返回占位 hidden，不调 pypto host。

### 2026-08-13 阶段 5 — KV 实际形状是拆开的 4D

- 引擎起来了：权重 27.52 GiB，KV 24.46 GiB / 160256 tokens（1252 pages × 128）。
- 第一次真实 generate（22 tokens）在 `stack_vllm_kv_as_contract` 失败：
  `got (1252, 128, 8, 128)`。
- 原因：昇腾 allocate 路径是每层 `(k, v)` 两个 4D 页，不是 `(2, P, S, H, D)`。适配层误把 tuple 的 `[0]` 当成 virtual-engine。
- 处理：`split_vllm_layer_kv` 同时接受 5D stacked 和 `(k, v)` 4D pair。

### 2026-08-13 阶段 6 — 整池 cat OOM

- `torch.cat` 40 层 K 还要再申请 12.97 GiB；当时已占用 53.5 / 61.3 GiB。
- 改成 `compact_vllm_kv_for_contract`：只按 `block_table`/`slot_mapping` 收集用到的 page，映射成紧凑 id，算完 `scatter` 回 vLLM 原页。分页管理仍在 vllm-ascend。

### 2026-08-13 阶段 7 — 第一次真正调到 prefill_fwd

- compact KV 通过；`load_kernels` + 编译成功（约 1 分钟）。
- 执行时报：`Tensor at position 0 is on npu:0, expected CPU`。
- pypto L2 runner 只接受 CPU `torch.Tensor` 或 `DeviceTensor`。
- 直接把 DeviceTensor 传给 `@pl.jit` 会丢掉 shape/dtype，特化失败。
- 处理：`kernel.compile(*torch_args)` 再用 `compiled(*DeviceTensor)`。
- DeviceTensor 包 torch NPU `data_ptr` 会在 `get_tensor_data`/`memcpy` 段错误（Worker 地址空间不是 torch 的）。
- 改回 CPU 入参：权重留在 CPU；NPU 上留一块同尺寸 reservation，避免 KV 池把卡吃满。执行后把 logits/compact KV copy 回 NPU。

### 2026-08-13 阶段 8 — 真机 a2a3 执行 prefill 被核数卡住

- `RunConfig(platform="a2a3")` 后不再走 sim。
- 真机报错：`REQUIRE_SYNC_START_INVALID`，`require_sync_start` 要的 block 数超过本卡物理核。
- 本卡 `aclrt`：AIC=20 / AIV=40。`prefill_fwd` 里 `GATE_UP/DOWN_PROJ/SILU/ATTN_PHASE=24`，`ROPE=32`，`FINALIZE=48`，且有 `sync_start=True`。
- 这是契约 kernel 按 24 核 A2 切的，不改 `pypto-lib` tiling 无法在这张 20 AIC 的 A3 上跑通 fused prefill。
- 适配层、KV 分页、权重打包、独立模型类和 CPU 单测（12 passed）已在 vllm-ascend 落地。

### 2026-08-13 阶段 9 — 把 prefill SPMD 降到 20 AIC

- `prefill_fwd.py`：`ATTN_PHASE` / `GATE_UP` / `DOWN_PROJ` / `SILU` 24 → 20。
- decode 侧 `QWEN3_PA_BLOCK_DIM=20`（`fa_fused` 也是 `sync_start=True`）。

### 2026-08-13 阶段 10 — hard syncall 与 20 核不满编

- 降到 20 后编译失败：`HardSyncallOccupancy`，`qk_pv_skew_probe` 里 hard `syncall` 要求占满编译器认定的 24 core-group。
- 处理：两处 `sync_start` attention SPMD 改为 `pl.system.available_cluster_count()`，stride 同步。

### 2026-08-13 阶段 11 — 通路跑通但输出是 `!::::`

- 冷启动 1 打出 `PYPTO_QWEN3_STAGE` prefill + 32 步 decode，文本是 `!:::::::::::::::::::::::::::::::`。
- tokenizer：id 0 = `!`，id 25 = `:`。独立 `prefill_probe.py` 用同一套官方权重 + 全零 page，argmax 就是 `2`（logit≈55），说明 fused prefill 数值本身没坏。
- 探针里 host `k_cache` 全 0：`pl.Out` 会 D2H，但 KV 只标了 `pl.Tensor`，runtime 不当输出拷回。decode 一直在看空 cache。
- 处理：`prefill_fwd` / `decode_fwd` 的 `k_cache`/`v_cache` 改成 `pl.InOut`（与 contract 文档一致）。探针复查 KABS=119.5、VABS=53，top 仍是 `2`。

### 2026-08-13 阶段 12 — vLLM 路径 logits 全是 NaN

- 接上 InOut 后再走 `LLM.generate`：`PYPTO_QWEN3_LOGITS` 显示 prefill/decode 都是 `kabs=nan`、`top_val=nan`。
- 独立探针没有 NaN。差别是 vLLM 新分配的 KV 页未清零；attention 按 128 token tile 读整页，mask 位仍是 `0 * NaN = NaN`。
- 另：权重迭代器可能复用 staging buffer。`collect_hf_state_dict` 改为立刻 `cpu().clone()`。
- compact 只拷贝 `seq_lens` 覆盖到的 token，未使用尾部保持 0。
- 用户纠正：pypto 入口吃的是 **heap 上的 device 指针**，不该每次 `kernel(*CPU)` 再 malloc/H2D，更不该拿 torch NPU `data_ptr` 去包 DeviceTensor（地址空间不是 Worker 的）。
- 用户澄清：pypto heap 只是算子内部 workspace，不是放权重/KV 的地方。入口必须是 **torch_npu tensor 的 data_ptr**（`DeviceTensor(child_memory=True)`）。
- 小算子验证（`npu_ptr_probe.py`，`x+1`）：CPU 路径和 NPU `data_ptr` 路径都对，maxdiff=0。NPU 路径 `bind.args` 几乎为 0（不再 H2D 用户数据）。
- 14B 改为：权重量在 npu 上；compile 仍用 CPU 样例；execute 走 `ChipWorker.run(compiled, *DeviceTensor(torch_npu.data_ptr))`。

### 2026-08-14 阶段 13 — 入口全部改成 torch_npu data_ptr

- 独立 `prefill_probe.py`（`PYPTO_PROBE_NPU_PTR=1`）argmax 仍是 `2`；vLLM `LLM.generate` 同一套 DeviceTensor 却是 logits/kabs=NaN，输出 `!!!!`。
- `seq_lens` / `chunk_lens` **不是 Scalar**：契约是 `[BATCH] int32` 张量。单卡也要传 `tensor([22], int32)`，用 `pl.tensor.dim` / `pl.tensor.read(..., [b])`。
- 差别是 vLLM metadata 经常把这两个小张量留在 CPU，和 NPU 权重指针混绑。适配层 `materialize_npu_args` 先把**全部**入参（含 `[BATCH] int32`）搬到同一张 NPU 再 `wrap_torch_npu_ptr`。
- `wrap_torch_npu_ptr` 不再内部 `contiguous()` 出临时对象；调用方必须保住 owner。
- RMS / QK-norm / RoPE 钉死 fp32；线性/embed 钉死 bf16。离线入口关 V1 多进程，并留出 ChipWorker workspace。

### 2026-08-14 阶段 14 — ChipWorker 必须在 KV 池之前初始化

- 把 vLLM 第一次 prefill 的 25 个入参存成 `vllm_prefill_args.pt`，进程外 `replay_vllm_args.py`：`FINITE True ARGMAX 17`（token `2`）。参数本身没问题。
- 同一组参数在 vLLM 进程里跑（CPU 对照和 DeviceTensor）都是 NaN。`default_dtype` 仍是 fp32。
- 差别是 vLLM 已经占了权重 + KV，ChipWorker 的 heap 后申请，和 torch 缓存分配器抢 HBM。
- 处理：`load_weights` 把权重量到 NPU 后立刻建 `PyptoChipSession`，赶在 V1 给 KV 池画像/分配之前。生产路径仍是 torch_npu `data_ptr`，`seq_lens`/`chunk_lens` 仍是 `[BATCH] int32`。

### 2026-08-14 阶段 15 — ATB warmup 会把 fused host 打成 NaN

- 申请不到内存会报错；提前建 ChipWorker 后仍剩 32 GiB，还是 NaN，说明不是 malloc 失败。
- 干净进程 / 只 init HCCL：重放同一组参数 `FINITE ARGMAX 17`。
- 只跑 `torch_npu._npu_matmul_add_fp32`（worker `_warm_up_atb`）后再重放：`FINITE False MAX nan`。占 28 GiB 或 CPU bind 单独都不会。
- 处理：`PyptoQwen3ForCausalLM` 跳过 ATB warmup。vanilla 默认路径仍 warmup。

### 2026-08-14 阶段 16 — 双卡真 TP，不再 gather 回全宽

- 上一轮把权重 Megatron 切完后又 Gloo gather 回全宽，两卡各跑单卡 fused host，再对 `logits[0,:5120]` 做 allreduce/2。验证判成假 TP。
- `wo`/`w_down` 按层切 K 维（`shard_stacked_row_parallel`），不再把 stacked `[L*K, out]` 在 dim0 切成「前 20 层 / 后 20 层」。
- 计算路径只保留本 rank 分片：`wq` last-dim 2560，`wo` `[L*2560, 5120]`。
- 新 TP host：`pypto_qwen3_tp_kernels.py` 的分片 GEMM + `pypto_qwen3_tp_runner.py` 层循环；`o_proj` / `down_proj` 之后走 Gloo+SHMEM 上的 pypto allreduce。
- 入口：`examples/offline_pypto_qwen3_14b_tp2.py`；`vLLM.LLM(tp=2)` 仍受 `/dev/shm=64MiB` 限制。
- GEMM 必须用 M=TOK=32、K=256、N=128：M=1 数值错；K=128 不满足 512B 对齐；N=256 撑爆 L0B。
- 两次冷启动 greedy 文本都是 `2`；日志里 `PYPTO_QWEN3_ARGS wq=(204800, 2560)`，`boundary=o_proj:*` / `down_proj:*`。

### 2026-08-14 阶段 17 — 现有 TP2 实现的整网泳道 + torch profile

- 现有 TP2 不是一张 fused host：一次 forward = 40 层 × (q/k/v/o/gate/up/down + 2×publish + 2×allreduce) = **440 次** `ChipWorker.run`。L2 记录按 launch 落盘，同一 kernel 的 `dfx_outputs` 会覆盖，所以每次 run 后快照。
- allreduce 原先走 `_run_chip` + 默认 `CallConfig`，DFX 不生效。改为 `compiled.build_call_config(session.config)`，和 GEMM 一样能出 `l2_swimlane_records.json`。
- 采集脚本：`tests/pypto_qwen3_profiles/collect_qwen3_tp2_profiles.py`（torchrun 2 卡，Gloo）。关掉 per-launch `swimlane_converter`，只合并 raw records。
- 一份 records 末尾多了一个 `}`（覆盖写毛刺），`json.loads` 用 `raw_decode` 吃掉。
- 产物目录（与 TP1 对称）：`tests/pypto_qwen3_profiles/{tp1,tp2}/{swimlane,torch}/`；编译垃圾在 `scratch/`。
- **泳道**（rank0，prefill / decode 各 440 launch）：
  - `tests/pypto_qwen3_profiles/tp2/swimlane/{prefill,decode}.{json,png}`
  - Perfetto：`tp2/swimlane/{prefill,decode}_trace.json`
  - 设备时钟跨度 prefill 102 s / decode 87 s，但 packed AICore 只有 **283 ms**。图按层把 host 间隙去掉，否则毫秒级 kernel 在 100 s 轴上不可见。
  - 设备时间：gate/up/down 各 ~75 ms，q/o 各 ~22 ms，k/v 各 ~4 ms，publish+allreduce 合计 ~5.5 ms。墙钟主要在 host 间隙（Gloo barrier + torch RMS/attn）。
- **torch_npu profiler**（Level1 + PipeUtilization + `with_stack`，1 prefill + 4 decode）：
  - `tests/pypto_qwen3_profiles/tp2/torch/prof/`（`trace_view.json` ~305 MB，`python_function` 637k）
  - `tp2/torch/top_kernels.png`：热点是 `simpler_aicpu_exec_*` 与 `aicore_kernel_0`（ChipWorker / pypto），torch `aclnnMatmul` 等是层间 RMS/attn。
  - 集合通信：泳道里是 `*.publish` / `*.allreduce`；torch trace 里是 `invoke_pypto` + `gloo` barrier，没有 HCCL。
- 同一进程里先开 880 次 DFX 再打 profiler，会 AICPU `507018` 然后 HDC 断连。profiler 必须新进程、DFX 关掉。输出文本仍含 `2`。

### 2026-08-14 阶段 18 — TP2 整网 fused：P0 device allreduce

- 计划：prefill/decode 各一次 `ChipWorker.run`。通信必须 device 侧等对端，Gloo 只做窗口 rendezvous。
- SHMEM 槽从 `data_buf`+`out_buf` 改成 `data_buf`+`signal`（`[2,1] int32`）。`out_buf` 本来就没用。
- `pld.tensor.allreduce` 能编过，上卡 AICPU 507018。改成 example 同款 `notify(AtomicAdd)` + `wait(Ge)` + `remote_load`，仍是一次 run。
- 两个 DistributedTensor 的 orch 要 **两个** `device_ctx` 标量（同一个 CommContext 指针传两遍）。少传一个就是 507018。
- 探针 `examples/offline_pypto_qwen3_allreduce_fused.py`：与旧 publish+barrier 路径 `maxdiff=0.0`，`PYPTO_QWEN3_FUSED_ALLREDUCE_OK`。
- runner 仍走旧路径。40 层循环里 wait 的 expected 要按次递增（1..80）。

### 2026-08-14 阶段 19 — TP2 整网 fused：P1 骨架与编译限制

- 新增 `pypto_qwen3_tp_fused_ops.py` / `pypto_qwen3_tp_fused_layer.py` / `pypto_qwen3_tp_fused.py`。
- GEMM step 改 `@pl.jit.inline`，runner 的 chip 再包一层 `@pl.jit.incore`，否则 matmul 会落到 orch。
- 上卡环境必须 `source /mnt/workspace/inductor/env.sh`（ptoas 0.57）。`ptoas-bin` 要 GLIBC 2.34，本机没有；CANN 自带 ptoas 0.24 不认识 `pto.cmo.cacheinvalid`。
- `pl.create_tensor([32,5120])` 不能写在 incore 里（会当 UB tile，超 184KiB）。工作区要在 `@pl.jit` orch 里分配再当 InOut 传入。
- Decode 整层一个 incore 会撑爆 Vec；attention 拆成独立 incore，GQA 用 online softmax 按 32 行切 cache。
- **当前编译卡点**：同一张 `@pl.jit` orch 里第二次调用 comm incore（`get_comm_ctx` / notify / remote_load）会被当成 orch 代码，报 `undefined function pld.system.get_comm_ctx`。第一次 o 边界 allreduce 可以。所以一层 fused 还没编过 ptoas，generate 默认仍走 runner。
- 入口加了 `PYPTO_QWEN3_TP_FUSED=1` 开关；默认 0。
- CPU 单测 `tests/ut/models/test_pypto_qwen3_tp.py`：6 passed。日志：`{SCRATCH}/pypto_qwen3_tp2_fused_tests.log`。

### 2026-08-14 阶段 20 — P1 一层 fused 编译墙（详细）

#### 要解决的问题

Megatron TP=2 一层 forward 有两次集合通信：`o_proj` 之后一次 allreduce，`down_proj` 之后再一次。现有 `PyptoTpRunner` 每次 allreduce 都是独立 `ChipWorker.run`，中间还有 host `dist.barrier()`。整网 fused 要求 **一张 `@pl.jit` orch 里完成这两次 device 侧 wait**，否则 40 层仍是几百次 launch，泳道拼不起来。

P0 已经证明：**单独一次** allreduce 可以编过、上卡正确。

P1 卡在：同一张 chip 里放 **第二次** `notify/wait/remote_load` 时，当前 pypto + ptoas 0.57 过不去。

#### 仍然成立的事实（不要回退）

- 通信路径：Gloo PG + `torch.distributed._symmetric_memory` 窗口（槽名 `data_buf` + `signal`）+ pypto `notify(AtomicAdd)` / `wait(Ge)` / `tile.remote_load`。禁止 HCCL/HCCP。
- 两个 `DistributedTensor` 的 orch 必须传 **两个** `device_ctx` 标量（同一个 CommContext 指针传两遍）。少一个上卡 AICPU 507018。
- P0 探针 `examples/offline_pypto_qwen3_allreduce_fused.py`：`allreduce_sum_fused` vs 旧 publish+barrier，`maxdiff=0.0`，`PYPTO_QWEN3_FUSED_ALLREDUCE_OK`。
- P0 实现：`vllm_ascend/models/pypto_qwen3_tp.py` 的 `_compile_fused_allreduce`。orch 只调 **一个** `@pl.jit.incore fused_allreduce_step`。`remote_load` 形状是 `[1, HIDDEN]`，整颗 kernel 是纯 vector，没有 cube GEMM。
- `pld.tensor.allreduce`：能编过，上卡 AICPU 507018，已放弃。
- generate 默认 `PYPTO_QWEN3_TP_FUSED=0`，走 `PyptoTpRunner`。不要为了编一层把默认改成坏的 fused。
- CPU 单测 `tests/ut/models/test_pypto_qwen3_tp.py`：6 passed。

#### 环境

- `source /mnt/workspace/inductor/env.sh` 后 `ptoas 0.57`（`/mnt/workspace/inductor/pto/PTOAS/install/bin/ptoas`）。
- `ptoas-bin` 要 GLIBC 2.34，本机没有。CANN 自带 ptoas 0.24 不认识 `pto.cmo.cacheinvalid`，不能当后备。
- `PTO_PLATFORM=a2a3`，venv：`vllm-ascend/.venv/bin/python`。不要把 `/mnt/workspace/inductor` 放进 `PYTHONPATH`。
- 编译入口：`compile_fused_layer_chip(decode=False)`（`pypto_qwen3_tp_fused_layer.py`）。日志：`{SCRATCH}/pypto_qwen3_fused_layer_compile.log`。

#### 规格写法（当前树）

`vllm_ascend/models/pypto_qwen3_tp_fused_ops.py`：`@pl.jit.inline tp_allreduce`，body 与 P0 相同（store 本端 → `get_comm_ctx` → notify → wait → `remote_load`+add → consume notify/wait）。

`vllm_ascend/models/pypto_qwen3_tp_fused_layer.py`：`@pl.jit.incore tp_o_down_boundaries` 里按顺序：

1. `tp_allreduce(o_partial, …, credit+1, credit+2)`
2. `residual_add` + `rms_hidden` + `gemm_h_inter_step`×2 + `swiglu` + `gemm_inter_h_step`
3. `tp_allreduce(down_partial, …, credit+3, credit+4)`
4. `residual_add`

`@pl.jit tp_fused_layer_prefill_chip` / `tp_fused_layer_decode_chip` 在 front GEMM/attn 之后 **只调这一次** `tp_o_down_boundaries`。credit 在 incore 里算成局部量（orch 不能把 `credit_base + 1` 当实参）。

`pypto_qwen3_tp_fused.py` 的 40 层 orch 同样每层调一次 `tp_o_down_boundaries`。

**这张图编不过。** 一层 hidden 对比 `examples/offline_pypto_qwen3_tp_fused_layer.py` 因此没跑成。

#### 复现

```bash
source /mnt/workspace/inductor/env.sh
source /mnt/workspace/inductor/shmem/install/set_env.sh
export PTO_PLATFORM=a2a3
cd /mnt/workspace/inductor/vllm-ascend
.venv/bin/python -c \
  "from vllm_ascend.models.pypto_qwen3_tp_fused_layer import compile_fused_layer_chip; compile_fused_layer_chip(False)"
```

#### 试过的写法和原文报错

下面每条都在本机编过，不是推想。试完已收回实验分叉（`tp_allreduce_tiled` / `tp_allreduce_guarded` / orch 里手写 `split_aiv`），树回到上面的规格写法。

**A. orch 里第二次通信**

结构：front incore（无 comm）之后，orch 调两次 comm。变体包括：

- 两个不同 `@pl.jit.incore`（`tp_allreduce_o` / `tp_allreduce_down`）
- 同一个 `@pl.jit.incore tp_allreduce_step` 展开调两次
- `for phase in pl.range(2): tp_allreduce_step(...)`
- orch 里两个 `for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):` 区域，区内 inline `get_comm_ctx`

第一次 comm 可以 outline 成 incore（早期 build 里能看到 `tp_allreduce_o.cpp`）。第二次一律被摊进 orch。

报错（`OrchestrationReferencesResolved`）：

```
Orchestration function 'tp_fused_layer_prefill_chip' references undefined
function 'pld.system.get_comm_ctx'. The Program must contain every callee
referenced from orchestration.
```

例：`build_output/_jit_tp_fused_layer_prefill_chip_20260814_125842/report/codegen_errors.txt`
以及 `{SCRATCH}/pypto_qwen3_fused_layer_compile.log`（`_20260814_130642`，当时第二段 `split_aiv` 落在 orch 第 327 行）。

含义：`get_comm_ctx` 只允许出现在 InCore。第二次调用没有留在 Program 里的 comm 函数上，被当成 orch 代码，verifier 找不到这个 callee。

文档补充：HOST `pld.tensor.allreduce` **禁止**出现在 for/while。InCore 的 notify/wait 按文档可以循环；但 **从 orch 发起的第二次 comm 调用** 在这版编译器上仍会被摊进 orch。

**B. 一张 mixed incore 里两处 `remote_load`**

结构：就是当前 `tp_o_down_boundaries`——vector 通信 + cube GEMM 在同一个 `@pl.jit.incore`。`remote_load` 试过 `[1, HIDDEN]`（与 P0 相同）和 `[TOK, 128]`。

ptoas 0.57：

```
Failed to compile group 'tp_o_down_boundaries'
[tp_o_down_boundaries_aic, tp_o_down_boundaries_aiv]:
ptoas compilation failed:
loc(".../pypto_qwen3_tp_fused_ops.py":270:9):
error: 'pto.partition_view' op size at dim 0 must be positive, got 0
Error: Failed to parse MLIR.
```

例：`build_output/_jit_tp_fused_layer_prefill_chip_20260814_130002/report/codegen_errors.txt`
同目录 `ptoas/tp_o_down_boundaries.pto` 里，**第一处** `remote_load` 仍是 `32x128`，**后面复制出来的**变成：

```
pto.partition_view ..., sizes = [%c0_index, %c0_index]
    -> !pto.partition_tensor_view<0x0xf32>
```

P0 纯 vector 的同一处 `remote_load` 不会这样。mixed（cube+vector）展开/切 AIV 时，第二份通信 tile 的 dim0 被切成 0。

**C. mixed incore 里用 `if` 区分 o 边界 / down 边界**

结构：`for phase in pl.range(2)` 里只留一处 `remote_load`，`if phase < 1:` 走 residual+FFN，else / 第二个 `if 0 < phase:` 走 residual tail。

C++ codegen 在吐 MLIR 之前就失败（还没到 ptoas parse）：

```
Internal error: IfStmt in-place return_var 'xn__phi_v6'
yields different backing SSAs across branches:
%xn__ssa_v0_view vs %hidden_out__ssa_v0_view
Check failed: inplace_return_ssa[i] == branch_yields[i]
at ../src/codegen/pto/pto_control_flow_codegen.cpp:420
```

去掉 else、改成两个独立 `if` 后，phi 换成 `'up__phi_v6'`：`%up__ssa_v0_view vs %signal__ssa_v0_view`。

对照：`pto/pypto/src/codegen/pto/pto_control_flow_codegen.cpp` 要求 if 两个分支对每个 in-place return 用同一块 backing SSA。同文件对 DistributedTensor 还有「不能在两个分支赋不同窗口」的检查，指向 GitHub issue #2027。
本仓库里能编过的 `if` 是 `write_kv` 那种：循环里 `if t < n_tok:` 只写 cache，没有 else，也不在同一层改一堆 InOut。

**D. `@pl.jit.incore` 里写 `pl.split_aiv`**

官方 `pto/pypto/docs/en/user/language/04-scopes.md`：通信应放进 `for aiv_id in pl.split_aiv(2, mode=NONE)`，cube 放 region 外；`auto_scope=False` / `split_aiv` **不能**标在 `.incore` 上。

本机：

```
LowerAutoVectorSplit: this pl.split_aiv region is nested inside a scope
and cannot be lowered — region lowering does not cross a scope boundary.
... a scope inside a function declared pl.FunctionType.InCore reaches
this pass intact.
```

把 `tp_o_down_boundaries` 改成 `@pl.jit` 再从 layer chip 调用：`Unsupported function call`（`@pl.jit` 不能调 `@pl.jit`）。
改成 `@pl.jit.inline` 让 `split_aiv` 拼进 orch：`IR auto-name base cannot contain reserved delimiter '__': __inline71`。
把 `split_aiv` 直接写在 layer `@pl.jit` 里：第一段通信能 outline，第二段回到 **A** 的 `get_comm_ctx`。

**E. orch 实参是表达式**

`tp_o_down_boundaries(..., credit_base + 1, ...)`：

```
Outer call to wrapper 'tp_o_down_boundaries' arg 15 is neither a variable
nor a recognized constant literal
(unsupported expression kind for orchestration codegen)
orchestration_codegen.cpp:2226
```

credit 必须在 callee 里算，或先赋给局部名再传入。这是附属问题，单独好改，过不了 A/B。

**F. 其它已排除**

- `pl.create_tensor([32,5120])` 写在 incore 里：按 UB tile 算，327680 > 188416。工作区必须 orch `create_tensor` 再当 InOut 传入。
- 整层一个 incore（含 GQA）：Vec 386176 overflow。attn 必须单独 incore；decode GQA 按 32 行 online softmax。
- GEMM `@pl.jit.inline` 被 orch 直接调用：`Misplaced tensor.matmul in Orchestration`。runner 的 chip 要再包 `@pl.jit.incore`。
- `pld.tensor.allreduce`：编过，上卡 507018。

#### 编译器墙上的两扇门（要继续必须动 pypto/ptoas）

只改 vllm-ascend 走不通。需要至少修一门：

1. **orch 第二次 comm**：`OutlineIncoreScopes` / `OrchestrationReferencesResolved` 把第二次 `get_comm_ctx` 留在 InCore callee 上，不要摊进 orch。修完后 orch 可以：front → comm → FFN incore → comm → tail，comm kernel 保持 P0 那种纯 vector。
2. **mixed kernel 第二处 `remote_load`**：ptoas 不要把后一份通信 tile 切成 `0x0`。修完后当前 `tp_o_down_boundaries`（一张 incore 里两次 allreduce + FFN）才编得过。

只开 1 就够 P1/P2（FFN 继续用现成 `gemm_*_incore`）。只开 2 则规格写法可以直接编。两门都不开，一层 fused 和 40 层一张图都做不了。

#### 当前仓库状态

| 路径 | 状态 |
| --- | --- |
| `pypto_qwen3_tp.py` P0 `allreduce_sum_fused` | 可用 |
| `PyptoTpRunner` + `offline_pypto_qwen3_14b_tp2.py` 默认 | 可用；两次冷启动 greedy 含 `2` |
| `tp_o_down_boundaries` + `compile_fused_layer_chip` | 规格在，ptoas 不过 |
| `examples/offline_pypto_qwen3_tp_fused_layer.py` | 因编译失败未跑 hidden 对比 |
| `PYPTO_QWEN3_TP_FUSED=1` 整网 generate | 未开，也编不过 |

未做：一层 vs runner hidden 对比、40 层一张图、fused 两次冷启动、单次 `ChipWorker.run` 的 L2 泳道。


### 2026-08-13 阶段 0 — 对齐接口

- 读完 `pypto-lib/models/qwen3_14b/{contract,weights,prefill_fwd,decode_fwd}.py` 与 vllm-ascend `AscendAttentionBackend.get_kv_cache_shape`。
- 确认 vanilla 单卡 generate 已在本机跑通；本目标必须打到 pypto stage marker，不能只跑 vanilla。
- 用户明确：改动只在 vllm-ascend；KV 管理留在 vllm-ascend。

### 2026-08-13 阶段 1 — 适配层 + CPU 单测

- 新增 `vllm_ascend/models/pypto_qwen3_adapter.py`：权重打包、block/slot 映射、KV view、prefill/decode 实参拼装。
- 新增 `tests/ut/models/test_pypto_qwen3_adapter.py`，从真实起点调用上述函数（不用硬编码期望张量、不 mock 被测函数）。
- `pytest tests/ut/models/test_pypto_qwen3_adapter.py`：9 passed。日志：`{SCRATCH}/pypto_qwen3_adapter_tests.log`。

### 2026-08-13 阶段 2 — 独立模型类

- 新增 `vllm_ascend/models/pypto_qwen3.py`：`PyptoQwen3ForCausalLM`。
  - 40 个标准 `Attention`，KV 仍由 V1 分配/分页。
  - `load_weights` 走 `pack_official_weights` → `prepare_qwen3_weights`。
  - `forward` 读 `attn_metadata` 的 `block_tables` / `slot_mapping` / `seq_lens`，调 `prefill_fwd` 或 `decode_fwd`。
  - stdout 打印 `PYPTO_QWEN3_STAGE qwen3_14b.prefill_fwd|decode_fwd`。
- 注册：`ModelRegistry.register_model("PyptoQwen3ForCausalLM", ...)`。
- 离线入口：`examples/offline_pypto_qwen3_14b.py`（device 1，`max_model_len=1024`，greedy 32）。

### 2026-08-14 阶段 21 — 必须修改 PyPTO：打通重复通信上下文

- 阶段 20 的第一扇门最终定位到 `MaterializeDistTensorCtx`：它只跟踪直接
  `Var` alias，没有保留 user-call 返回值和 loop carry 的 DistributedTensor
  来源。因此第一次 comm 返回的 `data/signal` 再传给第二次 comm 时，pass
  错误地在 orchestration 中合成 `pld.system.get_comm_ctx`。
- 只改 vllm-ascend 无法绕过这条编译器不变量，因此按用户许可对 PyPTO 做了
  最小通用修复：复用 return-lineage 信息，覆盖 tuple/direct call return，并
  追踪 `ForStmt` / `WhileStmt` 的 iter-arg 与 return-var。
- 回归覆盖三种路径：tuple 双返回、单个 DistributedTensor 直接返回、循环内及
  循环后重复 dispatch；完整 JIT `comm -> compute -> comm` 也验证两次调用都
  复用入口的 `data_ctx/signal_ctx`，orchestration 中不再出现
  `get_comm_ctx`。
- PyPTO 验证：Materialize pass 文件 `13 passed`；完整 JIT 定向用例通过；
  C++ 增量构建、clang-format、ruff、cpplint、markdownlint 和 diff-check 通过。
  没有修改 PTOAS。

### 2026-08-14 阶段 22 — 重复 SHMEM allreduce 的协议修正

- `symm_mem.empty` 不保证 signal 初值。启动时现在精确清零 signal 槽，执行
  `torch.npu.synchronize()`，再用 Gloo barrier 让两 rank 同时进入 device
  协议。
- 单次 publish barrier 不足以防下一轮覆盖 peer 尚未读完的 `data_buf`。协议改为
  publish `notify/wait`、remote load、consume `notify/wait` 两阶段握手；credit
  全程单调递增，不再阶段间手工归零。
- credit 必须是运行时 scalar。签名模式改为 `pl.RUNTIME`，避免样例值 1/2 被
  编译成常量；生成 orchestration 已确认从 `orch_args.scalar(0/1)` 读取。
- `_dispatch_chip` 对 torch_npu 指针输入在 ChipWorker 前后同步，避免两个运行时
  stream 竞态；编译目录加入 rank+PID，避免 torchrun 两进程覆盖同一个产物目录。
- 双卡交替 payload 压测：4 轮及 160 轮都在 rank0/rank1 得到 `maxdiff=0.0`；
  160 轮结束 `credits=320`，标记 `PYPTO_QWEN3_FUSED_ALLREDUCE_OK`。

### 2026-08-14 阶段 23 — 消除 MixedKernels 的 AIV/AIC 数据竞态

- 修完通信上下文后，一层图首次真正上卡，但 prefill/decode hidden maxdiff 分别
  约 16/24，K/V 约 4--8。生成代码显示同一 MixedKernels 中 AIC 已经 TLOAD
  `xn`，AIV 才在后面 TSTORE；QKV、attention、FFN 都存在同类跨 lane GM
  producer/consumer 竞态。
- 每层重构为 16 个有序纯 task，仍由最外层一个 `@pl.jit` 提交：
  RMS(AIV) -> QKV(AIC) -> norm/RoPE/cache(AIV) -> attention prepare(AIV) ->
  QK(AIC) -> softmax(AIV) -> PV(AIC) -> cast(AIV) -> O(AIC) -> allreduce(AIV)
  -> residual/RMS(AIV) -> gate/up(AIC) -> SwiGLU(AIV) -> down(AIC) ->
  allreduce(AIV) -> residual tail(AIV)。
- attention 显式物化完整 scores/probs；prefill 为 `[640,32]`，decode 为
  `[640,128]`。所有普通 scratch 拆成分阶段 SSA，40 层循环内逐层创建；只有
  hidden、KV cache、data/signal 是 loop carry。
- 一层双卡与 runner 对比正式通过：prefill/decode hidden maxdiff 都是 0.125，
  相对 RMSE 分别 0.70%/0.88%；K maxdiff 0.015625，V maxdiff 0.03125，KV
  相对 RMSE 0.28%--0.32%；两 rank 一致且 `credits=8`。验收同时限制非有限值、
  峰值和相对 RMSE，不用单一宽松 atol 掩盖错误。
- 完整 40 层 prefill/decode 静态编译都 `EXIT=0`。每阶段动态任务数为
  `1 + 40 * 16 + 2 = 643`；产物中 `MixedKernels=0`、`get_comm_ctx=0`、
  `partition_tensor_view<0x0=0`，每层两次 allreduce 都复用纯 AIV kernel。

### 2026-08-14 阶段 24 — 真实 Qwen3-14B 整网双卡验收

- `PYPTO_QWEN3_TP_FUSED=1` 下每 rank 保留真实 Megatron shard：
  `wq=(204800,2560)`、`wk/wv=(204800,512)`、`wo=(102400,5120)`、
  `w_down=(348160,5120)`；未 gather 回全宽。
- 两次独立 torchrun 冷启动均完成 22-token prefill 和至少一次 decode，均输出
  `OUTPUT: 2`、`PYPTO_QWEN3_14B_TP2_GENERATE_OK`，进程 `EXIT=0`。
- 为加强重复调用验证，离线入口新增可配置的 `PYPTO_PROMPT`、
  `PYPTO_EXPECT_TEXT`、`PYPTO_MIN_OUTPUT_TOKENS`，并用 Gloo MIN 汇总两 rank
  判定，失败时不再让另一 rank 卡在 barrier。默认题目也改为“用两句话介绍
  北京。”，默认至少生成 8 token；原来的“只回答数字”仍可通过环境变量复现。
- 换题“用两句话介绍北京。”后生成满 32 token，执行 1 次完整 prefill + 31 次
  完整 decode，文本以“北京是中国的首都，历史悠久……”开头；每次 decode
  都完成 40 层和 80 个 device allreduce，最终 `EXIT=0`。

### 2026-08-14 阶段 25 — 单次整图 L2 泳道与独立 torch profiler

- L2 模式在 rank0/rank1 对 prefill、decode 各捕获恰好一份
  `l2_swimlane_records.json`，标记均为 `ChipWorker.run=1 aicore_tasks=643`。
- 任务级泳道直接绘制 raw AICore task，而不是把整次 launch 画成一个长条：
  prefill wall 328.098 ms，decode wall 329.974 ms，均为 643 tasks / 6 个实际
  core lane；同时生成 Perfetto trace。
- torch profiler 在完全独立的新 torchrun 中采集 1 prefill + 4 decode。rank0、
  rank1 各自都生成非空 `kernel_details.csv` 和 `trace_view.json`；rank0 表含
  47 行、11 个有效 kernel 汇总，`top_kernels.png` 生成成功，进程 `EXIT=0`。
- `tests/pypto_qwen3_profiles/tp1/`、`scratch/` 继续由局部 `.gitignore`
  排除；本次新采集并校验过的 `tp2/` 泳道、Perfetto trace 和 torch profiler
  共约 6.1 MiB，按用户要求保留在交付目录中。旧的约 746 MiB 历史产物没有
  混入。

#### 最终状态与边界

| 路径 | 最终状态 |
| --- | --- |
| runner fallback | 保留，可用 `PYPTO_QWEN3_TP_FUSED=0` 显式回退 |
| fused 单层 prefill/decode | 双卡数值、KV、重复 credit 通过 |
| fused 40 层 prefill/decode | 各一次 `ChipWorker.run`，真实 14B 双卡生成通过 |
| 通信 | Gloo rendezvous + SHMEM + device notify/wait/peer load；无 HCCL/HCCP |
| L2 / torch profile | 分进程真机采集通过 |

本阶段的验收入口是离线 torchrun。vLLM EngineCore 的连续批处理、mixed
prefill/decode、paged-KV 回写与 preemption/cache-reuse 契约仍不属于本轮已验证
范围，不能把离线整网通过外推成通用在线 serving 已完成。

### 2026-08-14 阶段 26 — 生成与 profile 验收 fail-closed

- 离线生成每步先计算两 rank 的本地 argmax，再用 Gloo 广播 rank 0
  token；两 rank 继续使用同一 token，同时把本地 logits 不一致纳入最终
  MIN 判定。这既避免 EOS 分叉导致 SHMEM 死锁，也不会用广播掩盖
  rank 1 计算错误。
- 按原始计划把离线 TP2 入口的默认路径切为 fused 整网 host；显式设置
  `PYPTO_QWEN3_TP_FUSED=0` 仍可回退到 `PyptoTpRunner`。
- profile 采集同样广播并校验每步 token。L2 验收强制每个 stage 恰好
  643 条整图任务（AIV 402 / AIC 241）；torch profiler 强制
  `aicore_kernel_0` 的 step 恰好为 1 prefill + N decode，截断图和空产物不再
  可能误报 PASS。
- vLLM Engine TP=2 入口改为显式 `PYPTO_QWEN3_EXPERIMENTAL_TP_ENGINE=1`
  opt-in，并对多请求、mixed stage、prefill 超过 32 token、decode 非单 token
  或序列超过 128 直接报错，不再返回伪造的零 logits。显式 opt-in
  后同样默认走 fused，`PYPTO_QWEN3_TP_FUSED=0` 保留 runner 回退。
- 最终 token 协议用两进程 CPU Gloo 回归验证：一致 logits 在两 rank
  均 PASS，故意让 rank 1 argmax 不同时两 rank 均 FAIL。最终代码随后在不设
  `PYPTO_QWEN3_TP_FUSED` 的默认命令下做了两次独立真机冷启动，均确认进入
  fused 路径，两 rank 本地 logits 全程一致；两次均生成相同的 32 token
  北京文本，并以 `EXIT=0` 结束。
