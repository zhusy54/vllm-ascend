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
