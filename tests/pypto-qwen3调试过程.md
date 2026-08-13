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
