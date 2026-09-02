# DeepSeek V4 Decode CSA PyPTO L1 实现过程记录

## 0. 文档用途

本文档持续记录
`tests/DeepSeekV4_Decode_CSA_PyPTO_L1开发计划.md`
的实际实现过程。它不是只保留最终结论的结果报告，而是保留：

- 每次范围决策及其原因；
- 实际修改的文件与语义；
- 使用过的环境、commit 和命令；
- 编译、lower、Host UT、A3 ST、ACLGraph 和性能结果；
- 失败尝试、被证伪的假设及下一步修正；
- 用户工作树中已有改动与本任务改动的隔离边界；
- 尚未关闭的正确性、性能、生命周期和许可证风险。

后续每个修改批次都应在提交前更新本文档。尚未通过真实证据的项目必须写成“未验证”或“失败”，不得按计划或意图记为完成。

## 1. 最终确认的实现边界

### 1.1 仓库边界

本轮正式源代码只修改 `vllm-ascend`：

```text
vllm-ascend/
```

以下仓库仅作为依赖或只读参考，本轮不修改其源代码：

- `pypto-lib`：仅参考历史 DeepSeek V4 Flash DSpark CSA program；
- `pto/pypto`：使用已经存在的 L1、TRB、HBG、taskQueue 和 ACLGraph 能力；
- `pto/pypto/runtime`（simpler）：使用已经存在的 runtime；
- `vllm`：仅作为 vLLM-Ascend 的上游 Python 依赖。

若实现过程中发现必须修改上述依赖仓的框架缺陷，需要先记录可复现证据并向用户说明，不擅自扩张范围。

### 1.2 新代码目录

用户确认新 CSA 实现放在独立目录。正式实现目录为：

```text
vllm_ascend/ops/_pypto_dsv4_csa/
```

前导下划线表示当前是 private/experimental backend，不提前形成稳定公开 API。

对应测试、fixture、trace、Capsule、ACLGraph 和性能工具统一放在：

```text
tests/pypto_dsv4_decode_csa/
```

开发计划和过程记录继续放在 `tests/` 根目录，便于作为阶段性设计与验收依据。

### 1.3 功能边界

- 只覆盖 A2/A3 runtime 路径，实际上板使用当前 A3 双卡环境；不做 A5 和 simulator。
- 只做 DeepSeek V4 Flash 的 decode CSA，不部署完整模型。
- 目标是 native/PyPTO 单算子、状态化 trace、1/2/4-layer Capsule 的精度与性能对比。
- 最终 production 边界保持现有 `torch.ops.vllm.dsa_forward` schema。
- 首版不覆盖 `need_gather_q_kv=True`、DSA-CP、DCP、完整 Engine 调度和全模型精度。
- PyPTO L1 必须表现为 caller stream 上的一个普通算子；不跨单算子边界提前启动 scheduler，不使用 `rtStreamAddToModel`。
- TRB 与 HBG 测试必须在不同的新进程中运行。
- `pypto-lib` 中历史 program 不做修改，也不作为运行时 import 依赖。

## 2. 实现开始时的权威状态

记录日期：2026-09-02，时区 Asia/Shanghai。

### 2.1 仓库 commit

| 仓库 | commit |
|---|---|
| `vllm-ascend` | `e7cb166290dfcbf2f997aa67f01b323be643fe0e` |
| `vllm` | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| `pypto-lib` | `0073f4228811eae687f9049b417be803c75c49e1` |
| `pto/pypto` | `9cece0b730a96fe1a52c2637537132f524ffe1ea` |
| `pto/pypto/runtime` | `b6f905f63277597bd2547d672fd9d57b6013fca9` |

### 2.2 用户已有 dirty 文件

任务开始前 `vllm-ascend` 已有以下用户改动，本任务必须保留并避开：

```text
M  tests/ut/models/test_pypto_qwen3_adapter.py
M  vllm_ascend/models/pypto_qwen3.py
M  vllm_ascend/models/pypto_qwen3_adapter.py
M  vllm_ascend/worker/worker.py
?? atb_matmul_accum_atomic_state_leak_issue.md
?? pypto_init_soc_state_issue.md
?? vllm_ascend/models/pypto_qwen3_l1.py
```

本任务没有修改这些文件。`git diff --stat` 中显示的上述 tracked diff 不是本任务产生的，阶段性提交时必须用显式 pathspec 排除。

`pypto-lib` 原本已有用户修改：

```text
M models/qwen3_14b/decode_fwd.py
M models/qwen3_14b/prefill_fwd.py
```

本任务没有修改它们。

### 2.3 一次已撤销的范围偏差

在用户明确“`pypto-lib` 只做参考”之前，曾临时编辑
`pypto-lib/models/deepseek_v4_flash_dspark/decode_csa.py`。
用户澄清后已用 `apply_patch` 完整恢复，并验证：

```text
git diff --quiet -- models/deepseek_v4_flash_dspark/decode_csa.py
exit code: 0
```

因此 `pypto-lib` 当前没有本任务遗留的源代码修改。

## 3. Phase 0：环境探测与修复过程

### 3.1 初始问题

`vllm-ascend/.venv` 最初会从 user-site 加载旧的 `_task_interface.so`：

- 旧扩展 build commit：`824ff...`；
- 当前 simpler runtime commit：`b6f905f...`。

这会触发 source/build hash mismatch，不能作为 CSA 上板环境使用。问题是安装产物陈旧，不是本轮需要修改 PyPTO 或 simpler 源码的框架缺陷。

### 3.2 PTOAS 选择

当前可用的 A2/A3 PTOAS 是：

```text
/mnt/workspace/inductor/pto/PTOAS/build-v0.57-llvm21-cann9.2-clean/tools/ptoas
```

打包 wrapper 实际版本较旧并有 GLIBC 约束，因此构建前需要 `unset PTOAS_ROOT`，并将上面的本地 v0.57 路径放到 `PATH` 前部。

### 3.3 环境版本要求

目标 venv 是：

```text
/mnt/workspace/inductor/vllm-ascend/.venv
```

其中 Torch 版本是 2.12.0，不是 2.7。不能启用 `PYTHONNOUSERSITE=1`，因为当前环境有必要依赖位于 user-site；正确做法是把本次 native editable build 安装进目标 venv并验证实际 import 路径。

### 3.4 已执行的重建

已启动如下构建，均不修改依赖仓源码：

```bash
cd /mnt/workspace/inductor/pto/pypto
source .claude/skills/testing/load-env.sh
source /mnt/workspace/inductor/vllm-ascend/.venv/bin/activate
unset PTOAS_ROOT
export PATH=/mnt/workspace/inductor/pto/PTOAS/build-v0.57-llvm21-cann9.2-clean/tools/ptoas:$PATH

cd runtime
python -m pip install --no-build-isolation --no-deps \
  --config-settings=build.targets=_task_interface \
  --config-settings=build-dir=build/vllm-ascend-cp311-b6f905f -e .

cd ..
python -m pip install --no-build-isolation --no-deps \
  --config-settings=build-dir=build/vllm-ascend-cp311-9cece0b -e .
```

截至本记录首次创建时：

- simpler `_task_interface` editable build 已完成；
- PyPTO editable wheel 仍由原会话持续编译；
- 进程存在且 `cc1plus` 持续占用 CPU，因此不是假死，也没有重启构建；
- 尚未执行最终 import/hash 校验；
- 尚未占用 NPU。

完成后必须验证 `_task_interface.__build_commit__`、PyPTO Torch adapter 的 build Torch/torch_npu 版本以及 `.so` 的真实 import 路径。

## 4. Phase 1 初始代码落地

### 4.1 新增文件

当前在 private operator 目录新增：

```text
vllm_ascend/ops/_pypto_dsv4_csa/
├── __init__.py
├── config.py
├── contract.py
├── decode_compressor_ratio4.py
├── decode_indexer.py
├── decode_indexer_compressor.py
├── decode_sparse_attn_csa.py
├── kernel.py
├── qkv_proj_rope.py
└── rope_interleave.py
```

其中：

- `contract.py`、`__init__.py` 和静态 L1 factory 是本轮新增的 host/API 结构；
- primitive 初稿基于只读参考仓
  `pypto-lib/models/deepseek_v4_flash_dspark/`
  对应源码导入，随后改为 private package 内相对 import；
- 去除了参考文件中的 CLI、golden 和 standalone test 尾部，只保留 kernel 所需部分；
- `__init__.py` 延迟 import kernel，使 host-only contract 测试不要求安装 PyPTO。

### 4.2 静态 specialization contract

新增 `DecodeCSAProgramSpec`，当前字段包括：

- `batch`；
- 固定 `seq=8`；
- SWA、compressed、main state、inner state、indexer 五类物理 block 数；
- `layout_strides`；
- 包含以上所有字段的稳定 `key`。

首批 batch bucket 为 B4、B8、B12、B16。原因是 PyPTO L1 v1 拒绝 annotation 中的动态或非正维度，不能直接复用历史 `T_DYN/B_DYN` L2 入口。

六类 mutable state 已明确为：

1. compressed KV；
2. SWA KV；
3. main compressor state；
4. inner/indexer compressor state；
5. indexer K；
6. indexer scale。

对应 kernel ABI 中六者都声明为 `pl.InOut`，attention output 声明为 `pl.Out`。

### 4.3 Host contract 测试

新增：

```text
tests/pypto_dsv4_decode_csa/test_contract.py
```

覆盖：

- B4/B8/B12/B16；
- 非法 batch 和 seq；
- 五类非正物理 block count；
- 空、零、负 stride；
- `key` 包含全部物理容量和 stride family；
- 在独立子进程中安装 import guard，证明 import package 或 contract 不会 eager import `pypto`。

结果：

```text
.venv/bin/python -m pytest -q tests/pypto_dsv4_decode_csa/test_contract.py
31 passed

.venv/bin/python -m ruff check tests/pypto_dsv4_decode_csa/test_contract.py
All checks passed
```

这里只证明 host contract；不证明 kernel 可 lower、可 codegen 或可在 A3 执行。

## 5. 真实 A3 cache ABI 审计与修改

### 5.1 Indexer scale dtype

当前 A3 vLLM-Ascend 的 indexer scale cache 是 FP16。参考 kernel 初稿使用 FP32，已修改：

- `kernel.py` 顶层 `idx_kv_scale`：FP16；
- `decode_indexer.py` 参数：FP16；
- `decode_indexer_compressor.py` 参数：FP16；
- 写回前从 FP32 显式 cast 到 FP16；
- score 计算读取后再 cast 到 FP32。

尚需 A3 连续 decode 对比验证量化误差和状态写回。

### 5.2 Page-aware state 访问

初稿直接把以下 cache `reshape` 成逻辑连续二维 tensor：

- main compressor state；
- inner compressor state；
- indexer K；
- indexer scale。

这在物理 page 有 padding 时会跨页读写错误。第一轮大 patch 未能匹配源码上下文，`apply_patch` 原子失败，没有形成半修改状态；随后拆分为小 patch，改为：

```text
logical_row -> block_id + intra_block_offset
```

并通过 block table 或 slot mapping 对三维/四维 cache 直接 slice/read/write。

已删除的错误中间形式包括：

```text
compress_state_flat
idx_kv_cache_flat
idx_kv_scale_flat
```

`decode_indexer.py` 的 score tile 被约束为 `REDUCE_TILE <= BLOCK_SIZE`，并断言 tile 不跨 indexer page。

### 5.3 仍在核实的 stride 问题

“改成多维 slice”不自动等价于“支持任意 PyTorch `as_strided` view”。已经确认：

- L1 wrapper 会把实际 Torch `shape/stride/data_ptr` 作为 `ChipTensor.make_strided` 传给 runtime；
- `ChipTensor` 的 slice/offset 语义按
  `start_offset + sum(index[d] * stride[d])`；
- 首次成功 enqueue 会绑定 tensor stride，后续 stride family 变化会在 host 侧拒绝；
- `@pl.jit` specialization key 本身不包含 Torch stride；
- `@pl.jit` 当前也不支持把显式 `pl.TensorView(stride=...)` 保留到 specialization metadata。

因此必须进一步证明生成的 orchestrator/incore task 对外部 cache slice 使用 runtime `ChipTensor` stride，而不是在编译期重新假设 packed stride。当前结论是“正在审计”，不是已完成。

另外，当前 vLLM-Ascend 同时存在 v1/v2 cache reshape 路径；indexer K/scale 在某些当前 v2 路径中是独立连续分配，而 main/inner state 可能仍经过 page-padded view。必须针对本次实际 DSV4 decode 配置逐 tensor 记录真实 `shape/stride/storage_offset`，不能把某一条代码路径的推导泛化到全部 cache。

## 6. 静态 L1 lower 失败与修正记录

### 6.1 第一次失败：inline reshape 丢失 metadata

B4 `codegen_only` lower 首次失败：

```text
missing inferred tensor metadata for parameter 'position_ids'
```

原因是 `kernel.py` 将
`pl.reshape(position_ids, [tokens, 1])`
直接作为 `sparse_attn_csa` 的 inline callee 实参。

已改为：

```python
position_ids_t1 = pl.reshape(position_ids, [tokens, 1])
sparse_attn_csa(..., position_ids_t1, ...)
```

### 6.2 第二次失败：postponed annotation 无法解析局部容量

第二次 lower 失败：

```text
cannot infer parameter 'compress_state'
```

原因是 `from __future__ import annotations` 将嵌套 factory 中的
`main_state_blocks`、`inner_state_blocks`、`indexer_blocks`
保留为字符串，而 PyPTO 无法从该 wrapper 取得只用于 annotation 的局部值。

已移除 `kernel.py` 的 postponed annotations，使形参 annotation 在 factory 定义 nested program 时立即求值。这修复了形参 annotation 的局部容量解析。

### 6.3 第三次失败：正文 closure 常量不在 parser 符号表

最新 B4 lower 继续失败：

```text
kernel.py: rope_cos_t = pl.create_tensor([tokens, ...])
UndefinedVariableError: tokens
```

这证明移除 postponed annotations 只解决签名，不能解决函数正文中的 factory closure 常量。PyPTO 的文本/AST parser 不会把 nested factory 的任意正文闭包变量当作 DSL 常量。

当前状态：

- B4 尚未进入完整 IR；
- B8/B12/B16 因同一原因尚无有效 lower 证据；
- 不能声称静态 factory 已完成；
- 下一步需要改为 parser 原生支持的静态生成方式，例如顶层静态 program/template specialization；
- 不采用在正文加入无意义伪引用来“骗闭包”的脆弱方案。

## 7. Python 语法与清理状态

已执行：

```text
python -m compileall -q vllm_ascend/ops/_pypto_dsv4_csa
compileall-ok
```

这只证明 Python 语法可加载，不证明 PyPTO DSL 可解析。

该命令生成了 `__pycache__`，它们不是交付文件，阶段性提交前需要清理，并确认 `.gitignore`/status 中没有 bytecode。

## 8. 许可证与来源风险

从只读 `pypto-lib` 引入的 primitive 保留了原 CANN Open Software License Agreement Version 2.0 文件头；但文件头写有“See LICENSE in root”，而 `vllm-ascend` 根许可证是 Apache-2.0。另一个风险是当前 `kernel.py` 有较多逻辑派生自历史 `decode_csa.py`，但初稿只写了 Apache header。

因此在进入可提交的 production 代码前，必须明确以下之一：

1. 在 `vllm-ascend` 中按项目接受的第三方代码流程补齐准确的 license/NOTICE/provenance；或
2. 在具备授权的前提下重写为项目原生实现并保留合理 attribution；或
3. 得到维护者明确的许可证处理结论。

在此问题关闭前，当前 private 目录只能视为开发中的实现，不应直接对外发布。

## 9. 当前证据矩阵

| 项目 | 状态 | 证据/说明 |
|---|---|---|
| 独立 production 目录 | 已完成 | `_pypto_dsv4_csa/` 已建立 |
| pypto-lib 只读 | 已完成 | 临时偏差已恢复，目标文件 diff 为空 |
| Host static spec | 已完成第一版 | 31 项 UT 通过 |
| lazy import PyPTO | 已验证 | 两个独立子进程 import guard |
| 六类 mutable state 方向 | 已编码、未上板 | 六类 `pl.InOut` |
| A3 indexer scale FP16 | 已编码、未上板 | 三处 ABI 已改 |
| page-aware state indexing | 已编码、未 lower/上板 | flat state/cache 访问已移除 |
| 真实 runtime stride 传播 | 审计中 | wrapper/ChipTensor 已确认，task IR 尚待证明 |
| B4 static lower | 失败 | closure `tokens` 未被 parser 解析 |
| B8/B12/B16 lower | 未开始有效验证 | 被同一 factory 结构阻塞 |
| PyPTO/simpler venv 重建 | 进行中 | build session 仍活跃 |
| generic L1 TRB smoke | 未执行 | 等待 build/import 校验 |
| generic L1 HBG smoke | 未执行 | 必须独立新进程 |
| CSA eager A3 | 未执行 | kernel 尚未 lower |
| CSA ACLGraph | 未执行 | 依赖 eager/warmup |
| native/PyPTO golden | 未实现完整闭环 | 尚无真实 cache tuple fixture |
| Adapter/custom-op | 未实现 | 保持现有 schema 的设计已确认 |
| 1/2/4-layer Capsule | 未实现 | 后续阶段 |
| Stateful DecodeTrace | 未实现 | 后续阶段 |
| 性能对比 | 未执行 | 不可在正确性前宣称 |
| 许可证处理 | 未关闭 | 见第 8 节 |

## 10. 后续记录规则

每个批次至少追加：

1. 修改目的；
2. 修改文件；
3. 对 ABI、状态、生命周期和 capture 的影响；
4. 实际运行命令；
5. 完整的 pass/fail 结论；
6. 失败根因及下一步；
7. 是否使用 NPU、使用哪张 logical device；
8. 是否产生 commit，是否 push。

当前没有为本批次 commit，也没有 push。

## 11. 2026-09-02：PyPTO/simpler 环境闭环更新

本节之后的结论覆盖第 6、9 节中“build 仍在进行”和“B4 仍被 closure 阻塞”的旧状态；旧记录保留是为了说明失败与修正过程，不再代表当前状态。

### 11.1 开发环境

- vLLM-Ascend `.venv` 中的 `pypto` 实际从
  `/mnt/workspace/inductor/pto/pypto/python/pypto` 加载；
- `simpler` Python 包实际从
  `/mnt/workspace/inductor/pto/pypto/runtime/python/simpler` 加载；
- PyPTO 与 simpler task-interface 重建已完成，之前的 ABI/hash 不匹配已消除；
- lower/codegen 统一使用
  `/mnt/workspace/inductor/pto/PTOAS/build-v0.57-llvm21-cann9.2-clean/tools/ptoas`；
- 本批 host lower 没有使用 NPU。`npu-smi info` 在记录时显示 device 0 AICore 0%；不从该瞬时值推导其他 session 的历史使用情况。

## 12. AICore 对 runtime stride 的真实边界

### 12.1 审计结论

进一步审计 PyPTO L1 调用链后确认：

1. host L1 wrapper 会用 Torch tensor 的 `shape/stride/data_ptr`
   构造 `ChipTensor.make_strided`；
2. orchestration 层的 `ChipTensor` slice/view 能保留 stride 语义；
3. outlined AICore child kernel 的 tensor task arg 实际只消费
   `buffer address + start_offset`，不消费 runtime `ChipTensor.strides[]`；
4. child kernel 内部的多维 tensor 寻址仍使用编译期 packed stride。

因此，第 5.2 节的“对 page-strided 多维 tensor 直接 slice”仍然不足；它会在 host/orchestrator 层看似正确，但 child kernel 可能把下一页误当作紧邻的逻辑页。

### 12.2 本轮选择的零拷贝 ABI

不修改 PyPTO/simpler，也不在 decode 热路径制作 contiguous mirror。正式方案改为：

- adapter 在 prepare 阶段为每个 page-strided cache 建立
  `[1, physical_span]` 的零拷贝平坦 alias；
- alias 与原 tensor 共享 storage、`data_ptr` 和 `storage_offset`，并持有原 tensor 强引用；
- kernel 把 `block_id * page_stride + in_page_offset` 显式编入地址计算；
- 四个 page stride 是 `pl.Scalar[pl.INDEX]` 的 literal-specialized 参数，作为 static layout family 的一部分；
- 目前只接受“页内 canonical contiguous，页间可有 padding”的 A3 布局，其他 stride family 在 capture 外 fail-fast。

新增的 host 侧所有权对象和测试：

- `vllm_ascend/ops/_pypto_dsv4_csa/physical_storage.py`；
- `tests/pypto_dsv4_decode_csa/test_physical_storage.py`。

专项测试覆盖：同 storage/同地址、physical span 边界、padding canary、强引用保活、launch 不重建 tensor metadata，以及对 zero/negative/overlap/transpose/meta layout 的拒绝。

## 13. A3 四类 page-strided 物理布局锁定

`DecodeCSAPhysicalLayout` 已成为 `DecodeCSAProgramSpec` 的静态程序身份之一。当前 A3 contract 为：

| Tensor | 逻辑页 | dtype | page stride（elements） | physical span |
|---|---:|---:|---:|---:|
| main compressor state | `2 * 2048` | FP32 | 8192 | `(blocks - 1) * 8192 + 4096` |
| inner compressor state | `2 * 512` | FP32 | 1040 | `(blocks - 1) * 1040 + 1024` |
| indexer K | `32 * 128` | INT8 | 4160 | `(blocks - 1) * 4160 + 4096` |
| indexer scale | `32` | FP16 | 2080 | `(blocks - 1) * 2080 + 32` |

K 与 scale 的 stride 换算后都是 4160 bytes，对应它们共用的 indexer 物理页。当前 contract 拒绝页内 padding；这与 outlined child kernel 的实现能力保持一致，不对外声称支持任意 Torch stride。

Host contract + physical-storage 测试结果：

```text
43 passed
```

## 14. 平坦物理 alias 的 kernel 改造

已修改：

- `kernel.py`：四个参数从逻辑多维 cache 改为
  `[1, physical_span]`，并向 primitive 传入四个 page stride；
- `decode_compressor_ratio4.py`：main state 读写使用显式线性地址；
- `decode_indexer_compressor.py`：inner state、indexer K/scale 写回使用显式线性地址；
- `decode_indexer.py`：indexer score 路径用 `pl.load` 加 tile-level reshape，不对平坦 alias 做 tensor-level reshape。

修正过程曾触发并关闭以下 lower 错误：

1. Tensor/Tile 混用；
2. tile matmul 不支持该形式的 `b_trans`；
3. `row_sum` 需要显式 reduce tmp tile；
4. subscript 写回需要 Tensor RHS；
5. rank-reducing 5-argument slice 不被当前 lower 接受。

最终改用 rank-2 flat slice/显式 `pl.load`/`pl.store`，没有给 DSL 增加未验证的 ABI。

## 15. 四个 batch bucket 的 A2/A3 lower 证据

运行环境：`tensormap_and_ringbuffer`，`RunConfig(platform="a2a3")`，仅 host lower，未使用 NPU。

| Bucket | 结果 | 耗时 | IR 字符数 | 四物理 alias 的 `tensor.reshape` |
|---:|---:|---:|---:|---:|
| B4 | PASS | 约 10.8 s | 1,396,581 | 0 |
| B8 | PASS | 10.57 s | 1,396,619 | 0 |
| B12 | PASS | 10.68 s | 1,400,626 | 0 |
| B16 | PASS | 11.26 s | 1,400,633 | 0 |

命令核心形式：

```python
program = make_decode_csa_l1_program(DecodeCSAProgramSpec(batch=batch))
lowered = program.lower(config=RunConfig(platform="a2a3"))
```

这些结果证明四个静态 bucket 可进入完整 lower IR，且平坦物理 alias 没有被再次当作多维连续 tensor reshape。它们尚不等价于 PTOAS 编译成功、A3 eager 正确或 ACLGraph replay 正确；这三类证据需继续分层补齐。

## 16. 与之前 Qwen3-14B 路径的差异

Qwen3-14B 路径没有暴露本次问题，不是因为它已经证明 PyPTO child kernel 能消费任意 runtime stride，而是因为它显式规避了原始物理布局：

1. `compact_vllm_kv_for_contract()` 新分配连续 `key/value` contract buffer；
2. 只把当前 step 引用的 vLLM 物理页 copy 到连续 buffer；
3. PyPTO 消费的是压紧后的二维连续 tensor；
4. 运行后用 `scatter_contract_kv_to_vllm()` 写回原 cache。

这种方式可以作为正确性对照或早期实验 adapter，但不适合本次 CSA 正式性能路径：它会在 decode 热路径引入额外分配、D2D copy 和 scatter，还会掩盖真实 vLLM cache ABI。

CSA 因此使用“原 storage 零拷贝 alias + 显式 page stride 寻址”，而不复制 Qwen3 的 compact/scatter 方式。

## 17. 当前状态更新

| 项目 | 当前状态 |
|---|---|
| Host static/layout contract | 43 项专项 UT 通过 |
| B4/B8/B12/B16 A2/A3 lower | 全部通过 |
| 物理 alias 隐式 tensor reshape | 4 个 bucket 均为 0 |
| TRB PTOAS compile | 进行中，本节尚不记为 PASS |
| HBG PTOAS compile | 进行中，与 TRB 分开验证 |
| 正式 adapter/owner | 尚未完成；已具备零拷贝 physical alias 基础组件 |
| A3 eager/ACLGraph | 尚未执行 |
| native/PyPTO 精度对比 | 尚未执行 |
| commit/push | 本批次均未执行 |

## 18. lower IR 地址式和 scalar ABI 复核

独立只读复核在 B4 IR 中确认：

- main state：`block * 8192 + intra * 2048`；
- inner state：`block * 1040 + intra * 512`；
- indexer K：`block * 4160 + intra * 128`，写回与 score read 均使用；
- indexer scale：`block * 2080 + intra`，写回与 score read 均使用。

四个 scalar 名字在 lower IR 中只各出现于 public signature 一次，不再以变量形式出现于 body/child；这表明 literal specialization 已经折叠到 child 地址式，但编译后 callable 的四个 public scalar ABI slot 仍然存在。因此 adapter 不能依赖 Python 函数默认值，调用编译 artifact 时必须传入这四个值。

K 路径的形式为 `tile.load [1, 4096] -> tile.reshape [32, 128]`，scale 路径为 `tile.load [1, 32] -> tile.reshape`；这些 reshape 发生在已按物理地址 load 到 tile 之后，不是对外部 page-strided Tensor 做逻辑 flatten，因而不跨越页间 padding。

## 19. 六项 cache tuple 的正式 host adapter 基础

新增：

- `vllm_ascend/ops/_pypto_dsv4_csa/adapter.py`；
- `tests/pypto_dsv4_decode_csa/test_cache_adapter.py`。

`prepare_decode_csa_caches(cache_tuple, spec)` 现在执行以下工作：

1. 明确按 vLLM A3 ratio-4 顺序映射
   `(compressed KV, SWA KV, main state, inner state, indexer K, scale)`；
2. 校验六项均为 Tensor、处于同一 device，且 shape/dtype/stride 与 static spec 完全一致；
3. compressed KV/SWA KV 要求 packed BF16 布局；
4. 为四个 page-strided cache 在 prepare 阶段建立零拷贝 alias；
5. 校验 alias physical span 与 static program 完全一致；
6. 一次性创建不可变 `launch_arguments`，包含 6 个 cache 参数和 4 个 page-stride scalar；
7. `for_launch()` 每次返回同一份 mapping，不重建 tensor metadata。

它不调用 `.contiguous()`、`.clone()` 或 device copy。同一 `spec.key` 可以在 capture 外绑定到另一组 cache 地址；新 binding 仅创建 host Torch tensor metadata，不改变 static program identity。

本批 host 测试结果：

```text
test_contract.py + test_physical_storage.py + test_cache_adapter.py
57 passed, 14 warnings
ruff: All checks passed
git diff --check: passed
```

14 个 warning 均来自环境中 Torch `torch.jit.script_method` 的 deprecation warning，不是本 adapter 的失败或 fallback。

### 19.1 复核后修正：public state cache 是 4-D

首版 adapter 曾错误要求 main/inner state 输入已经是
`[blocks, 2, dim]`。只读复核发现真实 vLLM public cache tuple 是：

```text
main state:  [blocks, 2, 1, 2048], stride=(8192, 2048, 2048, 1)
inner state: [blocks, 2, 1,  512], stride=(1040,  512,  512, 1)
```

native `dsa_v1.py` 在调用 compressor 前才执行 `squeeze(-2)`。若不修正，adapter 会在真实 tuple 进入 kernel 前就报 shape error。

现已修正为：

1. adapter 校验真实 4-D public shape/stride；
2. prepare 阶段一次性创建 `squeeze(-2)` 的 3-D view；
3. 再对 3-D view 创建 flat physical alias；
4. `PreparedDecodeCSACaches.source_caches` 持有全部原始 public tuple，alias owner 同时持有 squeezed view；
5. launch/capture 路径不再创建 `squeeze` 或 `as_strided` metadata。

修正后 adapter 专项结果：

```text
14 passed, 14 environment warnings
ruff: All checks passed
git diff --check: passed
```

## 20. TRB 四 bucket 完整 PTOAS compile

> **结论勘误（2026-09-02）**：本节记录的是首次编译事实，但当时
> child primitive 的 `pl.dynamic` 已反向传播到 public entry。虽然 PTOAS
> codegen 成功，artifact tensor metadata 中仍有 `-1`，无法通过 L1
> prepare。因此本节的 PASS 只能解释为“动态 IR/codegen 可通过”，不能解释为
> “可执行的 positive-static L1 artifact 已闭环”。根因、修正和重新验收见第
> 23 节；第 23 节结论优先于本节。

在不使用 NPU 的独立验证进程中，使用：

```text
runtime=tensormap_and_ringbuffer
RunConfig(platform="a2a3", runtime="tensormap_and_ringbuffer", codegen_only=True)
clean PTOAS v0.57
```

四个 bucket 均完成 PyPTO compile 和 PTOAS codegen：

| Bucket | 结果 | 耗时 | public params | `.pto` | PTOAS `.cpp` |
|---:|---:|---:|---:|---:|---:|
| B4 | PASS | 26.22 s | 49 | 41 | 41 |
| B8 | PASS | 26.56 s | 49 | 41 | 41 |
| B12 | PASS | 25.68 s | 49 | 41 | 41 |
| B16 | PASS | 25.78 s | 49 | 41 | 41 |

四个 scalar metadata 都是 `direction=In, dtype=index`，名字为：

```text
main_state_page_stride
inner_state_page_stride
indexer_k_page_stride
indexer_scale_page_stride
```

每个 `.pto` 的 `target_arch='a2a3'`，artifact 的 runtime metadata 为 TRB。该结果把证据从“可 lower”推进到“TRB 可完整编译”，仍不代表 A3 运行时正确性。

## 21. HBG 四 bucket 完整 PTOAS compile

> **结论勘误（2026-09-02）**：与第 20 节相同，本节首次生成的 HBG
> package 仍带动态 public tensor metadata，不能进入 L1 prepare。这里保留
> 原始 codegen 数据作为问题发现过程，不把它作为 HBG 可执行性证据。重新验收
> 以第 23 节及后续上板记录为准。

HBG 使用与 TRB 独立的新 Python 进程验证，未运行 NPU，没有在同一 runtime owner 中切换 TRB/HBG。配置为：

```text
runtime=host_build_graph
RunConfig(platform="a2a3", device_id=0,
          runtime="host_build_graph", codegen_only=True)
clean PTOAS v0.57
```

| Bucket | 结果 | 耗时 | `.pto` | generated kernel C++ |
|---:|---:|---:|---:|---:|
| B4 | PASS | 35.17 s | 41 | 43 |
| B8 | PASS | 36.17 s | 41 | 43 |
| B12 | PASS | 34.95 s | 41 | 43 |
| B16 | PASS | 34.12 s | 41 | 43 |

与 HBG package/ABI 相关的静态证据：

- runtime metadata 为 `host_build_graph`；
- package descriptor 为 `expected_arg_count=49`、`scalar_count=4`；
- orchestration 依次读取 `orch_args.scalar(0..3)`，对应四个 page stride；
- main/inner/K/scale 的 AICore PTO 中分别有稳定的
  `8192/1040/4160/2080` 地址乘法证据；
- `pypto_orchestration_requirements_v1()` 返回 0，没有宣告 Host 读取 device tensor data 的非法 requirement；
- 四个 bucket 的 token specialization 分别为 32/64/96/128。

需要明确限定：四个 scalar 是 public callable ABI 的一部分，但真正的 AICore 地址式已按 static layout 专门化为常量；当前不支持对已编译 callable 传入不同 stride 便在运行时改变寻址。新 stride family 必须生成另一个 static spec/artifact，adapter 会拒绝把不匹配的 tensor 传给现有 artifact。

临时产物保留在 `/tmp/hbg_csa_compile.19ZhfQ/`；这些不是仓库交付文件。

## 22. Phase 1 静态编译状态更新

> **已被第 23 节覆盖**：本节当时遗漏了 compiled metadata 的正整数维度
> 检查，因而把“能编译”误写成了“静态 artifact 可编译”。在第 23 节重新
> 验证之前，TRB/HBG 四桶状态应降级为“codegen PASS、L1 executable FAIL”。

| 项目 | 状态 |
|---|---|
| B4/B8/B12/B16 TRB compile | 4/4 PASS |
| B4/B8/B12/B16 HBG compile | 4/4 PASS，独立进程 |
| HBG illegal Host device-data requirement | 未发现，requirements=0 |
| 49-param public ABI | TRB/HBG 均已确认 |
| 四 page-stride scalar | `In/index`，adapter 显式传入 |
| 真实 A3 public 4-D state tuple | adapter 已接受并一次性 squeeze |
| A3 eager 数值 | 未验证 |
| A3 ACLGraph replay | 未验证 |
| native/PyPTO 六状态 golden | 未验证 |

所以此时可以声称“四 bucket 的 TRB/HBG 静态 artifact 都可编译”，但还不能声称 Phase 1 整体完成；其完成标准还包括 B4/S8 关键 position 的输出与六状态对比、真实 page-strided A3 运行证据。

## 23. 动态 public ABI 复核、根因与修正

### 23.1 为什么首次四桶 compile 通过仍不可执行

独立复核不再只看 `lower()` 和 PTOAS 的退出码，而是继续读取
`CompiledProgram._get_metadata()`，并对每一个 public tensor extent 做
`dim > 0` 检查。该复核推翻了第 20～22 节的静态闭环结论：

1. `kernel.py` 的 public annotation 虽然由 `spec` 写成静态整数；
2. 六个 vendored inline primitive 仍用 `T_DYN/B_DYN/*_SPAN_DYN` 等
   `pl.dynamic` 标注变量轴；
3. PyPTO JIT dependency specialization 会 leaf-first 把 child 的动态轴映射回
   caller 实参；
4. 最终 public `decode_csa_core` 的 hidden/output、block table、cache block
   数和四个 physical alias span 被改成 DynDim；
5. compiled metadata 把非 `ConstInt` extent 表示为 `-1`；
6. L1 runtime 在 operator state/prepare 阶段明确拒绝 `dim <= 0`。

首次 B4 artifact 的自动 debug runner 甚至把 19 个动态 extent 都用 `1`
回填；它能 codegen 并不意味着它能按 B4 的 `T=32`、真实 cache span 执行。
因此从本节开始，CSA L1 编译的最低验收条件固定为：

```text
PTOAS success
AND public param count/output index correct
AND every public tensor extent is a positive integer
AND hidden/output/cache dimensions equal DecodeCSAProgramSpec
AND runtime scalar ABI matches adapter
```

### 23.2 放弃的中间方案

曾尝试在 public body 中增加 tensor alias，并额外加入
`static_tokens/static_batch` scalar，把动态 child 与 public 参数隔开。该方案
先后失败于：

- 同一个 `T_DYN` 在依赖图中得到 `32` 与 runtime scalar 两种冲突绑定；
- 由 runtime scalar 创建的 `step_cos` 等 local tensor 无法在依赖调用前得到
  静态 tensor metadata；
- 即使继续补特殊规则，也会给 public ABI 平白增加两个 scalar，并掩盖
  child 注解才是动态污染源的事实。

该中间方案已经移除，public ABI 不包含 `static_tokens/static_batch`。

还验证过“按 spec 克隆 Python function、递归替换 annotations/globals”的
备选方案。它可让真实 QKV primitive 在 T=8 下 A2/A3 lower 成功，但需要依赖
`JITFunction` 私有字段、复制 closure/kwdefaults、处理递归 dependency alias，
维护成本高于本场景所需，因此未进入正式代码。

### 23.3 最终静态化方式

这些 primitive 只属于 `_pypto_dsv4_csa` 私有实现，并且只作为
`decode_csa_core` 的 `@pl.jit.inline` dependency 使用。最终采用 PyPTO
已经支持的 caller-specialization 方式：

- 删除 child primitive 的 `pl.dynamic` 轴声明；
- 只把曾带动态轴的参数写成 bare `pl.Tensor` / `pl.InOut[pl.Tensor]`；
- 固定维度的权重和 scratch annotation 继续保留；
- dependency metadata 从 public entry 的静态参数和 caller 内
  `pl.create_tensor`/`pl.reshape` 推导；
- public entry 内的 `tokens/batch` 从已静态的 tensor metadata 读取，不新增
  scalar ABI。

这样每个 `DecodeCSAProgramSpec` 仍生成独立 callable，B4/B8/B12/B16 的
public ABI 保持 positive-static；child 只复用算法源码，不再承诺跨 bucket 的
单 artifact 动态 shape。

### 23.4 Triton 风格直调所需的 scalar 修正

复核正式 `@pl.jit(execution="l1")` direct facade 后还发现：只有函数签名中
默认值明确为 `pl.RUNTIME` 的 scalar，直调路径才把它保留为 per-call ABI。
此前四个 page stride 默认绑定到 closure 整数；显式
`compile(..., scalar=pl.RUNTIME)` 虽会产生 49 参数 artifact，但随后用真实整数
直调会命中另一份把 scalar 专门化掉的 45 参数 cache key。

现已把四项统一声明为：

```python
main_state_page_stride: pl.Scalar[pl.INDEX] = pl.RUNTIME
inner_state_page_stride: pl.Scalar[pl.INDEX] = pl.RUNTIME
indexer_k_page_stride: pl.Scalar[pl.INDEX] = pl.RUNTIME
indexer_scale_page_stride: pl.Scalar[pl.INDEX] = pl.RUNTIME
```

真实值仍只能来自 `PreparedDecodeCSACaches.for_launch()`；它们属于
`spec.physical_layout`，不是允许用户任意改变 layout family 的接口。修正只让
正式 Triton 风格 direct call、显式 compile 和未来 backend owner 共享同一份
49 参数 artifact。

### 23.5 当前证据

Host 专项测试新增两类 guard：

- 四个 bucket 的 public annotation 全部为正整数，四个 page stride 默认值
  必须是同一个 `pl.RUNTIME` sentinel；
- 完整 inline dependency graph 不允许任何 annotation DynDim 回归。

结果：

```text
tests/pypto_dsv4_decode_csa: 62 passed, 14 environment warnings
targeted ruff: passed
git diff --check: passed
```

B4 使用 A2/A3 TRB 完整编译后的 metadata 为：

```text
param count: 49
output indices: [44]
hidden/output: [32, 4096]
non-positive public tensor extents: []
scalar ABI:
  main_state_page_stride/index/In
  inner_state_page_stride/index/In
  indexer_k_page_stride/index/In
  indexer_scale_page_stride/index/In
```

该批 lower/compile 未使用 NPU。四桶 TRB/HBG 正在用上述更严格的 metadata
规则重新验收；在重新验收结果写入记录前，不沿用第 20～22 节的 4/4
“可执行 artifact”表述。

## 24. 静态 ABI 修正后的四桶正式重编译

第 23 节修正后，TRB 与 HBG 分别在独立 Host 进程、独立输出目录中重新
编译 B4/B8/B12/B16。本轮没有使用 NPU，且验收脚本不再只看退出码，而是把
compiled metadata 与 `inspect.signature(program._func)` 逐参数比较。

### 24.1 TRB

| Bucket | Tokens | compile | 耗时 | `.pto` | PTOAS C++ |
|---:|---:|---:|---:|---:|---:|
| B4 | 32 | PASS | 26.19 s | 41 | 41 |
| B8 | 64 | PASS | 25.61 s | 41 | 41 |
| B12 | 96 | PASS | 26.40 s | 41 | 41 |
| B16 | 128 | PASS | 25.44 s | 41 | 41 |

每桶共同满足：

- `49 params = 45 tensor + 4 scalar`；
- `output_indices == [44]`；
- 45/45 tensor metadata shape 与该 bucket 的静态 annotation 完全一致；
- 每个 extent 都是 builtin `int > 0`，无 `-1`、无 DynDim；
- token/batch 轴分别严格为 `B*8` 和 `B`；
- 四个 flat alias 固定为
  `[1,2125824] / [1,270384] / [1,1064896] / [1,530432]`；
- 四个 runtime scalar 的名字、顺序、`index/In` dtype/direction 完全匹配。

产物保留于：

```text
/tmp/dsv4_csa_static_trb_recompile.iztlsgrs
```

### 24.2 HBG

| Bucket | Tokens | compile | 耗时 | metadata |
|---:|---:|---:|---:|---:|
| B4 | 32 | PASS | 35.43 s | PASS |
| B8 | 64 | PASS | 35.57 s | PASS |
| B12 | 96 | PASS | 34.43 s | PASS |
| B16 | 128 | PASS | 34.58 s | PASS |

除与 TRB 相同的 49 参数、45/45 静态 shape 检查外，HBG 还满足：

- artifact/runtime owner 均为 `host_build_graph`；
- descriptor `expected_arg_count=49`、`scalar_count=4`；
- `pypto_orchestration_requirements_v1() == 0`；
- 生成产物中 `_DYN`、`DynDim`、dynamic-shape 命中均为 0；
- 四个 runtime scalar 均进入对应 task params，不是只留在 public 声明；
- 每桶生成 41 个 PTO、43 个 generated kernel C++。

产物保留于：

```text
/tmp/hbg_csa_static_recompile.LgVYQ5
```

至此，第 20～22 节首次 artifact 的动态 ABI 缺陷已经由重新编译证据关闭；
“四桶均有 positive-static TRB/HBG artifact”可以恢复为有效结论，但它仍不
替代 A3 数值、真实 page-stride 和 ACLGraph 证据。

## 25. Phase 0 与 CSA B4 零工作量 A3 纵向闭环

### 25.1 环境阻断及修复

设备检查时 device0 为 AICore 0%、无列出的进程；本节全部上板使用 device0，
未使用 device1。

最小 L1 smoke 首次有两个环境失败，均在进入有效设备计算前明确定位：

1. `PATH` 误指到 PTOAS 上一级目录，JIT 自动生成 `skip_ptoas=True` 的
   compile-only artifact，缺 `kernel_config.py`；
2. 修正 PTOAS 后，host runtime `dlopen` 报系统 libstdc++ 缺
   `GLIBCXX_3.4.32`。

最终固定本轮运行环境为：

```bash
unset PTOAS_ROOT
export PATH=/mnt/workspace/inductor/pto/PTOAS/build-v0.57-llvm21-cann9.2-clean/tools/ptoas:$PATH
export LD_LIBRARY_PATH=/mnt/workspace/inductor/toolchains/gcc15/lib:${LD_LIBRARY_PATH:-}
```

不替换系统库，只给测试子进程使用构建匹配的 GCC 15 runtime。

### 25.2 PyPTO 通用 L1 基线

在两个全新进程中运行：

```text
tests/st/runtime/l1/test_l1_jit_aclgraph.py
platform=a2a3, device=0
```

结果：

| runtime | eager | 两 callable | 换 stream capture | Torch 前后节点 | 3 replay |
|---|---:|---:|---:|---:|---:|
| TRB | PASS | PASS | PASS | PASS | PASS |
| HBG | PASS | PASS | PASS | PASS | PASS |

两条路径都使用正式 Triton-style `@pl.jit(execution="l1")` facade 和默认
taskQueue adapter；测试结束前外部 sync、graph reset、显式 shutdown。

### 25.3 真实尺寸零 fixture

新增 `tests/pypto_dsv4_decode_csa/fixtures.py`，构造的不是缩小 kernel：

- B4/S8，全部权重、metadata 和 cache shape 都是正式尺寸；
- 六 cache tuple 与 vLLM A3 顺序一致；
- main/inner state 使用 4-D public view；
- main/inner/indexer K/scale 使用 `8192/1040/4160/2080` 真实 page stride；
- block table、slot mapping、window indices 都指向合法且按 request 隔离的
  physical block；
- adapter 在 capture 前创建四个稳定零拷贝 alias；
- 零权重让 expected `attn_out` 精确为零，用于先隔离 ABI/调度问题；该
  fixture 明确不冒充非零数值 golden。

### 25.4 CSA B4 TRB/HBG eager 与 ACLGraph

TRB 和 HBG 分别使用全新进程。每个进程先做 ordinary eager
compile/init/prepare/warmup 和外部 sync，再在独立 capture stream 捕获：

```text
torch.add(source, bias, out=hidden)
  -> decode_csa_core(..., attn_out=preallocated)
  -> torch.add(attn_out, 1, out=final)
```

对 `source = 0.0 / 2.0 / -3.0 / 7.5` 连续 replay，结果如下：

| runtime | eager output | return identity | capture | 4 replay | max error |
|---|---:|---:|---:|---:|---:|
| TRB | finite, max_abs=0 | PASS | PASS | PASS | 0 |
| HBG | finite, max_abs=0 | PASS | PASS | PASS | 0 |

两条路径均在 device0 执行；capture/replay 内没有 adapter 重建、tensor
allocation、compile、prepare 或 sync。图销毁后显式 shutdown 成功。TRB 首次
单独 eager 探索脚本因未调用 shutdown 在进程退出时打印了“borrowed worker
intentionally leaked”警告；正式 ACLGraph 验证脚本已补齐 reset/shutdown，未
再出现该警告。

该证据已经证明正式 41-child CSA program 能以一个普通 L1 op 进入 A3
ACLGraph，并正确维持 caller-stream 顺序；尚未证明非零算法精度和真实 native
`dsa_forward` 对齐，下一阶段继续以非零 golden 和 custom-op harness 关闭。

## 26. 仓内可重复 A3 TRB/HBG runner 证据

第 25 节的手工上板脚本已收敛为可重复公共 harness：

- `tests/pypto_dsv4_decode_csa/a3_smoke.py`；
- `tests/pypto_dsv4_decode_csa/subprocess_runner.py`。

2026-09-02 在 device0 的两个全新进程分别执行：

```bash
python -m tests.pypto_dsv4_decode_csa.subprocess_runner \
  --runtime tensormap_and_ringbuffer \
  --device 0 \
  --batch 4 \
  --artifact-dir /tmp/dsv4-csa-trb-runner.nZjb5m/artifacts \
  --result-json /tmp/dsv4-csa-trb-runner.nZjb5m/result.json

python -m tests.pypto_dsv4_decode_csa.subprocess_runner \
  --runtime host_build_graph \
  --device 0 \
  --batch 4 \
  --artifact-dir /tmp/dsv4-csa-hbg-runner.3cDmK7/artifacts \
  --result-json /tmp/dsv4-csa-hbg-runner.3cDmK7/result.json
```

两个子进程退出码都为 0，关键结果完全对称：

| runtime | eager max abs | preallocated identity | replay | replay max error | padding mismatch |
|---|---:|---:|---:|---:|---:|
| TRB | 0 | PASS | 4 | 0 | 四类全为 0 |
| HBG | 0 | PASS | 4 | 0 | 四类全为 0 |

TRB 的完整 JSON 为：

```json
{
  "batch": 4,
  "device": 0,
  "eager_max_abs": 0.0,
  "logical_nonzero": {
    "indexer_k": 0,
    "indexer_scale": 8192,
    "inner": 8192,
    "main": 32768
  },
  "output_is_preallocated": true,
  "padding_mismatches": {
    "indexer_k": 0,
    "indexer_scale": 0,
    "inner": 0,
    "main": 0
  },
  "replay_count": 4,
  "replay_max_error": 0.0,
  "runtime": "tensormap_and_ringbuffer"
}
```

HBG 除 `runtime="host_build_graph"` 外，其余字段与上述 JSON 一致。
两个 runtime 没有在同一 Python 进程内切换，因而不依赖跨 runtime
shutdown/re-init 行为。

该 runner 使用真实 B4/S8 尺寸和 49 参数 ABI，同时覆盖：

1. ordinary eager compile/prepare/warmup；
2. 返回张量与 capture 前预分配 output 的对象同一性；
3. 独立 capture stream 上的 Torch predecessor/PyPTO/Torch successor；
4. `0.0/2.0/-3.0/7.5` 四次 replay；
5. 非零 APE 触发的 main/inner state 真实写回；
6. main/inner/indexer K/indexer scale 四类物理页 padding canary；
7. graph reset 后显式、幂等 `pypto.l1.shutdown(device=0)`。

`indexer_k` 在零权重 fixture 中保持为 0 是预期结果；这条 smoke 的
写地址证据由 main/inner 非零计数和全部 padding canary 共同提供，
不将零权重结果误写成非零数值 golden。

## 27. Native metadata 直接 ABI 及两个被零 fixture 遮蔽的正确性问题

第 23～26 节使用的 49 参数 ABI 是一个可上板的过渡形态，但它仍由
Host fixture 预先展开了多组 per-token mapping/window tensor，不是最终
`torch.ops.vllm.dsa_forward` 能直接提供的 native metadata 形态。本轮对
`dsa_v1.py`、`ops/dsa.py`、A3 cache allocator 和 metadata builder 进行了逐字段
只读审计，将 public ABI 收敛为 44 参数：

```text
40 tensors + 4 runtime page-stride scalars
output_indices == [39]
```

关键变化是：

1. 移除 Host 预计算的 main/inner state row mapping、compressed/indexer
   slot mapping、`window_swa_indices/window_swa_lens` 和 flat SWA slot；
2. 直接接收 vLLM 的五张 block table：SWA、compressed、main state、
   inner state 和 indexer；
3. 直接接收 A2/A3 native `[T,2] int32` SWA slot mapping，其两列为
   `[physical_block, offset_in_block]`；
4. 只公开 `start_positions[B]` 与 `kv_seq_lens[B]`，position rows、压缩边界和
   causal SWA window 全部在单个 L1 op 内重建；
5. block-table row width 属于 static specialization identity，不把它误等同于
   本进程恰好分配的 physical block 数量。

这次审计还发现两个之前被零权重零输出遮蔽的算法问题：

- S=8、ratio=4 时，每个 request 存在两个压缩边界，偏移为
  `first = 3 - start % 4` 和 `first + 4`；旧 kernel 只写了第一个。main
  compressor 与 inner/indexer compressor 现均以 `B*2` pooled rows 处理两次
  state/cache 写回；
- indexer query 的 RoPE 必须按 token 使用 S=8 个不同位置，而 inner
  compressor 必须按两个压缩边界使用 compressor metadata 对应的
  cos/sin。旧实现把一个 request 的同一行 RoPE 复用到全部 token，现已
  拆成 `token_cos/sin[T]` 与 `compress_cos/sin[B*2]`。

CPU reference 与 Host UT 同步改为两边界模型，避免用错误的 reference
给错误 kernel 做“自洽验数”。

## 28. Qwen3-14B 为什么没有暴露 page-stride 问题

这不是“Qwen3 已证明 PyPTO child 可以透明消费任意非连续 Torch tensor”。
代码对比的结论是，Qwen3 在 PyPTO ABI 之前把问题绕开了：

1. `compact_vllm_kv_for_contract()` 先收集本轮引用的 physical pages；
2. 通过 `new_zeros` 创建新的连续 K/V contract buffer；
3. 在 Python 循环中把已用 page 拷入，PyPTO 结束后再 scatter 回 vLLM cache；
4. `materialize_npu_args()` 还会对所有非连续实参调用 `.contiguous()`。

因此 Qwen3 得到的是连续 ABI，代价是热路径 allocation、D2D copy、Python
page loop 与 copyback。DeepSeek V4 CSA 的四类特殊 cache 则是 vLLM-owned、
page-padded 的长期 `InOut` state，不能在 ACLGraph 内照搬这个办法。

本实现保持的解法是：在 prepare 阶段建立同 data pointer、同 storage 的
`[1, physical_span]` 零拷贝 alias，再由 child 用
`block * page_stride + in_page_offset` 显式寻址。正式 launch 不做
`.contiguous()`、allocation 或 copyback。

同时补上了 backend 约束：特殊 cache 经 adapter 后也已是 canonical flat
alias，因而 40 个 public tensor 在 `bind()` 时统一要求与编译 shape 对应的
canonical row-major stride。不再允许“第一次绑定就是稳定的非连续 layout”被
pin 成一个错误 specialization family。

## 29. 44 参数 ABI 的 Host 编译与 A3 TRB/HBG 重验

### 29.1 Host 证据

2026-09-02 重跑全部 CSA 专项 Host 测试：

```text
122 passed, 14 warnings
ruff format/check: passed
```

B4 TRB 真实 lower/compile 在下列目录成功：

```text
/tmp/dsv4-csa-b4-trb-static.9sawL0
44 params = 40 tensor + 4 scalar
output_indices = [39]
source parameter order == compiled parameter order
```

本轮首次 compile 暴露的 `missing inferred tensor metadata for parameter 'cos'`
不是外部非连续 tensor 错误，而是每 token/每压缩边界 RoPE 拆分后，inline
子函数无法自行推导内部 scratch 的 variable-axis metadata。修正为 B16
上限静态 scratch（`T_MAX=128`、`CMP_ROWS_MAX=32`）后问题关闭。之后又修正了
JIT body 中不被支持的 Python `float(WIN)` 常量转换，全部 lower 通过。

### 29.2 A3 TRB

全新进程、device0：

```text
/tmp/dsv4-csa-trb-abi44.88gpyt
eager_max_abs = 0
output_is_preallocated = true
replay_count = 4
replay_max_error = 0
padding_mismatches = {main:0, inner:0, indexer_k:0, indexer_scale:0}
```

### 29.3 A3 HBG

另一个全新进程、device0：

```text
/tmp/dsv4-csa-hbg-abi44.6nCf5Q
eager_max_abs = 0
output_is_preallocated = true
replay_count = 4
replay_max_error = 0
padding_mismatches = {main:0, inner:0, indexer_k:0, indexer_scale:0}
```

这两条证据正式取代第 25～26 节的 49 参数上板结论。它们已证明
44 参数 native-metadata ABI 保持 L1 caller-stream、taskQueue、ACLGraph replay 与
page-padding 安全性；但零权重 fixture 仍不能代替非零数值 golden，也不能代替
实际 `torch.ops.vllm.dsa_forward` production dispatch 证据。

## 30. Production dispatch 的纯 Host 契约测试

2026-09-02 新增
`tests/pypto_dsv4_decode_csa/test_dispatch.py`，不使用 NPU，通过只记录
调用的 fake backend 与 CPU tensor 把 production owner/custom-op 边界固定下来。
审查后确认，warmup/capture-ready 是
`device + runtime + batch specialization` 级别的状态，不是 layer 级别的
状态；多个 layer 可以共用一个 backend/context 和一次 prepare，但每次
调用仍重新 bind 该 layer 的权重、metadata 与 I/O 地址。

新增的 9 条测试覆盖：

1. 所有 batch bucket 先 compile，然后整个 device owner 只 prepare 一次；
2. 同 device/runtime 的多 layer 共用同一 owner，已 prepare 的 registry
   拒绝追加 bucket 或更换 cache/layout family；
3. 首次 ordinary eager 调用走 `warmup`，且 `bind(retain=False)`；后续
   eager 调用走 launch，仍是 transient binding；
4. 没有 eager warmup 的 capture 直接报错；warmup 后未经外部 sync
   及显式 `mark_warmups_quiesced` 的 capture 也直接报错；
5. 只有通过 capture 前置条件后才执行 `bind(retain=True)`，保留图节点
   所引用的参数快照；
6. cache data pointer/storage offset/shape/stride/dtype/device 任一改变都在
   bind 前报错，sleep/wake 后不会偷偷继续使用旧 alias；
7. `need_gather_q_kv=True` 的 FlashComm 路径在 bind 前拒绝；
8. `dsa_forward` 保持原 custom-op schema，先构造六类 cache，再将排序后
   metadata 交给私有 dispatch；dispatch 抛错时不回退 native，避免在
   mutable state 可能已被写后重复执行算子；
9. profiling（`attn_metadata is None`）与未安装 owner 的路径继续走
   native implementation。

执行结果：

```text
pytest test_dispatch.py + test_native_metadata.py: 12 passed, 14 warnings
ruff check: passed
ruff format --check: passed
```

本轮审查没有发现需要修改 `dispatch.py`、`native_metadata.py` 或
`dsa.py` 契约的问题；仅新增测试和本记录。

## 31. 目标 checkpoint 的真实量化 ABI、45 参数迁移与 A3 重验

### 31.1 为什么 44 参数虽然能跑，却不能接目标模型

对目标 `Eco-Tech/DeepSeek-V4-Flash-0731-w8a8` 的
`quant_model_description.json` 和 vLLM-Ascend 实际 weight loader 逐项复核后，
确认第 29 节的 44 参数 ABI 与目标 checkpoint 的量化布局不一致。ratio-4
layer 的真实布局为：

- `wq_a`、`wq_b`、`wkv`、`indexer.wq_b`：INT8
  `AscendW8A8DynamicLinearMethod`，每个输出列带 FP32 weight scale；
- `wo_a`、`wo_b`、main/inner compressor、`weights_proj`、norm、APE 和
  attention sink：FLOAT/BF16；
- 上述四个 INT8 linear 都必须是 symmetric、zero offset；当前 kernel 没有
  zero-point ABI，不能静默接受其他 W8 scheme。

旧 ABI 错把 `wq_a/wkv` 当 BF16，同时错把 `wo_b` 当 INT8 并传入
`wo_b_scale`。这不是 NZ/ND layout 差异，也不能靠 cold-path
dequant/requant 修补；那样会改变 production 的数值算法。第 29 节的 44 参数
TRB/HBG 结果仍可作为 caller-stream、ACLGraph 和 page-stride 的历史结构证据，
但不再作为目标 checkpoint 可接入证据。

### 31.2 生产 ABI 收敛为 45 参数

本轮按真实 checkpoint 做以下精确替换：

```text
新增：wq_a_scale  [1024] FP32
新增：wkv_scale   [512]  FP32
删除：wo_b_scale  [4096] FP32

最终：41 tensors + 4 runtime page-stride scalars = 45 parameters
唯一纯输出：attn_out，index = 40
```

完整权重语义为：

```text
wq_a         INT8 [4096, 1024]   + FP32 scale[1024]
wq_b         INT8 [1024, 32768]  + FP32 scale[32768]
wkv          INT8 [4096, 512]    + FP32 scale[512]
idx_wq_b     INT8 [1024, 8192]   + FP32 scale[8192]
wo_a         BF16 [8, 1024, 4096]
wo_b         BF16 [4096, 8192]
```

`wo_a` 的真实 A3 loaded shape 是 `[groups, group_in, o_lora]`，TP1 即
`[8,4096,1024]`；cold pack 会先做 ND format cast，再交换末两维得到 kernel
要求的 `[8,1024,4096]`。所有 pack、format cast、RoPE 截断和 Hadamard
归一化只允许发生在安装/capture 之前。

### 31.3 Q/KV 动态量化与 BF16 数值边界

production native 路径会在 `wq_a` 与 `wkv` 都为 W8A8 dynamic 时共享一次
hidden activation quantization。本轮 kernel 同样按 token 对 BF16
`hidden_states[T,4096]` 只量化一次，得到：

```text
hidden_i8[T,4096]
hidden_dequant_scale[T,1]
```

`wq_a` 与 `wkv` 复用这两个 scratch，各自执行 INT8 matmul/INT32 split-K
累加，再按：

```text
int32_acc * hidden_row_dequant_scale * weight_column_scale
```

恢复 FP32。native `npu_quant_matmul(..., output_dtype=BF16)` 在 RMSNorm 前存在
明确 BF16 输出边界，因此 PyPTO 与 CPU reference 均显式执行
`FP32 -> BF16(rint) -> FP32` 后再进入 RMSNorm/RoPE，不能直接拿高精度 FP32
继续计算。zero hidden 的 amax 使用 `1e-4` 下限，不会除零或产生 NaN。

输出投影也按真实模型改为：

```text
attention group result
  -> BF16 wo_a，FP32 accumulation
  -> 显式 BF16 rounding
  -> BF16 wo_b，各 group FP32 partial accumulation/sum
  -> BF16 attn_out
```

旧的 projected-group dynamic INT8 quantization、`wo_b` INT8 matmul及
`wo_b_scale` 已全部移除。CPU reference 和 reference tests 同步采用该数学
边界，不再用旧 ABI 自洽验错。

### 31.4 首版 production gate

当前实现明确只接受以下范围，不能把静态全局 shape 误用于默认 TP4 权重：

- TP1：`n_heads == n_local_heads == 64`，
  `n_groups == n_local_groups == 8`；
- 非 DSA-CP 的 `AscendDSAImpl`；
- ratio=4、S=8、无 `skip_topk`、无 FlashComm gather；
- native DSA multistream overlap 关闭；
- `max_model_len <= 16384`；
- linear 无 bias、RMSNorm 无 anti-method/m4 bias；
- 四个 INT8 linear 必须是 symmetric W8A8 dynamic 且 offset 全零。

目标 checkpoint 默认 TP4 的 local H16/G2 不能直接接当前 H64/G8 static
artifact。支持 TP4 需要在 capture 前完成必要的 shard gather 或另做 local-shape
kernel specialization；首版选择 fail-fast，不在 launch/capture 内引入 collective。

### 31.5 Host 编译与测试证据

45 参数 B4/S8 已分别完成真实 PTOAS codegen：

```text
TRB artifact: /tmp/dsv4-csa-b4-prod45-trb.2fU5k6
HBG artifact: /tmp/dsv4-csa-b4-prod45-hbg.wpIsAK

params = 45
tensors = 41
scalars = 4
output_indices = [40]
source parameter order == compiled parameter order
```

第一次 TRB compile 捕获到同一 JIT 函数内 `kv_acc` 被用于不同 shape 的
single-assignment 类型冲突；将 projection、RMS pass 和 NOPE pass 的 accumulator
改为独立变量后，TRB/HBG 均完整编译。该问题是 DSL 变量类型约束，不是外部
non-contiguous tensor 问题。

迁移当时的专项 Host 测试结果为：

```text
126 passed, 14 warnings
ruff check: passed
```

随后第 30 节新增 9 条 production dispatch 测试；最终全量数字应以本节之后的
统一回归记录为准。

### 31.6 device0 TRB/HBG eager 与 ACLGraph 重验

两个 runtime 继续使用完全独立的新 Python 进程，只使用 A3 device0。结果：

| runtime | eager max abs | output identity | replay | replay max error | padding mismatch |
| --- | ---: | --- | ---: | ---: | --- |
| TRB | 0 | PASS | 4 | 0 | main/inner/indexer K/indexer scale 全 0 |
| HBG | 0 | PASS | 4 | 0 | main/inner/indexer K/indexer scale 全 0 |

两条路径都发生了真实 main/inner state 写回：

```text
main nonzero  = 32768
inner nonzero = 8192
indexer scale nonzero = 8192
```

这证明生产 45 参数 ABI 在 device0 上仍保持：预分配输出、caller-stream
顺序、Torch predecessor/PyPTO/Torch successor capture、四次连续 replay，
以及 page padding 安全。它仍然是零 projection weight 的 ABI/scheduling
smoke；不能替代下一步真实非零 weight 的 native/PyPTO 输出和六类 mutable
state 对比，也不能替代真实 `torch.ops.vllm.dsa_forward` layer harness。

## 32. 生产 `torch.ops.vllm.dsa_forward` 边界的 A3 TRB/HBG 闭环

第 31.6 节仍是直接调用 JIT program 的底层 ABI smoke。本轮新增
`tests/pypto_dsv4_decode_csa/a3_dispatch_smoke.py`，把相同 B4/S8/45 参数
artifact 穿过完整 production dispatch 链路：

```text
install_pypto_dsv4_decode_csa
  -> shared DecodeCSADeviceOwner/PyPTODSABackend
  -> torch.ops.vllm.dsa_forward(hidden, need_gather, output, layer_name)
  -> ForwardContext.no_compile_layers[layer_name]
  -> filter_metadata（五族按 key 排序）
  -> _build_kv_cache（真实六项 tuple）
  -> layer-owned PyPTO dispatch
  -> taskQueue L1 operator on caller stream
```

测试使用生产 custom-op schema，没有新增 public op，也没有直接从 harness 调
`program(...)`。synthetic layer object 按 `ops/dsa.py` 的真实 object graph 提供：

- compressed KV：`dsa_attn.kv_cache`；
- SWA KV：`swa_cache_layer.kv_cache`；
- main state：`compressor.state_cache.kv_cache`；
- inner state：`indexer.compressor.state_cache.kv_cache`；
- indexer K/scale：`indexer.k_cache.kv_cache` 的二元 tuple。

五族 metadata 直接引用 fixture 中已有的五张 block table、A3
`swa_slot_mapping[T,2]`、start/seq tensors；只在 install 冷路径创建并校验
`query_start_loc=[0,8,...,B*8]`。capture/launch 不做 H2D、tensor copy、sync
或 lazy prepare。

### 32.1 地址与 owner 验证

ordinary eager 首次调用执行显式 warmup，外部同步后才调用
`mark_warmups_quiesced()`。ACLGraph capture 使用另一对全新
`hidden_states/output` storage，因此同时验证：

- eager 和 capture 的 input/output data pointer 确实不同；
- eager `bind(retain=False)` 后 backend retained binding 数量为 0；
- capture `bind(retain=True)` 后恰好保留 1 个 graph-visible 参数快照；
- 捕获顺序为 Torch add predecessor -> production dsa custom-op -> Torch add
  successor；
- 四次 replay 均只执行已捕获任务，不再次进入 Python bind/launch。

### 32.2 device0 实测

TRB 与 HBG 仍分别使用全新 Python 进程，结果完全对称：

```text
runtime: TRB / HBG
eager_max_abs: 0
replay_count: 4
replay_max_error: 0
eager_and_capture_addresses_differ: true
eager_bindings_retained: 0
capture_bindings_retained: 1
```

这条结果关闭了“只有 standalone program 能进图、production custom-op hook
没有真机证据”的缺口。由于 projection weights 仍为零，它证明的是 production
对象边界、地址 patch、taskQueue、stream 和 graph owner，而不是非零算法精度。

### 32.3 真实 weight pack 的 Host 契约

新增 `test_weights.py`，通过 FakeTensorMode 以真实大 shape 验证 25 个 layer-static
launch tensor，而不在 Host 实际分配巨量 storage。覆盖：

- 四个 dynamic-W8 linear 的 INT8 weight、FP32 scale、zero offset；
- `wo_a` 实际 rank-3 loaded layout 的 transpose；
- BF16 `wo_b`；
- RoPE 截断到 16384；
- Hadamard 的 `1/sqrt(128)` 归一化；
- TP1、非 CP、最大位置、quant method、offset、scale、linear/norm bias 和
  dtype 的 fail-fast。

该文件 26 条测试全部通过。包含 dispatch、metadata、kernel ABI、cache adapter、
reference 和 weight pack 的最终统一 Host 回归为：

```text
161 passed, 14 warnings
ruff check: passed
ruff format --check: passed
git diff --check: passed
```

## 33. 45 参数四 bucket 矩阵、多 graph 生命周期与 Host trace

### 33.1 B4/B8/B12/B16 全静态 artifact

在第 31 节 B4 之后，继续对 B8/B12/B16 分别做 TRB/HBG 完整 PTOAS
codegen。两个 runtime 的结果均为：

```text
params = 45
tensors = 41
scalars = 4
output_indices = [40]
source parameter order == compiled parameter order
all tensor dimensions are positive static integers
```

artifact 根目录为：

```text
TRB: /tmp/dsv4-csa-prod45-trb-matrix.iKiRTn/{b8,b12,b16}
HBG: /tmp/dsv4-csa-prod45-hbg-matrix.u2ilxP/{b8,b12,b16}
```

因此四个正式 bucket 均不再依赖 B4 外推，也没有旧版本 public metadata 中的
`-1` 动态维度。

### 33.2 四 bucket 独立 A3 eager/ACLGraph

TRB 与 HBG 各自用三个 fresh subprocess 在 device0 运行 B8/B12/B16，连同
第 31 节 B4，八组 runtime/bucket 组合全部通过：

| bucket | tokens | main nonzero | inner nonzero | replay | max error | padding mismatch |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B4 | 32 | 32768 | 8192 | 4 | 0 | 四类全 0 |
| B8 | 64 | 65536 | 16384 | 4 | 0 | 四类全 0 |
| B12 | 96 | 98304 | 24576 | 4 | 0 | 四类全 0 |
| B16 | 128 | 131072 | 32768 | 4 | 0 | 四类全 0 |

TRB JSON 根目录：

```text
/tmp/dsv4-csa-prod45-trb-st.7Lixtn
```

HBG JSON 根目录：

```text
/tmp/dsv4-csa-prod45-hbg-st.6IcfFX
```

所有组合都保持预分配 output identity、`replay_max_error=0`。main/inner
实际写回量随 batch 线性增长，而 main/inner/indexer K/indexer scale 的页间
padding canary 始终没有被修改。

### 33.3 同 context 四 callable、四 graph 同存

`a3_dispatch_smoke.py` 进一步增加 multi-bucket production custom-op 场景：

1. 在同一个 `DecodeCSADeviceOwner` 中按 B4/B8/B12/B16 注册四个 static
   callable；
2. 四个 program 全部 compile 后只执行一次 context `prepare()`；
3. 分别 ordinary eager warmup，外部统一同步并 mark quiesced；
4. 每个 bucket 使用独立 hidden/output/source/final buffer 与独立 capture stream；
5. 四张 ACLGraph 同时存活，每张图保留一个地址快照；
6. 串行交替 replay：

```text
B4 -> B16 -> B8 -> B12 -> B16 -> B4 -> B12 -> B8
```

7. 销毁 B8 graph 后再次 replay B16，验证 graph 局部销毁不会破坏另一个
   callable/graph owner。

TRB 和 HBG 仍由 fresh process 分开运行，两边结果相同：

```text
compiled_bucket_count = 4
retained_capture_bindings = 4
replay_max_error = 0
survivor_replay_after_destroy = true
```

这条证据关闭了四 bucket 同 context 注册、地址隔离、交替 replay 和局部
destroy 的结构生命周期矩阵。由于 PyPTO 当前独占全部 AICore，所有 replay
严格串行；没有把多 stream 同时发射解释为支持并发执行。

### 33.4 Stateful decode trace 的纯 Host 基础

新增 `trace.py`/`test_trace.py`，建立与 device runner 解耦的不可变生命周期
模型：

- 五个独立 cache family：compressed、main state、inner state、indexer、SWA；
- block size 分别为 32/2/2/32/32，compressed/indexer 显式 ratio=4；
- deterministic lowest-free-block allocator；
- transactional admit/ensure/release，容量不足不污染已发布 state；
- 每 step 按 retire -> admit -> active request 前进 S=8 的顺序；
- 自动选择 B4/B8/B12/B16 最小可容纳 bucket；
- padded rows 的 request/seq/position 全为 -1、五族 ownership 必须为空；
- active ownership overlap、double-free、重复 request ID 和非法释放均 fail-fast；
- retire 后的 block 可被后续 request 确定性复用。

seeded churn generator 覆盖 4/32/128 step，并在前四步强制经过四个 bucket，
后续反复 shrink/grow 触发回收复用。新增 25 条 Host 测试全部通过。它目前
只证明调度/ownership 模型；下一步仍需把每个 `DecodeStep` materialize 成真实
五族 device metadata，并执行 native/PyPTO 双 state world。

## 34. ACLGraph v1/v2 capture 信号收口

生产 custom-op 入口中发现 vLLM Ascend 两套 ACLGraph runner 使用了
不同的 capture 标记位置：

- v1/piecewise ACLGraph 在 `ForwardContext.capturing` 上设置标记；
- v2 full-graph ACLGraph 在 Ascend `_EXTRA_CTX.capturing` 上设置标记。

原先若只读取第一个信号，v2 capture 会被错认为 ordinary eager，进而：

1. capture 前 warmup/quiescence 门禁失效；
2. binding 以 transient 而非 retained capture snapshot 形式提交；
3. graph replay 可能引用已经释放的 per-launch owner/address snapshot。

修正方式是只在 vLLM Ascend wrapper 边界合并两个信号：

```python
capturing = bool(
    getattr(forward_context, "capturing", False)
    or _EXTRA_CTX.capturing
)
```

然后作为显式 `bool` 传入 layer owner。PyPTO backend 仍然不查询
capture state、不取 graph handle，所以没有破坏“PyPTO 对 ACLGraph
capture/replay 无感”的 L1 边界。owner 对非 `bool` 值 fail-fast，避免
`None` 等三态值漏入生命周期决策。

新增 Host 回归用例专门构造
`ForwardContext.capturing=False` 但 `_EXTRA_CTX.capturing=True` 的 v2 情形，
确认生产 `torch.ops.vllm.dsa_forward` 路径把 `capturing=True` 传到
PyPTO owner，且不进入 native fallback。`test_dispatch.py` 现为 10 条全部
通过，对应文件的 Ruff check/format 也全部通过。

## 35. 四层 Decode CSA Capsule 的 Host 拓扑与 owner 模型

新增 `capsule.py`/`test_capsule.py`，先把真实 NPU 执行器之上的多层
调度、所有权和拓扑契约固定下来。它支持 1/2/4 层以及：

- `NNNN`：四层全 native；
- `PPPP`：四层全 PyPTO；
- `NPNP` 与 `PNPN`：mixed backend 路由；
- `PP_same`：前两层共享 logical callable，但保持独立参数地址、
  weight owner 和 cache owner；
- `PP_distinct`：每层 callable ID 和 logical function ID 都隔离。

每层 shell 的调用顺序固定为：

```text
residual_clone -> hc_pre -> rms_norm -> dsa_forward -> hc_post
```

层间 bridge 支持 deterministic `identity_residual` 和 `low_rank_ffn`。
`CapsuleExecutor` protocol 把真实 tensor 数学与拓扑调度解耦；当前
`HostRecordingExecutor` 只产生 opaque lineage/address evidence，不伪造数值或
精度结论。

每层有唯一 `layer_name`、`WeightOwner` 和 `MutableCacheOwner`。
`CallRecord` 记录 backend、callable identity、input/output/argument address 以及
weight/cache owner；mutable cache 的 append-only audit trail 能证明多次执行时
只有所属层更新自己的 state。

这一阶段新增 30 条 Capsule Host 测试并全部通过。它关闭的是
拓扑、调用顺序、callable/address 隔离和 owner 边界；四层数值对齐与
ACLGraph 上板仍必须由后续真实 NPU executor 证明。

## 36. 当前 Host 全量回归基线

合并 capture 双信号回归与 Capsule Host 契约后，执行：

```bash
pytest -q tests/pypto_dsv4_decode_csa
```

结果为 `217 passed, 14 warnings`。警告全部来自环境中 TorchScript
deprecated API，没有 CSA 用例失败。同时对 production package、`dsa.py`
和全部 CSA test support 执行 Ruff check/format-check，37 个文件全部通过。

## 37. partial bucket 复审发现的真实缺口

在把 Host trace 向生产 metadata 接续时，重新对照
`model_runner_v1.py` 和 `AscendDSAMetadataBuilder` 的 graph padding 逻辑，
发现当前上板的 B4/B8/B12/B16 都是“整个 bucket 全 active”证据，
还不能代替 `num_reqs_actual < bucket_size` 的验收。

生产 graph padding 的关键行为是：

- `num_reqs_padded` 决定 block table、query-start-loc 和 hidden buffer 的
  static bucket shape；
- runner 把 padding token 的 common `slot_mapping` 填为 `-1`；
- runner 把 padding request 的 block-table row 填为 `0`；
- device `seq_lens[num_reqs:]` 被填为 `0`；
- DSA builder 的 `num_reqs_actual` 保留真实 request 数，并把 trailing
  `start_pos` 和自己的 block-table row 清零。

而当前 PyPTO adapter 仍强制
`num_reqs_actual == spec.batch`，kernel 中 main/inner ratio-4 compressor 又会对
所有 static request row 执行 state/cache write。因此如果只放宽 Host 校验，
padding row 会借用清零后的 block-table row 误写 physical block 0。
这不是一个可以只靠文档标注掉的边角问题，而是 partial-bucket
生产接入的 correctness blocker。

当前拟定的闭环原则是：

1. 不使用 Host scalar `num_reqs_actual` 作为 replay 时动态信号，
   因为 graph node 可能固定 capture 时的 scalar 快照；
2. 使用 runner 每步已经更新的 device metadata 作为 active/padded
   权威信号，优先候选是 static `[bucket]` `kv_seq_lens > 0`；
3. main/inner state 和 compressed/indexer cache 的每个 write 都必须受
   device-side active guard 保护；
4. SWA write 继续尊重 negative slot sentinel，或在 active guard 下从
   block table + position 等价恢复物理地址；
5. 新增 padding canary 用例必须同时检查 main state、inner state、
   compressed KV、indexer K/scale 和 SWA KV，不只看 output。

在该 device-side guard 和 A3 canary 通过前，partial bucket 保持
**unsupported/fail-fast**，不把 Host trace 的 padded-row 不变量误写成生产
kernel 已支持。

## 38. 性能证据 schema 与采样契约

新增 `benchmark.py`/`test_benchmark.py`，先固定性能实验的数据
边界，避免后续上板时只报一个 best latency 或把 compile/warmup
混进 steady state。

主要契约如下：

- `BENCHMARK_SCHEMA_VERSION = 1.0.0`；
- backend/mode/workload/specialization key 不可变，包含 bucket、S、
  context position、layer 数、actual request 数、topology 和 trace seed；
- warmup 与 steady sample 两个 phase 完全分离，warmup 原始值保留
  但不进入统计；
- `SamplingPolicy` 强制 20～50 次 warmup 和 100～1000 个
  steady sample；
- 两 case 支持 ABBA，任意数量 case 支持轮转；同一交替计划
  强制 workload 完全相同；
- 每次调用的 `host_enqueue`、`metadata_update`、`device_span`、
  `graph_replay` 独立分栏；eager 的 graph replay 显式记为 N/A；
- 额外逐样本计算 metadata 计入/不计入的 Host 入口耗时，
  不用两组不同实验的 summary 相减；
- steady 统计包含 min/mean/population std/p50/p90/p99；
- JSON 保留 raw samples + summary，CSV 按 metric 分行，两者都携带
  schema version、key 和 environment metadata。

accumulator 对缺失、多余、重复、负 duration、顺序错乱以及未完成
schedule 就 finalize 全部 fail-fast。新增 25 条 Host 测试全部通过，
Ruff check/format-check 也通过。这一步只建立了数据 schema 与采样
状态机；真实 native/PyPTO launch、NPU event/profiler 和同卡 ABBA 数据
仍由后续 A3 runner 填充。

## 39. 真实 native/PyPTO 双 state world 对比路径

### 39.1 测试对象

新增 `native_fixture.py`、`a3_native_compare.py` 和
`test_native_fixture.py`。这条路径不加载完整 checkpoint 和模型，但也不用
`SimpleNamespace` 伪装 native 实现：

1. 从本地 HF config 构建真实 `VllmConfig`/`ModelConfig`；
2. 构建 TP1、ratio-4 的真实 `DeepseekV4Attention`；
3. 使用真实 `AscendModelSlimConfig`，四个 linear 走
   `AscendW8A8DynamicLinearMethod`，七个 linear 走 FLOAT method；
4. 用固定 seed 初始化非零 synthetic 权重，所有 W8 offset 强制为
   zero，然后调用真实 `process_weights_after_loading`；
5. 使用五个真实 `AscendDSAMetadataBuilder` 生成 ratio-4 metadata；
6. native 与 PyPTO 各自持有一套地址不共享、初值完全相同的
   compressed KV、SWA KV、main state、inner state、indexer K 和
   indexer scale；
7. 两边都经过同一生产 `torch.ops.vllm.dsa_forward` 入口，
   native 先完成，再 install PyPTO owner，禁止失败后 fallback；
8. 比较 output 以及六类 mutable state 中任意一边相对 initial
   发生变化的 union write set；INT8 状态要求精确相等。

当前首轮只定义 B4/S8 全 active 的 eager 数值对比，不把它
外推成 partial bucket、ACLGraph 或四层已通过。Host 契约用例为
7 条，Ruff 与 pytest 全部通过。

### 39.2 device0 首次启动发现的 harness 顺序问题

首次 TRB fresh-process 命令在 device0 执行，在任何 CSA kernel 或
PyPTO compile 前失败：

```text
initialize_model_parallel(...)
  -> get_current_vllm_config()
  -> AssertionError: Current vLLM config is not set
```

原因是新 runner 先调用 `initialize_model_parallel`，后构建/设置
synthetic `VllmConfig`；当前 vLLM 2.12 要求 model-parallel 初始化本身也
必须在 `set_current_vllm_config(...)` 作用域内。这是 harness
lifecycle 顺序问题，与 tensor contiguous、CSA kernel 和 PyPTO runtime 无关。
修正方向是先创建 config owner，然后在 config context 内初始化
TP1，并只在分布式初始化成功后执行对称 destroy。

### 39.3 FLOAT 与 W8 Linear method 的真实包装层级

调整 config/model-parallel 顺序后，第二次 device0 fresh-process 运行在
kernel 发射前被 fixture 自检拦下：

```text
indexer.weights_proj: expected AscendUnquantizedLinearMethod,
got UnquantizedLinearMethod
```

这不是 production 权重类型错误，而是 harness 把所有 FLOAT 项误认为
同一层 wrapper。真实构造逻辑中：

- compressor/indexer-compressor 和 output projection 的六个 FLOAT linear
  使用 `AscendUnquantizedLinearMethod`；
- `indexer.weights_proj` 显式以 `quant_config=None` 构造，保留 vLLM
  上游的 `UnquantizedLinearMethod`。

已将两类方法分开做 exact-type 断言，避免为通过 fixture 而放宽真实
quant/FLOAT 契约。

### 39.4 metadata builder 中 Host/Device tensor 边界

第三次运行进入真实 `AscendDSAMetadataBuilder`，在比较
`is_prefilling` 与 `query_start_loc_cpu` 时报 Host/NPU device 不一致。根因是
synthetic common metadata 把 `is_prefilling` 建成了 NPU bool tensor，而 production
builder 的该字段是与 `query_start_loc_cpu` 配合的 Host metadata。

修正后该字段留在 CPU；block table、seq lens、start position、slot
mapping 等真正的 launch metadata 仍在 NPU。这一修正也说明本轮遇到的
不是“PyPTO 不支持非连续 tensor”。

### 39.5 `_C_ascend` custom-op 注册必须复用 production bootstrap

第四次运行在 builder 调用
`npu_sparse_attn_sharedkv_metadata` 时发现 `_C_ascend` namespace 没有对应 op。
本地 binary 中实际包含该符号，问题是 standalone harness 没有经过 worker
启动时的 custom-op 注册入口。

修正为 device 设置后显式调用 production `enable_custom_op()`，并对返回值
fail-fast；不在测试中伪造 metadata op 或将真实 builder 替换为
`SimpleNamespace`。

### 39.6 synthetic model 构造必须复刻 loader 的 default dtype

第五次运行首次进入真实 native CSA 计算，在
`npu_rms_norm_dynamic_quant` 报错：

```text
x: BF16, gamma: FP32
supported combination requires BF16 gamma for BF16 x
```

根因是 synthetic fixture 虽然在 target NPU device context 中构造 module，但没有
复制 vLLM model loader 外层的
`set_default_torch_dtype(model_config.dtype)`。因而 RMSNorm 参数按 PyTorch
默认 FP32 创建，与真实 BF16 模型不同。Qwen3-14B 正常加载路径本来就在
model dtype context 内构造，因此没有这个 harness 人造问题。

当前修正是在 `set_current_vllm_config`、BF16 default-dtype 和 target-device
三个 context 内构造 `DeepseekV4Attention`，然后再执行 synthetic weight
fill 和 `process_weights_after_loading`。APE、`attn_sink`、W8 weight/scale
等显式 dtype 保持原样，不做粗暴全局 cast；post-load 后额外断言
q/kv/main-compressor/inner-compressor 四类 norm gamma 全部为 BF16。

### 39.7 production bootstrap 不只是注册 C custom op

在 dtype 修复后继续对照真实 worker 启动链路，发现只调用
`enable_custom_op()` 仍是一个“半 bootstrap”状态：C++ custom op 已可用，但
Ascend pluggable layer registry 还没有按 production worker 注册。这会造成两个假象：

- `wo_a` 被构造成普通 upstream linear，不是真实
  `AscendColumnParallelLinear`；
- `indexer.weights_proj` 保留 upstream `UnquantizedLinearMethod`，而真实
  Ascend registry 生效后它与其他 FLOAT linear 一样使用
  `AscendUnquantizedLinearMethod`。

harness 现在复用 `AscendWorker` 的
`register_ascend_customop(vllm_config)` 完成 class/op 整体注册，然后再构造
attention。device0 实测确认 `wo_a` 的 loaded layout 为
`[8, 4096, 1024]`，七个 FLOAT linear 均走 Ascend method。因此之前的
upstream-method 结论已撤回，fixture 断言改为 production 事实。

### 39.8 真实 native 首次完整执行与 autograd 边界

补齐 production bootstrap 后，native TP1 ratio-4 CSA 在 device0 完整执行成功，
output 为 finite 且非零，说明 synthetic module、ModelSlim post-load、五族
metadata builder 和六族 cache world 已越过 native 启动阶段。

同一进程首次进入 PyPTO warmup 时，L1 binding 拒绝了
`gamma_cq`：真实 `AscendRMSNorm.weight` 是 `nn.Parameter`，默认
`requires_grad=True`。`torch.inference_mode()` 只改变新操作的 autograd 记录，不会
回头改写已有 Parameter 的 flag；而 `.contiguous()` 对已连续 tensor 是 no-op，
因此原 weight pack 可能把 Parameter 本身交给 L1。

这不只是 fixture 问题，也是真实 checkpoint 接入 blocker。正式 weight
pack 已将 detach 纳入 cold-path 契约：

- matrix/exact/scale/RoPE/wo_a/hadamard 的 packed tensor 全部显式
  `detach()` 后再 canonical contiguous；
- 保留 source layer 作为 strong owner，detach 只去掉 autograd 边，不破坏
  storage 生命周期；
- Host weight test 将所有 floating fake Parameter 设为
  `requires_grad=True`，并断言 27 个 packed static tensor 全部
  `requires_grad=False`。

相应 Host 回归为 `26 passed`，Ruff check/format-check 通过。synthetic
fixture 也会显式进入纯推理对象状态，但正式修复不依赖 fixture 代劳。

## 40. Partial bucket 的 device-side 生命语义

### 40.1 不再把 dynamic SWA slot mapping 塞入静态 L1 ABI

对照 `model_runner_v1.py` 与 `AscendDSAMetadataBuilder` 后确认，uniform
decode graph 的 bucket 为 B、实际 request 数为 A 时：

- query-start-loc 仍为 B 个完整 S=8 row；
- `num_decodes=B`，但 `num_actual_tokens=num_decode_tokens=A*8`；
- `num_reqs_actual=A`；
- `kv_seq_lens[A:B]` 每步在 device buffer 中清零；
- 五张 block table 的 padded row 为零；
- A3 builder 返回的 SWA slot mapping 只是 `[A*8, 2]`，不是
  `[B*8, 2]`。

因此 public L1 签名已删除 `swa_slot_mapping`，从 45 槽收敛为
44 槽（40 tensor + 4 runtime scalar，output index=39）。SWA write 由 AICore 在
runtime 中使用
`swa_block_table[request, position//32] + position%32` 恢复地址。这不新增
Host tensor，也不会把 A 固化成 capture-time scalar。

### 40.2 `kv_seq_lens > 0` 是当前纯 decode specialization 的 active guard

当前 S=8 纯 decode 语义下，合法 active request 在该步结束时 seq-len 至少为
8，而 graph padding 稳定将 tail 置零。所以复用已有 `[B]` device tensor
`kv_seq_lens` 作为 guard，不新增 active-mask ABI。

已完成的 source 修改点是：

- SWA write 在读 padded table row 前 guard；
- main compressor 的 state scatter/pool 对 inactive request 直接填零内部 row
  并跳过所有 state page 读写，compressed KV store 再做一次 guard；
- inner compressor 对称处理，indexer K/scale store 共享同一 guard；
- indexer score/top-k 原本就以 seq-len 计算 visible length，inactive row 保持
  `-1` top-k；
- sparse attention 的 padded SWA block 标为 invalid，不读 block0，最终
  padded `attn_out` 显式写零，避免 replay 残留。

Host metadata/ABI 契约已覆盖 A 一致性、`1<=A<=B`、A*8 counters、
44 槽顺序和 hot-bind 不读 device value；定向结果为 `50 passed`。
当前尚不把 partial bucket 标为完成：还必须通过 A3 TRB/HBG compile，
并在同一 captured graph 内轮换 A，证明 HBG lowering 保留的是 device
predicate 而不是 build-time 常量。

### 40.3 44 槽 guard 版 TRB/HBG codegen

在 A3/a2a3 platform 上对 B4/S8 新 entry 分别执行 annotation-only
compile，两种 runtime 均完成 PTOAS codegen：

```text
compiled tensormap_and_ringbuffer 44 (39,)
compiled host_build_graph          44 (39,)
```

这证明 `continue`/device conditional、六类 side-effect child 新增的
`kv_seq_lens` 传递和 44 槽 ABI 可以被 TRB/HBG compiler 接受。它仍只是
codegen 证据，不等于 graph replay 时 predicate 已实测为动态。

合并 ABI、weight detach 与 kernel source 修改后，全部
`tests/pypto_dsv4_decode_csa` Host 回归为 `263 passed, 14 warnings`；警告仍全部是
环境 TorchScript deprecated API。

44 槽版 TRB B4 full-active zero fixture 也在 device0 重新完成：ordinary eager、
Torch predecessor -> PyPTO -> Torch successor capture 和 4 次 replay 全部通过，
`output_is_preallocated=true`、`replay_max_error=0`，四类 page-strided storage 的
padding mismatch 均为 0。这条是 full-active 回归，partial A 动态轮换仍由下一条
canary matrix 验收。

### 40.4 Partial-bucket 的可执行 canary harness

新增 `a3_partial_bucket_smoke.py` 和 `test_partial_bucket.py`，将第 40.1～40.3
的设计结论收敛为一条 production custom-op 验证路径：

- 同一个静态 B 下生成 A<B 的 full-B metadata：`query_start_loc[B+1]`、
  五张 `block_table[B,*]`、`start_positions[B]`和 `kv_seq_lens[B]`；
- active row 只引用互斥的非 0 physical block，padded row 全部归零，
  所以 block 0 可以作为缺失 device guard 的强 canary；
- capture 只绑定一组稳定 device tensor 地址，replay 前只就地 copy
  五张 table、start 和 seq-len payload，不重走 Python binder；
- Host counters 明确保留 capture 时的 A，不假装 graph replay 会重新读取
  `num_reqs_actual`；device 真正的 active predicate 仅来自 `kv_seq_lens`；
- 六类 mutable state 均检查 block-0 canary，四类 page-strided storage
  同时检查 page padding canary；padded output 必须 exact zero；
- SWA 的 `table[row, position//32], position%32` 公式已在 31/32、63/64
  边界与 native slot mapping 精确一致。

新增 17 条 Host UT 全部通过；与 metadata/dispatch/JIT ABI/fixture 联合为
57 passed，Ruff check/format-check 通过。本节只表示 harness 已可执行，
TRB/HBG 的 A 动态轮换仍必须分别在 fresh A3 进程运行后才能标记为
supported。

## 41. Native full-width RoPE ABI 与二次 interleave 错误

### 41.1 差异定位证据

真实 TP1/ratio-4 native 与 PyPTO 的非零对比首次完整执行后，输出
已在 `atol=rtol=0.1` 内，但 mutable state 仍有大误差：SWA KV 约
4.98、compressed KV 约 4.02、indexer raw INT8 约 55。逐段比对证明：

- SWA 的 32/32 write rows、compressed 的 8/8 rows、indexer 的 8/8 rows
  完全相同，没有 only-native/only-PyPTO row；
- SWA NOPE `[0:448]` bit-exact，全部大误差都集中在 RoPE `[448:512]`；
- compressed NOPE 的 max 仅 `0.0078125`、mean 约 `2.2e-6`，大误差同样
  集中在 RoPE 区域；
- position 0 行不显示该差异，首个差异出现在非零 position，进一步
  排除了 page/slot/write-set 错位。

根因在 `ComplexExpRotaryEmbedding`：native `rope_dsv4.py` 已经用
`repeat_interleave(2)` 构造 full 64-column cache，即
`[c0,c0,c1,c1,...]`。weight pack 只做 reshape/cast，并没有把它还原成
half-width 表。native Q/KV rotary、compressor metadata 和 compressor 都逐列直接消费
这 64 列，只对 sin 施加交错符号。

原 PyPTO 在三处又把 `j>>1` 当作 gather index：Q/KV projection 中从
full row 二次 gather；token/indexer 和 compressor 只取前 32 列再 duplicate；
sparse-attention 的 inverse RoPE 也重建了一次。这会把正确的相邻 2 列频率
变成相邻 4 列频率，与上述分段差异完全吻合。

### 41.2 Production 修正

保持 L1 public ABI 不变，`freqs_cos/freqs_sin` 仍是 native full-width
`[max_position,64]`；内部改为：

- Q/KV projection：cos 直接 FP32 cast/copy，sin 只乘
  `[-1,+1,-1,+1,...]`，保留 `j^1` lane swap；
- token/indexer 与 main/inner compressor 的 RoPE row 都改为 full 64-column，
  `rope_interleave` 不再 gather/duplicate；
- sparse-attention inverse RoPE 直接消费 full row，仅保留 conjugate
  `[+1,-1,...]` sign 和 `j^1` swap；
- CPU mathematical reference 和 fixture 同步改为 full-width native contract，
  新增一条用不同频率对显式证伪“二次 interleave”的回归测试。

修改后 B4/S8 的 TRB 与 HBG annotation-only A3 codegen 都已成功；
`tests/pypto_dsv4_decode_csa` 全量 Host 回归为
`284 passed, 14 warnings`，警告全部来自环境 TorchScript deprecated API。
真实非零 native/PyPTO 数值复验将在 fresh device0 进程中进行，结果不在本节
预写。如 RoPE 修正后 indexer raw INT8 仍不 close，下一个已知候选是
Hadamard 的舍入链：native 为 BF16 linear 后乘 `1/sqrt(128)`，当前
PyPTO 是预归一化 BF16 weight 与 FP32 accumulation，必须根据新证据决定是否继续修正。

## 42. 真实非零 TP1 对齐与持久 SWA cache 伪依赖

### 42.1 对齐 native 数值边界

第 41 节的 full-width RoPE 修正只是第一层原因。继续逐段对比真实
`DeepseekV4Attention` 后，还补齐了以下 production 数值边界：

- native `ComplexExpRotaryEmbedding` 保存的是 FP32 full-width cos/sin；正式
  weight pack 与 L1 ABI 因而保持 FP32，不再提前降成 BF16；
- Q-A 反量化结果先物化为 BF16，再进入 RMSNorm/动态量化；
- Q/K 的 RMSNorm 结果在 RoPE 前先物化为 BF16；
- indexer query 投影也在 RoPE 前保留 native 的 BF16 物化点；
- Hadamard weight 保持未归一化 BF16 `+1/-1`，cube 的 FP32 累加结果先
  物化 BF16，再乘 `1/sqrt(128)`，随后再次物化 BF16并动态量化；inner K
  使用同一边界。

此外，真实 CANN compressor 把 compressor state 的 physical block 0 当作
特殊/不可写区域。最初 synthetic fixture 从 block 0 分配 main/inner state，
会制造 native/PyPTO 假差异；fixture 已改为 main/inner state 从 physical
block 1 开始，block 0 保留为 sentinel。SWA、compressed 和 indexer 的寻址
规则没有因此被一并改写。

一次失败尝试是把 Hadamard 的 BF16/scale 边界融合进 cube scope。该方案在
A3 codegen 暴露明确的 Vec 容量超限：

```text
Vec buffer usage 311296 > 188416
```

原因是融合后形成约 262144-byte 的跨核 ring buffer。最终保留 cube-only
FP32 GM handoff，并在后继 vector quant scope 内完成 BF16/scale 边界；这既
匹配 native 数值，也不超过 A3 Vec 容量。失败方案没有保留在正式源码中。

### 42.2 为什么此前“单步通过”仍不够

完成上述修正后，最初的 fresh TRB 单步对比显示六类 state 都 close：SWA
bit-exact，indexer K/scale bit-exact，compressed 最大误差 `0.0078125`，输出
也在预设 `atol=rtol=0.1` 内。但当时 harness 把 PyPTO 的第一次
`operator.warmup()` 直接当成正确性调用，所以它只证明 warmup 路径，不能
证明后续普通 `operator.launch()` 与 warmup 等价。

为消除这个混淆，`a3_native_compare.py` 改成：

1. 在安装 owner 后保存六类 PyPTO cache 的独立 device clone；
2. 执行一次牺牲性 warmup；
3. 由 caller 做 device synchronize；
4. 从 device clone 原位恢复六类 cache，并再次由 caller synchronize；
5. 所有正式 correctness step 都只走普通 launch；
6. `--sync-between-correctness-steps` 只用于诊断连续普通 launch，不影响
   warmup 的必需外部 quiescence。

按这个严格流程，普通 launch 首次暴露 SWA RoPE 最大误差 `5.484375`；是否
在两个普通 launch 之间同步对结果没有影响。NOPE、indexer、其他 state 的
write-set 和地址都一致，因此可以排除：

- warmup 未外部同步；
- HostArgs/Tensor lease 提前失效；
- 两次 Host enqueue 异步重叠；
- step 1 的 slot mapping 覆盖 step 0；
- 普通的非连续 tensor ABI 问题。

逐 step snapshot 进一步证明：损坏发生在 warmup 后的第一个普通 launch
本身，而不是第二个 position 才出现。该 launch 只更新 32 个合法目标 row，
没有写目标集合之外的 row。

### 42.3 `kv_touch` 因果实验

`decode_sparse_attn_csa.py` 继承了历史 reference 的一个 WAR marker：

```python
ori_kv_flat[0:T, 0:HEAD_DIM] = ori_kv_flat[0:T, 0:HEAD_DIM]
```

它原本试图弥补 scalar-driven gather 没有自动形成完整 cache dependency 的
问题，但并非抽象 marker。B4 artifact 的生成代码证明它会提交一个真正的
`kv_touch` AIV kernel，而且这里的全局 `T` 是 128，不是运行 bucket 的
32 个 token；每次实际 load/store `128 x 512` 个 BF16 持久 SWA 元素。

旧 orchestration 中：

- `csa_cache_writeback` 写 `kv_cache`；
- `kv_touch` 对同一 storage 做大块 `add_inout` 自拷贝；
- `qk_pv` 只显式依赖 `qk_plan_tid`，没有显式依赖 writeback TaskId。

在 fresh 进程中仅临时删除 `kv_touch`，其余代码、地址、position、warmup/reset
流程不变，再执行一个普通 launch。结果 SWA 最大误差从 `5.484375` 降为
`0`，其余 state 维持原结果。这构成了因果 A/B，而不只是基于源码的猜测。

### 42.4 正式依赖修复

正式代码没有简单删除 WAR marker 后依赖隐式 alias 分析，而是建立真实的
producer/consumer edge：

```text
csa_cache_writeback --TaskId--> qk_pv dynamic gather
```

具体改动为：

- `csa_cache_writeback` 从 `for ... in pl.spmd(...)` 改为
  `with pl.spmd(...) as writeback_dep`，在 task body 内用
  `pl.tile.get_block_idx()` 取得 lane；
- `writeback_dep` 作为 `pl.Scalar[pl.TASK_ID]` 只在内部穿过
  `sparse_attn_csa` / `sparse_attn_csa_heads`；public 44-slot L1 ABI不变；
- `qk_pv` 的显式依赖改为
  `deps=[qk_plan_tid, writeback_dep]`；
- 永久删除会写用户持久 cache 的 `kv_touch` 自拷贝。

生成的 A3 orchestration 已直接确认：writeback task 的返回值被保存为
`PTO2TaskId writeback_dep`，`qk_pv` 的 dependency array 长度为 2，依次包含
`qk_plan_tid` 和 `writeback_dep`，artifact 中不再生成 `kv_touch` kernel。

正式 TRB B4/S8 的第一条严格复测命令为：

```bash
python -m tests.pypto_dsv4_decode_csa.a3_native_compare \
  --runtime tensormap_and_ringbuffer --device 0 \
  --warmups 0 --iterations 1 --start-position 0 \
  --correctness-steps 1 --master-port 29653
```

它执行的是“牺牲 warmup + cache restore + 一个普通 launch”，结果：

- output：`max_abs_error=0.036376953125`，close；
- compressed KV：`max_abs_error=0.0078125`，close；
- SWA KV：`max_abs_error=0`，bit-exact；
- main state：`max_abs_error=2.60770320892334e-08`，close；
- inner state：`max_abs_error=2.2351741790771484e-08`，close；
- indexer K：`max_abs_error=0`，bit-exact；
- indexer scale：`max_abs_error=0`，bit-exact；
- 最终 `close=true`。

该命令只有一个 timing iteration，不形成性能结论。两步无中间同步、HBG
codegen/执行与 ACLGraph 复验在本节记录时仍在继续，不能由这条 TRB 单步
结果提前外推。

### 42.5 两步假差异：真实 metadata 的进程级 RoPE runtime buffer

正式删除 `kv_touch` 并补上 writeback dependency 后，单步已经严格通过，
但早期的两步诊断仍报 SWA RoPE 最大误差 `5.484375`。这次差异不是
第二个 kernel 生命周期问题，而是测试 harness 不符合真实 metadata 的时序。

逐步六 cache snapshot 先证明：

- native 与 PyPTO 在 step0/step1 都只改本步目标 row，
  `changed_outside_target_rows=0`；
- step1 的真实 SWA slot mapping 是每个 request 的 offset `8..15`，
  Host contract test 同时检查二维 native mapping 和 linear mapping；
- native/PyPTO 的 write-set 一致，所以不是 address 或 page-table 计算错误；
- `npu_sparse_attn_sharedkv` 路径也没有在本诊断中改写目标集合之外的
  `ori_kv` row。

根因是 `AscendDSAMetadataBuilder` 的 decode 路径调用
`get_cos_and_sin_dsa(..., use_cache=True)`。该函数会把当步 cos/sin copy 到
`RopeGlobalState.runtime_buffer`，metadata 只持有 `buf[:num_tokens]` view。原 harness
先构造所有 `native_metadatas`，因此 position=8 的 step1 build 会覆写 position=0
的 step0 view；之后第一个 native forward 实际用了 position=8 RoPE，却按
position=0 slot 写回。这恰好解释了“单步过，两步的 step0 已经错”。

harness 现改为真实 production 时序：每步 materialize native metadata 后立即
forward，不在前一步消费 RoPE view 前构造后一步。每个 bundle 仍持有
真实 builder/buffer owner。SAS/QLI metadata buffer 由每步新建的 builder 自有，
没有同样的跨 builder 全局 view 覆写问题。PyPTO 路径消费的是完整
RoPE table 与 device `start_positions`，不依赖该 per-token runtime-buffer view。

修正后的 fresh-process 两步无中间同步命令：

```bash
python -m tests.pypto_dsv4_decode_csa.a3_native_compare \
  --runtime tensormap_and_ringbuffer --device 0 \
  --warmups 0 --iterations 1 --start-position 0 \
  --correctness-steps 2 --master-port 29657
```

结果 `close=true`：

- output `max_abs_error=0.036376953125`，close；
- compressed KV `max_abs_error=0.0078125`，close；
- SWA KV 两步 active elements `32764`，`max_abs_error=0`，bit-exact；
- main/inner state 分别为 `2.9802322387695312e-08` / `2.2351741790771484e-08`；
- indexer K 与 scale 都 `max_abs_error=0`，bit-exact。

单步串行稳定计时（`warmups=5, iterations=20`）为 native
`1.9573308527469635 ms`、PyPTO TRB `1.106735342182219 ms`，本次样本约
`1.7686x`。这是 synthetic TP1 B4/S8 算子级串行对比，不是整模型性能结论。

当前签名下的 HBG 纯 Host codegen + assemble 也已通过：

- runtime: `host_build_graph`；
- public parameters: 44，output index: `[39]`；
- final callable: 40 tensor signatures + 4 scalar signatures。

该 HBG 结论只证明当前 `writeback_dep` 内部签名能完成 A3 Host 生成与
装配，本轮没有宣称 HBG 真机执行或 ACLGraph replay 已复验。最终定向
Host suite 为 `335 passed`，Ruff check/format 通过。

### 42.6 四步连续 decode state

两步闭环后继续使用同一份 mutable state，在不插入 step 间外部
sync 的情况下执行 positions `0/8/16/24`：

```bash
python -m tests.pypto_dsv4_decode_csa.a3_native_compare \
  --runtime tensormap_and_ringbuffer --device 0 \
  --warmups 0 --iterations 1 --start-position 0 \
  --correctness-steps 4 --master-port 29659
```

fresh device0 结果 `close=true`：

- 四步 output 合并后 `max_abs_error=0.036376953125`；
- compressed KV：`16384` 个 active element，`max_abs_error=0.0078125`；
- SWA KV：`65528` 个 active element，`max_abs_error=0`；
- main/inner compressor state 分别为 `2.9802322387695312e-08` 和
  `4.0978193283081055e-08`；
- indexer K：`4055` 个 active element，bit-exact；
- indexer scale：`32` 个 active element，bit-exact。

这条结果同时验证了修正后的 per-step native metadata materialization 没有再把
后续步 RoPE view 提前覆写到前一步。

### 42.7 Nonzero 单 custom-op ACLGraph native/PyPTO 对比

新增 tests-only `a3_native_aclgraph_compare.py`，不改 production kernel。它构造两张
完全独立的 ACLGraph：

- native graph 和 PyPTO graph 的 capture region 各只包含一个
  `torch.ops.vllm.dsa_forward`，没有 Torch predecessor/successor 来掩盖单 op 边界；
- 两侧共用同一个真实 TP1 ratio-4 layer/权重，但 input、output、capture stream
  和六类 cache world 全部地址隔离；
- native 先 warmup/capture，然后才将同一 layer rebind 到 PyPTO world 并 install
  owner；已 capture 的 native node 仍持有自己的 device address；
- native 和 PyPTO 都显式证明 eager warmup IO 地址与 capture IO 地址不同，
  其中 PyPTO 完成了真实 warmup -> capture address patch；
- capture 前先将 nonzero `base_hidden` copy 到 capture input 并 sync，避免 graph
  capture 执行时把 `empty` 垃圾值送入 dynamic quant/compressor；
- capture 后先同步对应 stream，再原位恢复六份 cache；每次 correctness replay
  前两个 world 都恢复同一逻辑初值，然后只 patch 固定 input address 的内容；
- 销毁顺序为先 sync、reset 两张 graph，再 close PyPTO backend；
- PyPTO capture 后 backend 精确保留一个 capture address snapshot。

fresh device0/TRB 使用 input scale `0.5/-1.0/1.75` 连续做三次独立初值
replay，三次都 `close=true`：

- output 每次 `max_abs_error=0.036376953125`，在当前明确的
  `atol=rtol=0.1` 验收线内；
- SWA KV 每次 bit-exact；
- indexer K/scale 每次 bit-exact；
- compressed KV 最大误差为 `6.103515625e-05`；
- main/inner state 三次最大误差上界分别为
  `5.21540641784668e-08` / `4.470348358154297e-08`。

首个单次 timing 样本是 native `0.873992 ms`、PyPTO `0.849992 ms`。另一个
fresh process 中使用 `warmups=5, iterations=20`、且仍按 native 先、PyPTO 后串行
测得 native `0.667953 ms`、PyPTO `0.803727 ms`。两组样本的相对顺序不一致，
因此当前不用它们声称稳定 speedup；要形成稳定性能结论，仍需 ABBA/多轮
交错采样以排除顺序、频率与首次 replay 影响。这比只报一个有利数字更可靠。

随后在另一个 fresh device0 进程中使用同一 runner 完成 HBG 上板。三组
input scale 的 output 和六类 state 同样全部 `close=true`，SWA/indexer 仍为
bit-exact，双 world/IO 地址隔离、warmup -> capture address patch 和单个 retained
binding 全部为真。两份完整日志分别为：

- TRB：`/tmp/csa_native_aclgraph_first.log` 和
  `/tmp/csa_native_aclgraph_perf.log`；
- HBG：`/tmp/csa_native_aclgraph_hbg_first.log` 和
  `/tmp/csa_native_aclgraph_hbg_perf.log`；两次均显式记录 `exit_code=0`。

HBG 的性能结果不能忽略：`warmups=0, iterations=1` 时 PyPTO graph 为
`305.486112 ms`；`warmups=5, iterations=20` 后仍为 `319.842420 ms`，而同进程
native graph 为 `0.671216 ms`。因此这不是一次冷启动 cache miss，而是当前
HBG L1 路径的确定性性能边界。本节只证明 HBG ACLGraph 的功能/状态正确，
不将它标记为性能可用；后续必须对 HBG 的 per-replay host-build/H2D 和 device
orchestrator 时间做独立 profile。

### 42.8 Partial-bucket 的 block 0 踩踏与 AICore runtime `continue` 降低缺陷

为验证固定 bucket `B=4` 内 `A=num_reqs_actual` 在 replay 间变化时，
padding request 不会通过全 0 block table 写住理 block 0，
`a3_partial_bucket_smoke.py` 对 six-cache world 的 block 0 先写入非零 canary，
并在 `seeded` / `installed` / ordinary eager / graph capture / 每次 replay
后逐阶段 snapshot。测试还不仅检查 fixture 原始 table，而是：

- 对每个 payload 断言 active table row 全部非 0，padding row 全部为 0；
- 直接调用 production `bind_decode_csa_native_metadata`；
- 断言 launch binding 的七个 metadata tensor 正是
  `DecodeStepMetadataBufferOwner` 持有的对象，且 `data_ptr` 完全相同；
- 每次 replay 只原位 patch 固定地址 metadata，不 rebind graph node。

首个 TRB 真机运行没有通过 canary，而且破坏范围很有辨识度：

```text
compressed_kv:          1024
swa_kv:                    0
main_compressor_state:  4096
inner_compressor_state: 1024
indexer_k:               256
indexer_scale:             2
```

不能把这些 mismatch 当成“block 0 本来就可写”而放宽断言。
生成 artifact 给出了更直接的证据：source 中以下形态的 dynamic
control flow：

```python
if inactive:
    zero_scratch_rows()
    continue
write_persistent_state_or_cache()
```

在当前 outlined AICore lowering 中，生成 C++ 虽然保留了
`if (seq_len <= 0) { ... }` 的清零分支，但没有让 `continue` 跳过
该 request 后续展开的 scatter/write body。因此 padding row 仍会读取
全 0 page table，并写入 physical block 0。这不是 metadata/binder 错位，
也不是 capture 时更换了地址，而是 DSL 控制流 lowering 不能承载
这种 runtime early-exit 语义。

正式修复不再使用 runtime `continue`，而将 active predicate 放到每个
持久 state/cache write 之前：

- main/inner scatter 对 inactive row 将 `state_block_i32` 改为显式
  `pl.cast(-1, pl.INT32)`，随后复用 `state_block_i32 >= 0` 的真实写门禁；
- main compressed-cache write 以及 inner indexer-K/indexer-scale write，把所有
  持久写完整包在 `kv_seq_lens[c_idx] > 0` 内；
- inactive row 的 scratch `pooled_kv` 仍显式清 0，但 scratch 初始化不再
  被误用为控制持久写的 early exit；
- Host 源码契约测试禁止这两个 compressor 重新引入代码行级
  `continue`，并要求显式 positive active guard。

第一次修正在 Host codegen 时又暴露了类型问题：裸 `-1` 被推导为
`INDEX`，而 page table read 是 `INT32`。最终以显式 `pl.INT32` cast
消除歧义，没有为了通过 codegen 而改变 metadata ABI。

修正后使用 fresh device0 串行完成两个 runtime：

```bash
python -m tests.pypto_dsv4_decode_csa.a3_partial_bucket_smoke \
  --runtime tensormap_and_ringbuffer --device 0

python -m tests.pypto_dsv4_decode_csa.a3_partial_bucket_smoke \
  --runtime host_build_graph --device 0
```

两者均为 `B=4`，capture `A=2`，同一张 graph 依次 replay
`A=1/3/2/4/1`，并且：

- 每轮 `output_max_abs=0.0`，`padded_output_nonzero=0`；
- metadata address 全程稳定；
- `seeded` / `installed` / eager / capture / 五次 replay 后，六类
  block 0 mismatch 均为 0；
- main/inner state、indexer K/scale 的 page-padding mismatch 均为 0；
- TRB 完整日志为 `/tmp/csa_partial_bucket_trb_v3.log`；
- HBG 完整日志为 `/tmp/csa_partial_bucket_hbg.log`；
- 两个 fresh process 都显式记录 `exit_code=0`。

这个结论的边界是“B4 固定 graph 内 A 可变，padding 不写 block 0”。
它还不能替代 B4/B8/B12/B16 的多 bucket/multigraph 矩阵，也不证明
inactive row 可以去掉 full-B scratch 计算；后者是性能优化问题，不应与
本轮的持久 state 安全门禁混为一件事。

## 43. 四层 Capsule 的 Host/生命周期骨架

### 43.1 当前已实现范围

新增 `a3_capsule_smoke.py` 与对应 Host contract test。runner 支持：

- 1/2/4 层；
- `PPPP`、`PP_same`、`PP_distinct`；
- 每层独立 layer name、weight owner、metadata、六类 cache、输入/输出 buffer；
- 全部 callable 共用同一个 PyPTO backend 和同一个 prepared L1 context；
- `PP_same` 复用同一个 static spec、prepared operator 和
  `DecodeCSADeviceOwner` warmup/capture-ready facade；
- `PP_distinct` 用不同静态 spec/operator 验证 func/callable 隔离；
- 一张 ACLGraph 内包含多个 `dsa_forward` 节点及 torch `fill/add/mul` bridge；
- capture buffer 与 eager buffer 地址独立，多次 replay 原位更新固定输入。

resource audit 不只检查 Host label，还检查真实对象和 tensor address：layer、
weight、metadata、prepared cache、eager/capture IO 跨层及跨类别不能 alias；
backend/context 必须全层共享；prepared operator identity 必须与
`callable_group` 的相等关系双向一致。

### 43.2 两个审计后修正的问题

第一次实现有两个不会被 zero-output 表面结果自动揭示的问题：

1. capture context 退出后没有先同步 capture stream，而下一步默认流 eager
   会访问同一组 cache。ACLGraph capture 可能执行一次 captured nodes，因此
   这里存在跨流共享 state 竞态。现在在 capture 外、任何后继 eager 之前
   显式 `capture_stream.synchronize()`；该同步属于测试 caller，不在 L1 op
   或 graph 内。
2. `PP_same` 的两层最初各自创建一份 DeviceOwner facade。虽然二者指向同一
   backend/record.operator，但 `_warmup_enqueued` 独立，导致同一个 operator
   warmup 两次。现在按 `callable_group` 共享 facade，调度序列明确为第一层
   `warmup`、第二层普通 `launch`，同一 specialization 只 warmup 一次。

Host 反例测试覆盖 split backend、split context、伪共享 operator、metadata/
cache 跨类别 alias 和错误 warmup facade。定向 Capsule suite 为
`60 passed`，Ruff check 通过，未使用 NPU。

### 43.3 必须保留的证据边界

当前 A3 Capsule runner 仍是全 PyPTO、zero-weight、residual/affine bridge 的
结构与生命周期 smoke。它尚未实现计划中的真实 HC-pre、RMSNorm、HC-post、
low-rank FFN，也没有建立 nonzero native/PyPTO 双 state world；eager 与
graph 仍共享每层 cache，zero weight 使结果不受 state 影响。因此即使后续
这条 A3 smoke 通过，也只能证明多 node、callable/address/stream 生命周期，
不能宣称完成四层非零 stateful 精度验收。

## 44. 连续普通 launch 的两层假失败与最终两步闭环

### 44.1 中间 sync A/B 排除 L1 跨调用重叠

第 42 节建立正式 `writeback TaskId -> qk_pv` 依赖并移除
`kv_touch` 后，一个严格普通 launch 已经通过。但早期的两步诊断仍报
SWA RoPE 区最大差异 `5.484375`。为判断是否为 L1 共享 workspace、
hidden AICore stream 或 taskQueue 导致的跨调用重叠，分别执行了：

- 两个普通 launch 连续 enqueue，最后一次外部 synchronize；
- 两个普通 launch 之间额外执行 caller 侧 synchronize。

两者得到完全相同的失败，而且输出、NOPE 区、compressed cache、
main/inner state 与 indexer K/scale 仍 close。因此不能把该现象归因为
“缺少 stream sync”；同一 caller stream 的 FIFO 与 PyPTO 单算子内部 fork/join
并未在这个实验中失效。

### 44.2 真正原因是 native harness 预构造污染 RoPE runtime buffer

`AscendDSAMetadataBuilder` 的 decode 路径调用
`get_cos_and_sin_dsa(..., use_cache=True)`。后者不为每个 metadata 创建独立
cos/sin storage，而是把当前 position 的数据 copy 到
`RopeGlobalState.runtime_buffer` 中按 group 共享的稳定 buffer，并返回
`buf_cos[:num_tokens]` / `buf_sin[:num_tokens]` view。这是为 ACLGraph 保持地址稳定的
production 设计，正常 scheduler 会“建当前步 metadata -> 立即 forward”。

原测试为了一次生成所有 callable，先构造了整个
`native_metadatas` 列表：

```text
build position 0 metadata  -> shared rope buffer = position 0..7
build position 8 metadata  -> 覆写同一 shared rope buffer = position 8..15
run position 0 native call -> 仍持有该 buffer view，实际读到 position 8..15
```

这解释了为什么一步通过，而只要在执行前“多构造一个未来步骤”，
第一步就立即在 SWA RoPE 区不一致。这不是 position 8 slot mapping 写回
position 0，也不是 `npu_sparse_attn_sharedkv` 原位改写历史 cache。

正式修正为：native 每个 decode step 都在调用前立即运行五个真实
metadata builder，强持 bundle/builder/buffer owner，然后立即执行 custom-op。
PyPTO 一侧仍可预构造多步 metadata，因为它的 44-slot ABI 不消费该
per-token runtime-buffer view，而是消费 process-pinned full RoPE table 和每步独立的
device `start_positions`。

### 44.3 fresh device0 正式两步结果

修正 harness 后，在 device0 、TRB runtime、B4/S8、两个连续普通 launch、
中间无 sync 的 fresh 进程中完成严格复验，退出码为 0：

- 两步输出总体 `max_abs_error=0.036376953125`，close；
- SWA 两步活跃元素 `32764`，`max_abs_error=0`，bit-exact；
- compressed KV `max_abs_error=0.0078125`，close；
- main state `max_abs_error=2.9802322387695312e-08`，close；
- inner state `max_abs_error=2.2351741790771484e-08`，close；
- indexer K 与 FP16 scale 均 bit-exact；
- 全部六类 state 与输出的 `close=true`。

这条证据同时验证了普通 launch 而非只是 warmup、每步 tensor/metadata
地址 patch、两步持久 state 以及无内部/中间 stream sync。它不代替
ACLGraph 复验。

### 44.4 初步性能信号的边界

在同一 B4/S8 synthetic W8 真实算子上，一次稳定采样得到：

```text
native serial: 1.9573308527469635 ms
PyPTO TRB:     1.1067353421822190 ms
ratio:         1.768562707040305x
```

这个数值只是当前 synthetic TP1、native production overlap 关闭、单进程短采样下的
方向性信号；尚不具备 ABBA、p50/p90/p99、profiler device span 和 native overlap
baseline，不能当作最终性能结论。

## 45. Stateful trace 的 full-B production metadata materializer

### 45.1 Host 数据模型与生产 ABI 映射

在 `tests/pypto_dsv4_decode_csa/metadata.py` 建立了不依赖 NPU runtime 的第二层
trace materializer，将不可变 `DecodeStep` 转成深不可变的
`DecodeStepMetadataPayload`：

- 显式保留 `A=num_reqs_actual` 和静态 `B=bucket_size`；
- 生成 full-B、固定 S=8 的 `query_start_loc`；
- 生成 full-B `start_positions` 和 `kv_seq_lens`，padding 行两者均为 0；
- 生成 `cmp_block_table`、`compress_state_block_table`、
  `inner_compress_state_block_table`、`idx_block_table` 和 `swa_block_table`；
- active request 紧密排在 `[0,A)`，padding 行在 `[A,B)` 且五张表全 0；
- 五个 cache namespace 始终分开，不因 block ID 整数值相同而混淆所有权。

默认物理映射只给 main/inner compressor state 的 trace block ID 加 1，
保留 CANN compressor state physical block 0 sentinel。compressed/indexer/SWA 仍允许
physical block 0；padding 行安全性依赖正式 kernel 的
`kv_seq_lens[row] > 0` active guard。为了完成更强 canary 测试，materializer
也允许测试侧将五个 namespace 全部偏移 `+1`。

### 45.2 graph-stable device metadata owner

`StagedDecodeStepMetadataPayload.from_host()` 在 capture/replay 外将 Host payload
转成 canonical contiguous INT32 tensor。`DecodeStepMetadataBufferOwner` 为一个
static B/spec 创建一套固定地址 buffer，后续 `apply_staged_payload()` 只做：

1. 在第一个 `copy_` 前完成 spec/device/shape/query-row 全部校验；
2. 原位更新五张 block table、`start_positions` 和 `kv_seq_lens`；
3. 不更换任何被捕获 tensor 地址；
4. 不同步、不分配 device tensor；
5. 强持所有仍可能被异步 copy 消费的 staging source。

caller 只能在外部 stream/device synchronize 证明 copy 完成后，调用
`release_staging_owners_after_sync()` 释放这些 source。一个 owner 只服务一个
static B/spec；多 bucket 必须使用多个 owner，不在 replay 期间 resize/rebind。

### 45.3 Host 验证结果

新增 15 条 metadata 专项测试，覆盖：

- A/B 分离与 full-B padding sentinel；
- 五族 table 映射与 state block 0 sentinel；
- 与正式 `bind_decode_csa_native_metadata` 的 duck-typed ABI 兼容；
- admit/advance/retire/compact/block reuse；
- 固定地址 owner 的原位更新和失败前不改写；
- 4/32/128-step seeded churn 物化；
- width/capacity/bucket/offset 错误的 fail-fast。

seeded churn 的 target actual 序列从满桶 `4/8/12/16` 改为
`3/7/11/15/9/6/2/5`，仍在前四步覆盖 B4/B8/B12/B16，但现在每步都真正
满足 `A < B`。

验证证据：

- metadata + trace + partial + native-metadata 定向测试：`71 passed`；
- `tests/pypto_dsv4_decode_csa` 当时全量 Host 回归：`335 passed`；
- 仅有 14 个环境既有 TorchScript deprecated warning；
- 新增/修改文件 Ruff check 和 format-check 通过；
- 本节不使用 NPU。

### 45.4 必须保留的未完成边界

1. staging/from_host 可以分配并 H2D，只能在 capture/replay 外预先完成；
   `apply_staged_payload()` 的 `copy_` 也必须由 caller stream 正确排序在 replay 前。
2. materializer 只生成 metadata，不负责新 request 复用已回收 block 前的
   cache 清零/初始化。后续 A3 runner 必须按 native/vLLM 语义显式完成该动作，
   否则所谓“无残留泄漏”测试没有意义。
3. `derive_trace_metadata_requirements()` 可以推导 128-step trace 需要的更大
   table width 和 physical capacity，但默认 production 容量未必能装下它；A3 内存与
   compile 可行性尚未验证。
4. 本节只形成 Host/binder 证据，尚未形成连续 trace 的 A3 执行、
   native/PyPTO 双 state world 与 ACLGraph replay 证据。

## 46. Production dispatch Phase 2 Host 契约再审计

### 46.1 已确认的主链路

本轮只做 Host 侧静态审计与 fake-backend UT，没有使用 NPU。逐项对照
Phase 2 后确认 production 路径已具备：

- `dsa_forward` 仍保持原有四参数 custom-op schema 和
  `mutates_args=["output"]`；
- layer 从 `ForwardContext.no_compile_layers[layer_name]` 取得，metadata
  仍用 native 相同的 prefix filter + key sort，A3 六 cache tuple 仍由
  `_build_kv_cache()` 构造；
- v1 `ForwardContext.capturing` 与 v2 `_EXTRA_CTX.capturing` 在 vLLM
  wrapper 边界合并，PyPTO backend 不查询 graph/capture 状态；
- 首次 ordinary eager 只做 warmup，caller 外部同步并显式标记
  quiesced 后才能 capture；未 warmup 或未 quiesce 都在 bind/enqueue
  前失败；
- 安装 owner 后任何 adapter/bind/enqueue 异常直接向上传播，不会在
  六类 mutable state 可能已开始改写后 fallback native；
- eager binding 依靠 taskQueue tensor lease，不在 Python backend 中累积；
  capture binding 强持 layer/weight/cache/metadata/I/O 参数快照到 context
  close；
- 普通 hidden/output/metadata 地址变化每次重新 bind，不改变 static
  program identity；六 cache storage 变化需要显式 reinstall，不偷用旧
  physical alias。

### 46.2 修正的两个明确缺口

1. **`Out` 的 storage alias 原先未被拒绝。**
   编译 ABI 把 `attn_out` 定义为纯 `Out`，但 production owner 原先只检查
   shape/dtype/contiguous，同 storage 的 `hidden_states` view 仍可进入
   backend。现在 bind 前对全部 tensor argument（hidden、weight、metadata
   与六类 cache alias）做 storage identity 检查；只要 output 共享 storage
   就 fail-fast。这是 Host metadata 检查，不读 device 值、不同步。
2. **registry 原先会丢失 prepare 失败的 cleanup owner。**
   backend 本身能接管 PyPTO `cleanup_context`，但
   `DecodeCSADeviceOwnerRegistry.get_or_create()` 只在 `prepare()` 成功后才保存
   owner。若 context 部分初始化后失败，返回栈退出后将只剩不可重建的
   process/device claim，上层无法通过完整 backend 重试 close。现在 registry
   将该 owner 以 `cleanup-only` 状态强持，原始异常也携带
   `decode_csa_cleanup_owner`；后续 install/launch 拒绝复用，但 caller 仍可通过
   owner `close()` 使用 backend 已有的失败保留/重试语义。

另外，未 warmup capture 的门禁被前移到 hot metadata binding 之前。
这不改变正常 capture 路径，但保证明知 specialization 未就绪时不先构造
调用快照。

### 46.3 Host 回归

新增 3 条 dispatch UT，分别覆盖 cleanup-only owner 强持与 close、
capture gate 早于 metadata binding，以及 output/hidden same-storage alias 在
backend bind 前失败。定向执行结果：

- dispatch/backend/native-metadata/cache-adapter/contract：`120 passed`；
- `tests/pypto_dsv4_decode_csa` 全量 Host 回归：`347 passed`；
- 仅有 14 个环境既有 TorchScript deprecated warning；
- 修改文件 Ruff check 与 format-check 全部通过；
- 本节全程未调用 NPU。

### 46.4 本轮未改变的有意边界

1. public custom-op schema 只能声明显式参数 `output` 可变；六类 cache
   由 `ForwardContext` 隐式取得，无法出现在 `mutates_args`。PyPTO artifact
   内部已将它们全部声明为 `InOut`，但若未来让更高层 PyTorch
   functionalization/重排跨过这个现有 PrivateUse1 边界，仍需单独处理
   hidden side effect 语义；本轮不改 public schema。
2. capture binding 仍按 context 粒度保留，不能在单个 ACLGraph destroy 时精确
   回收；这是当前不取 graph handle 的有意安全策略。
3. hidden/output/metadata 地址可以 per-call patch，但六 cache 的 storage 地址
   变化仍必须 reinstall layer owner；这是 page-strided physical alias 的当前契约，
   不应误读为所有参数都能在已 capture graph 中任意换址址。

## 47. Phase 7 Host-only 交付件与最终证据覆盖规则

### 47.1 强制 supersession 规则

> **重要：最终证据会覆盖历史 45-slot 证据。**
>
> 当前 public L1 ABI 只有 **44 slots**：40 个 tensor、4 个 runtime scalar，
> 唯一纯 Out 为 index 39。第 33 节的 B4/B8/B12/B16、多 graph 和
> zero-program 结果来自旧 45-slot ABI，只能作为历史结构可行性证据，不能继续
> 充当当前支持矩阵。

证据覆盖链如下：

1. 第 33 节使用 45-slot zero program，当时能够证明四 bucket 和 graph owner 的
   早期路径可执行，但不能证明当前 kernel 语义。
2. 第 40 节删除 `swa_slot_mapping`，ABI 从 45 slots 收缩为 44 slots。
   该变更后，所有旧 artifact 和 captured graph 都不再是当前源码的产物。
3. 第 42 节又把持久 SWA writeback 改成真实 `TaskId -> qk_pv` 依赖，并删除
   为调度而引入的 `kv_touch`。即使是更早的 44-slot 产物，只要生成于该语义
   变更之前，也不能代表最终源码。
4. 最新的最终 44-slot + writeback 依赖已经在 fresh-process TRB/HBG 中完成
   partial-bucket 严格 canary，以及 B4/B8/B12/B16 四 graph 同存、交替 replay、
   局部 destroy 后 survivor replay。它们覆盖旧 45-slot 对 active-row guard、
   写隔离和多 graph 生命周期的结论，但使用 zero projection weight，不能自动
   升级为 B8/B12/B16 非零算法精度证据。
5. 可正式引用的 A3 证据必须对应最终源码、逻辑 `device0`、TRB/HBG 各自独立
   的 fresh Python 进程、明确的 process exit 0，以及与测试目标匹配的结果摘要。
6. Host UT、codegen 成功、runner 存在或 `/tmp` 临时日志不会自动升级为 A3
   支持证据。后续非零矩阵若在本节后继续产生，应由后续过程记录和持久结果覆盖
   本节状态，不能反向改写成“本节已经验证”。

### 47.2 Phase 0～7 当前 status 总表

`Host 已验证` 表示对应契约有纯 Host UT；`A3 已验证` 只用于本记录明确保存
板上结果的窄范围。`Host 骨架` 不等于能在 A3 上正确运行。

| Phase | 当前状态 | 已有的有效证据 | 必须明确保留的未完成项 |
| --- | --- | --- | --- |
| Phase 0：环境与最小 L1 基线 | Host 已验证；A3 部分已验证 | 五仓 commit、Python 3.11/Torch 2.12、PTOAS/GCC runtime 可由只读脚本复核；最小 L1 与 production B4 路径已有 eager/ACLGraph 板上记录 | 尚无正式 profiler 证明整条 production 路径内部无 sync；只读检查不证明 device 健康或空闲 |
| Phase 1：`decode_csa_core` 与静态 entry | Host 已验证；最终 B4 非零 A3 已验证；四 bucket zero-weight A3 已验证 | 正式实现位于 `_pypto_dsv4_csa`；六类 state 为 `InOut`；page-strided cache、FP16 scale、最终 44-slot B4 非零连续步骤已对齐 native；最终源码的 B4/B8/B12/B16 已执行 zero-weight 四 graph 路径 | zero-weight 四 bucket 不能替代 B8/B12/B16 非零 output + 六 state 数值矩阵；后续非零结果以后续记录为准 |
| Phase 2：vLLM ABI adapter 与单 custom-op | Host 已验证；最终 B4 A3 已验证 | `ForwardContext` metadata/capture signal、warmup-before-capture、失败不 fallback、output alias、cleanup-only owner、地址 patch 和 cache reinstall 契约均有 UT；最终 44-slot B4 TRB/HBG 有单 custom-op 非零 ACLGraph 记录 | capture snapshot 仍是 context 粒度；六 cache storage 变化不支持在已安装 owner 上透明 patch；单 B4 结果不能外推为所有 bucket 的非零精度 |
| Phase 3：四层 Capsule | Host 骨架 | 1/2/4-layer topology、layer-local owner/cache、backend route 和 zero-fixture runner 有 Host 契约与骨架 | 尚无 fresh-process A3 Capsule 黄金结果；真实 HC-pre/RMSNorm/HC-post、low-rank bridge、非零 `NNNN/PPPP/NPNP/PNPN`、逐层精度和 mixed backend 均未完成 |
| Phase 4：Stateful trace 与 block 生命周期 | Host 已验证；partial bucket zero-weight A3 已验证 | 4/32/128-step deterministic trace、admit/advance/retire/compact/reuse 的 Host materializer 与 graph-stable metadata owner 有 UT；TRB/HBG partial bucket 已通过 block-0/page-padding canary、padded output 清零和固定 metadata 地址 | partial bucket 非零 native/PyPTO 数值、retire/reuse 和连续 trace 尚无 A3 闭环；新 request 复用已回收 block 前的 cache 清零/初始化语义仍待落实 |
| Phase 5：多 bucket ACLGraph | 最终四 bucket zero-weight A3 已验证；完整矩阵未完成 | 最终 44-slot B4/B8/B12/B16 已分别在 fresh-process TRB/HBG 中完成四 graph 同存、交替 replay、局部 destroy 和 survivor replay | 该证据使用 zero weight；当前源码的非零多 bucket/multi-graph 数值、destroy 后 recapture、四层 graph 与跨 bucket stateful trace 尚未在本节闭环 |
| Phase 6：性能闭环 | 结果 schema/runner 已有；未验收 | 已有方向性 TRB 短样本与 HBG ACLGraph 稳定样本；HBG 功能通过但约 305～320 ms，native graph 约 0.67 ms | 无 native production-overlap baseline、同卡 ABBA、p50/p90/p99、Host enqueue/taskQueue dequeue/device span 拆分、四层/trace 性能和正式 profiler；不能声称生产性能可用 |
| Phase 7：测试收敛与交付 | Host-only 交付件已补；整体未完成 | 本节新增 README、只读环境检查脚本与 Host UT；既有 fixture/harness/result schema 集中在 `tests/pypto_dsv4_decode_csa/` | A3 黄金路径尚未收进正式 one-card ST；长稳/性能尚未收进 nightly/manual suite；缺少与最终源码完整对应的持久 result bundle/profiler；许可证/NOTICE/provenance 未闭环 |

### 47.3 新增 Host-only 交付件

1. `tests/pypto_dsv4_decode_csa/README.md`：固定真实环境命令、`device0`、
   TRB/HBG fresh-process 规则、当前 44-slot runner 命令、证据覆盖规则以及
   能与不能得出的结论。
2. `tests/pypto_dsv4_decode_csa/check_environment.py`：只使用 Python 标准库
   做只读检查。它通过 `find_spec()` 定位模块但不导入
   Torch/torch_npu/PyPTO/simpler/vLLM；唯一子进程操作为
   `git -C <repo> rev-parse HEAD`。它不运行 `npu-smi`、不初始化 NPU、
   不编译、不写文件，也不修改系统。
3. `tests/pypto_dsv4_decode_csa/test_environment_check.py`：覆盖成功报告、
   device/Python/PTOAS/GCC 错误、过期 commit/源码路径、导入集合不变和 README
   证据边界。

实际工作区执行只读检查时，五仓 commit、Python 3.11 venv、
Torch/torch_npu 2.12、PyPTO/simpler distribution、源码模块来源、
`.../tools/ptoas/ptoas` 实际可执行文件与 GCC 15 runtime 全部通过；
JSON 明确记录 `npu_probe_performed=false`。PTOAS 检查刻意指向最后一级
可执行文件，而不是把具有目录搜索权限的 `tools/ptoas/` 目录误判为 executable。

本次 Host-only 验证结果：

- 环境检查定向 UT：`9 passed`；
- `tests/pypto_dsv4_decode_csa` 全量 Host 回归：`357 passed`；
- 存在 14 条环境既有 TorchScript deprecated warning；
- 本节全程不使用 NPU，不 commit，不 push。

### 47.4 交付前最大缺口

1. **发布阻断：** private kernel primitive 的 CANN Open Software License 与
   仓库 Apache-2.0 之间的 NOTICE/provenance 尚未由维护者确认。
2. **状态轨迹阻断：** 最终 44-slot 的 retire/reuse、4/32/128-step stateful
   trace 和 cache block 复用前初始化还没有 TRB/HBG A3 数值闭环。
3. **数值矩阵阻断：** zero-weight partial/multi-graph 结果不证明各 bucket 的
   非零 output 与六 state 精度；本节不预写仍在后续产生的非零结果，以后续记录
   与持久 result 为准。
4. **子系统阻断：** 四层 Capsule 仍是 Host/runner 骨架，真实
   HC/RMSNorm/bridge、mixed backend 和逐层非零精度没有形成 A3 证据。
5. **性能阻断：** HBG 当前功能可执行但比 native graph 慢数百倍；TRB 也没有满足
   同卡 ABBA、production overlap、分位数和 profiler 要求。
6. **测试交付阻断：** A3 short ST、nightly/manual 矩阵、最终源码对应的 durable
   JSON/CSV/profiler bundle 尚未收敛。`/tmp` 文件或历史命令记录不等价于这些
   正式交付件。

## 48. 最终 44-slot 四 bucket 与非零数值矩阵补齐

### 48.1 证据对应的最终源码语义

本节所有结果都重新生成于当前 production 源码，不再引用第 33 节的
45-slot 历史 artifact。对应契约为：

- 总计 44 个参数 slot：40 个 tensor 和 4 个 runtime page-stride scalar；
- 唯一纯 `Out` 为 slot 39，六类持久 cache/state 全部为 `InOut`；
- SWA writeback 使用真实 `writeback TaskId -> qk_pv` dependency；
- artifact 不再包含过渡性 `kv_touch` kernel；
- TRB 和 HBG 的正式结果分别来自 fresh Python process，不在同一进程切换
  runtime；
- 仅使用逻辑 `device0`。

当时 production Python tree 的内容 SHA256 为
`50115110dbbe6d8aaabacb769a41f931e8b904c91d866c50160b66453467f752`。
该 hash 是对 `_pypto_dsv4_csa/` 正式实现树计算，不把之后仍在迭代的
test runner 当成 production 源码变化。

### 48.2 B4/B8/B12/B16 多 graph 生命周期

分别在 TRB 和 HBG fresh process 内同时创建 B4/B8/B12/B16 四个
specialization 和四张 ACLGraph，按下列顺序串行 replay：

```text
B4 -> B16 -> B8 -> B12 -> B16 -> B4 -> B12 -> B8
```

两个 runtime 均得到：

- `compiled_bucket_count=4`；
- `retained_capture_bindings=4`；
- 全部 replay 的 `max_error=0.0`；
- 销毁 B8 graph 后，B16 survivor graph 仍可正确 replay；
- process exit code 为 0。

这条矩阵使用 zero projection weight，所以它正式证明的是：最终 ABI、
static callable 隔离、四个 capture binding 强持、交替 replay 和局部 destroy
生命周期。它不单独证明非零算法精度。

### 48.3 B8/B12/B16 非零 native/PyPTO eager 对比

为消除“只有 B4 非零、其他 bucket 只有 zero-weight”的证据空洞，
`a3_native_compare.py` 改为按命令行 `batch` 构造真实
`DecodeCSAProgramSpec`，随后在 device0 上分别以 fresh process 执行。

TRB 结果：

| Bucket | output max abs | compressed KV max abs | SWA | main state max abs | inner state max abs | indexer K/scale | 结果 |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| B8 | 0.036376953125 | 0.0078125 | bit-exact | 2.9802322387695312e-08 | 2.2351741790771484e-08 | bit-exact | PASS |
| B12 | 0.036376953125 | 0.0078125 | bit-exact | 2.9802322387695312e-08 | 2.9802322387695312e-08 | bit-exact | PASS |
| B16 | 0.036376953125 | 0.0078125 | bit-exact | 2.9802322387695312e-08 | 4.0978193283081055e-08 | bit-exact | PASS |

完整日志为：

- `/tmp/csa_native_b8_trb_final.log`；
- `/tmp/csa_native_b12_trb_final.log`；
- `/tmp/csa_native_b16_trb_final.log`。

HBG 也分别以 fresh process 执行 B8/B12/B16 非零对比，output 与六类
state 全部通过。日志为：

- `/tmp/csa_native_b8_hbg_final.log`；
- `/tmp/csa_native_b12_hbg_final.log`；
- `/tmp/csa_native_b16_hbg_final.log`。

这些 runner 输出的一次 timing 只是功能诊断量，不是 Phase 6 benchmark。
它们仍然给出一个不应隐藏的工程边界：TRB B8/B12/B16 诊断值约为
2.80/3.38/3.85 ms，而 HBG 约为 587.36/586.71/585.93 ms。因此 HBG
的功能矩阵已扩展到四 bucket，但其当前性能仍明确不可用。

### 48.4 持久结果包与证据边界

新增版本化结果包：

`tests/pypto_dsv4_decode_csa/results/20260903_device0_final44_functional_matrix.json`

它保存了五仓基线中与执行直接相关的 commit、production tree hash、
44-slot ABI、partial-bucket canary、四 bucket multi-graph 和非零数值矩阵，
并已通过 `python -m json.tool` 语法校验。结果包显式将一次 latency
标记为非 benchmark，也保留了下列未完成边界：

1. partial-bucket 与 multi-graph 仍是 zero-weight guard/lifecycle 证据；
2. B8/B12/B16 是非零 eager 对比，当时非零单 op ACLGraph 证据仍只有 B4；
3. 128-step TRB 原始 INT8 indexer K 在极少量量化边界上出现 1 LSB
   差异，未在本节伪装成 bit-exact PASS；
4. 本节不声称完整 Engine/模型、HC/MoE shell、TP/EP/HCCL、A5 或
   simulator 已验证。

## 49. 128-step TRB 与 INT8 K/FP16 scale 联合量化契约

### 49.1 为什么不能只放宽全局容差

最初的 128-step B4 TRB 运行中，output、compressed KV、SWA、main/inner
state 和 indexer scale 全部通过，但 raw INT8 indexer K 有极少量 1 LSB
差异。原始整数 comparator 因此正确地给出 `close=false`。

本轮没有提高全局 `atol/rtol`，也没有将 INT8 粗暴改成浮点近似比较。
新增独立 `indexer_quantized_comparison.py`，把 raw INT8 K 与对应的 FP16 scale
当作一个量化对象审计，并保留两层结论：

1. raw `states.indexer_k.close` 仍保留 bit-exact 语义，有任何整数差异就为
   `false`；
2. 整体数值验收额外检查独立的 `indexer_quantized.acceptable`，仅当下列
   条件全部满足时才允许量化边界 fallback：
   - native/PyPTO write-set 完全相同；
   - FP16 scale bit-exact；
   - raw 单元素最多差 1 bin；
   - raw mismatch fraction 不超过 `1e-4`；
   - 每个反量化误差不超过该物理行 1 个 scale quantum 加 `1e-7`
     绝对浮点 slack；
   - 所有 scale 与反量化值有限。

报告不只保存 PASS/FAIL，还保存 raw/scale/dequant 差异数、fraction、
max-bin、首个物理 row/column/block/value、token provenance 和映射 decode step。
Host 反例覆盖 write-set 不同、scale 不同、2-bin 差异、密度超标、
反量化误差超标与非有限值，不允许任一异常被 fallback 掩盖。

### 49.2 device0 上的 128-step 无中间同步复验

使用下列语义重新执行：

- runtime：`tensormap_and_ringbuffer`；
- bucket/seq：B4/S8；
- `start_position=0`；
- `correctness_steps=128`；
- `sync_between_correctness_steps=false`；
- 只在 128 步全部 enqueue 后由 caller 外部 quiesce；
- 逻辑 `device0`，fresh Python process；
- 完整日志：`/tmp/csa_native_b4_trb_128step_quantized_contract.log`。

进程 `exit_code=0`，整体 `close=true`。主要结果：

| 对象 | 结果 |
| --- | --- |
| output | close，`max_abs_error=0.04019451141357422` |
| compressed KV | close，`max_abs_error=0.0078125` |
| SWA KV | close，`max_abs_error=0.015625` |
| main state | close，`max_abs_error=4.470348358154297e-08` |
| inner state | close，`max_abs_error=2.9802322387695312e-08` |
| indexer scale | bit-exact，1024 个已写物理行 |
| raw indexer K | 明确非 bit-exact；4/131072 个元素差 1 bin |
| joint quantized object | `acceptable=true` |

联合量化报告的完整数值为：

- native/PyPTO written rows 均为 1024，`write_sets_equal=true`；
- `raw_mismatch_elements=4`；
- `raw_mismatch_fraction=3.0517578125e-05`；
- `raw_max_bin_error=1`；
- `scale_mismatch_elements=0`；
- `dequantized_nonzero_error_elements=4`；
- `dequantized_max_abs_error=0.019317626953125`；
- `dequantized_mean_abs_error=5.745096132159233e-07`；
- `dequantized_max_scale_quanta=1.0`；
- `dequantized_over_policy_elements=0`；
- `one_bin_fallback_eligible=true`。

首个 raw mismatch 在 physical row 2233、column 54，即 block 69 内 offset 25；
native/candidate 分别为 `-6/-5`，两者 scale 都为 `0.0180816650390625`。
该物理行映射到 request 1 的 absolute position 740～743，首个可能写入它的
decode step 为 92。因此这不是无来源的 cache 污染，而是一个可定位、
低密度的量化边界差异。

### 49.3 仍然保留的边界

1. 这条 128-step 是四个固定 request 连续 advance，尚不是
   admit/retire/compact/block-reuse churn；
2. 正式异步证据没有中间 sync，因此不会用同步掩盖 L1 调度问题；
   若之后运行逐步 snapshot 诊断，必须明确标注它只是定位证据；
3. 本次 `native_ms=2.5608` / `PyPTO_ms=2.4761` 是一次诊断 timing，
   不符合 ABBA/p50/p90/p99 要求，不用于声称稳定 speedup；
4. 联合契约不是通用 INT8 容差；只要 scale/write-set/bin 误差/密度/
   反量化误差中任一超过上述独立上限，整体仍必须失败。

## 50. 四层非零 Capsule 的逐层 oracle 与 TRB/HBG mixed-backend 闭环

### 50.1 为什么四层比较不能只看最终输出

首版四层 runner 把 `NNNN` 与候选链分别完整执行，再逐层直接比较。当
PyPTO 在上游 layer 产生已经处于单层容差内的 BF16 差异后，下一层 native
reference 与 candidate 实际收到的输入已经不同；此时下一层 cache 差异混合了
“当前层实现差异”和“上游输入差异”，不能再作为当前层严格 oracle。

最终采用三个互相独立的六-cache world：

1. `NNNN` world：形成端到端 native baseline；
2. local-oracle world：每一层都用该层**实际 candidate 输入**直接调用 production
   native `impl.forward`，只用于当前层局部输出与状态比较；
3. candidate world：按 `PPPP`、`NPNP` 或 `PNPN` 路由执行，并且只有该 world
   的输出继续传播到下一层。

因此局部门禁与链路累计误差是两套独立政策：局部仍使用
`atol=0.1, rtol=0.1`；四层最终输出默认使用 root-sum-square 绝对预算
`chain_atol=0.2`。六类 mutable state 和 INT8/scale 联合量化契约只使用
same-input local oracle，不因放宽最终链路容差而放宽。

### 50.2 上板前连续修正的 harness 问题

这条路径不是一次写完即通过。按出现顺序修正了以下问题，并全部保留为实现
过程的一部分：

1. 多层 synthetic attention 使用临时 prefix 后再改名，触发
   `static_forward_context` 重复注册；修为从构造开始就使用最终唯一 layer prefix。
2. RoPE/ModelSlim 配置只描述单个旧 prefix，四层构造时查不到真实 key；修为一次
   构造包含四个最终 prefix 的 combined description。
3. 最初误取模型前四层，其中并非全部是目标 ratio-4 CSA；修为从
   `FLASH.compress_ratios` 选择真实 ratio-4 layer，当前为模型层 2/4/6/8。
4. 初版逐层 comparator 让 local oracle 消费 NNNN 输入，而不是 candidate 实际
   输入，形成上述 upstream-drift 假失败；修为三-world same-input local oracle。
5. 直接调用 native 实现时漏装真实 `ForwardContext`；修为在
   `override_forward_context` 与 `torch.inference_mode()` 下调用。
6. native `impl.forward` 的 metadata 参数要求可变 list，而 harness 传入 tuple；
   修为保持 owner 持有 tuple，但调用边界显式转换为 list。

这些修正只发生在 vLLM-Ascend 测试 harness；没有修改 native DSA、PyPTO、
simpler 或 pypto-lib。

### 50.3 device0 TRB 结果

三种候选拓扑均在独立 fresh Python process、A3 device0 上退出 0：

| candidate topology | 最终 max abs | 最终 mean abs | close | 日志 |
| --- | ---: | ---: | --- | --- |
| `PPPP` | 0.10400390625 | 0.015396037840901045 | true | `/tmp/csa_native_capsule_pppp_trb_v7.log` |
| `NPNP` | 0.072998046875 | 0.012139094136585982 | true | `/tmp/csa_native_capsule_npnp_trb.log` |
| `PNPN` | 0.1123046875 | 0.018917381348728668 | true | `/tmp/csa_native_capsule_pnpn_trb.log` |

每条结果都逐层比较 DSA output、layer output、compressed KV、SWA KV、main
compressor state、inner compressor state、indexer K 与 indexer scale；并记录
四套 attention/weight owner 及 12 个互不别名的 mutable cache world。不能用
最终 `close=true` 代替这些局部证据。

### 50.4 device0 HBG 结果

TRB 进程全部退出后，又为每个拓扑分别启动新的 HBG process；没有在一个进程中
切换 runtime：

| candidate topology | 最终 max abs | 最终 mean abs | close | 日志 |
| --- | ---: | ---: | --- | --- |
| `PPPP` | 0.10400390625 | 0.015396037840901045 | true | `/tmp/csa_native_capsule_pppp_hbg.log` |
| `NPNP` | 0.072998046875 | 0.012139094136585982 | true | `/tmp/csa_native_capsule_npnp_hbg.log` |
| `PNPN` | 0.1123046875 | 0.018917381348728668 | true | `/tmp/csa_native_capsule_pnpn_hbg.log` |

三条 HBG 日志的进程退出码均为 0，顶层 `close=true`，逐层 local comparison
也全部通过。TRB/HBG 得到相同数值不是同进程缓存复用的结果。

### 50.5 本阶段仍不能声称的内容

当前非零 runner 的层壳仍是 `DSA output + residual + deterministic affine
bridge`，明确输出 `deterministic_bridge_only=true`。它已经证明多层权重/cache
隔离、同 callable 多地址 patch、mixed native/PyPTO 顺序与 HBG eager 路径，
但**尚未**证明真实 `npu_hc_pre_v2 -> RMSNorm -> dsa_forward -> npu_hc_post`
层壳。`PP_same/PP_distinct` 当前也只有 zero-weight ACLGraph/Host ownership
骨架，不能用本节的 `PPPP` 推导为 distinct callable 已完成。后续必须分别补齐，
不能把当前结果写成完整 Phase 3 已完成。

## 51. Stateful churn 首次上板暴露的非零 prefix 初始化错误

### 51.1 首次 4-step TRB 结果

新增 `a3_stateful_churn_compare.py` 后，首次使用旧 deterministic trace 在 fresh
device0/TRB 进程运行 4 step。该进程编译 B4/B8/B12/B16 四个 static entry 后，
在 step 0 失败，完整日志为：

`/tmp/csa_stateful_churn_trb_4.log`

step 0 是 B4/A3，三个新 request 的 `sequence_length_before` 分别为
120、24、3。native/PyPTO 的 write-set 相同，padded output、block0 canary 和
unexpected-row 检查均干净，但 compressed KV/SWA/state 与 indexer quantized
value 明显不同；例如 raw indexer K 有 760/768 个元素不同，最大 115 bins。
这不是可接受的量化边界，也没有被 fallback 放过，runner 正确退出 1。

### 51.2 根因是测试状态语义非法，不是把容差调大

trace 把新 request 直接 admit 到非零历史长度，却只把它新获得的 cache/state
block 清零。真实 `sequence_length_before=120` 表示 prefill/此前 decode 已经产生
完整历史 cache 和 compressor state；“非零历史长度 + 空 cache”不是合法的
vLLM request 状态。CSA 又是历史状态相关算法，因此不能把该输入下 native/PyPTO
差异解释为 operator 不一致，更不能提高容差让测试通过。

正式 churn runner 将只把新 request 从 position 0 admit；之后由连续 decode
自然形成历史 state。retire 后复用的 block 也只在新 position-0 request 获得时
按 empty-prefix 语义清空。2K/8K 边界必须另行使用真实 prefill/合法预装 state，
不能再用“伪非零 prefix + 清零 block”模拟。

### 51.3 同时发现的两个门禁缺口

只读复审还发现：

1. 旧 trace 前四步只是 A3/A7/A11/A15 单调增员，没有 retire/reuse；把其称为
   4-step churn 不准确。正式 4-step trace 必须包含非尾部 retire、survivor row
   compact、replacement admission 和真实 block reuse，并由 UT 显式断言。
2. 首版 `indexer_quantized.acceptable` 相对全局 initial 统计所有累计写行；后期某个
   step 的局部高密度错误可能被历史大量正确行稀释。正式门禁必须增加 step-local
   delta 行量化比较；累计/final 报告可以保留，但不能替代逐步门禁。

上述修正完成并重新上板前，Phase 4 状态仍是“runner 骨架与首个有效失败证据”，
不是 churn correctness PASS。

### 51.4 第二次 4-step TRB：生命周期全部通过，暴露离散小样本门槛

将 admission prefix 固定为 0，并把前四步改成真正包含非尾部 retire、survivor
compact、replacement admission 与五类 cache block reuse 后，在 fresh device0/TRB
进程再次运行，完整日志为：

`/tmp/csa_stateful_churn_trb_4_v2.log`

这次 B4/B8/B12/B16 四个 bucket 均完成编译并执行到 step 3。四步中的 output、
除量化对以外的 mutable state、逐步 write-set、复用事件、block0 canary、未授权行
均通过。唯一失败是 step 3 当前写入的 4096 个 indexer K 元素中有 1 个 raw INT8
差异；native/PyPTO 相差一档，scale bit-exact，反量化误差恰为一个 scale quantum。
这说明 lifecycle 与地址复用已经走通，失败来自门禁对离散样本数的定义，而不是用
历史正确行稀释当前错误。

原规则直接检查 `mismatch_fraction <= 1e-4`，对 4096 元素的 step-local 样本等价于
“必须零差异”；它与已经明确接受的稀有、一档、scale 完全相同的量化边界契约不
一致。规则因此改为：允许数量是
`max(1, floor(compared_k_elements * 1e-4))`。这个 `1` 只是离散计数下限，不是
额外放宽：默认策略下两个差异仍失败，任意两档差异、scale 不一致、超过一个 scale
quantum、write-set 不同或非有限值也仍失败。逐步门禁继续只看本 step 的显式写行；
累计报告只作为诊断，不能替代逐步结论。

针对该边界新增 Host 反例与正例：单行 128 个 K 元素中 1 个合规边界差异通过，
2 个差异失败；1024 行历史中末行 2 个差异仍可证明累计统计会通过、step-local
统计会失败。相关 28 项定向测试、ruff check、ruff format check 均已通过。
修改规则后的第三次 device0 重跑及 32/128-step 证据将在后续小节追加；在它们退出
0 之前仍不把 Phase 4 标为完成。

### 51.5 第三次 4-step TRB 通过

应用离散小样本契约后，在另一个 fresh device0/TRB process 使用相同 seed 重跑，
进程退出码为 0，完整日志为：

`/tmp/csa_stateful_churn_trb_4_v3.log`

结果不是只看顶层布尔值：

| step | bucket/actual | 本步 reuse event | output max abs | step-local raw mismatch / allowed | step close |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | B4/A3 | 0 | 0.036376953125 | 0 / 1 | true |
| 1 | B8/A7 | 11 | 0.03812670707702637 | 0 / 1 | true |
| 2 | B12/A11 | 19 | 0.03684043884277344 | 0 / 1 | true |
| 3 | B16/A15 | 19 | 0.03744983673095703 | 1 / 1 | true |

四步合计 49 个 reuse event。每一步六类 state 的 native/PyPTO changed-row set
相同且是 legal-row 子集，block0 与四类 page-padding canary 均未改变；PyPTO 的
padded output 始终为 0。step 3 的唯一差异仍是同一物理位置的一档 INT8 差异，
scale bit-exact、反量化误差一个 scale quantum，因而不是通过扩大普通浮点
`atol/rtol` 获得的假通过。native padded output 非零只作为 production native
实现行为诊断，不是 PyPTO 输出契约。

本条证明了 4-step 的真实 admit/retire/compact/reuse 闭环；32/128-step 与 HBG
仍需独立 fresh-process 证据，所以 Phase 4 尚未整体完成。

### 51.6 32-step TRB 长化验证通过

随后使用 fresh device0/TRB process 运行 32-step，进程退出码为 0，完整日志为：

`/tmp/csa_stateful_churn_trb_32.log`

该 trace 共发生 1654 个五族 block reuse event，bucket 命中分布为 B4 8 次、
B8 12 次、B12 8 次、B16 4 次；32/32 个 step 的 `close=true`。全链最大 output
absolute error 为 `0.042495667934417725`。只有 step 3 出现 1 个、允许 1 个的一档
INT8 量化边界差异，后续 28 步没有继续累积或扩散；其余 31 步 step-local raw
indexer K 全等。每一步 PyPTO padded output 均为 0，最终 native/PyPTO 的四类
page-padding canary 计数也全部为 0。

这条结果同时反证了离散下限没有把“每步都可漂一个”的系统性偏差隐藏起来：若
存在第二个同 step 差异，step-local 门禁会立即失败；本次唯一边界差异没有传播到
后续 churn。128-step TRB 与两种 HBG 长度仍待独立进程验证。

### 51.7 128-step TRB 首次运行在 step 0 失败

首次 128-step fresh device0/TRB 运行没有通过，日志为：

`/tmp/csa_stateful_churn_trb_128.log`

该命令最初使用 `python ... 2>&1 | tee ...`，没有启用 shell `pipefail`，因此外层
shell 错误返回了 `0`（`tee` 的退出码），但日志内有 Python traceback、
`AssertionError` 和 CANN application exception。此后所有上板管道必须先执行
`set -o pipefail`；是否通过以 runner JSON/traceback 和真实 pipeline exit code
共同判断，绝不再把 `tee` 的 0 当作算子通过。

失败发生在 candidate 的 step 0，而不是跑到后期才漂移：output 仍满足普通浮点
门禁，但 compressed KV、SWA KV 与 indexer K/scale 明显不一致；indexer 当前写入
768 个元素中有 762 个 raw mismatch，最大 122 bins，6 个 scale 全不相同，联合
量化契约正确拒绝。main/inner compressor state 仍分别只差约 `2.61e-8` 和
`2.24e-8`，六类 changed-row set 仍完全相同且没有越界行。这不是 1-bin 离散边界，
也不能靠容差处理。

Host 对比确认 step 0 的 request、start position、KV length、首个有效 block ID 与
4/32-step 一致；差别来自为完整 trace 推导的静态容量：

| trace | compressed/swa/main/inner/indexer blocks | compressed/swa/main/inner/indexer table width |
| --- | --- | --- |
| 4 | 16 / 16 / 125 / 125 / 16 | 1 / 1 / 16 / 16 / 1 |
| 32 | 17 / 22 / 221 / 221 / 17 | 2 / 8 / 128 / 128 / 2 |
| 128 | 23 / 46 / 605 / 605 / 23 | 8 / 32 / 512 / 512 / 8 |

因此当前首要怀疑是大静态 table/capacity specialization 暴露了 kernel、ABI、
workspace 或 runtime task graph 边界，而不是 churn 生命周期本身。定位将比较
32/128 codegen、缩小执行步数但保留 128 容量，并逐族隔离 table width/capacity；
在根因修复并以 `pipefail` 重跑退出 0 前，128-step 明确记为 FAIL，Phase 4 不完成。

### 51.8 静态 128-step specialization 本身不足以触发失败

为避免把“128-step trace”错误等价为一个变量，runner 已先拆出
`spec_steps`（决定静态 capacity/table width 与编译 artifact）和
`execute_steps`（实际执行并比较的 candidate prefix）。在 fresh device0/TRB
进程中运行 `spec_steps=128, execute_steps=1`：

`/tmp/csa_stateful_churn_trb_spec128_exec1.log`

该命令使用 `set -o pipefail`，Python/pipeline 真实退出码为 0，runtime close
成功。它使用与完整 128-step 完全相同的静态容量和 table width，只执行 trace 的
第一个 step；output、六类 state、changed-row set、联合 INT8 K/scale 以及 padding
canary 均通过。因此 128 specialization 的大静态 extent **不是充分条件**，不能再
把根因简单归结为大 table 的 codegen。

随后在 fresh device0/TRB 进程中按完整 `--steps 128` 重跑：

`/tmp/csa_stateful_churn_trb_128_v2.log`

本次同样启用 `pipefail`，真实退出码为 1，并在 candidate step 0 稳定复现原失败：
compressed KV 最大差约 `3.789`、SWA KV 最大差约 `3.328`，indexer K 的
768 个当前写入元素中 762 个不等且最大差 122 bins，6 个 scale 全不等；main/
inner 仍只差约 `2.61e-8`/`2.24e-8`，写集合仍一致。与单步通过相比，完整 runner
在 candidate step 0 之前还存在两个耦合差异：它保留了全部 128 份 device metadata
payload owner，并先执行了 128 次 native DSA 调用形成设备/runtime 前置状态。

下一步不再继续盲猜 kernel extent，而是正交拆分：

1. 只增加 retained metadata owner 数量，native 仍只执行一步；
2. 只增加 native precondition/burn 步数，candidate 和 retained owner 保持最小；
3. 两者都增加，确认最小可复现组合；
4. 增加“native 前预先 pack PyPTO weights”诊断，区分 native 调用后的 source/pack
   状态与设备全局状态；
5. 若 native burn 单独足以触发，再重点验证 ATB/AIC 全局 SoC state。当前 PyPTO
   child wrapper 会调用 `set_atomic_none()`，但这不等价于完整
   `AscendC::InitSocState()`，所以只能作为待证假设，不能在证据前归因。

在上述正交矩阵闭环前，128-step 仍保持 FAIL；4/32-step 与 ACLGraph 已通过的结论
不受影响。

### 51.9 正交上板确认触发条件是长 native 前置执行

四轴 runner 完成 Host 合同后，在三个互相独立的 fresh device0/TRB process 上执行
了以下诊断；所有命令都从 shell 起点启用 `set -o pipefail`：

| `spec/execute/native/retained` | prepack | 结果 | 日志 |
| --- | --- | --- | --- |
| `128/1/1/128` | 否 | PASS，`close=true` | `/tmp/csa_stateful_churn_trb_diag_native1_retained128.log` |
| `128/1/128/1` | 否 | FAIL，candidate step 0 | `/tmp/csa_stateful_churn_trb_diag_native128_retained1.log` |
| `128/1/128/1` | 是 | FAIL，candidate step 0 | `/tmp/csa_stateful_churn_trb_diag_native128_retained1_prepack.log` |

第一项保留全部 128 份 staged device metadata owner，但 native 只执行一步；它的
compressed/SWA/main/inner 全部 close，indexer K 的 768 个比较元素 raw bit-exact，
6 个 scale bit-exact。这证明强保活 metadata 数量、由此产生的 allocator 压力和
packed-weight 地址变化单独都不足以触发失败。

第二项只保留最小 trace metadata（外加所有矩阵固定存在的四 bucket sacrificial
warmup owner），但先完整执行 128 次 native；随后第一个 PyPTO step 精确复现旧
签名：compressed KV 最大差 `3.7890625`、SWA KV 最大差
`3.3277854919433594`、indexer K 762/768 个 raw mismatch 且最大差 122 bins、
6 个 scale 全不等；main/inner 仍分别只差约 `2.61e-8`/`2.24e-8`，write set、
block0 和 page padding 地址边界仍正确。由此把触发条件收窄为“长 native 执行后
遗留的 device/runtime 状态”，而不是 128 静态 specialization 或 metadata 保活。

第三项在 native 前就完成 PyPTO weight pack，安装 dispatch 仍严格放在 native
之后；失败数值与第二项相同。因此可以继续排除 native 之后 source weight 被修改、
weight pack 时机或 packed-weight allocator 地址作为根因。下一步应采用诊断性 SoC
state scrub 验证具体硬件状态，并优先比较完整 `AscendC::InitSocState()` 与当前
child wrapper 只有 `set_atomic_none()` 的差异。修复必须落在单 kernel 自身的入口
状态合同内，不能靠跨算子提前启动或隐藏同步规避。

## 52. 目标追加：功能闭环后继续优化并以超过 native 为目标

用户在功能开发过程中明确追加：不能以“功能符合预期”作为任务终点；所有功能性
开发完成后，要继续尽力优化性能，目标是超过 native 方式。

本轮将“超过”固定为可审计契约：

1. 使用相同 B/S/context/layer/trace workload 和相同输入、state、精度门禁；
2. native 主基线是 production 默认配置，包括其真实 multi-stream overlap；
3. 在同一逻辑 device0 上按 ABBA/轮转采样，至少 20 次 warmup、100 次样本，
   报 p50/p90/p99/min/std；
4. cold-start/validation 排除项和 steady-state 必计项采用下文完整定义；
5. Host enqueue、taskQueue dequeue、device span、graph replay、metadata update
   分开报告；地址稳定与地址变化分别测量；
6. 不允许关闭 native 优化、降低精度、缩小 PyPTO workload、只取最好样本，或
   越过 L1 单算子边界制造结论；
7. forced-serial native 只用于解释 overlap 收益，不能替代 production native
   成为胜负基线。

随后用户进一步明确：PyPTO 第一次算子编译及其他 warmup 阶段不属于性能目标的
胜负范围，只要求后续常态化性能超过 native。因此正式报告会把首次 program
compile、PTOAS/codegen、binary 注册、runtime/context/owner 创建与 prepare、
weight pack、全部 ordinary/ACLGraph warmup、ACLGraph capture、最终采样地址的首次
ordinary invocation 或 replay、一次性 structure-cache 填充、event pool/handle
初始化，以及 correctness golden 的生成与比较分列为 cold-start/validation 数据；
steady-state ABBA 采样从它们全部结束并由 caller 外部 quiesce 后开始。

每次常态调用真实发生的参数校验、tensor 地址/scalar patch、taskQueue
enqueue/dequeue、AICPU/AICore device scheduler 和 kernel execution 成本仍保留；
允许单独拆栏，但不能借“warmup 排除”从正式主指标删除。尤其需要区分：最终采样
地址第一次调用的一次性 binding/cache 填充可以排除，但真实 workload 若在稳态
持续改变 tensor 地址或 scalar，由此反复产生的 patch/cache miss 必须计入自身路径。

性能工作顺序保持“先功能、后优化”。任何优化都必须重新通过单 op、四层、
stateful churn、TRB/HBG ACLGraph 与六类 state 正确性矩阵。当前 HBG 数百毫秒
诊断值仍属于首要性能缺陷；在正式 ABBA runner 和 profiler 完成前，不声称已经
超过 native，也不把本目标标记为完成。

## 53. 真实 HC attention-half 一层链路首次上板通过

### 53.1 运行口径

在 fresh device0/TRB process 上运行一层 `PPPP` candidate，shell 使用默认的
`fidelity`，而不是旧的 deterministic affine-only 诊断模式。完整日志为：

`/tmp/csa_real_hc_1l_pppp_trb.log`

本次命令从一开始启用 `set -o pipefail`，Python/pipeline 真实退出码为 0。实际链路
是生产形态的
`npu_hc_pre_v2 -> AscendRMSNorm -> DSA -> npu_hc_post`；输入形状为
`[32, 4, 4096]` BF16，HC-pre 产生 `[32, 4096]` BF16 hidden、`[32, 4]`
FP32 post 和 `[32, 4, 4]` FP32 comb，HC-post 恢复 `[32, 4, 4096]` BF16。
NNNN reference、same-input local native oracle 和 PyPTO candidate 使用相互独立的
mutable state world；local oracle/candidate 共享完全相同的 normalized tensor、
residual、post 和 comb，仅 DSA backend 不同。

### 53.2 数值与状态证据

runner 最终 `close=true`，且：

- DSA output 最大绝对误差 `0.0341796875`，一层 HC-post 后最终输出最大绝对误差
  `0.03424072265625`，均通过预先固定的 `atol=rtol=0.1` 门禁；
- compressed KV 当前有效区最大绝对误差 `1.52587890625e-05`，SWA KV bit-exact；
- main/inner compressor state 最大绝对误差分别为
  `2.682209014892578e-07` 和 `2.384185791015625e-07`；
- indexer K 与 FP16 scale 均 bit-exact，1024 个本次比较 K 元素的 raw mismatch 为
  0，反量化误差也为 0；
- 三个 cache world 的六类 storage 地址互不重叠，真实 HC operator、shape、dtype、
  tensor 地址、RMSNorm owner/weight 与 HC 参数地址均已写入结果 evidence；
- `real_hc_complete=true`、`local_oracle_same_candidate_inputs=true`、
  `candidate_sacrificial_warmup_reset=true`。

这条证据把 Phase 3 从“只验证 deterministic shell”推进到真实 HC
attention-half 的一层功能闭环，但仍不能代表四层 PPPP/NPNP/PNPN、HBG 或
ACLGraph 已通过，也不包含完整 DecoderLayer 的第二段 HC、FFN/MoE。

### 53.3 上板后只读复审发现的结果闭包缺口

静态复审确认真实 HC 的参数顺序、shape/dtype、RMSNorm 与 HC-post 调用均和生产
`deepseek_v4.py` 对齐，同时发现三项需要在扩大上板矩阵前修紧的问题：

1. `A3NativeCapsuleCompareResult.close` 尚未把 resource evidence、same-input、
   sacrificial-warmup reset 和 topology/evidence 对应关系全部纳入自证闭包；当前
   runner 在构造结果前会独立校验资源，所以本次结果不是假通过，但落盘对象本身
   仍可能在字段被破坏后错误保持 `close=true`。
2. 某些初始化/cleanup 失败路径没有把可重试 owner 附到异常；这不影响本次成功
   退出，但不满足失败后由调用方显式重试 close 的所有权契约。
3. packed weight evidence 只检查非空和 owner 不同，尚未直接证明跨 layer packed
   storage 地址集合不相交。

上述三项将补充反例测试并修正后再运行多层真实 HC；修正不得改变已通过的数值
容差，也不得由 GC/atexit 偷做 runtime close。

### 53.4 结果闭包修紧与四层 TRB 拓扑矩阵

复审缺口已修正：result 现在现场重算 topology/resource/HC evidence，要求 same-input
local oracle、sacrificial warmup reset 和六类 state family 均闭合；同步、卸载 hook 或
runtime owner close 任一步失败时，都保留 primary error，并把未关闭 owner 以
`decode_csa_cleanup_owner` 挂到异常供显式重试，GC/atexit 不偷 close。packed weight
按层内和跨层地址集合检查不重叠；`freqs_cos/freqs_sin/hadamard_idx` 是有意共享的
只读静态参数，单独记录并要求所有 PyPTO 层一致。相关证据伪造与 cleanup 反例的
53 项独立定向测试通过，ruff check/format 和 diff-check 通过。

随后在三个 fresh device0/TRB process 上运行四层真实 HC fidelity shell：

- PPPP：`/tmp/csa_real_hc_4l_pppp_trb.log`
- NPNP：`/tmp/csa_real_hc_4l_npnp_trb.log`
- PNPN：`/tmp/csa_real_hc_4l_pnpn_trb.log`

三条命令均启用 `set -o pipefail`，真实退出码和顶层 `close` 均为 0/true；三种结果
的 `topology_evidence_complete`、`resource_evidence_complete`、
`real_hc_complete`、`shell_evidence_complete`、`execution_contract_complete` 均为
true。PPPP 的四层 PyPTO packed weight 地址集合互不相交，四层 attention/wrapper/
prepared-weight、HC 参数、RMSNorm 与三套 mutable cache world 都独立；共享 runtime
owner 和共享静态 RoPE/Hadamard 则符合显式合同。NPNP/PNPN 分别证明 native 在
PyPTO 前后以及 PyPTO 在 native 前后时，真实 HC 串链与 state 可见性均正确。

PPPP 每层 DSA output 最大绝对误差依次为约 0.03418、0.03882、0.04639、0.04248，
四层最终 output 最大绝对误差约 0.10596，在预先固定的四层 `chain_atol=0.2`、
`chain_rtol=0.1` 内通过；所有 layer-local 六类 state 与 indexer 联合量化门禁通过。
这仍是 production HC attention-half 加人工 bridge，而不是完整 DecoderLayer；但相比
一层，它已覆盖真实 HC 四层累积、独立资源和两种 native/PyPTO 交错方向。

## 54. Stateful 多 bucket ACLGraph 首个 TRB 闭环通过

### 54.1 Host 合同与上板形态

新增的 `a3_stateful_aclgraph_compare.py` 不把四个 static specialization 伪装成一张
动态图，而是对 B4/B8/B12/B16 各 capture 一张固定地址 ACLGraph；四张图共享一个
process-pinned PyPTO runtime/cache owner 和同一组六类 mutable state。所有
graph-visible metadata、hidden、output 及每步 D2D source 都在 capture 前分配；
replay 只在对应 graph stream 上排入新 block 初始化、metadata/hidden D2D copy、
output canary fill 与 `graph.replay()`，同步发生在 caller 的 replay 后验数边界。

Host 定向验证覆盖了四图路由、地址不变、重复 metadata update、result 证据闭包、
capture/replay 源码反例和 CLI 参数门禁；与正式 benchmark 合计 54 项测试通过，ruff
check、format check 和 `git diff --check` 通过。

### 54.2 device0/TRB 4-step 结果

fresh device0/TRB process 的完整日志为：

`/tmp/csa_stateful_aclgraph_trb_4.log`

命令启用 `set -o pipefail`，真实退出码为 0，顶层 `close=true`。结果包含：

- `graph_model=one_graph_per_static_bucket`，四张 graph、一个 runtime owner，四个
  retained capture binding；
- replay 顺序 B4/A3、B8/A7、B12/A11、B16/A15，四步都属于 partial bucket；
- 四组 metadata 地址在 capture 前后及 replay 后逐项相同，hidden/output 地址稳定；
  warmup 与 capture 使用不同 IO 地址，所有 staging owner 在同步后归零；
- 四步合计 49 个 block reuse event；逐步 output、六类 state、step-local changed-row
  write-set、block0、page-padding canary 与 PyPTO padded output 全部通过；
- step 3 仍只有已经在 eager churn 中证明过的同一类 1-bin INT8 边界：4096 个本步
  K 元素中 1 个 mismatch、允许 1 个，scale bit-exact、反量化误差一个 scale
  quantum；没有新增或扩散的量化差异。

这条证据首次证明四个 static bucket graph 能在同一 PyPTO runtime/cache owner 下
按真实 admit/retire/compact/reuse 状态连续工作。它只完成 TRB/4-step；32/128-step
的多次同图 metadata 更新和 HBG pristine/working graph 恢复仍需分别验证。

### 54.3 device0/HBG 4-step 结果

随后在另一个 fresh device0/HBG process 运行完全相同的四 bucket/4-step 轨迹：

`/tmp/csa_stateful_aclgraph_hbg_4.log`

命令同样启用 `set -o pipefail`，真实退出码为 0，顶层 `close=true`。四张 HBG
specialization 各完成 capture 和 replay；metadata/IO 地址、retained binding、49 次
reuse、六类 state/write-set、block0/page-padding canary、padded output 和联合量化
结果均与 TRB 的通过证据一致。特别地，四张图依次 capture 时每次都在 sacrificial
执行后恢复初始 mutable state，replay 后没有出现 working graph 或 completion state
残留。这证明当前 HBG graph-as-tiling 生命周期至少在“四个 context、各一次
capture+replay”的 ACLGraph 组合下闭环。

因为每张图目前只 replay 一次，本条尚不能证明同一张 HBG 图在 metadata/address
内容变化后的重复 replay；该门禁由 32-step（每个 bucket 多次命中）承担。

### 54.4 TRB/HBG 32-step 重复 replay 均通过

在两个互相独立的 fresh device0 process 上分别运行 32-step：

- TRB：`/tmp/csa_stateful_aclgraph_trb_32.log`
- HBG：`/tmp/csa_stateful_aclgraph_hbg_32.log`

两条命令均启用 `set -o pipefail`，真实退出码均为 0，顶层均为 `close=true`。
相同轨迹的四张图分别 replay B4=8、B8=12、B12=8、B16=4 次，累计 1654 个
block reuse event。32/32 个 step 的 output、六类 state、step-local write-set、
block0/page-padding canary、PyPTO padded output 和联合量化门禁均通过；四组
metadata 与 IO storage 地址从 capture 到最后一次 replay 都没有变化，staging owner
每次在 caller 同步后清零。

HBG 与 TRB 得到相同的逐步 correctness 结论，证明当前 HBG package/working graph
并非只在第一次 replay 偶然可用：同一 captured node 在固定地址内容多次更新后，
能够反复恢复 completion/host-done 状态并重新执行。至此 stateful ACLGraph 的主要
复用验收已由 32-step 完成；128-step endurance 仍被 §51.9 已收窄的“长 native
前置执行遗留 device/runtime 状态”阻塞，必须先完成 SoC state 入口合同修复，再
重跑 eager 与 ACLGraph 128-step，不能把静态容量继续当成已证实根因。

## 55. HBG 稳态 320 ms 的静态根因与第一项收缩

旧 HBG ACLGraph 结果在 5 次 warmup、20 次 replay 后仍约 `319.842420 ms`，而
native graph 约 `0.671216 ms`；计时区间内部只有 `graph.replay()` 和批末 caller
同步，input copy、state restore、compile、capture 均在外部。因此它是稳态 replay
问题，不属于本轮明确排除的首次编译/warmup 成本。

静态审计确认 B4/B8/B12/B16 当前 orchestration 都只有 42 个顶层 submit，但 CSA
backend 创建 HBG L1 context 时没有指定 `ring_task_window`，因而继承 Simpler 历史
L2 默认 16384。其结果是每个 CANN-owned HostArgs/runtime image 约 93 MB；每次
replay 时 AICPU 都会重新 invalidate blob、将 pristine shared-memory/runtime-arena
复制到 working region、flush/invalidate 完整 capacity，然后再执行通用 scheduler
的 init、classify、poll、dispatch、destroy 和 deinit。Python backend、Host graph
builder、H2D graph build 与 taskQueue callback 不会在 replay 时重入，所以不能把
320 ms 解释成冷启动。

最初根据 42 个顶层 submit 将窗口收缩为 64，但首次 fresh device0/HBG 上板在
warmup 阶段正确触发 runtime fail-closed：追加下一项前
`scope_task_count=63, active_tasks=63/64`，并明确报告 HBG whole-graph-resident
scope 要求 task count 严格小于 window。日志为：

`/tmp/csa_hbg_window64_aclgraph_5w20i.log`

这证明顶层 submit 数不能直接当成 runtime scope task 数；内部展开后至少需要第
64 个 slot。该失败发生在 warmup，未产生任何性能样本，也没有被当作优化结果。
据此立即把仅属于本 CSA adapter 的安全收缩修正为：

- HBG 默认显式使用 `ring_task_window=128`，这是 device0 已证明大于实际 scope
  task 数的最小 2 次幂；
- TRB 仍传 `None`，行为完全不变；
- vLLM 私有 backend 不读取或公开 `PTO2_RING_TASK_WINDOW`；128 是本 specialization
  的内部已验证容量，不增加用户配置面，也遵守 vLLM-Ascend 环境变量集中管理规则；
- 未来 graph 超过窗口时依赖 runtime warmup fail-closed，再由实现代码显式提高并
  重新验证，不静默截断，也不让普通调用者猜测 scheduler 容量。

Host 回归已覆盖 HBG 内部默认值和 TRB 不变。历史通用 HBG 数据显示
compact arena + window 64 的 package 为 `1,097,392 B`、replay 为 `3.790 ms`，但
该窗口对 CSA 明确不足，而且这也不是本 CSA 的新实测，不能拿来宣称性能结果；
必须用修正后的 window 128 在 device0 重新运行正式
steady-state ABBA。即使下降到几毫秒，42 个异构 child task 与通用 scheduler 仍很
可能慢于 native 约 0.67 ms，后续还需按真实 live task 做稀疏 snapshot/cache
maintenance，最终可能需要面向 CSA DAG 的静态 wave/task-table dispatcher。

## 56. 128-step 失败与 SoC state 入口合同审计

### 56.1 证据已经把失败收窄到 native 长前置执行

128-step 诊断矩阵把 static specialization、retained metadata allocation、native
precondition 和 weight-pack 时机拆开后，得到如下因果边界：

- `spec128 / execute1 / native1 / retained1` 通过；
- `spec128 / execute1 / native1 / retained128` 通过，排除仅保持 128 份 device
  metadata owner 及 allocator layout 即触发错误；
- `spec128 / execute1 / native128 / retained1` 失败，且与完整 128-step
  的 step0 具有相同签名：main/inner state 正确，compressed KV、SWA KV 和
  indexer K/scale 大面积错误；
- 把 PyPTO packed weight 提前到 native128 之前，仍保持同样失败。

因此 static extent、metadata 强 owner、source weight 被 native 改写、pack 时机和
packed-weight allocator 地址都不再是主要候选；当前最强假设是 128 次 native
DSA 在物理 AI Core 上留下了 PyPTO child 入口没有完整恢复的 SoC
状态。这与已有 Qwen3/ATB atomic 泄漏案例是同一类边界缺陷，但本次
CSA 失败不能直接等同为 atomic，因为当前每个 child wrapper 已有
`set_atomic_none()`。

### 56.2 CANN 9.2 A2/A3 入口语义与实际 CSA 产物

本机 CANN 9.2 的 `AscendCUtils::InitSocStateImpl()` 在 `__NPU_ARCH__ == 2201`
下执行：

```cpp
set_atomic_none();
set_mask_norm();
if ASCEND_IS_AIC {
    set_l1_3d_size(0);
    set_padding(0);
} else {
    set_vector_mask(-1, -1);
}
```

对已上板的 B16/TRB 产物做静态盘点，共有 44 个硬件 child：13 个
AIC 和 31 个 AIV。盘点结果为：

- 44/44 wrapper 都有 `set_atomic_none()`；
- 31/31 AIV body 都有 `set_mask_norm()` 和 `set_vector_mask(-1, -1)`；
- 13/13 AIC 都没有 `set_l1_3d_size(0)` 或 `set_padding(0)`，其中 11/13
  连 `set_mask_norm()` 也没有；另外 2 个是 mixed source 在 AIC 侧也包含了
  vector 路径的 mask 设置，不能代表完整 AIC 初始化。

这与“只有 `set_atomic_none()` 不等价于 `InitSocState()`”的预期完全一致，
也为 CSA 大量 matmul/cache writer 错而部分 state path 正确的失败签名
提供了合理解释。但在用完整 `InitSocState()` 做真机 A/B 之前，仍只能
称为强假设，不把静态盘点写成已证实根因。

### 56.3 为什么不在 vllm-ascend 内加 extern “全核 scrub wave”

`@pl.jit.extern` 可以编译一个手写 AIC/AIV kernel，`pl.spmd_submit` 也能返回
TaskId，因此表面上可以在 CSA 开头发一次 `InitSocState()` wave，再让根
task 依赖该 TaskId。本轮明确不实现这个方案，原因不是语法做不到，
而是它无法建立所需的语义保证：

1. Init wave 是独立硬件 task。TaskId 只能证明“wave 完成后 consumer 才可运行”，
   不能把 `InitSocState()` 与 consumer 的第一条真实指令放在同一个
   `kernel_entry` 内。
2. CSA 自身是 44 个 child 的 DAG。一次入口 wave 之后，前一个 CSA child
   仍可修改某个物理 core 的全局状态，后一 child 可被调度到同核；一次
   wave 不能代替“每个 child 入口”的合同。
3. 如果改为每个 source-level scope 前加 wave，不仅会把 DAG 全局串行化，
   compiler 后续 outline/split/transform 仍可产生新 child；vllm-ascend 源码层
   无法证明没有遗漏任何最终硬件入口。
4. 全核覆盖也无法从 JIT 源码层严格建立。PyPTO 910B backend 静态
   SoC 模型是 24 AIC/48 AIV，而当前 A3 实际 runtime 可用为 20/40；准确
   数量只能由 AICPU 侧 `rt_available_cluster_count()`/`rt_available_aiv_count()`
   获得。用 24/48 做 `sync_start` 会触发超额 deadlock guard，去掉
   `sync_start` 后又无法将“确实触达每个可用物理 core”作为公开合同。
5. 该 wave 每次都会新增全核调度和 barrier，直接进入本轮要超过
   native 的 steady-state 成本；即使它在当前 128-step 轨迹上经验性消除
   错误，也只证明假设，不是可交付的产品修复。

特别地，“当前 CSA 的 AIV body 恰好都自行设置 mask，且静态搜索没有
CSA child 再设置 L1 3D/padding”只能说明单次 AIC wave 是一个有价值的临时
诊断。它依赖当前 PTOAS 产物的偶然形态，下一次 codegen 或新 primitive
便可破坏，因此按“不在业务仓内做不完整 hack”的原则放弃落码。

### 56.4 唯一可建立完整语义的修复边界

修复应位于 PyPTO 通用 hardware child wrapper 生成边界：在每个最终
`kernel_entry` 进入 runtime sub-block/SPMD 解析、参数解包和 PTOAS body 之前，
调用完整 `AscendC::InitSocState()`；CPU simulator/costmodel 路径为明确 no-op。
这样初始化与真实 consumer 在同一物理 core、同一 kernel invocation 内，同时
自动覆盖 outline、split、mixed AIC/AIV 以及未来新增的 child，无需业务代码
知道核数或重建 task DAG。

修复后的必要验收顺序为：

1. codegen 单测证明 AIC、AIV、split/mixed 每个 wrapper 都在任何现有 setup
   之前发出一次完整初始化；
2. 最小状态污染回归分别覆盖 AIC 和 AIV，不只测 atomic；
3. fresh device0 重跑 `spec128/execute1/native128/retained1` 诊断，再跑完整
   128-step eager TRB/HBG；
4. 重跑 128-step TRB/HBG ACLGraph，确认 capture/replay 没有状态漂移；
5. 重跑 steady-state ABBA；`InitSocState()` 本身是每次 child 的必要语义成本，
   不能排除在正式计时外，需在 task 数量/调度优化中吸收。

本轮根据范围约束不修改 `/pto/pypto` 或 simpler，也没有向
vllm-ascend 生产 kernel 提交 extern scrub wave；这是有意保留正确所有权
边界，不是未完成的临时遗漏。

### 56.5 Host 侧能力验证

为避免把“工具链根本不支持 extern/TaskId”误当成本节结论，使用当前
PyPTO main 和与 simpler `b6f905f63277` 匹配的 binding，运行了：

```text
tests/ut/jit/test_extern_kernel.py
tests/ut/language/parser/test_spmd_submit.py
tests/ut/codegen/test_orchestration_task_deps.py
```

结果为 `66 passed`。这批 Host 回归直接证明：

- `@pl.jit.extern` 能正确引入 AIC、AIV 和 mixed/dual-AIV 外部 kernel；
- `pl.spmd_submit(..., deps=[...])` 能产生可被后续 task 依赖的 TaskId；
- mixed group 可生成 AIC/AIV launch spec 和依赖。

所以不采用 scrub wave 是经过现有能力验证后的语义决定：这些能力只能
生成“前置独立 task”，仍无法把初始化放到 compiler 最终产生的每个
child `kernel_entry` 中。本轮未上 NPU，没有产生新的真机根因结论。

## 57. window 128 后的正式稳态性能基线

### 57.1 device0/window 128 首次实测

修正为 window 128 后，在另一个 fresh device0/HBG process 重跑 B4/S8 ACLGraph：

`/tmp/csa_hbg_window128_aclgraph_5w20i.log`

命令启用 `set -o pipefail`，真实退出码为 0，顶层 `close=true`；三组不同 input
scale 的 replay 均通过 output、六类 state 与地址隔离门禁。5 次 warmup 后的 20 次
计时得到：

- native graph：`0.6639058934524655 ms`；
- PyPTO HBG graph：`6.441163620911539 ms`；
- native/PyPTO 比值：`0.10307235346375367`，即 PyPTO 仍约慢 9.7 倍。

相较旧 `319.842420 ms`，window 128 将本 CSA 的真实稳态 replay 降低约 49.7 倍，
证明 16384 capacity 的 package restore/cache maintenance 确实是首要瓶颈；但它还
没有达到“超过 native”的目标，也不能用方向性单边计时替代正式 ABBA。剩余约
5.8 ms 差距不再可能靠排除首次编译或增加 warmup 消失，应继续缩减 live-range
snapshot/invalidate 和 42 个异构 child 的通用调度成本。

### 57.2 HBG 正式同卡 ABBA 稳态基线

随后使用独立 fresh device0/HBG process 运行正式口径：20 次 warmup、100 次
sample、同一 caller stream、ABBA 顺序、每 20 次 enqueue 由 caller 批量同步；
native 明确验证 production `multistream_dsv4_dsa_overlap=True`。日志为：

`/tmp/csa_hbg_window128_formal_abba_w20_s100.log`

命令真实退出码为 0，sample 前后 correctness gate 均 `close=true`，indexer K/scale
bit-exact。program compile、PTOAS/codegen、binary/context prepare、weight pack、
ordinary warmup、ACLGraph capture、最终采样地址的首次 replay、structure-cache
填充、event handle 初始化和 golden comparison 全部在正式 sample 外。稳态结果：

| backend | device p50 | device p90 | device p99 | Host replay p50 |
| --- | ---: | ---: | ---: | ---: |
| native production | `0.633330 ms` | `0.639970 ms` | `0.645454 ms` | `17.866 us` |
| PyPTO HBG/window128 | `6.423010 ms` | `6.456438 ms` | `6.476929 ms` | `17.161 us` |

PyPTO 的 Host graph replay enqueue 略低于 native，且都只有十几微秒；约 10.1 倍的
p50 差距完整落在 taskQueue consumer 到设备执行完成的 span，进一步证明当前主要
瓶颈是 HBG AICPU package restore/cache maintenance 与通用 42-child 调度，而不是
Python、首次编译、capture 或 Host enqueue。该正式结果满足用户追加的“只比较
常态化性能”口径，但结论是当前仍未超过 native；它是后续优化必须改善的基线，
不能被更有利的单次最小值替代。

### 57.3 TRB 正式同卡 ABBA 稳态基线

随后在另一个 fresh device0/TRB process 使用完全相同的正式口径运行：20 次
warmup、100 次 sample、同一 caller stream、ABBA 顺序、每 20 次 enqueue 由
caller 批量同步；native production 的 multi-stream overlap 保持开启。日志为：

`/tmp/csa_trb_formal_abba_w20_s100.log`

命令真实退出码为 0。采样前、采样后 correctness gate 均 `close=true`，indexer
K/scale 均 bit-exact；因此这 100 个样本没有被 §56 所述长 native 前置执行状态
问题污染。与 HBG 基线相同，program compile、PTOAS/codegen、binary/context
prepare、weight pack、普通 warmup、ACLGraph capture、最终采样地址的首次 replay、
structure-cache 填充、event handle 初始化和 golden comparison 全部在正式 sample
之前完成，不进入胜负口径。结果为：

| backend | device p50 | device p90 | device p99 | Host replay p50 |
| --- | ---: | ---: | ---: | ---: |
| native production | `0.641550 ms` | `0.648664 ms` | `0.652178 ms` | `18.546 us` |
| PyPTO TRB | `0.820920 ms` | `0.832736 ms` | `0.838568 ms` | `18.926 us` |

TRB 的 Host replay p50 只比 native 高 `0.380 us`，而 device span p50 高
`0.179370 ms`，即当前 TRB steady state 约为 native 的 `1.280` 倍、仍慢约
`27.96%`。差距同样不在首次编译、warmup、capture 或 Python enqueue，而是在
taskQueue consumer dequeue/launch 与设备 scheduler/44-child DAG 执行所覆盖的
稳态设备区间。与 HBG/window128 的约 `10.14` 倍相比，TRB 已明显接近 native，
因此后续性能优化应先用 profiler 将这 `0.179 ms` 拆成 AICPU scheduler、child
dispatch/barrier、AI Core 尾部和 native overlap 机会成本，再逐项 A/B；不能把
一次性阶段重新计入或移出任一方来改变结论。

### 57.4 qk_pv 按真实 AIC 数启动后的第一轮优化

device0 的公开属性为 `Ascend910_9362`、`cube_core_num=20`、
`vector_core_num=40`。但移植自参考实现的 `qk_pv` 把 launch width 和 lane stride
都固定为 24；这会把 24 个 block 投到 20 个 AIC 上，形成第二波调度尾部。该问题
不应通过把产品代码再硬编码成 20 修复，因为 A2/A3 不同 SKU 的可用核数仍可能
不同。当前 PyPTO 已提供 orchestration-side `pl.system.available_cluster_count()`，
同时 child 可用 `pl.tile.get_block_num()` 读取本次 blockDim，因此改为：

- `pl.spmd(pl.system.available_cluster_count(), ...)` 按本次 runtime 的实际 AIC 数
  启动 qk_pv；
- child 用 `pl.tile.get_block_num()` 同时计算 `qk_lane_iters` 和 strided item 下标；
- 不增加 public scalar，不查询 ACLGraph/capture 状态，也不改变 L1 单算子边界。

生成物确认 orchestration 发出
`set_block_num(rt_available_cluster_count())`，AIC/AIV 两个 child 都从参数读取
实际 block number。静态 ABI 回归为 `13 passed`，ruff 与 `git diff --check`
通过。随后在 fresh device0/TRB process 重跑完全相同的 20-warmup/100-sample
正式 ACLGraph ABBA：

`/tmp/csa_trb_runtime20_formal_abba_w20_s100.log`

真实退出码为 0，采样前后 correctness gate 均 `close=true`、indexer bit-exact：

| 版本/backend | device p50 | device p90 | device p99 | Host replay p50 |
| --- | ---: | ---: | ---: | ---: |
| 本轮 native production | `0.644820 ms` | `0.650516 ms` | `0.656232 ms` | `18.806 us` |
| 旧 PyPTO TRB/固定 24 | `0.820920 ms` | `0.832736 ms` | `0.838568 ms` | `18.926 us` |
| 新 PyPTO TRB/runtime 20 | `0.799960 ms` | `0.812162 ms` | `0.821298 ms` | `18.880 us` |

以两次 PyPTO p50 比较，该项降低 `20.960 us`、约 `2.55%`；同轮相对 native
仍为 `1.241` 倍，慢约 `24.06%`，尚余 `155.140 us`。Host replay 基本不变，
符合“只消除 device 侧超额 block 尾部”的预期。该优化保留，但不能据此把剩余
差距都归因于 qk_pv；下一步仍需用 profiler 分解 AICPU dispatch、各 child 时长与
native 多流 overlap。

## 58. steady-state profiler 审计与剩余开销归属

### 58.1 Host 路径不是当前主要差距

对 runtime-20 之后的生成物和 benchmark 分层重新审计，正式 ACLGraph 样本中
native/PyPTO 的 Host replay enqueue 均稳定在十几微秒，差异不足 1 微秒；而当时
device span 仍相差 155 微秒。因此后续优化不再围绕 Python wrapper、首次 JIT 或
capture 做文章，而直接审计设备内 task DAG。exact-bucket 产物在该阶段包含 43 个
不同 child binary、约 62 个真实硬件 task submit，以及 allocator/ring 依赖节点。
这解释了“顶层只有一个 ACLGraph node”并不等于设备侧只有一次 kernel dispatch。

### 58.2 profiler 必须 fail closed

benchmark 的可选 profiler 路径已经改为强校验产物，不再在 CANN 解析失败后把空
目录当作成功证据。最终候选上执行的诊断日志为：

`/tmp/csa_trb_qk_valid_only_profile.log`

原始目录为：

`/tmp/csa_trb_steady_profile_v2/decode_csa_tensormap_and_ringbuffer_aclgraph_2822003_1788377861494081243`

CANN export 生成的 `trace_view.json` 在 byte 273324 截断，全部 timeline/relation/
kernel parser 报错，同时缺少 `kernel_details.csv`。harness 因此抛出
`BenchmarkError` 并以非零退出，明确报告“profile analysis failed closed”。这次
诊断运行不属于正式计时样本；它既不能提供可信 child 级归因，也不推翻三个完全
独立、未开 profiler 的正式 ABBA 结果。后续只有在 parser 产出完整 JSON 与 kernel
表之后，才允许引用 profiler 数字。

## 59. exact shape、无效 work item 与被否决的实验

### 59.1 exact token bucket 与输出 scratch 收缩

正式 workload 是 B4/S8，即 32 个 token。早期内部仍携带 B16/T128 风格的输出和
部分 scratch。本轮让 specialization 的真实 token 轴进入 indexer、attention 和
output projection 的内部 shape，保留只有编译器静态 tile 确实需要的 `T_PAD`。
日志 `/tmp/csa_trb_exact_bucket_formal_abba_w20_s100.log` 得到：

| backend | p50 | p90 | p99 |
| --- | ---: | ---: | ---: |
| native | `661.730 us` | `668.120 us` | `674.899 us` |
| PyPTO | `755.840 us` | `773.022 us` | `781.165 us` |

paired p50 gap 从 runtime-20 的 `155.140 us` 收缩为 `94.110 us`。该结果同时包含
更小的 output projection、plan 展开和 scratch 节点，不能把全部收益归因到某一个
child kernel。

### 59.2 无效 sparse block 不再制造 GM 工作

随后清理 padded/不可见 sparse block 的工作：无效项不再写可选 block scratch，
QK/PV 与 merge 继续由有效 mask 保护。日志
`/tmp/csa_trb_skip_invalid_formal_abba_w20_s100.log` 得到 native/PyPTO p50 分别为
`628.820/712.520 us`，paired gap 为 `83.700 us`。这项保留。

审计也纠正了一个容易夸大的结论：这一版本之前的无效项已经不会真的执行完整
QK/PV，剩余浪费主要是 `qk_order` 仍携带无效 tail，AIC lane 仍要遍历并判断，
而不是几十 MiB 的无效 GM 数据搬运。

### 59.3 grouped output projection 实验被回退

为减少 8 组 `proj_a -> cast -> proj_b` 的 child dispatch，曾实验把 group 合并到
较少 scope。功能性四步回归通过，但正式日志
`/tmp/csa_trb_group_merged_formal_abba_w20_s100.log` 的 native/PyPTO p50 为
`630.920/771.340 us`，paired gap 反而扩大到 `140.420 us`。原因是减少 task 数
同时破坏了原有 group 间流水和 AIC/AIV overlap；“child 更少”本身不是性能保证。
该实验完整回退，不进入最终候选。

### 59.4 indexer scratch 机械 exact-T 化被回退

另一个实验把 indexer 的所有内部行轴机械收缩到 exact `t_dim`。功能正确，但
`weights_proj_reduce` 的 tile/布局和调度收益不足以抵消新形态的 codegen，正式日志
`/tmp/csa_trb_exact_indexer_formal_abba_w20_s100.log` 得到 native/PyPTO p50
`636.550/742.080 us`，paired gap `105.530 us`，比 skip-invalid 基线更差。因此只
保留真正能减少 work 的 exact shape，不把“scratch 越小越快”作为未经实测的规则。

## 60. 短上下文 indexer selection-set 快路径

### 60.1 为什么 `visible_len <= 512` 不需要 score/sort

CSA indexer 最终只选择最多 512 个 compressed KV index。当本 token 的可见
compressed 长度不超过 512 时，所有可见 index 必然全部入选，score 的具体大小
不会改变入选集合。基于这个融合算子内部合同，本轮做了两层优化：

1. `idx_topk_full` 从 `[tokens, 4096]` 收缩为 `[tokens, 512]`；
2. `visible_len <= 512` 时直接生成逻辑 `arange`，余下位置填 `-1`；超过 512 时
   保留完整 score/sort/merge-sort 路径。

这里刻意使用“selection-set 等价”，而不声称 raw top-k 顺序等价。当前 A3 native
QLI arch32 即使小于 512 也按 score 排序；直接 `arange` 改变 attention 的 128-row
分块和 online-softmax 累加顺序，所以可能有容差内浮点差异。融合输出和六类状态
门禁仍必须通过；如果未来公开 standalone indexer 的 raw ordered-topk API，这条
快路径不能直接复用其契约。

仅启用 direct-topk、尚未跳过上游 score DAG 时，日志
`/tmp/csa_trb_topk_direct_formal_abba_w20_s100.log` 的 native/PyPTO p50 为
`628.950/706.850 us`，gap `77.900 us`，相对上一候选约再收回 5.8 微秒。

### 60.2 device scalar gate 与九个 predicate task

为了把无意义的上游计算一起跳过，新增的是 PyPTO program 内部 scratch
`need_index_score: [1] INT32`，不是 public ABI 参数；44-slot public ABI 仍为
40 tensors + 4 scalars，输出仍是 slot 39。gate 被并入既有 `csa_rope_step`
orchestration scope：每次调用先写 0，只要任一 active request 同时满足 cache
capacity 与最后可见 compressed extent 大于 512 就写 1，因此每次 replay 都从本次
device metadata 重新计算，不依赖 capture 时的 Python 值。

以下九个只服务于 score 的 task 全部增加同一个 device predicate，并且每个都直接
依赖 gate task，确保 predicate false 时由 runtime retire 并结算 fanout：

1. `idx_qr_proj_matmul`；
2. `idx_qr_proj_dequant`；
3. `qr_rope_swap_idx`；
4. `qr_rope`；
5. `qr_hadamard_matmul`；
6. `qr_hadamard_quant`；
7. `weights_proj`；
8. `weights_proj_reduce`；
9. `score`。

`indexer_compressor` 仍必须运行，因为它更新当前 token 的 indexer KV/cache state。
top-k 明确等待 `score_tid`；当 predicate false 时，该 TaskId 仍以 retired 状态满足
依赖。原先两个 `pl.at` 单任务因不支持 predicate，转换为 `pl.spmd(1, ...)`。

### 60.3 512 边界与长上下文验证

压缩倍率为 4，单 token 的精确边界为：position 2046 对应 511；2047～2050 对应
512；2051 对应 513。S8 batch 中，`start_position=2043` 是最后一个所有 token 都走
快路的窗口，2044 是第一个 mixed 窗口。已在 device0/TRB 分别通过：

- start 0 的 4-step nonzero stateful correctness；
- `/tmp/csa_trb_indexer_predicate_boundary2043.log`；
- `/tmp/csa_trb_indexer_predicate_boundary2044.log`；
- start 8191 长上下文 slow-path；
- B4/B8/B12/B16 partial bucket ACLGraph，padded output 与 canary 均干净。

因此 gate 不是只为 benchmark start=0 写死的 Host 分支，512 两侧和真实长上下文
路径都编译并在设备上执行过。

### 60.4 HBG 的当前限制

当前 simpler 的 HBG resident recorder 尚不能把 task predicate 表达到 resident
graph 中；遇到 predicate 会将 recording 标为 unsupported，并安全重跑 ordinary
path。因此这项优化在 TRB 是正式性能路径，在 HBG 只保证功能正确，不能把 HBG
fallback 的约 9 ms 当作 resident HBG 性能。修复方向是以后给 simpler 增加 graph
predicate 表达，而不是在 vLLM kernel 中删除 device gate 或探测 capture 状态。

## 61. QK plan 只调度真实有效 work item

完成 indexer gate 后，QK plan 仍把每个 token 的所有 optional sparse block 放进
`qk_order`，child 只是在执行时判断无效。最终候选改为：

- 每 token 始终保留 block0；padding token 由 block0 写零 seed，避免 merge 读取
  未初始化 scratch；
- optional block 仅在有效时 append；
- `qk_wcur[0]` 成为真实 valid item count；
- AIC child 读取该动态计数，并用 runtime 可用 AIC 数做 exact lane loop bound，
  不再遍历无效 tail。

这不改变 public ABI、输出数学或 graph 地址，只缩短每次 replay 的设备工作队列。
TRB 四步 stateful correctness 通过；最终 HBG 四 bucket/四步 ACLGraph 回归也通过，
详见 §63。

## 62. 最终 TRB 稳态验收

### 62.1 判定规则

为了避免单进程抖动产生选择性结论，最终验收使用至少三个 fresh process；每个进程
在 device0、同一 caller stream 上按 ABBA 顺序比较，native production overlap
保持开启，每边至少 20 次独立 warmup 和 100 个正式 sample。对每个分位数定义：

```text
Dq = native_q - pypto_q
```

先在每个 fresh process 内求 paired D，再取三个 D 的中位数。当前目标的明确通过
条件是 `median(D50) > 0`、`median(D90) >= 0`、`median(D99) >= 0`，并且 sample
前后 output、六类 mutable state、indexer quantization/write-set 与 canary 全部
通过。TRB ACLGraph 是当前正式性能验收路径；HBG predicate resident 能力未完成
前单独报告功能与 fallback 性能，不能混入 TRB 胜负。

### 62.2 三个 fresh process 原始结果

最终 qk-valid-only 候选的完整日志为：

- `/tmp/csa_trb_qk_valid_only_formal_abba_w20_s100.log`；
- `/tmp/csa_trb_qk_valid_only_formal_abba_run2.log`；
- `/tmp/csa_trb_qk_valid_only_formal_abba_run3.log`。

单位均为微秒：

| process | native p50/p90/p99 | PyPTO p50/p90/p99 | D50/D90/D99 |
| --- | --- | --- | --- |
| run1 | `633.460 / 639.878 / 643.360` | `613.110 / 627.598 / 636.228` | `+20.350 / +12.280 / +7.132` |
| run2 | `645.020 / 652.750 / 655.972` | `628.720 / 644.216 / 650.940` | `+16.300 / +8.534 / +5.032` |
| run3 | `625.040 / 630.470 / 633.401` | `619.290 / 632.836 / 642.396` | `+5.750 / -2.366 / -8.996` |
| paired-D 中位数 | — | — | `+16.300 / +8.534 / +5.032` |

单个 run3 的 p90/p99 有反向抖动，但预先定义的跨进程 paired-D 中位数三项均为正，
因此正式稳态判定为 PASS。不能把 run1 的最好结果单独宣传，也不能把 run3 删除。
该结论只覆盖已声明的 B4/S8 short-context fixture 和 TRB ACLGraph；长上下文与更大
bucket 已完成功能门禁，但尚未形成同样的三进程性能胜负结论。

### 62.3 本轮性能改善的累计含义

原始 TRB p50 为 `820.920 us`，同轮 native 为 `641.550 us`，paired gap
`-179.370 us`。最终三进程中位 paired p50 已变为 PyPTO 领先 `16.300 us`。主要
贡献依次来自 runtime 实际 AIC 数、exact token shape、无效 sparse work 清理、
short-context selection-set/predicate gate，以及 QK valid queue。由于不同 fresh
process 的 native 自身会漂移十几微秒，累计值只能用于解释优化方向；最终是否
通过始终以同轮 ABBA paired-D 规则为准。

## 63. 最终 HBG 回归与 DSL 修正过程

### 63.1 predicate fallback 下的功能回归

最终候选在 fresh device0/HBG process 运行：

```text
python -m tests.pypto_dsv4_decode_csa.a3_stateful_aclgraph_compare \
  --runtime host_build_graph --device 0 --steps 4 --master-port 29875
```

完整日志为 `/tmp/csa_hbg_final_qk_valid_correctness4_retry2.log`，真实退出码为 0。
结果包含一个 runtime owner、B4/B8/B12/B16 四张 graph、四个 partial-bucket replay；
四组 metadata/hidden/output 地址 capture 前后稳定，warmup 与 capture 地址不同，
pending staging owner 同步后均为 0。每步 output、六类 mutable state、legal
write-set、block0/page-padding canary 与 indexer quantization 均通过，顶层
`close=true`。

该结果证明 HBG 在 recorder 不支持 predicate 时能安全 ordinary-fallback 并保持
ACLGraph 功能语义；它不证明 predicate 已驻留到 HBG graph，也不作为本轮性能 PASS。

### 63.2 两次失败的编译修正也保留记录

把原 `pl.at` 转为可带 predicate 的 `spmd(1)` 后，为消除单 block index warning，
曾把 `qr_rope_swap_idx` 和 `weights_proj_reduce` 的 Tensor 写改成 `pl.store`。第一次
HBG 编译在 `/tmp/csa_hbg_final_qk_valid_correctness4.log` 明确失败：`tile.store`
要求 TileType，实际目标/参数是 TensorType。这一写法未上设备。

随后恢复 Tensor slice assignment，但暂时移除了 `pl.tile.get_block_idx()`；第二次
编译 `/tmp/csa_hbg_final_qk_valid_correctness4_retry.log` 被 parser 正确拒绝：
`spmd` body 既未消费 block index，也未 dispatch child，每个 block 会执行相同工作。

最终写法同时满足两层合同：继续使用 Tensor slice assignment，并让 slice 起点显式
由 `block_idx * tile_extent` 计算。即使 blockDim 当前为 1，block index 仍真实进入
数据流；第三次编译和完整 HBG 四 bucket 回归通过。两次失败均为 Host 编译期错误，
没有产生错误 device execution 或污染正式性能样本。

## 64. 用户追加后的最终性能计时边界

本轮目标正式修订为：不要求把 PyPTO 第一次算子编译和任何 warmup 成本摊进胜负，
只要求可持续重复调用的常态化性能超过 native。为了让“稳态”不能被任意解释，正式
`SamplingPhase.SAMPLE` 开始前必须完成并由 caller quiesce/synchronize 的排除项为：

- 第一次 program compile；
- PTOAS/codegen；
- binary registration、materialization；
- runtime/context/owner 创建和 prepare；
- 一次性 weight pack；
- ordinary eager 与 ACLGraph 的全部 warmup；
- ACLGraph capture；
- 最终采样地址组合上的第一次 ordinary invocation/replay；
- 一次性 structure/template cache population；
- event pool、handle 等一次性初始化；
- correctness golden 的创建和比较。

每个正式稳态样本仍必须完整包含：

- 真实 per-call 参数、dtype/shape/stride/device 校验；
- tensor 地址和 scalar patch；
- taskQueue enqueue 与 dequeue；
- AICPU/AICore scheduler；
- 所有 child kernel 的真实执行。

如果 tensor 地址或 scalar 在服务运行中反复改变，并反复引发 cache miss，该 miss 是
稳态工作的一部分，不能继续称为 warmup 后排除。地址 churn benchmark 必须固定地址
序列、working-set 和 cache capacity；只允许把 native/PyPTO 共用、与 backend 无关
且完全相同的 fixture 输入内容更新放到共同计时边界外，任何 PyPTO 专属 metadata
生成、绑定或 patch 都必须计入。最终三进程结果按这一口径取得并已经超过 native。

## 65. 最终静态回归、结果固化与提交边界

### 65.1 移除不必要的用户环境开关

按仓库 `AGENTS.md` 复核后，发现 private backend 曾直接读取
`PTO2_RING_TASK_WINDOW`。该变量属于 PyPTO runtime，不在 vLLM-Ascend 集中
`envs.py` 管理体系内，而且本 CSA 已在 device0 证明 HBG window 128 足够；把
scheduler 容量暴露给普通算子用户也违背“L1 外观保持简单”的目标。因此最终代码
删除该环境读取和合法/非法 override 分支，HBG 始终由 private backend 显式传内部
常量 128，TRB 仍传 `None`。未来 specialization 超过容量时由 runtime warmup
fail-fast，再在实现代码中调整和重新上板，不要求用户猜容量。

这项清理发生在三轮 TRB 计时之后，但不改变计时路径：正式 TRB 从未使用 HBG
window，且计时时环境未设置该变量。持久结果 JSON 同时保存计时源码树 hash
`8ff1f35b...` 与清理后当前源码树 hash `b2da9609...`，没有用后者冒充原始计时
产物。

### 65.2 最终 Host/静态检查

完成上述清理后重新执行：

```text
pytest -q tests/pypto_dsv4_decode_csa
ruff check vllm_ascend/ops/dsa.py \
  vllm_ascend/ops/_pypto_dsv4_csa tests/pypto_dsv4_decode_csa
jq empty tests/pypto_dsv4_decode_csa/results/20260903_device0_trb_steady_state_performance.json
```

结果为 `515 passed, 14 warnings`，ruff 全通过，结果 JSON 语法通过。14 条 warning
全部是环境中 `torch.jit.script_method` 的既有 deprecation warning，不是新增测试
失败。对所有新源码、测试、Markdown 和 JSON 另做 untracked whitespace check，
没有 trailing whitespace/error。

### 65.3 持久性能结果

新增：

`tests/pypto_dsv4_decode_csa/results/20260903_device0_trb_steady_state_performance.json`

文件保存三个 fresh process 的 PID、日志路径与 SHA256、generated artifact、原始
p50/p90/p99、paired difference、Host enqueue 诊断、采样前后 correctness、源码
hash、完整计时边界、HBG 排除原因以及 profiler fail-closed 诊断。run3 的负 p90/
p99 没有被删除。该 JSON 是可审计结果摘要，原始 `/tmp` 日志仍是本机证据；若要在
另一环境长期归档，还应把三份原始日志转存到正式 artifact storage。

### 65.4 许可证决定了本轮只能做文档 checkpoint

最终来源审计确认：private package 中 7 个 primitive 保留 CANN Open Software
License Agreement v2 文件头并与只读 pypto-lib 基线存在大量共同代码；其文件头
又错误指向 Apache-2.0 的仓库根 `LICENSE`。`kernel.py` 虽标 Apache-2.0，但结构
也派生自历史 `decode_csa.py`，在没有重许可或完整多许可证方案前同样按待定处理。

因此当前实现可以继续留在本地工作树和上板验证，但不能静默作为“纯 Apache-2.0
功能提交”发布或 push。最安全的阶段性动作是只提交计划、过程记录、README 和两份
结果 JSON；production hook、private kernel package 与其测试保持未提交，直到以下
任一方案闭环：

1. 有权主体明确将相关 primitive/`kernel.py` 重许可为 Apache-2.0，并留存依据；
2. 项目维护者接受多许可证分发，同时补 CANN v2 协议全文、NOTICE/provenance、
   wheel/sdist packaging、文件头引用和项目 license metadata；
3. 对相关实现做可证明的独立重写并完成 provenance review。

本轮明确不 stage 用户已有的 Qwen3/worker 改动、根目录问题记录或
`pypto_qwen3_l1.py`，也不修改/提交只读 pypto-lib、PyPTO 或 simpler。

### 65.5 最终代码审查保留的两个 P2 边界

对 short-index gate、qk-valid-only、两个 predicate `spmd(1)` 和静态 ABI 测试做
只读最终审查，未发现 P0/P1 correctness blocker。依赖闭合点包括：九个 score-only
task 均直接依赖 gate；false predicate 的 retired `score_tid` 能释放 top-k fanout；
`qk_order` 容量为每 token 1 个 seed 加最多 4 个 optional block；padding seed 避免
merge 未初始化读；两个 blockDim=1 的 Tensor slice 都由恒为 0 的 block index 落在
合法范围；长上下文仍进入原 score/sort 路径。

仍保留两个 P2 边界：

1. HBG predicate 功能依赖当前匹配 simpler 在 `recording.unsupported` 后安全重新走
   ordinary orchestration 的语义。当前版本和 device0 四 bucket 已验证，但尚无公开
   capability handshake；若部署换成不保证该 fallback 的 runtime，backend 应在
   prepare 时 fail-fast，而不能默认为 supported。当前结果不得称为 resident HBG。
2. callable 是 vLLM 内部 custom-op，device metadata 被视为受信输入。现有 builder
   保证 padding start 为 0、active position 落在 RoPE 表内、可见 block id 落在对应
   physical cache capacity；adapter 为避免 NPU `.item()`/同步只验证 shape、dtype、
   stride、device，不在热路径把 device 值拉回 Host。若未来把 callable 开放为泛用
   public API，必须增加 device-side position/block-id 上下界 guard；不能沿用“调用者
   一定正确”的内部合同。

静态测试当前以源码/合同断言为主，尚未生成并解析每个 bucket 的完整 IR/DAG 来验证
所有 dependency。这是后续回归增强项，不改变本轮已有的 TRB/HBG 真机正证据。
