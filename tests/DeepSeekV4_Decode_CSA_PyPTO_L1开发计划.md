# DeepSeek V4 Flash Decode CSA 以 PyPTO L1 接入 vLLM-Ascend 的开发计划

## 0. 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档性质 | 开发计划、测试设计、执行状态与验收依据 |
| 目标硬件 | Ascend A3，仅覆盖 A2/A3 runtime 路径；本阶段不覆盖 A5/A5 simulator |
| 可用环境 | 单机双卡 A3；主验证采用单卡 TP1，另一张卡仅用于并行生成数据或独立调试 |
| 目标场景 | DeepSeek V4 Flash 的 decode CSA，不部署完整模型，不验证 prefill/MoE/sampler |
| PyPTO 接口 | L1 单算子模式，使用 caller stream，支持 TRB 与 HBG，能够被 ACLGraph 透明捕获 |
| vLLM 接口 | 保持 `torch.ops.vllm.dsa_forward` 的 custom-op 边界 |
| 正式实现位置 | kernel、静态 L1 entry、ABI contract 与 backend adapter 统一位于 `vllm_ascend/ops/_pypto_dsv4_csa/` |
| 历史参考位置 | `/mnt/workspace/inductor/pypto-lib/models/deepseek_v4_flash_dspark/` 仅作为只读算法和 fixture 参考，本任务不修改 pypto-lib |
| 测试实现位置 | `tests/pypto_dsv4_decode_csa/`，Host UT 与 A3 ST 从该目录的公共 fixture/harness 派生 |
| 默认 workload | DSpark target verify，`S=8`（1 个 target token + 7 个 draft token） |
| 静态 bucket | `B4/B8/B12/B16`，对应 `T=B*S=32/64/96/128`；每个 bucket 是独立 L1 specialization |
| 默认子图 | 4 个真实尺寸、独立权重和独立 cache 的 CSA layer |
| 文档状态 | 主体实现与 A3 单算子闭环完成；B4/S8/C8191 TRB ACLGraph 稳态性能通过；发布提交等待许可证处理 |

本文档中的“native”是指当前 vLLM-Ascend 的 DeepSeek V4 DSA/CSA 实现；“PyPTO”是指新的 PyPTO L1 backend；“CSA Capsule”是本文设计的、有状态但不包含完整 vLLM Engine 的 decode CSA 子图运行器。

## 1. 结论与核心决策

本任务不以减少层数的完整 DeepSeek V4 模型作为首要验证载体。直接把官方模型的 `num_hidden_layers` 改成 1～4，仍会引入 MoE、embedding、LM head、模型加载、checkpoint key、MTP 和 scheduler 等与 CSA 无关的显存和故障面，而且不能提高 CSA 对比的可信度。

本任务采用三级证据中的前两级，并显著加强第二级：

1. 单个 `torch.ops.vllm.dsa_forward` 原位接入测试，用于证明 production custom-op 边界、L1 caller-stream 语义、ACLGraph capture/replay、精度与单算子性能。
2. 一个默认包含 4 个 CSA layer 的有状态 Decode CSA Capsule，用于模拟连续 decode、batch bucket、paged cache 分配与回收、不同 layer/callable、多 custom-op 节点、torch 算子交错及多 ACLGraph 生命周期。
3. 完整 vLLM Engine 或缩减模型不属于本阶段必交付项。只有未来需要验证 request scheduler、模型加载器、分布式 TP/EP、sampler 或外部服务 API 时才追加。

强化后的第二级不是“静态调用两个 attention layer”，而是一个尽量复用 vLLM 数据结构和 metadata 生成规则的 stateful decode subsystem：

```text
Synthetic Decode Schedule
  request admit / step / retire / block reuse / bucket switch
                              |
                              v
                    Decode Trace Materializer
                              |
              +---------------+----------------+
              |                                |
              v                                v
       Native State World                PyPTO State World
       independent caches                independent caches
              |                                |
              v                                v
       4-layer CSA Capsule               4-layer CSA Capsule
              |                                |
              +---------------+----------------+
                              |
                              v
             per-step output + all mutable-state comparison
```

以下决策是本阶段的硬约束：

- 只做 decode CSA；不为得到“模型能出 token”的表面结果引入完整网络。
- 生产替换边界保持为 `dsa_forward`，不把当前包含 HC 的 PyPTO program 原封不动塞入该边界。
- 以 pypto-lib 当前 `attention_csa` 为只读算法参考，在 `vllm_ascend/ops/_pypto_dsv4_csa/` 内维护与 vLLM `dsa_forward` 对齐的正式 `decode_csa_core`；本任务不向 pypto-lib 写入 kernel、adapter、测试或兼容补丁。
- 正式 kernel 与正式 adapter 必须同属 `vllm_ascend/ops/_pypto_dsv4_csa/`，不把生产 kernel 留在测试目录，也不再建立第二份可独立演进的 production kernel。
- L1 首版只接受静态正整数 shape；固定支持 `B4/B8/B12/B16`、`S=8` 四组 specialization，不把 pypto-lib 的 `B_DYN/T_DYN` 动态入口直接注册为 L1 callable。
- HC-pre、输入 RMSNorm 和 HC-post 作为真实 surrounding torch/NPU op 放在 Capsule 层壳中，而不是重复融合进 `dsa_forward`。
- 六类被写入的 cache/state 都必须在 PyPTO 编译签名、adapter 所有权和测试 oracle 中显式建模为 `InOut`：SWA KV、compressed KV、main compressor state、inner/indexer compressor state、indexer K、indexer scale。
- A3 indexer scale cache 的正式 ABI 是 FP16 storage；kernel 内部可以转成 FP32 计算，但写回必须恢复为 FP16，不能用 FP32 mirror 冒充生产 ABI。
- 正式实现必须直接支持 vLLM A3 的 page-strided cache view，所有 cache 寻址都以 block、offset 和实际 stride/page stride 为准；禁止用逻辑 `reshape/flatten` 跨越 page padding。
- 首先用 TRB（`tensormap_and_ringbuffer`）跑通功能和错误闭环，再启用 HBG（`host_build_graph`）做同矩阵验证与性能比较。
- TRB 与 HBG 的正式测试矩阵必须分别在全新进程中执行；不把同进程 runtime 切换作为本阶段支持项或验收前提。
- production-style 测试必须使用 torch_npu taskQueue adapter；不允许 taskQueue 初始化失败后静默退回 Python raw-stream 路径。
- capture、replay 和 L1 launch 内不允许编译、lazy prepare、device allocation、H2D staging 或 synchronize。
- 当前 PyPTO 独占全部 AICore，同一 device 上只允许串行执行；本计划不把多流并发作为支持能力。
- 性能结论必须在同一张卡、同一 workload、同一状态轨迹下交替测量；不能用两张卡各跑一套后直接比较。

## 2. 背景与当前实现边界

## 2.1 vLLM-Ascend 的现有 custom-op 边界

当前 DeepSeek V4 sparse attention wrapper 位于 `vllm_ascend/ops/dsa.py`。其 `forward()`：

1. 接收已经进入 attention 子层的 `hidden_states`；
2. 由 PyTorch wrapper 分配 `output`；
3. 调用 `torch.ops.vllm.dsa_forward(hidden_states, need_gather_q_kv, output, layer_name)`；
4. custom-op 从 `ForwardContext` 中取得 layer object、attention metadata 和 KV cache；
5. native backend 完成 Q/KV prolog、compressor、indexer、sparse attention 和 output projection。

相关代码：

- [`vllm_ascend/ops/dsa.py`](../vllm_ascend/ops/dsa.py)
- [`vllm_ascend/attention/dsa_v1.py`](../vllm_ascend/attention/dsa_v1.py)
- [`vllm_ascend/device/device_op.py`](../vllm_ascend/device/device_op.py)

`dsa_forward` 通过 `PrivateUse1` 注册，是当前为 ACLGraph 保持的正式算子边界。首版接入不新造另一个模型级 public op，也不把 cache tensor 全部暴露到 Python 用户接口；测试 harness 可以直接访问这些 hidden mutable states 做验数。

## 2.2 Decoder layer 中的真实上下游

DeepSeek V4 attention 半层的实际顺序位于 `vllm_ascend/models/deepseek_v4.py`：

```text
residual clone
  -> hc_pre
  -> input_layernorm
  -> self_attn / torch.ops.vllm.dsa_forward
  -> hc_post
```

随后才进入 FFN/MoE 半层。本任务只替换其中的 decode CSA custom-op，但 Capsule 中保留真实 HC-pre、RMSNorm 和 HC-post，以验证 PyPTO 节点与其前后 torch/NPU task 的 stream 顺序与数值连接。

相关代码：

- [`vllm_ascend/models/deepseek_v4.py`](../vllm_ascend/models/deepseek_v4.py)

## 2.3 pypto-lib 历史 program：只读参考边界

历史 program 位于工作区：

```text
/mnt/workspace/inductor/pypto-lib/models/deepseek_v4_flash_dspark/decode_csa.py
```

当前 `attention_csa_test` 已经覆盖：

- HC-pre；
- attention RMSNorm；
- Q/KV 和 RoPE；
- ratio-4 main compressor；
- inner compressor；
- indexer；
- sparse attention；
- grouped output projection；
- HC-post。

其边界比 vLLM `dsa_forward` 更宽，因此不能直接做一对一替换。现有签名还只有主 `kv_cache` 被显式声明为 `pl.InOut`，但 `compress_state`、`inner_compress_state`、`cmp_kv`、`idx_kv_cache` 和 `idx_kv_scale` 同样存在执行期写入。正式实现必须在 vLLM-Ascend 自己的 kernel contract 中重新核对所有 side effect，并补齐方向元数据。

本任务对该目录采用严格只读原则：

- 可以阅读其算法拆分、参数维度、调度方式和 standalone fixture；
- 可以用其输出生成离线或运行时 golden；
- 不在 pypto-lib 中提取或提交新的 L1 entry；
- 不修改 pypto-lib 的 dtype、layout、cache ABI 或 `InOut` 声明来迁就本任务；
- 正式实现的唯一源码所有权位于 `vllm_ascend/ops/_pypto_dsv4_csa/`；如果参考实现与真实 vLLM A3 ABI 冲突，以 vLLM-Ascend 的真实 ABI 为准，并用对照测试记录差异。

现有 Flash 配置提供本阶段的真实维度基线：

| 参数 | 值 |
| --- | ---: |
| hidden size | 4096 |
| attention heads | 64 |
| head dim | 512 |
| RoPE head dim | 64 |
| Q LoRA rank | 1024 |
| O LoRA rank | 1024 |
| O groups | 8 |
| sliding window | 128 |
| compressor ratio | 4 |
| index heads | 64 |
| index head dim | 128 |
| index top-k | 512 |
| paged cache block size | 32 |
| DSpark verify width `S` | 8 |

现有 standalone fixture 的请求轴支持 4 的倍数，首轮使用 `B=4`，随后扩到 `B=8/12/16`。

历史入口中的 `B_DYN/T_DYN` 只能继续用于参考或 L2 golden。PyPTO L1 当前要求 tensor annotation 的每一维都是正整数，因此正式 L1 entry 必须按 `B4/B8/B12/B16` 分别生成静态 callable；动态入口不能作为任一 bucket 的兜底路径。

## 2.4 PyPTO L1 的既有契约

PyPTO L1 的完整设计与实现记录位于：

```text
/mnt/workspace/inductor/pto/pypto/tests/PyPTO_L1与ACLGraph完整设计文档.md
/mnt/workspace/inductor/pto/pypto/tests/PyPTO_L1与ACLGraph实现过程记录.md
```

本计划必须遵守以下既有契约：

- AICPU task 使用 caller stream；内部 AICore stream 不对外暴露。
- launch 不同步 caller stream，也不调用 device synchronize。
- launch 不创建 workspace、Runtime、KernelArgs、event、binary handle 或 HBG working slot。
- workspace 仍由 PyPTO context 内部准备并持有。
- taskQueue 路径使用 `.stream(false)`、`RunOpApiV2`、C++ Tensor lease 和 allocator `recordStream`。
- TRB/HBG 的 task 参数必须形成独立的 runtime-owned snapshot，不能被下一次 Host 调用覆盖。
- HBG graph package 是 tiling-like task 参数；immutable source、CANN-owned snapshot 和 context-owned mutable execution slot 必须分离。
- HBG resident registry 是 Context-owned，不允许退回 DSO-global 可变 registry。
- L1 路径不调用 `aclrtBinaryUnLoad`/`rtsBinaryUnload`，图可见 binary/code handle 按进程 pin。
- PyPTO 不查询 capture 状态，不获取 graph handle，不调用 `rtStreamAddToModel`。
- 当前同一 context/device 不支持并发执行；可检测到的冲突 fail-fast。

## 3. 目标、非目标与成功定义

## 3.1 功能目标

1. PyPTO `decode_csa_core` 能从 vLLM `dsa_forward` 获得等价输入、metadata、权重和 cache，并在 caller stream 上作为一个 L1 op 入队。
2. native/PyPTO backend 可在测试中明确切换；切换不能改变 custom-op 的外部 schema。
3. eager 与 ACLGraph capture/replay 都能执行同一套 decode step。
4. 同一编译产物可以服务支持范围内的不同 tensor 地址；普通 eager 地址变化不得重新编译结构图。
5. ACLGraph 使用每个 bucket 的固定 graph buffer，replay 时只更新 buffer 内容和外部支持的 runtime state，不在 PyPTO 内感知 capture/replay。
6. 默认 4 层 Capsule 支持全 native、全 PyPTO 和 mixed backend 拓扑。
7. 连续 decode trace 支持 request 加入、推进、结束、cache block 回收和重新分配。
8. 所有被修改的 cache/state 都能在 native/PyPTO 两个独立 world 之间逐步比较。
9. TRB 与 HBG 都完成 eager、ACLGraph、连续 replay 和多 layer 验证。
10. 形成可重复运行的正确性与性能命令、结果目录和 profiler 解析规则。

## 3.2 正确性目标

- 单步输出误差可解释并满足预先定义的容差。
- 连续 4、32 和 128 步后误差不出现非预期发散。
- 每一种 mutable cache/state 的有效写入区域与 native 对齐。
- 未写入区域保持原值，不允许 padded/inactive request 污染 cache。
- cache block 回收并复用后，新 request 不读取前任 request 的残留状态。
- 四层之间不发生 layer-name、weight、func-id、cache 或 metadata 串扰。
- mixed backend 的最终输出与对应逐层 reference 一致。

## 3.3 稳态性能目标（不计首次编译和任何 warmup）

功能性开发全部闭环后，本计划继续进行尽力而为的性能优化，最终目标是在
**相同 workload、相同正确性门禁、同一张 A3、native production 默认优化保持开启**
的条件下，使 PyPTO 路径超过 native。第一轮先建立可复现 baseline，再依据
profiler 拆解：

- native custom-op device span；
- PyPTO TRB device span；
- PyPTO HBG device span；
- host enqueue latency；
- torch_npu taskQueue dequeue latency；
- ACLGraph replay latency；
- 四层 Capsule 稳态 latency；
- 地址命中与地址变化时的 Host cache 开销；
- graph bucket 切换成本。

性能优化不能通过越过单算子边界、提前启动私有 AICPU stream、内部 synchronize 或 capture 私有 API 达成。
也不能通过降低数值门禁、缩小实际 batch/context/layer workload、只挑最好一次或
关闭 native overlap 来制造“超过 native”的结论。不得把任一方稳态调用中因真实
地址/scalar 变化而反复发生的 patch 或 cache miss 移出自身口径；最终采样地址第一
次调用所产生的一次性 binding/structure-cache 填充则对双方一致作为冷启动处理。
若经过合理优化仍未超过，必须保留可复现 profiler 与未关闭瓶颈，不能把目标悄悄
降级为仅完成测量。

“超过 native”的正式胜负口径只比较冷启动结束后的常态化 steady state。下列
项目必须在正式 `SAMPLE` phase 之前全部完成，作为 cold-start/validation
数据单独记录，不进入胜负样本：

- 首次 program 编译和 PTOAS/codegen；
- binary 注册、materialize 和加载；
- runtime/context/owner 创建、prepare 和一次性 callable 准备；
- weight pack 和 preparation；
- ordinary eager、ACLGraph 及正式 benchmark 的全部 warmup；
- ACLGraph capture 和 graph build；
- 在最终采样地址上的第一次 ordinary invocation 或第一次 replay；
- 一次性 structure-cache/binding-cache 填充和 event pool/handle 初始化；
- correctness golden 生成、执行、比较以及 validation-only copy。

进入 `SAMPLE` 前必须由 caller 外部同步并 quiesce，确认上述任务均已结束；
不能只是 Host 代码已经返回。

正常服务调用路径中真实发生的参数校验、tensor 地址/scalar patch、taskQueue
enqueue/dequeue、AICPU/AICore device scheduler 和 kernel execution 均属于被测路径
的必要成本，必须进入主胜负口径；允许拆栏报告，但不允许从主指标中删除。固定
captured binding 的 ACLGraph replay 本来不发生 Python 侧地址/scalar patch，因此
不要求为它人为伪造一次 patch；这类结果也只能证明固定 binding 的稳态性能。
真实稳态 workload 若持续改变 tensor 地址或 scalar，并因此反复发生 patch、
binding-cache 替换或 cache miss，这些反复成本必须保留在对应样本中；只有最终
采样工作集的首次 binding/一次性结构 cache 建立可以归入冷启动。

固定 captured binding 的正式主指标继续使用完整 replay 外围的同 caller-stream
`device_span`。如果待验收调用真实存在 start event 入队之前的 Host validation、
地址/scalar patch 或 cache lookup，则不得把 `host_enqueue` 与 `device_span` 相加（两者
可能重叠）；该场景改用稳态 batch critical-path 作为主指标：从 batch 第一个调用的
backend-specific per-call 工作开始计时，到最后一个调用在 caller stream 上 quiesce
结束，扣除双方完全等量的 fixture 内容更新后除以调用数。同时保留 `host_enqueue`
和 `device_span` 作为归因子指标。这样既覆盖 Host patch，也不会重复计算 Host/device
重叠时间。

正式达标必须对每个准备宣称达标的目标 `(runtime, mode, workload)` 使用至少
**3 个相互独立的 fresh Python process** 重复实验。每个 process 内 native 与
PyPTO 必须使用同一张 A3、同一 caller stream、同一 ABBA 交替方案、相同输入和
等价初始 state、相同数值门禁；native 必须保持 production multi-stream overlap
等默认优化。每个 backend 每个 process 的冷启动阶段必须包含至少 20 次 warmup；
该阶段全部完成并由 caller quiesce 后，随后采集至少 100 个 `SAMPLE` 样本。

对每个 fresh-process 重复 `r` 和分位点 `q ∈ {50, 90, 99}`，定义：

```text
D[q, r] = native_latency[q, r] - pypto_latency[q, r]
```

正值表示 PyPTO 更快。只有同时满足
`median_r(D[50, r]) > 0`、`median_r(D[90, r]) >= 0` 和
`median_r(D[99, r]) >= 0`，才能在该目标配置上宣称稳态性能超过 native。
所有 process 都必须通过采样前后正确性门禁并逐轮保留原始样本；不得把多个
process 的样本合并后只挑有利分位数。TRB ACLGraph 是本轮稳态性能主验收路径；
HBG 在 resident graph 对 predicate/动态调度具备完整表达前，必须保功能和单列
性能，但其通用 graph restore 架构开销不否定已经按上述门槛成立的 TRB 单算子
稳态结论；任何单独的 HBG“超过 native”声明仍必须满足同一套门槛。

## 3.4 非目标

- 不运行完整 DeepSeek V4 权重。
- 不验证 prefill。
- 不实现或验证 MoE、shared expert、hash layer、embedding、LM head、sampler、tokenizer。
- 不验证 vLLM request scheduler、HTTP server、模型加载器或 checkpoint loader。
- 不做生成文本级精度或困惑度评估。
- 不把 Capsule latency 外推成完整模型 tokens/s。
- 首版不做 TP2/HCCL、EP、DP、DCP 或跨卡通信。
- 不支持 A5，也不运行 A5 simulator。
- 不声称支持同一 device 上的并发 graph replay。
- 不在本阶段设计 PyPTO workspace 外置或完整并发 slot 化。

## 4. 总体架构

## 4.1 测试系统分层

```text
+-----------------------------------------------------------------------+
| Test Scenario                                                         |
| fixed case / continuous trace / graph lifecycle / performance profile |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| DecodeTrace                                                           |
| request ids, positions, seq_lens, actual/padded rows, block ownership  |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| MetadataMaterializer                                                  |
| vLLM common metadata -> DSA metadata -> PyPTO tensor ABI adapter       |
+----------------------+-----------------------------+------------------+
                       |                             |
                       v                             v
+----------------------------------+  +----------------------------------+
| NativeStateWorld                 |  | PyPTOStateWorld                  |
| weights + six cache families     |  | cloned weights + cache families  |
| native layer modules             |  | PyPTO L1 callable/session         |
+----------------------+-----------+  +------------------+---------------+
                       |                                 |
                       v                                 v
+----------------------------------+  +----------------------------------+
| Native 4-layer CSA Capsule       |  | PyPTO 4-layer CSA Capsule         |
| actual HC/norm + dsa_forward     |  | actual HC/norm + dsa_forward     |
+----------------------+-----------+  +------------------+---------------+
                       |                                 |
                       +----------------+----------------+
                                        |
                                        v
+-----------------------------------------------------------------------+
| Comparator / Invariant Checker / Profiler Reporter                    |
+-----------------------------------------------------------------------+
```

## 4.2 核心组件

| 组件 | 职责 | 明确不负责 |
| --- | --- | --- |
| `DecodeScenario` | 声明 batch、S、起始位置、步数和 request churn | 直接创建 NPU tensor |
| `DecodeTrace` | 表达每一步 request 与 block ownership 的不可变记录 | 执行算子 |
| `CacheBlockAllocator` | 在测试域内分配、释放、复用逻辑 block | 模拟完整 vLLM scheduler |
| `MetadataMaterializer` | 尽量复用 vLLM builder 生成真实 metadata | 修改 PyPTO runtime |
| `PyPTOAbiAdapter` | 将 vLLM tensor/layout 转为 `decode_csa_core` ABI | 每次调用转置/复制静态权重 |
| `LayerState` | 持有单 layer 的权重、cache、native/PyPTO callable | 跨 layer 共享 mutable cache |
| `CSALayerCapsule` | 运行 HC-pre、norm、CSA、HC-post | 运行完整 MoE |
| `TorchBridge` | 在层间制造真实 torch task 和数据依赖 | 冒充 DeepSeek V4 FFN 精度 |
| `CSACapsuleRunner` | 执行 1/2/4 层和 backend 拓扑 | 启动 vLLM Engine |
| `ACLGraphBucketManager` | prepare、warmup、capture、replay、destroy | 在 replay 时 lazy compile |
| `StateComparator` | 输出、cache、未写区、block ownership 验证 | 隐式同步性能计时区间 |
| `ProfileReporter` | 输出统一 JSON/Markdown/CSV 指标 | 把不同卡结果直接相减 |

## 4.3 独立的 native/PyPTO state world

同一测试 case 必须创建两个完全独立的 mutable world：

```text
immutable initial weights --------------------+
                                               +--> native weights/view
                                               +--> PyPTO packed weights/view

immutable initial cache image ----------------+
                                               +--> native cache world
                                               +--> PyPTO cache world
```

两个 backend 不能交替修改同一份 cache 后再比较。静态权重可以共享同一原始 Host fixture，但其 NPU storage、预打包 layout 或 quantization view 必须满足各 backend 的生命周期契约。

## 4.4 正式代码与测试代码的所有权

本任务采用单一正式实现源，不再把“先放测试包、验证后再搬生产目录”作为默认方案：

```text
vllm_ascend/ops/_pypto_dsv4_csa/
  kernel/core/static entry       正式 PyPTO CSA kernel
  contract/config               正式 A3 ABI 与 specialization contract
  backend/adapter               正式 dsa_forward backend、owner 与参数适配

tests/pypto_dsv4_decode_csa/
  fixture/trace/graph/compare   测试数据、state world、ACLGraph owner 和 oracle

pypto-lib/models/deepseek_v4_flash_dspark/
  historical reference         只读参考，不是本任务产物的运行时依赖或修改目标
```

具体原则：

- `vllm_ascend/ops/_pypto_dsv4_csa/` 是 kernel 与 adapter 的唯一权威实现；内部可继续拆成多个小文件，但不能在 tests 或 pypto-lib 再维护另一份正式 kernel。
- `tests/pypto_dsv4_decode_csa/` 可以持有 canonical fixture、reference runner、metadata materializer、graph owner、性能 runner 和诊断工具，但不得复制出另一份可独立演进的 kernel 算法。
- `dsa.py` 只保留现有 custom-op schema 和一个窄 backend 分派点；具体 PyPTO owner、static program registry、weight pack、ABI validation 和 launch 均进入 `_pypto_dsv4_csa/`。
- 对 pypto-lib 的任何观察都记录为“历史参考行为”；正式契约由本文件、vLLM A3 实际 cache tuple 和 `_pypto_dsv4_csa/contract.py` 共同约束。
- 正式代码必须能在没有把 pypto-lib 加入 `PYTHONPATH` 的环境中导入、编译和运行，避免形成未声明的跨仓源码依赖。

## 5. Production custom-op 接入设计

## 5.1 保持外部 schema

首版保持：

```python
torch.ops.vllm.dsa_forward(
    hidden_states,
    need_gather_q_kv,
    output,
    layer_name,
)
```

不在用户层新增 `ctx.operator(...).prepare().warmup()` 风格调用，也不要求模型 forward 显式传入 PyPTO context。PyPTO owner 应封装在 backend adapter 中，在 capture 外由测试 fixture 或未来模型初始化流程显式 warmup。

`output` 继续由 PyTorch wrapper 管理：

- 普通 eager 可以由 torch allocator 创建；
- direct custom-op harness 为保证地址和计时边界清晰，使用预分配输出；
- ACLGraph bucket 使用 capture 前已经建立的固定 graph buffers；
- PyPTO 只写入外部提供的输出地址，不为输出分配内存。

## 5.2 Backend 选择

测试代码必须支持显式 dependency injection：

```python
backend = NativeDSABackend(...)
backend = PyPTODSABackend(runtime="tensormap_and_ringbuffer", ...)
backend = PyPTODSABackend(runtime="host_build_graph", ...)
```

`NativeDSABackend` 可以继续复用现有实现；`PyPTODSABackend`、program registry、static entry factory、A3 ABI validation 和 weight pack 必须实现在 `vllm_ascend/ops/_pypto_dsv4_csa/`。测试可以 dependency-inject 该正式 backend，但不另造仅供测试的 backend 实现。

如果未来加入生产环境变量，必须在 `vllm_ascend/envs.py` 集中定义，不能在 hot path 中散落读取字符串。测试本身优先使用显式配置，避免进程全局环境污染不同参数化 case。

## 5.3 Adapter 生命周期

建议状态机：

```text
Created
  -> Compiled
  -> Prepared
  -> WarmedUp
  -> CaptureReady
  -> Active
  -> RetainedUntilProcessExit / ExplicitlyShutdown
```

约束如下：

- compile 与 prepare 必须发生在 ACLGraph capture 外。
- 每个 specialization 至少在普通 eager 下 warmup 一次，caller 外部同步后才能 capture。
- capture 内首次看见未准备 specialization 必须明确报错。
- production-style taskQueue 初始化失败必须报错，不能像现有实验 adapter 那样静默切换 `use_task_queue=False`。
- direct/raw-stream 模式只保留给明确标记的 bring-up 测试。
- Python GC 和 `atexit` 不调用 runtime close。
- 不显式 shutdown 时 owner 安静 pin 到进程结束。
- 显式 shutdown 前，测试负责同步、销毁全部相关 graph，并证明不存在未完成 task。
- TRB 与 HBG 各自拥有进程级 owner 生命周期；测试 runner 必须以 subprocess/全新 Python 进程切分两套 runtime，不在一个进程中先 shutdown TRB 再初始化 HBG。

## 5.4 Specialization key

缓存 key 至少包含：

- device type 与 device id；
- PyPTO runtime：TRB 或 HBG；
- program identity/content hash；
- batch bucket、`S`、token count；
- tensor dtype、shape、stride 与 layout kind；view 的 storage offset 作为调用期地址/越界校验信息记录，除非它改变 kernel 可见布局，否则不单独制造编译 specialization；
- 六类 cache 的实际物理 block 数、block size、page stride/row stride family；
- A3 indexer scale storage dtype（固定为 FP16）；
- quantization format 与 scale layout；
- 所有影响编译拓扑的 scalar bit pattern；
- AICore worker/core 配置；
- ABI version。

普通 eager 的 tensor 地址变化不能导致重新编译；HBG Host plan 可以对 per-call 地址做 patch，但不能重新生成与地址无关的结构模板。ACLGraph replay 使用 capture node 持有的地址和参数快照，本计划不自行实现 graph update API。

这里区分“编译 artifact key”和“首次 enqueue 后的 runtime binding”：PyPTO 编译缓存自身可能不把 torch stride 纳入源码级 cache key，但 adapter 必须在选择/复用 callable 前校验完整 stride/layout family；同一 context 中已经绑定某组 shape/dtype/stride 的 operator 不允许被另一组 stride family 复用。普通 storage address 和 view 起点变化由 per-call `data_ptr` 表达，不应仅因地址变化重新编译；若 storage offset 会改变 kernel 可见布局或越界证明，则由 contract validation 明确拒绝，而不是静默生成连续副本。

## 6. `decode_csa_core` 的提取与 ABI 对齐

## 6.1 边界定义

新的 PyPTO program 只覆盖：

```text
normalized hidden_states [T, D]
  -> Q/KV projection and normalization
  -> RoPE
  -> SWA KV scatter
  -> main ratio-4 compressor
  -> inner compressor
  -> indexer and top-k
  -> compressed sparse attention
  -> grouped output projection
  -> output [T, D]
```

以下逻辑不进入 `decode_csa_core`：

- HC-pre；
- input RMSNorm；
- HC-post；
- FFN/MoE 半层；
- sampler 或 token acceptance。

这样才能与 native `dsa_forward` 做 apples-to-apples 的输出、cache 和性能比较。

## 6.2 Tensor 方向

在最终签名前必须基于实际 kernel 写集合完成一次 side-effect audit。预期至少包含：

| 数据 | 预期方向 | 说明 |
| --- | --- | --- |
| normalized hidden states | In | 每步输入 |
| Q/KV/O weights and scales | In | prepare 后常驻，调用期间只读 |
| RoPE/cos/sin 或等价参数 | In | 由当前 position 选择 |
| metadata/slot mappings/block tables | In | 每个 decode step 更新内容 |
| SWA KV cache | InOut | 写入当前 token，读取窗口历史 |
| compressed KV cache | InOut | 写入压缩结果，读取历史 |
| main compressor state | InOut | 跨 decode step 持久状态 |
| inner/indexer compressor state | InOut | 跨 decode step 持久状态 |
| indexer K cache | InOut | INT8 cache |
| indexer scale cache | InOut | 与 indexer K cache 同步更新 |
| output | Out | caller 分配 |

任何实际写入但仍声明为普通 `pl.Tensor` 的参数都必须修正。测试还要通过填充 canary 和写区分析验证编译方向声明没有漏项。

## 6.3 vLLM A3 cache tuple 映射

A3 的 `dsa_forward` 从 context 构造的逻辑 cache tuple 为：

```text
0: compressed KV cache
1: SWA KV cache
2: main compressor state cache
3: indexer/inner compressor state cache
4: indexer K cache
5: indexer scale cache
```

adapter 必须显式映射到 PyPTO 参数，不能依赖位置恰好相同。映射测试至少验证：

- cache tuple 长度和每一项是否允许为 `None`；
- compress ratio 为 4 时全部所需 cache 均存在；
- A3 slot mapping 是 `[block_id, offset]` 的二维形式；
- PyPTO 当前使用的 flattened physical slot id 转换无溢出、无负值误解释；
- block size 使用真实 metadata/config 值，不在 adapter 中硬编码多份 `32`；
- cache shape、stride、layout 与 compiler metadata 完全一致。

### A3 page-strided cache 的强制契约

正式 adapter 传入的是 vLLM A3 创建的真实 cache view，而不是为了迁就 PyPTO 临时制作的 contiguous mirror。该 view 可能由 `as_strided` 建立，逻辑 block 之间存在 page padding。典型 block-size 32 布局包括：

| Cache family | 逻辑视图示意 | 典型物理 page stride | 正式要求 |
| --- | --- | ---: | --- |
| main compressor state | `[blocks, 2, 2048]`, FP32 | 8192 elements | 按 block page stride 寻址，不把相邻 block 当作紧邻的 4096 elements |
| inner/indexer compressor state | `[blocks, 2, 512]`, FP32 | 1040 elements | 保留每个 page 尾部 padding |
| indexer K | `[blocks, 32, 1, 128]`, INT8 | 4160 elements | 按 `[block, offset, head, channel]` 计算地址 |
| indexer scale | `[blocks, 32, 1, 1]`, FP16 | 2080 elements | storage dtype 是 FP16；不能按 FP32 连续数组解释 |

这些数值用于锁定当前 A3 参考布局和构造测试，但实现不能把同一个常数散落写入多个 kernel；权威 stride 来自实际 tensor/contract，并作为 specialization validation 的一部分。若运行环境返回不同但受支持的 page stride，应生成或选择对应静态 layout specialization；若不受支持，应在 capture 外 fail-fast。

所有 cache kernel 都必须避免对 page-strided 三维 view 直接执行会跨 block 合并的逻辑 `reshape/flatten`。合法地址应由 `block_id`、block 内 `offset`、内层 index 和真实 stride 显式计算。Host adapter 也不得通过每次 launch 的 contiguous copy 来掩盖寻址错误；否则既破坏地址更新能力，也使正确性和性能结论偏离真实 vLLM ABI。

### A3 indexer scale 的强制 dtype

`idx_kv_scale` 的对外 storage ABI 固定为 FP16。允许 kernel 在寄存器/UB 中将已加载 scale 转换为 FP32 做计算，写回时必须显式转换为 FP16。native/PyPTO golden 同时比较反量化后的数值误差和 FP16 cache 的写入区域；不创建常驻 FP32 shadow cache，也不在 hot path 做 FP16/FP32 全量转换。

## 6.4 Metadata 映射原则

优先复用 vLLM-Ascend 的 metadata builder。只有 PyPTO ABI 需要不同表示时才在 adapter 尾部生成：

- flattened original slot mapping；
- compressed slot mapping；
- main/inner state slot mapping；
- indexer slot mapping；
- window SWA indices 和有效长度；
- per-request KV sequence lengths；
- main/inner/index block tables；
- position ids。

转换函数必须是纯函数或显式写入调用方提供的稳定 buffer。capture/replay hot path 中不能创建新 device tensor，也不能通过 `.item()` 把 device scalar 拉回 Host。

## 6.5 权重与 layout

PyPTO fixture 和 vLLM module 的权重存储方向、quantized layout、NZ/ND 表示不一定相同。实现必须分成两层：

1. 一次性 weight pack：模型/fixture 初始化时完成转置、contiguous、quant scale 重排和 format cast；
2. 每次 launch：只传稳定 device storage，不再转置、复制或重新量化。

正确性首轮可以使用从同一 canonical FP32/BF16 seed 生成的两套 backend layout。第二轮必须验证从 vLLM module 中读取的真实存储 layout 能被一次性 pack，并记录 pack 后的额外显存。pack 成本不计入单算子 latency，但必须单独报告初始化耗时和常驻内存。

## 6.6 静态 bucket L1 entry

PyPTO L1 当前不接受包含 `B_DYN/T_DYN` 或任一非正维度的 tensor signature。本任务不把动态 shape 入口作为首版能力，正式 kernel 必须提供按静态 spec 生成 callable 的 factory，概念接口如下：

```python
@dataclass(frozen=True)
class DecodeCSAStaticSpec:
    batch: int                       # 4 / 8 / 12 / 16
    seq: int                         # 首版固定 8
    ori_blocks: int
    cmp_blocks: int
    main_state_blocks: int
    inner_state_blocks: int
    indexer_blocks: int
    block_size: int                  # 当前 A3 为 32
    main_state_block_size: int
    inner_state_block_size: int
    indexer_scale_dtype: str         # 当前 A3 固定 fp16
    layout_version: int


make_decode_csa_l1_program(
    spec: DecodeCSAStaticSpec,
    runtime: Literal[
        "tensormap_and_ringbuffer",
        "host_build_graph",
    ],
) -> Callable
```

静态 entry 的强制规则：

- `tokens = batch * seq`，四个首版 specialization 分别为 `(B,T)=(4,32)/(8,64)/(12,96)/(16,128)`；
- 每一个 tensor annotation 都使用具体正整数，不调用 `bind_dynamic`，也不以最大 B16 entry 代替其余 bucket；
- cache 物理 block 数进入 spec，不能在 kernel 内固定成 synthetic fixture 的 `512/256/260` 等容量；
- 六类 mutable cache/state 全部声明为 `pl.InOut`，attention output 声明为 `pl.Out`，其余参数只读；
- entry 调用共享的 inline `decode_csa_core`，算法实现不因 runtime 或 bucket 复制四份；
- factory 可以按 immutable spec/runtime 做进程内缓存，但 tensor address 不进入 program identity；
- eager 可以使用 PyTorch allocator 提供的输出；ACLGraph capture 必须显式传入 capture 前分配的固定输出 buffer；
- 每个 `(device, runtime, spec, stride/layout family)` 在普通 eager 中成功 warmup 并由 caller 外部同步后，才允许进入 capture。

静态 B bucket 只表示 graph/callable 的 padded batch 容量。`num_reqs_actual`、inactive row sentinel 和 metadata 内容仍然按 step 更新，因此必须测试 `actual < bucket` 时不会写入任何 cache。

## 6.7 Artifact metadata 与加载前校验

每个正式编译 artifact 至少记录：

- source/content hash 与 program identity；
- runtime、device/SoC 和 ABI/layout version；
- `B/S/T` 与六类 cache 的物理 block 数；
- 所有 tensor 的 shape、dtype、direction 和 stride/layout family；
- indexer scale FP16 storage contract；
- quantization/scale layout、worker/core 配置和影响拓扑的 scalar specialization；
- PyPTO、simpler、pto-isa、vLLM-Ascend commit 及构建版本。

加载或首次调用前必须检查编译 metadata：所有维度为正、tensor/scalar 数量不超过 L1 限制、输出方向只有预期 `Out`、六类状态均为 `InOut`、runtime 与请求一致且没有意外 distributed/SDMA 路径。任何 shape/dtype/stride/page-layout 不一致都在 capture 外报错，不能在 capture 内重编译、重新 pack 或制作 mirror。

## 7. 强化版 Decode CSA Capsule

## 7.1 Layer 数量

runner 支持 `num_layers in {1, 2, 4}`：

- 1 层：最快 bring-up 和单 layer 精度定位；
- 2 层：验证两个独立 callable/cache；
- 4 层：默认验收与性能模式，验证重复调用、数值累积、registry 和多 node graph 生命周期。

不把 8 层设为默认，因为层数增加对状态语义覆盖的边际收益有限；若 4 层稳定后需要压力测试，可增加 8 层长稳测试，但不阻塞首版验收。

## 7.2 单层壳

每个 `CSALayerCapsule` 的 fidelity mode 使用：

```text
x_hc [T, HC_MULT, D]
  -> residual clone
  -> real hc_pre
  -> real RMSNorm
  -> dsa_forward backend
  -> real hc_post
  -> y_hc [T, HC_MULT, D]
```

HC 参数在 native/PyPTO 两个 world 中取相同初始值。CSA backend 之外的操作完全相同，从而：

- 保留真实输入分布转换；
- 验证 torch/NPU predecessor 和 successor 顺序；
- 防止只测裸 kernel 时遗漏 taskQueue/caller-stream 问题；
- 仍可把差异归因到 CSA backend。

## 7.3 层间 Torch bridge

由于本阶段不实例化完整 MoE，层间使用显式的 deterministic bridge。建议提供两种模式：

1. `identity_residual`：少量 `add`/`mul`/activation，用于最小 graph 和 stream 顺序测试；
2. `low_rank_ffn`：真实 NPU matmul + activation + residual，但使用小 bottleneck，制造更接近 decoder 的数据变换和 allocator 压力。

bridge 在 native/PyPTO world 中使用相同参数和完全相同实现。其结果只用于形成后续 layer 输入和整体子图性能，不声称等价于 DeepSeek V4 MoE。

## 7.4 Backend 拓扑矩阵

四层模式至少执行：

| 名称 | Layer 0 | Layer 1 | Layer 2 | Layer 3 | 主要目的 |
| --- | --- | --- | --- | --- | --- |
| `NNNN` | Native | Native | Native | Native | golden baseline |
| `PPPP_TRB` | PyPTO | PyPTO | PyPTO | PyPTO | TRB 完整替换 |
| `PPPP_HBG` | PyPTO | PyPTO | PyPTO | PyPTO | HBG 完整替换 |
| `NPNP` | Native | PyPTO | Native | PyPTO | torch/native 与 PyPTO 交错 |
| `PNPN` | PyPTO | Native | PyPTO | Native | PyPTO 后序 native 可见性 |
| `PP_same` | 同一个 callable 重复使用 | 同左 | 独立 callable | 独立 callable | 同 callable 多 node 与地址快照 |
| `PP_distinct` | 独立 callable | 独立 callable | 独立 callable | 独立 callable | func-id 与 registry 隔离 |

`PP_same` 仍应使用不同 layer weights/cache 地址，以验证同一结构模板下 per-call tensor 地址 patch；`PP_distinct` 可以通过不同静态属性或独立编译 identity 强制形成多个 callable。

## 8. Stateful decode trace

## 8.1 Trace 数据模型

建议定义不可变结构：

```python
@dataclass(frozen=True)
class RequestStep:
    request_id: int
    sequence_length_before: int
    token_positions: tuple[int, ...]
    logical_blocks: tuple[int, ...]
    active: bool


@dataclass(frozen=True)
class DecodeStep:
    step_id: int
    bucket_size: int
    num_reqs_actual: int
    requests: tuple[RequestStep, ...]
    retired_request_ids: tuple[int, ...]
    admitted_request_ids: tuple[int, ...]
```

Trace 先在 Host 上完整生成并校验，再 materialize 为两个 state world 的 device buffers。这样同一条 trace 可以稳定复现 native/PyPTO 差异，且性能测量不包含随机调度逻辑。

## 8.2 首轮 DSpark decode 语义

默认：

- 每个 active request 每步验证 `S=8` 个 token；
- `T = bucket_size * S`；
- 首版真实 request 数按 PyPTO 当前约束从 4 开始；
- ACLGraph bucket 为 `B=4/8/12/16`；
- `num_reqs_actual <= bucket_size`；
- inactive/padded row 的 slot mapping 必须使用明确 invalid sentinel，且所有写 kernel 都必须屏蔽。

普通非 speculative decode 的 `S=1` 是单独 specialization，不与首轮 `S=8` 混用。若后续要支持 `S=1`，必须重新核对 task topology、window indices、compressor state 更新次数和性能，不仅是把 tensor 第一维改小。

## 8.3 必测位置边界

Trace 至少覆盖：

- compressor 尚未形成完整 group 的位置；
- ratio-4 边界前后；
- sliding window `127/128/129`；
- paged cache block `31/32/33`；
- 2K 上下文；
- 8K 上下文；
- 接近当前 fixture 最大上下文但不越界的位置；
- 不同 request 拥有不同 sequence length 的混合 batch。

## 8.4 Request churn 场景

固定测试轨迹示例：

```text
step 0:  admit A, B, C, D                    -> B4
step 1:  advance A, B, C, D                  -> B4
step 2:  retire B; admit E using fresh block -> B4
step 3:  retire A; admit F, G, H, I           -> B8
step 4:  advance C..I                        -> B8
step 5:  retire C, D, E; return their blocks -> B8
step 6:  admit J, K; reuse returned blocks   -> B8
step 7:  compact active requests             -> B4 or B8
...
```

必须有专门场景强制新 request 复用已退休 request 的 physical block，并在分配前按 vLLM 预期执行初始化/覆盖。测试同时验证：

- block ownership 在任何一步都不重叠；
- block table 和 slot mapping 相互一致；
- inactive row 不拥有可写 slot；
- native/PyPTO 使用同一逻辑轨迹但不同 physical tensor storage；
- block 复用后的输出不依赖前任 request 的隐含残留。

## 9. ACLGraph 设计

## 9.1 Bucket owner

每个 backend/runtime/bucket 拥有独立 graph owner：

```text
GraphKey = (
    backend,
    runtime,
    bucket_size,
    S,
    cache_physical_block_counts,
    num_layers,
    topology,
    dtype_layout_signature,
    page_stride_signature,
)
```

Graph owner 持有：

- 固定输入、输出与 metadata buffers；
- graph capture stream 及外部同步责任；
- captured graph object；
- 所引用的 layer weights/cache/context owner；
- replay 计数与 profiler 标签。

PyPTO 不持有或查询 ACLGraph object。

## 9.2 标准生命周期

```text
construct layer/callable
  -> compile outside capture
  -> prepare outside capture
  -> ordinary eager warmup on caller stream
  -> caller external synchronize
  -> create/fill fixed graph buffers
  -> capture on capture stream
  -> external synchronize
  -> update buffer contents
  -> replay N times
  -> external synchronize for validation/timing boundary
  -> destroy/reset graph
  -> after all graphs/tasks are gone, optional PyPTO shutdown
```

以下行为在 capture/replay 期间直接视为失败：

- 首次编译或加载 binary；
- 首次注册 callable；
- workspace 或 HBG slot 扩容；
- tensor layout pack；
- Host 读取前序 device tensor 值来动态建 HBG topology；
- 内部 stream/device synchronize；
- taskQueue drain；
- 获取 capture model 并手工挂 hidden stream。

## 9.3 多 graph 生命周期矩阵

至少验证：

1. warmup stream 与 capture stream 不同；
2. `B4` capture/replay/destroy；
3. `B4` 和 `B8` 同时存活，外部严格串行交替 replay；
4. `B4 -> B8 -> B12 -> B4 -> B16 -> B8`；
5. destroy `B4` 后 `B8` 仍可 replay；
6. destroy 全部 graph 后重新 capture `B4`；
7. TRB 全部 case 在一个或一组 TRB 专用新进程执行，HBG 全部 case 在另一组 HBG 专用新进程执行；本阶段不执行也不验收同进程 runtime 切换；
8. shutdown 重复调用幂等，失败时 owner 保留并可重试。

“同时存活”不表示并发 replay。由于当前 context workspace/HBG working slot 只有一份，两张 graph 的 replay 必须由 caller 明确串行并在跨 stream 时满足 quiescence 契约。

测试 runner 必须把 runtime 作为进程启动参数，而不是同一 pytest 进程中的普通参数化值。子进程需要分别记录 artifact、日志和退出状态，防止前一个 runtime 的 process-pinned binary、context owner 或 DSO/static state 污染后一个 runtime 的结论。

## 9.4 Replay 时的输入更新

每个 step 把新内容写入固定 graph input buffers：

- hidden state；
- positions；
- sequence lengths；
- block tables；
- slot mappings；
- window indices/lens；
- `num_reqs_actual` 对应的 device-visible表示。

capture node 的 storage 地址不变。必要的数据 copy/update 作为 graph 前序 torch task 或由测试明确放在计时区间外，不能让 PyPTO launch 自己完成隐式 H2D。

## 10. 正确性验证计划

## 10.1 比较层次

1. PyPTO standalone golden：确认 `decode_csa_core` 从宽边界 program 中正确抽取。
2. 单 `dsa_forward` native vs PyPTO：确认 ABI adapter 和 custom-op 接入。
3. 单层 HC-surrounded Capsule：确认真实前后算子组合。
4. 四层全 native vs 全 PyPTO：确认累积结果和独立 layer state。
5. mixed backend：确认互操作和 caller-stream 顺序。
6. continuous trace：确认跨 step cache 生命周期。
7. ACLGraph replay：确认 capture snapshot、固定 buffers 和 HBG restore。

## 10.2 每步比较对象

每一步都比较：

- layer 0～3 的 CSA 输出或至少最终 layer 输出；
- Capsule 最终 `x_hc`；
- NaN/Inf 数量；
- cosine similarity；
- max absolute error；
- max relative error；
- 超过阈值元素比例。

debug/边界 step 额外比较：

- 每层 SWA KV 的当前写入位置；
- compressed KV 的新增位置；
- main compressor state；
- inner/indexer compressor state；
- indexer K cache；
- indexer scale cache；
- 未写区域 canary；
- 被回收/重新分配 block 的完整内容；
- 可选的 top-k index debug 输出。

## 10.3 初始容差

现有 standalone fixture 可作为初始参考，而不是最终不可调整规格：

- `x_out`：以 `4e-3` 级相对误差阈值和不超过约 `0.8%` 的超差比例开始；
- BF16 KV：以 `rtol=1/128`、小绝对容差开始；
- INT8 K cache：量化值应逐元素一致，若量化 tie/舍入路径不同则必须同时比较反量化值和 scale；
- FP32 compressor state：先要求严格或非常紧的容差，若不同归约顺序造成误差再基于证据放宽。

任何容差放宽都必须记录：首次分歧位置、对应 kernel、误差分布及其是否随 decode step 增长。禁止只提高全局 `rtol` 使测试通过。

## 10.4 不变量检查

- 所有 tensor 的 dtype/shape/stride/layout 与编译 metadata 一致。
- 所有输出和 mutable state 位于当前 device。
- layer i 只写 layer i 的 cache。
- request r 只写其 slot mapping 指向的 block/offset。
- padded slot 保持 canary。
- tensor Python 引用在 enqueue 后立即释放时，taskQueue Tensor lease 仍保证 storage 存活。
- eager 连续调用使用不同输入/输出地址时结果正确。
- 同一结构模板的地址变化不触发不必要的重新编译。
- capture 后 replay 不进入 Python backend builder 或 lazy prepare。

## 10.5 失败用例

至少覆盖：

- capture 中首次调用未 warmup specialization；
- taskQueue adapter 不可用；
- 错误 device/current device；
- 错误 dtype、shape、stride 或 output alias；
- 缺少任一 ratio-4 cache；
- 非法 slot mapping/block id；
- 超出 prepared HBG working-slot capacity；
- 尝试并发或未 quiesce 的跨 stream 调用；
- graph 尚存活时错误 shutdown 流程的测试侧保护；
- runtime/callable identity 不匹配；
- 同 layer name 重复注册冲突。

错误必须在能够 Host 校验时 fail-fast；部分 prepare 后失败必须保留可诊断 owner，不能假装初始化从未发生。

## 11. 性能验证计划

## 11.1 比较对象

| ID | 实现 | 模式 |
| --- | --- | --- |
| N-E | native vLLM CSA | eager，生产默认配置 |
| N-G | native vLLM CSA | ACLGraph |
| P-T-E | PyPTO L1 TRB | eager + taskQueue |
| P-T-G | PyPTO L1 TRB | ACLGraph + taskQueue |
| P-H-E | PyPTO L1 HBG | eager + taskQueue |
| P-H-G | PyPTO L1 HBG | ACLGraph + taskQueue |
| M-G | mixed native/PyPTO | 四层 ACLGraph |

如果 native DSA 默认开启多 stream overlap，报告两条 native baseline：

- native production overlap；
- native forced-serial。

PyPTO 当前独占全部 AICore，不能只与较弱的 serial baseline 比较后宣称生产端到端收益。生产 overlap 与 PyPTO 的差异必须保留在报告中。

## 11.2 Workload 矩阵

首轮最小矩阵：

| Batch bucket | `S` | Context positions | Layers |
| ---: | ---: | --- | ---: |
| 4 | 8 | 128、2K、8K | 1、4 |
| 8 | 8 | 128、2K、8K | 1、4 |
| 12 | 8 | 128、2K、8K | 1、4 |
| 16 | 8 | 128、2K、8K | 1、4 |

正确性稳定后追加：

- mixed sequence lengths；
- `num_reqs_actual < bucket_size`；
- request churn trace；
- B4/B8/B12/B16 graph 交替 replay；
- 相同 callable 与不同 callable 的四层拓扑。

## 11.3 计时规则

- 冷启动排除项与稳态必计项严格遵循 §3.3 的完整定义，不允许用本节的测量细分
  改变正式胜负口径。
- 首次 program compile、PTOAS/codegen、binary 注册/materialize、
  runtime/context/owner 创建和 prepare、weight pack、全部 ordinary/ACLGraph
  warmup、ACLGraph capture、最终采样地址的第一次 ordinary invocation/replay、
  一次性 structure-cache/binding-cache 填充、event pool/handle 初始化和
  correctness golden 生成/执行/比较不计入 steady-state latency；必须完成
  全部项目并由 caller 外部同步、quiesce 后才进入 `SAMPLE`。
- 每个 backend 在每个 fresh process 中至少执行 20 次 warmup replay，再采样至少
  100 次；推荐范围为 20～50 次 warmup、100～1000 次 sample。
- 报告 median、p90、p99、min 和标准差，不只报告最好一次。
- host enqueue 用 Host 高精度时钟测量，不在每次迭代后同步。
- device span 使用 profiler/event 批量测量，避免逐次 event sync 改变队列行为。
- cache clone、golden 计算、比较和 trace materialize 不计入算子时间。
- 只有 backend-independent、双方完全等量且位于相同边界的 fixture/input content
  更新，才可提供诊断性的“计入”和“不计入”两栏；任何 PyPTO 特有的 metadata、
  地址/scalar 准备、参数校验、tensor 地址/scalar patch、taskQueue enqueue/dequeue、
  device scheduler 与 kernel execution 必须计入正式主口径，不能因另列子项而扣除。
- 正式比较使用至少 3 个 fresh process；每个 process 内两个 backend 在同一
  device、同一 caller stream 上按 ABBA 顺序运行，并使用同一 workload、输入、
  等价初始 state 和正确性门禁，降低温度/频率漂移影响。native 保持
  production overlap 开启。
- 每个 ABBA leg 必须从等价 state snapshot 开始，或双方推进同一个 trace step；地址
  churn 实验还必须固定 unique address/scalar 序列、working-set 和 cache capacity，
  循环容量驱逐后稳态反复出现的 miss 必须计时，不能先预热有限全集来隐藏它。
- 第二张 A3 可以同时用于生成 fixture 或独立调试，但最终 native/PyPTO 对比必须来自同一张卡。
- 每个 process 分别计算 `D[q,r] = native_latency[q,r] - pypto_latency[q,r]`；
  正式达标需同时满足 `median(D50) > 0`、`median(D90) >= 0` 和
  `median(D99) >= 0`，详细定义见 §3.3。

## 11.4 Profiling 检查点

Profiler 必须回答：

- `dsa_forward` 外部是否表现为一个普通 custom-op；
- caller stream 上 predecessor、AICPU task、join、successor 顺序是否正确；
- 是否存在内部 `streamSynchronize/deviceSynchronize`；
- taskQueue dequeue 是否仍有异常长尾；
- HBG 是否避免每次重新构造与地址无关的结构模板；
- tensor 地址或 scalar 变化时是否只做 per-call patch；
- 四层中是否发生不必要的 binary registration、H2D 或 allocator 调用；
- PyPTO 内部 AICore 利用率、尾部空洞和 scheduler 开销；
- native multi-stream overlap 被替换后损失了多少可重叠时间。

## 11.5 结果输出

每次正式实验保存：

- git commit/hash：vLLM、vLLM-Ascend、PyPTO、simpler、pypto-lib；
- CANN、torch、torch_npu、Python 版本；
- device id、SoC、频率/功耗信息；
- backend/runtime/specialization key；
- workload 与 trace seed；
- 正确性摘要 JSON；
- latency CSV/JSON；
- profiler trace；
- 一份自动生成的 Markdown 汇总。

性能报告不能只保留 profiler 截图。

## 12. 分阶段开发计划

## Phase 0：环境与最小 L1 基线

### 工作项

1. 固定本次使用的 vLLM、vLLM-Ascend、PyPTO、simpler 和 pypto-lib commit。
2. 在 vLLM-Ascend Python 环境中重新构建当前 PyPTO/simpler extension，消除 source hash 不一致。
3. 使用满足 ABI 的 GCC runtime `libstdc++`，避免加载系统旧版本。
4. 在 A3 上运行 PyPTO 最小 L1 eager 与 ACLGraph smoke test。
5. 验证 taskQueue adapter 开启且没有 silent fallback。
6. 记录设备空闲状态与基线环境。

### 完成标准

- 当前源码对应的 PyPTO extension 可以从 vLLM-Ascend venv 导入。
- 一个简单 PyPTO L1 op 在 caller stream 上通过 eager 和 ACLGraph replay。
- profiler 中没有 PyPTO 内部同步。

## Phase 1：在正式目录建立 `decode_csa_core` 与静态 L1 entry

### 工作项

1. 只读审计 pypto-lib `attention_csa` 的数据流、算法拆分和 side effect，把审计结论写入 vLLM-Ascend 测试记录，不修改参考仓。
2. 在 `vllm_ascend/ops/_pypto_dsv4_csa/` 中建立正式 `decode_csa_core`，把 HC-pre、attention norm 和 HC-post 留在 program 边界外。
3. 将 SWA KV、compressed KV、main state、inner state、indexer K 和 indexer scale 六类 mutable cache/state 全部标记为 `InOut`。
4. 定义 `DecodeCSAStaticSpec` 和 `B4/B8/B12/B16` 静态 L1 entry factory，所有 annotation 使用具体正整数，并让物理 cache block 数进入 spec。
5. 将 A3 indexer scale storage ABI 固定为 FP16，kernel 内按需升精度计算、FP16 写回。
6. 将 compressor/indexer/cache 访问改为真实 page-strided 寻址，删除任何会跨 page padding 的逻辑 flatten/reshape 假设。
7. 在 `tests/pypto_dsv4_decode_csa/` 为 core 新建 standalone fixture，重用相同 canonical 数据，同时覆盖 contiguous 诊断 fixture 和真实 A3 page-strided fixture。
8. 比较历史宽边界 reference 与“外部 HC/norm + 新 core”的结果和六类 cache；参考结果不要求 pypto-lib 成为正式实现的运行时依赖。
9. 分别在新进程编译/运行 TRB 与 HBG artifact，核对 HBG requirements 中不存在非法 Host device-tensor data access。
10. 校验 artifact metadata 中的静态 shape、方向、FP16 scale、cache capacity、stride/page layout、runtime 和版本字段。

### 完成标准

- B4、S8、关键 position 下，新旧 program 输出和六类 mutable state 一致。
- core 的外部输入输出与 vLLM `dsa_forward` 语义对齐。
- side-effect audit 有明确表格和测试，不再只有主 KV cache 被验收。
- B4/B8/B12/B16 四个 static spec 均可编译，metadata 中没有动态或非正维度。
- 真实 A3 page-strided cache 与 FP16 indexer scale fixture 通过；contiguous fixture 不能替代该验收项。

## Phase 2：vLLM ABI adapter 与单 custom-op harness

### 工作项

1. 构造真实或最接近真实的 `AscendDeepseekSparseAttention` layer object。
2. 构造 `ForwardContext.no_compile_layers`、attention metadata 和 A3 六 cache tuple。
3. 在 `vllm_ascend/ops/_pypto_dsv4_csa/` 实现一次性 weight pack、metadata/slot mapping adapter、static program registry 和 owner。
4. 在 `dsa_forward` backend 内调用同目录的正式 PyPTO L1 op；测试不得换用 tests 内的 kernel/backend 副本。
5. 保持 custom-op schema 与 output mutation 契约不变。
6. 建立 native/PyPTO 双 state world。
7. 完成 eager 单步、连续多步和地址变化测试。
8. 完成单 op ACLGraph capture/replay。

### 完成标准

- 同一输入/初始 state 下，native 与 PyPTO 输出和六类 state 满足容差。
- warmup stream 与 capture stream 不同时仍能 replay。
- capture 中没有 lazy compile/prepare/allocation/sync。
- taskQueue 不可用时 production-style case 明确失败。
- vLLM `_build_kv_cache()` 返回的真实六 tuple 可以不经 contiguous/FP32 mirror 直接通过 ABI validation 并执行。

## Phase 3：四层 Decode CSA Capsule

### 工作项

1. 实现可配置 1/2/4 layer 的 `CSALayerCapsule`。
2. 接入真实 HC-pre、RMSNorm 和 HC-post。
3. 实现 deterministic `identity_residual` 与 `low_rank_ffn` bridge。
4. 为每层创建独立权重、cache、layer name 和 backend owner。
5. 实现 `NNNN`、`PPPP`、`NPNP`、`PNPN`、`PP_same`、`PP_distinct`。
6. 为每一层和最终输出增加可选 debug capture 点。
7. 验证相同 callable 多地址 patch 与不同 callable func-id 隔离。

### 完成标准

- 四层 eager 下所有拓扑可重复运行。
- 全 PyPTO 与 native baseline 满足逐层和最终精度要求。
- mixed backend 没有 stream 顺序或 storage 生命周期问题。
- 四层之间无 cache/weight/callable 串扰。

## Phase 4：Stateful trace 与 cache block 生命周期

### 工作项

1. 实现不可变 `DecodeTrace` 和 deterministic seed。
2. 实现测试域 `CacheBlockAllocator`。
3. 尽量复用 vLLM metadata builder 生成每步 metadata。
4. 支持 request admit、advance、retire、compact 和 block reuse。
5. 支持 actual request 与 padded bucket 分离。
6. 增加 canary、write-set 和 ownership 检查。
7. 执行 4、32、128 step trace。

### 完成标准

- block ownership 始终合法。
- padded row 不修改任何有效/无效 cache 区域。
- 回收复用 block 后 native/PyPTO 仍一致。
- 128 step 不出现不可解释的误差发散或 state 泄漏。

## Phase 5：多 bucket ACLGraph

### 工作项

1. 实现 B4/B8/B12/B16 graph buffer owner。
2. 每个 specialization 在 capture 外 prepare/warmup。
3. 捕获单 op 和四层 Capsule 两类 graph。
4. 运行固定 graph 和跨 bucket trace。
5. 执行 graph 同时存活、串行交替 replay、局部 destroy 和重新 capture。
6. 分别完成 TRB 和 HBG 矩阵。
7. 验证 HBG 每次 replay 从 pristine package 恢复 mutable working slot。

### 完成标准

- 所有 bucket 连续 replay 正确。
- B4/B8/B12/B16 交替使用不串 state。
- 销毁一张 graph 不影响其他 graph。
- 全部 graph 销毁和外部 quiesce 后，optional shutdown 行为符合契约。
- TRB/HBG 结果来自各自全新进程；任何同进程切换结果都不作为本阶段证据。

## Phase 6：性能基线与优化闭环

### 工作项

1. 建立 native serial/native production overlap baseline。
2. 测量 PyPTO TRB/HBG eager 与 graph。
3. 分离单 op、四层 Capsule 和 continuous trace 指标。
4. 分析 taskQueue dequeue、Host plan cache、地址 patch 和 device scheduler。
5. 对每个优化建立前后 profiler 和正确性回归。
6. 输出同卡 ABBA 对比报告。
7. 在功能矩阵保持通过的前提下迭代优化，目标是 PyPTO steady-state latency
   低于 native production overlap baseline；优先处理 device scheduler/AICore
   尾部、taskQueue dequeue、Host structure-cache 与 per-call address patch。

### 完成标准

- 命令和结果可重复。
- 指标包含 Host、device 和 graph replay 三种口径。
- 任何性能优化均不破坏 L1 单算子边界和正确性矩阵。
- 明确记录 PyPTO 独占 AICore 对 native overlap 的机会成本。
- 最终报告以同 workload、同卡 ABBA 的 native production 为主要胜负基线；
  forced-serial native 只作为诊断，不作为达标替代物。
- 冷启动排除项和稳态必计项严格遵循 §3.3；不得使用缩略口径改变正式胜负结论。
- 每个目标配置有至少 3 个 fresh-process 同卡同流 ABBA 重复，每个 backend
  每轮至少 20 次 warmup 和 100 个稳态样本。
- 达标判定使用 §3.3 定义的 `D = native - PyPTO`，并同时满足
  `median(D50) > 0`、`median(D90) >= 0`、`median(D99) >= 0`。
- 目标为 PyPTO 超过 native production；若尚未达到，本阶段保持未完成并给出
  profiler 证据、已尝试优化和下一瓶颈。

## Phase 7：测试收敛与交付

### 工作项

1. 把公共 fixture/harness 固定放入 `tests/pypto_dsv4_decode_csa/`，纯 Host adapter/state/trace 测试从该目录复用实现并放入 UT。
2. 把 A3 eager/ACLGraph 正确性放入 one-card ST。
3. 把长时间和性能矩阵放入 nightly/manual suite。
4. 添加运行 README、环境检查脚本和结果 schema。
5. 更新本计划中的实际结论、未完成项和已知限制。

### 完成标准

- 核心失败路径有 UT。
- A3 黄金路径有稳定 ST。
- profiling 命令与结果格式进入仓库。
- 文档不把未验证路径标成 supported。

## 13. 权威文件组织

以下是本任务已经确认的源码所有权边界。文件可以在同一 package 内按 review 做小粒度拆分或合并，但不能改变“正式 kernel 和 adapter 都在 `_pypto_dsv4_csa`、测试 harness 在 `tests/pypto_dsv4_decode_csa`、pypto-lib 只读”的三条边界：

```text
vllm_ascend/
  ops/
    dsa.py                         # 保持 custom-op；增加最小 backend 分派点
    _pypto_dsv4_csa/
      __init__.py                  # 只导出正式 backend/factory/contract
      config.py                    # A3/Flash 静态配置与 static spec
      contract.py                  # 六 cache ABI、direction、dtype/stride 校验
      kernel.py                    # static L1 entry factory 与 decode_csa_core 编排
      qkv_proj_rope.py             # 正式 kernel 子模块
      rope_interleave.py           # 正式 kernel 子模块
      decode_compressor_ratio4.py  # page-aware main compressor
      decode_indexer_compressor.py # page-aware inner compressor
      decode_indexer.py            # page-aware indexer；FP16 scale storage
      decode_sparse_attn_csa.py    # sparse attention 与 output projection
      backend.py                   # PyPTODSABackend、owner、program registry
      adapter.py                   # weight pack、metadata/cache 参数适配

tests/
  pypto_dsv4_decode_csa/
    __init__.py
    config.py                      # A3/Flash/DSpark test constants
    fixtures.py                    # deterministic weights/initial cache
    reference.py                   # native/历史算法 golden runner，不复制正式 kernel
    trace.py                       # DecodeScenario/DecodeTrace
    block_allocator.py             # test-only cache ownership model
    metadata.py                    # vLLM metadata materializer
    pypto_adapter.py               # test inspection helpers
    layer_capsule.py               # HC/norm/CSA/HC shell
    runner.py                      # native/PyPTO state worlds
    graph.py                       # ACLGraph bucket owner
    compare.py                     # tensor/cache/invariant comparison
    profiling.py                   # benchmark/report helpers
    subprocess_runner.py           # TRB/HBG 分进程执行与结果汇总

  ut/
    pypto/
      test_dsv4_csa_trace.py
      test_dsv4_csa_metadata_adapter.py
      test_dsv4_csa_state_ownership.py
      test_dsv4_csa_backend_state.py

  e2e/
    pull_request/
      one_card/
        pypto/
          test_dsv4_decode_csa_l1.py
          test_dsv4_decode_csa_aclgraph.py
    nightly/
      one_card/
        pypto/
          test_dsv4_decode_csa_long_trace.py
          test_dsv4_decode_csa_performance.py
```

边界说明：

- backend 未完全打开生产开关时，可以仍由测试通过 dependency injection 激活，但 backend 和 adapter 的源码仍然放在 `_pypto_dsv4_csa/`，不先放 tests 后搬迁。
- `tests/pypto_dsv4_decode_csa/` 是公共测试支持目录；`tests/ut`、one-card ST 和 nightly 用例只负责引用它并声明场景，避免各套测试复制 fixture/trace/graph 逻辑。
- 不新增另一个 production model 文件；`dsa.py` 的外部 schema 不变。
- pypto-lib 不出现在本树中，也不是 package install/import 依赖。它只在文档、golden 来源和版本记录中作为只读参考出现。

## 14. 测试矩阵

## 14.1 Host UT

| 类别 | 用例 |
| --- | --- |
| Trace | deterministic、request churn、block reuse、invalid ownership |
| Metadata | slot mapping flatten/expand、block size、padded sentinel、mixed seq len |
| Cache ABI | 六 tuple 映射、缺项、错误 dtype/shape/layout |
| Page layout | A3 page stride、padding canary、禁止跨 page flatten、非支持 stride fail-fast |
| Static entry | B4/B8/B12/B16 正维度 metadata、cache capacity、六 InOut、runtime 隔离 |
| Weight pack | canonical 到 native/PyPTO layout、只执行一次、cache key |
| Backend state | 未 prepare、重复 warmup、错误 device、poisoned owner |
| Topology | layer name 唯一、same/distinct callable、backend route |
| Comparator | canary、write set、NaN/Inf、INT8 + scale 比较 |

## 14.2 A3 短 ST

| Runtime | Eager | ACLGraph | 1 layer | 4 layer | Mixed | Address change |
| --- | --- | --- | --- | --- | --- | --- |
| Native | 是 | 是 | 是 | 是 | 作为 mixed baseline | 是 |
| PyPTO TRB | 是 | 是 | 是 | 是 | 是 | 是 |
| PyPTO HBG | 是 | 是 | 是 | 是 | 是 | 是 |

短 ST 使用 B4、S8、1～4 个关键 position 和 4 个 decode step，目标是控制 PR 测试时间。

## 14.3 A3 长稳/性能

- B4/B8/B12/B16；
- context 128/2K/8K；
- 128 step churn；
- 100～1000 replay；
- 多个 graph owner 生命周期；
- full profiler；
- taskQueue dequeue 长尾统计；
- HBG structure-cache/address-patch 命中率。

## 15. 验收门槛

## 15.1 接口与生命周期

- [x] production custom-op schema 未因 PyPTO 改变。
- [x] PyPTO backend 使用 caller stream 和 taskQueue adapter。
- [x] capture/replay 内无 compile、prepare、alloc、H2D staging 或 sync。
- [x] warmup 是 capture 前显式流程，未 warmup capture 明确报错。
- [x] graph/context/binary/tensor owner 生命周期有测试覆盖。
- [x] 不调用 BinaryUnLoad，不调用 `rtStreamAddToModel`，不查询 capture handle。
- [x] TRB 与 HBG 的正式证据分别来自新进程，不依赖同进程 runtime 切换。
- [x] 正式 kernel/adapter 只来自 `vllm_ascend/ops/_pypto_dsv4_csa/`，运行时不导入 pypto-lib 或 tests 内 kernel。

## 15.2 功能与精度

- [x] `decode_csa_core` 与旧宽边界 program 对齐。
- [x] native/PyPTO 单 op 输出与全部 cache 对齐。
- [x] 默认四层 `PPPP` 对齐 `NNNN`。
- [x] `NPNP` 与 `PNPN` 正确。
- [ ] B4/B8/B12/B16 的静态 entry、artifact metadata 和运行结果正确。
- [x] 六类 mutable cache/state 均声明为 `InOut` 并逐步比较。
- [x] 真实 A3 page-strided cache 直接执行正确，无 contiguous mirror 或跨 page flatten。
- [x] indexer scale 使用 FP16 storage ABI，读写和容差正确。
- [x] padded request 不写 cache。
- [x] request retire/block reuse 正确。
- [ ] 128 step 无 state 泄漏和异常误差增长。

四 bucket 的最终源码均有 fresh compile artifact；B4 与 B16 有最终组合后的
nonzero A3 运行，B8/B12 只有较早版本的 nonzero 结果和最终源码 compile，因此第一项
保持未勾选。32-step TRB/HBG churn 已通过；128-step 被 native 长前置执行引发的
SoC state 问题阻断，不能用短化或 reset 掩盖，最后一项保持未勾选。

## 15.3 ACLGraph

- [x] TRB eager/capture/replay 通过。
- [x] HBG eager/capture/replay 通过。
- [x] warmup/capture 换 stream 通过。
- [x] 多 graph 同时存活、串行 replay 通过。
- [x] destroy 一张 graph 不影响另一张。
- [x] HBG 第二次及后续 replay 正确恢复 execution state。

## 15.4 性能证据

- [x] native serial 与 production overlap baseline 均存在。
- [ ] 单 op、四层 Capsule、continuous trace 三种口径齐全。
- [x] 每个正式声明达标的目标配置至少 3 个 fresh process，每轮同卡、同 caller stream、
  ABBA 交替测量，每个 backend 至少 20 次 warmup 和 100 个 sample。
- [x] native/PyPTO 使用同一 workload/输入、等价初始 state 和相同正确性门禁；
  native production overlap 保持开启。
- [x] 报告 p50/p90/p99，且 `D = native - PyPTO` 满足
  `median(D50) > 0`、`median(D90) >= 0`、`median(D99) >= 0`。
- [ ] Host enqueue、taskQueue dequeue、device span、graph replay 分开报告。
- [x] 冷启动排除项和稳态必计项严格遵循 §3.3，没有用缩略口径改变正式结论。
- [ ] 稳态中重复的地址/scalar patch 和 cache miss 均在样本内，没有被扩大
  warmup 或事前遍历工作集掩盖。
- [x] profiler 与源码审计证明没有隐藏同步或跨算子 early orchestration。

当前正式 PASS 只针对 fixed-binding B4/S8/C8191 TRB ACLGraph；paired
`D50/D90/D99` 三轮中位数为 `+13.300/+9.100/+7.068 us`。四层与 continuous
trace 尚未形成同等级正式性能结果，动态地址/scalar churn 也未按完整 Host-to-quiesce
critical path 验收，因此对应三项保持未勾选。ACLGraph 的 `graph_replay` 与
`host_enqueue` 是同一 producer 操作，device span 已覆盖 consumer dequeue；
尚无可靠的独立 dequeue-only 指标，所以不把“分开报告”伪标完成。

## 16. 风险清单与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 当前 PyPTO program 边界包含 HC | 无法与 `dsa_forward` 公平替换 | 先抽取 `decode_csa_core`，HC 留在 layer shell |
| mutable cache 方向声明不完整 | graph/编译器可能错误处理 side effect | side-effect audit + 全 cache InOut + canary 测试 |
| vLLM/PyPTO slot mapping 不同 | 写错 physical block | 单独 adapter、双向映射 UT、ownership check |
| 把 page-strided cache 当连续 tensor | 跨 page padding 读写，短 fixture 可能假通过 | block/offset/真实 stride 寻址 + padding canary + 禁止 hot-path contiguous mirror |
| A3 indexer scale 被声明为 FP32 | ABI 不匹配、地址步长和精度错误 | 对外 FP16 storage；kernel 内 FP32 compute、FP16 writeback |
| 静态 entry 硬编码 fixture cache 容量 | 真实 vLLM block id 越界或 shape binding 失败 | 物理 block 数进入 static spec/artifact key；从真实 cache tuple 派生并校验 |
| weight layout/NZ/quant 不同 | 错误结果或每次调用发生 pack | canonical fixture + 一次性 pack + launch 热路径断言 |
| B1～B3 需 padding | padded row 污染 cache | `num_reqs_actual` + invalid mapping + write-set 验证 |
| HBG Host builder读取 device data | 破坏 stream 语义 | requirements gate；topology 只能来自 Host scalar/descriptor |
| HBG working slot 容量不足 | 新 shape 在 capture 内失败 | prepare 前按最大 bucket sizing；超限图外 fail-fast |
| PyPTO 独占 AICore | 失去 native overlap，端到端收益变差 | 同时报 native overlap baseline；不只看 kernel 时间 |
| taskQueue dequeue 长尾 | Host latency 抵消 kernel 收益 | 单独采样 enqueue/dequeue，检查模板缓存和 callback 工作量 |
| 多 layer callable identity 冲突 | 第二层以后错误或挂死 | same/distinct callable 拓扑和 func-id 隔离测试 |
| ACLGraph 多 owner 生命周期错误 | destroy/replay 后 UAF 或 Conflict |多 graph 矩阵、Context-owned HBG registry、process-pinned code |
| 同进程切换 TRB/HBG 污染状态 | process-pinned binary/context/DSO 状态导致假失败 | runtime 作为 subprocess 参数，TRB/HBG 分别全新进程 |
| kernel 在 pypto-lib 与 vLLM 双份演进 | 修复和 ABI 漂移，结论无法复现 | `_pypto_dsv4_csa/` 为唯一正式源码；pypto-lib 严格只读参考 |
| 精度误差跨 step 放大 | 单步通过但真实 decode 失败 | 4/32/128 step 双 world 比较并定位首次分歧 |
| 测试 harness 偏离 vLLM metadata | 得到虚假的“可接入”结论 | 优先复用真实 builder；自造数据必须逐字段与 builder 对照 |
| reduced bridge 不等价于 MoE | 不能代表完整模型数值/性能 | 明确只报告 CSA subsystem；bridge 只用于依赖和压力 |

## 17. 明确保留的证据边界

完成本计划后，可以证明：

- PyPTO 能以 L1 普通 custom-op 的形式替换 DeepSeek V4 Flash decode CSA；
- native/PyPTO 在真实尺寸、真实 metadata/cache ABI 和连续 decode state 下的数值差异；
- PyPTO TRB/HBG 在 ACLGraph 中的 capture/replay 与生命周期；
- 多个 layer、多个 callable、多个 graph bucket 和 mixed torch/native/PyPTO 节点可以串行正确工作；
- CSA 子系统级性能收益和代价。

完成本计划后仍不能证明：

- 完整 DeepSeek V4 能在当前双卡上部署；
- vLLM scheduler、模型 loader、TP/EP/HCCL、MoE、sampler 和 server API 与该 backend 完全集成；
- 完整模型 tokens/s 一定提升；
- 同一 device 上并发 replay 安全；
- A5 或其他 SoC 可用。

这些限制必须保留在最终报告中，不能因 Capsule 的覆盖较强就把它描述为完整模型 E2E。

## 18. 建议的阶段性提交

遵循“阶段性 commit、未经明确要求不 push”的工作方式。建议提交切分：

1. `test: add DeepSeek V4 decode CSA fixtures and ABI checks`
2. `feat: add static PyPTO DeepSeek V4 decode CSA kernels under vLLM ops`
3. `feat: add PyPTO adapter for vLLM DSA custom op`
4. `test: add stateful multi-layer decode CSA capsule`
5. `test: cover TRB and HBG ACLGraph decode traces`
6. `perf: add decode CSA profiling and reports`
7. `docs: record DeepSeek V4 decode CSA validation results`

每个 commit 都应能够独立解释其测试范围；不要把 PyPTO、simpler、vLLM-Ascend 多仓的无关修改混成一个 commit。涉及不同仓库时分别记录依赖 commit hash。

本任务的 kernel、adapter 和测试提交均属于 vLLM-Ascend 仓库。pypto-lib 只记录参考 commit，不产生本任务提交；除非后续另有明确任务，不能借本计划修改或提交 pypto-lib。

## 19. 开发执行检查表

### 开始实现前

- [x] 确认 A3 device 空闲。
- [x] 记录五个相关仓库/子模块 commit。
- [x] 修复 PyPTO extension/source hash 和 `libstdc++` ABI 环境。
- [x] 确认当前 dirty worktree 中用户改动并避开覆盖。
- [x] 确认首轮采用 B4、S8、TP1、A3，并准备 B8/B12/B16 静态 spec。
- [x] 确认正式源码位于 `_pypto_dsv4_csa/`、测试支持位于 `tests/pypto_dsv4_decode_csa/`，pypto-lib 保持只读。
- [x] 从真实 A3 cache tuple 记录六类 shape/dtype/stride/page stride，确认 indexer scale 为 FP16。

### 每个 Phase 结束

- [x] 更新本计划中的实际结论和偏差。
- [x] 运行对应 Host UT。
- [x] 运行对应 A3 ST。
- [x] 保存完整命令、日志、结果和 profiler。
- [x] 检查没有新增 capture 内 allocation/sync。
- [ ] 阶段性 commit；没有用户明确要求时不 push。

### 宣布任务完成前

- [ ] 所有验收项有可追踪证据。
- [x] native/PyPTO 使用相同 workload 和初始 state。
- [x] 性能在同卡交替测量。
- [x] 所有 mutable state 都已比较。
- [x] TRB/HBG 均有 ACLGraph 多次 replay 结果。
- [x] TRB/HBG 结果来自独立新进程并记录各自退出状态。
- [ ] B4/B8/B12/B16 均有静态 artifact metadata 与最终源码 nonzero 正确性证据。
- [x] page-strided cache、FP16 indexer scale 和六类 InOut 均有专项测试证据。
- [x] 报告明确写出未覆盖完整 Engine 和完整模型。

阶段性 commit 当前被许可证/NOTICE/provenance 决策阻断，而不是遗忘。未完成验收项
具体为 B8/B12 最终组合 nonzero A3、128-step native 长前置状态问题、动态 binding
性能、四层/trace 正式性能和完整 Engine（后者本来就是非目标）；它们都在 §15 和
过程记录第 92 节逐项保留，不能因为单算子性能达标而自动勾选。

## 20. 推荐的首条实现路径

为尽快得到有价值的板上反馈，第一条纵向闭环按以下顺序执行：

```text
修复构建环境
  -> B4/S8 decode_csa_core standalone TRB eager
  -> 全 mutable-state golden
  -> vLLM dsa_forward 单 op native/PyPTO shadow
  -> 单 op TRB ACLGraph 8 次 replay
  -> 1-layer real HC/norm Capsule
  -> 4-layer PPPP + NPNP eager
  -> 4-layer TRB ACLGraph
  -> 同矩阵 HBG
  -> 32-step request churn + block reuse
  -> B4/B8/B12/B16 graph 交替
  -> 正式 profiling
```

这条路径每一步都能形成独立结论。如果在 HBG、大 bucket 或连续 trace 上遇到问题，已经完成的较小闭环仍然可以用于定位，而不会退化成“完整模型启动失败，无法判断 CSA 是否正确”。
