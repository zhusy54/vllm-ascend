# PyPTO WSE 阶段 1A T01/T02/T03 验证记录

## 1. 结论

阶段 1A 的 P0 项 T01/T02/T03 已在两张 Ascend950PR 上完成 NPU surrogate 预验证：
Attention 进程和 WSE surrogate 进程分别绑定 NPU 0/1，通过 CANN ACL VMM 导出、跨进程
导入和 P2P 映射完成双向 Device Memory 传输及设备侧变换往返。

- T01 NPU→WSE surrogate：4/4 基础大小通过，共 1,118,272 B。
- T02 WSE surrogate→NPU：4/4 基础大小通过，共 1,118,272 B。
- T03 NPU→WSE surrogate 变换→NPU：4/4 基础大小通过，单向 payload 共 1,118,272 B，
  双向 wire bytes 共 2,236,544 B。
- 基础大小：64 B、4 KiB、64 KiB、1 MiB。
- Transport：`cann_acl_vmm_p2p`，scope 为 `HOST_LOCAL`。
- 数据操作：`aclrtMemcpy(..., ACL_MEMCPY_DEVICE_TO_DEVICE)`。
- T03 变换：WSE surrogate 直接在 VMM Device Memory 上执行
  `aclnnInplaceBitwiseXorScalar(LOW8(sequence_id))`。
- Host bounce 和 T03 Host intermediate payload 均为 0 B；Host 初始化写入和最终结果读回各
  3,354,816 B，二者仅用于测试准备和最终校验，不位于设备间往返路径。
- 资源清理：两端 imported window、owned window 和 ACL runtime 均关闭；驱动日志显示 P2P HBM 当前分配归零。
- 脱敏证据：[pypto-wse-stage1a-evidence.json](pypto-wse-stage1a-evidence.json)

本结果的能力等级仅为 **C1（NPU surrogate only）**。`evidence_status` 仍为
`SIMULATION`，真实 NPU-WSE 能力为 `NOT_ESTABLISHED`；本结果不声明 C2/C3。

## 2. 实现边界

阶段 1A 未运行 PyPTO task、FFN、DAG 或 vLLM：

1. 两个独立 Python 进程分别初始化自己的 ACL Device Context。
2. 每端在本地 Device 分配 1 MiB 逻辑 VMM window，实际按 2 MiB granularity 映射。
3. TCP bootstrap 只交换 VMM opaque handle 和 metadata，不传 payload。
4. 发送端先用 H2D 初始化本地 window，再从本地设备地址向对端导入地址执行 D2D。
5. T01/T02 接收端仅在 D2D 返回并同步后 D2H 一次，校验完整 payload 和 SHA-256。
6. T03 由 Attention 端 H2D 初始化后 D2D 发送，WSE surrogate 在本地 Device Memory 原地 XOR，
   再 D2D 返回；Attention 端只在往返结束后 D2H 一次做最终校验。
7. teardown 先关闭 imported mapping，再经 `DETACHED` barrier 释放 owned physical window。

Stage 1A 允许 Host 逐 case 通知和校验，因此控制面除 T01/T02 的 8 个
`TRANSFER_COMPLETE` / 8 个 `VERIFIED` 外，T03 还有各 4 个 `ROUND_TRIP_FORWARD`、
`ROUND_TRIP_RETURN` 和 `ROUND_TRIP_VERIFIED`。这不能作为 Host-free invocation 热路径或 C2 证据。

## 3. 硬件证据

最终功能代码 revision 为 `a435d28a`，通过 `task-submit` 锁定 NPU 0/1，两个启动顺序均通过：

| task | 启动顺序 | T01 | T02 | T03 | P2P 日志 | 清理 |
| --- | --- | --- | --- | --- | --- | --- |
| `task_20260908_151056_312564524948` | Attention-first | 4/4 | 4/4 | 4/4 | 双端存在 | 通过 |
| `task_20260908_151122_312816728635` | WSE-first | 4/4 | 4/4 | 4/4 | 双端存在 | 通过 |

最终 WSE-first 运行中，T01/T02 的 1 MiB D2D 区间分别约为 1.45 ms 和 1.55 ms；这些
数值包含 Python 循环和 16 次 64 KiB 阻塞 copy，只用于发现明显风险，不是性能验收结果。
T03 的 1 MiB forward、设备 XOR、return 分别约为 1.46 ms、0.03 ms、1.58 ms；ACLNN 每次
请求 16,777,728 B 临时 workspace，后续性能与内存阶段需评估复用或更轻量的变换实现。

## 4. 已确认限制

当前 CANN 9.2.0 / Driver 25.7.rc1.6 组合存在需要继续跟踪的原始 copy 行为：

- 单次 64 B、4 KiB 和 64 KiB `ACL_MEMCPY_DEVICE_TO_DEVICE` 校验正确。
- 单次 1 MiB 调用返回成功，但目标 checksum 不一致；诊断任务为
  `task_20260908_141717_281007320337`。
- 改用 `ACL_MEMCPY_INTER_DEVICE_TO_DEVICE` 时首个调用返回错误码 `100000`；诊断任务为
  `task_20260908_141610_28060536058`。
- 当前验证实现因此固定使用 64 KiB D2D chunk，1 MiB payload 对应 16 次 copy，并在每条
  observation 中记录 `transfer_chunks` 和 `max_transfer_chunk_bytes`。

分块是阶段 1A 验证策略，不应直接成为正式 Transport ABI。进入性能设计前需要确认单次
大块 copy 错误的 CANN 约束、正确 API 或 SDK 缺陷归属。

## 5. 自动门禁

C1 surrogate 结论采用 fail-closed 判定，必须同时满足：

- 两方向全部基础大小 checksum 一致；
- T03 全部基础大小的输入、预期输出和最终输出 checksum 一致；
- source/destination 均标记为 Device Memory；
- T03 transform memory 为 Device Memory，transform API 固定为
  `aclnnInplaceBitwiseXorScalar`，且 `host_intermediate_payload_bytes == 0`；
- backend、handle kind、copy kind 和 64 KiB 分块数与契约一致；
- `host_bounce_bytes == 0` 且未使用 fallback；
- 两端驱动日志均出现 P2P enable 和 P2P HBM 释放证据；
- imported window、owned window 和 runtime 全部关闭；
- Attention-first 与 WSE-first 至少各有一次成功运行；
- artifact 不包含 raw shareable handle 或设备地址。

离线单元测试覆盖阶段 0 回归、ACL ABI/VMM 生命周期、双端协议、T03 设备变换、证据契约
和脱敏汇聚。Stage 1A 已结束；后续 T04 设备闭环结果见
[pypto-wse-stage1b-record.md](pypto-wse-stage1b-record.md)。
