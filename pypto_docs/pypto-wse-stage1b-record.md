# PyPTO WSE 阶段 1B T04/T05 验证记录

## 1. 结论

阶段 1B 的 T04/T05 已在两张 Ascend950PR 上完成 NPU surrogate 预验证：Attention
进程和 WSE surrogate 进程分别绑定 NPU 0/1，各自只启动一次 AIV kernel。两个 kernel
通过 CANN ACL VMM P2P 映射，在 Device Memory 中完成单 slot 闭环及双 slot 流水验证。

T04 结果：

- 固定 slot：0。
- sequence：1～100，全部使用不同的确定性 pattern。
- payload：每次 4 KiB；每个成功运行累计输入加输出 819,200 B。
- Attention device driver：生成 input、发布 submission、等待 completion、逐字校验 output。
- WSE surrogate device service：等待 submission、校验 input、XOR 变换、写回 output 并发布 completion。
- 两端 device report 均为 100/100，checksum、marker、sequence、generation 和 timeout 错误均为 0。
- READY 后 Host task 消息、completion 消息、payload 和其他控制消息均为 0。
- Host bounce：0 B；资源清理通过。
- 脱敏证据：[pypto-wse-stage1b-evidence.json](pypto-wse-stage1b-evidence.json)

T05 结果：

- `slot_count=2`、`max_inflight=2`，sequence 1～100；slot 0/1 各完成 50 次。
- slot 0 使用 4 KiB payload，slot 1 使用 64 KiB payload；每次运行累计双向传输
  6,963,200 B。
- driver 每轮先取得两个 credit 并发布一对请求，service 固定先完成 slot 1，再完成 slot 0。
- 每次运行观测到 50 次乱序 completion，均按 slot、sequence 和 generation 正确匹配。
- 100 个 task 均进入终结态，100 个 credit 全部归还；slot 覆盖和所有校验错误均为 0。
- 两种启动顺序的 READY 后 Host 热路径计数均为 0，Host bounce 为 0 B，资源清理通过。
- 脱敏证据：[pypto-wse-stage1b-t05-evidence.json](pypto-wse-stage1b-t05-evidence.json)

当前能力结论仍为 **C1（NPU surrogate only）**，`c2_status` 为 `NOT_ESTABLISHED`。
T04/T05 证明单 slot 和双 slot device loop 成立；完成 T06～T12 前不能声明达到 C2。
真实 NPU-WSE 能力仍为 `NOT_ESTABLISHED`，`evidence_status` 仍为 `SIMULATION`。

## 2. 设备闭环

T04 使用两个纯 CCEC AIV kernel，均在 READY 前由 Host 启动一次：

1. Attention driver 直接向对端导入的 VMM input window 写入 4 KiB pattern。
2. driver 执行 `DSB_ALL`，再通过 `st_dev` 发布 submission sequence。
3. WSE surrogate service 使用 `ld_dev` 在设备侧等待 sequence，随后读取并校验完整 input。
4. service 对每个 64-bit word 执行 `LOW8(sequence_id)` 重复字节 XOR，并直接写入对端 output window。
5. service 执行 `DSB_ALL`，再发布 completion sequence。
6. driver 在设备侧等待 completion，并立即校验 generation、sequence、status、checksum、头尾标记及全部 output word。
7. driver 校验成功后才复用 slot 0 并提交下一个 sequence。

Host 在 hot path 中只阻塞等待整个 device kernel 结束，不逐 sequence 调用 copy、transform、
doorbell 或 completion API。运行结束后，Host 只读回每端一条 64 B 汇总 report。

## 3. 双 slot 流水

T05 使用独立的 driver/service AIV kernel 和两个 1 MiB payload slot。driver 在等待前先后
发布 slot 0 的 4 KiB 请求和 slot 1 的 64 KiB 请求，使两个 slot 同时处于非 FREE 状态。
service 确认两个 submission 均可见后，故意按 slot 1、slot 0 的顺序完成，从而验证乱序
completion 不能释放错误 slot 或 credit。

两端使用独立的 per-slot signal 和 descriptor cache line。每次复用 slot 前，driver 检查上一
sequence 的 completion；service 和 driver 均校验 generation、sequence、slot、payload size、
checksum 及头尾标记。输入发布和输出完成继续采用 `st_dev` + `DSB_ALL` 可见性顺序。

## 4. 硬件证据

T04 最终功能代码 revision 为 `3a2582c5`；T05 最终功能代码 revision 为 `b225e1cf`。
通过 `task-submit` 锁定 NPU 0/1，两项验证的两种启动顺序均通过：

| task | 启动顺序 | T04 | Driver/Service launch | Host 热路径 | 清理 |
| --- | --- | --- | --- | --- | --- |
| `task_20260908_160054_350511627711` | Attention-first | 100/100 | 1 / 1 | 全部为 0 | 通过 |
| `task_20260908_160216_3526421445` | WSE-first | 100/100 | 1 / 1 | 全部为 0 | 通过 |

| task | 启动顺序 | T05 | 双 slot / credit / 乱序 | Host 热路径 | 清理 |
| --- | --- | --- | --- | --- | --- |
| `task_20260908_162927_381083423560` | Attention-first | 100/100 | 50+50 / 100=100 / 50 | 全部为 0 | 通过 |
| `task_20260908_163002_381447212224` | WSE-first | 100/100 | 50+50 / 100=100 / 50 | 全部为 0 | 通过 |

WSE-first 运行中，driver 和 service 分别报告约 26.23M 和 25.70M device cycles；Host 观测
整个 kernel 分别约 26.4 ms 和 25.9 ms。该数据只用于发现明显风险，不是吞吐或时延验收结果。

kernel binary SHA-256 已逐端记录。驱动日志同时证明 P2P enable，并在 teardown 后显示
P2P HBM 当前分配归零。

T05 的 Attention-first driver/service 分别报告约 205.32M/204.27M device cycles，WSE-first
分别约 205.44M/204.13M device cycles。该数据只用于发现明显风险，不是吞吐或时延验收结果。

## 5. 自动门禁

T04 采用 fail-closed 判定，必须同时满足：

- driver/service execution context 均为 `AIV_DEVICE_KERNEL`；
- 两端各只发生一次 kernel launch；
- device submission、completion 和 validated sequence 均为 100；
- 两端 report 的所有错误计数为 0；
- input-before-command 与 output-before-completion 使用显式 `st_dev`、`DSB_ALL` 和 signal publish；
- READY 后 Host task/completion/control 消息和 payload 字节均为 0；
- Host bounce 和 fallback 均为 0；
- 两种启动顺序均通过；
- imported window、owned window、kernel binary/stream 和 ACL runtime 全部关闭；
- artifact 不包含 raw shareable handle 或设备地址。

T05 在上述约束之外，还要求：

- slot 0/1 各处理 50 次，`max_inflight` 必须为 2；
- 两种 payload 大小必须分别为 4 KiB 和 64 KiB；
- acquired、terminal 和 returned credit 均为 100；
- 每轮 slot 1 先于 slot 0 完成，乱序 completion 计数必须为 50；
- slot overwrite、sequence 串扰及所有数据校验错误必须为 0。

离线相关单元测试共 106 项通过，覆盖 Stage 0/1A 回归、AIV binary 生命周期、T04/T05
契约、结果聚合与证据脱敏。下一步按计划实现 T06 队列满和背压，再验证 T07 顺序可见性。
