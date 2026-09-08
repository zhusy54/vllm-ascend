# PyPTO WSE 阶段 1B T04～T08 验证记录

## 1. 结论

阶段 1B 的 T04～T08 已在两张 Ascend950PR 上完成 NPU surrogate 预验证：Attention
进程和 WSE surrogate 进程分别绑定 NPU 0/1，每个 generation 各自只启动一次 AIV kernel。
两个 kernel 通过 CANN ACL VMM P2P 映射，在 Device Memory 中完成单 slot 闭环及双 slot 流水验证。

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

T06 结果：

- service 等待设备侧背压探针，不提前消费已占用的 slot 0/1，确定性形成满队列。
- driver 在两个 credit 均被占用时尝试第三个请求，得到一次 `NO_CREDIT` 并转入 pending。
- `submission_retry_spins=0`：pending 请求不循环尝试提交，也不修改任一在途 slot。
- service 恢复后 pending 请求取得归还的 credit，最终 4/4 task 完成，credit 4/4 归还。
- 两种启动顺序的 slot overwrite、sequence、checksum、marker 和 timeout 错误均为 0。
- READY 后 Host task、completion、payload 和其他控制消息均为 0，Host bounce 为 0 B。
- 脱敏证据：[pypto-wse-stage1b-t06-evidence.json](pypto-wse-stage1b-t06-evidence.json)

T07 结果：

- 连续传输 100 个 1 MiB round-trip，交替复用 slot 0/1，每次运行累计双向 200 MiB。
- 每个 sequence 使用唯一头尾标记；两端各校验 13,107,200 个 64-bit payload word。
- input payload、descriptor 和 submission，以及 output、completion metadata 和 signal 均使用
  显式 `st_dev` + `DSB_ALL` 顺序。
- driver 在 completion signal 后立即校验完整 output，100 次检查的额外等待均为 0 cycle。
- 两种启动顺序的旧 payload、输入不完整、提前 completion、checksum 和头尾标记错误均为 0。
- 脱敏证据：[pypto-wse-stage1b-t07-evidence.json](pypto-wse-stage1b-t07-evidence.json)

T08 结果：

- generation G 完成 4/4 请求后，两端依次关闭 kernel、导入 window、自有 window 和 ACL runtime。
- 双方确认 G 代资源释放后才创建 G+1；两端旧 peer handle 在重分配前后各探测一次，
  4/4 次均被 `aclrtMemImportFromShareableHandle` 以错误码 `507899` 拒绝。
- G+1 的正常请求先占用唯一 credit，随后注入 G 代 descriptor；service 拒绝旧 descriptor，
  并在正常 completion 前注入 G 代 completion。
- driver 拒绝旧 completion，未归还 credit，且确认 G+1 slot 保持占用；随后正常请求完成，
  `old_completion_credit_releases=0`、`progress_after_stale=1`。
- 两端新旧 opaque handle 均不同，handle collision 和意外导入成功均为 0。
- 两种启动顺序的 5/5 正常请求、旧代隔离、Host 零热路径及双代资源清理均通过。
- 脱敏证据：[pypto-wse-stage1b-t08-evidence.json](pypto-wse-stage1b-t08-evidence.json)

当前能力结论仍为 **C1（NPU surrogate only）**，`c2_status` 为 `NOT_ESTABLISHED`。
T04～T08 证明单 slot、双 slot 流水、满队列背压、顺序可见性及 generation 隔离成立；完成 T09～T12 前不能
声明达到 C2。
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

## 4. 队列满和背压

T06 的 driver 先提交 sequence 1/2，占满两个 slot 和两个 credit；第三次提交只执行一次
credit 判定。credit 为零时，driver 记录 `NO_CREDIT`，保留 sequence 3 的 pending 状态，
但不写 payload、descriptor 或 doorbell。service 在设备侧看到该记录后才恢复消费，因此
不会因两个 kernel 的调度时序差异而跳过满队列状态。

第一个 completion 归还 credit 后，driver 才把 pending sequence 3 写入已经终结的 slot 0；
后续 sequence 4 同样等待 slot 1 的 credit。设备报告同时要求满队列深度为 2、背压事件为
1、提交重试自旋为 0，并在恢复后完成全部 4 个请求。

## 5. 顺序与可见性

T07 使用两个 1 MiB slot，并以 `max_inflight=1` 交替复用。driver 先写完完整 input 和唯一
头尾标记，再执行 `DSB_ALL`，最后发布 descriptor 和 submission signal。service 收到 signal
后立即校验 descriptor、完整 payload、checksum 和标记，并显式检查是否混入同一 slot 的
前一个 sequence。

service 写完完整 output 后执行 `DSB_ALL`，再发布 completion metadata 和 signal。driver
观察到 signal 后仅执行可见性 fence，不进行 sleep 或定时等待，随即读取 completion 和全部
1 MiB output。任何不完整 output 都会同时进入 premature completion 和数据错误计数。

## 6. Generation 隔离

T08 使用连续的 G/G+1 两代独立 VMM 资源和两轮 AIV kernel。G 代完成并 drain 后，两端先
解除 peer 映射、释放自有 window 并关闭 ACL runtime，再通过控制面互相确认资源已关闭。
G+1 runtime 随后直接尝试导入旧 peer handle，要求 CANN 拒绝；新 window 分配并交换 manifest
后再次尝试旧 handle，以覆盖资源重用造成的 ABA 风险。证据只记录错误码和 opaque SHA-256，
不记录 raw shareable handle 或设备地址。

G+1 driver 先发布正常请求，使单 slot、单 credit 处于占用状态，然后向独立 stale lane 发布
G 代 descriptor。service 看到正常请求但暂不处理，先拒绝旧 descriptor，并向 driver 的 stale
lane 发布 G 代 completion。driver 在正常 completion 尚未出现时拒绝旧 completion，确认 credit
仍为 0、当前 slot 未被释放，再向 service 确认；service 此后才处理正常请求。最终正常请求继续
完成，证明旧 completion 不会释放新 slot 或阻断 G+1 进展。

## 7. 硬件证据

T04 最终功能代码 revision 为 `3a2582c5`，T05 为 `b225e1cf`，T06 为 `0badeef2`，
T07 为 `2602dc49`，T08 为 `5787f66a`。通过 `task-submit` 锁定 NPU 0/1，五项验证的两种启动顺序均通过：

| task | 启动顺序 | T04 | Driver/Service launch | Host 热路径 | 清理 |
| --- | --- | --- | --- | --- | --- |
| `task_20260908_160054_350511627711` | Attention-first | 100/100 | 1 / 1 | 全部为 0 | 通过 |
| `task_20260908_160216_3526421445` | WSE-first | 100/100 | 1 / 1 | 全部为 0 | 通过 |

| task | 启动顺序 | T05 | 双 slot / credit / 乱序 | Host 热路径 | 清理 |
| --- | --- | --- | --- | --- | --- |
| `task_20260908_162927_381083423560` | Attention-first | 100/100 | 50+50 / 100=100 / 50 | 全部为 0 | 通过 |
| `task_20260908_163002_381447212224` | WSE-first | 100/100 | 50+50 / 100=100 / 50 | 全部为 0 | 通过 |

| task | 启动顺序 | T06 | 尝试 / NO_CREDIT / retry spin | 恢复后进展 | 清理 |
| --- | --- | --- | --- | --- | --- |
| `task_20260908_165831_811913425` | Attention-first | 4/4 | 5 / 1 / 0 | 4/4 | 通过 |
| `task_20260908_165931_1548322410` | WSE-first | 4/4 | 5 / 1 / 0 | 4/4 | 通过 |

| task | 启动顺序 | T07 | payload | 立即检查 / 延迟 | Host 热路径 | 清理 |
| --- | --- | --- | --- | --- | --- | --- |
| `task_20260908_171913_14417018689` | Attention-first | 100/100 | 1 MiB | 100 / 0 cycle | 全部为 0 | 通过 |
| `task_20260908_171952_14879930649` | WSE-first | 100/100 | 1 MiB | 100 / 0 cycle | 全部为 0 | 通过 |

| task | 启动顺序 | T08 | 旧 handle 拒绝（前/后） | 旧 descriptor/completion | 信用误释放 | 清理 |
| --- | --- | --- | --- | --- | --- | --- |
| `task_20260908_175522_39823812451` | Attention-first | 5/5 | 2/2 | 1/1 均拒绝 | 0 | 通过 |
| `task_20260908_175612_40680326335` | WSE-first | 5/5 | 2/2 | 1/1 均拒绝 | 0 | 通过 |

WSE-first 运行中，driver 和 service 分别报告约 26.23M 和 25.70M device cycles；Host 观测
整个 kernel 分别约 26.4 ms 和 25.9 ms。该数据只用于发现明显风险，不是吞吐或时延验收结果。

kernel binary SHA-256 已逐端记录。驱动日志同时证明 P2P enable，并在 teardown 后显示
P2P HBM 当前分配归零。

T05 的 Attention-first driver/service 分别报告约 205.32M/204.27M device cycles，WSE-first
分别约 205.44M/204.13M device cycles。该数据只用于发现明显风险，不是吞吐或时延验收结果。

T06 的 Attention-first driver/service 分别报告约 0.86M/0.52M device cycles，WSE-first
分别约 1.62M/0.53M device cycles。该数据只用于发现明显风险，不是吞吐或时延验收结果。

T07 的 Attention-first driver/service 分别报告约 6.306B/6.293B device cycles，WSE-first
分别约 6.306B/6.292B device cycles。该数据只用于发现明显风险，不是吞吐或时延验收结果。

T08 的每次运行包含 G/G+1 两轮 kernel。Attention-first 的 driver/service 合计约
1.36M/5.57M device cycles，WSE-first 合计约 4.53M/1.69M device cycles。该数据只用于发现
明显风险，不是吞吐或时延验收结果。

## 8. 自动门禁

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

T06 在通用约束之外，还要求：

- 两个初始 request 都处于在途状态时，第三次提交必须返回一次 `NO_CREDIT`；
- 第三个 request 保持 pending，`submission_retry_spins` 必须为 0；
- pending 前后不得覆盖 slot 0/1，峰值队列深度不得超过 2；
- service 必须先观测暂停状态，再恢复消费并完成 4/4 request；
- acquired、terminal 和 returned credit 均为 4。

T07 在通用约束之外，还要求：

- 100 个 1 MiB payload 和 100 个唯一尾标记全部通过；
- 每端校验的 payload word 必须为 13,107,200；
- input publish、input visibility check、output publish 各发生 100 次；
- completion signal 后立即检查 100 次，`post_completion_delay_cycles` 必须为 0；
- stale payload、incomplete payload 和 premature completion 错误必须为 0。

T08 在通用约束之外，还要求：

- G 代必须完成 4/4 请求并完整关闭资源，G+1 必须使用重新分配的 window 和新 runtime；
- 两端旧 peer handle 在 G+1 重分配前后各导入一次，4 次探测必须全部失败；
- 两端新旧 local/peer opaque handle 不得碰撞，旧 handle 意外导入成功次数必须为 0；
- G+1 必须各注入并拒绝一个 G 代 descriptor 和 completion；
- 旧 completion 不得归还当前 credit，driver/service 都必须确认当前 slot 保持占用；
- 拒绝旧流量后 G+1 正常请求必须继续完成，所有 generation、sequence 和数据错误必须为 0；
- 每端只允许每代一次 kernel launch，共两次，双代全部资源必须关闭。

离线相关单元测试共 156 项通过，覆盖 Stage 0/1A 回归、AIV binary 生命周期、T04～T08
契约、结果聚合与证据脱敏。下一步按计划实现 T09 超时状态机。
