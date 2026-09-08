# PyPTO WSE 阶段 1C T09～T12 验证记录

## 1. 结论

阶段 1C 的 T09～T12 已在两张 Ascend950PR 上完成 NPU surrogate 验证。统一收集器同时校验
既有 T01～T08 脱敏证据后，阶段 1 的 Host-local NPU surrogate 能力达到 **C2**：

- `capability_level=C2`；
- `c2_status=ESTABLISHED_HOST_LOCAL_NPU_SURROGATE`；
- `claim_scope=HOST_LOCAL_NPU_SURROGATE_ONLY`；
- `evidence_status=SIMULATION`；
- 真实 NPU-WSE 能力仍为 `NOT_ESTABLISHED`。

该结论证明同 Host、双 NPU、CANN ACL VMM P2P 原型中的 device-driven command、completion、
timeout、generation 和 drain 行为成立，不证明真实 WSE、RoCE/UB 或跨 Host C3 能力。

脱敏证据：[pypto-wse-stage1c-evidence.json](pypto-wse-stage1c-evidence.json)

## 2. T09 持续运行

每次运行包含 100 次预热和 10,000 次正式 round-trip。driver/service 各只启动一次 AIV
kernel，两个 payload slot 每轮保持两个 request 在途；Host 在 READY 后只等待整个 kernel，
不逐 request 推动 progress。

正式请求使用确定性混合负载：

| payload | 次数 |
| --- | ---: |
| 64 B | 4,000 |
| 4 KiB | 4,000 |
| 64 KiB | 1,900 |
| 1 MiB | 100 |

每 100 个正式请求由 device report 累计一个 progress checkpoint，共 100 个。两种启动顺序均满足：

- 10,100/10,100 request 完成，slot 0/1 各 5,050；
- 10,100 credit 获取并归还，无永久 slot/credit 丢失；
- queue stall、checksum、marker、sequence、generation 和 timeout 错误全部为 0；
- 每端校验 30,803,200 个 64-bit word；
- READY 后 Host task/completion/control/payload 均为 0；
- 稳态 RSS 最大增量 208,896 B，Host CPU 最大占用约 9.37%；
- teardown 后 P2P HBM 分配归零。

Attention-first driver/service 分别约 14.879B/14.866B device cycles，WSE-first 分别约
14.915B/14.901B cycles。该数据仅作风险观测，不是性能验收结论。

## 3. T10 WSE 无响应

driver 提交一个 4 KiB request 后，service 只发布 `PAUSED` 状态而不发布 completion。双方
设备 kernel 使用 500,000,000 cycles 的配置 timeout，随后将 generation 标记 unhealthy。

两种启动顺序均满足：

- request 终结为 `WSE_UNRESPONSIVE_TIMEOUT`，没有成功 completion；
- generation 停止接收新 request；
- 唯一 slot 保持 quarantine，credit 不归还且不复用；
- `forged_success_completions=0`、`slot_reuse_after_timeout=0`；
- driver/service timeout 分别落在 500,000,019～500,000,052 和
  500,000,005～500,000,006 cycles；
- 故障结束后资源完整清理。

## 4. T11 进程异常退出

launcher 在两个 endpoint 都完成 READY 后发送真实 `SIGKILL`。两次任务分别终止 WSE
surrogate Host 进程和 Attention Host 进程；存活端不依赖 Host 心跳推进设备状态机。

两种故障方向均满足：

- victim exit code 为 `-9`；
- survivor 在约 20,000,000,000 cycles 后返回明确 timeout/unhealthy，不无限等待；
- survivor 执行 best-effort cleanup，旧 peer handle 导入以错误码 `507899` 被拒绝；
- launcher 立即以 generation G+1 重启 T04，恢复运行 100/100；
- 新 session 不复用旧资源。

该测试只覆盖同 Host 进程退出，不能等价证明真实 Host 掉电、网络分区、单向黑洞或 NIC reset。

## 5. T12 正常 drain/close

每次运行先接受 4 个 request，再由设备侧停止 admission，并拒绝一次 post-stop submission。
service 处理完所有已接受请求后才确认停止。Host 随后严格执行：

```text
STOP_NEW_SUBMISSIONS
  → DRAIN_ACCEPTED_REQUESTS
  → VALIDATE_CREDITS_RETURNED
  → STOP_SERVICE
  → RELEASE_PEER_IMPORT
  → UNREGISTER_LOCAL_WINDOW
  → FREE_LOCAL_MEMORY
  → DESTROY_RUNTIME
```

两种启动顺序均满足：

- 4/4 accepted request 都有 terminal outcome；
- credit 4/4 获取并归还；
- `early_window_releases=0`；
- 两端八步关闭顺序一致；
- kernel、peer import、owned window 和 runtime 每端重复 close 一次，合计 8/8 幂等成功；
- duplicate close 拒绝和 window、queue、registration、context 残留均为 0。

## 6. 实机任务

最终功能代码 revision 为 `e59b9a00`。

| Case | task | 启动/故障 | 结果 |
| --- | --- | --- | --- |
| T09 | `task_20260908_191247_80932828907` | Attention-first | PASS |
| T09 | `task_20260908_191328_81178525074` | WSE-first | PASS |
| T10 | `task_20260908_191410_8194701298` | Attention-first | PASS |
| T10 | `task_20260908_191437_82545123675` | WSE-first | PASS |
| T11 | `task_20260908_191133_80518318851` | WSE victim | PASS |
| T11 | `task_20260908_191503_8319349735` | Attention victim | PASS |
| T12 | `task_20260908_191600_83616312940` | Attention-first | PASS |
| T12 | `task_20260908_191627_83882520827` | WSE-first | PASS |

所有 evidence 只保留 opaque handle SHA-256、返回码、kernel hash、设备汇总 report 和日志 hash；
不包含 raw shareable handle 或设备地址。

## 7. C2 边界与下一步

T01～T12 已满足计划中的同 Host C1、C2 和稳定性断言，因此可以开始把已验证接口封装为
PyPTO `RemoteDeviceEndpoint`。后续仍需：

1. 将本地 ACL VMM P2P prototype 封装为 PyPTO endpoint，而不是把测试 kernel 直接接入 vLLM；
2. 使用 PyPTO Scheduler 重跑 submit/completion 和 Remote Task DAG 验证；
3. 有真实 WSE 后替换 surrogate service；
4. 使用目标 RoCE/UB 在双 Host 上执行 C3 gate；
5. C3 通过前不得声明真实 NPU-WSE 跨节点数据面成立。
