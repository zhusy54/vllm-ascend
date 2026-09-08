# PyPTO WSE 阶段 0 验证记录

## 1. 结论边界

本轮使用两张 Ascend NPU 分别承担 Attention 端和 WSE surrogate 端，仅验证阶段 0
的协议、进程边界、能力盘点和控制面启动条件。

- 验证 profile：`NPU_SURROGATE`
- Transport scope：`SIMULATION`
- 本轮最高可声明等级：C0
- 不声明：真实 WSE、跨 Host、Device Memory 双向传输、C1、C2 或 C3

阶段 0 通过只表示契约和验证入口足以支持后续阶段 1A。数据传输必须使用后续运行证据
单独判定，不能由 API 名称、符号存在或 Host TCP 通信推断。

## 2. 当前环境基线

| 项目 | 当前值 | 状态/证据 |
| --- | --- | --- |
| NPU | 8 × Ascend950PR，单卡 128 GiB HBM | `npu-smi info`，健康状态均为 OK |
| Driver | 25.7.rc1.6 | `npu-smi info` |
| CANN | 9.2.0 | `ascend-toolkit/latest` 解析结果 |
| PyPTO | 0.2.1 | Python distribution metadata |
| Simpler | `407438ef677b9a5787c4d645e8cae4f7d4919d4e` | 参考 checkout revision |
| 目标平台 | `a5` | Ascend950 系列映射 |
| 目标 Runtime | `tensormap_and_ringbuffer` | Simpler onboard Runtime |
| NPU 隔离 | `task-submit` | 两卡任务必须持有设备锁 |

最终证据由采集工具重新生成；本表只是启动基线，不替代运行 artifact。

## 3. 运行环境

当前仓库使用独立虚拟环境，不复用参考 Simpler checkout 的 `.venv`：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install pytest==9.1.1 ruff
python -m pip install --no-build-isolation --no-deps \
  "${SIMPLER_ROOT:-../simpler-mix-spmd-sync-start}"
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

安装前必须确认参考 checkout 的 `python/`、`src/`、`simpler_setup/`、`CMakeLists.txt`
和 `pyproject.toml` 没有未提交改动。采集结果必须记录实际 revision，不能只记录路径。

## 4. 固定进程模型

```text
launch_local
  ├── exec Attention endpoint → one task-submit assigned NPU
  └── exec WSE surrogate      → another task-submit assigned NPU
```

- Launcher 不初始化 Device Runtime，不持有设备资源。
- 两端使用独立 Python 进程、Device Context、日志目录和 Runtime 实例。
- 禁止初始化设备后 `fork`；端点必须通过独立解释器 `exec` 启动。
- Bootstrap 使用 TCP loopback，只传递 JSON 控制帧。
- 设备号来自 `$TASK_DEVICE`，两个角色必须绑定不同设备。
- 至少验证 Attention-first 和 WSE-first 两种启动顺序。

## 5. 控制面与数据面边界

阶段 0 控制面允许：

- `HELLO/CAPABILITIES`
- `READY`
- `HEALTH`
- `STOP`
- generation 级错误

阶段 0 禁止：

- invocation 级 task descriptor
- Tensor payload 或 output
- invocation 级 completion
- Host 逐 slot credit 或逐任务唤醒

阶段 0 没有 invocation 热路径。Launcher 仍需统计每类消息和 payload 字节，为阶段 1
复用；若出现禁止消息，C0 直接失败。

## 6. 初始能力盘点

| 能力 | 当前判断 | 阶段 0 证据要求 |
| --- | --- | --- |
| NPU Device 初始化/释放 | 待运行验证 | 两个独立 L2 Worker 初始化并正常 close |
| NPU Device Memory allocate/free | API 待映射 | Runtime 调用点和所有权说明 |
| Memory register/unregister | 未知 | 实际 backend 调用与失败语义 |
| Export/import remote handle | 高风险 | A5 VMM handle 或其他已执行路径；不能使用 Host 指针代替 |
| NPU→surrogate data operation | 阶段 1A | Device Memory 数据正确性和 backend identity |
| surrogate→NPU data operation | 阶段 1A | Device Memory 数据正确性和 backend identity |
| Device-side submit/completion | 阶段 1B | Device trace、doorbell 和 completion 证据 |
| Visibility fence/flush | 未知 | 明确 API、方向和作用域 |
| Backend identity/counters | 未知 | 查询接口或 trace/counter 来源 |

已知限制：Simpler 当前 `host_tcp` remote buffer 使用 Host 侧 session storage，只能作为
控制面和生命周期参考，不能作为 Device Memory、C1 或 C2 证据。A5 URMA 代码存在但受
能力宏门控，同样不能在未运行前标记为可用。

## 7. 阶段 0 交付与通过条件

- [ ] 环境和能力采集 artifact 完整，未知项明确标记。
- [ ] Manifest、handle、buffer、queue、descriptor 和 completion 契约冻结。
- [ ] 两个独立进程能分别初始化不同 NPU 并完成 READY/STOP。
- [ ] `SIMULATION` 与 `HOST_LOCAL`、`NETWORK_REMOTE` 严格区分。
- [ ] Host 消息计数、Device 日志和 Runtime 清理证据可回收。
- [ ] 失败、超时、generation、drain 和重复 close 行为已定义并测试。
- [ ] 阶段 1 T01～T12 测试入口、优先级和停止条件已记录。

阶段 0 收口时更新本清单，并附脱敏后的 `environment.json`、双方 manifest、
`transport.json` 和 `result.json`。任何未满足项都必须保留为阻塞或高风险，不能用
NPU surrogate 结果替代真实 WSE 结论。
