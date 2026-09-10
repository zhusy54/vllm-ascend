# PyPTO代理层A-B-C原型验证记录

## 1. 结论

2026-09-09在同一Host的Ascend 950PR设备0和1上完成V00～V06。
原型范围内结论为**PASS**：使用第二张NPU仿真WSE Device时，Bootstrap子系统可以建立
双进程、双Device的VMM P2P数据面；代理服务可以在Host只提交一次请求并等待最终结果的条件下，
由两个常驻AIV Kernel完成`A(NPU) -> B(WSE) -> C(NPU)`。

本结论不覆盖真实WSE、跨Host transport、PyPTO Compiler/Scheduler、并发请求或故障路径。
运行时`npu-smi`将设备0和1显示为`Alarm`，但本矩阵的ACL初始化、Kernel执行、P2P访问和释放均
成功；该健康告警需要在正式环境验证前单独排查。

## 2. 实现边界

- `BootstrapManager`只编排三阶段初始化；`HostControlRpc`隐藏进程/TCP实现，
  `AscendVmmMemoryProvider`独占ACL初始化、VMM allocate/import/map/peer access/release。
- 初始化明确拆为`launch_wse_host()`、`launch_npu_host()`和`build_communication()`；实测事件
  顺序为`wse_host_ready -> npu_host_ready -> communication_built`。
- `EndpointBundle`只注入NPU进程自己的local/peer shared Device地址、layout、lease和远端生命周期
  控制；不暴露allocator、H2D/D2H、kernel launch或release。
- `PseudoPyptoDistributedService`提供typed `initialize/execute/health/drain/close`；
  `NpuExecutionBackend`自行分配本地输入/输出内存并直接执行H2D/D2H；固定任务为
  `uint32 A=x+1, B=2*A, C=B+3`。
- `WseBackend`属于伪PyPTO，拥有本地lifecycle/report内存并启动、检查和停止resident B kernel；
  不进入逐请求Host控制路径。
- Host逐请求路径只有input H2D、request descriptor/signal H2D、final signal/completion D2H和
  final payload D2H；A/B中间数据和B completion均不经过Host。

## 3. 测试结果

| Case | 启动顺序 | generation | 请求数 | payload | A/B/C计数 | 错误 | Host中间字节 | 释放 |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| V01 | Attention-first | 1 | 0 | 无 | 0/0/0 | 0 | 0 | CLEAN |
| V01 | WSE-first | 1 | 0 | 无 | 0/0/0 | 0 | 0 | CLEAN |
| V02 | Attention-first | 1 | 1 | 4 KiB | 1/1/1 | 0 | 0 | CLEAN |
| V03 | Attention-first | 1 | 4 | 64 B～1 MiB | 4/4/4 | 0 | 0 | CLEAN |
| V03 | WSE-first | 1 | 4 | 64 B～1 MiB | 4/4/4 | 0 | 0 | CLEAN |
| V04 | Attention-first | 1 | 100 | 混合 | 100/100/100 | 0 | 0 | CLEAN |
| V05 | Attention-first | 1 | 1 | 4 KiB | 1/1/1 | 0 | 0 | CLEAN |
| V06 | Attention-first | 1 | 1 | 4 KiB | 1/1/1 | 0 | 0 | CLEAN |
| V06 | Attention-first | 2 | 1 | 4 KiB | 1/1/1 | 0 | 0 | CLEAN |

补充证据：

- V00：39个隔离单元测试通过，覆盖typed API、三阶段初始化、HostControlRpc、外部VMM provider、
  本地/共享内存拆分、Device通信ABI、依赖方向、BUSY、close幂等和evidence fail-closed。
- V03两种启动顺序的输入和最终结果各传输1,118,272字节，结果逐元素匹配`2*x+5 mod 2^32`。
- V04输入和最终结果各传输1,768,000字节，request ID为1～100且A/B/C均执行100次。
- 每个generation两端各分配1个owned window、各建立2个mapping；退出后live mapping均为0，
  lease状态均为`RELEASED`。
- 每个generation Attention driver和WSE B kernel各launch一次。
- 每个generation只向PyPTO注入进程本地shared地址；两端PyPTO本地内存均在外部VMM释放前释放。
- Attention Host控制RPC仅包含BUILD_COMMUNICATION、START、HEALTH、DRAIN、CLOSE和RELEASE，
  不包含逐请求EXECUTE或A/B completion消息。
- V06 generation 1和2使用不同run ID及lease，并分别完成释放。

脱敏摘要见`pypto-wse-proxy-abc-evidence.json`。完整逐次运行产物默认写入gitignored的
`pypto_test/artifacts/`。

## 4. 复现命令

```bash
bash pypto_test/build_kernels.sh
../.venv/bin/python -m pytest -q pypto_test/tests
../.venv/bin/python -m pypto_test.validation.collect_evidence \
  --attention-device 0 --wse-device 1 \
  --artifact-dir pypto_test/artifacts/full
```

本次Kernel SHA256：

- `abc_driver.o`: `6a73992ccdd10bf34b6ac6e7f12c39087851bad024cf342715de5d42d5e9ca78`
- `b_service.o`: `12fbb6c350745d26885d404fc0d1cf416fa4606adc6e831ecdb489f6149ee331`

本次完整`summary.json` SHA256：
`7f6d6247e38c7aa8270954a12c55599ca8cb3a22264fc9630d77ae5f398b67eb`。
