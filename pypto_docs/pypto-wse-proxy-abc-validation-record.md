# PyPTO代理层A-B-C原型验证记录

## 1. 结论

2026-09-09在同一Host的Ascend 950PR设备0和1上完成V00～V06。
原型范围内结论为**PASS**：以第二张NPU作为WSE surrogate时，Bootstrap子系统可以建立
双进程、双Device的VMM P2P数据面；代理服务可以在Host只提交一次请求并等待最终结果的条件下，
由两个常驻AIV Kernel完成`A(NPU) -> B(WSE surrogate) -> C(NPU)`。

本结论不覆盖真实WSE、跨Host transport、PyPTO Compiler/Scheduler、并发请求或故障路径。
运行时`npu-smi`将设备0和1显示为`Alarm`，但本矩阵的ACL初始化、Kernel执行、P2P访问和释放均
成功；该健康告警需要在正式环境验证前单独排查。

## 2. 实现边界

- `BootstrapManager`协调manifest/handle交换和attach；两侧本地MemoryManager独占通信内存
  allocate/import/detach/free及Device Context生命周期。
- `EndpointBundle`只提供借用视图、窄化的`DeviceExecutionPort`和远端生命周期控制接口，
  不暴露allocate/import/map/free。
- `PseudoPyptoDistributedService`提供`initialize/execute/health/drain/close`；固定任务为
  `uint32 A=x+1, B=2*A, C=B+3`。
- `NpuSurrogateBackend`只启动、检查和停止resident B kernel，不进入逐请求数据路径。
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

- V00：24个隔离单元测试通过，覆盖固定契约、generation/lease/layout、API所有权、BUSY、
  close幂等和evidence fail-closed。
- V03两种启动顺序的输入和最终结果各传输1,118,272字节，结果逐元素匹配`2*x+5 mod 2^32`。
- V04输入和最终结果各传输1,768,000字节，request ID为1～100且A/B/C均执行100次。
- 每个generation两端各分配1个owned window、各建立2个mapping；退出后live mapping均为0，
  lease状态均为`RELEASED`。
- 每个generation Attention driver和surrogate B kernel各launch一次。
- V06 generation 1和2使用不同run ID及lease，并分别完成释放。

脱敏摘要见`pypto-wse-proxy-abc-evidence.json`。完整逐次运行产物默认写入gitignored的
`pypto_test/artifacts/`。

## 4. 复现命令

```bash
bash pypto_test/build_kernels.sh
.venv/bin/python -m pytest -q pypto_test/tests
.venv/bin/python -m pypto_test.collect_evidence \
  --attention-device 0 --surrogate-device 1 \
  --artifact-dir pypto_test/artifacts/full
```

本次Kernel SHA256：

- `abc_driver.o`: `b493bbfcd937d9bfa6eac489ec85ad2909727ac934a7d1be12849f1e341398e4`
- `b_service.o`: `1ce6ac848b89a55817742ca12460625453762aeaa7d853b31bfc50f52a733a09`

本次完整`summary.json` SHA256：
`039e28b0ed9c177f2d1e8f2bb3f876c80a56b97a8d215db43defe0d24d4d9fdf`。
