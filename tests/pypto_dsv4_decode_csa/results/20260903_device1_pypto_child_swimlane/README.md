# PyPTO Decode CSA 逐 AICore Child-Task 泳道

本目录保存 2026-09-03 在 A3 device 1 上，用当前最终 Decode CSA TRB program
采集的 Simpler chip-swimlane 结果。其目标是展开 PyPTO 外层 AICPU scheduler 所调度的
每个 child task，并按真实物理核显示，而不是再次绘制 CANN profiler 只能看到的
`simpler_aicpu_l1_exec_*` 和 `aicore_kernel_0` 外层区间。

## 直接查看

将 `pypto_child_task_swimlane.json` 拖入 Perfetto：

- `Worker View`：主视图；包含 `AIC_0` 至 `AIC_19`、`AIV_20` 至 `AIV_59`，每个矩形
  都是一个实际 child task；
- `Scheduler View`：同一批 task 的 dispatch、ready、执行与完成关系；
- `AICPU Scheduler`：4 条 scheduler 线程泳道；
- `AICPU Orchestrator`：orchestrator 阶段。

本次共记录 834 个 child task，覆盖全部 60 个 AICore：

| 项目 | 数值 |
| --- | ---: |
| AIC 物理核 | 20 |
| AIV 物理核 | 40 |
| AIC child task | 369 |
| AIV child task | 465 |
| child task 总数 | 834 |
| 每核 child task 数范围 | 10–21 |
| callable/function 数 | 41 |
| 结构依赖边 | 220 |

名称表包含最终融合 kernel `idx_qr_dequant_rope` 的 64 个实例，不包含已经被替代的
`idx_qr_proj_dequant` 和独立 `qr_rope`，因此这不是复用优化前的历史泳道。

## 文件说明

- `pypto_child_task_swimlane.json`：可直接导入 Perfetto 的完整逐核泳道，是本目录的
  主交付物；
- `chip_swimlane_records.json`：Simpler 从设备收集的原始 task 时间记录；
- `deps.json`：独立 dependency pass 生成的结构任务、tensor 和依赖边；
- `name_map.json`：runtime function ID 到 PyPTO kernel 名称的映射。

## 口径边界

PyPTO borrowed-device L1 按设计关闭 Simpler DFX，不能在 ACLGraph capture/replay 中直接
打开 chip-swimlane。因此本次让**同一个最终 TRB callable、相同静态
B4/S8/C8191 参数契约和相同 task DAG**在独立进程中走一次 L2 chip-swimlane。输入使用
生产 shape 的全零 fixture，`start_position=8191`，执行后做了全零输出校验。

这份数据可用于分析 child task 的真实执行核、并行排布、依赖和 scheduler gap；L2
采集的绝对总时延以及 DFX 开销不能当作 L1 ACLGraph 性能，也不能和 device 0 上的
Native/PyPTO CANN profile 数值直接相减。全面的 Native/PyPTO 外层对比 profiling 仍保留在
同级 `../20260903_device0_native_vs_pypto_diagnostic_profile/`。

## 复现

仓库内的 `../../a3_child_swimlane.py` 固定采用当前 TRB program 和上述 fixture 口径。
在 vLLM-Ascend 根目录、torch 2.12 环境中执行：

```bash
PTOAS_ROOT=/mnt/workspace/inductor/pto/PTOAS/install-v0.57-llvm21-cann9.2-clean \
LD_LIBRARY_PATH=/mnt/workspace/llvm-project-vpto-src/build-release-cann9.2-clean/lib:/mnt/workspace/inductor/toolchains/gcc15/lib:${LD_LIBRARY_PATH} \
.venv/bin/python -m tests.pypto_dsv4_decode_csa.a3_child_swimlane \
    --device 1 --start-position 8191
```

生成器会先在独立进程采集 `deps.json`，再执行不带 dep-gen 扰动的 clean timing pass，
最后调用 Simpler converter 生成 Perfetto JSON。
