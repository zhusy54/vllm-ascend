# Device 0 Native 与 PyPTO 对比诊断 Profile

本目录汇聚 2026-09-03 在 A3 device 0 上生成的一次有效 Native/PyPTO 同轮对比
profiling 产物。测试采用 TRB（`tensormap_and_ringbuffer`）、ACLGraph、TP1、B4/S8、
context position 8191、ratio4，并在同一个 fresh process 中按 ABBA 顺序执行。

## 主要入口

- `operator_comparison_analysis.md`：Native/PyPTO 算子、双流 overlap、AICPU/AICore
  envelope 和最终差距的完整闭合分析，应当先读这一份。
- `native_operator_breakdown.csv`：Native 38 个可见算子的逐项耗时。
- `gap_decomposition.csv`：两轮 MODEL 差距的机器可读闭合数据。
- `profiler_output/trace_view.json`：完整时间线，可直接用于 timeline 分析。
- `profiler_output/kernel_details.csv`：按 kernel 分解 Native 与 PyPTO device 时间。
- `profiler_output/task_time.csv`：device task 明细。
- `profiler_output/operator_details.csv`：框架算子明细。
- `profiler_output/ascend_pytorch_profiler_0.db`：PyTorch Profiler 数据库。
- `profiler_output/analysis.db`：CANN 分析数据库。
- `benchmark_run.txt`：产生本轮 profile 的完整运行日志，含 benchmark 配置、正确性门禁和
  正式性能样本。
- `SHA256SUMS`：从 `/tmp` 汇聚时记录的原始文件哈希。

原始 CANN CSV 使用 CRLF 行尾；本目录通过局部 `.gitattributes` 将它们标记为
binary，以确保 Git 不转换原始字节，也不产生无意义的逐行 diff。

## 结果口径

Profiler 在正式 `20 warmup + 100 sample` ABBA benchmark 完成后，额外执行一轮
不进入正式 percentile 的两轮诊断 ABBA。因此，timeline 用于拆分 kernel、并行重叠、
AICPU scheduler 与 AICore envelope；不能把 profiler 下的时延当成正式性能结果。

该进程的正式 device p50/p90/p99 为：

- Native：`704.220 / 711.950 / 722.980 us`
- PyPTO：`769.010 / 780.876 / 790.904 us`

诊断 profile 前后，output、六类 mutable state 以及 indexer raw K/scale 正确性门禁均
通过。

## 重要限制

这份 profile 对应优化过程中的中间实现，主要用于解释当时约 `64.790 us` 的 p50 差距；
它早于最终保留的 tile-major weight pack、提交顺序调整和 `idx_qr_dequant_rope` 融合，
因此不能代表最终源码的 kernel 拆分或最终性能结论。

最终源码的正式三进程稳态对比结果仍以同级目录中的
`../20260903_device0_trb_steady_state_performance.json` 为准。该最终结果没有配套的
torch_npu profiler timeline。

## 原始位置

汇聚前的 profiler 输出位于：

```text
/tmp/csa_profile_current_sqlite_20260903/
  decode_csa_tensormap_and_ringbuffer_aclgraph_3289559_1788390800801559439/
  6c42b4b8ccc347b39c28d0410ff20ed7_3289559_20260903071320804_ascend_pt/
  ASCEND_PROFILER_OUTPUT/
```

原始运行日志位于：

```text
/tmp/csa_trb_current_mature8191_aclgraph_profile_sqlite_round1.log
```
