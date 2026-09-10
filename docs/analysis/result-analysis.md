# 气泡、等待链与瓶颈分析

分析命令只读取已保存产物，不重新编译或仿真：

```bash
npu-ooo analyze \
  --run-dir out/attention-static \
  --compare-dir out/attention-dynamic
```

默认输出到 `<run>/08_analysis/`：

```text
analysis.json       结构化观测与推导
comparison.json     公平性、周期差和 control hash
report.md/html      可读报告与联合时间线
program_hierarchy.json
tisa_graph.dot / tisa_graph_dynamic.dot
```

## 统计口径

- EU busy 只使用 payload task 的物理 `[start,finish)`，不使用 TISA issue→complete；
- 每个实例的 global physical-work window 分为 startup、steady gap 和 drain；只有 steady gap
  默认标为可优化候选；
- Static 优先使用实际 `STATIC_*_WAIT` 区间；其他等待从 receive/issue、依赖完成和 bound
  descriptor 推导；旧 trace 无法确定的 blocker 标为 `unknown_from_saved_trace`；
- stall reason 可以重叠，不直接求和形成总周期分解；
- “最长等待”只是候选瓶颈，必须结合资源利用率、等待链和单因素实验验证。

等待记录指向具体 producer、condition、dependency kind、provenance；fence 还记录 buffer
slot。由此可以追踪：

```text
MXU bubble -> target TISA wait -> load producer -> allocation fence -> old consumer
```

## 比较与受控实验

Static/Dynamic 对比首先检查 `shared_workload_hash`、machine hash 和 timing provider。正文
workload 不同会给出警告，不将周期差解释为调度收益。

已提供的 Attention 控制成本实验：

- 默认 control：1432 cycles；
- 显式零成本：1408 cycles；
- 4-cycle control：1632 cycles；
- control width 4：1430 cycles。

该结果验证控制延迟会改变总周期；而小 Matmul 将 control width 从 1 改为 4 仍为 83 cycles，
说明“限制参数变宽”不自动带来收益。所有数值来自 analytical、uncalibrated 模型。
