# 进度日志

## 2026-08-20：项目初始化

### 已完成

- 确认 GitHub active account 为 `cxjiang-bk`；
- 创建私有仓库 `cxjiang-bk/npu-ooo-simulator`；
- 克隆到 `/home/lora/OpenTPU/npu-ooo-simulator`；
- 确定项目独立于 `operator-opt`，旧仓库只作为参考；
- 编写总体架构、路线图、任务计划和研究发现；
- 明确第一条闭环是 2mm + configurable DMA/MXU + Static/Dynamic trace。
- 在 `codex/initial-research-plan` 分支提交规划文档，并创建 draft PR：
  `https://github.com/cxjiang-bk/npu-ooo-simulator/pull/1`。

### 当前状态

阶段 2/3 进行中：Model/Benchmark、Operator Graph、MachineConfig、Schedule/Tile/Execution IR 已有标准库实现；默认 2mm 已完成 tile 展开、matmul load/compute/store lowering，并接入三种 deterministic analytical scheduling policy。下一步是拆出独立 event simulator、trace CSV/PNG exporter、窗口/ROB 约束和 dynamic scoreboard。

### 创建/修改文件

- `README.md`
- `docs/architecture.md`
- `docs/roadmap.md`
- `task_plan.md`
- `findings.md`
- `progress.md`

### 本轮实现

- 新增 `src/npu_ooo/ir/schedule.py`：显式 tile factor、loop order、residency 和 stage schema；提供默认 2mm schedule。
- 新增 `src/npu_ooo/ir/tile.py`：展开实际 tile bounds，保留边界 tile，并建立跨算子 tile dependency。
- 新增 `src/npu_ooo/ir/execution.py`：primitive task、BufferRegion、读写地址范围和显式 predecessor graph。
- 新增 `src/npu_ooo/lowering/matmul.py`：2mm `load -> matmul -> store` lowering、K 方向累加依赖、producer-consumer region 依赖和 MAC/traffic 统计。
- 新增 `src/npu_ooo/scheduler/core.py`：sequential、static pipeline、dynamic ready queue，以及 task timing、stall 分解、Perfetto trace JSON。
- 新增 `src/npu_ooo/trace/export.py` 和 `src/npu_ooo/cli.py`：命令行运行 2mm，并导出 summary JSON、task CSV 和无依赖 SVG 泳道图。
- 新增 `tests/test_pipeline.py`：覆盖边界 tile、2mm 统计、跨算子依赖、policy 差异和 architecture profile 差异。
- `docs/model-layer.md`
- `docs/operator-taxonomy.md`

## 验证记录

| 检查 | 结果 | 状态 |
|---|---|---|
| GitHub repository created | `cxjiang-bk/npu-ooo-simulator` | pass |
| Local remote | `origin` points to GitHub repository | pass |
| Planning commit | `24d9497` on `codex/initial-research-plan` | pass |
| Draft PR | `cxjiang-bk/npu-ooo-simulator#1` targets `main` | pass |
| Old `operator-opt` planning files | no remaining changes from this session | pass |
| Implementation code | intentionally not created | pass |
| Paper benchmark/model-layer review | Table IX, compiler stack, TISA operand/instruction fields reviewed | pass |
| Python unit tests | 11 tests passed | pass |
| Python compileall | `src` and `tests` compile successfully | pass |
| `git diff --check` | no whitespace errors | pass |

### 当前 2mm analytical 结果

默认 `M=128,K=64,L=96,N=80`、fp16、minimal profile 下：

```text
sequential          25856 cycles
static_pipeline     21184 cycles
dynamic_ready_queue 17984 cycles
```

这些数值用于验证调度趋势和 trace 语义，manifest 中应标记为 `calibration_status=analytical`。

## 五问重启检查

| 问题 | 答案 |
|---|---|
| 我在哪里？ | 新项目阶段 0：Model IR、算子分类和契约冻结 |
| 我要去哪里？ | 2mm Static/Dynamic 可配置仿真第一里程碑 |
| 目标是什么？ | 从顶层算子到参数化 NPU cycle simulator 的完整研究栈 |
| 我学到了什么？ | 见 `findings.md` |
| 我做了什么？ | 见本日志和 `task_plan.md` |
