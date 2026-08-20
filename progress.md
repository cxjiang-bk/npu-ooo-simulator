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

阶段 4 进行中、阶段 5/6 已启动：Model/Benchmark、Operator Graph、MachineConfig、Schedule/Tile/Execution IR 和独立 analytical event backend 已实现；2mm 已完成 tile 展开、matmul lowering、queue/ROB/window 约束、completion wake-up、CSV/SVG/Perfetto trace、显式 static stage reservation、runtime RAW/WAR/WAW range scoreboard，以及 stall/ROB/queue occupancy 指标。`sweep-two-mm` 已能批量扫描 architecture × tile size × policy × window × ROB；residual-add、row-reduce、softmax 和 RMSNorm 已完成独立 lowering/CLI，混合 lowering registry 和 decoder block fragment 已接入，下一步是 LayerNorm 和更广的 workload sweep。

## 2026-08-20：混合算子 lowering 与 PNG 泳道

### 已完成

- 新增 `LoweringRegistry`：按 semantic operator type 注册 Matmul、Elementwise、Reduce、Softmax、RMSNorm lowerer；scheduler 不增加算子分支。
- 新增 `lower_mixed_graph/lower_mixed_model`：对 heterogeneous Operator Graph 按拓扑逐算子 lowering，重新编号全局 `program_order`，并根据显式 `DataEdge` 和 root-memory `BufferRegion` overlap 注入跨算子 store -> load 依赖。
- 新增 `default_mixed_schedule`：为每个 semantic operator 生成 resolved tile sizes、loop order 和 topology stage。
- 新增 decoder one-block benchmark：`RMSNorm -> Matmul -> ResidualAdd`，支持 `decoder-block` CLI，生成完整 Model/Operator/Tile/Execution graph、summary、CSV、Perfetto、SVG/PNG 泳道。
- 新增 `write_png` trace exporter，使用本机 ImageMagick/librsvg 进行 SVG 栅格化；PNG 后端缺失时会给出明确错误。
- 新增 LayerNorm lowering/benchmark/CLI：显式 `reduce_sum -> layernorm_mean -> center -> reduce_sum_square -> layernorm` 双 barrier primitive DAG。

### 验证

```text
37 tests passed
decoder-block static_pipeline: 9504 cycles, 30 tiles, 114 tasks
decoder-block dynamic_ready_queue: 9504 cycles, same graph and machine
cross-operator dependencies: 24
layernorm static_pipeline: 3808 cycles; dynamic_ready_queue (critical_path): 4696 cycles
```

当前结果仍为 `calibration_status=analytical`。Decoder fragment 在 minimal profile 下 Static/Dynamic 恰好同周期；LayerNorm 则出现 dynamic critical-path 慢于 static 的反例。两者都说明动态机制、priority heuristic、window/ROB 和 barrier 结构必须作为独立实验维度。

### 2026-08-20：阶段 4 后端语义推进

- 新增 `StaticPipelineConfig`：支持 `stage_count`、stage offsets、modulo initiation interval 和按 task 的精确 issue reservation；CLI 支持 `--static-stage-offsets` / `--static-stage-ii`。
- 2mm lowering 为每个 task 保留 `iteration` 元数据，因而可直接生成 dual/triple stage reservation。
- 新增 runtime `AddressScoreboard`：只追踪 active task 的 `BufferRegion`，issue 前阻塞 RAW/WAR/WAW 冲突，COMPLETE 后释放并继续调度；不再通过改写 execution graph 模拟硬件 scoreboard。
- 统一输出 `stall_cycles_by_reason`、`stall_by_reason`、`pipeline_drain_cycles`、`queue_occupancy_timeline` 和观测到的 `address_hazards`。
- 新增 dual/triple reservation、RAW/WAR/WAW、sweep artifact、elementwise/reduce/softmax/RMSNorm 和 tile-size sweep micro-tests；当前测试总数 33，全部通过。
- 新增 `sweep-two-mm` CLI：每个 architecture/tile size/policy/window/ROB 组合独立写 manifest、summary、tasks、Perfetto 和 SVG，并在顶层汇总 CSV/JSON。
- 新增 `elementwise` benchmark 与通用 `lower_elementwise_graph`，验证 semantic operator 不依赖 matmul 专用 lowering。
- 新增 row-reduce partial chain 和 softmax composite lowering；动态 priority 现在支持 `critical_path` 与 `oldest_first`。
- 新增 RMSNorm `square -> reduce_sum_square -> rmsnorm` lowering，覆盖 decoder block 常见归一化路径。

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
- CLI 进一步导出 Model/Operator/Schedule/Tile/Execution Graph JSON、DOT 和 `operator_graph.svg`，把计算图 artifact 与时间线 artifact 明确分开。
- 新增 `src/npu_ooo/simulator/core.py`：独立的 analytical timing model 和离散事件 backend，实际处理 instruction queue、ROB、dependency window、in-flight tile、unit queue/II、completion wake-up 和 backpressure 指标。
- `schedule_execution_graph()` 现在只是 scheduler policy wrapper；`--dependency-window`、`--rob-entries` 等 CLI 参数可以覆盖运行时容量。
- 新增可选 address scoreboard prepass：根据 `BufferRegion` 增加 RAW/WAR/WAW predecessor，并通过 `--address-scoreboard` 记录到 manifest。
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
| Python unit tests | 15 tests passed | pass |
| Python compileall | `src` and `tests` compile successfully | pass |
| `git diff --check` | no whitespace errors | pass |

### 当前 2mm analytical 结果

默认 `M=128,K=64,L=96,N=80`、fp16、minimal profile 下，event backend（默认容量 `ROB=8`、`dependency_window=8`）当前为：

```text
sequential          25856 cycles
static_pipeline     17984 cycles
dynamic_ready_queue 17984 cycles
```

这些数值用于验证调度趋势和 trace 语义，manifest 中应标记为 `calibration_status=analytical`。

## 五问重启检查

| 问题 | 答案 |
|---|---|
| 我在哪里？ | 新项目阶段 4：event backend、runtime window 和 address scoreboard |
| 我要去哪里？ | 完善 static dual/triple pipeline，并接入可校准 MXU/memory timing |
| 目标是什么？ | 从顶层算子到参数化 NPU cycle simulator 的完整研究栈 |
| 我学到了什么？ | 见 `findings.md` |
| 我做了什么？ | 见本日志和 `task_plan.md` |
