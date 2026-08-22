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
- 新增 `sweep-workloads`：统一扫描 elementwise、LayerNorm、decoder fragment 等 workload 与 architecture/policy/window/ROB/tile-size，并为每个 case 导出完整计算图和泳道 artifact。
- 新增 MachineConfig JSON round-trip、`load_machine_config` 和所有 workload CLI 的 `--machine-config`，允许外部版本化架构文件替换内置 profile。
- sweep 在外部 MachineConfig 模式下允许自定义 architecture label，manifest 同时保留 label 和 stable machine hash。
- 新增 `TimingTableModel` 和所有 workload CLI 的 `--timing-config`；外部 duration/II table 可以覆盖 primitive timing，未命中 task 回退 analytical，结果 backend 写入 manifest/summary。
- 新增 `configs/timing/attention_probe.json` 可运行示例，并修复模块 CLI 对缺失 timing 文件的友好错误输出。
- `sweep-workloads` 增加 `--dynamic-priorities` 维度；LayerNorm priority demo 捕获 `critical_path` 慢于 `oldest_first` 的可复现实验。
- 新增单头 attention fragment 和 `attention` CLI：`QK^T -> Softmax -> PV` 复用 registry，保留矩阵乘/softmax 跨算子依赖；static=4520、dynamic critical-path=4532（analytical）。
- 新增 `transformer-block` skeleton：LayerNorm + 单头 attention + MLP + residual 串接为 9-operator mixed graph；默认 30 tiles/126 tasks/28 handoffs，static/dynamic critical-path 均 10540 cycles（analytical）。
- 新增模型层 `model-block` preset：BERT、GPT-J、LLaMA2-7B 和 DeepSeek-R1-16B 均可实例化为带 native metadata 的 one-block proxy；默认使用小 shape，`--tokens/--sequence/--head-dim/--intermediate` 可覆盖，DeepSeek 明确标注 dense/MoE 未确认且不包含 expert routing。
- `sweep-workloads` 已注册上述四个模型 preset，可在不改变 scheduler/backend 的前提下生成模型 × policy × window/ROB × tile-size 的配对 artifact。
- 模型 preset sweep 已验证：统一 `16x16x16x32` proxy shape 生成 16 个 case；按 preset 区分小规模 shape 的单独 Static/Dynamic probes 产生 BERT 1372、GPT-J 1900、LLaMA2 2528、DeepSeek 3392 cycles（均 analytical，不能替代真实模型 benchmark）。

### 验证

```text
37 tests passed
decoder-block static_pipeline: 9504 cycles, 30 tiles, 114 tasks
decoder-block dynamic_ready_queue: 9504 cycles, same graph and machine
cross-operator dependencies: 24
layernorm static_pipeline: 3808 cycles; dynamic_ready_queue (critical_path): 4696 cycles
small workload sweep demo: 6 cases (3 workloads x 2 policies), each with graph JSON and PNG swimlane
attention fragment: 12 tiles, 54 tasks, 8 cross-operator dependencies
transformer block skeleton: 9 operators, 30 tiles, 126 tasks, 28 handoffs
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

## 2026-08-21：ExecuTorch、Runtime 与热插拔 Backend 架构更新

### 本轮完成

- 重新定义总体架构为：`Frontend Adapter -> Canonical Operator IR -> Compiler PassManager -> Schedule/Tile -> CompiledProgram/TISA Command -> RuntimeSubmission -> Device Backend`；
- 选择 ExecuTorch/`torch.export()` 作为第一模型前端，暂不把完整 MLIR/IREE toolchain 引入第一版；
- 明确 `CompiledProgram` 与 `RuntimeSubmission` 的职责边界：前者保存逻辑 task/region/dependency/address expression，后者绑定 shape/state、物理地址、command buffer 和提交顺序；
- 将 Runtime software dynamic 与 TISA device dynamic 分开，规划四种组合实验：static/dynamic runtime × static/dynamic device；
- 将 backend 设计为 `TimingProvider`、`EventBackend`、可选 `SystemBackend` 三层热插拔接口；
- 保留当前 analytical discrete-event backend 作为默认 TISA device model，并规划 SCALE-Sim、Ramulator/DRAMSys、RTL/Verilator、Gemmini/VTA 和 gem5/SALAM 的组合接入边界；
- 更新 `README.md`、`docs/architecture.md`、`docs/model-layer.md`、`docs/roadmap.md`、`task_plan.md` 和 `findings.md`，新增阶段 8-12 的开发计划和验收条件；
- 本轮未修改实现代码，未改变现有 simulator 行为。

### 当前工作位置

```text
已有：手写 Model/Operator Graph -> TileGraph -> ExecutionGraph -> analytical device simulator
下一步：ExecuTorch -> Canonical OperatorGraph -> 自动 Compiler PassManager
随后：RuntimeSubmission -> 热插拔 Device Backend -> runtime/device 组合实验
```

### 验证

- 文档检查：架构图、IR 层次、runtime/device 边界、backend 分层和 roadmap 已同步；
- 代码/测试：本轮没有实现代码变更，因此未重新运行测试；后续阶段 8 开始前需先建立 frontend adapter micro-test。

## 2026-08-21：TISA 抽象层复核与 IR gap 识别

### 本轮确认

- 重新从论文原文核对 Table II/III、Section III-VI 和 Algorithm 1/2：TISA 是硬件消费的 tile-level scheduling-semantics ISA，不是最终 per-unit 微指令，也不是普通 compiler-only IR；
- 确认论文 dynamic scheduler 的关键实现位于 AI-core hardware layer：per-unit WQ/IQ/Fu、7--9 cycle dispatch 和 RTL synthesis；host software runtime 只负责 descriptor/command stream 入口；
- 确认论文 compiler 在 tile granularity 截止 lowering：`TISAInstruction` 之后才进入 backend-specific execution ISA；
- 确认 StableHLO 在论文中承担 semantic OpType 对齐，不只是 graph serialization；ExecuTorch adapter 必须保留 source/composite provenance；
- 识别当前设计的主要 gap：`TileInstance -> ExecutionTask` 跳过了结构化 `TISAInstruction`，导致 OpType、Operand(TileShape/TileMem/AccessType)、UnitMap、typed Deps 和 partial-ready condition 没有独立契约；
- 将目标路径修正为：`TileInstance -> TISAProgram -> hardware-like TISA scheduler -> backend primitive ExecutionTask`；
- 新增 `docs/tisa-alignment.md`，集中记录论文 TISA 字段、当前 IR 映射、抽象层次偏差和迁移顺序；
- 更新 `README.md`、`docs/architecture.md`、`docs/operator-taxonomy.md`、`docs/roadmap.md`、`task_plan.md` 和 `findings.md`，将 TISA semantic IR、backend expansion 和硬件/runtime 边界写入持久化文档；
- 本轮仍未修改实现代码。

### 环境检查

- 当前 Python 环境未安装 `torch` 和 `executorch`；本轮只完成论文/架构复核，阶段 8 实现前需单独固定依赖版本并运行 export smoke test。

### 当前设计结论

现有 Model IR、OperatorGraph、ScheduleSpec、MachineConfig、Static/Dynamic baseline 和 analytical event simulator 可以保留；下一阶段不能继续只扩展 primitive lowering，应先冻结 `TISAInstruction/TISAProgram` 和 typed dependency schema。

## 2026-08-21：阶段 8 前端与统一 compiler 第一版

### 已完成

- 新增 `ir/tisa.py`：冻结最小 `TileMem`、`TISAOperand`、`UnitMap`、typed `TISADependency`、`TISAInstruction`、`TISAProgram` 和 `BackendArtifact` schema；每条 TISA instruction 绑定一个 semantic tile，payload 通过 `tisa_id -> ExecutionTask tuple` 关联。
- 新增 `frontend/bridge.py`：`FrontendImport` 统一 framework bridge 输出；`JsonGraphAdapter` 作为无外部依赖的 canonical graph 入口；`TorchExportAdapter` 对 `torch.export` 做惰性导入和明确缺依赖错误。
- 新增 `compiler/pipeline.py`：从 Canonical OperatorGraph 自动生成默认 Schedule/Tiling、复用 lowering registry、构造一条 TISA instruction/semantic tile，并输出 `BackendArtifact`，保持现有 analytical ExecutionGraph 可供 simulator 回归。
- 修正 TISA codegen 的 EU 边界：同一 tile 内的 `DMA -> Tensor -> DMA` primitive 不再错误地打包为一条跨 EU TISA instruction，而是按连续 resource group 输出 `load<dma>`、`gemm<tensor>`、`store<dma>`；Vector/ARU 内部的 `reduce/exp/normalize` 仍作为一个 semantic payload。一个 tile 可有多条 TISA instruction，但每条只绑定一个 EU 类别，并通过 `semantic_tile_id` 保留 tile provenance。
- 新增 `compile-model` CLI：输入 `operator_graph.json`，自动导出 frontend/canonical/schedule/tile/TISA/backend/execution artifact 和泳道图；不再要求为该入口手写 benchmark-specific lowering 分支。
- 新增 `tests/test_frontend_compiler.py`，覆盖 JSON adapter、shape symbol resolve、TISA/payload 数量对齐和无 torch 时的 frontend 诊断。
- 更新 `README.md` 与 `docs/architecture.md`：明确 `torchxla` 是 framework bridge，StableHLO 是 bridge 后的 portable semantic graph IR，ExecuTorch/torch.export 是本项目可替代入口，不属于 backend 或 device scheduler。

### 验证

- `PYTHONPATH=src python3 -m npu_ooo.cli two-mm ...` 回归通过：minimal/static_pipeline 仍输出 204 primitive tasks、17984 analytical cycles。
- `PYTHONPATH=src python3 -m npu_ooo.cli compile-model --graph-json out/rmsnorm-dynamic/operator_graph.json ...` 通过：36 EU-bound TISA instructions、60 primitive tasks、3472 analytical cycles，并生成 `tisa_program.json`、`backend_artifact.json` 和 swimlane artifact。
- TISA 边界 smoke：2mm 的 60 tiles 生成 144 条 EU-bound TISA instructions；attention 的 DMA/Tensor/Vector groups 均通过 `CompiledArtifact.validate()`。
- `git diff --check` 通过。
- 当前环境没有安装 `pytest`；使用 `PYTHONPATH=src python3 -m unittest discover -s tests -q` 验证通过，当前共 62 tests；另以 CLI 和 Python smoke test 验证核心路径。

### 当前限制

- `TorchExportAdapter` 已建立边界，但当前环境没有 `torch`/`executorch`，且 torch FX metadata/operator overload 的版本兼容性尚未在本机验证。
- 目前 TISA program 是对既有 primitive lowering 的语义封装；device scheduler 仍消费 primitive ExecutionGraph。下一阶段必须实现 TISA target scheduler：先 issue TISA tile，再激活不可跨 tile 重排的 payload group。
- 当前默认 schedule 仍是 compiler 内的 deterministic heuristic，不是完整 MLIR/StableHLO pass manager；StableHLO adapter 和 composite semantic preservation 后续接入同一 FrontendImport。
