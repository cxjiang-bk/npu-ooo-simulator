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

## 2026-08-23：论文形态 StableHLO 前端闭环

### 本轮完成

- 新增 `src/npu_ooo/frontend/stablehlo_codegen.py`：从 canonical frontend graph 生成可检查的 textual StableHLO module，并保留 `source_graph_id`、frontend 和 operator count provenance。
- `StableHLOAdapter` 扩展为通用常见 dot 维度解析：支持 rank-2、rank-3 batching、隐式 activation leading dimensions、共享二维 RHS batch broadcast、`rhs_transposed` 和 `transpose_dims`。
- 新增 `SoftmaxFusionPass` 与 `LayerNormFusionPass`，使 generated StableHLO 的 primitive chain 在 graph compiler 中恢复成语义算子；RMSNorm 复用现有 `RMSNormFusionPass`。
- `CompiledArtifact` 保存 `stablehlo` 和 `source_frontend` 两个可选中间产物；新增 through-StableHLO compiler API。
- CLI `compile-model --through-stablehlo` 已跑通 attention block，并按 staged output 写入 `00_frontend/generated.mlir`、`stablehlo_module.json` 和 `source_frontend_import.json`。
- 旧的 rank-3 dot rejection regression 已更新为 batched matmul import regression；新增 TorchExport round-trip 等价性测试。

### 结果

```text
attention_block (B=1,S=4,H=8,tile=4)
semantic operators: 13
tiles:             33
primitive tasks:   134
TISA instructions: 112
analytical cycles: 590
```

直接 TorchExport 和 StableHLO round-trip 路径的语义算子类型、tile 数、TISA 数和 primitive task 数一致；metadata/provenance 会记录不同 frontend 来源，因此完整 JSON hash 不要求相同。

### 验证

- `PYTHONPATH=src python3 -m unittest discover -s tests -q`：81 tests passed，11 skipped。
- `python3 -m compileall -q src tests examples`：通过。
- `git diff --check`：通过。
- CLI smoke：`compile-model --torch-module examples.torch_models:attention_block --through-stablehlo` 成功生成 staged artifacts 和 swimlane。

### 当时边界（已由下一节的官方 StableHLO 接入取代）

- 这一阶段的 generated StableHLO 仅是 dependency-light textual subset；现已降级为显式 `textual` regression backend。

## 2026-08-23：官方 StableHLO dialect/parser/verifier

### 已完成

- 从 OpenXLA StableHLO 官方 `dev-wheels` 安装并验证
  `stablehlo-1.12.1.1751868740+6f7b4ab8-cp312-linux_x86_64`；真实导入路径是
  `mlir.ir` 与 `mlir.dialects.stablehlo`，dialect 需显式注册。
- 新增 `src/npu_ooo/frontend/stablehlo_official.py`：官方 generator 输出合法
  StableHLO，adapter 通过官方 MLIR parser/verifier 校验并导回统一 FrontendImport。
- 修复轻量文本过去不符合官方语义的部分：reduce 使用 reducer region，reduction
  默认移除轴，keepdim 通过 `broadcast_in_dim` 表达；参数/标量 broadcast 显式化；
  transpose 使用正式 permutation；RMSNorm 使用 mean 而非未经归一化的 sum。
- compiler/CLI 增加 `official|auto|textual` backend。正式默认是 `official` 且失败即报错；
  `auto` 的回退状态和原因进入 manifest，避免实验静默使用 fallback。
- `--stablehlo-file` 以及 `compile_stablehlo_text/file/module()` 同样默认官方验证，不再
  只有 `--through-stablehlo` 使用 verifier。
- 当前机器 Python 3.12 用户 site-packages 已安装官方 wheel；默认 Python 3.14 不兼容
  cp312 wheel，真实命令使用 `/usr/bin/python3.12`。

### 已验证

```text
official StableHLO import + parse + verify: passed
official examples/stablehlo/matmul.mlir compile: passed
official attention CLI: 13 ops / 33 tiles / 112 TISA / 134 tasks / 590 cycles
official/textual attention structural counts: identical
```

### 尚未完成

- 官方 StableHLO wheel 不等于 PyTorch exporter。项目 generator 继续提供当前宽覆盖
  legalization；真实 torch-xla exporter 已接入 Matmul、attention micrograph 和完整
  attention block；torch-mlir 尚未接入。
- 当前 official importer 的支持范围仍是一结果、静态 shape 和本项目算子集合；完整
  tuple/control-flow/dynamic-shape/layout/quantization 不在本轮完成范围。

### 最终回归

```text
Python 3.14（无 torch/StableHLO）：86 tests passed，16 skipped
Python 3.12（torch 2.9.1 + official StableHLO + torch-xla）：86 tests passed
compileall（Python 3.14/3.12）：passed
git diff --check：passed
```

## 2026-08-23：torch-xla 官方 exporter 插件

### 已完成

- 验证 PyPI `torch-xla==2.9.0` 提供 cp312 manylinux wheel，并与本机
  `torch==2.9.1+cu128` 在 PJRT CPU backend 下兼容。
- 新增 `TorchXLAStableHLOExporter`，直接调用官方
  `exported_program_to_stablehlo()`，记录 exporter version、StableHLO bytecode size/hash，
  并用独立 OpenXLA StableHLO bindings 再做 parse/verify。
- compiler API 与 CLI 新增 `stablehlo_exporter=project|torch-xla`；torch-xla 强制搭配
  `stablehlo_backend=official`，任何 export/import unsupported 都直接失败。
- 增加 `examples.torch_models:attention_micrograph` 三输入 factory 和真实 regression。

### 已验证

```text
torch-xla Matmul export: passed
torch-xla attention micrograph: 3 semantic ops / 36 tiles / 104 TISA / 132 tasks
direct TorchExport 与 torch-xla 路径的 semantic/tile/TISA/task 数：一致
torch-xla CLI dynamic_ready_queue: 559 analytical cycles
```

### 后续 importer 工作

- [x] torch-xla `nn.Linear` 的 flatten/dot/bias/unflatten 已恢复为 rank-3 semantic
  BatchedMatmul 加 broadcast Add，显式 weight transpose 继续 fold 到 Matmul metadata。
- [x] `[1, 4, 8]` 的直接 row-wise LayerNorm 和 `[2, 4, 8] -> [1, 8, 8]` 的展开形态
  均已从多结果 `stablehlo.batch_norm_training` 加 affine 恢复；不满足 reshape/feature
  等价约束时不做错误泛化。
- [ ] 一般多结果消费、复杂 MLIR region、layout、动态 shape constraint、更广模型导出和
  TISA device scheduler 仍是后续阶段。

## 2026-08-23：torch-xla 完整 attention block recovery

### 已完成

- 官方投影层保留 `batch_norm_training` 的 `epsilon` 和 `feature_index`，为每个 MLIR result
  建立独立身份；未使用的 mean/variance 可被安全省略，一旦后续 op 或 return 消费次要
  result 就显式报错。
- 新增 `RecoverStableHLOLayerNormPass`：只在直接 row-wise 或
  `[1, prod(outer), hidden]` reshape 形态、internal scale/offset 为 1/0 且外部 affine
  shape 匹配时恢复 LayerNorm；batch=2 的完整 block 已加入回归。
- 新增 `RecoverStableHLOFlattenedLinearPass`：恢复共享 flatten 输入和四组
  dot/bias/unflatten，重建 rank-3 batch/M/N/K 维度，再由既有 transpose fold、lowering、
  TISA 和 backend 继续处理。
- recovery 只消费官方 verifier 通过后的 StableHLO，不读取 source TorchExport graph，
  不会静默切换 project exporter。

### 验证

```text
torch-xla attention block: 13 semantic ops / 33 tiles / 112 TISA / 134 tasks
direct TorchExport 与 torch-xla 的 semantic op 多重集、tile/TISA/task 数：一致
Python 3.14：86 tests passed，16 skipped
Python 3.12 + torch/StableHLO/torch-xla：86 tests passed
compileall（Python 3.14/3.12）：passed
git diff --check：passed
```

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

## 2026-08-22：按阶段整理输出 artifact

### 已完成

- 新增 `src/npu_ooo/trace/layout.py`，统一定义 `00_frontend` 到 `07_trace` 八个阶段目录、文件映射、输出目录 README 和 `artifact_index.json`。
- 所有通用 JSON/CSV/SVG/PNG/Graphviz exporter 自动写入规范阶段目录；普通 benchmark、`compile-model` 和 sweep case 共用同一布局。`sweep-two-mm` case 现在也会保存模型、算子图、schedule/tile 和 backend graph，而不只保存最终仿真结果。
- 顶层保留 `manifest.json`、`summary.json`、`artifact_index.json` 和 `README.md`；原有平级文件名以相对符号链接保留，兼容现有 CLI/tests/脚本。
- README 的项目目录和运行输出说明已改为中文，并加入从 frontend 到 trace 的完整目录树、每层职责、典型查看顺序和 sweep 说明。
- `docs/architecture.md` 同步记录 staged artifact schema，避免文档仍描述旧的 flatten 输出。

### 验证

- 全量 `PYTHONPATH=src python3 -m unittest discover -s tests -q`：62 tests passed。
- 普通 `two-mm`、`compile-model` 和 `sweep-two-mm` smoke 均生成八个阶段目录；`artifact_index.json` 可列出规范文件，兼容符号链接可正常被旧测试读取。
- `git diff --check` 通过。

## 2026-08-22：明确前端自动编译推进计划

### 本轮结论

- 当前单算子和模型 block 冒烟仍主要由 `benchmarks/*.py` 手动构造 `ModelSpec/OperatorGraph`，再调用固定的 `default_*_schedule` 和专用/registry lowering；它们是稳定的 backend/simulator baseline，不是从真实 PyTorch 算子自动编译的前端。
- `compile-model` 已能从 Canonical JSON 经过 `FrontendImport -> Schedule/Tile -> TISA -> BackendArtifact`，但 JSON 本身已经是规范化输入，尚未覆盖真实 `torch.export` 导入和完整 pass pipeline。
- 阶段 9 拆分为 9.1-9.7：依赖与输入契约、TorchExport/ATen 导入、Canonicalization PassManager、decomposition/fusion、自动 schedule/tile、TISA/backend codegen、真实前端 CLI 与回归。
- 第一轮前端范围冻结为静态 shape、推理、小型真实 PyTorch 模型，优先 RMSNorm、LayerNorm、Linear/Matmul、ResidualAdd、Softmax 和 attention micrograph；暂不承诺完整 ATen、训练、动态控制流或完整 StableHLO/MLIR 工具链。

### 本轮变更

- 更新 `task_plan.md`，补充阶段 9 前端自动编译子计划、输入输出和验收标准。
- 本轮未修改实现代码，未改变现有 benchmark、simulator 或输出格式。

## 2026-08-22：前端真实导入与首个 composite pass

### 已完成

- 扩展 `TorchExportAdapter`：支持传入 `model_id`/`variant`/`shape_environment`，保留 `torch.export` 与 GraphModule provenance，并记录 graph inputs/outputs、parameter/activation metadata、frontend target、input occurrences 和 JSON-safe constant args。
- 扩展 FX/ExportedProgram 导入边界：处理 `placeholder`、`get_attr`、`call_function`、`call_method`、`call_module` 和 `output`；对 matmul、reduce、norm、softmax、elementwise 的基础 target 做语义映射；不再静默丢弃非 `call_function` 节点。
- 新增 `src/npu_ooo/compiler/passes.py`：`PassManager`、`CanonicalizeGraphPass` 和第一版 `RMSNormFusionPass`。RMSNorm pass 识别显式 `mul -> reduce -> add(epsilon) -> rsqrt -> mul`，生成语义 RMSNorm，并保留 source op/fusion provenance。
- `compile_operator_graph` 已在 schedule/lowering 前统一执行 PassManager，并把 pass diagnostics 写入 `CompiledArtifact`；新增 `SchedulePlanner` 入口，当前封装既有确定性 heuristic 并标记 `source=automatic-planner`。
- 新增公开 API：`compile_torch_module` 和 `compile_torch_exported_program`，真实模型无需先手工落成 Canonical JSON 即可进入统一 compiler pipeline。
- 新增 frontend contract/fusion/compile-to-TISA 测试；全量测试从 62 增至 64，均通过。

### 验证

```text
PYTHONPATH=src python3 -m unittest discover -s tests -q: 64 tests passed
python3 -m compileall -q src tests: passed
git diff --check: passed
fake ExportedProgram RMSNorm: rmsnorm -> 2 tiles -> 6 TISA instructions -> 10 primitive tasks
```

### 当前限制

- 当前环境 Python 3.14 未安装 `torch`、`executorch` 或 `stablehlo`，因此还没有运行真实 `torch.nn.Module -> torch.export` API smoke；已用 duck-typed ExportedProgram graph 契约覆盖相同导入逻辑。
- RMSNorm fusion 当前只覆盖静态二维、显式算术模式；native LayerNorm tuple output、复杂 broadcast、动态 shape 和完整 attention fusion 留待后续 pass。
- 真实 probe 已确认 `aten.softmax` 可以进入现有 lowering；`aten.linear` 现在能正确识别为 matmul 并保留 bias input，但在 bias decomposition 完成前会显式失败，避免静默丢失 bias；affine `native_layer_norm` 和包含 transpose 的完整 MultiheadAttention 同样暂未宣称支持。

### 真实 PyTorch smoke

- 发现系统 Python 3.12 环境已提供 PyTorch 2.9.1，使用 `PYTHONPATH=src python3.12` 验证真实 `torch.export`。
- `torch.nn.Module` RMSNorm 展开实际使用 `aten.mean`，补充 `mean -> reduce` 映射；真实 target 使用 `aten::mul`/`aten::rsqrt`，fusion target matching 统一规范化 `::`/`.`。
- 真实端到端结果：`rmsnorm` semantic op、2 个 tile、6 条 TISA instruction、10 个 primitive payload task，`CompiledArtifact.validate()` 通过。
- 增加 `test_real_torch_rmsnorm_compiles_to_tisa`；无 torch 环境自动 skip，Python 3.12+torch 环境已实际通过。

### 本轮遇到并修复的错误

| 错误 | 原因 | 修复 |
|---|---|---|
| `no registered lowering for operator type 'aten.mean'` | PyTorch 2.9 RMSNorm 导出使用 `aten.mean`，导入器只映射了 `sum` | 将 `aten.mean`/`aten::mean` 归一到 `reduce` |
| RMSNorm 未触发 fusion，最终 `mul_1` shape mismatch | 真实 FX target 使用 `aten::` 而 fusion matcher 只匹配 `aten.` | matcher 统一把 `::` 转换为 `.` |

## 2026-08-22：StableHLO frontend bridge 第一版

### 已完成

- 新增 `src/npu_ooo/frontend/stablehlo.py` 和公开 `StableHLOAdapter`：支持 `from_text()`、`from_file()`、`from_module()` 以及包含 `mlir/text` 的 payload wrapper。
- StableHLO adapter 解析第一版 textual MLIR subset：`func.func` 参数、tensor type、constant、单结果 `stablehlo/mhlo` elementwise/reduce/dot/softmax operation 和 return；保留 frontend target、常量参数、graph inputs/outputs、entry point 与 parser provenance。
- StableHLO 图汇入同一个 `FrontendImport`，因此不需要修改 PassManager、SchedulePlanner、TISA 或 backend；StableHLO `multiply/reduce/add/rsqrt/multiply` 已复用 RMSNorm fusion。
- 增加 StableHLO textual RMSNorm fixture：导入后能生成 semantic RMSNorm，并通过自动 tiling、TISA/backend artifact validation。
- 更新 README、`docs/architecture.md` 和 `task_plan.md`，明确 StableHLO adapter 的职责和当前 subset 边界。

### 验证

```text
PYTHONPATH=src python3 -m unittest discover -s tests -q: 66 tests passed, 1 skip
PYTHONPATH=src python3.12 -m unittest tests.test_frontend_compiler -q: 10 tests passed
python3 -m compileall -q src tests: passed
git diff --check: passed
```

### 当时限制（历史记录）

- 当时本机没有 StableHLO/MLIR bindings；该限制已在 2026-08-23 的官方 StableHLO 接入中解除，`torch_xla` 仍未安装。
- StableHLO parser 目前只支持单结果和受控 textual subset；tuple result/native LayerNorm、复杂 region、layout encoding 和动态 shape constraint 仍需后续 adapter pass。

## 2026-08-22：StableHLO CLI 与真实 PyTorch 常用算子闭环

### 已完成

- StableHLO `dot_general` 现在从函数签名的结果侧读取 shape，并解析 compact/canonical contracting 与 batching dimensions；rank-2/rank-3 batched dot、共享二维 RHS broadcast 和可下沉的转置元数据会进入 canonical matmul lowering，真正不支持的维度布局仍在 frontend boundary 显式报错。
- 新增 `compile_stablehlo_text/file/module` API；`compile-model` 通过互斥的 `--graph-json`/`--stablehlo-file` 支持两种入口，并加入可直接运行的 `examples/stablehlo/matmul.mlir`。
- PyTorch `graph_signature` 参数分类已接入，parameter/buffer/user input 不再都记为普通 input。
- 新增 Linear decomposition：`aten.linear(x, W[N,K], bias[N]) -> Matmul(rhs_transposed) -> broadcast Add`；backend payload 对 RHS 使用 `load_transpose`。
- affine LayerNorm 支持 weight/bias tile load，LayerNorm 与 RMSNorm 支持 `[batch, sequence, hidden]` 多 outer dimensions。
- Softmax 保留 positional axis，并按物理 tensor axis 构造 region，支持多 outer dimensions 和非末轴 reduction。
- attention micrograph 的单用途 `K.transpose(-2,-1)` 自动 fold 到 batched Matmul；真实 `QK^T -> Softmax -> PV` 可从 `torch.nn.Module` 编译到 TISA/backend。
- `compile-model --torch-module MODULE:FACTORY --input-shape ...` 已接入真实 PyTorch CLI；仓库内 pre-norm attention block 示例生成 13 个 canonical operators、33 个 tiles、112 条 TISA instructions（其中 18 条 `load_transpose`）和 134 个 primitive tasks。
- `pyproject.toml` 新增 `torch` 可选依赖并锁定本轮真实 smoke 使用的 `torch==2.9.1`；未验证的 StableHLO/torch-xla 发行包没有被写成伪依赖。
- 同一真实 attention block 分别运行 `static_pipeline` 与 `dynamic_ready_queue`：Canonical Graph、TileGraph、TISAProgram 和 ExecutionGraph 的 SHA-256 完全相同；当前 minimal/small-shape 两者均为 590 analytical cycles，说明公平输入已固定，但该 case 本身没有表现出动态调度收益。
- PyTorch RMSNorm 与 StableHLO RMSNorm 已验证 canonical op、维度、tile 数和 TISA instruction 数一致。

### 验证

```text
PYTHONPATH=src python3 -m unittest discover -s tests -q: 78 tests passed, 8 skipped
PYTHONPATH=src:. /usr/bin/python3.12 -m unittest tests.test_frontend_compiler -q: 22 tests passed
python3 -m compileall -q src tests: passed
git diff --check: passed
```

### 仍待完成

- StableHLO 官方 parser/verifier、reducer region 与真实 torch-xla exporter 已接入；
  通用 tuple/layout/dynamic-shape smoke 尚未完成。
- 完整 Transformer block、Conv/ResNet 与模型加载 CLI 尚未实现；rank-3 Linear 已支持共享二维 weight/bias 的 batch broadcast，但更复杂的 batch broadcasting 尚未覆盖。
- SchedulePlanner 仍是确定性 shape-aware heuristic，不是 architecture-aware cost model；该时点 simulator 仍消费 backend primitive graph，阶段 9.8 已在后续完成 TISA target 接入。

# 2026-08-24：阶段 9.8 TISA Device Scheduler 启动

## 本轮目标

- 将 `TISAProgram` 从编译描述产物提升为 device scheduler 的真实输入。
- Static 按 TISA program order 发射且不 bypass；Dynamic 在 reception/dependency window 内选择 ready TISA instruction。
- 一条 TISA instruction issue 后才激活对应 backend primitive payload，禁止 primitive 跨 instruction 进入全局 OOO window。
- trace 同时记录 TISA ISSUE/COMPLETE 和 primitive START/COMPLETE；`compile-model` 默认使用 TISA target，旧算子命令继续保留 primitive baseline。

## 实施拆分

1. 加强 TISAProgram/BackendArtifact payload 完整性与依赖一致性校验。
2. 新增 TISA device event simulator 和 Static/Dynamic policy。
3. 接入 compile-model、summary/manifest 和双层泳道图。
4. 增加 micro golden case 与完整 attention 回归。
5. 下一批再重构 `TileGraph -> TISA -> backend primitive` 的 codegen 方向。

## 启动时基线

- 工作区已有上一轮泳道图图例/cycle 轴修改：`src/npu_ooo/trace/export.py`、`tests/test_simulator.py`。
- Python 3.12 全量回归：90 tests passed。
- attention project-StableHLO 路径：33 tiles / 112 TISA instructions / 134 primitive tasks / 590 primitive-baseline analytical cycles。

## 遇到的问题

| 问题 | 尝试次数 | 处理 |
| --- | ---: | --- |
| 强化 TISA dependency 校验后，Softmax/RMSNorm/LayerNorm 出现 dependency source 位于 consumer 之后 | 1 | 根因是旧 codegen 按 tile/primitive 顺序输出而非 TISA DAG 拓扑顺序；新增稳定拓扑排序，并保留 `source_program_order` provenance |
## 2026-08-24：阶段 9.8 TISA device scheduler（第一版完成）

- 修复 TISA codegen 的粗粒度分组：由 `resource + compute` 改为 `resource + primitive`，避免 Softmax reduction barrier 在合并后形成伪 TISA cycle；目标前端/仿真测试 42 项通过。
- 新增独立 TISA scheduler micro-test，覆盖 Static/Dynamic 共用 artifact、critical-path reorder、dependency completion、payload 本地顺序执行、window=1 和非法 payload 契约；5 项通过。
- `compile-model` 默认调度目标切换为 `tisa`，保留 `--scheduler-target primitive` 兼容 baseline；新增 `tisa_instructions.csv` 和 TISA/primitive 双层泳道输出。
- 首次 CLI 回归失败：`tisa_instructions.csv` 将 `TaskTiming.task_id` 展开为未声明列；已改为显式映射到 `tisa_id`，避免 CSV schema 混用。
- 完整 attention official-StableHLO smoke：13 semantic operators / 33 tiles / 120 TISA instructions / 134 primitive tasks；Static=876 cycles，Dynamic critical-path=564 cycles。
- Static/Dynamic 的 `tisa_program.json` 与 `backend_artifact.json` SHA-256 分别完全一致；两边 `tisa_decision_count=120`，证明策略只改变 issue order。
- 双层 PNG 已人工检查：TISA/DMA、TISA/MXU、TISA/ARU 与 primitive DMA/MXU/ARU lane 同时可见，图例和 cycle 刻度正常。
- 目标回归 47 项通过；最终完整回归 `PYTHONPATH=src:. /usr/bin/python3.12 -m unittest discover -s tests -q` 为 95 tests passed。
- 显式 primitive 兼容 smoke 通过：manifest 为 `scheduler_target=primitive`，instruction timing 为空且不生成 `tisa_instructions.csv`。

## 2026-08-24：阶段 9.9 TISA-first Backend Codegen 完成

- 目标：将 compiler 顺序从 `primitive ExecutionGraph -> TISA` 调整为 `TileGraph -> TISAProgram -> backend payload`，同时保留现有 analytical lowerer 作为第一种 payload backend。
- 设计边界：`TISASemanticBuilder` 只读取 `OperatorGraph/ScheduleSpec/TileGraph/MachineConfig`，生成 semantic `op_type`、`TISAStage`、logical operands 和 typed dependencies；`AnalyticalBackendCodegen` 再调用现有 lowerer 生成 `ExecutionGraph` 并按稳定 `(tile_id, tisa_stage)` contract 绑定 payload。
- 第一批覆盖 `matmul/batched_matmul/gemv`、`elementwise/residual_add`、`reduce`、`softmax`、`rmsnorm`、`layernorm`；未覆盖的 primitive 必须在 backend binding 阶段显式失败，不能静默丢 task。
- 已完成 TISA-first 主路径：`compile_operator_graph()` 先构造并传递唯一 `TileGraph` 给 `TISASemanticBuilder`，再调用 `AnalyticalBackendCodegen`；旧的 `ExecutionGraph -> TISA` helper 已从 pipeline 移除。
- backend lowerer 通过稳定的 `(tile_id, primitive)` payload contract 绑定任务；`lower_mixed_graph()` 支持复用调用方 TileGraph，避免 backend 阶段重复 tiling。
- 修正 Softmax 的跨 tile sum barrier 与 LayerNorm 的 variance barrier；验证器改为检查 backend owner edge 在 TISA dependency DAG 中可达，避免绑定到某个 primitive lowerer 的直接边形状。
- `TISAInstruction.op_type` 现在表示 scheduler-visible stage（如 `load`、`matmul`、`reduce_sum`、`load_transpose`），`attributes.semantic_op_type` 保留 composite semantic operator family，兼顾论文的 stage issue 粒度与前端语义 provenance。
- 验证：`tests.test_frontend_compiler` + `tests.test_tisa_simulator` 共 38 项通过；`unittest discover -s tests -q` 共 95 项通过；`compileall` 通过。

## 2026-08-24：阶段 10.0 RuntimeSubmission v1

- 阶段 9.9 已提交为 `cedd49b`（`完成阶段9.9 TISA-first backend codegen`），提交前全量 96 tests passed。
- 新增 `src/npu_ooo/ir/runtime.py`：定义 `BufferBinding`、`RuntimeOperandBinding`、`RuntimeCommandChunk` 和 `RuntimeSubmission`，所有结构均有独立 validation 与 JSON 序列化。
- 新增线性 physical buffer allocator：按 resolved tensor shape/dtype 分配互不重叠的 DRAM 地址，支持外部 base address 和 alignment。
- `create_runtime_submission()` 将 logical TISA operands 绑定到 physical buffer；当阶段 9.9 的 logical tile shape 仍过粗时，显式记录 `size_source=buffer_capacity_fallback`，避免把近似范围伪装成 compiler 已给出的精确地址。
- static runtime 按 TISA program order 提交；dynamic-ready runtime 对 TISA dependency DAG 做 fanout-first ready-queue 拓扑排序，保证 dependency-before-consumer，但不改变编译产物和 device issue policy。
- `compile-model` 新增 `--runtime-policy`、`--runtime-chunk-size`、`--runtime-base-address`、`--runtime-alignment`、`--runtime-launch-latency` 和 `--runtime-synchronization-cycles`，并输出 `05_runtime/runtime_submission.json`；manifest 分别记录 runtime/device policy、chunk/buffer count、device 起止周期和端到端周期。
- RuntimeSubmission 已进入 TISA simulator 的 descriptor reception path：每个 chunk 完成后 descriptor 才可进入接收窗口；static device 保持接收顺序，dynamic device 只在已接收窗口内做 ready selection。
- SVG 新增 `Runtime/Submit[0]` 泳道与图例，Perfetto 使用独立 `pid=0` 输出 runtime submit 事件；`compile-model` 同时落盘 `07_trace/perfetto.json`。
- address scoreboard 在存在 RuntimeSubmission 时消费 concrete physical address range；同一 logical tensor 的不相交绑定可并发，重叠绑定仍产生 RAW/WAR/WAW stall。没有 RuntimeSubmission 时保留 logical `TileMem` fallback。
- 新增 `run_runtime_device_matrix()` 与 `compile-model --runtime-device-matrix`：四种 runtime/device policy 组合共享同一 `BackendArtifact` 和 physical allocation，分别输出 runtime submission、summary、Perfetto、SVG/PNG，并汇总 `sweep.csv/json`。
- dynamic runtime submission 使用 tile-affine fanout-first 拓扑顺序：保持 dependency-before-consumer，同时优先补齐已开始 tile 的 ready stage，避免 static device 的有限 in-flight tile window 被不完整 tile packet 占满而死锁。
- compiler TISA region legalizer v1 已从 semantic operator/tile bounds 计算 dense tensor 的 starts、tile shape、byte offset 和 byte size，覆盖 matmul/batched_matmul/gemv、broadcast elementwise、reduce、softmax、RMSNorm 和 LayerNorm affine 参数；two-matmul 324 个 TISA operands 全部使用 `size_source=tile_mem`，不再依赖 coarse fallback。
- Runtime buffer allocator 新增显式 `linear|lifetime_reuse` policy；`derive_tensor_lifetimes()` 计算 TISA first/last use，`derive_tensor_reuse_pairs()` 只允许 dependency DAG 上全序的 tensor pair 复用地址，并写入 lifetime provenance。2mm 已验证 `A -> D` 可复用而 `C -> D` 不可复用。
- Runtime command chunk 新增 `availability_cycle`；static runtime 对队头 descriptor 阻塞，dynamic runtime 在依赖允许且 request 已 available 时 bypass 独立 descriptor。summary/manifest 分开记录 `runtime_submit_busy_cycles`、`runtime_request_wait_cycles` 和 `runtime_submit_cycles`。

## 2026-08-24：阶段 11.0 Backend Contract 第一版

- 新增 `src/npu_ooo/backend/contracts.py`：定义 `BackendCapabilities`、`TimingProvider`、`CodegenBackend`、`EventBackend`、`SystemBackend` protocol；capability 校验覆盖 primitive/resource/memory 和 calibration status。
- 新增 `src/npu_ooo/backend/registry.py`：注册 `analytical` 与 `timing_table` 两个 timing provider；现有 `AnalyticalTimingModel`/`TimingTableModel` 通过 adapter 接入，不改变原有 timing 数值。
- `compile-model` 新增 `--timing-provider`，manifest 增加 `timing_provider` 和 `timing_backend_capabilities`；TISA simulator 对声明 capabilities 的 provider 执行显式校验，unsupported payload 直接失败。
- 110 项测试通过；当前阶段 11 仍缺少真实 EventBackend/SystemBackend adapter 和 SCALE-Sim/RTL 外部 timing 接入。
- StableHLO Matmul CLI smoke：15 TISA / 21 primitive / 3 physical buffers / 33 operand bindings / 5 chunks（chunk size=3）；runtime/device policy 均为 dynamic_ready_queue，device cycles 仍为 120。
- 全量回归：`PYTHONPATH=src:. /usr/bin/python3.12 -m unittest discover -s tests -q` 共 103 tests passed；`compileall` 与 `git diff --check` 通过。
- 官方 StableHLO Matmul + policy matrix smoke：3 TISA / 4 primitive / 1 command chunk；runtime submit=2、device start=2、device finish=54、device cycles=52、synchronization=5、end-to-end=59 cycles，四个 matrix cell 均生成独立 Perfetto trace。

## 2026-08-24：阶段 11.1 Analytical EventBackend

- 新增 `AnalyticalEventBackend`，把现有 `simulate_tisa_artifact()` 封装为正式 device event engine；scheduler policy、RuntimeSubmission、TimingProvider 和 SimulatorConfig 仍作为独立输入，未改变 TISA issue/payload run-to-completion 语义。
- 新增 `EventBackendRegistry`，默认注册 `analytical_event`；`schedule_tisa_program()` 改为通过 EventBackend 调用，不再直接绑定具体 TISA simulator。
- `compile-model` 新增 `--event-backend analytical_event`；runtime/device 四象限实验复用所选 backend，顶层和 case manifest 分别记录 event backend、capabilities、timing provider、codegen backend 与 runtime backend。
- analytical event capability 的 resource 范围由 `MachineConfig` 决定，不硬编码 DMA/MXU/ARU 名称；primitive capability 仍显式枚举并在运行前验证。
- 官方 StableHLO Matmul + policy matrix smoke 保持 3 TISA / 4 primitive / runtime submit 2 / device 52 / synchronization 5 / end-to-end 59 cycles，证明 adapter 迁移未改变数值。
- 相关 backend/TISA/runtime matrix 测试 16 项通过；下一步补齐 analytical CodegenBackend adapter/registry。

## 2026-08-24：阶段 11.2 Analytical CodegenBackend

- 新增 `AnalyticalCodegenBackend` 和 `CodegenBackendRegistry`，默认注册 `analytical`；adapter 包装现有 `AnalyticalBackendCodegen`，但对外严格实现 `CodegenBackend.lower(...) -> BackendArtifact`。
- `compile_operator_graph()` 现在固定先以 `TISASemanticBuilder` 构造 backend-independent TISAProgram，再从 codegen registry materialize payload；默认路径生成的 TISAProgram 与 BackendArtifact 和迁移前保持相同。
- 所有统一编译入口（canonical JSON、StableHLO、torch.export、torch-xla StableHLO、ModelInstance）均支持传入同一个 CodegenBackend；CLI 新增 `--codegen-backend analytical`。
- compiler attributes、顶层 manifest 与 policy-matrix case manifest 都记录 codegen/runtime/event/timing backend 和相应 capability metadata。
- 官方 StableHLO Matmul + policy matrix smoke 保持 3 TISA / 4 primitive / runtime submit 2 / device 52 / synchronization 5 / end-to-end 59 cycles；backend/frontend/runtime 目标回归 44 项通过。
- 下一步：为 matmul 引入可选外部 MXU timing adapter，未覆盖 primitive 必须显式保留 analytical fallback 或失败，并将 calibration source 写入 manifest。

## 2026-08-24：阶段 11.3 External Systolic-MXU Profile Adapter

- 环境检查确认本机没有安装 SCALE-Sim、Ramulator 或 gem5，也没有可复用的 SCALE-Sim checkout；因此没有伪造 runtime SCALE-Sim integration。
- 新增 `SystolicMXUProfileTimingProvider` 和 `systolic_mxu_profile` registry entry：读取 versioned `npu_ooo.systolic_mxu_profile.v1` JSON，按 Matmul task 的 `(batch,m,n,k)` 精确查找 duration/II。
- profile 只负责 MXU；non-matmul primitive 明确委托 `AnalyticalTimingModel`，未命中 Matmul 由 `unmatched_matmul=analytical|error` 控制。这样可研究局部校准对 TISA OOO trace 的影响，不会把 mixed timing 宣称为全芯片 RTL 结果。
- Timing provider 的 calibration status 现在进入 common primitive simulator 和 TISA simulator 的 `timing_calibration_status`/effective `calibration_status`；顶层与 policy matrix manifest 新增 `timing_provider_metadata`。MXU-only profile 的 effective status 显式写成 `mixed:<profile-status>+analytical-fallback`。
- 新增 `configs/timing/systolic_mxu_matmul_example.json`，其 source 明确标为格式示例而非测量。官方 StableHLO Matmul smoke 命中 `(1,4,12,8)` profile，保持 3 TISA/4 primitive，runtime submit=2、同步=5、总周期从 analytical 的 59 变为 64；profile 标记为 `source-derived-example`，仿真有效状态标为 mixed。
- 下一步：实现 SCALE-Sim exporter 或真实 RTL trace importer，将外部原始结果转换为此 profile；外部工具仍只校准 timing，不替代 TISA scheduler。

## 2026-08-24：阶段 11.3 Coverage 可解释性补充

- 增加 `timing_provider_coverage`：profile provider 按唯一 compiled `ExecutionTask` 集合统计 exact Matmul profile match、unmatched Matmul、unknown-shape Matmul 和 non-Matmul analytical fallback。
- coverage 不在 `timing()` 内计数，因为 critical-path 分析、候选筛选与实际 issue 会多次查询同一 task；直接计数会错误地把 scheduler 查询放大为执行量。
- common primitive simulator、TISA simulator、顶层 manifest 与四象限 case manifest 都输出同一 coverage 结构，为后续比较 dynamic/static 时的校准范围提供证据。

## 2026-08-24：阶段 11.4 RTL Completion Trace Importer

- 新增 `src/npu_ooo/backend/rtl_trace.py`，定义 `npu_ooo.rtl_completion_trace.v1` 的
  JSON/CSV 输入契约，记录 descriptor issue、compute start/done 和 PSB write completion。
- importer 支持 `compute_start_to_compute_done` 与 `descriptor_issue_to_done` 两种显式
  interval；默认前者，避免把 MXU instruction manager/PSB write 的端到端周期误当作
  isolated `matmul` primitive latency。
- 同一 `(batch,m,n,k)` 的重复样本支持 `max`、`median`、`p95` 聚合；II 优先读取显式
  值，否则从起始事件间隔推导，单样本回退到 duration。
- 新增 `import-rtl-trace` CLI、`docs/rtl-calibration.md` 和
  `examples/rtl/mxu_completion_trace.json`，生成已有 `systolic_mxu_profile.v1`，并在
  profile metadata 中保留 interval、aggregation、source、record/shape 数量和校准状态。
- 默认 unmatched Matmul 为 `error`；只有显式选择 analytical fallback 才允许混合 timing。
- 当前仍未接入 VCD/VPD/FSDB、VCS、Verilator 或 SCALE-Sim exporter；下一阶段是连接真实
  trace exporter，并校验 RTL signal boundary 与 TISA backend payload boundary。

## 2026-08-24：阶段 11.4 interval safety guard

- `SystolicMXUProfileTimingProvider` 读取 profile metadata 中的 `interval`；旧 profile 无该
  字段时保持兼容，compute interval 仍可用于 isolated matmul。
- provider 对 `descriptor_issue_to_done` profile 的 isolated `matmul` timing 查询显式失败，
  并在 capability metadata 中标记 `isolated_matmul_compatible=false`。
- 新增回归覆盖该拒绝路径；后续若要使用 descriptor-to-PSB latency，必须先把
  `CodegenBackend` payload 边界提升为 full MXU instruction。

## 2026-08-24：阶段 11.5 MXU VCS console-log adapter

- 新增 `src/npu_ooo/backend/rtl_log.py` 和 `import-rtl-log` CLI，解析仓库
  `rtl/unit_test/mxu/tb_mxu.sv` 的 `Prepared instruction`、`instruction accepted`、
  `Done Signal` 输出，生成 `npu_ooo.rtl_completion_trace.v1`。
- 将 testbench 的 `K1` 按显式 `k_per_tile`（默认 RTL `K0=8`）还原为 profile 的 `k`，并将
  shape、acceptance、done 的 provenance 写入 metadata/record attributes。
- 明确该 testbench 只有 descriptor-to-completion 边界；跳过 `END instruction accepted` 和
  `task_done=1` 控制事件，避免 FIFO 配对错位。没有制造 compute-start marker。
- synthetic VCS log -> trace -> profile 链路已验证；下一步仍是让 RTL testbench 导出
  matrix-array handshake/final compute boundary，或在 backend 中新增 full-MXU payload contract。
