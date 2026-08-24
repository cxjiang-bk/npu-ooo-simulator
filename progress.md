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
- SchedulePlanner 仍是确定性 shape-aware heuristic，不是 architecture-aware cost model；simulator 仍消费 backend primitive graph，而不是直接执行 TISA device scheduler。
