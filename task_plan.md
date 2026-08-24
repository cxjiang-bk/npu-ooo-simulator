# 任务计划：参数化 NPU OOO 编译与仿真框架

## 目标

独立构建从顶层算子到参数化 cycle simulator 的完整研究栈，在公平条件下比较 Static 与 TISA-like Dynamic tile scheduling，并生成总周期、stall 分解和泳道图。

## 当前阶段

阶段 9 已完成第一版自动前端、TISA device scheduler 和 TISA-first backend codegen。当前处于阶段 10：`TISAProgram/BackendArtifact -> RuntimeSubmission -> descriptor reception -> TISA device scheduler` 已形成第一条闭环，runtime launch/synchronization 与 device cycles 分层统计，默认零开销保持旧 baseline。下一步补齐同一 compiled artifact 上的 static/dynamic runtime x static/dynamic device 批量对照。阶段 4-6 的 analytical primitive-task baseline 继续保留为兼容对照。

阶段 9 的前端目标边界：先支持静态 shape、推理场景和小型真实 PyTorch 模型，覆盖 RMSNorm、LayerNorm、Linear/Matmul、ResidualAdd、Softmax 和 attention micrograph；不在第一轮承诺完整 ATen、动态控制流、训练语义或完整 StableHLO/MLIR 工具链。现有手写 benchmark 继续作为 canonical IR/lowering/simulator 回归基线，不能被自动前端重构破坏。

## 阶段状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | Model IR、MachineConfig、IR、trace 和 experiment schema | in_progress |
| 1 | Model IR、MachineConfig 与基础 Operator Graph IR | completed |
| 2 | 2mm tile instance 和 primitive lowering | completed |
| 3 | Static discrete-event simulator 与 trace | completed |
| 4 | Dynamic/TISA-like scheduler baseline | completed |
| 5 | Elementwise/Reduce/Softmax/Attention/Model presets baseline | completed |
| 6 | Architecture x Schedule x Policy 实验框架 baseline | completed |
| 7 | TileFlow/SCALE-Sim/RTL 校准 | pending |
| 8 | TISA Contract + ExecuTorch Frontend Adapter | completed |
| 9 | Compiler PassManager 与自动 TileGraph | in_progress |
| 10 | Runtime Submission 与 runtime/device 分层仿真 | completed |
| 11 | Hot-pluggable Timing/Event/System Backend | in_progress |
| 12 | 模型级自动编译与 TISA 实验矩阵 | pending |

## 阶段 0 检查表

- [x] 新建独立 GitHub 仓库；
- [x] 确定系统分层和项目边界；
- [x] 确定分阶段路线图和第一里程碑；
- [ ] 冻结 MachineConfig 字段和版本策略；
- [x] MachineConfig canonical JSON round-trip 与 CLI 外部配置入口；
- [x] 外部 MachineConfig sweep 的自定义 architecture label；
- [x] TimingTable JSON 覆盖入口与 analytical fallback；
- [ ] 冻结 Model/Benchmark IR 的 normalized schema；
- [ ] 冻结 `evaluation_scope=one_block|layer|full_model` 语义；
- [ ] 冻结 semantic operator 与 lowering primitive taxonomy；
- [x] 首批 semantic operator 的 lowering registry 与 mixed-graph handoff 契约；
- [x] LayerNorm mean/variance barrier lowering 与 micro-test；
- [x] workload sweep 的 dynamic priority 维度与 Static 配对 baseline；
- [x] 单头 attention `QK^T -> Softmax -> PV` mixed graph 与 CLI；
- [x] LayerNorm + attention + MLP + residual transformer block skeleton；
- [x] BERT/GPT-J/LLaMA2/DeepSeek 的 proxy model preset 与 `model-block` CLI；
- [ ] 冻结 Model/Operator/Schedule/Tile/Program/Runtime/Execution IR 的 normalized schema；
- [ ] 冻结 ExecutionTask dependency/address schema；
- [ ] 冻结 FrontendAdapter/TISAProgram/RuntimeSubmission schema；（RuntimeSubmission v1 已有第一版实现，完整 runtime timing schema 仍待冻结）
- [ ] 冻结 Runtime policy 与 Device Scheduler policy 的独立配置键；
- [ ] 冻结 CodegenBackend/TimingProvider/EventBackend/SystemBackend capability contract；
- [ ] 冻结 simulator event/tie-break 语义；
- [ ] 冻结 trace/summary/manifest schema；
- [x] 为 dual/triple pipeline 编写手算 golden case（dual reservation + drain 已由测试固定；stage_count 支持 triple）。

阶段 0 目前已落地 Model/Operator/MachineConfig、Schedule/Tile/Execution、trace/summary/manifest 基础 schema；核心 dual/triple golden case 已有，完整 normalized schema 版本策略和真实 ISA 契约仍待冻结。

## 阶段 9 前端自动编译子计划

| 子阶段 | 目标 | 主要产物 | 验收标准 |
|---|---|---|---|
| 9.1 前端环境与输入契约 | 固定 PyTorch/torch.export（后续再接 ExecuTorch/StableHLO）版本，定义真实模型输入、参数、shape constraint、dtype 和 provenance | frontend dependency lock、`FrontendImport` v1、错误诊断规范 | 一个真实 RMSNorm `torch.nn.Module` 能导出；缺依赖、动态 shape、unsupported op 都有明确错误 |
| 9.2 TorchExport/ATen 导入 | 将 `ExportedProgram/GraphModule` 的 placeholder、call_function、get_attr、output 映射到 Canonical OperatorGraph | `TorchExportAdapter` 完整实现、tensor/value/constant metadata | RMSNorm、LayerNorm、Linear、ResidualAdd、Softmax、Matmul 的 graph JSON 与手写 canonical graph 结构等价 |
| 9.3 Canonicalization PassManager | 统一 op name、dtype/shape、广播、常量、别名和数据边；保留 composite provenance；把低级 ATen 序列识别成 semantic operator 或显式 unsupported | pass registry、pass diagnostics、canonical IR verifier | 相同语义的不同 PyTorch 写法得到稳定 graph；每个 pass 可单独 dump 前后 IR |
| 9.4 Decomposition/Fusion | 建立受控的 ATen decomposition 和 composite pattern（先 RMSNorm/LayerNorm/attention），区分 semantic fusion 与 backend primitive expansion | decomposition registry、fusion groups、source-to-semantic mapping | RMSNorm 不依赖用户手写 graph；attention 能保留 QK/Softmax/PV 的语义边界；unsupported pattern 不静默丢失 |
| 9.5 自动 Schedule/Tiling | 用 op lowering registry 的元数据选择 tile factors、loop order、residency、stage，并生成真实边界 TileInstance | `SchedulePlanner`、`TileGraph`、shape-aware cost hooks | 改变输入 shape/tile size 会自动改变 tile 数、边界和地址；不再为每个 benchmark 写 `default_*_schedule` |
| 9.6 TISA/Backend codegen | 从 TileGraph/semantic lowering 生成 EU-bound TISA instructions、typed deps、logical address expressions 和 primitive payload | TISA program、backend artifact、payload map、compiler manifest | 每个 semantic tile 的 DMA/Vector/Tensor 分组、依赖和 payload 可验证；生成结果能复用现有 analytical simulator |
| 9.7 真实前端 CLI 与回归 | 增加 `compile-torch` 或 `compile-model --frontend torch-export`，统一输出 00-07 artifact；保留旧 benchmark 命令作为 baseline | CLI、golden fixtures、前后 IR dump、trace | 一条真实 PyTorch RMSNorm 命令完成 export -> graph -> tile -> TISA -> backend -> simulation，并与手写 baseline 对比周期/泳道 |
| 9.8 TISA Device Scheduler | 让 device scheduler 消费 TISAProgram，并在 TISA issue 后原子激活绑定 payload | TISA event simulator、Static/Dynamic TISA policy、双层 trace、payload contract tests | scheduler 决策数等于 TISA instruction 数；primitive 不跨 instruction 全局重排；Static/Dynamic 共享同一 program/payload |
| 9.9 TISA-first Backend Codegen | 调整 compiler 方向，使 TISAProgram 先于 backend primitive payload 生成 | semantic TISA builder、backend payload lowerer、capability verifier | completed：compiler 不从 ExecutionTask 反推 TISA；backend 复用入口 TileGraph；替换 payload backend 不改变 TISAProgram |

### 阶段 9 的实施顺序

```text
9.1 环境与契约
  -> 9.2 TorchExport/ATen 导入
  -> 9.3 PassManager/Canonicalization
  -> 9.4 decomposition + semantic fusion
  -> 9.5 自动 schedule/tile
  -> 9.6 TISA/backend codegen
  -> 9.7 真实前端 CLI 与回归
  -> 9.8 TISA device scheduler
  -> 9.9 TISA-first backend codegen
```

每个子阶段都必须保留 JSON graph、diagnostics 和可复现测试输入；先完成 RMSNorm 单算子闭环，再扩展到 LayerNorm/Linear/attention，最后才接模型 one-block preset。RuntimeSubmission 和 TISA device scheduler 不插入 9.1-9.7 的中间步骤，避免前端问题与后端调度问题混在一起。

### 阶段 9 当前进度

- 9.1 已完成第一版：`FrontendImport` 保留 model/variant/frontend/shape/provenance；Torch adapter 缺依赖时在 frontend boundary 报错；输入输出、parameter/activation metadata 和 constant args 已纳入导入结果。
- 9.2 已完成第一版并经过真实 PyTorch 2.9.1 验证：支持 placeholder/get_attr/call_function/call_method/call_module/output，使用 graph signature 区分 parameter/buffer/user input；已覆盖 Matmul/BatchedMatmul、Reduce、Elementwise、Norm、Softmax 和 transpose metadata。
- 9.2 StableHLO 正式分支已接入官方 OpenXLA bindings：`OfficialStableHLOGenerator` 输出合法 region/broadcast/dot/transpose，`OfficialStableHLOAdapter` 执行 dialect registration、MLIR parse/verify 并导回统一 pipeline；`StableHLOAdapter` textual subset 仅保留为显式 regression backend。真实 torch-xla exporter 已接入，torch-mlir、通用 tuple/layout/dynamic-shape 仍待完成。
- 9.3 进行中：`PassManager` 已统一 alias、推导 data edge，并加入 Linear decomposition 与 RHS transpose-to-Matmul fold；pass 前后独立 artifact dump 仍待实现。
- 9.4 已完成首批模式：RMSNorm composite fusion 支持静态多 outer dimensions；Linear 自动拆为 transposed Matmul + broadcast Add；StableHLO primitive chain 可恢复 Softmax/LayerNorm/RMSNorm；真实 attention micrograph 保留 `QK^T -> Softmax -> PV` 三个 semantic 边界。
- 9.5 已建立入口：新增 `SchedulePlanner`，当前封装既有确定性 shape-aware heuristic；后续再替换为 architecture-aware cost model。
- 9.6 已可复用：前端编译 API 生成 `TISAProgram`/`BackendArtifact`，payload ownership、单资源约束和跨 payload dependency 已纳入 verifier；codegen 方向仍待 9.9 重构。
- 9.7 基础入口完成：`compile-model` 支持互斥的 canonical JSON、StableHLO file 和 `--torch-module MODULE:FACTORY`，并通过 `--through-stablehlo` 选择论文形态 round-trip；StableHLO 默认 `official` 且不静默回退；均输出完整 staged artifact 和泳道图。标准模型 preset、动态 shape 与复杂输入 binding 尚未完成。
- 9.8 已完成第一版：`compile-model` 默认使用 TISA target，Static/Dynamic 消费同一 artifact，payload run-to-completion，输出 TISA/primitive 双层 trace；attention 回归为 120 次 TISA decision / 120 instructions。
- 9.9 已完成第一版：`TISASemanticBuilder` 只消费 graph/schedule/tile/machine，先构造 stage、logical operands 和 typed dependencies；`AnalyticalBackendCodegen` 再复用同一 `TileGraph` 生成并绑定 primitive payload。`BackendArtifact.validate()` 检查 payload ownership、单资源约束、TISA 依赖可达性和未绑定 task；覆盖 matmul/batched_matmul/gemv、elementwise/residual_add、reduce、softmax、rmsnorm、layernorm。前端+TISA 回归 38 项、全量回归 95 项通过。
- 10.0 已完成第一版：新增 `BufferBinding`、`RuntimeOperandBinding`、`RuntimeCommandChunk`、`RuntimeSubmission` 和线性 allocator；command chunk completion 驱动 descriptor reception，runtime/device policy 独立，launch/synchronization latency 分层进入 summary、manifest、SVG 和 Perfetto。启用 scoreboard 时优先消费 runtime physical address range。
- 10.1 已完成第一版：`--runtime-device-matrix` 在一次编译结果和一次 physical allocation 上运行 static/dynamic runtime x static/dynamic device 四种组合；每个 cell 输出独立 submission/summary/trace，顶层汇总周期和相对 static/static speedup。
- 10.2 已完成第一版：compiler TISA region legalizer v1 已覆盖 matmul/batched_matmul/gemv、broadcast elementwise、reduce、softmax、RMSNorm 和 LayerNorm（含 affine 参数），将 dense starts/shape 转为 byte offset/size；runtime 新增 descriptor `availability_cycle`、static 队头等待、dynamic bypass、`runtime_request_wait_cycles`，以及基于 TISA dependency proof 的 `lifetime_reuse` allocator。动态 shape/非 dense layout 仍显式记录 fallback。
- 10.3 下一步：把 request availability 从静态 JSON 映射扩展为 runtime event/state provider，并校准 buffer reuse 与 memory-bank/port timing；随后进入阶段 11 hot-pluggable backend contract。
- 11.0 进行中：新增 `BackendCapabilities`、`TimingProvider`、`EventBackend`、`SystemBackend`、`CodegenBackend` protocol 和 timing provider registry；`compile-model` 可选择 analytical/timing_table provider，manifest 写入 capability/calibration metadata，TISA simulator 在 provider 声明 capability 时执行显式校验。

阶段 4-6 的“completed”表示基线闭环完成，不表示后续功能不再扩展。下一轮工作必须保持这些 baseline 的输入/输出兼容，并以自动前端和 runtime/backend 分层作为增量演进。

## 第一里程碑

```text
2mm
  -> tile graph
  -> configurable DMA/MXU backend
  -> sequential/static dual/dynamic dual
  -> total cycles + stall breakdown
  -> Perfetto trace + PNG swimlane
```

验收必须覆盖两个 architecture profile，并证明 Static/Dynamic 只改变 scheduler policy。

当前已有两个 architecture profile 的 analytical cycle 对比、CSV/SVG/PNG/Perfetto 输出、ROB/window/queue 指标、显式 static stage reservation、runtime address scoreboard 和 occupancy timeline；`sweep-two-mm` 已能批量生成 architecture × policy × window × ROB 结果。混合 decoder block 已完成第一条跨算子 lowering 闭环；真实 MXU/memory timing 校准仍待后续提交。

## 下一阶段关键问题

1. ExecuTorch Core ATen 到 Canonical OperatorGraph 的最小公共字段是什么？
2. `TISAInstruction` 如何同时表达 OpType、Operand、UnitMap、typed Deps 和 runtime-bindable address expression？
3. `TISAInstruction -> BackendArtifact(descriptor + payload) -> ExecutionTask` 如何保持 scheduler 在 tile 粒度决策？
4. Runtime command-buffer chunk、launch latency 和 software ready queue 如何加入总周期而不改变 device graph？
5. Backend plugin 是只替换 TimingProvider，还是需要替换完整 EventBackend？
6. TISA 论文中的 tile address/range 依赖应在 compile-time graph、runtime binding 还是 device scoreboard 之间如何分工？
7. Latency model 的 analytical、source-derived 和 RTL-observed 状态如何进入配置与 manifest？
8. 当前 NPU ISA 中哪些指令应视作一个 primitive task，哪些需要拆成 issue 和 completion 两个事件？

## 已做决策

| 决策 | 理由 |
|---|---|
| 新建 `cxjiang-bk/npu-ooo-simulator`，不在 `operator-opt` 上实现 | 研究后端需要独立的契约、测试和演进节奏，旧仓库只作为参考 |
| 后端以 MachineConfig 驱动 | 支持不同 NPU、资源数量、queue/window、latency 和 bandwidth 探索 |
| Static/Dynamic 共用 graph 和 simulator | 将性能差异严格归因到 scheduler policy |
| 第一条闭环使用 2mm | 同时具备 producer-consumer pipeline 和可手算规模，适合验证依赖与 overlap；Model IR 仍从第一天保留 |
| 在 Operator Graph 上增加 Model/Benchmark IR | 论文 benchmark 横跨 CNN/encoder/decoder、prefill/decode 和不同 batch/seq，单个算子图无法表达这些 workload 语义 |
| semantic operator 与 lowering primitive 分离 | 保留 Attention/Softmax/Norm/MoE 的调度语义，同时允许硬件 timing 拆成 vector/reduction/transfer tasks |
| TISAInstruction 独立于 TileInstance 和 ExecutionTask | TileInstance 只描述 tile bounds；TISAInstruction 携带 OpType、Operand、TileMem、AccessType、UnitMap 和 typed Deps；ExecutionTask 是 TISA 之后的 backend-specific primitive，不应成为唯一 scheduler IR |
| 论文中的 dynamic scheduler 归入 device backend | 论文虽称 runtime scheduler，但明确给出 AI-core 7--9 cycle dispatch、per-unit WQ/IQ/in-flight table 和 RTL synthesis；host runtime 只负责 descriptor/command stream 接口，核心 reorder/issue 是硬件行为 |
| TISA 是 virtual/high-level ISA，但具有硬件消费的 ISA 契约 | 它不是最终 MXU/Vector/DMA 微指令，也不是纯 compiler IR；Epoch 有 binary encoding，TISA metadata 由硬件 scheduler 直接读取，backend 再 lower 到 per-unit execution ISA |
| BackendArtifact 同时保留 descriptor 和 execution payload | backend-specific codegen 可在 runtime 前产生 per-unit binary/primitive template，但硬件 scheduler 只读取 TISA descriptor 并以 tile 为单位触发关联 payload；不能让 primitive 自行进入全局 OOO window |
| 使用 GraphTemplate + GraphInstance | 避免重复 block 展开成巨型 graph，同时保留 layer/template provenance |
| Trace 同时输出 cycle-native CSV/JSON 和 Perfetto JSON | 前者适合测试与数据分析，后者适合交互式泳道观察 |
| Conv2D 后置 | halo、padding、layout 会过早扩大 lowering 复杂度 |
| 第一版不宣称 cycle-accurate | timing model 需要经过外部模型和 RTL observation 分层校准 |
| scheduler policy 与 event backend 分离 | policy 只选择 ready task；queue、ROB、II、资源占用和 completion wake-up 由 simulator 统一处理 |
| `SimulatorConfig` 覆盖 MachineConfig runtime capacity | 便于直接 sweep instruction queue、ROB、dependency window、in-flight tile，而不修改编译图 |
| address scoreboard 作为可选 runtime layer | 基于 active `BufferRegion` 生成 RAW/WAR/WAW issue stall，COMPLETE 后释放范围；不改写默认 graph，方便和 compile-time dependency 做公平对照 |
| ExecuTorch 作为第一模型前端 | `torch.export()`/Core ATen graph 已经规范化 PyTorch 输入、shape constraint 和 backend partition；先解决自动模型导入，不立即承担完整 MLIR/IREE 集成成本 |
| Runtime 与 Device Backend 分离 | Runtime 负责 buffer/address binding、command submission 和软件动态行为；Device Backend 负责已提交 task 的 static/dynamic issue、ROB、scoreboard 和 cycle timing |
| TISAProgram/RuntimeSubmission 作为新边界 | Compiler 输出可复用的语义 tile instruction 和逻辑地址表达式；Runtime 绑定物理地址并分批提交；Device Backend 不反向修改图或 schedule |
| Backend 热插拔 | 以 `TimingProvider`、`EventBackend`、`SystemBackend` 分层；当前 analytical backend 是默认实现，SCALE-Sim、Ramulator/DRAMSys、RTL/Verilator 和 gem5/SALAM 作为可选实现 |
| Artifact 输出布局 | 编译/仿真输出按 `00_frontend` 到 `07_trace` 分阶段保存；顶层保留 manifest/summary/index，旧文件名用相对符号链接兼容 | 既便于按处理链路定位问题，又不破坏已有脚本 |
| 四种 runtime/device 组合必须可比较 | `static runtime + static/dynamic device` 与 `dynamic runtime + static/dynamic device` 分离报告，避免把软件提交收益误归因于 TISA issue |
| 没有单一外部 simulator 直接替代当前 backend | gem5 更强在 full-system，Gemmini/VTA 更强在具体 NPU，SCALE-Sim 更强在 MXU，Timeloop 更强在 mapping；当前 TISA tile OOO 语义继续由本项目维护 |

## 暂不做

- 完整 ISA binary encoder；
- 完整 Conv2D data-layout/halo 优化；
- RTL/UVM 集成；
- 未校准 energy 常数；
- 将 TileFlow aggregate pipeline cycle 当作 event trace。
- 在 ExecuTorch adapter 完成前继续增加 benchmark-specific graph builder。
- 在 TISAInstruction/typed dependency schema 冻结前继续扩大 primitive lowering 覆盖。
- 把 runtime submission order 和 device issue order 写成同一个时间线或同一个 policy。
- 让外部 timing backend 改写 compiler graph，破坏 Static/Dynamic 公平性。
- 把当前 analytical timing 或 address prepass 宣称为真实 TISA/RTL scoreboard。
- 把 DeepSeek-R1-16B 未经配置证据直接假设为 dense 或 MoE。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| 旧仓库规划追加补丁首次锚点不匹配 | 1 | 使用稳定尾部锚点追加，之后按用户要求完整移除本轮追加内容 |
| artifact 验证命令包含 `rm -rf` 被执行策略拒绝 | 1 | 不清理目录，直接由确定性导出覆盖同名 artifact |
| LayerNorm barrier 测试错误假设首 tile 完成均值 | 1 | 按实际 M 外层/N 内层 tile 顺序断言每行最后一个 reduction tile |
| Attention tile count 测试把三个阶段相乘 | 1 | 按 QK、Softmax、PV 三个独立 operator 的 tile 数求和，默认小图为 12 |
| Transformer block handoff count 测试把 tile/task 数混同 | 1 | 依赖计数按 root-memory 相交的 store/load 边统计，默认 skeleton 为 8 |
| timing-config 使用文档占位路径导致 FileNotFoundError | 1 | 添加仓库内 smoke table，README 改用真实路径，CLI 捕获 ValueError 输出简洁错误 |
| 模型 preset sweep 默认维度过大导致事件仿真耗时过长 | 1 | 为 `sweep-workloads` 增加 `--model-tokens/--model-sequence/--model-head-dim/--model-intermediate` 覆盖；大规模 proxy 维度改为按模型分批运行 |
| TISA resource-only compute grouping 产生 group-level cycle | 1 | 按 `resource + primitive` 分组并稳定拓扑排序；禁止未经 DAG contraction 验证的 payload fusion |
| TISA CSV 展开 `task_id` 与 `tisa_id` schema 冲突 | 1 | 显式映射 instruction timing 字段，不直接展开 primitive-oriented `TaskTiming.to_dict()` |
| 文档检索命令中的 Markdown 反引号被 shell 当作命令替换 | 1 | 后续 shell pattern 不使用反引号，改用单引号或去掉该检索项 |

## 备注

## 2026-08-23：TorchExport -> StableHLO -> TISA round-trip

- 新增 `StableHLOGenerator`/`StableHLOModule`，从 `FrontendImport` 生成依赖轻量的 textual StableHLO primitive module。
- 新增 `compile_frontend_import_through_stablehlo()`、`compile_torch_exported_program_through_stablehlo()` 和 `compile_torch_module_through_stablehlo()`，明确执行 `TorchExport -> generated StableHLO -> StableHLOAdapter -> PassManager -> TISA`。
- StableHLO parser 已支持 rank-2/rank-3 `dot_general`、显式 batching dimensions、rank-N activation 乘共享二维权重的 RHS batch broadcast、末两维转置元数据和 reducer/transpose attributes。
- 新增 `SoftmaxFusionPass`、`LayerNormFusionPass`；StableHLO primitive chain 会恢复为 semantic Softmax/LayerNorm，RMSNorm 复用既有 fusion pass。
- `compile-model --through-stablehlo` 输出 `00_frontend/generated.mlir`、`stablehlo_module.json` 和 `source_frontend_import.json`，其余阶段继续使用同一套 graph/tile/TISA/backend/simulator。
- attention round-trip 初始 primitive baseline 曾验证为 13 semantic operators、33 tiles、112 TISA instructions、134 primitive tasks、590 cycles；修复 composite 分组后当前为 120 TISA instructions/134 primitive tasks，TISA scheduler static=876、dynamic=564 cycles。
- 当前 round-trip 的 graph metadata/provenance 与 direct path 不逐字节相同，这是预期的 frontend provenance 差异；语义类型、shape、tile count 和 TISA count 已回归。
- textual emitter/parser 仍不是完整 StableHLO/MLIR compiler，不支持 tuple result、复杂 region、layout、动态 shape 约束和完整 XLA legalization。

## 2026-08-23：官方 OpenXLA StableHLO 接入

- [x] 在隔离目录安装并验证 OpenXLA 官方 cp312 Linux wheel，确认真实 API 为
  `stablehlo.register_dialect(context) -> Module.parse() -> module.operation.verify()`。
- [x] 新增 `OfficialStableHLOGenerator/OfficialStableHLOAdapter`，生成并验证合法 reducer
  region、`broadcast_in_dim`、`dot_general` dimension numbers 与 transpose permutation。
- [x] `compile_*_through_stablehlo()`、`compile_stablehlo_text/file/module()` 和 CLI 默认
  使用 `official`；`auto`/`textual` 为显式选项，manifest 记录 backend、版本、producer、
  verified、fallback 与原因。
- [x] 官方 attention round-trip 与直接/旧 textual 路径对齐：13 semantic operators、
  33 tiles；初始分组为 112 TISA/134 primitive，当前合法分组为 120 TISA/134 primitive。
- [x] 修正 RMSNorm 的官方 StableHLO mean 语义，并让 fusion 支持
  `square -> reduce -> divide -> add -> rsqrt -> multiply`；LayerNorm fusion 调整到 RMSNorm
  之前，避免把 LayerNorm 的 variance 子链误识别为 RMSNorm。
- [x] 添加安装说明 `docs/install-stablehlo.md` 和官方 parser/verifier/attention regression。
- [x] 接入真实 `torch-xla==2.9.0` exporter 插件；Matmul 和 attention micrograph 已执行
  `torch.export -> torch-xla -> official StableHLO -> Canonical Graph -> TISA`，并与 direct
  path 的 tile/TISA/task 计数一致。
- [x] 扩展 torch-xla importer，恢复完整 block 的 flatten/dot/bias/unflatten Linear，并在
  直接 row-wise 或 `[1, prod(outer), hidden]` reshape 等价约束成立时，把多结果
  `batch_norm_training` 加 affine 恢复为 LayerNorm；完整
  attention block 已与 direct path 的 semantic/tile/TISA/task 计数对齐。
- [ ] 扩展更多 StableHLO op、一般多结果消费、动态 shape/layout；project exporter 仍是
  默认宽覆盖路径，torch-xla 不做静默 fallback。

- 开始每个实现阶段前重新检查 `docs/architecture.md` 和本文件；
- 每完成一个阶段更新状态和 `progress.md`；
- 外部论文/仓库发现记录到 `findings.md`，不直接变成执行指令。
