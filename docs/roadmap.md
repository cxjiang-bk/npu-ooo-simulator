# 实施路线图

## 总体策略

先建立一个小而完整、可手算验证的 2mm 闭环，再把模型前端、统一 compiler、软件 runtime、TISA device scheduler 和热插拔 timing backend 分层推进。每个阶段都必须产生可独立验收的 artifact，避免把 runtime 提交开销、硬件 issue 语义和外部 timing 校准混成一个不可解释的总周期。

当前已经完成的是“手写 Model/Operator Graph -> TileGraph -> ExecutionGraph -> analytical device simulator”基线。下一阶段的重点不再是继续增加 benchmark-specific builder，而是把这条基线替换为可自动导入、可提交、可换 backend 的统一路径。

## 阶段 0：契约冻结

目标：确定 MachineConfig、Model/Benchmark IR、Operator/Schedule/Tile/Execution IR、Trace 和 Experiment manifest 的 schema。

交付物：

- MachineConfig schema 与校验规则；
- Model/Benchmark、Operator/Schedule/Tile/Execution IR 字段定义；
- scheduler policy 接口；
- event/summary/manifest schema；
- 两个手算 pipeline 示例。

验收：给定一张 dual-stage/triple-stage DAG，能够仅根据文档人工推导预期时间线；所有字段都能解释论文图中的 iteration、stage、尾部 drain 和总周期。

## 阶段 1：MachineConfig、Model IR 与基础 Operator IR

目标：实现通用架构配置、Model/Benchmark IR 和 Operator Graph IR。

首批 profile：

- `minimal`: 单 DMA、单 MXU、简单 SRAM；
- `lpu-derived`: 从已有 LPU 参数导入，但不依赖其 RTL 运行环境；
- `wide-mxu`: 修改 MXU 数量/shape/II 的探索 profile。

验收：

- profile 可序列化、校验并生成稳定 hash；
- ModelSpec 支持 graph template、重复 block、shape environment 和 prefill/decode phase；
- Table IX 的每个 model/config/phase 组合可以表达为独立 BenchmarkCase；
- Operator Graph 保存 model/layer/template provenance；
- 非法 memory parent、path 和 resource 引用在仿真前失败；
- 修改 queue、bandwidth、MXU shape 不需要修改 simulator 代码。

## 阶段 2：2mm Tile Graph

目标：跑通 `Model/Benchmark -> Operator Graph -> Schedule -> Tile Instance -> Execution Graph`。

交付物：

- Matmul lowering；
- 2mm benchmark；
- tile region/address 计算；
- DMA/MXU primitive task；
- RAW producer-consumer 依赖。

验收：

- tile 数量、边界和依赖可手算；
- task ID 和拓扑顺序可重复；
- aggregate MAC 和 transfer bytes 与参考模型对账；
- compiler 输出 execution graph JSON，尚不要求运行动态调度。

## 阶段 3：Static Simulator 与泳道图

目标：实现确定性离散事件 simulator 和静态基线。

策略：

- Sequential；
- Static dual-stage pipeline；
- Static triple-stage pipeline。

交付物：

- event engine；
- resource/queue/II model；
- events CSV、summary JSON、Perfetto trace；
- 论文风格静态泳道图。

验收：

- 串行、双资源重叠、队列满、pipeline drain micro-test 与手算一致；
- 2mm 在不同 bandwidth/MXU profile 下周期变化可解释；
- 静态泳道能显示资源空泡和等待原因。

## 阶段 4：Dynamic/TISA-like Scheduler

目标：在相同 execution graph 上加入动态 ready queue 和 completion wake-up。

递进实现：

1. dependency-ready + resource-ready；
2. configurable out-of-order window；
3. oldest/critical-path/locality priority；
4. address-range RAW/WAR/WAW scoreboard；
5. queue/ROB/backpressure。

验收：

- 复现 sequential/static dual/dynamic dual/static triple/dynamic triple 五种图；
- Dynamic 不越过真实数据依赖；
- window=1 退化为近似 in-order；
- 扩大 window 的收益和代价能从 trace/queue 指标解释；
- Static/Dynamic 只切换 policy，不改变 graph 或 latency。

## 阶段 5：模型与算子覆盖扩展

目标：验证后端不是 2mm 专用 simulator，并覆盖论文 benchmark 所需的模型结构。

顺序：

1. Elementwise/Reduce（elementwise/residual-add、row-reduce 已完成 lowering 闭环）；
2. ResNet bottleneck：Conv2D/Norm/Activation/Residual/Pooling；
3. Softmax/LayerNorm/RMSNorm composite lowering（softmax、LayerNorm 和 RMSNorm 已展开 composite primitive）；
4. Decoder block：已先接入 `RMSNorm -> Matmul -> ResidualAdd` 混合 fragment，并新增单头 `QK^T -> Softmax -> PV` attention fragment 与 LayerNorm + Attention + MLP skeleton；下一步扩展 QKV/Attention/MLP/RoPE/KV cache；
5. BERT/GPT-J/LLaMA2/DeepSeek benchmark templates；已新增明确标注 proxy/shape-only 的 `model-block` presets，真实模型算子仍待扩展；
6. Conv2D halo/layout 和 optional MoE routing 作为后续扩展。

验收：每种 P0 semantic operator 都有独立 lowering、micro-test 和至少一组 Static/Dynamic trace；模型层能实例化 CNN、encoder 和 decoder template；scheduler 中无模型或算子名称分支。

## 阶段 6：实验框架

目标：系统比较架构与调度参数。

实验矩阵：

```text
architecture profile
  x benchmark shape
  x tiling schedule / tile size
  x scheduler policy
  x queue/window setting
```

交付物：

- 一条命令运行对比实验；
- `sweep-two-mm` 批量扫描 architecture × tile size × policy × window × ROB；
- `sweep-workloads` 已能扫描多个算子/模型 fragment × architecture × tile size × policy × window × ROB；
- `sweep-workloads` 已注册 BERT/GPT-J/LLaMA2/DeepSeek proxy preset，并支持 `--model-*` shape override；
- 每个 case 独立 manifest；
- 汇总 CSV/JSON；
- 单 workload CLI 已统一输出 SVG 和 PNG 泳道图；
- total cycle、speedup、utilization、stall、drain、buffer/queue peak；
- 可复现的泳道图目录。

## 阶段 7：外部模型与硬件校准（研究准备，正式接入后置）

目标：明确外部工具的能力边界和校准数据格式；正式 backend 接入放到阶段 11 之后，避免在接口未冻结时绑定某个工具。

- TileFlow/Timeloop 对账 mapping、traffic 和 aggregate trend；
- SCALE-Sim 校准 MXU dataflow 和 systolic timing；
- VTA/Gemmini 对照 queue/dependency 语义；
- Verilator/RTL trace 校准 unit latency、II、buffer port/bank conflict；
- 最终接入当前 NPU ISA/profile。

任何未经 RTL/hardware observation 校准的数字，manifest 中保持 `analytical` 或 `source-derived` 状态。

## 阶段 8：TISA Contract 与 ExecuTorch 导入

目标：先冻结论文对齐的 TISA semantic instruction/backend artifact contract，再让 benchmark 从“手写 OperatorGraph”迁移到真实 PyTorch 导出图，同时不改变现有 TileGraph/ExecutionGraph baseline 契约。

交付物：

- `TISAInstruction`、`Operand`、`TileMem`、`UnitMap`、typed `Deps` 和 partial-ready condition schema；
- `TISAProgram`、`BackendArtifact`、descriptor-to-payload association schema；
- `FrontendAdapter` 抽象和 source provenance schema；
- ExecuTorch/`torch.export()` adapter；
- Core ATen 到 Canonical OperatorGraph 的规范化映射；
- shape constraint、dtype、layout、constant/parameter metadata 保留；
- source module/ATen provenance 和 composite semantic boundary 保留，禁止在 frontend 阶段把 Attention/Softmax/Norm 直接打散成 primitive-only graph；
- `compile-model` 或等价统一入口；
- 一个由 PyTorch module 自动生成的 2mm/MLP/attention micro graph。

首批支持的输入算子：

```text
aten.mm / aten.matmul
aten.add / aten.mul
aten.reshape / aten.transpose
aten.sum / aten.amax
aten.softmax
```

验收：同一个 PyTorch module 经过 `torch.export()` 和当前手写 benchmark，能够生成语义等价的 Canonical OperatorGraph；`OpType` 能映射到稳定的 semantic taxonomy，并保留 composite boundary；模型名称不出现在 scheduler 或 device backend 分支中；导入失败必须在 frontend boundary 给出明确诊断。

阶段 8 的 TISA schema 必须先于大规模 frontend/lowering 实现冻结。当前环境尚未安装 `torch`/`executorch`，实现时需固定依赖版本并先通过一个 export smoke test。

## 阶段 9：统一 Compiler PassManager 与自动 TileGraph

目标：从 Canonical OperatorGraph 自动完成 decomposition、shape/layout、fusion、memory planning、tiling 和 primitive lowering。

交付物：

- pass manager 和 pass-level diagnostics；
- ATen composite op decomposition；
- shape/layout inference 与 liveness metadata；
- 默认 tiling planner，输出现有 `ScheduleSpec`；
- region-aware tile dependency builder，替换过度保守的 all-to-all expansion；
- `TISAInstruction`/`TISAProgram` 语义层：`OpType`、`Operand(TileShape/TileMem/AccessType)`、`Attributes`、`UnitMap`、typed `Deps`；
- `BackendArtifact` 契约：TISA descriptor 与 backend execution payload/opcode/kernel reference 的稳定关联；
- `TISAProgram -> backend primitive ExecutionGraph` 的显式 lowering；
- 手写 benchmark builder 降级为测试 fixture，而不是生产入口。

验收：

- `torch.export -> OperatorGraph -> ScheduleSpec -> TileGraph -> ExecutionGraph` 一条命令跑通；
- scheduler 的 target mode 在 TISAInstruction 粒度 issue，primitive ExecutionTask 只作为 backend timing expansion；
- simulator 可以预生成 primitive template，但 tile issue 后才激活对应 payload，禁止 primitive 跨 tile 越界重排；
- 2mm、elementwise、reduce/softmax、LayerNorm 至少各有一条自动编译路径；
- 编译输出保留 model/layer/operator/tile provenance；
- Static/Dynamic 接收完全相同的 compiled graph；
- tile 数量、边界、MAC、traffic 和 region dependency 可与现有 golden case 对账。

## 阶段 10：Runtime Layer 与 Command Submission

目标：显式模拟软件 runtime 在设备执行之前的动态事务，并把 runtime latency 与 device cycles 分开统计。论文中的关键 dynamic tile scheduler 归入 device backend；host runtime 只负责 descriptor/command stream 的生成、绑定和提交。

交付物：

- `TISAProgram` 到 `RuntimeSubmission` 的 binding 过程；
- logical tensor region 到 physical buffer address 的 memory planner/allocator；
- static runtime 和 dynamic ready-queue runtime 两种提交策略；
- command-buffer chunk、queue depth、launch latency、event/synchronization 模型；
- TISA descriptor reception buffer，以及进入 device per-unit WQ/IQ 的提交路径；
- runtime trace lane 和 `runtime_submit_cycles` 等 summary 指标；
- 四种组合实验：static/dynamic runtime × static/dynamic device。

验收：

- runtime submission order 与 device issue order 在 trace 中分开可见；
- 改变 launch latency 或 command-buffer chunk 不修改 device task graph；
- 同一 compiled program 可以在不同 physical address allocation 下重放；
- 总周期能拆分为 runtime submit、device execution 和 synchronization 三部分。

## 阶段 11：Hot-pluggable Device Backend Contract

目标：把当前 analytical event simulator 变成一个默认 backend，并允许外部 timing/event/system backend 热插拔。

交付物：

- `TimingProvider`、`EventBackend`、`SystemBackend` 三层接口；
- `CodegenBackend` 接口：TISAProgram -> BackendArtifact；
- backend registry/factory 和配置文件选择机制；
- 现有 analytical backend 的无行为变化迁移；
- SCALE-Sim timing adapter 的最小实现（先覆盖 matmul/MXU）；
- timing source、backend name、calibration status 写入 manifest；
- backend capability validation，明确支持的 primitive/resource/memory 特性。

候选 backend 顺序：

```text
analytical event backend
    -> SCALE-Sim MXU timing
    -> Ramulator2/DRAMSys memory timing
    -> RTL/Verilator unit timing
    -> Gemmini/NVDLA/gem5-SALAM system integration
```

验收：切换 backend 不需要修改 compiler pass 或 scheduler policy；未覆盖的 primitive 必须显式回退或失败，不能静默混用未标定的数字；同一 trace schema 可以比较不同 backend。

## 阶段 12：模型级自动编译与 TISA 实验矩阵

目标：将自动前端、runtime 和 device backend 连接到论文模型 benchmark，并保持 proxy 与真实配置的边界清晰。

顺序：

1. decoder one-block：QKV、attention、RoPE、KV-cache、MLP、residual；
2. BERT encoder block；
3. ResNet bottleneck/stage，补充 Conv2D halo/layout；
4. GPT-J/LLaMA2 prefill/decode；
5. DeepSeek-R1 配置确认后的 dense/MoE routing；
6. full-model repetition、persistent state 和 request-level runtime。

实验矩阵扩展为：

```text
frontend
  x model/benchmark case
  x compiler pass/tile schedule
  x runtime policy/submission chunk
  x device scheduler policy
  x MachineConfig
  x timing/event backend
```

验收：每个 case 具有独立 frontend/compiler/runtime/device manifest；能够分别报告 runtime dynamic 的收益和 TISA device dynamic 的收益；模型 proxy、source-derived 和 RTL-observed 结果不能混在同一统计组中。

## 第一里程碑定义

第一里程碑完成条件：

```text
2mm benchmark
+ 2 个 architecture profile
+ sequential/static dual/dynamic dual
+ total cycle 和 stall breakdown
+ Perfetto trace 和 PNG swimlane
+ 手算 micro-test
+ 可复现实验 manifest
```

这是后续 Attention/TISA 复现的稳定底座。

## 下一里程碑定义

```text
torch.export / ExecuTorch micro model
  + automatic Canonical OperatorGraph
  + default tile planner
  + RuntimeSubmission with physical addresses
  + analytical device backend
  + static/dynamic runtime × static/dynamic device comparison
  + unchanged Perfetto/swimlane schema
```

下一里程碑完成后，项目才真正具备“从模型导出到 runtime 再到 TISA device simulator”的闭环；SCALE-Sim、RTL 和 gem5 集成应建立在该闭环之上，而不是替代它。
