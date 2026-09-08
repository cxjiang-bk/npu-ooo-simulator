# 整体架构

## 总体流程图

```mermaid
flowchart TB
    subgraph Frontend[前端：真实 PyTorch 输入]
        A[PyTorch nn.Module] --> B[torch.export]
        B --> C[Torch-XLA: ATen -> StableHLO]
        C --> D[官方 StableHLO parse / verify]
    end

    subgraph Compiler[编译器：论文 GC / FC / TISA Generator]
        D --> E[GC: Canonical IR、tiling、依赖]
        E --> F[Semantic TileGraph]
        F --> G[FC: 符号 TISA 方言]
        G --> H[TISA Generator: 虚拟 TISA]
        H --> I[Target Lowering: TargetPlan]
        I --> J[最终 TISA + Backend payload + MemoryPlan]
    end

    subgraph Runtime[Runtime]
        J --> K[基址/动态参数绑定、command chunk、descriptor arrival]
    end

    subgraph Device[Device scheduler]
        K --> L[reception / WQ / IQ / ROB]
        L --> M{Device policy}
        M --> N[Static: program order]
        M --> O[Dynamic: ready queue + OOO]
    end

    subgraph Backend[Backend timing/event]
        N --> P[ExecutionTask payload]
        O --> P
        P --> Q[completion feedback]
        Q -.-> L
    end

    P --> R[cycles / stalls / utilization]
    P --> S[swimlane / Perfetto trace]
```

`Static` 与 `Dynamic` 共享同一份 `BackendArtifact`。Runtime policy 和 device policy
分别描述 descriptor 到达与到达后的 issue 选择，是两个独立实验轴。

## 1. 分层契约

生产输入是 PyTorch `nn.Module`。各层的输入输出如下：

| 层 | 输入 | 输出 | 语义职责 |
| --- | --- | --- | --- |
| Frontend | module + example tensors | ExportedProgram、StableHLO module、provenance | 捕获 PyTorch 语义并完成官方 StableHLO 验证 |
| GC | StableHLO projection | `GCArtifact` | canonicalization、semantic recovery、tiling、region/state dependency |
| FC | `GCArtifact` | `TISADialectProgram` | 生成逻辑 transfer/compute、显式 source/destination、符号 buffer、visibility/owner/domain 和抽象 UnitMap |
| TISA Generator | TISA 方言 | 虚拟 `TISAProgram` | 规范化 descriptor，不读取目标 route |
| Target Lowering | 虚拟 TISA + `MachineConfig` | `TargetPlan`、目标 TISA | 统一决定 memory、每跳 route/engine、target stage、operand 和依赖展开 |
| Backend payload | `TargetPlan` | `ExecutionGraph` | 按目标计划生成 payload；实际 region 用于一致性校验 |
| Memory planner | `TargetPlan` + payload resource declaration | `MemoryPlan` | 容量、对齐、allocation、alias、slot 和复用依赖 |
| Runtime | `BackendArtifact.memory_plan` + invocation bindings | `RuntimeSubmission` | 绑定地址空间基址、动态参数、command chunk、到达时间和同步 |
| Device scheduler | submission + policy | schedule result | 处理 queue/ROB、依赖、资源和 issue |
| Event backend | issued instruction + payload | events、cycles、trace | 计算执行时序并输出可视化 |

编译层保留语义信息，后端层提供硬件执行细节，runtime 层管理 invocation 生命周期。

## 2. 主调用链

CLI 入口位于 `src/npu_ooo/cli.py`：

```text
main
  -> run_compile / run_simulate / run_compile_and_sim
  -> compile_torch_module
       -> TorchExportAdapter
       -> Torch-XLA exporter
       -> shape specialization
       -> OfficialStableHLOAdapter
       -> GraphCompiler
       -> FusionCompiler
       -> TISAGenerator
       -> TargetLowerer
       -> CodegenBackend
       -> RuntimeSubmission
       -> schedule_tisa_program
       -> trace writer
```

`compile_operator_graph()` 接受已经导入的 Canonical graph，用于单独验证 GC、FC 和
backend 契约。用户 CLI 提供 `compile`、`simulate` 和 `compile-and-sim` 三个入口：前者
生成 compile package，第二个只消费 package，第三个串联两者。

## 3. Frontend

### 3.1 PyTorch 与 torch.export

```python
exported_program = torch.export.export(module, args, **export_kwargs)
```

`ExportedProgram` 包含 FX/ATen graph、graph signature、参数与 buffer 描述以及 shape
constraint。example tensors 确定本次编译的 rank、dtype 和 shape。源图摘要写入
`00_frontend/source_frontend_import.json`，用于 provenance 和前后端对照。

### 3.2 Torch-XLA 与官方 StableHLO

Torch-XLA 将 ATen 语义导出为 StableHLO。项目保存可读 MLIR、bytecode hash、版本和
导出 provenance 到 `00_frontend/generated.mlir` 与 `stablehlo_module.json`。

`OfficialStableHLOAdapter` 使用 OpenXLA bindings 完成：

```text
register StableHLO dialect
  -> Module.parse(text)
  -> module.operation.verify()
  -> canonical assembly
```

项目维护 StableHLO semantic family 到 Canonical/TISA/backend capability 的映射。
`StableHLOOpCapabilityRegistry` 负责单条 operation；`SemanticFusionPatternRegistry`
负责经过 shape、常量和数据流证明的多节点语义恢复。operation capability 诊断包含原始
名称、规范名称、缺失注册项和已知 operation 集合。

dtype 名称由 IR 级共享 registry 解析。StableHLO 的 `i32/i64/ui*` 和 PyTorch/Canonical
别名使用同一 capability family 与 storage byte width，避免索引 tensor 在编译边界被误判。

### 3.3 Shape specialization

Torch-XLA 对动态 shape 生成 `get_dimension_size`、shape tensor 和 dynamic
operation。compiler 在官方 parse/verify 前执行 operation-level specialization：

- `get_dimension_size -> reshape -> concatenate -> maximum` 可求值时，广播转换为
  静态 `broadcast_in_dim`；
- 常量 start 的 `dynamic_slice` 按 StableHLO clamp 语义转换为 `slice`；
- 常量 shape tensor 的 `dynamic_reshape` 在目标维度为正数且元素总数守恒时转换为
  `reshape`；
- shape-only SSA 在转换后清理，variant、shape environment 和诊断写入 artifact。

这一阶段产出经过语义验证的 StableHLO module。常量索引在 specialization 中静态化；
运行时索引保留为 `DynamicIndexExpr`，由 runtime `DynamicIndexBinding` 提供本次 invocation
的具体值。`dynamic_update_slice` 同时生成 `stateful/state_id/state_buffer` contract，并
将结果 alias 到 persistent state buffer。runtime 先按 StableHLO clamp 规则求解索引，再
使用 buffer 的 dense 或显式 byte strides 计算动态窗口的 physical offset/span；resolved
region、索引和 provenance 进入 TISA operand 与 address scoreboard。未解析的 StableHLO
layout encoding 经过统一 resolver：可验证的 strides/minor-to-major 进入 concrete stride
metadata，opaque encoding 保留 conservative logical region。

## 4. Graph Compiler（GC）

GC 输入是官方 StableHLO projection，输出 `GCArtifact`：

```text
OperatorGraph
ScheduleSpec
Semantic TileGraph
fusion / residency / locality metadata
typed tile dependencies
initial software order
per-pass graph snapshots
```

### 4.1 Canonical IR

`TensorSpec` 保存 shape、dtype、source kind、layout source 和 layout encoding。
`OperatorSpec` 保存 semantic family、输入输出、iteration/reduction dims、StableHLO
provenance 和 backend capability key。`DataEdge` 连接 producer、consumer 和 tensor。
零秩 tensor 使用 shape `()`，并以单元素 metadata 参与 elementwise region。

### 4.2 Pass pipeline

默认 pass 顺序：

```text
CanonicalizeGraphPass
LinearDecompositionPass
RecoverStableHLOLayerNormPass
RecoverStableHLOFlattenedLinearPass
FoldTransposeIntoMatmulPass
LayerNormFusionPass
RMSNormFusionPass
SoftmaxFusionPass
RotaryEmbeddingRegionPass
AttentionRegionPass
SwiGLUFusionPass
MoEDispatchRegionPass
```

GC 通过 fixed-point recovery 处理同一图中的多个规范化节点。复合语义以 region metadata
保留可观察成员：

- Attention region 记录 `QK^T -> score transform -> Softmax -> probability transform
  -> PV`，成员继续生成独立 TISA；
- RoPE region 记录 `value * cos + rotate_half(value) * sin`，Q/K 路径和旋转元数据
  继续可见；
- SwiGLU semantic operator 收敛 `logistic -> silu multiply -> gate multiply`，内部
  primitive 由同一 vector payload 承担；
- MoE dispatch region 保留 router softmax、top-k mask、per-expert dispatch weight、
  expert output 和 combine 角色；成员继续生成独立 TISA，以暴露 expert 分支并行；
- KV-cache recovery 识别固定窗口 `slice(cache) + concatenate(update)`，生成带
  `state_id/state_buffer` 的 `kv_cache_update`。

`softmax_algorithm` 是 Softmax lowering 属性：`materialized` 生成完整中间结果，
`online` 生成跨 reduction tile 的 `(max, sum)` state chain。该属性沿 GC、FC 和
backend payload 传递，device policy 独立配置。

### 4.3 Tiling、locality 与依赖

`SchedulePlanner` 的 baseline：

```text
tile_size(dim) = min(requested tile size, resolved extent)
loop_order = iteration dims + reduction dims
stage_id = operator topological order
```

`--tile-size-candidates` 启用 `cost-model-v1`，按 tile 数、估算计算周期、root
traffic 和 local working-set 计算候选分数，选择分数最低的候选并记录
`candidate_costs` 与 `selected_tile_size`。

`MachineConfig.operation_placements` 按 operation/operand role 描述执行单元、目标存储和
逐跳 route。planner 将其写入 residency/ping-pong intent；CodegenBackend 才负责实际
allocation 与复用，Runtime 不重新选择 placement。

`build_tile_graph()` 为每个 `TileInstance` 记录 tile id、operator id、coordinates、
bounds 和 semantic metadata。跨算子边使用 `logical_tensor_region_v1`：producer 与
consumer tile 的逻辑 region 重叠时建立 `TileDependency`，并保存 hazard kind、两侧
logical region、ready condition 和 provenance。数据流边使用 RAW；reduction、state、
accumulate 和 buffer-reuse 分别使用项目扩展关系。Matmul 的 M/N/K、broadcast
elementwise、reduce/norm、卷积/池化 halo 和 full-tensor transform 各有对应投影规则。
映射信息不足时采用记录在统计中的 conservative overlap。

普通 reshape/transpose 使用 full-tensor DMA transform。slice 使用 output-tile copy，
其 source operand 记录动态索引表达式；runtime 绑定后使用动态窗口的具体物理区间。静态 `broadcast_in_dim` 按
输出域切 tile，并依据 `broadcast_dimensions` 投影源 operand region。卷积和 pooling
输入 region 包含 window/kernel halo。FC `TileMem` 保存 scope、logical address
expression、concrete offset/size、`strides_bytes`、`stride_expr`、layout 和 dtype
metadata；可验证 stride 生成 concrete interval，opaque encoding 保留 logical region
并使用 conservative overlap。

## 5. Fusion Compiler（FC）

FC 消费 `GCArtifact` 中的 `OperatorGraph`、`ScheduleSpec` 和 `TileGraph`，
把每个 semantic tile stage 具体化为目标无关的 TISA 方言 operation：

```text
OpType / semantic family
Operands: TileShape + TileMem + AccessType
UnitMap
typed dependencies + readiness condition + provenance
fusion / reorder attributes
backend payload recipe
```

核心过程：

1. 验证 `GCArtifact`；
2. 按 TileGraph 拓扑顺序选择 operator family 的 stage 模板；
3. 将 tile bounds 投影到输入输出 tensor，构造逻辑 `TISAOperand`；抽象 transfer 同时
   声明 source 与 destination buffer；
4. 写入 operand role、符号 buffer id、Private/Local/Shared visibility、owner/domain、
   抽象 `dma/tensor/vector` UnitMap、readiness condition 和 payload recipe；
5. 投影 region/state/accumulate/buffer-reuse 边，并补齐同一 tile 的 stage 顺序；
6. 稳定拓扑排序得到 `program_order`，生成 `TISADialectProgram`。

FC Matmul 的边界固定为：

```text
abstract load(source -> tile buffer)
  -> abstract tensor compute(lhs/rhs -> output accumulator)
  -> abstract store(output accumulator -> destination)
```

其中 `Shared` 表示图/程序域可见的逻辑 tensor，`Private` 表示单 tile 计算链拥有的临时
operand，`Local` 表示需要跨同一输出的 reduction tile 保存的局部状态。它们不是具体
memory 的别名；lhs/rhs 即使都是 Private，仍可在目标后端分别映射到 LMB/RMB。

FC 不读取 GM/UB/LMB/RMB、GDMA/LDMA 或 route hop。每条抽象 operation 绑定一个资源
类别。Softmax 的
`reduce_max/exp/reduce_sum/normalize` 作为 VE payload primitive，保持 semantic
instruction 的整体依赖和完成边界。

## 6. TISA Generator 与 TISAProgram

`TISAGenerator` 将 TISA 方言规范化为虚拟 `TISAProgram`，保留 FC 已确定的抽象 stage
边界，不进行 target placement。该结果独立保存在
`03_tisa/virtual_tisa_program.json`。每条 `TISAInstruction` 包含：

```text
tisa_id / tile_id / operator_id
op_type
TISAOperand(tile shape, TileMem, access type)
UnitMap
typed dependency: kind、condition、provenance
semantic metadata
payload_ref
```

此时 `TileMem.memory_space=None`，`scope/visibility` 表达抽象可见性；CodegenBackend
物化后，对外暴露的
`CompiledArtifact.tisa_program` 与 `BackendArtifact.program` 是同一个最终目标程序。

`ExecutionTask` 属于 backend payload，表示同一 TISA issue 后在目标 execution unit
内执行的步骤。全局 scheduler 以 TISA instruction 为唯一调度单位；payload lane 事件
用于 timing 和泳道图。每个 task 的 predecessor metadata 保存对应 GC edge 的
`hazard_kind`、`condition`、logical region 和 provenance；trace 的 `WAKE_UP`、`ISSUE`、
`COMPLETE` 与 address scoreboard event 直接消费这份 metadata。

## 7. BackendArtifact

`CodegenBackend` 接收 `TISAProgram` 并生成：

```text
BackendArtifact {
  program: final target TISAProgram
  target_plan: TargetPlan
  execution_graph: ExecutionGraph
  payloads: tisa_id -> ExecutionTask ids
  memory_plan: MemoryPlan
}
```

目标存储物化遵循同一来源：

```text
FC symbolic operand / abstract instruction
  -> TargetPlan(MachineConfig placement, route, target instruction, operand)
  -> final TileMem(symbolic id, target buffer id, memory, range)
  -> backend BufferRegion generated/bound from the same TargetPlan
  -> MemoryPlan(allocation, offset, lifetime, slot, alias)
```

`TargetPlan` 保存 abstract instruction 到一个或多个 target instruction 的映射，以及每跳
source/destination、engine、layout transform、provenance、展开后的依赖和最终 MemoryPlan。
Matmul payload
直接遍历 TargetPlan 生成，不再独立计算 route 或约定同名 stage key。其他现有 payload
lowerer 也必须把 region 绑定到已规划 operand；payload 不再反向定义最终 TISA operands。

Backend-private Softmax/Norm scratch 不属于 FC 外部访存契约，由 backend 在 allocation 前以
`internal_resource_contract=explicit_target_plan` 加入 TargetPlan，然后接受同样的容量、
地址和 payload 覆盖检查。

最终 TISA operands 必须覆盖 payload 的每个 read/write region，artifact 校验会检查
buffer identity、memory、allocation 和范围。局部 packed tile 使用
`valid_bytes` 表示有效数据/搬运量，`size_bytes` 表示地址包围跨度；二者不会再混用。

分配器按 memory space 检查对齐和容量。不同逻辑 buffer 可以通过显式 `allocation_id`
复用同一物理 slot；编译器同时加入 RAW/WAR/WAW 或 `BUFFER_REUSE/allocation_released`
依赖，避免动态 issue 提前覆盖。Tensor alias/view 使用独立 `buffer_id` 和显式 `alias_of`，
而不是靠 tensor 名推断。

默认 lowering registry 覆盖：

```text
matmul / batched_matmul / gemv
elementwise / residual_add
reduce / softmax / rmsnorm / layernorm
reshape / transpose / conv2d / pooling
dtype_convert / kv_cache_update
embedding gather
```

backend 通过 `TimingProvider` 提供 duration 和 initiation interval；`EventBackend`
把 issue、task start、task done、TISA completion 和 resource release 写入统一 trace。

## 8. Runtime

`RuntimeSubmission` 完成：

```text
MemoryPlan allocation -> 本次执行的地址空间基址
TISA buffer_id -> physical range
command chunk
descriptor available cycle
launch latency
synchronization cost
```

Runtime 可以和编译阶段分开执行。`compile` 将以下文件组成可复用的 compile package：

```text
01_gc/canonical_graph.json
02_fc/tisa_dialect.json
03_tisa/virtual_tisa_program.json
03_tisa/tisa_program.json
04_backend/backend_artifact.json
04_backend/target_plan.json
04_backend/memory_plan.json
04_backend/machine.json
manifest.json
```

`simulate --compile-dir <package>` 只读取这些编译产物，然后在本次 invocation 中完成
编译计划的基址绑定、dynamic index/layout binding、descriptor 提交和 device/backend timing。它不
导入 PyTorch，也不重新执行 Torch-XLA、GC 或 FC。多个 `simulate` 命令可以复用同一个
package，对比兼容的 MachineConfig 参数、timing provider、runtime policy 和 device policy。
容量、带宽、latency、port/bank 和 unit 数量可以覆盖（新容量仍须容纳原计划）；改变
memory identity/parent、alignment、transfer connectivity/engine/transform 或 operand
placement 会改变 topology hash，因此必须重新编译。旧 package 若没有 TargetPlan v1 或
MemoryPlan v2，独立 simulate 会明确拒绝，避免把符号 TISA 默认为某个目标存储。

命令参数遵循同一边界：`compile` 只接受前端、shape、tile/GC 和 codegen 选项；`simulate`
接受 runtime、scheduler、MachineConfig 覆盖和 timing/event backend；`compile-and-sim` 将
两组参数组合为一次端到端执行。编译包因此不携带某次仿真的运行时地址、descriptor 顺序
或调度结果。

Runtime policy 表示 descriptor 的生成和提交顺序；device policy 表示已到达 TISA
instruction 的 issue 选择。四种组合由 `--runtime-device-matrix` 一次编译后运行。

Runtime submission 还携带 `DynamicIndexBinding`。binding 的 expression id 必须匹配
TISA 的 `dynamic_index` metadata，值的 rank 按 expression contract 校验。runtime 为每个
operand 记录 clamp 后的索引、dynamic region、physical offset/span 和 provenance；TISA
address scoreboard 直接消费这些范围。`RuntimeLayoutBinding` 可以在每个 invocation 为
外部/runtime-owned buffer 绑定具体 shape、byte strides、layout 和 offset；目标局部副本
仍保持编译期 packed layout。bank-aware 映射使用这些 resolved 地址。

固定窗口 KV-cache 携带 `state_id/state_buffer`。`RuntimeStateRegistry` 绑定稳定的
persistent address、memory scope 和容量；`RuntimeSequence` 为同一 `BackendArtifact`
创建多次 invocation，并加入：

```text
invocation[n-1] --state_complete(state_id)--> invocation[n]
```

sequence simulator 合并每个 invocation 的事件和 timing，输出
`STATE_RELEASE/STATE_WAIT/STATE_READY`。当前 state contract 定义固定 shape、unit
stride、固定窗口和顺序 decode；动态 position、paged cache、跨 request ownership 和
完整 cache layout 作为后续 runtime capability。

`RuntimeSequence` 同时承担 paper-matrix 的 request replay。Stateful request 使用相邻
`state_complete` 边；stateless request 使用相同 compile package 和 buffer contract，按
配置的 inter-request gap 顺序重放。sequence 汇总对所有 invocation 的 runtime submit、
device 和 synchronization 周期求和。

## 9. Device Scheduler

`schedule_tisa_program()` 消费 BackendArtifact、MachineConfig、RuntimeSubmission、
SimulatorConfig、TimingProvider 和 EventBackend。

`analytical_event` 使用事件级基线；`cycle_event` 使用显式 reception FIFO、per-EU
WQ/IQ/Fu 和按 descriptor 提交顺序退休的 ROB。其周期顺序固定为
`retire → complete → wakeup → issue → select → dispatch → receive`。
`MachineConfig.scheduler.pipeline` 或 `--scheduler-config` 配置控制延迟和带宽。
完整接口、手算案例和设计假设见 [device-scheduler.md](device-scheduler.md)。

```text
static_pipeline:
  按 program order 与依赖约束 issue

dynamic_ready_queue:
  在 dependency window / ROB / ready queue 内选择
  已到达、依赖满足、UnitMap 可用的 TISA instruction
```

可配置参数包括 instruction queue depth、ROB entries、dependency window、ready queue
depth、max inflight tiles、address scoreboard 和 dynamic priority。

`address scoreboard` 的 RAW/WAR/WAW 观察包含 predecessor、successor、tensor、memory、
condition 和 provenance。若冲突来自编译期 GC edge，记录中嵌入同一条 TISA dependency；
若冲突只由运行时物理区间产生，记录 `address_scoreboard` 作为来源。

memory bank scoreboard 读取 `MachineConfig.memory_levels` 的 bank 数、bank width、
read/write ports，为 active TISA instruction 建立 analytical reservation，并记录
`memory_bank_block_events`。该模型提供结构冲突趋势；真实 SRAM/DRAM 时序由专用
memory backend 提供。

## 10. 可插拔后端与配置

| 接口 | 输入/输出 | 当前实现 |
| --- | --- | --- |
| `CodegenBackend` | virtual TISA -> TargetPlan -> final TISA/payload/MemoryPlan | analytical |
| `TimingProvider` | ExecutionTask -> duration/II | analytical、timing_table、systolic_mxu_profile |
| `EventBackend` | TISA + payload -> event execution | analytical_event、cycle_event |

`MachineConfig` 描述 execution unit 数量、memory hierarchy、interconnect、队列容量和
默认 timing。配置变化作用于 backend 和 scheduler 参数，IR schema 保持一致。

## 11. 输出与复现

artifact 按 `00_frontend` 到 `07_trace` 分层。比较策略时固定 module、example
shape/dtype、Torch-XLA/StableHLO version、tile size、MachineConfig、BackendArtifact、
TimingProvider 和 RuntimeSubmission；实验变量明确写入 manifest。

`compile_statistics.json` 保存 per-operator tile/TISA/payload、MAC、root traffic 和
dependency 数量。`manifest.json` 保存 frontend path、工具版本、machine hash、backend、
policy、TISA instruction count、cycle 和 calibration status。

## 12. 当前范围与扩展项

当前生产链路覆盖：

- Matmul、batched Matmul、GEMV、elementwise、reduce、Softmax、LayerNorm、RMSNorm；
- Attention、SwiGLU、RoPE、Conv2D、BatchNorm inference、max/avg pooling；
- reshape/transpose、slice、静态 broadcast、scalar tensor、dtype convert；
- token/position/type embedding gather、causal mask、重复 Transformer/ResNet block；
- DeepSeek dense 与外部 top-k mask 驱动的 MoE dispatch region；
- 固定窗口 KV-cache、dynamic_update_slice state contract 与多步 RuntimeSequence；
- analytical、timing table、systolic MXU profile 和 RTL completion trace importer。

扩展项按以下顺序推进：

1. 更复杂 StableHLO layout dialect 的扩展与 bank-aware memory timing 校准；
2. online Softmax 的数值 rescale、最终 normalization 与 workspace 生命周期；
3. 论文 WQ/IQ/Fu 容量、dispatch/wake-up/issue/completion 控制开销的硬件校准；
4. SCALE-Sim、Ramulator2/DRAMSys、RTL/Verilator 和 system simulator adapter；
5. 动态 top-k、token compaction/expert capacity 与精确论文模型拓扑。

当前结果标签为 `TISA instruction-level analytical scheduling baseline`。加载相应
profile 后，结果标签随 manifest 的 calibration status 变化；trace schema 和编译产物
保持一致。
