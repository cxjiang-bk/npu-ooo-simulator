# 总体架构

## 1. 系统边界

本项目研究的是 NPU 上的 tile-level scheduling，而不是仅做 loop mapping，也不是第一步就复刻某一款 NPU 的 RTL。系统同时区分三个动态层次：编译器生成静态任务描述，runtime 动态完成 buffer 绑定和任务提交，device backend 在已提交任务窗口内执行 TISA-like 静态/动态 issue。

```text
PyTorch nn.Module
    -> torch.export.ExportedProgram
       |-- A. torch-xla exporter -> official StableHLO ---------|
       |-- B. project legalizer -> verified StableHLO ---------+--> FrontendImport
       `-- C. direct TorchExportAdapter ------------------------|         |
                                                                         v
External StableHLO -> official parser/verifier -> StableHLOAdapter ------+
Canonical JSON / Model preset -> JsonGraphAdapter / model instantiate ---+
                                                                         |
                                                                         v
Model + Canonical Operator IR
          |
          v
Compiler PassManager
  - normalize/decompose/shape/layout
  - fusion/liveness/memory planning
  - tile planning and region dependencies
          |
          v
Schedule/Tiling IR
          |
          v
Tile Instance IR
          |
          v
TISAProgram / Semantic Tile Instruction IR
  - OpType/Operand/UnitMap/typed Deps
  - logical addresses and scheduling metadata
  - descriptor/command stream templates
          |
          v
Backend Codegen / Adapter
  - Simulator artifact: descriptor + analytical payload
  - NPU artifact: TISA metadata + per-unit execution payload
  - CPU artifact: functional-validation kernels
          |
          v
Runtime / Loader
  - shape/state and physical-address binding
  - command/descriptor submission
          |
          v
TISA Device Scheduler
  - reception buffer and per-unit WQ/IQ
  - typed dependency/resource checks
  - static/dynamic tile issue
          |
          v
Backend Execution/Timing
  - DMA/MXU/Vector execution payload
  - hot-pluggable event/timing model
          |
          v
                 Trace + Cycle Summary
```

Model 层决定 workload 的拓扑、重复结构、shape environment 和 execution phase；Operator 层决定单个算子的数学语义；Compiler PassManager 决定规范化、分解、融合、layout、liveness 和 tiling；TISAProgram 描述可提交的语义 tile 指令；Backend Codegen 为每条 TISA 指令生成保留 semantic descriptor 的 target artifact；Runtime/Loader 绑定动态地址并提交；TISA Device Scheduler 决定 tile issue；execution backend 执行对应 payload。这些层不能混为一谈。

论文中的 `Framework bridge -> Graph compiler -> Fusion compiler -> TISA generator -> backend-specific code generation -> runtime interface` 对应本项目的 Frontend Adapter、Compiler PassManager、Schedule/Tiling、TISAProgram、Backend Codegen、RuntimeSubmission 和 TISA Device Scheduler。

### 1.1 Runtime 与 Device Backend 的边界

Runtime 是软件层，处理较粗粒度的动态事务：

```text
model phase / request selection
buffer allocation and lifetime
physical base-address binding
command-buffer construction
kernel/tile batch submission
host/device event and synchronization
```

Device Backend 是设备内部时序层，处理已提交命令的细粒度执行：

```text
command decode
dependency-ready and resource-ready
static reservation or dynamic ready-queue issue
ROB/window/backpressure
RAW/WAR/WAW address scoreboard
DMA/MXU/ARU start and complete events
```

因此 `runtime submission order` 不等于 `device issue order`。同一个 `TISAProgram` 可以由不同 runtime 提交策略送入同一个 Device Backend，也可以在同一个 runtime 下切换 static 和 TISA-like dynamic device policy。

## 1.2 Model IR：为什么需要这一层

只拥有 Operator Graph 不足以复现论文的 benchmark 表。以下信息不属于单个 operator：

- 模型家族、版本和 block template；
- block 重复次数、权重共享和跨层连接；
- batch、sequence length、image resolution、hidden/head dimensions；
- inference phase：`train`、`prefill`、`decode`；
- KV cache、causal mask、position state 和其他 runtime state；
- dtype、quantization、layout 和 benchmark warmup/repetition；
- optional routing，例如 MoE token dispatch 和 expert capacity。

因此 Model IR 不保存完整权重，也不把每一层无条件展开成巨型图，而是同时支持 `GraphTemplate` 和 `GraphInstance`：模板表达重复 block，实例化参数表达当前 benchmark case。

一个最小 Model IR 可以表示为：

```text
ModelSpec {
  name, family, version
  shape_env, dtype, layout
  execution_phase
  graph_templates, top_level_nodes
  persistent_state
}

BenchmarkCase {
  model_id
  evaluation_scope: one_block | layer | full_model
  batch, sequence_length, image_size
  phase: prefill | decode | train
  precision, quantization
  model_config_id, architecture_config_id
}
```

`ModelSpec` 经过实例化后才生成 Operator Graph IR。这样同一个 LLaMA block 可以复用于不同 sequence length、batch 和 prefill/decode case。

## 2. 编译、运行时与设备 IR

### 2.0 Framework bridge：torch-xla、StableHLO 与并行 frontend 路线

论文的 frontend 不是直接从 PyTorch kernel 进入 TISA。其层次是：

```text
PyTorch/JAX/TF
    -> torch-xla framework bridge
    -> XLA / StableHLO portable graph IR
    -> MLIR Graph Compiler
    -> Fusion Compiler / TISA dialect
    -> TISA generator
```

`torch-xla` 是导出/连接框架，属于 compiler pipeline 最上游的 framework bridge；
`StableHLO` 是 bridge 之后的可移植 semantic graph IR，位置类似本项目的
`Canonical OperatorGraph`，但不是最终 TISA 指令。它保留跨框架可识别的算子语义，
供 Graph Compiler 做 decomposition、fusion、layout、tiling 和 shape analysis。

前端不是一条互相替代的单链，而是三条并行路线。它们服务于不同研究目的，并在
`FrontendImport / Canonical OperatorGraph` 边界汇合：

| 路线 | 编译链 | 定位 | 是否经过 StableHLO |
| --- | --- | --- | --- |
| A. 论文对齐主路径 | `PyTorch -> torch.export -> torch-xla -> official StableHLO -> importer` | 正式研究 PyTorch 模型和新标准算子 | 是，StableHLO 由 torch-xla 产生 |
| B. project legalizer 路径 | `PyTorch -> torch.export -> TorchExportAdapter -> project legalizer -> official verifier -> importer` | 小算子、pattern、round-trip 和错误定位 | 是，StableHLO 由项目受控 emitter 产生 |
| C. direct/debug 路径 | `PyTorch -> torch.export -> TorchExportAdapter`，或 `Canonical JSON -> JsonGraphAdapter` | 无 XLA 依赖基线、单元测试和前后端隔离调试 | 否 |

外部 StableHLO 文件是路线 A/B 之外的标准 IR 入口：必须先经过 official parser/verifier，
再由同一个 StableHLO importer 汇入 Canonical OperatorGraph。未来 ONNX、ExecuTorch 或
torch-mlir adapter 也只能增加新的上游入口，不能改变汇合点后的契约。

各组件归属必须明确：

| 组件 | 归属 | 本项目是否维护其算子覆盖 |
| --- | --- | --- |
| `torch.export` | PyTorch | 否，只适配其 `ExportedProgram` 契约 |
| `exported_program_to_stablehlo` | PyTorch/XLA (`torch-xla`) | 否，标准 PyTorch/ATen legalization 由上游负责 |
| StableHLO dialect/parser/verifier/bytecode | OpenXLA StableHLO 官方仓库 | 否，使用官方 wheel 和兼容性规则 |
| `OfficialStableHLOGenerator` / `stablehlo_codegen.py` | 本项目 | 是，但仅维护受控 regression 子集 |
| StableHLO importer / semantic-family recovery | 本项目 | 是，这是前端主要扩展点 |
| Canonical Graph / TISA / backend lowering | 本项目 | 是，这是硬件能力建模边界 |

#### 路线 A：torch-xla 论文对齐主路径

```text
PyTorch nn.Module
    -> torch.export.ExportedProgram
    -> torch_xla.stablehlo.exported_program_to_stablehlo()
    -> StableHLO text + bytecode
    -> OpenXLA StableHLO dialect registration / parse / verify
    -> StableHLO op import + semantic-family recovery
    -> Canonical OperatorGraph
```

这是后续支持标准 PyTorch 算子的主方向。PyTorch/ATen 到 StableHLO 的 legalization 由
torch-xla 负责，本项目不为每个新 PyTorch 算子重复编写 emit 函数。本项目维护的边界是：

```text
StableHLO operation
    -> semantic family / composite pattern
    -> Canonical OperatorSpec
    -> TISA/backend capability
```

因此“torch-xla 能导出”不等于“backend 已支持”。例如 `torch.sin` 可以动态导出为通过
官方 verifier 的 `stablehlo.sine`；如果 semantic-family registry 尚未把它归入 pointwise，
或者 backend 没有对应 vector/scalar lowering，编译必须在 capability boundary 明确失败，
不能绕回 source FX graph，也不能静默改用 project legalizer。

当前通过 CLI 使用路线 A 时必须显式指定：

```text
--through-stablehlo
--stablehlo-exporter torch-xla
--stablehlo-backend official
```

`TorchXLAStableHLOExporter` 保存 human-readable module、bytecode size/hash、exporter version
和 producer；`OfficialStableHLOAdapter` 再独立执行官方 verifier。manifest 必须记录
`stablehlo_producer=torch-xla`、版本、verifier 和 `stablehlo_fallback=false`。

#### 路线 B：project legalizer 回归路径

```text
TorchExportAdapter
    -> Canonical source graph
    -> OfficialStableHLOGenerator / StableHLOGenerator
    -> official verifier 或 textual regression parser
    -> StableHLO importer
    -> Canonical OperatorGraph
```

`OfficialStableHLOGenerator` 和 `stablehlo_codegen.py` 中的固定 emit 函数是项目内
legalizer，只覆盖当前受控算子集合。它们用于构造确定性 fixture、验证 importer/fusion、
比较 direct 与 StableHLO round-trip，以及在 torch-xla 输出过于复杂时缩小问题；它们
不是完整 XLA compiler，也不再作为扩展标准 PyTorch 算子的主入口。

项目 legalizer 的 `official` backend 仍必须输出官方语法，并通过 OpenXLA parser/verifier。
旧 `StableHLOGenerator/StableHLOAdapter` 的 `textual` backend 只服务无官方依赖的 regression；
`auto` 的任何回退都必须进入 manifest。当前 CLI 为兼容已有 smoke 仍默认
`stablehlo_exporter=project`，正式论文实验必须显式选择路线 A；待 importer/backend
semantic-family 覆盖达到基线后，再单独切换默认值。

#### 路线 C：direct/debug 与外部 IR 路径

Direct TorchExport 路径跳过 StableHLO，用于判断问题发生在 framework export、StableHLO
import，还是 graph/TISA/backend。Canonical JSON 继续提供完全无 PyTorch/XLA 依赖的
回归基线。两者都不是论文最终 frontend，但对于分层验证不可删除。

三条路线的公平性比较不能依赖 SSA 名称、参数顺序或完全相同的拓扑序。最低比较项为：

```text
semantic operator family + tensor shape
dependency topology
tile count and tile bounds
TISA instruction/dependency count
backend primitive task count
cycle and trace
```

只有汇合后的 semantic graph、TISA 和 backend capability 相同，调度策略对比才有意义。

真实 PyTorch 路径当前已验证：

```text
aten.linear
  -> Matmul(rhs_transposed) + broadcast Add

K.transpose(-2, -1) -> rank-3 aten.matmul
  -> batched Matmul(rhs_transposed/load_transpose)

RMSNorm arithmetic chain
  -> semantic RMSNorm
```

当前 project legalizer round-trip 为：

```text
TorchExportAdapter
    -> OfficialStableHLOGenerator (generated.mlir)
    -> OpenXLA MLIR parse + StableHLO verify
    -> OfficialStableHLOAdapter
    -> Canonicalize/Decompose/Fold/Fusion Passes
    -> Schedule/Tiling/TISA/Backend
```

在路线 B 中，Softmax、RMSNorm、LayerNorm 先展开为合法 StableHLO primitive chain，再由
对应 FusionPass 恢复语义算子。路线 A 则消费 torch-xla 真实产生的 StableHLO pattern。
官方 bindings 只解决 dialect/grammar/verifier 真实性；动态 shape、layout、复杂 region、
通用多结果消费仍需要扩展 StableHLO importer 和 semantic-family recovery。两条路径都复用
`OfficialStableHLOAdapter` 和汇合点后的下游。

路线 A 当前已验证 Matmul、attention micrograph 与包含 `nn.Linear/nn.LayerNorm` 的完整
attention block。
Graph Compiler 会恢复 flatten/unflatten Linear，并在直接 row-wise 或
`[1, prod(outer), hidden]` reshape 的数学等价约束成立时，把多结果
`batch_norm_training` 加 affine 恢复成 LayerNorm。默认 project exporter 继续承担当前
宽覆盖 smoke；`--stablehlo-exporter torch-xla` 用于论文同构路径，且不会静默 fallback。

LayerNorm、RMSNorm 和 Softmax lowering 支持多个静态 outer dimensions；Softmax 的
物理 BufferRegion 顺序按原 tensor axis 恢复，不把非末轴 reduction 错写成末轴访问。
这些 pass 仍属于本项目的最小 Graph/Fusion Compiler，不等价于论文产业编译器的完整
layout、memory planning、cost model 或 StableHLO region 处理能力。

`compile-model` 当前有三个互斥入口：`--graph-json`、`--stablehlo-file` 和
`--torch-module MODULE:FACTORY`。PyTorch CLI 由零参数 factory 构造 `nn.Module`，再按
重复的 `--input-shape` 生成静态浮点 example inputs；复杂模型输入仍通过
`compile_torch_module()` API 注入，避免 CLI 猜测 tokenizer、mask 或 cache 语义。

### 2.1 Model/Benchmark IR

描述 workload 的模型级语义：

```text
model family and version
block templates and repetition
shape environment and execution phase
state/cache/parameter metadata
benchmark case and measurement protocol
```

首批 model family：

- `cnn_residual`：ResNet50；
- `encoder_transformer`：BERT-Base；
- `decoder_transformer`：GPT-J、LLaMA2；
- `decoder_reasoning`：DeepSeek-R1-16B；
- `moe_decoder`：作为可选扩展，不对 DeepSeek-R1 是否使用 MoE 做未经证实的假设。

### 2.2 Operator Graph IR

描述计算语义，不包含具体硬件：

```text
Tensor: name, shape, dtype, layout
Operator: type, inputs, outputs, iter dims, reduce dims, attributes
Edge: producer, consumer, tensor
```

首批 operator type：

- `matmul`；
- `elementwise`；
- `reduce`；
- `softmax`，先作为可展开的 composite op；
- Conv2D 只预留接口，后续再处理 halo、padding 和 layout。

### 2.3 Schedule/Tiling IR

描述编译期 mapping 决策：

```text
loop order
tile factors
temporal/spatial mapping
fusion/stage boundaries
tensor residency
buffer assignment
explicit child dependencies
```

这一层允许手写 schedule，也允许以后接 TileFlow/Timeloop 或其他 mapper。

### 2.4 Tile Instance IR

把 schedule 中的切分规则展开成实际运行实例：

```text
tile_id
operator_id
iteration coordinates
logical bounds
tile shape
input/output regions
stage_id
program order
```

`M=32` 是 schedule factor；`M=[64,96)` 才是 tile instance。边界 tile 必须保留实际 shape。

### 2.5 SemanticTileInstruction / TISA IR

这一层是 compiler 和 TISA device scheduler 之间的稳定契约。TISA 不是最终的 MXU/Vector/DMA 微指令，也不是只存在于 compiler 内部的普通 IR；论文将它定义为硬件消费的 tile-level scheduling-semantics ISA。Epoch 上有具体 binary encoding，但其语义粒度高于传统 per-unit execution ISA，最终仍需要 backend-specific lowering。

当前项目的三个对象应明确区分：

```text
TileInstance
  几何/迭代层：这个 operator 在哪些 bounds 上计算

SemanticTileInstruction / TISAInstruction
  语义/调度层：这个 tile 做什么、读写什么、可去哪个 unit、依赖什么

ExecutionTask
  backend 层：某个 TISA instruction 在具体 backend 中展开出的 primitive
  例如 DMA burst、MXU issue、ARU reduction、store
```

一个 TISA instruction 的规范字段是：

```text
TISAInstruction = (OpType, Operands, Attributes, UnitMap, Deps)

Operand = (TileShape, TileMem, AccessType)
TileMem = (base, scope)

OpType       semantic identifier, e.g. GEMM/SOFTMAX/LOAD
TileShape    symbolic or concrete computational bounds
TileMem      logical/physical address expression and memory scope
AccessType   R / W / RW
Attributes   reorder constraints, barrier/sync, partial-ready condition
UnitMap      (unit class, quantity, affinity)
Deps         (source, RAW/WAR/WAW, condition)
```

`TileInstance` 目前只有 `operator_id`、coordinates、bounds 和 stage；它在粒度上接近论文中的 tile，但还不是可供 TISA scheduler 使用的完整指令。后续必须由 compiler 将 TileInstance 和 operand region 组装为 `TISAInstruction`，而不是直接把它降成失去语义的 primitive task。

一个命令模板可以在 runtime 阶段绑定 batch、KV-cache base、buffer base 和实际 tile 坐标，再形成一次 `RuntimeSubmission`。TISA scheduler 应在 TISAInstruction 粒度做 run-to-complete、non-preemptive 的 tile-level issue；只有在真正的 backend timing 或 RTL adapter 内部，才展开为 `ExecutionTask` primitive。

### 2.6 BackendArtifact / Runtime Binding IR

BackendArtifact 保存与 TISA descriptor 关联的 target execution payload；Runtime 不重新编译数学语义，而是为 `BackendArtifact` 提供一次执行所需的动态状态：

```text
submission_id
program_id
shape and phase bindings
logical-to-physical buffer map
command-buffer chunks
submit order and queue
launch/event synchronization
runtime overhead model
```

`RuntimeSubmission` 的输出是 Device Backend 可以接收的命令包。静态 runtime 可以按编译顺序提交；动态 runtime 可以根据软件 ready queue、buffer availability 或 request state 选择提交顺序，但两者必须引用同一份 compiled task metadata。

### 2.7 Primitive Execution Graph

一个 TISA instruction 根据 MachineConfig 和 backend capability lower 成若干 primitive task。当前项目的 `ExecutionGraph` 是默认 analytical backend 的 primitive 输入；TISA 对齐后，它应当成为 `TISAProgram -> BackendLowering -> ExecutionGraph` 的结果，而不是 compiler 的唯一中间表示：

```text
task_id
tile_id
opcode/primitive
resource_class
source/target buffers
read/write address ranges
latency request
predecessors
program-order tag
```

例如 Matmul tile 可以 lower 为：

```text
load-left -> load-right -> mxu-compute -> psum-update -> store-output
```

具体是否需要某一步由架构 profile 和 buffer residency 决定，而不是由 scheduler 猜测。

Runtime 绑定物理地址后，`BufferRegion` 同时保存 logical tensor region 和 concrete memory placement。这样同一个 execution graph 可以在不改写依赖拓扑的情况下，重放不同的 static/dynamic device policy；如果研究 buffer allocation 本身，则把 allocation policy 作为单独的 runtime 实验维度记录。

## 3. 参数化 MachineConfig

MachineConfig 是 TileFlow emitter、operator lowering 和 simulator 的共同事实来源。

除内置 profile 外，`MachineConfig.to_dict()` 产生的 canonical JSON 可以通过 `MachineConfig.from_dict()` / `load_machine_config()` 读回，并由 CLI 的 `--machine-config` 直接使用。这样实验矩阵可来自 profile 名称，也可来自版本化的外部架构文件；两者都必须将 `stable_hash()` 写入 manifest。

### 3.1 Memory

每个 memory level 至少包含：

```text
name, parent, capacity_bytes
read/write bandwidth
read/write latency
read/write ports
bank count, bank width
alignment
```

不预设固定命名。以下两种都应该可表达：

```text
DRAM -> SRAM -> RF
GM -> UB -> LMB/RMB/PMB/PSB/ARB
```

### 3.2 Execution Unit

每类 unit 至少包含：

```text
name, count, supported_ops
queue_depth, issue_width
pipeline_depth
latency model
initiation_interval
```

`latency` 与 `initiation_interval` 必须分离，以表达流水化单元连续启动不同 tile 的能力。

### 3.3 Transfer Path

数据通路由配置描述：

```text
source, target, engine
channel_count
bandwidth, setup_latency
optional transform and transform_latency
overlap capability
```

不能在代码里固定 `_route(GM, UB) -> GDMA`。当前 LPU 路径只是一份 profile。

### 3.4 Scheduling Capacity

动态调度实验需要显式配置：

```text
global instruction queue depth
per-resource queue depth
ROB/out-of-order window entries
maximum in-flight tiles
dependency tracking entries
issue width
```

这些参数必须进入 experiment manifest，否则无法解释动态调度收益来自何处。

## 4. Frontend Adapter 与 Compiler PassManager

### 4.1 并行 Frontend Adapter 与能力边界

当前 PyTorch 主入口是 `torch.export.ExportedProgram`。其后并行进入 torch-xla、project
legalizer 或 direct adapter，具体路线遵循 2.0 节。ExecuTorch 仍可作为未来的模型交付、
partition 和 runtime 集成入口，但当前不把它写成已经接入的第一前端，也不让它替代
StableHLO semantic layer。

Frontend 扩展采用两级 capability registry，而不是继续扩张单个 emitter：

```text
StableHLOImportCapability
  - op name / version / result arity
  - shape and dtype constraints
  - semantic family
  - composite recovery requirements

BackendCapability
  - semantic family / TISA OpType
  - supported dtype/layout/rank
  - required unit and memory behavior
  - lowering and timing provider
```

第一层 `StableHLOImportCapability` 的首个阶段已在
`src/npu_ooo/frontend/stablehlo_semantics.py` 中实现，当前覆盖 operation name、operand arity、
semantic family、recovery 标记和 backend key。importer 现在按完整 operation name
精确查询，而不是用 `add`、`mul`、`exp` 等 substring 猜测类型；未注册 operation 会在
frontend capability boundary 直接失败，不会以一个未知 `op_type` 延迟到 lowering 才报错。
当前首批覆盖：

```text
pointwise unary
  abs / cosine / exponential / log / negate / rsqrt / sine / sqrt / tanh

pointwise binary
  add / divide / maximum / minimum / multiply / power / subtract

structural and reduction
  dot_general / reduce / transpose / reshape / broadcast_in_dim

recovery or compatibility
  batch_norm_training / softmax / rms_norm / layer_norm
```

每个导入的 operation 至少携带：

```text
stablehlo_op             精确 StableHLO operation identity
semantic_family          Canonical OperatorGraph 分类
semantic_op              family 内的具体操作，如 sine、maximum、add
operand_arity            StableHLO 原始 operand 数，包括常量和重复 operand
requires_recovery        是否必须由 pattern/recovery pass 消除或改写
backend_capability_key   backend/timing 查询键，如 pointwise.sine
```

这里必须区分 `operand_arity` 与 canonical `inputs`：例如 `x * x` 的两个 StableHLO operand
会在 canonical tensor input 列表中去重；`maximum(x, 0)` 的常量 operand 不需要生成一次
内存 load。两者都仍然是二元运算，因此具体语义和 timing 使用 `operand_arity=2`，而 backend
task 的 `input_count` 只表示实际加载的 tensor 数。pointwise identity 和 capability key
会继续保留到 TileInstance、ExecutionTask 与 TISA metadata，供分阶段产物检查、timing table
和 trace 分析使用。

StableHLO version/result arity 与 shape/dtype constraint 仍需补入第一层。第二层
`BackendCapability` 也是下一阶段工作；当前 `LoweringRegistry` 只按 semantic family 选择
lowering，dtype/layout/rank、EU 支持范围和 timing provider 尚未统一成声明式 registry。

一个新标准 PyTorch 算子的接入流程是：先确认 torch-xla 可以导出且官方 verifier 通过，
再增加或复用 StableHLO semantic-family 映射，最后检查 TISA/backend capability。只有新的
复合模式或新的硬件语义族才应新增专用 recovery/lowering；普通 pointwise 算子不应各写
一套 graph emitter 和 backend pipeline。

FrontendAdapter 只负责输入格式转换和 provenance 保留，不负责 tile mapping 或 scheduler policy。至少要保留：

```text
source node and target operator id
tensor shape/dtype/layout
constant/parameter metadata
shape symbols and constraints
model/layer/block provenance
execution phase and persistent state
```

ONNX、ExecuTorch 和 Torch-MLIR 作为后续 adapter，不改变 Canonical OperatorGraph 的
下游契约。StableHLO 是当前跨框架 semantic canonicalization 层，torch-xla 是 PyTorch
标准算子到该层的主 exporter；direct TorchExport 和 Canonical JSON 保留为调试基线。

### 4.2 统一 Compiler PassManager

benchmark 不再直接构造后端 graph。统一 pipeline 至少包含：

```text
Import
  -> Normalize / Decompose
  -> Shape and Layout Inference
  -> Constant Folding / Canonicalize
  -> Pattern Fusion and Partition
  -> Liveness / Memory Planning
  -> Tile Planning
  -> Region Dependency Analysis
  -> TISA semantic builder / TISAProgram Verify
  -> Backend payload lowering / BackendArtifact Verify
```

“统一流程”不意味着所有算子使用同一个 lowerer。Matmul、Reduce、Softmax、Conv2D 仍然需要语义专用 lowering，但由 PassManager 通过 registry 调用；benchmark 只提供输入模型和 case 参数，不再负责 lower 顺序、buffer handoff 或 scheduler 分支。

### 4.3 Compiler、Backend Codegen 与 Runtime 的契约

Compiler 输出：

```text
TISAProgram
  - semantic tile instruction stream (one or more EU-bound instructions per tile)
  - OpType and operator/tile provenance
  - Operand(TileShape/TileMem/AccessType)
  - typed Deps(RAW/WAR/WAW + condition)
  - UnitMap(resource class, quantity, affinity)
  - logical buffer regions and runtime-bindable address expressions
  - backend lowering/timing keys
```

Backend Codegen 输入 `TISAProgram` 后输出 target artifact：

```text
BackendArtifact
  - TISA scheduling descriptor retained for each tile instruction
  - backend execution payload/opcode/kernel reference
  - descriptor-to-payload association
  - binary/layout/serialization metadata
```

Simulator backend 可以把 artifact 表示为内存中的 descriptor + primitive template；真实 NPU backend 可以通过 LLVM/custom codegen 将 TISA metadata 和 unit execution payload 编入 binary；CPU backend 可以串行执行同一语义用于功能验证。

Runtime/Loader 输入 `BackendArtifact` 后输出：

```text
RuntimeSubmission
  - concrete shape/state bindings
  - physical buffer addresses
  - command-buffer chunks
  - software submission order
  - launch/event metadata
```

TISA Device Scheduler 只能消费 `RuntimeSubmission` 中的 TISA descriptor，不能反向修改 OperatorGraph 或 ScheduleSpec。Backend execution unit 只执行与 descriptor 关联的 payload。需要改变 tile、fusion 或 residency 时，必须重新走 compiler pipeline。

## 5. Operator Lowering Registry

每个算子 lowering 插件负责：

1. 计算 tile 的输入输出 region；
2. 根据 residency 生成 transfer；
3. 生成 compute/reduce primitive；
4. 生成 tile 内依赖；
5. 记录 resource、duration/II hint 和 timing key，具体 timing 由可插拔 TimingProvider 在 device simulation 时解析。

Backend lowering 和 device scheduler 都不包含算子专用分支：baseline scheduler 消费统一的 `ExecutionTask`，TISA target scheduler 消费统一的 `TISAInstruction`；semantic operator-specific logic 只存在于 compiler lowering registry。

当前 registry 已支持 `matmul/batched_matmul/gemv`、`elementwise/residual_add`、`reduce`、`softmax`、`rmsnorm` 和 `layernorm`。`lower_mixed_graph` 对 heterogeneous graph 按拓扑逐算子调用插件，再将每个插件的任务合并为一个 ExecutionGraph：

```text
Operator A store(root, tensor T)
       -- DataEdge(T) + BufferRegion overlap -->
Operator B load(root, tensor T)
```

这是一版保守的跨算子 handoff，明确要求上游写回 root memory、下游从 root memory 读取。它适合先比较调度策略，避免 scheduler 猜测 tensor 地址；后续 mapping/residency 优化可以替换为 local-memory handoff 或真正的 fusion lowering，而不改动 policy 接口。

首批 lowering 顺序：

```text
Matmul -> 2mm -> Elementwise -> Reduce/Softmax -> Attention
```

## 6. 依赖模型

Execution Graph 同时保留三类依赖：

- 显式数据流依赖：producer tile 到 consumer tile；
- 编译期顺序约束：stage、barrier、固定 program order；
- 地址范围依赖：RAW、WAR、WAW。

TISA IR 还必须显式保留论文中的 typed dependency：

```text
Deps = (source, type, condition)
type      RAW / WAR / WAW
condition full-region-ready / partial-region-ready / barrier / token
```

第一版可以根据精确 tensor region 自动推导 `RAW/WAR/WAW`，但不能只把它压缩为字符串 predecessor。TISA device backend 再将同一 metadata 放进 per-unit waiting table、issue queue 或 address scoreboard，在条件满足时唤醒 ready tile。

## 7. SchedulerPolicy

Device scheduler policy 接受同一个 `RuntimeSubmission`/`TISAProgram` 和 `MachineConfig`；Runtime policy 独立接受 `TISAProgram`、buffer state 和 runtime config。

### 7.1 Sequential

按 iteration-major/program order 执行，前一个 iteration 完成后才推进下一个，作为最保守基线。

### 7.2 StaticPipeline

编译期确定：

```text
task order
resource reservation
stage offset
optional modulo initiation interval
```

支持 dual-stage 和 triple-stage，但 stage 名称来自 graph，不写死 `M0/S/M1`。`StaticPipelineConfig` 可用 `stage_offsets + initiation_interval_cycles + task.attributes[iteration]` 生成 reservation，也可用 `task_issue_cycles` 精确指定每条指令的 issue cycle。没有 reservation 配置时，`static_pipeline` 保持 deterministic program-order list scheduling，便于和旧基线对照。

### 7.3 DynamicReadyQueue

运行时维护 ready queue、resource queue 和 in-flight window。候选 priority 至少包括：

- oldest-first；
- critical-path-first；
- resource-locality-first；
- iteration-first，作为静态顺序近似对照。

当前 simulator 将可执行的动态启发式显式化为 `SimulatorConfig.dynamic_priority`：`critical_path`（默认）或 `oldest_first`。这使得 softmax 等多阶段 DAG 可以把“动态 ready queue 机制”和“具体优先级函数”作为两个独立实验维度报告。

TISA-like policy 现在支持可选 device-side address range scoreboard、窗口大小和 completion wake-up；scoreboard 不改写编译期图，而是只追踪 active task 的地址范围，因此能在 trace 中区分编译期依赖与设备运行时 address stall。

## 8. Runtime 与 Device Backend

### 8.1 Runtime Policy

这里需要区分论文术语和系统层次。论文称其为 “runtime scheduler”，但 Section III/V 的关键实现是 AI-core 内的 hardware-consumed scheduler：论文明确给出 7--9 cycle 的 dispatch budget、per-unit WQ/IQ/in-flight table 和 RTL synthesis。宿主软件 runtime 只负责把带 TISA metadata 的 descriptor/command stream 送入接口；真正的 tile reorder、hazard check 和 issue 属于 Device Backend。

本项目的 host/runtime policy 研究软件层的动态行为：

```text
submission order
command-buffer chunk size
software ready queue
buffer allocation/reuse
launch latency
host/device synchronization
```

它可以选择先提交哪个 kernel/tile batch，但不会替代设备内部的 TISA issue policy。若暂时把 host runtime 开销设为零，仍然可以独立复现论文的 hardware dynamic scheduler；只有加入 command submission、launch latency 和 buffer allocation 后，才研究 compiler/runtime/device 的端到端总周期。

### 8.2 Device Scheduler Policy

Device scheduler 继续使用现有的：

```text
sequential
static_pipeline
dynamic_ready_queue
```

`compile-model` 默认从已进入设备窗口的 `TISAInstruction` 中选择下一条可发射 tile；
`--scheduler-target primitive` 保留原来从 `ExecutionTask` 选择 task 的兼容 baseline。
TISA instruction issue 后，绑定 payload 才在所选 EU instance 上按本地拓扑顺序运行，
payload primitive 不进入全局 OOO window。Static/Dynamic 共用同一份 `TISAProgram`、
`BackendArtifact`、地址、依赖、timing 和 MachineConfig；RuntimeSubmission 尚未接入，
当前等价于一次性提交整个 descriptor stream。

### 8.3 热插拔 Backend 分层

Backend 不作为单一巨型类实现，而是拆成可替换的四种能力：

```text
CodegenBackend
  TISAProgram -> descriptor + target execution payload

TimingProvider
  task/resource -> duration + initiation interval

EventBackend
  issue/start/complete
  queue/ROB/scoreboard
  memory port/bank and event timing

SystemBackend (optional)
  runtime submission
  DMA/host/device interaction
  full-system memory and synchronization
```

第一版实现：

```text
SimulatorCodegenBackend
    + TISA descriptor / primitive template artifact
    +
AnalyticalTimingProvider
    + Current DiscreteEventBackend
```

后续可插拔实现：

```text
LLVMNpuCodegenBackend        # TISA metadata + per-unit binary payload
CpuValidationCodegenBackend # serial functional validation
SCALE-SimTimingProvider       # MXU/systolic timing
Ramulator/DRAMSysProvider     # DRAM timing
RTL/VerilatorEventBackend     # unit/bank/handshake calibration
Gem5/SALAMSystemBackend       # CPU + NPU full-system study
```

所有 backend 都必须遵守同一份 `TISAProgram/RuntimeSubmission` 和 trace schema；backend 不能改变 scheduler policy 的语义。Manifest 必须记录：

```text
frontend
compiler_pipeline
codegen_backend
runtime_backend
device_backend
timing_provider
machine_hash
calibration_status
```

## 9. Discrete-event Simulator

核心事件：

```text
ISSUE
START
COMPLETE
WAKE_UP
STALL_BEGIN / STALL_END
```

Simulator 维护：

- resource instance 与 next-issue time；
- resource queue occupancy；
- task dependency count；
- in-flight/ROB state；
- buffer occupancy 和 live allocation；
- event priority queue；
- deterministic tie-break order。

每个 task 的开始时间由依赖、资源、queue、initiation interval 和 buffer 状态共同决定。

当前实现将这一层拆成两个接口：

```text
SchedulerPolicy
  -> baseline: 从 visible ready queue 选择 ExecutionTask
  -> TISA target: 从 descriptor/WQ visible window 选择 TISAInstruction

TimingModel / EventBackend
  -> baseline: 计算 ExecutionTask issue/start/complete
  -> TISA target: 将已 issue 的 TISAInstruction 展开为 instruction-local backend primitive timing
  -> 维护 unit queue、II、ROB、dependency window、tile window
  -> 在 COMPLETE 时唤醒后继 task
```

`AnalyticalTimingModel` 只消费 `ExecutionTask.duration_cycles` 和 `MachineConfig` 的 unit 默认值；它是可替换的 timing provider，不代表真实硬件。TISA target 把一个 payload 内 primitive duration 按本地拓扑顺序相加，形成该 instruction 的 run-to-completion duration。`SimulatorConfig` 可以覆盖 instruction queue、ROB、ready queue、dependency window 和 max in-flight tiles。

`TimingTableModel` 是第一种外部校准入口：它从 JSON 读取 primitive/resource/task 的 duration 和 initiation interval，未命中的 task 回退到 `AnalyticalTimingModel`。因此 SCALE-Sim、RTL waveform 或硬件 counter 的局部结果可以先转换为 table，再通过同一 scheduler 重放；manifest 的 `backend` 会区分 table 名称和 analytical。

`--address-scoreboard` 启用 device-side range scoreboard：根据相同 `tensor/memory` 上的 `BufferRegion` 重叠关系，在 issue 前检查 active task 并产生 RAW/WAR/WAW stall；COMPLETE 后释放范围并继续调度。当前地址来自 canonical `ExecutionTask` metadata，尚不是从真实 TISA binary 动态解析出的硬件 scoreboard。

## 10. Trace 与结果

一次仿真至少输出：

```text
manifest.json
artifact_index.json
00_frontend/{model_spec,benchmark_case,model_instance,frontend_import}.json
00_frontend/{generated.mlir,stablehlo_module.json,source_frontend_import.json}  # round-trip 路径
01_graph_ir/{operator_graph.json,operator_graph.svg,operator_graph.dot}
02_schedule_tile/{schedule.json,tile_graph.json,tile_graph.dot}
03_tisa/{tisa_program.json,compiled_artifact.json}
04_backend/{machine.json,backend_artifact.json,execution_graph.json}
05_runtime/address_dependencies.json
06_simulation/{summary.json,tasks.csv,tisa_instructions.csv}
07_trace/{perfetto.json,swimlane.svg,swimlane.png}
```

artifact 只按编号目录组织，不在顶层生成同名副本或符号链接；复用旧输出目录时会清理
历史版本留下的已知扁平 artifact。单次实验顶层只保留 `README.md`、`manifest.json` 和
`artifact_index.json`，后者记录每个阶段实际生成的文件。这样可以按
`frontend -> graph -> schedule/tile -> TISA -> backend -> runtime -> simulation/trace`
顺序检查一次实验，而不必在一个目录中区分几十个平级文件。

`events.csv` 使用 cycle-native 字段：

```text
instruction_id, task_id, tile_id, operator, OpType
unit_class, unit_instance, UnitMap
issue_cycle, start_cycle, end_cycle
wait_reason, typed_deps, predecessors
operand_regions, memory_scopes
```

`trace.json` 使用 Chrome Trace Event `X` 事件，可直接在 Perfetto 中按 resource/resource instance 展示。

`summary.json` 至少包含：

```text
total_cycles
speedup_vs_baseline
resource_utilization
stall_cycles_by_reason
pipeline_drain_cycles
queue_occupancy_timeline
queue_peak_occupancy
buffer_peak_usage
completed_tile_count
runtime_submit_cycles
runtime_launch_count
command_buffer_bytes
device_cycles
runtime/device synchronization stalls
```

## 11. 公平比较约束

比较 device Static 与 Dynamic 时，以下内容必须一致：

- workload 和 tensor layout；
- tile decomposition；
- buffer allocation；
- tensor address/range；
- dependency graph；
- latency model；
- hardware resource count；
- simulation seed 和 tie-break rule。

此外，runtime policy、command-buffer chunk size、buffer allocation policy 和 launch overhead 必须作为单独实验键。推荐至少保留四种组合：

```text
static runtime + static device
static runtime + dynamic device
dynamic runtime + static device
dynamic runtime + dynamic device
```

这样才能区分收益来自软件提交顺序，还是来自 TISA hardware issue。若同时改变 tile、地址分配或硬件，结果必须作为独立实验维度报告。

## 12. 校准边界

第一版是 architecture exploration model。验证分三层：

1. 手算 micro-case：验证 scheduler/simulator 语义；
2. TileFlow/SCALE-Sim：对账计算量、搬运量和 aggregate 趋势；
3. Verilator/RTL/hardware counters：校准 latency、queue、bank conflict 和 observed cycle。

只有第三层完成后，相关结果才可标记为 RTL-observed 或 cycle-accurate。
