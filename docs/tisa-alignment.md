# TISA 论文语义对齐

论文 TISA 与项目的 `TileInstance` 位于相近的 tile 粒度。TileInstance 表达切分范围；
`TISAInstruction` 在此基础上携带 scheduler 所需的 operand、地址、访问类型、依赖和
execution-unit 映射。DMA、MXU、Vector 微指令属于 TISA payload 的下一层。

## 1. 抽象层次

论文字段：

```text
Operand = (TileShape, TileMem, AccessType)
TileMem = (base, scope)
TISA_Inst = (OpType, Operands, Attributes, UnitMap)
Deps = (src, RAW | WAR | WAW | STATE | ACCUMULATE, condition)
```

项目对应层次：

```text
PyTorch / StableHLO graph
        |
        v
GC: graph optimization + tiling + typed tile dependency
        |
        v
software-scheduled Semantic TileGraph
        |
        v
FC: TISA dialect
        |
        v
virtual TISAInstruction          target-independent
        |
        v
TargetPlan / target TISA         device scheduler 可见
        |
        v
backend ExecutionTask payload    DMA/MXU/Vector 单元内部执行
```

FC 用抽象 transfer/compute 表达 semantic tile；一条抽象 transfer 若跨越多个 target EU，
由 FC 之后的 Target Lowering 展开为多条 scheduler-visible TISA，并用 typed dependency
串联。每条 target instruction 绑定一个主要 UnitMap，payload 只包含该 EU 内部的步骤。

论文明确给出 TISA operand、GC/FC/TISA generator 的抽象关系以及硬件 scheduler 的调度
粒度，但没有公开 Epoch generator/backend 内部由谁完成全部 buffer allocation。本项目将
目标 placement 与有限容量 allocation 放在 CodegenBackend 边界，是为保证契约可审计而作
的实现选择，不把它表述为 Epoch 的已证实内部实现。

## 2. 字段对应

| 论文概念 | 当前实现 | 说明 |
| --- | --- | --- |
| tile bounds | `TileInstance.bounds` | 静态 shape、边界 tile 和 logical region |
| OpType | `TISAInstruction.op_type`、`semantic_family` | 复合语义保持 semantic op，stage 按 EU 划分 |
| TileShape | `TISAOperand.tile_shape` | resolved shape；symbolic binding 使用 normalized shape environment |
| TileMem scope | `scope` | `program`、`operator:<id>`、`tile:<id>` 等逻辑所有权范围 |
| visibility | `visibility/owner/domain` | Private/Local/Shared 可见性、owner 和资源域，不等同具体 memory |
| operand role | `role` | lhs/rhs/output/state/scratch；参与目标 memory 选择 |
| symbolic buffer | `symbolic_buffer_id` | FC source、tile buffer、accumulator 的逻辑身份 |
| target memory | `memory_space` | 只在 Target Lowering 后出现 GM/UB/LMB/RMB/PSB 等实例 |
| allocation | `buffer_id/allocation_id` | target copy 与最终物理分配身份 |
| AccessType | operand/buffer access | read、write、read-write |
| Attributes | `TISAInstruction.attributes` | readiness、region、state、fusion 和 reorder |
| UnitMap | `TISAInstruction.unit_map` | execution unit 类别与数量 |
| typed Deps | `TISADependency` | kind、condition、provenance；GC 同时保存 logical region |
| WQ/IQ/Fu | `cycle_event` 的 per-EU 队列和 operand tracking | 时钟边界与容量显式可配；ROB 为项目扩展 |

memory bank scoreboard 读取最终 Runtime operand 的 physical scope/address，并结合
MachineConfig 的 bank、width、read/write port 形成 analytical structural reservation。

## 3. 编译路径

```text
PyTorch nn.Module
  -> torch.export.ExportedProgram
  -> Torch-XLA StableHLO
  -> official StableHLO parse/verify
  -> GCArtifact / Semantic TileGraph
  -> TISADialectProgram
  -> virtual TISAProgram
  -> TargetPlan / target TISA
  -> backend payload / MemoryPlan
  -> final TISAProgram + BackendArtifact
```

Torch-XLA 负责 ATen 到 StableHLO。项目维护 semantic family 到 Canonical/TISA/backend
capability 的映射。复合算子 recovery 依据图结构、shape、常量和数据流证明，模型名称
作为 provenance 字段。

`softmax_algorithm` 是 Softmax lowering 属性：materialized 生成完整中间结果，
online 生成 reduction tile 的 `(max, sum)` state chain。它与 static/dynamic device
policy 独立，沿 GC、FC 和 backend payload 传播。

## 4. Scheduler 的输入

全局 scheduler 处理 loader 已绑定的 `BoundTISADescriptor`；其中复用同一个
`TISAInstruction` 语义，不另造重复 ISA：

```text
Runtime loader
  -> TISAInstruction + 本次地址/依赖/completion token + arrival envelope
TISA ready
  -> Deps、UnitMap、queue/ROB、已接收地址冲突检查
  -> issue 整条 instruction
  -> ExecutionBackend 执行 payload 并返回 physical-done/partial-ready
  -> completion 带宽/延迟接受后唤醒后继 TISA
```

在 minimal profile 上，Matmul tile 形成：

```text
TISA DMA(DRAM -> SRAM) -> TISA MXU(SRAM) -> TISA DMA(SRAM -> DRAM)
```

在 lpu-like profile 上，显式 placement 为：

```text
lhs:    GM -> UB -> LMB
rhs:    GM -> UB -> RMB
output: PSB -> UB -> GM

TISA GDMA -> TISA LDMA -> TISA MXU(LMB,RMB -> PSB)
          -> TISA ARU -> TISA GDMA
```

FC 对 minimal 和 lpu-like 都只生成 `load → tensor compute → store`。多跳 route 在
Target Lowering 阶段按 EU 边界展开；TargetPlan 同时生成 target operands 与
abstract→target 映射，Matmul payload 直接消费该计划，不从 task graph 反推边界。

Softmax 的 `reduce_max/exp/reduce_sum/normalize` 属于同一 VE payload。payload lane
事件进入 timing 和泳道图，TISA 依赖保持全局可见。

阶段 4 的最小 Attention 片上交接是项目 target-lowering 优化：当 QK Matmul 的最终片上
output 与 Softmax input 在 memory、role、layout、dtype 和 tile geometry 上完全兼容且
fan-out=1 时，TargetPlan 可以消除 root store/load，并将 Softmax 依赖改接到片上 producer
completion。论文支持 locality-aware tile flow，但没有公开这一项目规则的具体判定与归属，
因此结果不标为作者实现的逐项复现。

## 5. Runtime 与 device scheduler

```text
Host Runtime
  消费 MemoryPlan，绑定地址空间基址、动态参数、command chunk、descriptor arrival、同步

TISA Device Scheduler
  reception、WQ/IQ、依赖检查、资源检查、OOO issue、completion

Execution Backend
  已加载 payload 的 can-accept、EU busy/II、task trace 和物理完成反馈
```

论文的 tile-by-tile OOO 决策位于 device hardware；runtime 控制 descriptor 的可见时间。
项目用 `--runtime-policy` 和 `--policy` 分别研究两层，`--runtime-device-matrix`
一次编译后运行四种组合。

逐周期配置与论文公开机制/项目假设的区分见 [device-scheduler.md](device-scheduler.md)。

## 6. 当前实现与扩展项

当前实现：

- PyTorch -> Torch-XLA -> official StableHLO -> GC/FC/TISA；
- semantic Softmax/Norm、Attention、SwiGLU、RoPE 和 KV-cache region；
- embedding gather 与 scheduler-visible MoE dispatch region；
- instruction-level static/dynamic scheduler；
- descriptor arrival、queue/ROB/window、resource、completion feedback analytical model；
- invocation-scoped bound descriptor、提交 envelope、静态计划和 runtime alias token；
- scheduler/execution 分离：设备仲裁器不读取 ExecutionGraph/payload primitive；
- analytical、timing table、systolic MXU profile 和 RTL importer；
- `payload_ready:<task_id>` partial-ready 原型和 memory bank scoreboard。
- GC `TileDependency` 的 hazard kind、logical region、condition 和 provenance，并向
  TISA dependency 与 compile statistics 贯通。
- `MachineConfig.operation_placements`、多跳 transfer stage 和 target `MemoryPlan`；
- FC 符号 source/destination、Private/Local/Shared、role/owner/domain；
- `TargetPlan` 的 abstract→target mapping、route hop/engine、target operands 和 provenance；
- TISA、payload 与 RuntimeSubmission 共享 `buffer_id/allocation_id`，局部 slot 的复用由
  RAW/WAR/WAW/BUFFER_REUSE 依赖保护；
- compile package schema v2 与 topology hash 检查。

扩展项：

- 论文全部 operation/model block 的 semantic coverage；
- 更复杂 StableHLO layout dialect 的扩展与 bank-aware memory timing 校准；
- 论文 WQ/IQ/Fu 容量、dispatch/wake-up/issue/completion 控制开销；
- 完整 RTL 与真实芯片 timing calibration；
- online Softmax 数值 rescale、最终 normalization 和 workspace 生命周期。
- Matmul 之外的 operation 使用显式 operation-class placement；复合算子 payload 仍经
  TargetPlan recipe adapter，逐算子专用多跳 target codegen 是后续能力；
- bank-aware placement 优化、fragmentation/eviction、跨 core memory routing；

当前结果标签为 `TISA instruction-level analytical scheduling baseline`。profile 加载后，
manifest 记录对应 calibration status，trace schema 保持一致。
