# TISA 对齐说明

本文档记录论文《Dynamic Scheduling for AI Accelerators via TISA》与本项目 IR/调度器的对应关系。结论先行：TISA 与本项目的 `TileInstance` 处于相近的 tile 粒度，但 TISA 不是裸 tile 坐标，也不是最终的 MXU/Vector/DMA 微指令；它是由硬件消费的、带调度语义的 tile-level virtual ISA。TISA descriptor 和 backend execution payload 可以在编译阶段一起生成，但只有 descriptor 进入 TISA scheduler 的可见调度状态。

## 1. 论文中的抽象层次

论文定义：

```text
Operand = (TileShape, TileMem, AccessType)
TileMem = (base, scope)

TISA_Inst = (OpType, Operands, Attributes, UnitMap)
Deps = (src, type, condition)
type = RAW | WAR | WAW
```

语义含义是：

```text
TileShape
    这个 tile 计算哪个逻辑范围，边界 tile 可以是非满 tile

TileMem
    这个 tile 的 operand 位于哪个地址范围和 memory scope

AccessType
    对 operand 是读、写还是读写

OpType
    语义身份，例如 GEMM、SOFTMAX；不是简单的 resource/opcode 字符串

Attributes
    reorder constraint、同步要求、partial-ready 条件等

UnitMap
    允许的 unit class、数量和 affinity

Deps
    typed dependency 以及满足依赖的条件
```

TISA 的位置可以表示为：

```text
High-level framework / StableHLO
        |
        v
Graph + Fusion + Tiling compiler
        |
        v
TISA semantic tile instruction       <-- 论文的调度契约
        |
        v
per-unit execution ISA / DMA-MXU-Vector micro-events
```

论文明确描述 compiler 在 tile granularity 截止 lowering；TISA instruction 在 Epoch 上还有 binary encoding，但它补充而不是替代 per-unit execution ISA。

论文的 framework bridge 选择 torchxla -> XLA/StableHLO 还有一个重要含义：StableHLO 不只是导入格式，而是用来保持跨框架的 semantic OpType。ExecuTorch 可以作为当前 PyTorch 入口，但 adapter 必须保留高层/composite provenance；如果只输出碎片化 Core ATen primitive，仍然会复现论文所批评的 semantic erosion。长期可以采用 `ExecuTorch -> Canonical OperatorGraph` 或 `ExecuTorch -> StableHLO -> Canonical OperatorGraph` 两条入口，但二者必须汇聚到同一 TISA semantic taxonomy。

## 2. 与当前 IR 的逐项对照

| 论文概念 | 当前项目 | 判断 |
|---|---|---|
| tile computational bounds | `TileInstance.bounds` | 基本对应，但目前只按 operator schedule 展开 |
| tile index/provenance | `TileInstance.coordinates`, `operator_id` | 已有，需要补 model/layer/template provenance 传递 |
| `OpType` | `OperatorSpec.normalized_type` | 语义来源已有，但没有复制到独立的 scheduler instruction |
| `TileShape` | `TileInstance.bounds` + operand shape | 部分对应，尚未按每个 operand 表达 |
| `TileMem(base, scope)` | `BufferRegion` | 最接近，但当前 base 主要是 offset/starts，scope 与 address expression 不结构化 |
| `AccessType` | `BufferRegion.access` | 已有 `READ/WRITE/READ_WRITE`，可直接演进 |
| `Attributes` | `OperatorSpec.attributes` / `ExecutionTask.attributes` | 分散保存，缺少 TISA reorder/sync/partial-ready schema |
| `UnitMap` | `ExecutionTask.resource` | 不足；只有一个 resource 名称，没有 quantity/affinity/合法 unit class 集合 |
| typed `Deps` | `TileDependency` / `ExecutionTask.predecessors` | 不足；当前 predecessor 是无类型字符串边 |
| hardware in-flight semantic table `Fu` | simulator active task/address scoreboard | 部分对应，但当前跟踪的是 primitive task，不是 TISA instruction |
| per-unit WQ/IQ | simulator ready set/resource state | 行为上部分对应，结构上还没有 TISA reception/WQ/IQ 层 |

## 3. 当前设计的真实问题

当前路径是：

```text
TileInstance
    -> operator-specific lowering
    -> ExecutionTask
    -> Static/Dynamic scheduler
```

论文对齐后的目标路径应为：

```text
TileInstance
    -> TISAInstruction
    -> backend codegen: descriptor + execution payload
    -> runtime/loader submission
    -> TISA device scheduler reads descriptor
    -> execution unit runs backend payload
    -> ExecutionTask timing/events
```

如果直接在 `ExecutionTask` 粒度做动态 issue，会产生三个偏差：

1. 一个论文 tile 被拆成多个 primitive 后，scheduler 可以在 tile 内部重新排序，违反论文的 run-to-complete、non-preemptive tile boundary 语义；
2. `SOFTMAX/GEMM/ATTENTION` 的 operator identity、UnitMap 和 typed dependency 在 primitive lowering 时被打散，scheduler 无法按 semantic compatibility 做合法性判断；
3. `BufferRegion` 的地址重叠只表示一种 hazard 结果，无法区分 compiler typed dependency、TISA condition 和 device-observed conflict。

这并不意味着当前 `ExecutionTask` 没有价值。它应保留为 backend timing/event 层，用来表达：

```text
TISA GEMM tile
    -> DMA load A
    -> DMA load B
    -> MXU execution
    -> optional writeback
```

真实编译器可以在 runtime 之前生成这些 execution payload，但 scheduler 只能根据与 payload 绑定的 TISA descriptor 发射整个 tile instruction。在 simulator 中可以预先展开 primitive template，但必须以一个不可被跨 tile 重排的 instruction group 执行，不能把 primitive graph 当成 TISA scheduler 的唯一输入。

## 4. 动态调度属于哪一层

论文中的 “runtime scheduler” 容易造成误解。原文同时给出：

```text
software/runtime interface:
    binary emits per-tile descriptors
    descriptors populate waiting/issue queues

AI-core hardware scheduler:
    reception buffer
    per-unit WQ/IQ
    semantic conflict check
    in-flight table Fu
    out-of-order issue
    completion feedback
```

论文还报告了 7--9 cycle 的 tile dispatch 和 RTL synthesis，并指出控制处理器上的 software runtime 需要 microsecond 级开销，不能承担 tile-level dispatch。因此核心 dynamic reorder/issue 是硬件执行，host runtime 只负责提交带 TISA metadata 的 descriptor stream。

项目中应拆成：

```text
Host Runtime
    选择提交哪个 command chunk
    绑定物理地址和动态 state
    处理 launch/event/synchronization

TISA Device Scheduler
    接收 TISAInstruction
    路由到 per-unit WQ
    形成 IQ
    检查 typed deps / TileMem / UnitMap
    以 tile 为单位 issue 和 complete

Backend Timing/Event Layer
    将已 issue 的 TISA tile 展开为 DMA/MXU/ARU timing
```

## 5. 迁移顺序

不需要推翻当前 Model IR、OperatorGraph、ScheduleSpec、MachineConfig 和 analytical event backend。推荐按以下顺序修正：

```text
1. 冻结 TISAInstruction schema
   OpType, Operand, TileMem, AccessType, Attributes, UnitMap, Deps

2. 为现有 TileInstance 生成 TISAInstruction
   先覆盖 2mm、elementwise、softmax

3. 把 typed Deps 从 region overlap 推导出来
   保留 RAW/WAR/WAW 和 partial/full readiness

4. 新增 TISA scheduler mode
   scheduler 选择 TISAInstruction，而不是 primitive ExecutionTask

5. 为每个 device backend 实现 TISA -> primitive expansion
   codegen 生成 descriptor + execution payload
   analytical、SCALE-Sim、RTL/Verilator 分别实现自己的 expansion/timing

6. 保留 ExecutionTask trace
   同时新增 TISA issue/complete trace，区分 semantic tile 和 primitive lane
```

在第 4 步完成前，当前 dynamic 结果只能称为 `primitive-task analytical dynamic scheduling baseline`，不能直接宣称已经复现论文中的 TISA tile scheduler。
