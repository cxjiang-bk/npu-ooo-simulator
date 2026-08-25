# TISA 论文语义对齐

TISA 与项目中的 `TileInstance` 处于相近粒度，但两者不是同一个对象。`TileInstance` 描述计算切分；`TISAInstruction` 在此基础上增加硬件调度所需的 operand、地址、访问类型、依赖和 execution-unit 映射。TISA 也不是最底层 DMA/MXU/Vector 微指令。

## 1. 抽象层次

论文中的核心字段是：

```text
Operand = (TileShape, TileMem, AccessType)
TileMem = (base, scope)
TISA_Inst = (OpType, Operands, Attributes, UnitMap)
Deps = (src, RAW | WAR | WAW, condition)
```

对应的编译和执行层次是：

```text
PyTorch / StableHLO graph
        |
        v
graph optimization + tiling
        |
        v
TileInstance
        |
        v
TISAInstruction              device scheduler 可见
        |
        v
backend execution payload    DMA/MXU/Vector 等单元内部执行
```

一个 semantic tile 若跨多个 execution-unit 类型，compiler 会按 unit/stage 边界生成多条 TISA instruction，并用 typed dependency 串联。每条 instruction 只映射到一种主要 `UnitMap`，其 payload 可以包含该 execution unit 内部的多个执行步骤。

## 2. 当前字段对应

| 论文概念 | 当前实现 | 当前边界 |
| --- | --- | --- |
| tile bounds | `TileInstance.bounds` | 已支持静态 resolved shape 与边界 tile |
| `OpType` | `TISAInstruction.op_type`、`semantic_family` | 部分 composite 会按资源 stage 拆分 |
| `TileShape` | `TISAOperand.tile_shape` | 尚缺通用 symbolic shape binding |
| `TileMem` | `TISAOperand.tile_mem` | 已有 scope/offset/size，地址表达式仍需扩展 |
| `AccessType` | operand/buffer access | 已支持 read/write/read-write |
| `Attributes` | `TISAInstruction.attributes` | partial-ready/reorder 仍需继续校准 |
| `UnitMap` | `TISAInstruction.unit_map` | analytical backend 当前主要验证 quantity=1 |
| typed `Deps` | `TISADependency` | 已有 RAW/WAR/WAW 与 readiness condition |
| WQ/IQ/Fu | TISA simulator queue/ROB/in-flight state | 是行为模型，不宣称 RTL 微结构等价 |

## 3. 当前唯一编译路径

```text
PyTorch nn.Module
  -> torch.export.ExportedProgram
  -> Torch-XLA StableHLO
  -> official StableHLO parse/verify
  -> Canonical OperatorGraph
  -> ScheduleSpec / TileGraph
  -> TISAProgram
  -> CodegenBackend
  -> BackendArtifact(TISA descriptors + bound payload)
```

Torch-XLA 负责 ATen 到 StableHLO 的 legalization。项目只维护 StableHLO semantic family 到 Canonical/TISA/backend capability 的映射，不接受手写 StableHLO 文件，也不维护另一套项目自有 StableHLO emitter。

Torch-XLA 可能把 Softmax、Norm 等复合算子展开成基础 StableHLO operations。项目的 recovery/fusion pass 根据图结构、shape 和常量恢复调度所需的语义边界，不能按模型名或 factory 名匹配。

## 4. Scheduler 调度什么

全局 static/dynamic scheduler 只调度 `TISAInstruction`，不调度独立的 backend `ExecutionTask`：

```text
TISA instruction ready
  -> device scheduler 检查到达时间、Deps、UnitMap、queue/ROB 和地址冲突
  -> issue 整条 TISA instruction
  -> backend 在 instruction 边界内执行绑定 payload
  -> completion 唤醒后继 instruction
```

例如 Matmul tile 可以形成：

```text
TISA load-A -> TISA load-B -> TISA matmul -> TISA store
```

其中一条 `TISA matmul` 的 analytical payload 仍可包含 execution-unit 内部 timing task，但这个 task 不会重新进入全局 OOO ready queue。因此不同 tile 的内部步骤不会绕过 TISA dependency 被任意重排。

`ExecutionTask` 仍然有价值：它是 backend timing/event 表达，也是泳道图中 DMA、MXU、Vector lane 的来源。它不是论文 scheduler 的输入 ISA。

## 5. Runtime 与硬件动态调度

论文所称 runtime/interface 与核心硬件调度必须分开：

```text
Host Runtime
  绑定物理地址、组织 command chunk、控制 descriptor 到达和同步

TISA Device Scheduler
  接收 descriptor、进入 WQ/IQ、检查依赖与资源、OOO issue、处理 completion

Backend Timing/Event
  模拟已 issue instruction 在执行单元中的时序
```

论文报告 tile dispatch 的 cycle 级开销和 scheduler RTL synthesis，并指出软件控制处理器无法以微秒级开销承担 tile-by-tile reorder。因此论文的关键 OOO 决策属于 device hardware；host runtime 只影响 descriptor 何时可见。

项目用 `--runtime-policy` 研究第一层，用 `--policy` 研究第二层。`--runtime-device-matrix` 在一次编译后运行四种组合，避免把软件提交收益误算成 TISA device OOO 收益。

## 6. 当前完成度

已经完成：

- PyTorch 到 Torch-XLA/official StableHLO 的唯一前端；
- TileGraph 到 TISA descriptor 与 BackendArtifact；
- TISA instruction 粒度的 static/dynamic scheduler；
- descriptor arrival、queue/ROB/window、资源状态和 completion feedback 的 analytical 模型；
- instruction 与 payload 两层 trace；
- analytical、table 和 MXU RTL profile timing source。

尚未完成：

- 论文全部 operation/模型 block 的 semantic coverage；
- 通用 partial-ready、精确 typed hazard 和 memory bank/port conflict；
- 与论文实现一致的 WQ/IQ/Fu 容量和 dispatch pipeline；
- 完整 RTL 或真实芯片时序校准。

因此当前结果应称为“TISA instruction-level analytical scheduling baseline”，不能称为论文硬件的 cycle-accurate 复现。
