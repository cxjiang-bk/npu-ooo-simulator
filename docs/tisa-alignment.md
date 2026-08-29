# TISA 论文语义对齐

TISA 与项目中的 `TileInstance` 处于相近粒度，但两者不是同一个对象。`TileInstance` 描述计算切分；`TISAInstruction` 在此基础上增加硬件调度所需的 operand、地址、访问类型、依赖和 execution-unit 映射。TISA 也不是最底层 DMA/MXU/Vector 微指令。

## 1. 抽象层次

论文中的核心字段是：

```text
Operand = (TileShape, TileMem, AccessType)
TileMem = (base, scope)
TISA_Inst = (OpType, Operands, Attributes, UnitMap)
Deps = (src, RAW | WAR | WAW | STATE | ACCUMULATE, condition)
```

对应的编译和执行层次是：

```text
PyTorch / StableHLO graph
        |
        v
Graph Compiler (GC): graph optimization + tiling + typed tile dependencies
        |
        v
software-scheduled TileGraph
        |
        v
Fusion Compiler (FC): TISA dialect
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
| `OpType` | `TISAInstruction.op_type`、`semantic_family` | 复合算子保持 semantic op，按 EU stage 拆分 |
| `TileShape` | `TISAOperand.tile_shape` | 已支持 resolved shape；symbolic shape 仍需完整 binding/legalization |
| `TileMem` | `TISAOperand.tile_mem` | 已有 scope、逻辑 `address_expr`、concrete offset/size、`strides_bytes`/`stride_expr`/`layout`；静态 broadcast 按输出 tile 映射源 region；未知 StableHLO encoding 保留 `layout_encoding` 并采用 conservative interval；dtype 能力由 machine strict/fallback policy 统一校验 |
| `AccessType` | operand/buffer access | 已支持 read/write/read-write |
| `Attributes` | `TISAInstruction.attributes` | 已记录 stage readiness；simulator 支持显式 `payload_ready:<task_id>` partial-ready 原型 |
| `UnitMap` | `TISAInstruction.unit_map` | analytical backend 当前主要验证 quantity=1 |
| typed `Deps` | `TISADependency` | 已有 RAW/WAR/WAW/STATE/ACCUMULATE 与显式 readiness condition |
| WQ/IQ/Fu | TISA simulator queue/ROB/in-flight state | 是行为模型，不宣称 RTL 微结构等价 |

可选的 memory bank scoreboard 根据 `MachineConfig.memory_levels` 的 bank 数量、bank
宽度和读写端口数建立 active reservation。它用于研究 TileMem 访问造成的结构性
阻塞，默认关闭，且不替代真实 memory timing backend；对应结果应标注为 analytical
structural-conflict model。

## 3. 当前唯一编译路径

```text
PyTorch nn.Module
  -> torch.export.ExportedProgram
  -> Torch-XLA StableHLO
  -> official StableHLO parse/verify
  -> Graph Compiler (GC)
  -> GCArtifact / software-scheduled TileGraph
  -> Fusion Compiler (FC)
  -> TISADialectProgram
  -> TISA Generator
  -> TISAProgram
  -> CodegenBackend
  -> BackendArtifact(TISA descriptors + bound payload)
```

Torch-XLA 负责 ATen 到 StableHLO 的 legalization。项目只维护 StableHLO semantic family 到 Canonical/TISA/backend capability 的映射，不接受手写 StableHLO 文件，也不维护另一套项目自有 StableHLO emitter。

Torch-XLA 可能把 Softmax、Norm 等复合算子展开成基础 StableHLO operations。项目的 recovery/fusion pass 根据图结构、shape 和常量恢复调度所需的语义边界，不能按模型名或 factory 名匹配。

GC 的 `softmax_algorithm` 只是 Softmax lowering 的算法属性：`materialized` 对应完整
中间结果 materialization，`online` 对应跨 reduction tile 的状态链。它与 device
侧 `static_pipeline`/`dynamic_ready_queue` 是两条独立配置轴；同一 Softmax algorithm
可以分别用两种 scheduler 比较。

## 4. Scheduler 调度什么

全局 static/dynamic scheduler 只调度 `TISAInstruction`，不调度独立的 backend `ExecutionTask`。FC 对 Softmax、RMSNorm、LayerNorm 等复合算子只生成语义 TISA operation；`reduce_max/exp/reduce_sum/normalize` 等步骤属于该 instruction 的 backend-local payload：

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

其中一条 `TISA matmul` 或 `TISA softmax` 的 analytical payload 仍可包含 execution-unit 内部 timing task，但这些 task 不会重新进入全局 OOO ready queue。因此不同 tile 的内部步骤不会绕过 TISA dependency 被任意重排。

实现边界需要特别区分：默认 analytical Softmax payload 是 materialized
row-wise `max/sum -> exp -> normalize`，并非论文式 online state update。
显式配置 `softmax_algorithm=online` 时，项目提供一个 scheduler-level analytical
state-chain payload：每个 reduction tile 的 `online_update` 读取并更新前一 tile
的 `(max, sum)` 状态。它用于观察状态依赖对 OOO 调度的影响，不执行完整数值算法的
rescale、最终 normalization 和 workspace 管理；跨 tile 的 primitive 边仍只属于
backend payload，只有 GC/FC 生成的 TISA `STATE` 边进入全局 scheduler。

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
- 论文 GC、FC、TISA Generator 的显式 Python 语义边界与阶段产物；
- 复合 Softmax/Norm 的 semantic TISA instruction 与 backend-local primitive payload 分离；
- TISA instruction 粒度的 static/dynamic scheduler；
- descriptor arrival、queue/ROB/window、资源状态和 completion feedback 的 analytical 模型；
- instruction 与 payload 两层 trace；
- analytical、table 和 MXU RTL profile timing source。

尚未完成：

- 论文全部 operation/模型 block 的 semantic coverage；
- GC 自动生成的通用 partial-ready、精确 typed hazard 和 memory bank/port conflict；
- 与论文实现一致的 WQ/IQ/Fu 容量和 dispatch pipeline；
- 完整 RTL 或真实芯片时序校准。

动态索引边界：shape specialization 已将可求值的常量起点 `dynamic_slice` 按
StableHLO clamp 语义改写为静态 `slice`，因此不会把动态 start 误当作静态地址；未求值
的 start 和 `dynamic_update_slice` 仍在 importer 前显式失败。后者需要 runtime state、
地址绑定和跨 invocation 语义共同确定，不能仅靠文本替换支持。

常量 shape tensor 驱动的 `dynamic_reshape` 已在同一 specialization pass 中静态化，只有
输入/输出元素总数守恒且目标维度均为正整数时才会改写为官方 `reshape`；运行时 shape
仍保持显式失败。

因此当前结果应称为“TISA instruction-level analytical scheduling baseline”。默认编译
路径仍是 completion-boundary 语义；`payload_ready:<task_id>` 只作为校准 backend
和 micro-test 的 partial-ready 原型，不能称为论文硬件的 cycle-accurate 复现。
