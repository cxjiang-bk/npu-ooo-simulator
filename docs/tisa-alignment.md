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
TISAInstruction                  device scheduler 可见
        |
        v
backend ExecutionTask payload    DMA/MXU/Vector 单元内部执行
```

一个 semantic tile 跨越多个 execution unit 时，compiler 按 unit/stage 边界生成多条
TISA instruction，并用 typed dependency 串联。每条 instruction 绑定一个主要 UnitMap；
payload 可以包含该 execution unit 内部的多个步骤。

## 2. 字段对应

| 论文概念 | 当前实现 | 说明 |
| --- | --- | --- |
| tile bounds | `TileInstance.bounds` | 静态 shape、边界 tile 和 logical region |
| OpType | `TISAInstruction.op_type`、`semantic_family` | 复合语义保持 semantic op，stage 按 EU 划分 |
| TileShape | `TISAOperand.tile_shape` | resolved shape；symbolic binding 属于扩展项 |
| TileMem | `TISAOperand.tile_mem` | scope、logical address expression、offset/size、stride/layout metadata |
| AccessType | operand/buffer access | read、write、read-write |
| Attributes | `TISAInstruction.attributes` | readiness、region、state、fusion 和 reorder |
| UnitMap | `TISAInstruction.unit_map` | execution unit 类别与数量 |
| typed Deps | `TISADependency` | RAW/WAR/WAW/STATE/ACCUMULATE/BUFFER_REUSE |
| WQ/IQ/Fu | simulator queue/ROB/in-flight | 可配置行为模型，参数来自 MachineConfig |

memory bank scoreboard 读取 MachineConfig 的 bank、width、read/write port，形成
analytical structural reservation，独立记录 memory conflict stall。

## 3. 编译路径

```text
PyTorch nn.Module
  -> torch.export.ExportedProgram
  -> Torch-XLA StableHLO
  -> official StableHLO parse/verify
  -> GCArtifact / Semantic TileGraph
  -> TISADialectProgram
  -> TISAProgram
  -> BackendArtifact
```

Torch-XLA 负责 ATen 到 StableHLO。项目维护 semantic family 到 Canonical/TISA/backend
capability 的映射。复合算子 recovery 依据图结构、shape、常量和数据流证明，模型名称
作为 provenance 字段。

`softmax_algorithm` 是 Softmax lowering 属性：materialized 生成完整中间结果，
online 生成 reduction tile 的 `(max, sum)` state chain。它与 static/dynamic device
policy 独立，沿 GC、FC 和 backend payload 传播。

## 4. Scheduler 的输入

全局 scheduler 处理 `TISAInstruction`：

```text
TISA ready
  -> 到达时间、Deps、UnitMap、queue/ROB、地址冲突检查
  -> issue 整条 instruction
  -> backend 执行绑定 payload
  -> completion 唤醒后继 TISA
```

Attention tile 形成：

```text
TISA load-A -> TISA load-B -> TISA matmul -> TISA store
```

Softmax 的 `reduce_max/exp/reduce_sum/normalize` 属于同一 VE payload。payload lane
事件进入 timing 和泳道图，TISA 依赖保持全局可见。

## 5. Runtime 与 device scheduler

```text
Host Runtime
  物理地址、command chunk、descriptor arrival、同步

TISA Device Scheduler
  reception、WQ/IQ、依赖检查、资源检查、OOO issue、completion

Backend Timing/Event
  已 issue instruction 的 task duration、II 和事件
```

论文的 tile-by-tile OOO 决策位于 device hardware；runtime 控制 descriptor 的可见时间。
项目用 `--runtime-policy` 和 `--policy` 分别研究两层，`--runtime-device-matrix`
一次编译后运行四种组合。

## 6. 当前实现与扩展项

当前实现：

- PyTorch -> Torch-XLA -> official StableHLO -> GC/FC/TISA；
- semantic Softmax/Norm、Attention、SwiGLU、RoPE 和 KV-cache region；
- instruction-level static/dynamic scheduler；
- descriptor arrival、queue/ROB/window、resource、completion feedback analytical model；
- analytical、timing table、systolic MXU profile 和 RTL importer；
- `payload_ready:<task_id>` partial-ready 原型和 memory bank scoreboard。

扩展项：

- 论文全部 operation/model block 的 semantic coverage；
- symbolic shape、dynamic index、dynamic layout 和 stride-aware transform；
- 论文 WQ/IQ/Fu 容量、dispatch/wake-up/issue/completion 控制开销；
- 完整 RTL 与真实芯片 timing calibration；
- online Softmax 数值 rescale、最终 normalization 和 workspace 生命周期。

当前结果标签为 `TISA instruction-level analytical scheduling baseline`。profile 加载后，
manifest 记录对应 calibration status，trace schema 保持一致。
