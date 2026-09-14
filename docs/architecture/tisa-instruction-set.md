# TISA 指令集

## 1. 文档定位

本文定义当前项目的 scheduler-visible TISA 指令集。TISA instruction 是一个 tile
stage 的调度描述符，携带 tile operand、执行资源和 typed dependency。它位于 semantic
operator 与 ExecutionTask payload 之间：

```text
semantic operator / semantic region
        -> tile stage
        -> TISAInstruction
        -> ExecutionTask payload
```

`TISAInstruction` 是 device scheduler 的调度单位。一个 instruction 的 payload 可以在同
一个 execution unit 内包含多个内部 primitive；payload 的内部依赖、时延和 lane trace 由
`ExecutionBackend` 管理。

当前编译路径包含两个 TISA 形态：

1. FC 生成 virtual TISA。它保留 semantic tile stage 和抽象 `UnitMap`，其
   `scheduler_visible` 属性为 `false`。
2. Target Lowering 根据 `MachineConfig` 展开 memory route、transfer hop 和 execution
   engine，生成最终 target TISA。最终程序设置 `scheduler_visible=true`，由 Runtime 和
   device scheduler 消费。

因此本文的“指令”包含两层相关内容：stage-level TISA primitive 是稳定的语义接口，target
TISA 可以把一个 transfer stage 展开成多个 route-hop instruction。当前设计定义了
descriptor、资源、依赖和 payload 绑定契约；binary opcode、寄存器编码和汇编语法属于后续
硬件 backend contract。

## 2. 指令格式

当前 `TISAInstruction` 的逻辑格式如下：

```text
TISAInstruction {
    tisa_id:       unique instruction id
    tile_id:       source tile id
    operator_id:   semantic operator id
    op_type:       stage primitive; target TISA 的调度操作名
    operands[]:    TISAOperand
    unit_map:      UnitMap
    dependencies[]: TISADependency
    attributes:    stage, semantic, readiness and target metadata
    payload_ref:   backend payload handle
}
```

每个 `TISAOperand` 由以下字段组成：

```text
TISAOperand {
    name
    tile_shape
    tile_mem
    access_type: read | write | read_write
}
```

`TileMem` 同时记录逻辑和目标绑定信息：

| 字段组 | 当前含义 |
| --- | --- |
| `base`、`tensor` | 逻辑 tensor 或 buffer 名称 |
| `scope`、`visibility` | program/operator/tile 所属范围和 Shared/Local/Private 可见性 |
| `role`、`owner`、`domain` | lhs、rhs、output、state、scratch 等角色及资源归属 |
| `logical_starts`、`logical_shape` | tile 在逻辑 tensor 中的区域 |
| `offset_bytes`、`size_bytes`、`strides_bytes`、`address_expr` | 地址范围、布局和 stride |
| `memory_space` | Target Lowering 选择的 GM、UB、LMB、RMB、PSB 等目标 memory |
| `symbolic_buffer_id`、`buffer_id`、`allocation_id` | 逻辑 buffer、target buffer 和物理分配身份 |
| `valid_bytes`、`dtype`、`layout` | 有效数据量、数据类型和布局 |

## 3. 执行资源

抽象 TISA 使用 `UnitMap` 描述 instruction 请求的资源类别；target TISA 再把它绑定到
具体 engine。

| `UnitMap.unit` | 语义 | 当前典型 primitive |
| --- | --- | --- |
| `dma` | 数据搬运、布局转换和索引读取 | `load`、`store`、`copy`、`transpose`、`gather` |
| `tensor` | 矩阵或张量计算 | `matmul`、`conv2d` |
| `vector` | 向量计算、归约和复合向量算子 | `elementwise`、`reduce`、`softmax`、各类 norm |

`quantity` 表示资源数量，`affinity` 表示 data、matrix 或 vector affinity。LPU-like
target 会把抽象 `dma` 映射到 GDMA/LDMA，把 `tensor` 映射到 MXU，把 `vector` 映射到
ARU/Vector engine；具体名称由 machine profile 的 placement 和 route 决定。

## 4. 当前 TISA stage 指令

下表列出 `_stages_for_tile()` 当前生成的 stage primitive；Target Lowering 将这些 stage
绑定到 route 和 engine 后形成 scheduler-visible target TISA。括号中的 semantic family
是该 stage 服务的上层语义。

| `op_type` | UnitMap | semantic family | 指令职责 |
| --- | --- | --- | --- |
| `load` | `dma` | 矩阵、卷积、向量、归约、norm、pool、复合算子 | 将输入 tile 从 Shared/root region 搬到目标 local operand |
| `store` | `dma` | 产生输出的计算 stage | 将 output tile 从 local operand 写回 Shared/root region |
| `copy` | `dma` | `reshape`、`slice`、`concatenate` | 按逻辑区域或线性 segment 搬运数据 |
| `transpose` | `dma` | `transpose` | 执行布局或维度置换 |
| `gather` | `dma` | `embedding` | 使用索引从 embedding table 读取 tile |
| `matmul` | `tensor` | `matmul`、`batched_matmul`、`gemv` | 执行 tile matrix multiply；`gemv` 在当前 TISA 中使用矩阵 stage |
| `conv2d` | `tensor` | `conv2d` | 执行卷积 tile |
| `elementwise` | `vector` | `elementwise`、`residual_add` | 执行 pointwise、broadcast、bias 或 residual add |
| `reduce` | `vector` | `reduce` | 沿 reduction axis 累积 tile partial result |
| `batch_norm` | `vector` | `batch_norm` | 执行 BatchNorm inference tile |
| `pool` | `vector` | `pool` | 执行 pooling window 或 global pooling tile |
| `softmax` | `vector` | `softmax` | 执行 materialized 或 online softmax tile |
| `rmsnorm` | `vector` | `rmsnorm` | 执行 RMSNorm tile |
| `layernorm` | `vector` | `layernorm` | 执行 LayerNorm tile |
| `swiglu` | `vector` | `swiglu` | 执行 gate activation、乘法和可选 dtype conversion |
| `kv_cache_update` | `vector` | `kv_cache_update` | 按 dynamic index 更新 persistent KV-cache window |

`load`、`compute`、`store` 是 stage key；`op_type` 保存该 stage 的具体 primitive。矩阵
乘法和卷积的 reduction tile 在最后一个 reduction tile 生成 `store`，前序 tile 通过
`ACCUMULATE`/`STATE` dependency 保持 partial result。Transform 和 embedding 直接生成
`copy`、`transpose` 或 `gather` stage，因此不需要额外的 load/store stage。

### 4.1 Target transfer instruction

Target Lowering 会根据 route 把抽象 transfer 展开为一个或多个 transfer-hop TISA。每个
hop 带有 source/destination memory、engine、route hop 编号和 transfer pair metadata。
当前 transfer primitive 包括：

| primitive | 产生位置 | 语义 |
| --- | --- | --- |
| `load` | 输入 route hop | 从上一级 memory 读取输入 tile |
| `load_transpose` | 输入 route 的布局转换 | 读取并完成目标布局转换 |
| `store` | 输出 route hop | 将 output tile 写入下一级 memory |
| `copy` | route 或 transform 的直接复制 | 保持数据值并改变 buffer/location |
| `transpose` | route 或显式 transpose transform | 在搬运过程中完成 layout transpose |

Matmul 的典型 lpu-like route 为：

```text
GM --GDMA--> UB --LDMA--> LMB/RMB --MXU--> PSB --ARU--> UB --GDMA--> GM
```

每一个箭头对应一个 target TISA transfer 或 compute instruction；scheduler 使用这些
target instruction 的 `UnitMap`、operand region 和 dependency 进行 issue。

## 5. Backend payload primitive

Payload primitive 是一次 TISA issue 后在一个 execution unit 内执行的步骤。它们保留
payload 内部数据流，TISA scheduler 以 stage instruction 的完成边界观察它们。

### 5.1 通用 payload

| payload primitive | 所属 TISA stage | 说明 |
| --- | --- | --- |
| `load`、`load_transpose`、`copy`、`transpose` | `load` 或 transfer-hop | 输入搬运和可选布局转换 |
| `store` | `store` | 输出搬运 |
| `matmul` | `matmul` | MXU matrix multiply |
| `conv2d` | `conv2d` | tensor convolution |
| `elementwise` | `elementwise` | 向量 pointwise 计算 |
| `reduce` | `reduce` | 向量归约 |
| `batch_norm` | `batch_norm` | BatchNorm 计算 |
| `pool` | `pool` | pooling 计算 |
| `gather` | `gather` | embedding table gather |

### 5.2 复合 vector payload

| semantic family | payload primitive sequence |
| --- | --- |
| materialized `softmax` | `reduce_max -> exp -> reduce_sum -> normalize` |
| online `softmax` | `online_update` |
| `rmsnorm` | `square -> reduce_sum_square -> rmsnorm` |
| `layernorm` | `reduce_sum -> layernorm_mean -> center -> reduce_sum_square -> layernorm` |
| `swiglu` | `logistic -> silu_multiply -> dtype_convert* -> gate_multiply` |
| `kv_cache_update` | `kv_cache_update` |

`dtype_convert*` 表示由 operator attributes 中的 `conversion_steps` 决定的零个或多个
转换步骤。上述步骤共享所属 TISA stage 的 resource、payload handle 和 completion
边界；单个 payload primitive 的 timing 由 `ExecutionTask` 和 backend timing provider
提供。

## 6. Semantic operator 到 TISA stage

当前 semantic operator 的 stage 序列如下。`store(last reduction tile)` 表示只有完成
该输出 reduction 的最后一个 tile 才执行输出写回。

| SemanticOpType | TISA stage sequence | 资源序列 | 说明 |
| --- | --- | --- | --- |
| `matmul`、`batched_matmul`、`gemv` | `load -> matmul -> store(last reduction tile)` | DMA -> Tensor -> DMA | 矩阵乘法和 partial accumulation |
| `conv2d` | `load -> conv2d -> store(last reduction tile)` | DMA -> Tensor -> DMA | 卷积 tile |
| `elementwise`、`residual_add` | `load -> elementwise -> store` | DMA -> Vector -> DMA | pointwise、broadcast 和残差相加 |
| `reduce` | `load -> reduce -> store(last reduction tile)` | DMA -> Vector -> DMA | reduction 输出 |
| `softmax` | `load -> softmax -> store` | DMA -> Vector -> DMA | materialized/online 由 `softmax_algorithm` 选择 |
| `rmsnorm` | `load -> rmsnorm -> store` | DMA -> Vector -> DMA | RMS 统计量和归一化由 payload 展开 |
| `layernorm` | `load -> layernorm -> store` | DMA -> Vector -> DMA | mean、center、variance 和 normalize 在 payload 中完成 |
| `swiglu` | `load -> swiglu -> store` | DMA -> Vector -> DMA | gate activation、SiLU 和可选 dtype conversion |
| `kv_cache_update` | `load -> kv_cache_update -> store` | DMA -> Vector -> DMA | state buffer 和 dynamic window 通过 attributes 描述 |
| `batch_norm` | `load -> batch_norm -> store` | DMA -> Vector -> DMA | BatchNorm inference |
| `pool` | `load -> pool -> store` | DMA -> Vector -> DMA | pooling window |
| `reshape` | `copy` | DMA | 通过 output-domain tile 和 logical segment 完成 reshape |
| `transpose` | `transpose` | DMA | 直接表达 layout transpose |
| `slice`、`concatenate` | `copy` | DMA | 每个 output tile 可包含多个 source segment |
| `embedding` | `gather` | DMA | index tensor 通过 runtime operand binding 提供 |

Attention、FlashAttention、RoPE 和 MoE 属于 semantic region 或 fusion family。它们沿用
上述基础 stage 表达可调度的成员 tile：

```text
attention       -> projection/matmul + softmax + matmul + elementwise
flash_attention -> QK matmul + online softmax + PV matmul
RoPE            -> load + elementwise/transform + store
moe_dispatch    -> gather/copy + expert matmul + swiglu/elementwise + combine
```

region metadata、state contract 和 provenance 保存在 `TISAInstruction.attributes`；每个
成员 tile 仍然拥有独立的 operand、UnitMap 和 typed dependency。

## 7. Operand 与 memory contract

TISA operand 描述 scheduler 进行 readiness、alias 和 resource 检查所需的完整区域：

1. `tile_shape` 给出本次 instruction 的逻辑 tile 形状。
2. `TileMem.logical_starts/logical_shape` 给出 tensor 内的逻辑区域。
3. `memory_space`、`buffer_id`、`allocation_id` 和 byte range 给出 target placement 后的
   物理身份。
4. `access_type` 区分 read、write 和 read-write；同一物理 allocation 的复用由 typed
   dependency 保护。
5. `address_expr`、dynamic index 和 stride metadata 使 runtime binding 可以解析动态
   slice 或 KV-cache window 的实际范围。

`valid_bytes` 表示有效数据量，`size_bytes` 表示地址包围跨度。Target Lowering、MemoryPlan
和 payload 使用相同的 `buffer_id/allocation_id`，从而让 scheduler 的 alias 检查与 backend
实际访问保持一致。

## 8. Dependency contract

`TISADependency` 的格式为：

```text
TISADependency {
    source
    kind: RAW | WAR | WAW | STATE | ACCUMULATE | BUFFER_REUSE | CONTROL
    condition
    provenance
}
```

当前常用 condition 包括：

| condition | 触发含义 |
| --- | --- |
| `input_region_ready` | producer 已使输入区域可读 |
| `output_region_ready` | producer 已使输出区域可写或可消费 |
| `operand_regions_ready` | 指令的所有计算 operand 已 ready |
| `semantic_tile_ready` | 复合 semantic tile 的输入条件已满足 |
| `full_region_ready` | 整个 producer region 完成 |
| `target_buffer_ready` | target route 已完成对目标 buffer 的写入 |
| `state_complete` | state/accumulation 链达到完整完成点 |
| `allocation_released` | 物理 allocation 可以安全复用 |
| `payload_ready:<task_id>` | backend payload 的指定 partial-ready 点已经到达 |

Dynamic scheduler 和 Static Streams 都消费同一套 TISA dependency。普通 dependency 使用
producer 的完整 completion；`payload_ready:<task_id>` 使用 backend 的 partial-ready
feedback；allocation reuse 使用 physical range 和 `allocation_released` 条件。`provenance`
保留 GC tile dependency、target route expansion 和 runtime alias 的来源，便于验证和 trace
审计。

## 9. TISA、payload 与 scheduler 的边界

```text
TISAInstruction
  -> scheduler admission / dependency check / issue / completion

ExecutionTask payload
  -> one execution unit 内部步骤、timing、II、partial-ready 和 lane trace
```

Device scheduler 不遍历 `ExecutionGraph` 来重新推断指令边界；它读取最终 target TISA 的
descriptor、operand、UnitMap 和 dependency，并将 `payload_ref` 交给 ExecutionBackend。
Backend 负责验证 payload ownership、`can_accept`、EU busy/II 和 physical completion。

Static Streams 和 Dynamic ready queue 共享同一个最终 TISA program、MemoryPlan、dependency
语义和 ExecutionBackend。两种 device policy 只改变 issue 选择方式，不改变本指令集的
operation、operand 或 completion contract。

## 10. 代码索引

| 内容 | 代码位置 |
| --- | --- |
| `TISAInstruction`、`TISAOperand`、`TileMem`、`UnitMap`、`TISADependency` | `src/npu_ooo/ir/tisa.py` |
| `AccessType`、`ExecutionTask`、`BufferRegion` | `src/npu_ooo/ir/execution.py` |
| stage 和 abstract primitive 生成 | `src/npu_ooo/compiler/tisa_dialect.py` |
| semantic operator lowering registry | `src/npu_ooo/lowering/registry.py` |
| target transfer-hop 展开 | `src/npu_ooo/backend/target_lowering.py` |
| backend payload lowerer | `src/npu_ooo/lowering/*.py` |
| device scheduler 消费 TISA | `src/npu_ooo/scheduler/`、`src/npu_ooo/simulator/` |
