# 算子分类与覆盖计划

## 1. Semantic operator 与 lowering primitive

Semantic operator 保留模型和调度所需的上下文，作为 OperatorGraph/TISA 的 `OpType`：

```text
matmul, conv2d, attention, softmax, layernorm, rmsnorm,
gelu, silu, residual_add, embedding, moe_dispatch, ...
```

Lowering primitive 表达硬件执行任务：

```text
load, store, copy, transpose,
mxu_gemm, vector_add, vector_mul, vector_exp,
reduce_max, reduce_sum, barrier, cache_read, cache_write
```

TISA scheduler 看到 semantic OpType、TileMem 和 UnitMap；backend payload 负责 primitive
timing。一个 semantic tile 跨越 DMA、Tensor、Vector 等 EU 时，compiler 按 EU 边界
生成多条 TISA instruction，并使用 typed dependency 串联。同一 EU 内部的 primitive
保留自身的数据依赖和 reduction barrier，由 payload graph 表达。

每条 semantic tile instruction 包含：

```text
TISAInstruction
  OpType
  Operands: TileShape + TileMem(base, scope) + AccessType
  Attributes: reorder/sync/readiness
  UnitMap: unit class, quantity, affinity
  Deps: source, RAW/WAR/WAW/STATE/ACCUMULATE, condition
```

lowering 顺序：

```text
semantic operator
  -> TileInstance
  -> TISAInstruction
  -> backend primitive tasks
```

## 2. Benchmark 算子需求

| 类别 | 算子 | 主要 benchmark | 优先级 |
|---|---|---|---|
| Dense linear algebra | GEMM、batched GEMM、GEMV、linear | BERT/GPT-J/LLaMA/DeepSeek | P0 |
| CNN | Conv2D、1x1 Conv、depthwise Conv | ResNet50 | P0 |
| Attention | QKV projection、QK^T、scale/mask、softmax、PV | Transformer family | P0 |
| Normalization | LayerNorm、RMSNorm、BatchNorm inference | BERT/decoder/ResNet | P0 |
| Vector activation | ReLU、GELU、SiLU/Swish、clamp | ResNet/BERT/LLaMA | P0 |
| Elementwise/fusion | add、mul、sub、div、bias、residual add | all | P0 |
| Reduction | reduce-max、reduce-sum、mean、variance、RMS | norm/softmax | P0 |
| Tensor transform | reshape、transpose、permute、slice、concat、pad | all | P0 |
| Embedding/state | embedding lookup、gather、position embedding、KV-cache read/write | decoder/BERT | P1 |
| Positional encoding | RoPE、ALiBi、causal mask | GPT-J/LLaMA/decoder | P1 |
| Pooling | max-pool、avg-pool、global-avg-pool | ResNet50 | P1 |
| MoE/routing | top-k、token dispatch、expert GEMM、combine | optional MoE | P1 |
| Quantization | cast、dequant、quant、scale/zero-point | future BF16/INT8 | P2 |
| Distributed | all-reduce、all-gather、all-to-all | future multi-chip | P2 |

## 3. 模型 block 分解

### ResNet50

```text
Conv2D / 1x1 Conv
BatchNorm inference 或 folded affine
ReLU
MaxPool / GlobalAvgPool
Residual Add
Linear
Transpose/Layout transform
```

BatchNorm fold 到 Conv 的选择由 frontend import 决定，provenance 记录实际图形态。

### BERT

```text
Embedding + position/type embedding
Q/K/V linear projections
batched QK^T
attention mask + scale
Softmax
PV batched GEMM
LayerNorm
GELU MLP
Residual Add
reshape/transpose
```

### GPT-J / LLaMA2 / DeepSeek decoder

```text
Embedding
Q/K/V projection
RoPE 或 model-specific position transform
causal attention mask
QK^T + Softmax + PV
KV-cache read/write
LayerNorm 或 RMSNorm
GELU 或 SiLU/SwiGLU MLP
Residual Add
LM head / Linear
```

prefill 与 decode 复用 semantic operators，tile shape、KV-cache region 和并发度随 phase
配置变化。

## 4. Composite semantic op

以下语义保留为 composite operator，并在 backend 选择 primitive：

- attention：保留 QK-softmax-PV context；
- softmax：保留 reduction axis、mask 和 numerical policy；
- layernorm/rmsnorm：保留统计量和归一化关系；
- silu/swiglu：保留 activation-gate 组合关系；
- moe_dispatch：保留 routing、capacity 和 token ownership。

## 5. 覆盖顺序

```text
P0-A: matmul, elementwise, reduction, transpose/copy
P0-B: softmax, layernorm/rmsnorm, conv2d, residual add
P1-A: embedding, RoPE, causal mask, KV cache
P1-B: pooling, SwiGLU, optional MoE routing
P2: quantization and distributed communication
```

新增算子从真实 PyTorch module 经 Torch-XLA 导入，semantic taxonomy 保持模型无关。
模型专用名称归属于 provenance 和 benchmark registry。

## 6. Backend 最低契约

```text
semantic_op_type()
infer_output_shapes()
tile_access_regions()
required_buffers()
lower_to_tisa_instructions()
dependency_rules()
backend_expand_to_tasks()
latency_request()
```

Scheduler 依据 OpType、UnitMap 和 typed dependency 工作；模型专用分支属于 frontend
provenance 层。
