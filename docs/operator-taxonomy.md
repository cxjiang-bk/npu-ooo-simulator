# 算子分类与覆盖计划

## 1. 两种层次必须分开

### Semantic operators

保留模型和调度需要的上下文，作为 Operator Graph/TISA 的 `OpType`：

```text
matmul, conv2d, attention, softmax, layernorm, rmsnorm,
gelu, silu, residual_add, embedding, moe_dispatch, ...
```

### Lowering primitives

由 semantic operator 拆出的硬件任务：

```text
load, store, copy, transpose,
mxu_gemm, vector_add, vector_mul, vector_exp,
reduce_max, reduce_sum, barrier, cache_read, cache_write
```

TISA-like scheduler 应看到 `OpType=SOFTMAX` 以及它的 `TileMem/UnitMap`，而不是只看到失去上下文的四个 `exp/add/div`。Primitive task 仍然需要存在，供 timing simulator 计时和画 lane。

## 2. 按论文 benchmark 的算子需求

| 类别 | 必要算子 | 主要 benchmark | 优先级 |
|---|---|---|---|
| Dense linear algebra | GEMM、batched GEMM、GEMV、linear | BERT/GPT-J/LLaMA/DeepSeek | P0 |
| CNN | Conv2D、1x1 Conv、可选 depthwise Conv | ResNet50 | P0 |
| Attention | QKV projection、QK^T、scale/mask、softmax、PV | Transformer family | P0 |
| Normalization | LayerNorm、RMSNorm、BatchNorm inference | BERT/decoder/ResNet | P0 |
| Vector activation | ReLU、GELU、SiLU/Swish、clamp | ResNet/BERT/LLaMA | P0 |
| Elementwise/fusion | add、mul、sub、div、bias、scale、residual add | all | P0 |
| Reduction | reduce-max、reduce-sum、mean、variance、RMS | norm/softmax | P0 |
| Tensor transform | reshape、transpose、permute、slice、concat、pad | all Transformer/CNN | P0 |
| Embedding/state | embedding lookup、gather、position embedding、KV-cache read/write | decoder/BERT | P1 |
| Positional encoding | RoPE、ALiBi、causal mask | GPT-J/LLaMA/decoder | P1 |
| Pooling | max-pool、avg-pool、global-avg-pool | ResNet50 | P1 |
| MoE/routing | top-k、token dispatch、expert GEMM、combine | optional MoE | P1 |
| Quantization | cast、dequant、quant、scale/zero-point | future BF16/INT8 cases | P2 |
| Distributed | all-reduce、all-gather、all-to-all | future multi-chip | P2 |

## 3. P0 具体拆分

### ResNet50

```text
Conv2D / 1x1 Conv
BatchNorm inference or folded affine
ReLU
MaxPool / GlobalAvgPool
Residual Add
Linear
Transpose/Layout transform
```

BatchNorm inference 是否单独存在由 frontend import 决定；如果已被 fold 到 Conv 权重和 bias，必须在 provenance 中记录，而不是重复模拟。

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
RoPE or model-specific position transform
causal attention mask
QK^T + Softmax + PV
KV-cache read/write
LayerNorm or RMSNorm
GELU or SiLU/SwiGLU MLP
Residual Add
LM head / Linear
```

`prefill` 和 `decode` 复用 semantic operators，但 tile shape、KV-cache region 和并发度不同。

## 4. 哪些应该作为 composite op

以下算子不建议一开始直接打散为 primitive-only graph：

- `attention`：保留 QK-softmax-PV 的 operator context；
- `softmax`：保留 reduction axis、mask 和 numerical policy；
- `layernorm/rmsnorm`：保留统计量和归一化关系；
- `silu/swiglu`：保留 activation-gate 组合关系；
- `moe_dispatch`：保留 routing、capacity 和 token ownership。

lowering backend 再根据 MachineConfig 选择 primitive task。这样 dynamic scheduler 能按 semantic compatibility 和 UnitMap 做更准确的判断。

## 5. P0 到 P2 的落地顺序

```text
P0-A: matmul, elementwise, reduction, transpose/copy
P0-B: softmax, layernorm/rmsnorm, conv2d, residual add
P1-A: embedding, RoPE, causal mask, KV cache
P1-B: pooling, SwiGLU, optional MoE routing
P2: quantization and distributed communication
```

第一里程碑仍然可以只用 `matmul + copy + elementwise` 跑 2mm；但 Model IR 从现在开始就按上述 taxonomy 设计，避免之后为了支持 ResNet/LLM 重做接口。

## 6. 每个算子 backend 的最低契约

```text
semantic_op_type()
infer_output_shapes()
tile_access_regions()
required_buffers()
lower_to_execution_tasks()
dependency_rules()
latency_request()
```

Scheduler 不应出现 `if op_name == "ProduceQ"` 这类分支。模型专用名字只能存在于 frontend provenance 和 semantic OpType 映射中。
