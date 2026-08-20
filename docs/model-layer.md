# Model IR 与 Benchmark Layer

## 1. 结论

应该在 Operator Graph 之上增加 Model/Benchmark Layer，而且它不是可选的装饰层。

原因是论文的评估对象不是几个孤立算子，而是不同模型和运行阶段：

| Model | Family | Case shape/phase |
|---|---|---|
| ResNet50 | CNN residual | FP16, batch=128, image=224x224 |
| BERT-Base | encoder Transformer | FP16, batch=64, seq=128 |
| GPT-J-6B | decoder Transformer | FP16, batch=1, seq=512, prefill |
| LLaMA2-13B | decoder Transformer | FP16, batch=1, seq=512, prefill |
| DeepSeek-R1-16B | decoder/reasoning Transformer | BF16, batch=50, seq=100, prefill |
| DeepSeek-R1-16B | decoder/reasoning Transformer | BF16, batch=50, seq=700, decode |

`Operator IR` 只能表达 `Matmul`、`Softmax` 或 `Conv2D`；它不能表达 GPT-J 有多少个 block、decode 时 KV cache 如何增长、ResNet 的 residual stage 如何重复，也不能表达 benchmark 的 batch/sequence/phase 配置。

## 2. Model 层和 Operator 层的职责

```text
Model IR
  “运行哪个模型、哪个阶段、什么 shape？”

Operator Graph IR
  “这个模型实例由哪些语义算子和 tensor 组成？”

Schedule/Tiling IR
  “每个算子如何切 tile、融合、驻留和排序？”

Tile/TISA IR
  “每个 tile 需要什么资源、访问哪些地址、依赖什么？”
```

Model 层不应该直接描述 MXU、ARU 或 DMA；硬件信息属于 MachineConfig 和 Tile/TISA 层。

## 3. 推荐的 Model IR 结构

### 3.1 ModelSpec

```text
ModelSpec
  id: stable model identifier
  family: cnn_residual | encoder_transformer | decoder_transformer | moe
  variant: model/version string
  dtype_policy
  layout_policy
  shape_symbols
  graph_templates
  top_level_graph
  persistent_state
```

### 3.2 GraphTemplate

```text
GraphTemplate
  id: e.g. transformer_block, resnet_bottleneck
  parameters: hidden, heads, intermediate, head_dim, ...
  nodes: operator template nodes
  edges: tensor/dataflow edges
  repetitions: symbolic or concrete count
  shared_parameters: weight tying metadata
```

模板机制很重要：GPT-J/LLaMA2 的完整模型不应手写数十层重复图；应定义一个 decoder block，再通过 `num_layers` 实例化。ResNet 也应定义 bottleneck/basic block 和 stage repetition。

### 3.3 BenchmarkCase

```text
BenchmarkCase
  model_id
  case_id
  batch
  sequence_length / image_height / image_width
  phase: train | prefill | decode
  dtype: fp16 | bf16 | fp32 | int8 | ...
  quantization
  causal / mask policy
  cache policy
  warmup, repetitions, seed
  architecture_profile
  scheduler_profile
```

Table IX 中的每一行应成为一个独立 `BenchmarkCase`，不能把 `DeepSeek prefill` 和 `DeepSeek decode` 混成同一个 case。

## 4. 运行阶段模型

### CNN inference

主要状态是输入 feature map、weight 和 residual buffer；通常没有 token-by-token state。

### Encoder inference

BERT 的 self-attention 通常是双向 mask，输入 sequence 长度固定在一个 case 内。Model IR 需要保留 mask policy，但不需要 KV cache decode state。

### Decoder prefill

一次处理完整 prompt：

```text
QKV projection -> attention over prompt -> MLP -> residual
```

KV cache 在层内写入，矩阵形状通常是 batch/sequence 维度较大，适合研究跨 tile pipeline。

### Decoder decode

每一步通常只生成一个或少量 token：

```text
new-token projection -> read historical KV cache -> attention -> MLP
```

decode 和 prefill 必须使用不同的 shape environment、cache access pattern 和 benchmark case；不能只把 `seq` 改成另一个整数。

### Optional MoE

如果模型或未来 benchmark 使用 MoE，Model IR 需要表达：

```text
router -> top-k selection -> token dispatch -> expert GEMMs
       -> combine/gather -> residual
```

但在没有模型配置证据时，不应把 DeepSeek-R1-16B 固定判定为 MoE。MoE 作为 model family/capability 保留即可。

## 5. 模型到算子的实例化流程

```text
ModelSpec + BenchmarkCase
        |
        v
instantiate templates and symbolic shapes
        |
        v
Operator Graph IR
        |
        v
fusion groups / schedule candidates
        |
        v
Tile Instance IR
```

实例化阶段需要保存 provenance：

```text
model_id
template_id
layer_index
block_index
operator_id
```

这样泳道图可以回答“哪个 tile 属于第几层的哪个 operator”，而不只显示一串匿名 `MXU_37`。

## 6. Model 层的最小可行范围

第一阶段不需要导入完整 HuggingFace 权重，也不需要执行真实数值推理。先支持三类结构化 model template：

1. `two_matmul_chain`：验证 model -> operator -> tile 的最小闭环；
2. `resnet_bottleneck`：验证 Conv/Norm/Activation/Residual；
3. `decoder_block`：验证 QKV/Attention/Norm/MLP/Residual 和 prefill/decode shape。

在此基础上再添加 `bert_encoder`、`gptj_decoder`、`llama_decoder` 和 optional `moe_decoder`。

## 7. Benchmark case 与实验隔离

模型配置和硬件配置必须分开：

```text
ModelSpec/BenchmarkCase
  = what workload runs

MachineConfig
  = on what accelerator it runs

SchedulePolicy
  = how tiles are issued
```

同一个 `BenchmarkCase` 可以在不同 architecture profile 和 scheduler policy 下重复运行，实验结果才能归因于硬件或调度策略。
