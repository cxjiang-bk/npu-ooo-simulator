# 研究决策记录

## 核心边界

```text
PyTorch nn.Module
  -> torch.export
  -> Torch-XLA StableHLO
  -> official StableHLO verify/import
  -> Canonical OperatorGraph
  -> Schedule / TileGraph
  -> TISAProgram
  -> BackendArtifact
  -> RuntimeSubmission
  -> TISA device scheduler
  -> backend timing/event trace
```

用户前端只接受真实 PyTorch module。项目不维护 StableHLO emitter，不接受手写 StableHLO/JSON 作为生产入口，不为模型名称添加 compiler 或 scheduler 分支。

## 语义与执行层次

- `TileInstance` 表达切分范围；`TISAInstruction` 增加 `OpType`、`TileShape`、`TileMem`、`AccessType`、`UnitMap` 和 typed dependencies；
- 全局 OOO scheduler 只 issue `TISAInstruction`；`ExecutionTask` 是 issue 后在 execution unit 内部执行的 backend payload；
- runtime 负责物理地址、command chunk、descriptor 到达和同步；device scheduler 负责 queue/ROB/依赖/资源/OOO issue；
- static 与 dynamic 必须共享同一 compiled artifact、MachineConfig、timing source 和 runtime submission，除非实验变量就是对应层策略。

## Backend 定位

| Backend | 项目中的职责 |
| --- | --- |
| analytical | 默认可解释的 tile/event baseline |
| timing table | 按 task/primitive 覆盖局部时序 |
| systolic MXU profile | 重放离线 MXU shape/duration/II profile |
| RTL completion importer | 将 JSON/CSV/VCS log 转为 versioned profile |
| SCALE-Sim/RTL/Verilator（后续） | 提供局部硬件 timing，不接管 TISA 语义 |

## 公平性规则

1. 改变调度策略不能重新切 tile、改变地址或换 timing provider；
2. runtime dynamic 的收益和 device dynamic 的收益分开统计；
3. analytical、source-derived、RTL-observed 结果分组，不能混成一个绝对性能数字；
4. 未覆盖的 StableHLO operation 必须显式失败或由用户明确允许的 fallback 记录，不得静默丢失语义。

## 待验证问题

- 完整 attention/decoder/ResNet/BERT block 所需 StableHLO capability；
- region-aware dependency、partial-ready 和 memory bank/port conflict；
- 论文 WQ/IQ/Fu 容量与 dispatch pipeline 的硬件校准；
- 当前 NPU ISA 中 issue/completion、SET/WAIT/FENCE 和 buffer address 的精确语义。

## 2026-08-29：LLaMA2 decode/cache 验证

- `LLaMA2DecodeOneBlock` 采用 `x:[B,1,H]`、`cache:[B,heads,W,D]`、`update:[B,heads,1,D]`
  的固定窗口布局；`cache[...,1:,:]` 与新 token 沿 `-2` 拼接后参与 QK/PV attention。
- 当前 Torch-XLA/StableHLO/GC 链路会将 K/V 两条 cache 更新分别恢复为
  `kv_cache_update(state_id=arg18/arg11)`；输出 cache 与输入 cache alias 到同一物理地址。
- 两次 invocation 必须复用同一 `BackendArtifact` 与 persistent state registry；sequence
  simulator 用 `state_complete` 串联 invocation。该边界表达 runtime state 生命周期，尚不
  表示动态 position 写入、paged cache、GQA/MQA layout 或跨 request ownership。
- decode workload 的周期是 analytical 结果，只用于比较 static/dynamic 调度趋势；不能
  与论文 A100/epoch 绝对时间直接比较。

## 2026-08-29：ResNet convolution/pooling 语义边界

- Torch-XLA 对 inference `Conv2d` 产生 `stablehlo.convolution`，常见 padding 以
  `dense<1> : tensor<2x2xi64>` 打印；官方投影必须只读取 dense payload，再将标量扩展
  为每个空间维度的 low/high pair。
- inference `BatchNorm2d` 产生五输入 `stablehlo.batch_norm_inference`；未使用的
  `num_batches_tracked` 仍可能出现在 Torch Export placeholder 集合中，应在 source
  graph 边界丢弃该死的零秩 placeholder。
- MaxPool/AvgPool 由 `stablehlo.reduce_window` 加 reducer region 表达。当前 Canonical
  `pool` 只接受 rank-4 NCHW、N/C-preserving window、unit dilation；`maximum` 和
  `add` reducer 可区分，平均池化的 scale/divide 保持为后续 elementwise 节点。
- 卷积与池化输入 tile 不是输出矩形的简单同构：kernel/window 会产生空间 halo。TileGraph
  和 FC `TileMem` 必须使用相同的 halo 区域，否则 backend primitive 的 root-memory
  producer/consumer 边会多于 TISA dependency，artifact validation 会拒绝该结果。

## 2026-08-29：Torch-XLA dynamic shape 图形态

- 即使提供具体 example input，带 `torch.export.Dim` 的 Torch-XLA StableHLO 仍保留
  `tensor<?x...>`，并插入 `get_dimension_size -> reshape/concatenate/maximum ->
  dynamic_broadcast_in_dim` shape program。
- `shape_environment={symbol: value}` 不能安全地直接替换上述程序；正确实现需要在官方
  StableHLO 上执行 shape specialization、constant propagation 和 dynamic-to-static
  legalization，再进入现有 Canonical importer。

## 2026-08-30：dynamic broadcast specialization 实测

- 当前安装的 StableHLO 1.12.1 Python bindings 未注册 `stablehlo-refine-shapes` 等现成
  pass；因此实现独立 operation-level pass，而不是调用不存在的 pass 名称或做 `?` 文本
  替换。
- Torch-XLA dynamic add 的 shape program 可由常量与 `get_dimension_size` 求值；转换后
  static module 能通过 official parse/verify，并可继续生成 Canonical IR/TISA。
- 该 pass 只承诺静态 specialization 子集；未支持的 dynamic operation 保持显式失败，
  不改变“unsupported 不静默降级”的总原则。

## 2026-08-30：静态 broadcast tile 语义缺口

- Canonical importer 已保存 `stablehlo.broadcast_in_dim` 的
  `broadcast_dimensions`，但 planner 仍把所有 `reshape` 当成 full-tensor transform；因此
  广播只有一个 TileGraph 节点，跨算子依赖无法暴露输出域的 tile 并行性。
- 广播不是普通 reshape。输出 tile 到源 tensor 的映射必须按 `broadcast_dimensions`
  投影：未映射的输出轴不读取源数据，源 extent 为 1 的轴始终读取 `[0:1]`，其余轴读取
  对应输出 tile 区域。
- GC TileGraph、FC `TileMem` 和 analytical backend 必须使用同一映射，否则 TISA RAW
  dependency 与 backend root-memory handoff 会不一致，`BackendArtifact.validate()` 会失败。
- 本阶段只处理静态 `broadcast_in_dim`。动态广播仍要求 official StableHLO shape
  specialization；普通 reshape/transpose 继续保持 full-tensor transform。

- 当前 `TensorSpec.validate()` 要求至少一个维度；PyTorch 的 `x + scalar` 可能在
  Torch-XLA 中表现为零秩常量或 `broadcast_in_dim`。scalar 支持必须先定义零秩 tensor 的
  metadata、字节大小和 region 表示，再接入 elementwise tile 映射，不能简单把 shape 改成
  `(1,)`，否则会改变 StableHLO rank/广播语义。

## 2026-08-27：阶段 1A 首个增量

## 2026-08-30：scalar reduce compatibility

- 引入零秩常量后，StableHLO `reduce` 的 init SSA value 也会进入 tensor map；因此 reduce
  projection 必须显式裁剪为单个数据输入，同时把 init 保留在 `constant_args`，否则既有
  norm/attention 图会在 analytical lowering 阶段失败。

## 2026-08-30：dtype 与 multi-result 契约

- 当前 lowering 的 dtype-byte 表覆盖常用 f16/bf16/f32/f64 和整数类型，但未知 dtype
  会默认走 2-byte，存在静默错误风险；因此 machine `supported_dtypes` 与
  `dtype_policy={strict,fallback}` 在 compiler 入口统一校验，fallback 只能用于显式标记的
  analytical 结果。
- Official StableHLO projection 不能把任意 multi-result op 截断为第一个结果。唯一保留
  的例外是现有 LayerNorm recovery 依赖的 `batch_norm_training`；其 secondary result
  仍在被消费或返回时失败。

- `stablehlo.convert` 已注册为单操作 elementwise capability；Canonical Operator 保留
  `source_dtype`、`target_dtype`、`conversion_kind=dtype_cast`，并验证 convert 不改变 shape。
- 未注册 operation 现在会报告原始/规范化名称、缺失的 `StableHLOOpCapability` 注册项和
  当前已知 operation 集合；不会静默跳过或降级。
- 该增量使 FP16 Transformer benchmark 的 `convert` importer 阻塞解除；当前 BERT、GPT-J、
  LLaMA2 micro workload 可生成 TISA。ResNet 仍在 `stablehlo.convolution` capability 边界
  显式失败；DeepSeek BF16 在 CPU Torch-XLA 上仍受设备 dtype 限制。
- StableHLO 使用 `f16/f32` 类型拼写，内部 TISA/tiling/runtime 字节表已补齐这些别名；否则
  `f32` 会错误走默认 2-byte 路径，导致 TileMem、residency 和 traffic 统计不一致。

## 2026-08-28：Fusion Pattern Registry 设计边界

- `StableHLOOpCapabilityRegistry` 只负责单条 StableHLO operation 的解析与语义投影；
  多节点语义恢复由独立的 `SemanticFusionPatternRegistry` 管理。
- 首个 registry 增量只注册已经实现的 LayerNorm recovery、LayerNorm、RMSNorm 和
  Softmax pattern，不添加尚未实现的 Attention/SwiGLU 占位能力。
- pattern priority 使用 GC pipeline 的全局顺序。默认 pipeline 将结构 pass 与 registry
  pattern 按 priority 合并，从而保持现有八个 pass 的顺序、名称和 fixed-point 行为不变。
- Attention 后续应以可观察 region 表达 `QK^T matmul -> Softmax -> PV matmul`，而不是
  融合为一个对 scheduler 不透明的单节点算子。

## 2026-08-28：阶段 1B Attention/SwiGLU 验收

- Attention region 保留 7 个可见成员（QK、3 个 score/mask transform、Softmax、
  probability reshape、PV），只通过 `semantic_regions` 与 tile/TISA attributes 传递 region。
- PreNorm decoder 和 LLaMA2 micro workload 的 SwiGLU 已恢复；LLaMA2 中真实存在的
  `f32 -> f16 -> f32` round-trip 由两个 `dtype_convert` backend primitive 保留。
- 全量测试 79 项通过；BERT/GPT-J one-block 回归已覆盖 region、TISA 成员集合和 payload
  ownership。下一阶段不再修改 scheduler，而是扩展 RoPE/KV-cache 的 state/layout 语义。

## 2026-08-28：Attention/SwiGLU 实际 StableHLO 图形态

- 真实 multi-head Attention 在 GC 中不是相邻的三节点，而是
  `QK matmul -> reshape -> scale -> mask add -> softmax -> reshape -> PV matmul`；
  region recovery 必须允许 shape-preserving elementwise/transform bridge，同时保留所有成员。
- Torch-XLA 的 SwiGLU 尾部稳定表现为
  `logistic(gate) -> gate * logistic(gate) -> silu(gate) * up`。gate/up projection
  Matmul 和 bias add 位于该 pattern 外部，适合作为 semantic SwiGLU 的两个输入。
- Attention region metadata 不改变 scheduler-visible TISA；SwiGLU semantic op 对应 vector
  TISA，内部 logistic、SiLU multiply、gate multiply 由同一 TISA 的 backend-local payload 承担。
