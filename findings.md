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

## 2026-08-27：阶段 1A 首个增量

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
