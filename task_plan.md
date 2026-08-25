# 任务计划

## 目标

建立从真实 PyTorch 算子/模型到 TISA device simulator 的可配置研究栈，用同一份编译产物比较 static 与 dynamic 调度，并输出可解释的周期、stall 和泳道图。

## 当前状态

当前已完成阶段 1-3，并完成论文 GC/FC 阶段对齐的第一版：

1. `torch.export -> Torch-XLA -> official StableHLO -> Canonical OperatorGraph` 唯一前端；
2. Graph Compiler（GC）：Canonical graph passes、统一 tile planner、region/state tile dependency；
3. Fusion Compiler（FC）与 TISA Generator：TISA dialect proxy、semantic descriptor 和 backend payload ownership；
4. RuntimeSubmission、物理地址绑定、TISA instruction 粒度的 static/dynamic simulator；
5. 可配置 MachineConfig，以及 codegen、timing、event backend registry；
6. 分阶段 artifact、泳道图、Perfetto trace 和 RTL completion profile importer。

已经删除的生产路径：手写 `src/npu_ooo/benchmarks`、直接 JSON/StableHLO 输入、项目自有 StableHLO emitter、算子专用 CLI 和 primitive scheduler 入口。

## 阶段 1：前端语义覆盖

目标：输入只接受真实 `torch.nn.Module`，通过 Torch-XLA 获得 StableHLO，不按模型名写分支。

- [x] Matmul、elementwise、reduce、Softmax、RMSNorm、LayerNorm 基础路径；
- [x] attention micrograph 的 `QK^T -> Softmax -> PV`；
- [x] 首个静态 shape multi-head attention：scale、additive mask、head reshape、output projection；
- [ ] decoder block：RoPE、KV-cache、SwiGLU/MLP、residual；
- [ ] ResNet bottleneck：Conv2D、BatchNorm inference、ReLU、pooling；
- [ ] BERT/GPT-J/LLaMA2/DeepSeek 的真实 one-block module。

每个新增能力都必须同时补 semantic capability、graph recovery/fusion、TISA stage、backend lowering 和真实 PyTorch 回归。

## 阶段 2：编译与 TISA 正确性

- [x] PassManager、统一 tile planner、TISA descriptor 和 backend payload ownership；
- [x] 明确 GCArtifact、TISADialectProgram、TISAProgram 三个编译阶段契约；
- [x] 复合 Softmax/Norm 保持 semantic TISA op，内部 primitive 退回 backend payload；
- [x] 第一版 region-aware tile dependency，无法证明映射时显式保守回退；
- [x] 保存每个 GC pass 的输入/输出图与诊断 dump；
- [x] 生成 capacity-aware residency 和多 tile ping-pong intent metadata；
- [x] 为 FC stage 和 typed dependency 写入显式 readiness condition；
- [x] Softmax 提供默认 materialized 与显式 analytical online state-chain 两种 payload 语义；
- [ ] 完善 symbolic/dynamic shape、layout、broadcast 和边界 tile；
- [x] 输出 per-operator tile、TISA、MAC、root traffic 和 dependency statistics；
- [x] 输出每个 pass 前后的独立 graph dump；
- [ ] 为多种 tile candidate 增加可解释 cost model。

验收：相同 module、shape、tile size、MachineConfig 和 backend 产生稳定 compiled artifact；static/dynamic 只改变 policy。

## 阶段 3：设备调度与后端

- [x] reception availability、queue、ROB/window、资源占用、completion feedback 的 analytical 模型；
- [x] 补齐论文语义的 partial-ready 原型、typed hazard、address conflict 和可选 memory bank/port conflict；
- [ ] 校准 dispatch/wake-up/issue/completion 控制开销；
- [ ] 接入 SCALE-Sim 类 MXU timing、memory timing 和 RTL/Verilator 局部时序；
- [ ] 每个 backend 声明 capability、timing interval 和 calibration status。

## 阶段 4：论文实验矩阵

固定以下维度进行公平比较：

```text
PyTorch module / input shape / phase
  x tile candidate
  x runtime policy
  x device scheduler policy
  x MachineConfig
  x timing/event backend
```

prefill 与 decode 必须是不同 case；analytical、source-derived 和 RTL-observed 结果必须分组统计。完成 one-block 后再扩展 full-model repetition 和 request-level runtime。

## 当前下一步

下一步细化 FC 的 TISA dialect metadata、strided `TileMem` expression，并补齐 online
Softmax 的 rescale、最终 normalization 与 workspace 生命周期语义。memory bank
scoreboard 已作为默认关闭的 analytical structural-conflict 模型接入。
之后继续加入 stride-aware transform、RoPE、KV-cache 和 SwiGLU，形成第一个真实 decoder
block。scheduler 微结构和外部 timing backend 放在语义契约稳定之后。
