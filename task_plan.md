# 任务计划

## 目标

建立从真实 PyTorch 算子/模型到 TISA device simulator 的可配置研究栈，用同一份编译产物比较 static 与 dynamic 调度，并输出可解释的周期、stall 和泳道图。

## 当前状态

当前已完成阶段 1-3：

1. `torch.export -> Torch-XLA -> official StableHLO -> Canonical OperatorGraph` 唯一前端；
2. Canonical graph passes、统一 tile planner、TISA-first codegen；
3. RuntimeSubmission、物理地址绑定、TISA instruction 粒度的 static/dynamic simulator；
4. 可配置 MachineConfig，以及 codegen、timing、event backend registry；
5. 分阶段 artifact、泳道图、Perfetto trace 和 RTL completion profile importer。

已经删除的生产路径：手写 `src/npu_ooo/benchmarks`、直接 JSON/StableHLO 输入、项目自有 StableHLO emitter、算子专用 CLI 和 primitive scheduler 入口。

## 阶段 1：前端语义覆盖

目标：输入只接受真实 `torch.nn.Module`，通过 Torch-XLA 获得 StableHLO，不按模型名写分支。

- [x] Matmul、elementwise、reduce、Softmax、RMSNorm、LayerNorm 基础路径；
- [x] attention micrograph 的 `QK^T -> Softmax -> PV`；
- [ ] 完整 multi-head attention：scale、mask、reshape、output projection；
- [ ] decoder block：RoPE、KV-cache、SwiGLU/MLP、residual；
- [ ] ResNet bottleneck：Conv2D、BatchNorm inference、ReLU、pooling；
- [ ] BERT/GPT-J/LLaMA2/DeepSeek 的真实 one-block module。

每个新增能力都必须同时补 semantic capability、graph recovery/fusion、TISA stage、backend lowering 和真实 PyTorch 回归。

## 阶段 2：编译与 TISA 正确性

- [x] PassManager、统一 tile planner、TISA descriptor 和 backend payload ownership；
- [ ] 细化 region-aware tile dependency，减少不必要的 all-to-all 等待；
- [ ] 完善 symbolic/dynamic shape、layout、broadcast 和边界 tile；
- [ ] 输出 pass 前后 graph、tile 数、MAC、traffic 和 dependency diagnostics；
- [ ] 为多种 tile candidate 增加可解释 cost model。

验收：相同 module、shape、tile size、MachineConfig 和 backend 产生稳定 compiled artifact；static/dynamic 只改变 policy。

## 阶段 3：设备调度与后端

- [x] reception availability、queue、ROB/window、资源占用、completion feedback 的 analytical 模型；
- [ ] 补齐论文语义的 partial-ready、typed hazard 和 memory bank/port conflict；
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

优先完成一个真实 multi-head attention 或 decoder block，并同时细化 tile region dependency。前端语义稳定后再增加 scheduler 微结构和外部 timing backend，避免用不完整的图去校准硬件时序。
