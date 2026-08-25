# 当前进度

## 已完成

- 唯一生产入口：`compile-model --torch-module MODULE:CLASS_OR_FACTORY`；
- 完整前端链路：`PyTorch -> torch.export -> Torch-XLA -> official StableHLO -> Canonical IR`；
- StableHLO 官方 parse/verify 与 semantic capability boundary；
- 自动 graph pass、tile planner、TISA-first builder 和 analytical backend payload；
- TISA instruction 粒度 static/dynamic device scheduler；
- runtime 地址分配、command chunk、descriptor availability 和 runtime/device 四组合矩阵；
- 可配置 MachineConfig、codegen/timing/event registry；
- staged output：`00_frontend` 到 `07_trace`、manifest、artifact index、SVG/PNG/Perfetto；
- MXU RTL completion trace 和 VCS console log 的离线 profile importer；
- 删除手写 benchmark、直接 StableHLO/JSON 输入和旧 primitive scheduler 生产入口；
- 54 个回归测试、真实 attention CLI smoke 均通过。

## 真实验证

```text
Python 3.12
torch 2.9.1
torch-xla 2.9.0
StableHLO 1.12.1...
attention micrograph: 19 TISA instructions, 136 analytical cycles
```

## 当前限制

- StableHLO semantic importer 仍只覆盖已注册 operation；
- tile planner 还是确定性启发式，没有 cost model；
- 跨算子依赖偏保守，symbolic/dynamic shape 与复杂 layout 尚未完成；
- analytical event backend 不是 RTL cycle-accurate；
- 当前 MXU VCS log 主要提供 descriptor-to-completion 区间；
- 论文中的完整 ResNet50、BERT、GPT-J、LLaMA2、DeepSeek block 尚未形成可复现实验集。

## 下一步

先扩展真实 PyTorch multi-head attention/decoder block，再补 region-aware dependency 和 pass diagnostics；随后接入更细的 device queue/hazard 模型和外部 MXU/memory timing。
