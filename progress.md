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
- 首个两头 Attention block 已覆盖 head reshape/permute、scale、additive mask、Softmax、PV、output projection 和 residual；
- reshape/transpose 已生成 DMA-bound TISA transform，TileGraph 已使用逻辑 tensor region 建边；
- `compile_statistics.json` 已输出 per-operator tile/TISA/payload、MAC、root traffic 和 dependency 统计；
- 当前回归为 55 项；一次编译的四组合实验中 device static=2344、dynamic=2119 analytical cycles。

## 真实验证

```text
Python 3.12
torch 2.9.1
torch-xla 2.9.0
StableHLO 1.12.1...
multi-head attention: 48 tiles, 121 TISA instructions
minimal analytical device: static 2344, dynamic 2119 cycles
```

## 当前限制

- StableHLO semantic importer 仍只覆盖已注册 operation；
- tile planner 还是确定性启发式，没有 cost model；
- reshape/transpose 仍是 full-tensor transform，symbolic/dynamic shape 与 stride-aware layout 尚未完成；
- analytical event backend 不是 RTL cycle-accurate；
- 当前 MXU VCS log 主要提供 descriptor-to-completion 区间；
- 论文中的完整 ResNet50、BERT、GPT-J、LLaMA2、DeepSeek block 尚未形成可复现实验集。

## 下一步

先把真实 Attention 扩展为 pre-norm decoder block，并细化 stride-aware transform 与 pass dump；随后接入更细的 device queue/hazard 模型和外部 MXU/memory timing。
