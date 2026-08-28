# 当前进度

## 已完成

- 唯一生产入口：`compile-model --torch-module MODULE:CLASS_OR_FACTORY`；
- 完整前端链路：`PyTorch -> torch.export -> Torch-XLA -> official StableHLO -> Canonical IR`；
- StableHLO 官方 parse/verify 与 semantic capability boundary；
- 自动 graph pass、tile planner、TISA dialect semantic builder 和 analytical backend payload；
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
- 当前回归为 62 项；一次编译的四组合实验中 device static=2344、dynamic=2119 analytical cycles。
- TISA readiness 已增加 completion-boundary 解释器和显式 `payload_ready:<task_id>` partial-ready 原型；后者可在 source payload 完成前唤醒依赖，并输出 `TISA_PARTIAL_READY` trace 事件。
- 已接入可选 memory bank/port scoreboard：依据 `MachineConfig.memory_levels` 的 bank 和读写端口容量记录结构冲突；默认关闭，不改变既有 baseline。
- GC 的 LayerNorm recovery 已改为 fixed-point pass，可完整处理同一 module 中多个 Torch-XLA `batch_norm_training` 规范化节点。
- RMSNorm recovery 已支持 Torch-XLA 的 `power -> reduce -> rsqrt -> affine` 形式及中间 reshape，并在 backend payload 中建模 affine weight read。
- 首个真实 PyTorch pre-norm decoder block 已编译并完成 static/dynamic device 仿真，覆盖 RMSNorm、multi-head attention、masked Softmax、residual 和 SwiGLU/MLP；RoPE 与 KV-cache 尚未加入。

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
- GC 当前只生成 completion-boundary readiness；真实 partial-tile producer 语义仍需由 backend/calibration 端接入；
- memory bank scoreboard 目前是 analytical structural-conflict model，不是 cycle-accurate SRAM/DRAM backend；logical scope 且无 runtime physical binding 时不会强行猜测 bank；
- 论文中的完整 ResNet50、BERT、GPT-J、LLaMA2、DeepSeek block 尚未形成可复现实验集。

## 下一步

保持模型到 TISA 为当前主线：先补 Attention region 与 SwiGLU semantic pattern，并用
BERT/GPT-J one-block 回归验证；随后实现 LLaMA2 RoPE/KV-cache，以及 ResNet
Conv2D/BatchNorm/pooling capability。scheduler/backend 校准继续后置。

## 2026-08-27：阶段 1A

- 完成 `stablehlo.convert` capability/import 和 dtype conversion metadata；
- 完成缺失 StableHLO capability 的显式诊断；
- 新增 2 个 capability boundary regression tests；
- 全量回归：69 tests passed；BERT/GPT-J/LLaMA2 micro workload 已能从 PyTorch 生成 TISA；
- 补齐 StableHLO `f16/f32` 到 TISA/runtime 的 dtype-byte 规范化别名；
- 下一项：建立 Fusion Pattern Registry，随后补 Transformer/ResNet 模型语义。

## 2026-08-28：阶段 1B Fusion Pattern Registry

- 新增独立的 `SemanticFusionPatternRegistry`，与单操作
  `StableHLOOpCapabilityRegistry` 保持职责分离；
- 注册现有 LayerNorm recovery、LayerNorm、RMSNorm、Softmax 多节点 pattern；
- 默认 GC pipeline 由结构 pass 与 semantic pattern priority 确定性合并，八个 pass 的
  历史顺序、名称、fixed-point 与 dump 行为保持不变；
- 新增 5 项 registry 回归；全量回归为 78 tests passed；
- BERT、GPT-J、LLaMA2、DeepSeek prefill/decode micro workload 均重新生成有效 TISA；
  ResNet 仍按预期在 `stablehlo.convolution` capability boundary 显式失败；
- 下一项：实现 Attention region 与 SwiGLU pattern，并以 BERT/GPT-J one-block 回归验证。

## 2026-08-28：阶段 1B Attention/SwiGLU

- 新增 `recover_attention_region`：识别真实 Torch-XLA 图中的
  `QK matmul -> score transform -> Softmax -> probability transform -> PV matmul`，
  生成非 opaque region metadata，但保留每个成员为 scheduler-visible TISA；
- 新增 `fuse_swiglu`：将 `logistic -> silu multiply -> gate multiply` 恢复为一个
  `swiglu` semantic operator，projection Matmul 保持 region 外部；
- 新增 SwiGLU analytical lowering，compute payload 在同一 vector EU 内执行
  `logistic/silu_multiply/gate_multiply`；对 LLaMA2 Torch-XLA 的 `f32 -> f16 -> f32`
  round-trip 保留显式 `dtype_convert` payload，不丢失 dtype 语义；
- FC、lowering registry、backend capability registry 和 TileGraph metadata 已同步；
- MHA、pre-norm decoder、BERT/GPT-J/LLaMA2 micro 回归及 static/dynamic 仿真通过；
- BERT/GPT-J one-block 已有独立 Attention region、SwiGLU payload 和 artifact validity
  回归；全量测试为 79 tests passed；
- 下一项是开始 LLaMA2 RoPE/KV-cache 需求分析。
