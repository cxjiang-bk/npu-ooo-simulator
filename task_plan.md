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
- [x] 首个静态 shape pre-norm decoder block：RMSNorm、attention、residual、SwiGLU/MLP；
- [x] decoder block extensions：RoPE、固定窗口 KV-cache 与多步 decode sequence；
- [x] ResNet bottleneck micro：Conv2D、BatchNorm inference、ReLU、MaxPool；
- [x] BERT/GPT-J/LLaMA2/DeepSeek 的真实 one-block module（LLaMA2 decode/cache 为 scaled micro workload）。

每个新增能力都必须同时补 semantic capability、graph recovery/fusion、TISA stage、backend lowering 和真实 PyTorch 回归。

## 阶段 2：编译与 TISA 正确性

- [x] PassManager、统一 tile planner、TISA descriptor 和 backend payload ownership；
- [x] 明确 GCArtifact、TISADialectProgram、TISAProgram 三个编译阶段契约；
- [x] 复合 Softmax/Norm 保持 semantic TISA op，内部 primitive 退回 backend payload；
- [x] 第一版 region-aware tile dependency，无法证明映射时显式保守回退；
- [x] 保存每个 GC pass 的输入/输出图与诊断 dump；
- [x] LayerNorm recovery 对多实例图执行 fixed-point，避免第二个规范化节点残留到 FC；
- [x] RMSNorm recovery 支持 Torch-XLA power/reshape/affine 链，并将 weight 作为 semantic operand 传入 payload；
- [x] 生成 capacity-aware residency 和多 tile ping-pong intent metadata；
- [x] 为 FC stage 和 typed dependency 写入显式 readiness condition；
- [x] Softmax 提供默认 materialized 与显式 analytical online state-chain 两种 payload 语义；
- [ ] 完善 symbolic/dynamic shape、layout、broadcast 和边界 tile；
  - [x] Torch-XLA 常见 dynamic broadcast 与常量起点 `dynamic_slice` 的 shape specialization 子集；
  - [ ] 完整 dynamic index/layout legalization；
- [x] 为 FC/TISA `TileMem` 增加 concrete stride、stride expression 和 layout metadata；
- [x] 输出 per-operator tile、TISA、MAC、root traffic 和 dependency statistics；
- [x] 输出每个 pass 前后的独立 graph dump；
- [x] 为多种 tile candidate 增加可解释 cost model。

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

当前矩阵入口与输出契约：

- [x] `paper-matrix` 对 registry case 执行一次编译并复用同一 `BackendArtifact`、
  `program_id` 和 buffer binding；
- [x] 默认比较 device static/dynamic，显式 `--runtime-device-matrix` 才运行四组合；
- [x] 为每个 `<case-id>/<variant>/` 保存共享 `00_frontend` 到 `04_backend` compiler
  artifact，并将策略的 `05_runtime` 到 `07_trace` 结果放入 `policy_matrix/`；
- [x] 在矩阵根目录写入 `matrix_index.json`，用于识别本次 case 清单并诊断旧输出目录；
- [ ] 补齐 full-model repetition、request-level runtime，以及 source-derived/RTL-observed
  分组统计。

`micro` 是默认快速 proxy；`paper_shape` 只用于接近论文形状的代表性输入，预计需要
更大的内存和更长的编译时间，不能直接替代完整论文 benchmark。

## 当前下一步

### 阶段 1A：公共 StableHLO 前端稳定性

- [x] `stablehlo.convert` capability 和 source/target dtype 语义；
- [x] 未注册 StableHLO operation 显式报告缺失 capability 与已知 operation 集合；
- [ ] scalar/multi-result/layout/broadcast 的通用 metadata；
  - [x] 静态 `broadcast_in_dim` 输出域 tile、源 operand region 与边界 tile 契约；
  - [x] scalar elementwise operand 的零秩/常量 metadata 与 tile region 契约；
  - [x] StableHLO tensor encoding 的来源 metadata 与未知 layout 的 conservative interval 契约；
  - [x] 未知 StableHLO multi-result operation 的显式失败；`batch_norm_training` 保留既有 recovery 例外；
- [x] Torch-XLA/PJRT dtype 兼容性矩阵和严格/fallback policy。

### 阶段 1B：模型到 TISA 的语义覆盖

- [x] Fusion Pattern Registry 基础设施与现有 LayerNorm/RMSNorm/Softmax pattern；
- [x] Attention region 与 SwiGLU semantic pattern；
- [x] BERT/GPT-J one-block 的真实前端回归；
- [x] LLaMA2 的 RoPE、固定窗口 KV-cache 和 prefill/decode micro workload；
- [x] ResNet micro 的 Conv2D、BatchNorm inference、pooling；完整 ResNet50 仍需扩展；
- [ ] DeepSeek 结构确认以及 dense/MoE 路径。

阶段 1A 完成后继续细化 FC 的 TISA dialect metadata、strided `TileMem` interval，
再推进上述模型语义。scheduler 微结构和外部 timing backend 保持在编译语义稳定之后。

### 阶段 1C：跨 invocation runtime state

- [x] 建立 persistent state registry，稳定绑定 `state_id` 与物理 buffer；
- [x] 建立多步 `RuntimeSequence`，显式记录 invocation 间 state-complete 依赖；
- [x] sequence simulator 复用同一 compiled artifact，拼接多次 invocation 的周期与事件；
- [x] 用固定窗口 KV-cache 两步 decode 回归验证地址稳定、状态依赖和 static/dynamic 周期。

阶段 1C 的 contract 仍只覆盖固定窗口、静态 shape、unit stride 和顺序 decode；动态
position 写入、跨 request 生命周期和真实 cache layout 留到后续阶段。

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Scalar constants caused reduce lowering to see two inputs | 1 | Keep reducer-init SSA value in `constant_args`; canonical reduce retains only the first data operand |
