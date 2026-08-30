# 当前进度

## 当前快照

项目处于“模型到 TISA 语义稳定、后端校准扩展”的阶段。生产链路和实验目录已经固定：

```text
PyTorch nn.Module
  -> torch.export
  -> Torch-XLA StableHLO
  -> official StableHLO verify/import
  -> GC / TileGraph
  -> FC / TISA dialect
  -> TISAProgram / BackendArtifact
  -> RuntimeSubmission
  -> static 或 dynamic device scheduler
  -> analytical / calibrated backend
  -> cycles、stall、swimlane、Perfetto
```

当前回归：119 tests passed，`git diff --check` clean。

## 已交付能力

### 前端与编译

- 统一入口 `compile-model --torch-module MODULE:CLASS`；
- torch.export 源图 provenance、Torch-XLA StableHLO 导出和官方 MLIR parse/verify；
- StableHLO operation capability registry、semantic fusion/recovery registry；
- GC pass pipeline、固定点规范化、tile planner、candidate cost model；
- region-aware dependency、卷积/池化 halo、静态 broadcast、scalar region；
- TileMem 的 stride、layout、dtype metadata；
- 常量 dynamic broadcast、dynamic_slice、dynamic_reshape specialization；
- GC/FC/TISA 三阶段 artifact 与逐 pass dump；
- materialized/online Softmax payload 属性。

### 模型 benchmark

- 两头 multi-head Attention；
- pre-norm decoder block：RMSNorm、Attention、residual、SwiGLU/MLP；
- BERT、GPT-J、LLaMA2、DeepSeek dense one-block；
- LLaMA2 RoPE、固定窗口 KV-cache、prefill/decode micro；
- ResNet bottleneck micro：Conv2D、BatchNorm inference、ReLU、MaxPool；
- registry 六个 case 与 paper-matrix 批处理。

### Runtime、scheduler 与 backend

- RuntimeSubmission 的地址绑定、command chunk、descriptor arrival；
- RuntimeStateRegistry、RuntimeSequence 和 state-complete invocation 链；
- static_pipeline 与 dynamic_ready_queue 共用同一 BackendArtifact；
- queue/ROB/window、UnitMap 资源、typed dependency、address scoreboard；
- 可选 memory bank/port structural-conflict model；
- analytical、timing table、systolic MXU profile 和 RTL completion importer；
- staged output、周期/stall 汇总、泳道图和 Perfetto trace。

## 当前能力边界

以下能力已定义为后续扩展项，并在编译或运行时保留清晰的语义入口：

- dynamic index 与 dynamic_update_slice：扩展 dynamic index metadata、state binding 和
  runtime address expression；
- stride-aware transform 与完整动态 layout：扩展可验证 stride、layout lowering 和
  memory timing；
- online Softmax 数值实现：扩展 rescale、最终 normalization 和 workspace 生命周期；
- 完整 ResNet/BERT/GPT-J/LLaMA2 repetition 与 DeepSeek MoE routing；
- 论文 WQ/IQ/Fu 容量、dispatch/wake-up/issue/completion 控制开销校准；
- SCALE-Sim、Ramulator2/DRAMSys、RTL/Verilator 和 system simulator backend。

当前默认结果标签为 TISA instruction-level analytical scheduling baseline。RTL profile
加载后，manifest 记录 source、interval、aggregation 和 calibration status。

## 关键里程碑

### 2026-08-29：Runtime state 与 LLaMA2 decode

固定窗口 KV-cache 通过 `state_id/state_buffer` 建立 persistent binding；两次 invocation
复用同一 BackendArtifact，由 `state_complete` 串联并合并事件。LLaMA2 decode one-block
生成 scaled micro workload，输出包含 runtime sequence artifact。

### 2026-08-29：ResNet capability

官方 StableHLO convolution、batch_norm_inference、reduce_window capability 接入
Canonical/GC/FC/backend。卷积和 pooling 的 halo region 进入 TileMem 与 dependency。

### 2026-08-29：GC cost model

SchedulePlanner 支持 tile candidate、计算/traffic/working-set cost，并记录
`candidate_costs` 与 `selected_tile_size`。

### 2026-08-30：shape、layout、dtype 语义

dynamic broadcast、常量 dynamic_slice、常量 dynamic_reshape specialization 接入官方
StableHLO verify；scalar tensor、layout encoding、dtype policy 和 multi-result boundary
进入 Canonical/TISA contract。

### 2026-08-30：论文矩阵

六个 registry case 使用统一 paper-matrix 入口。共享编译产物位于
`00_frontend` 到 `04_backend`，策略结果位于 `policy_matrix/05_runtime` 到
`07_trace`，根目录写入 `matrix_index.json`、`sweep.csv/json`。

## 下一步

按以下顺序推进：

1. symbolic shape、dynamic index 和 layout binding；
2. DeepSeek capability 与完整模型 repetition；
3. scheduler 微结构和控制开销校准；
4. 外部 timing/memory/RTL backend；
5. source-derived 与 RTL-observed 论文矩阵。
