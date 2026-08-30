# 后续开发路线

## 当前基线

项目已跑通以下生产链路：

```text
PyTorch nn.Module
  -> torch.export
  -> Torch-XLA StableHLO
  -> official StableHLO verify/import
  -> Canonical OperatorGraph
  -> ScheduleSpec / TileGraph
  -> TISAProgram / BackendArtifact
  -> RuntimeSubmission
  -> static 或 dynamic TISA device simulation
  -> 周期、泳道图和 trace
```

当前基线包含真实 PyTorch module 入口、Matmul/elementwise/reduce/Softmax/LayerNorm/
RMSNorm/Attention/SwiGLU/RoPE/Conv2D/BatchNorm/pooling 语义、TISA instruction 粒度
static/dynamic 调度、runtime 地址与四组合 policy、可配置 MachineConfig、可替换
codegen/timing/event backend、RTL completion profile importer，以及分阶段 artifact。

默认 cycle 来源为 analytical backend。RTL-observed profile 通过 manifest 的 calibration
status 单独标识，结果用于与 analytical 趋势分组比较。

## 阶段 A：前端与模型覆盖

目标：让论文模型 block 统一由 PyTorch module 驱动，沿 Torch-XLA -> StableHLO -> GC ->
FC -> TISA 链路进入 simulator。

已完成：

- 静态 Attention、pre-norm decoder、BERT/GPT-J/LLaMA2/DeepSeek dense one-block；
- LLaMA2 RoPE、固定窗口 KV-cache 和多步 RuntimeSequence；
- ResNet bottleneck micro 的 Conv2D、BatchNorm inference、ReLU、pooling；
- StableHLO capability registry 与 semantic fusion registry。

当前工作项：

1. 完成 DeepSeek dense/MoE 结构的 capability 清单；
2. 扩展完整 ResNet50、BERT、GPT-J、LLaMA2 的层重复与论文形状 proxy；
3. 将 embedding、position embedding、causal mask 和更多 layout 变体纳入统一 registry。

每个新能力遵循：

```text
Torch-XLA StableHLO operation
  -> semantic capability/import
  -> graph recovery/fusion
  -> TISA stage
  -> backend lowering
  -> PyTorch regression
```

模型名称作为 provenance 和 benchmark registry 字段。

## 阶段 B：编译语义与可审计性

目标：让 tile、依赖、内存流量和 shape 变换都能从 artifact 逐项核对。

已完成：

- PassManager、统一 tile planner、candidate cost model；
- region-aware dependency、卷积/池化 halo、静态 broadcast 和 scalar region；
- TileMem 的 stride expression、layout metadata、dtype policy；
- GC pass dump、compile statistics、residency/ping-pong intent；
- 常量 dynamic broadcast、dynamic_slice、dynamic_reshape specialization；
- readiness condition、state/accumulate/buffer-reuse dependency。

当前工作项：

1. 建立 symbolic shape 的统一 binding contract；
2. 扩展 dynamic index、dynamic layout 和 stride-aware transform；
3. 为每类 layout、transpose、broadcast、reduction 建立可验证的 region rule；
4. 为完整模型 proxy 增加 shape/traffic 对账样例。

验收标准：同一 module、shape、tile、MachineConfig 和 backend 产生稳定 artifact hash；
static/dynamic 的差异来自 policy；小图的 tile、MAC、traffic 和 dependency 可手算核对。

## 阶段 C：Device scheduler 对齐

目标：逐步复现论文 WQ/IQ/Fu/ROB 和 completion feedback 的行为。

已完成：

- reception availability、queue、ROB/window、资源占用和 completion feedback analytical
  模型；
- typed RAW/WAR/WAW/STATE/ACCUMULATE、address scoreboard、partial-ready 原型；
- 可选 memory bank/port structural-conflict 模型；
- queue、dependency、resource、memory stall 计数和泳道事件。

当前工作项：

1. 根据论文和 NPU ISA 参数化 dispatch、wake-up、issue、completion 控制开销；
2. 校准 WQ/IQ/Fu 容量、dispatch width、in-flight tile 和 queue backpressure；
3. 让 partial-ready 从 backend calibration contract 进入 GC 语义；
4. 增加逐 cycle micro-test 对账和稳定的 stall taxonomy。

## 阶段 D：可插拔硬件 backend

目标：在 compiler 与 scheduler 接口稳定的基础上替换时序来源。

推进顺序：

```text
analytical baseline
  -> SCALE-Sim 类 MXU timing
  -> Ramulator2 / DRAMSys memory timing
  -> RTL / Verilator unit timing
  -> system simulator adapter
```

每个 backend 声明 operation、resource、memory capability、timing interval 和 calibration
status。Backend 切换保留 TISAProgram、trace schema 和实验 manifest 结构。

## 阶段 E：论文实验矩阵

固定维度：

```text
PyTorch model / input shape / phase
  x tile candidate
  x runtime policy
  x device policy
  x MachineConfig
  x timing/event backend
```

registry case：

- ResNet50 bottleneck；
- BERT encoder block；
- GPT-J one block；
- LLaMA2 prefill/decode one block；
- DeepSeek dense prefill/decode one block，MoE 作为后续 case。

矩阵入口 paper-matrix 每个 case 编译一次，在共享 BackendArtifact 和 buffer binding
上运行 policy 组合。micro 用于回归，paper_shape 用于 representative proxy。输出
按 case/variant 保存共享的 00_frontend 到 04_backend，策略目录保存
05_runtime 到 07_trace，根目录通过 sweep.csv/json 汇总。

下一阶段优先级：

1. B 阶段 symbolic/dynamic shape、layout 和动态索引语义；
2. A 阶段 DeepSeek 与完整模型 repetition；
3. C 阶段 scheduler 微结构校准；
4. D 阶段外部 timing/memory/RTL backend；
5. E 阶段 source-derived 与 RTL-observed 论文矩阵。
