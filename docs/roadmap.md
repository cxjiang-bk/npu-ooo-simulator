# 后续开发路线

## 当前基线

项目已经跑通一条唯一的生产流程：

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

当前已具备：

- 真实 PyTorch module 的命令行入口；
- Matmul、elementwise、reduce、Softmax、LayerNorm、RMSNorm 的基础语义导入与 lowering；
- TISA instruction 粒度的 static/dynamic 调度；
- runtime 地址分配、提交策略和 device policy 四组合实验；
- 可配置 MachineConfig，以及 codegen、timing、event 三类 backend 接口；
- analytical timing、timing table、MXU RTL completion trace/profile 导入；
- 分阶段 artifact、泳道图、Perfetto trace 和 manifest。

当前 cycle 主要来自 analytical backend。没有加载 RTL-observed profile 的结果只适合比较调度趋势，不能作为芯片绝对性能结论。

## A. 扩大真实前端覆盖

目标：让论文中的模型 block 直接以 PyTorch `nn.Module` 输入，不添加模型专用 builder。

优先顺序：

1. 完成 Linear、batched Matmul、reshape/transpose、bias、activation 的组合覆盖；
2. 支持完整 multi-head attention，包括 scale、mask、head reshape 和 output projection（首个静态 shape 两头 Attention 已完成）；
3. 支持 decoder block 的 RoPE、KV-cache、SwiGLU/MLP 和 residual；
4. 支持 ResNet bottleneck 所需的 Conv2D、BatchNorm inference、ReLU 和 pooling；
5. 接入 BERT、GPT-J、LLaMA2、DeepSeek 的一个真实 block，最后再扩展 full model。

每次扩展都沿同一边界完成：

```text
Torch-XLA StableHLO operation
  -> semantic capability/import
  -> graph recovery/fusion（需要时）
  -> TISA stage
  -> backend lowering
  -> PyTorch 端到端测试
```

验收标准：不按 module 名称匹配；不维护项目自有 StableHLO emitter；不允许 unsupported operation 静默降级。

## B. 提升编译正确性与可探索性

目标：在扩大模型覆盖之前，使 tile 数量、依赖和内存流量都可核对。

- 将跨算子的保守依赖细化为基于 tile region 的依赖（第一版 logical region 映射已完成）；
- 完善 symbolic/dynamic shape 与边界 tile；
- 为 layout、transpose、broadcast、reduction barrier 建立显式规则；
- 给 TISA operand 增加可绑定的地址表达式和 memory scope；
- 在统一 planner 中加入多个合法 tile candidate 和可解释 cost model；
- 输出每个 pass 前后的诊断、MAC、传输字节和依赖统计。

验收标准：Static 和 Dynamic 严格复用同一份 `BackendArtifact`；相同输入与配置生成稳定 hash；小图的 tile、依赖、MAC 和 traffic 可手算对账。

## C. 提升 device scheduler 对论文的对齐度

目标：从 instruction-level analytical baseline 逐步逼近论文硬件调度器。

- 明确 reception queue、per-unit WQ/IQ、ROB/Fu 和 completion feedback；
- 实现并验证 typed RAW/WAR/WAW、`payload_ready:<task_id>` partial-ready 原型和 address conflict；
- 校准 dispatch、wake-up、issue 和 completion 的控制开销；
- 验证同一 TISA instruction 的 backend payload 只在 instruction 内部执行，不进入全局 OOO window；
- 为 queue full、dependency wait、resource busy、memory conflict 分别记录 stall 原因。

验收标准：micro-test 能逐 cycle 对账，`window=1` 的退化行为可解释，dynamic 的收益可以从 ready/issue/stall trace 直接定位。

## D. 扩展可插拔硬件 backend

目标：在不修改 compiler 和 scheduler policy 的情况下替换硬件时序来源。

推进顺序：

```text
analytical baseline
  -> SCALE-Sim 类 MXU timing
  -> Ramulator2/DRAMSys 类 memory timing
  -> RTL/Verilator unit timing
  -> system simulator adapter
```

每个 backend 必须声明支持的 operation/resource/memory capability、时序区间和校准状态。未覆盖的任务只能显式失败，或在用户显式允许时回退并记录 mixed calibration。

验收标准：切换 backend 不改变 TISAProgram；trace schema 保持一致；同一 workload 可以比较不同 MachineConfig 与 timing source。

## E. 建立论文实验矩阵

模型 block 覆盖后，固定以下实验维度：

```text
PyTorch model / shape / phase
  x compiler tile candidate
  x runtime policy
  x device scheduler policy
  x MachineConfig
  x timing/event backend
```

首批 case：ResNet50 bottleneck、BERT encoder block、GPT-J one block、LLaMA2 one block，以及确认结构后的 DeepSeek block。Prefill 与 decode 必须作为不同 case，proxy、source-derived、analytical 和 RTL-observed 结果不得混在同一统计组。

## 下一阶段优先级

当前最高优先级是 A 和 B：先用真实 PyTorch module 扩展一个完整 attention/decoder block，同时完善 region dependency 与编译统计。完成后再集中增加 scheduler 微结构细节和外部 backend，避免用不完整的图语义校准硬件时序。
