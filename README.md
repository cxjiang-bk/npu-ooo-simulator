# npu-ooo-simulator

面向 NPU tile 级静态与动态调度研究的参数化编译、仿真和可视化框架。

项目目标是建立一条独立、可复现的研究链路：

```text
模型/benchmark case
  -> Model IR
  -> Operator Graph IR
  -> Schedule/Tiling IR
  -> Tile Instance IR
  -> Primitive Execution Graph
  -> Static 或 Dynamic Scheduler
  -> 参数化离散事件 Simulator
  -> 总周期、利用率、stall 分解和泳道图
```

当前阶段只冻结架构和实施计划，不包含 simulator 实现。

模型层是必要的：论文 benchmark 同时覆盖 ResNet50、BERT、GPT-J、LLaMA2 和 DeepSeek-R1，并区分 CNN、Transformer、prefill、decode、batch、sequence length 和 dtype。Operator IR 负责描述“一个算子做什么”，Model IR 负责描述“哪些算子以什么拓扑、重复次数和运行阶段组成一个 workload”。

## 研究目标

- 支持 GEMM、2mm、elementwise、reduction/softmax 和 Attention benchmark；
- 支持 CNN、encoder Transformer、decoder Transformer、长上下文和可选 MoE workload；
- 在同一 tile graph 和同一硬件配置上公平比较 Static 与 Dynamic 调度；
- 支持 sequential、static dual/triple pipeline 和 TISA-like dynamic dual/triple pipeline；
- 后端架构参数可配置，不绑定某一版 NPU RTL；
- 输出 cycle-native event trace 和 Perfetto/Chrome Trace 泳道图；
- 允许后续使用真实 ISA、RTL/Verilator 或硬件计数器校准 timing model。

## 设计原则

1. **架构与调度分离**：MachineConfig 描述硬件，SchedulerPolicy 只描述 issue 决策。
2. **Static/Dynamic 共用执行模型**：tile、地址、依赖、buffer、资源和 latency 完全一致。
3. **Mapping 与 timing 分层**：TileFlow/Timeloop 可提供 mapping 与 aggregate cost，但不冒充 per-tile 时间线。
4. **算子 lowering 可扩展**：scheduler 不依赖 `ProduceQ` 等特定算子或 tensor 名称。
5. **先可解释、再校准**：第一版是确定性的离散事件模型，不宣称 RTL cycle-accurate。

## 文档

- [总体架构](docs/architecture.md)
- [实施路线图](docs/roadmap.md)
- [持续任务计划](task_plan.md)
- [研究发现与决策](findings.md)
- [进度日志](progress.md)

## 计划目录

```text
npu-ooo-simulator/
├── configs/
│   ├── architectures/       # MachineConfig profiles
│   ├── schedules/           # static/dynamic scheduler configs
│   └── experiments/         # benchmark matrix
├── benchmarks/              # operator/fusion graph descriptions
├── src/npu_ooo/
│   ├── ir/                  # model, operator, schedule, tile, execution IR
│   ├── arch/                # machine schema, validators, profile importers
│   ├── lowering/            # operator tile -> primitive tasks
│   ├── scheduler/           # sequential, static pipeline, dynamic ready queue
│   ├── simulator/           # deterministic discrete-event engine
│   ├── trace/               # CSV/JSON/Perfetto exporters
│   └── cli/                 # compile, simulate, compare
├── tests/
└── docs/
```

## 第一条闭环

第一条可运行路径固定为：

```text
2mm
  -> Model IR benchmark case
  -> 手写 Operator Graph
  -> 显式 tiling schedule
  -> tile instance graph
  -> configurable DMA/MXU tasks
  -> sequential/static pipeline
  -> dynamic ready queue
  -> Static/Dynamic 对比泳道图
```

在该闭环稳定后，再加入 ARU/reduction、Attention 和更复杂的硬件资源。

## 参考项目

- TileFlow/Timeloop：tiling、mapping、memory traffic 和 aggregate cost；
- TVM-VTA：静态 LOAD/COMPUTE/STORE pipeline 和 dependency token；
- Gemmini：load/execute/store queues、ROB 和参数化 accelerator；
- SCALE-Sim：systolic array timing 和 memory bandwidth 模型；
- Perfetto：多资源泳道和事件分析。

这些项目只作为架构与实现参考。该仓库保持自己的 canonical IR、MachineConfig 和 trace schema。
