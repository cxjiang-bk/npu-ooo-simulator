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

当前已经具备 2mm、residual-add、row-reduce、softmax 和 RMSNorm 五条可执行算子闭环，并新增一个 `RMSNorm -> Matmul -> ResidualAdd` decoder block 混合图。混合图通过 lowering registry 按拓扑逐算子展开，再按显式 tensor edge 和 root-memory region overlap 建立跨算子依赖。所有 workload 都经过模型实例化、显式 tiling、primitive execution graph，以及基于 MachineConfig 的 sequential/static-pipeline/dynamic-ready-queue analytical scheduler。输出包含总周期、等待分解和 Perfetto/Chrome Trace 事件；数值仍标记为 analytical，尚未宣称 RTL cycle-accurate。

模型层是必要的：论文 benchmark 同时覆盖 ResNet50、BERT、GPT-J、LLaMA2 和 DeepSeek-R1，并区分 CNN、Transformer、prefill、decode、batch、sequence length 和 dtype。Operator IR 负责描述“一个算子做什么”，Model IR 负责描述“哪些算子以什么拓扑、重复次数和运行阶段组成一个 workload”。

## 研究目标

- 支持 GEMM、2mm、elementwise/residual-add、reduction/softmax 和 Attention benchmark；
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

### Elementwise benchmark

`elementwise` 命令使用一个两输入 residual-add 图，lowering 为 `load -> ARU elementwise -> store`，与 2mm 共用 scheduler 和 simulator：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli elementwise \
  --arch minimal \
  --policy dynamic_ready_queue \
  --dependency-window 4 \
  --rob-entries 4 \
  --output-dir out/elementwise-dynamic
```

`reduce` 显式保留跨 reduction tile 的 partial accumulation chain；`softmax` 则展开为 `reduce_max -> exp -> reduce_sum -> normalize` composite primitives：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli softmax \
  --arch minimal \
  --policy dynamic_ready_queue \
  --dynamic-priority oldest_first \
  --dependency-window 8 \
  --rob-entries 8 \
  --output-dir out/softmax-dynamic
```

`rmsnorm` 展开为 `load -> square -> reduce_sum_square -> rmsnorm -> store`，并把最终 sum barrier 传播到同一行的每个 normalize tile：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli rmsnorm \
  --arch minimal \
  --policy dynamic_ready_queue \
  --output-dir out/rmsnorm-dynamic
```

`layernorm` 显式展开均值和方差两条 barrier：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli layernorm \
  --arch minimal \
  --policy dynamic_ready_queue \
  --output-dir out/layernorm-dynamic
```

其 primitive 图为 `load -> reduce_sum -> layernorm_mean -> center -> reduce_sum_square -> layernorm -> store`；当前 epsilon 和数值近似只保留在 operator attributes，周期仍来自 analytical timing。

`decoder-block` 使用同一套 registry 和 simulator 运行一个 decoder one-block fragment：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli decoder-block \
  --arch minimal \
  --policy dynamic_ready_queue \
  --dependency-window 8 \
  --rob-entries 8 \
  --output-dir out/decoder-block-dynamic
```

该命令会同时生成 RMSNorm、Matmul、ResidualAdd 的 execution graph，以及跨算子 root-memory handoff 依赖。Static 与 Dynamic 应使用同一输出目录之外的同一 graph/machine 配置进行对比。

当前已实现到 `Execution Graph -> analytical event simulator -> SchedulerResult`，输出 CSV/SVG、Perfetto/Chrome Trace 和 runtime queue/ROB 指标。后端仍是 analytical timing，不是 RTL cycle-accurate；真实 ISA/RTL timing 后续通过 `TimingModel` 接入。

### 快速运行

无需安装第三方运行时依赖时，可直接从源码运行：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli two-mm \
  --arch minimal \
  --policy dynamic_ready_queue \
  --dependency-window 8 \
  --rob-entries 8 \
  --output-dir out/two-mm-dynamic
```

命令会同时生成编译图和调度结果：

```text
model_spec.json         模型模板
benchmark_case.json     本次 benchmark 参数
model_instance.json     shape 实例化后的模型
operator_graph.json     semantic operator/tensor 计算图
operator_graph.svg      可直接查看的顶层计算图
schedule.json           tile factor、loop order、stage
tile_graph.json         具体 tile 实例和依赖
execution_graph.json    load/matmul/store task 及依赖
address_dependencies.json  运行时观测到的 RAW/WAR/WAW 冲突
machine.json            本次 MachineConfig
manifest.json           配置 hash、policy、周期和统计
tasks.csv               task start/finish 时间
swimlane.svg            资源泳道图
swimlane.png            PNG 资源泳道图（由 ImageMagick/librsvg 导出）
perfetto.json           Perfetto/Chrome Trace
```

同时提供 `operator_graph.dot`、`tile_graph.dot`、`execution_graph.dot`，用于 Graphviz 或其他图分析工具。`--arch` 可选 `minimal`、`wide-mxu`、`lpu-like`；`--policy` 可选 `sequential`、`static_pipeline`、`dynamic_ready_queue`。

运行时容量可以通过 `--instruction-queue-depth`、`--rob-entries`、`--max-inflight-tiles`、`--dependency-window` 和 `--ready-queue-depth` 覆盖 MachineConfig 默认值；实际生效值会写入 `manifest.json` 和 `summary.json`。`--address-scoreboard` 启用运行时 range scoreboard：活跃 task 的重叠 `BufferRegion` 会产生 RAW/WAR/WAW issue stall，完成后释放并唤醒等待者。动态 policy 可用 `--dynamic-priority critical_path|oldest_first` 切换启发式。静态流水线可用 `--static-stage-offsets 0,200 --static-stage-ii 250` 显式指定 stage reservation；不提供该参数时保留默认 program-order static baseline。

批量比较使用 `sweep-two-mm`。它对每个 architecture/policy/window/ROB 组合重新执行相同的 2mm lowering 和 simulator，并为每个组合写入独立目录：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli sweep-two-mm \
  --architectures minimal,wide-mxu \
  --policies static_pipeline,dynamic_ready_queue \
  --tile-sizes 16,32 \
  --windows 4,8 \
  --robs 4,8 \
  --output-dir out/sweep-two-mm
```

顶层的 `sweep.csv` / `sweep.json` 汇总 tile size、total cycles、相对 static 的 speedup、ROB/ready peak、stall 分解和 pipeline drain；各 case 子目录保留 `manifest.json`、`summary.json`、`tasks.csv`、`address_dependencies.json`、`perfetto.json`、`swimlane.svg` 和 `swimlane.png`。

## 参考项目

- TileFlow/Timeloop：tiling、mapping、memory traffic 和 aggregate cost；
- TVM-VTA：静态 LOAD/COMPUTE/STORE pipeline 和 dependency token；
- Gemmini：load/execute/store queues、ROB 和参数化 accelerator；
- SCALE-Sim：systolic array timing 和 memory bandwidth 模型；
- Perfetto：多资源泳道和事件分析。

这些项目只作为架构与实现参考。该仓库保持自己的 canonical IR、MachineConfig 和 trace schema。
