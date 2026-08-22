# npu-ooo-simulator

面向 NPU tile 级静态与动态调度研究的参数化编译、仿真和可视化框架。

项目目标是建立一条独立、可复现的研究链路：

```text
PyTorch / ONNX / StableHLO
  -> ExecuTorch / Frontend Adapter
  -> Model + Canonical Operator Graph IR
  -> Compiler PassManager
  -> Schedule/Tiling IR
  -> Tile Instance IR
  -> TISAProgram / Semantic Tile Instruction IR
  -> RuntimeSubmission (shape/state/address/command binding)
  -> TISA Device Scheduler (Static 或 Dynamic)
  -> Backend primitive expansion/timing
  -> 热插拔 Timing/Event/System Backend
  -> 总周期、runtime/device 分解、stall 和泳道图
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

1. **编译、runtime 与设备调度分离**：Compiler 生成 task，Runtime 负责提交，Device Scheduler 只负责已提交 task 的 issue 决策。
2. **Static/Dynamic 共用执行模型**：tile、地址、依赖、buffer、资源和 latency 完全一致。
3. **TISA 语义层独立存在**：TileInstance 描述几何 bounds，TISAInstruction 描述 OpType/Operand/UnitMap/typed Deps，ExecutionTask 只表示 backend primitive。
4. **Backend 热插拔**：TimingProvider、EventBackend 和可选 SystemBackend 通过统一 compiled/runtime contract 接入。
5. **Mapping 与 timing 分层**：TileFlow/Timeloop 可提供 mapping 与 aggregate cost，但不冒充 per-tile 时间线。
6. **算子 lowering 可扩展**：scheduler 不依赖 `ProduceQ` 等特定算子或 tensor 名称。
7. **先可解释、再校准**：第一版是确定性的离散事件模型，不宣称 RTL cycle-accurate。

## 文档

- [总体架构](docs/architecture.md)
- [TISA 对齐说明](docs/tisa-alignment.md)
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
│   ├── frontend/            # JSON/canonical bridge and optional torch.export adapter
│   ├── compiler/            # canonical graph -> schedule/tile/TISA/backend pipeline
│   ├── runtime/             # planned: buffer binding and submission
│   ├── backend/             # planned: hot-pluggable backends
│   ├── ir/                  # model, operator, schedule, tile, program, execution IR
│   ├── arch/                # machine schema, validators, profile importers
│   ├── lowering/            # operator tile -> primitive tasks
│   ├── scheduler/           # sequential, static pipeline, dynamic ready queue
│   ├── simulator/           # deterministic discrete-event engine
│   ├── trace/               # CSV/JSON/Perfetto exporters
│   └── cli.py              # compile, simulate, compare entry points
├── tests/
└── docs/
```

### Framework bridge 层级

论文中的 `torchxla` 属于最上游的 framework bridge：它把 PyTorch/JAX/TensorFlow
程序导出到 XLA/StableHLO 图。`StableHLO` 是 bridge 之后、Graph Compiler 之前的
可移植图/算子 IR；它不是 TISA，也不是 NPU backend ISA。Graph Compiler 在
StableHLO 上做 fusion、tiling、layout 和 memory planning，Fusion Compiler/TISA
Generator 再输出语义 tile 指令。

本项目先采用不依赖大型 MLIR 工具链的等价边界：

```text
torch.export / ExecuTorch       (framework bridge, optional dependency)
canonical JSON / in-memory IR   (framework-independent bridge for tests)
              -> Canonical OperatorGraph
              -> compiler pipeline (schedule -> TileGraph -> TISAProgram)
              -> BackendArtifact (descriptor + analytical primitive payload)
```

`TorchExportAdapter` 只在被调用时导入 `torch`，所以没有 PyTorch 的环境仍可使用
JSON/canonical frontend 和后端 simulator。未来的 StableHLO adapter 应汇入同一个
`FrontendImport`，而不应把 StableHLO node name 直接暴露给 scheduler。

## 第一条闭环

当前已完成的第一条可运行路径是：

```text
2mm
  -> Model IR benchmark case
  -> 手写 Operator Graph（仅作为 baseline/fixture）
  -> 显式 tiling schedule
  -> tile instance graph
  -> configurable DMA/MXU tasks
  -> sequential/static pipeline
  -> dynamic ready queue
  -> Static/Dynamic 对比泳道图
```

ARU/reduction、Attention、模型 proxy 和多架构 sweep 已在该 baseline 上扩展；下一步重点转向自动 frontend、runtime submission 和热插拔 backend。

下一条开发路径将替换手写 graph 入口：

```text
PyTorch module
  -> torch.export / ExecuTorch
  -> Canonical OperatorGraph
  -> Compiler PassManager
  -> TISAProgram
  -> RuntimeSubmission
  -> analytical TISA Device Scheduler
  -> primitive timing backend
```

runtime 和 device scheduler 将分别记录提交顺序与硬件 issue 顺序，并支持 static/dynamic runtime × static/dynamic device 的四种组合实验。

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

`attention` 提供首个 attention 级闭环，固定为单头、无 mask/cache 的 `Q @ K^T -> Softmax -> P @ V`：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli attention \
  --arch minimal \
  --policy dynamic_ready_queue \
  --output-dir out/attention-dynamic
```

这是 attention 数据流和调度实验的最小 fragment，不等同完整 GQA/MQA、RoPE、KV-cache 或 flash attention 实现。

`transformer-block` 在此基础上连接 LayerNorm、attention、MLP 和两次 residual：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli transformer-block \
  --arch minimal \
  --policy dynamic_ready_queue \
  --output-dir out/transformer-block-dynamic
```

它是 shape-only 的 decoder block skeleton，默认生成 9 个 semantic operators、30 tiles、126 primitive tasks；目前仍不包含真实 Q/K/V projection、GQA、RoPE、KV-cache、causal mask 或 fused flash attention。

`model-block` 在同一 skeleton 上增加模型级 benchmark preset。当前提供 `bert-base`、`gpt-j`、`llama2-7b` 和 `deepseek-r1-16b`，每个 preset 都把 native model metadata 和明确的 proxy assumptions 写入 `model_spec.json` / `benchmark_case.json`。默认使用小型 proxy shape，便于快速观察调度；可用 `--tokens`、`--sequence`、`--head-dim` 和 `--intermediate` 放大或缩小实例：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli model-block \
  --model-preset llama2-7b \
  --tokens 16 \
  --sequence 16 \
  --head-dim 16 \
  --intermediate 32 \
  --policy dynamic_ready_queue \
  --output-dir out/model-block-llama2
```

这些 preset 是模型层和调度链路的 proxy benchmark，不等同于真实 GPT-J/LLaMA2/BERT/DeepSeek 的完整算子图。尤其 DeepSeek-R1 的 dense/MoE 配置仍需外部模型配置确认；当前 preset 明确标记为 dense shape-only proxy，不包含 expert routing。

模型 preset 也可直接加入统一 sweep（这里用小 tile 只做调度趋势探针）：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli sweep-workloads \
  --workloads bert-base,gpt-j,llama2-7b,deepseek-r1-16b \
  --architectures minimal \
  --policies static_pipeline,dynamic_ready_queue \
  --windows 4 \
  --robs 4 \
  --tile-sizes 16 \
  --model-tokens 16 \
  --model-sequence 16 \
  --model-head-dim 16 \
  --model-intermediate 32 \
  --output-dir out/sweep-model-proxies
```

`--model-*` 只作用于命名 model preset；同一组覆盖值会让各模型共享 shape，适合先验证调度公平性。要比较不同 proxy 规模，应为每个 preset 单独运行 `model-block`，并保留 `proxy_shape` 作为结果索引。

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

同时提供 `operator_graph.dot`、`tile_graph.dot`、`execution_graph.dot`，用于 Graphviz 或其他图分析工具。`--arch` 可选 `minimal`、`wide-mxu`、`lpu-like`；`--machine-config path/to/machine.json` 可以直接加载 canonical MachineConfig JSON，用于探索任意自定义 memory/unit/path 参数；`--policy` 可选 `sequential`、`static_pipeline`、`dynamic_ready_queue`。manifest 中的 `machine_hash` 是实际加载配置的稳定标识。

运行时容量可以通过 `--instruction-queue-depth`、`--rob-entries`、`--max-inflight-tiles`、`--dependency-window` 和 `--ready-queue-depth` 覆盖 MachineConfig 默认值；实际生效值会写入 `manifest.json` 和 `summary.json`。`--address-scoreboard` 启用运行时 range scoreboard：活跃 task 的重叠 `BufferRegion` 会产生 RAW/WAR/WAW issue stall，完成后释放并唤醒等待者。动态 policy 可用 `--dynamic-priority critical_path|oldest_first` 切换启发式。静态流水线可用 `--static-stage-offsets 0,200 --static-stage-ii 250` 显式指定 stage reservation；不提供该参数时保留默认 program-order static baseline。

Timing 也可以从 JSON 表覆盖，而不修改 lowering 或 simulator：

```json
{
  "name": "rtl_probe_v0",
  "entries": {
    "matmul": {"duration_cycles": 64, "initiation_interval_cycles": 4},
    "MXU:matmul": {"duration_cycles": 64, "initiation_interval_cycles": 4}
  }
}
```

仓库提供了一个可以直接运行的示例表 `configs/timing/attention_probe.json`：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli attention \
  --arch minimal \
  --timing-config configs/timing/attention_probe.json \
  --policy dynamic_ready_queue \
  --output-dir out/attention-calibrated
```

匹配优先级是 `timing_key`、task id、`resource:primitive`、primitive、resource、`default`；未覆盖的 task 回退到 analytical timing。结果的 `backend` 会记录 timing table 名称。`path/to/timing.json` 只是占位写法，不能直接作为命令运行。

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

跨算子和跨模型结构的实验使用 `sweep-workloads`。它复用 lowering registry，把每个 workload 编译到同一套 ExecutionGraph/SchedulerResult artifact：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli sweep-workloads \
  --workloads elementwise,layernorm,decoder-block \
  --architectures minimal,wide-mxu \
  --policies static_pipeline,dynamic_ready_queue \
  --windows 4,8 \
  --robs 4,8 \
  --tile-sizes 16,32 \
  --dynamic-priorities critical_path,oldest_first \
  --output-dir out/sweep-workloads
```

每个 case 目录同时保留 `operator_graph.json`、`execution_graph.json`、`summary.json`、`tasks.csv`、`perfetto.json`、`swimlane.svg` 和 `swimlane.png`，顶层 `sweep.csv/json` 增加 workload、dynamic priority 字段并计算相对 static 的 speedup。Static baseline 会对每个 priority 值重复记录，确保比较键完全一致。

使用外部架构文件时，sweep 的 architecture label 可以是任意字符串；实际硬件参数来自 `--machine-config`，label 和 `machine_hash` 会同时写入 manifest。

## 参考项目

- ExecuTorch：`torch.export()`、Core ATen graph 和 backend partition；
- TVM/TileLang：Relax/TensorIR 分层、tile schedule、pipeline 和 layout；
- TileRT：软件 tile task runtime、event 和 compute/I/O overlap；
- TileFlow/Timeloop：tiling、mapping、memory traffic 和 aggregate cost；
- TVM-VTA：静态 LOAD/COMPUTE/STORE pipeline 和 dependency token；
- Gemmini/NVDLA：参数化或工业风格 accelerator queue、DMA、scratchpad 和 RTL 参考；
- SCALE-Sim：systolic array timing 和 memory bandwidth 模型；
- Ramulator2/DRAMSys：DRAM 请求和 bank timing；
- gem5/gem5-SALAM：可选的 CPU + NPU full-system backend；
- Perfetto：多资源泳道和事件分析。

这些项目只作为架构与局部 backend 参考。该仓库保持自己的 canonical IR、TISAProgram/RuntimeSubmission contract、MachineConfig 和 trace schema。
