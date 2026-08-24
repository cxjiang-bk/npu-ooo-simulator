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

## 项目目录与输出目录

```text
npu-ooo-simulator/
├── configs/                 # 可版本化的架构、时序和实验配置
│   ├── architectures/       # MachineConfig 架构 profile
│   ├── schedules/           # Static/Dynamic 调度参数
│   ├── timing/              # primitive 时序覆盖表
│   └── experiments/         # benchmark 矩阵
├── benchmarks/              # 算子、融合图和模型 proxy 定义
├── src/npu_ooo/
│   ├── frontend/            # framework bridge：JSON、torch.export/ExecuTorch 适配器
│   ├── compiler/            # Canonical Graph -> Schedule/Tile -> TISA/Backend
│   ├── runtime/             # 规划中：地址绑定、command buffer 和提交策略
│   ├── backend/             # 规划中：可热插拔的 timing/event/system backend
│   ├── ir/                  # Model、Operator、Schedule、Tile、TISA、Execution IR
│   ├── arch/                # MachineConfig schema、校验和 profile 导入
│   ├── lowering/            # semantic operator -> backend primitive task
│   ├── scheduler/           # sequential、Static pipeline、Dynamic ready queue
│   ├── simulator/           # 确定性离散事件仿真和 address scoreboard
│   ├── trace/               # JSON、CSV、SVG、PNG、Perfetto 导出与输出布局
│   └── cli.py               # 编译、仿真、sweep 和对比实验入口
├── tests/                   # 单元测试和 golden case
├── docs/                    # 架构、TISA 对齐、路线图和研究记录
├── task_plan.md             # 当前阶段和验收条件
├── findings.md              # 论文/开源项目研究结论
└── progress.md              # 持续进度与验证日志
```

每次运行命令的 `--output-dir` 都会生成一个独立的实验目录。例如：

```text
out/attention-dynamic/
├── README.md                # 本次运行的目录说明和推荐查看顺序
├── artifact_index.json      # 阶段目录到文件的机器可读索引
├── manifest.json            # benchmark、架构、policy、backend、配置 hash
├── 00_frontend/             # 输入模型、benchmark case、frontend provenance
│   ├── model_spec.json
│   ├── benchmark_case.json
│   ├── model_instance.json
│   ├── generated.mlir       # --through-stablehlo 时生成的 StableHLO 文本
│   └── stablehlo_module.json
├── 01_graph_ir/             # Canonical OperatorGraph 和图可视化
│   ├── operator_graph.json
│   ├── operator_graph.dot
│   └── operator_graph.svg
├── 02_schedule_tile/        # tile factor、loop order、边界 tile 和依赖
│   ├── schedule.json
│   ├── tile_graph.json
│   └── tile_graph.dot
├── 03_tisa/                 # TISA descriptor、语义指令和编译产物
│   ├── tisa_program.json
│   └── compiled_artifact.json
├── 04_backend/              # MachineConfig、backend payload 和 primitive graph
│   ├── machine.json
│   ├── backend_artifact.json
│   └── execution_graph.json
├── 05_runtime/              # 地址绑定/运行时观察到的依赖和 hazard
│   └── address_dependencies.json
├── 06_simulation/           # 周期、stall、队列指标和每个 task 的时序
│   ├── summary.json
│   └── tasks.csv
└── 07_trace/                # 适合浏览器/图形工具查看的时间线
    ├── perfetto.json
    ├── swimlane.svg
    └── swimlane.png
```

目录编号对应从前端到后端的实际处理顺序，便于定位问题：

1. `00_frontend` 为空或内容不对，说明模型导入/shape 约束有问题；
2. `01_graph_ir` 反映规范化后的算子语义和 tensor 拓扑；
3. `02_schedule_tile` 反映切分、边界 tile、loop order 和 compile-time 依赖；
4. `03_tisa` 用来检查 `OpType`、`TileMem`、`UnitMap` 和 typed dependency 是否保留；
5. `04_backend` 用来检查具体机器参数以及 TISA 到 primitive payload 的展开；
6. `05_runtime` 用来区分地址冲突等运行时观察，不把它们误认为编译期依赖；
7. `06_simulation` 和 `07_trace` 用来比较总周期、stall 分解和不同调度策略的泳道。

当前已有的 primitive baseline 命令仍直接生成 `ExecutionGraph`，因此它们的
`03_tisa/` 可能暂时为空；`compile-model` 路径已经生成 `TISAProgram`/`BackendArtifact`，
后续 TISA target scheduler 接入后，所有自动编译路径都会填充该目录。

为兼容旧脚本，常用文件名仍会在实验目录顶层生成相对符号链接，例如
`operator_graph.json -> 01_graph_ir/operator_graph.json`。规范位置以编号目录为准；
批量 sweep 的 `sweep.csv`、`sweep.json` 和每个 case 目录仍保留在 sweep 根目录下，
便于批量扫描工具直接读取。

### Framework bridge 层级

论文中的 `torchxla` 属于最上游的 framework bridge：它把 PyTorch/JAX/TensorFlow
程序导出到 XLA/StableHLO 图。`StableHLO` 是 bridge 之后、Graph Compiler 之前的
可移植图/算子 IR；它不是 TISA，也不是 NPU backend ISA。Graph Compiler 在
StableHLO 上做 fusion、tiling、layout 和 memory planning，Fusion Compiler/TISA
Generator 再输出语义 tile 指令。

本项目现在提供正式的官方 StableHLO 路径，以及仅用于回归测试的轻量路径：

```text
torch.export                    (PyTorch capture)
              -> torch-xla exporter 或 project legalizer
              -> official StableHLO parser + verifier
              -> Canonical OperatorGraph
              -> compiler pipeline (schedule -> TileGraph -> TISAProgram)
              -> BackendArtifact (descriptor + analytical primitive payload)
```

`TorchExportAdapter` 只在被调用时导入 `torch`，所以没有 PyTorch 的环境仍可使用
JSON/canonical frontend 和后端 simulator。正式路径由 `OfficialStableHLOGenerator`
生成标准 reducer region、`broadcast_in_dim`、`dot_general` 和 transpose 属性，随后由
`OfficialStableHLOAdapter` 注册官方 dialect，执行 `Module.parse()` 和
`module.operation.verify()`。验证后的模块再汇入同一个 `FrontendImport`，scheduler
不会依赖 MLIR node name。旧 `StableHLOGenerator/StableHLOAdapter` 只保留为显式
`textual` regression backend，不用于正式实验结果。

官方 wheel 的安装与版本说明见 [docs/install-stablehlo.md](docs/install-stablehlo.md)。
需要注意，官方 StableHLO 包不是 PyTorch exporter。默认 `project` exporter 由本项目
完成当前宽覆盖的 `torch.export -> StableHLO` legalization；论文同构实验使用已接入的
`torch-xla==2.9.0` exporter。Matmul、`QK^T -> Softmax -> PV` micrograph 和包含
LayerNorm/四个 Linear 的完整 attention block 均已走通。两者复用相同官方 verifier 和
下游 compiler contract，manifest 会明确记录 producer，不能混为同一个实现。

当前真实 PyTorch 模型可以通过 Python API 进入统一编译入口：

项目把已验证版本声明为可选依赖，可在具备对应 PyTorch wheel 的 Python 环境中安装：

```bash
python3.12 -m pip install -e '.[torch]'
```

```python
from npu_ooo.arch import minimal_machine_config
from npu_ooo.compiler import compile_torch_module

compiled = compile_torch_module(
    module.eval(),
    (example_input,),
    minimal_machine_config(),
    model_id="my_model",
)
```

该 API 执行 `torch.export -> FrontendImport -> PassManager -> Canonical OperatorGraph ->
SchedulePlanner -> TileGraph -> TISAProgram -> BackendArtifact`。当前 PassManager 已
包含结构规范化、Linear decomposition、RHS transpose fold 和 RMSNorm pattern fusion；
真实 PyTorch 2.9.1 已验证 Linear、三维 RMSNorm/LayerNorm、Softmax 和
`QK^T -> Softmax -> PV` attention micrograph。

StableHLO textual MLIR 可以直接经 CLI 编译并仿真：

```bash
PYTHONPATH=src:. /usr/bin/python3.12 -m npu_ooo.cli compile-model \
  --stablehlo-file examples/stablehlo/matmul.mlir \
  --stablehlo-backend official \
  --arch minimal \
  --policy dynamic_ready_queue \
  --output-dir out/stablehlo-matmul
```

真实 PyTorch module 也可通过零参数 factory 进入同一命令。默认路径直接使用
`torch.export -> Canonical OperatorGraph`；加上 `--through-stablehlo` 可以显式走
论文形态的 StableHLO round-trip：

```bash
PYTHONPATH=src:. /usr/bin/python3.12 -m npu_ooo.cli compile-model \
  --torch-module examples.torch_models:attention_block \
  --input-shape 1,4,8 \
  --input-dtype float32 \
  --tile-size 4 \
  --arch minimal \
  --policy dynamic_ready_queue \
  --output-dir out/torch-attention-block
```

```bash
PYTHONPATH=src:. /usr/bin/python3.12 -m npu_ooo.cli compile-model \
  --torch-module examples.torch_models:attention_block \
  --input-shape 1,4,8 \
  --tile-size 4 \
  --arch minimal \
  --policy dynamic_ready_queue \
  --through-stablehlo \
  --stablehlo-backend official \
  --output-dir out/torch-attention-stablehlo
```

使用 torch-xla framework bridge 的完整 attention block：

```bash
PYTHONPATH=src:. PJRT_DEVICE=CPU /usr/bin/python3.12 -m npu_ooo.cli compile-model \
  --torch-module examples.torch_models:attention_block \
  --input-shape 1,4,8 \
  --tile-size 4 \
  --arch minimal \
  --policy dynamic_ready_queue \
  --through-stablehlo \
  --stablehlo-backend official \
  --stablehlo-exporter torch-xla \
  --output-dir out/torch-xla-attention
```

这条命令真实执行
`torch.export -> torch_xla.stablehlo.exported_program_to_stablehlo -> official verifier ->
Canonical Graph -> TISA`。当前完整 block 结果为 13 个 semantic operators、33 tiles、
112 TISA instructions、134 primitive tasks；与直接 TorchExport 路径的结构计数一致。
torch-xla 合法地把 V projection 排到 Softmax 之后，因此不要求两条路径的 SSA 名称和
拓扑序逐项相同。

启用该选项后，`00_frontend/` 会额外生成 `generated.mlir` 和
`stablehlo_module.json`；`frontend_import.json` 表示 StableHLO 重新导入后的图，
`source_frontend_import.json` 保留原始 TorchExport 图。编译器会在 StableHLO primitive
链上执行 Softmax、LayerNorm、RMSNorm fusion，然后继续使用同一套 tile/TISA/backend
实现。`stablehlo_module.json` 和 `manifest.json` 会记录官方 StableHLO 版本、producer、
verifier 状态与 fallback 状态；默认 `official` 不会静默回退。

`--input-shape` 可重复指定多个 tensor 输入。当前 CLI 生成静态浮点输入，适合算子和
小模型编译/性能 smoke；需要 tokenizer、整数输入、KV cache 或自定义样本的模型使用
`compile_torch_module()` Python API 传入真实 example inputs。

该命令会在 `00_frontend` 到 `07_trace` 中生成 Canonical Graph、TileGraph、
TISAProgram、backend payload、周期汇总和泳道图。当前官方路径已支持 rank-2、rank-3、
共享二维 RHS、合法 reducer region、显式 broadcast 和常见 `dot_general`，并能从 primitive
链恢复 Softmax/LayerNorm/RMSNorm，并恢复 torch-xla 的
`flatten -> dot_general -> bias -> unflatten` Linear pattern。对
`batch_norm_training -> affine` 的 LayerNorm 恢复采用严格等价条件：输入必须是直接
row-wise 形态，或由 reshape 严格变换成 `[1, prod(outer), hidden]`；不满足时显式失败。
尚未实现的是完整
torch-xla/XLA legalization、动态 shape、通用 tuple result、复杂 region/layout、完整标准
模型 one-block preset 和 TISA device scheduler。不能把官方 verifier 和这些受约束的
recovery pass 等同于完整 XLA Graph Compiler。

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

命令会按 `00_frontend` 到 `07_trace` 的阶段目录生成编译图和调度结果；各目录的
文件含义见上面的输出树。顶层 `artifact_index.json` 会列出本次实际生成的规范文件，
不需要逐个猜测文件位置。Graphviz 文件位于 `01_graph_ir`、`02_schedule_tile` 和
`04_backend`，时间线位于 `06_simulation`/`07_trace`。顶层同名符号链接仅用于兼容旧
脚本，不代表新的规范布局。

`--arch` 可选 `minimal`、`wide-mxu`、`lpu-like`；`--machine-config path/to/machine.json`
可以直接加载 canonical MachineConfig JSON，用于探索任意自定义 memory/unit/path 参数；
`--policy` 可选 `sequential`、`static_pipeline`、`dynamic_ready_queue`。manifest 中的
`machine_hash` 是实际加载配置的稳定标识。

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

每个 case 目录也采用相同的 `00_frontend` 到 `07_trace` 布局；顶层 `sweep.csv/json`
增加 workload、dynamic priority 字段并计算相对 static 的 speedup。Static baseline 会
对每个 priority 值重复记录，确保比较键完全一致。

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
