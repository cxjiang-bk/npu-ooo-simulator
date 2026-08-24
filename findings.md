# 研究发现与决策

## 需求

- 从顶层算子建立独立编译与仿真栈；
- 后端架构参数可配置，不能固定为当前 NPU；
- 支持不同调度策略的泳道图和整体执行周期；
- 复现 TISA 论文中的 sequential、static/dynamic dual/triple staged pipeline；
- 旧 `operator-opt` 只作为只读参考。
- 论文 benchmark 需要 Model/Benchmark layer，不能只建立孤立 Operator Graph。

## 已确认事实

- `operator-opt` 已覆盖 Fusion IR、Tiling Tree、Attention/2mm TileFlow mapping 和 aggregate cost，但缺少 per-tile execution lowering 和事件仿真；
- 其 `LpuTaskGraph` 有 resource/duration/predecessor/start/finish，可作为粗粒度参考，不足以表达 queue、II、地址范围和 runtime wake-up；
- TileFlow 的 `Sequential/Pipeline` 主要通过 cycle sum/max 构造 aggregate estimate，不直接产生论文所需的 per-tile issue/completion trace；
- 当前可运行后端有较多 Attention 专用命名和路径规则，因此新项目必须采用 operator lowering registry；
- Static 与 Dynamic 公平比较必须共享 tile graph、buffer、地址、依赖、latency 和 hardware config。
- TISA 原文的 compiler stack 是 `Framework bridge -> Graph compiler -> Fusion compiler -> TISA generator -> backend`，并使用 StableHLO/MLIR 保留 operator semantics；这直接支持在 Operator Graph 上增加 Model/Benchmark IR。
- Table IX 的 benchmark case 具有不同 model family、dtype、batch、sequence/image shape 和 prefill/decode phase；每一行都应是独立 BenchmarkCase。
- 论文 TISA instruction 不是只含 opcode：`OpType + Operands(TileShape/TileMem/AccessType) + Attributes + UnitMap`；这要求 semantic operator context 在 lowering 后仍保留。

## 开源参考定位

| 项目 | 参考内容 | 不直接复用的部分 |
|---|---|---|
| TileFlow/Timeloop | mapping、tiling、memory traffic、aggregate cost | per-tile event simulation |
| TVM-VTA | LOAD/COMPUTE/STORE、dependency token、static pipeline | VTA 固定 ISA/三级 pipeline |
| Gemmini | queues、ROB、access/execute decoupling | RISC-V/RoCC 和 Gemmini 专用实现 |
| SCALE-Sim | MXU/systolic timing、bandwidth/stall | 多 execution-unit OOO scheduler |
| Perfetto | 多 lane event trace | scheduling semantics |

## 技术决策

| 决策 | 理由 |
|---|---|
| canonical MachineConfig 独立于 RTL parser | 支持手写探索 profile、RTL-derived profile 和未来其他 NPU |
| 四层 IR 明确分离 | 防止 schedule factor、runtime tile 和 hardware task 混淆 |
| 独立 ExecutionTask graph | 现有 aggregate task graph 无法承载 tile address/queue/event 语义 |
| 确定性离散事件 simulator | 便于手算验证、回归和可复现实验 |
| latency 与 initiation interval 分离 | 必须表达流水化执行单元的 overlap |
| 2mm 先行，Attention 后接 | 先验证核心机制，再增加 softmax/barrier/cache 生命周期 |
| Model IR 先于 Operator Graph 实例化 | 模型重复 block、运行 phase、KV cache、mask 和 benchmark shape 不属于单个算子 |
| semantic operator 与 primitive task 分离 | Dynamic scheduler 需要知道 `SOFTMAX`/`RMSNORM` 等语义，而 simulator 仍需计时 `reduce_max/exp/reduce_sum` 等 primitive |
| ScheduleSpec 明确保存 tile factor、loop order、residency 和 stage | 让 mapping 结果可序列化，并与实际边界 tile、runtime task 解耦 |
| TileInstance 的 coordinates 保存 tile index，bounds 保存实际 `[start, stop)` | 边界 tile 不会被错误地当成满 tile，后续地址/traffic 统计可直接复用 |
| ExecutionTask 显式携带 BufferRegion 和 predecessor | 统一承载 TISA operand 的 TileShape/TileMem/AccessType 语义，scheduler 不需要猜 tensor 地址或依赖 |
| Matmul lowering 对每个 K tile 建立累加链，最终 tile 才生成 store | 保留 partial-sum 生命周期，同时让跨算子 producer store -> consumer load 依赖可观察 |
| Policy 只改变 ready-task 选择；task graph 与 MachineConfig 作为共享输入 | 保证 Static/Dynamic 周期差异归因于调度策略，而不是重新切 tile 或更换 timing model |
| scheduler policy 与 event backend 分离 | policy 只选择 ready task；event backend 统一处理 issue/start/complete、queue、ROB、II、in-flight tile 和 completion wake-up |
| `SimulatorConfig` 覆盖 MachineConfig runtime capacity | 可以对 dependency window、ROB、instruction queue、ready queue 和 tile window 做实验 sweep |
| address scoreboard 作为可选 runtime layer | 基于 active `BufferRegion` 生成 RAW/WAR/WAW issue stall，COMPLETE 后释放范围；不改写默认 graph，方便和 compile-time dependency 做公平对照 |
| Elementwise/residual-add 先于 reduce/softmax 接入 | 它能验证多输入同形 tile、ARU primitive 和 producer-consumer store/load 依赖，同时不引入 reduction barrier 与指数近似等额外语义 |
| Dynamic priority 必须作为独立实验维度 | Softmax 的 ARU/DMA 竞争反例中，`window=8/ROB=8` 下 critical-path heuristic 为 4808 cycles，oldest-first 为 3784 cycles；动态机制本身不保证某个启发式总是占优 |
| Tile size 属于 mapping 实验维度，不属于 scheduler policy | 2mm `tile_size=16` 与 `32` 产生不同 tile/task graph 和周期，但每个 tile size 内 Static/Dynamic 仍共享完全相同的 lowered graph；sweep manifest 必须把 tile size 单独记录 |
| RMSNorm 可先建模为 sum-square barrier | `square -> reduce_sum_square -> rmsnorm` 保留跨 reduction tile 的生命周期和完成依赖；epsilon/scale 的数值语义留在 operator attributes，当前 analytical timing 不宣称数值精确 |
| 混合图使用 lowering registry 而不是 scheduler 分支 | registry 按 semantic operator type 选择插件；dispatcher 只负责拓扑拼接、全局 program order 和显式 DataEdge 的 root-memory region handoff，因而 Static/Dynamic 仍消费同一 ExecutionGraph |
| 混合图首个 decoder fragment | `RMSNorm -> Matmul -> ResidualAdd` 覆盖 decoder block 常见 pre-norm/projection/residual 数据流；当前以 shape-only 权重和 conservative GM store/load handoff 表达，不等同完整 GPT-J/LLaMA attention block |
| PNG 泳道导出 | SVG 作为 canonical trace visualization，PNG 由可替换的 ImageMagick/librsvg 外部 rasterizer 生成；缺少转换器时应报告环境缺失，不把 SVG 冒充 PNG |
| LayerNorm barrier 建模 | 每个 row 先串行累加 `reduce_sum`，再发射单个 `layernorm_mean`；之后按 tile 做 `center` 和串行 `reduce_sum_square`，最终 `layernorm` 等待完整 variance barrier。该 DAG 比 RMSNorm 多一个全行统计阶段，适合观察 window/priority 对 barrier 的影响 |
| LayerNorm 动态反例 | 默认 `128x96`、minimal、同一 graph/machine 下，static pipeline 为 3808 cycles，dynamic `critical_path` 为 4696 cycles；动态 priority 不能被解释成总是优于 static，必须同时 sweep priority、window、ROB 并查看 stall/occupancy trace |
| 通用 workload sweep | `sweep-workloads` 对每个 workload/architecture/tile-size 缓存同一份 lowering，再在 policy/window/ROB 维度重放；每个 case 保留 semantic graph、execution graph 和 SVG/PNG/Perfetto，避免只比较汇总数字而看不到图结构 |
| 外部 MachineConfig | canonical `MachineConfig.to_dict()` 已支持 round-trip 和 CLI `--machine-config`；自定义 memory hierarchy、execution unit、transfer path 可以不改 simulator 代码直接进入实验，但仍需通过 schema validation |
| Custom profile label | `sweep-two-mm`/`sweep-workloads` 在提供 `--machine-config` 时允许任意 architecture label；label 只用于实验索引，真实配置由 JSON 和 `machine_hash` 唯一确定 |
| External timing table | `TimingTableModel` 支持 task id、resource:primitive、primitive、resource 和 default 五级匹配，未覆盖 task 回退 analytical；这提供了 SCALE-Sim/RTL 校准的最小可插拔接口 |
| External MXU profile | `SystolicMXUProfileTimingProvider` 已能读取 versioned JSON，按 `(batch,m,n,k)` 精确匹配 Matmul tile duration/II，并把非 MXU 或未命中 tile 的 fallback policy、source 和 calibration status 写入 manifest；它重放离线 profile，尚未直接调用 SCALE-Sim |
| Timing command error | 用户直接复制 `path/to/timing.json` 会触发文件不存在；已提供 `configs/timing/attention_probe.json` smoke table，并让 CLI 以简洁 `error:` 返回而不是 traceback |
| Priority sweep 反例 | `sweep-workloads --workloads layernorm --windows 8 --robs 8` 显示 static=3808；dynamic `oldest_first`=3808，而 dynamic `critical_path`=4696（speedup 0.811）。因此 priority 必须成为 manifest/sweep 的显式键 |
| Attention 首个闭环 | 单头无 mask/cache 的 `Q @ K^T -> Softmax -> P @ V` 由两个 Matmul 和一个 Softmax semantic op 组成，默认 `64x64x32` 生成 12 tiles、54 primitive tasks、8 个跨算子 handoff；minimal analytical profile 下 static=4520、dynamic critical-path=4532 |
| Transformer block skeleton | `LayerNorm -> QK^T -> Softmax -> PV -> residual -> MLP1 -> activation -> MLP2 -> residual` 默认生成 9 semantic operators、30 tiles、126 tasks、28 个 root-memory handoff；minimal analytical profile 下 static/dynamic critical-path 均为 10540 cycles，但这只是 shape-only skeleton |
| Model presets | `model-block` 已将 BERT、GPT-J、LLaMA2-7B 和 DeepSeek-R1-16B 暴露为模型层 preset；默认使用小型 proxy shape，native hidden/head/intermediate metadata 和 assumptions 会写入 ModelSpec/BenchmarkCase。DeepSeek preset 明确不对 dense/MoE 做事实判断，当前仅作为 dense shape-only proxy |
| Model sweep registration | 这些 preset 复用 `sweep-workloads` 的 lowering cache 和 Static baseline 配对逻辑，因此模型维度不会引入新的 scheduler 分支；`workload` 字段保留 preset 名称，case manifest 仍记录真实 model_id 和 proxy metadata |
| Model proxy sweep sizing | 四个 preset 使用同一 `16x16x16x32` proxy shape 时生成 16 个配对 case，Static/Dynamic 均为 1300 cycles；这只证明 graph/machine/policy 的公平配对。按 preset 区分小规模 proxy shape（BERT 16/16/16/32、GPT-J 16/16/24/48、LLaMA2 16/16/32/64、DeepSeek 16/16/40/80）后，minimal analytical cycles 分别为 1372、1900、2528、3392，仍属于 shape-only 趋势探针 |

## 视觉发现

论文示意图的五条时间线可以统一为 iteration-specific stage DAG：

```text
Sequential
Static dual-stage
Dynamic dual-stage
Static triple-stage
Dynamic triple-stage
```

彩色块表示不同 iteration 的 stage task；虚线表示静态 stage/iteration 边界；尾部阴影 `E*` 表示 pipeline drain 或依赖/资源造成的结束差异。新 simulator 必须显式输出 task start/end 和 drain cycles，不能只返回一个总 Cycle。

## 待验证

- TISA 原文对 tile dependency table、地址范围、窗口和完成事件的精确定义；
- 当前 NPU ISA 的 issue/completion、SET/WAIT/FENCE 和 buffer address 语义；
- 哪些 latency 可从 RTL source 提取，哪些必须通过 waveform/hardware counter 校准；
- TileFlow mapping 到新 Schedule IR 的完整信息保真度。
- Model import 的 StableHLO/ONNX/PyTorch adapter 最小公共字段；
- DeepSeek-R1-16B benchmark 的实际 dense/MoE 配置与 KV-cache/attention 细节；
- ResNet50 inference 中 BatchNorm 是否已 fold 到 Conv；

## 资源

- 论文：`/home/lora/OpenTPU/ooo_research/Song 等 - Dynamic scheduling for AI accelerators via TISA.pdf`
- 参考仓库：`/home/lora/OpenTPU/operator-opt`，仅只读使用；
- 新项目：`https://github.com/cxjiang-bk/npu-ooo-simulator`。

## 2026-08-22：Artifact 输出布局

- 之前每个 benchmark 输出目录把 model、graph、schedule、backend、simulation 和 trace 文件全部平铺，人工检查时难以对应编译链路。
- 现在规范布局使用八个编号目录：`00_frontend`、`01_graph_ir`、`02_schedule_tile`、`03_tisa`、`04_backend`、`05_runtime`、`06_simulation`、`07_trace`。
- 顶层保留 `manifest.json`、`summary.json`、`artifact_index.json` 和 `README.md`；旧的平级 artifact 文件名通过相对符号链接保留，确保已有测试、脚本和用户命令不需要立刻改写。
- `artifact_index.json` 是机器可读的阶段索引，输出目录中的 `README.md` 给出本次运行的中文查看顺序；sweep 的每个 case 复用相同布局，顶层 `sweep.csv/json` 仍用于汇总。

## 2026-08-21：Frontend、Runtime 与热插拔 Backend 决策

### 架构决策

- ExecuTorch/`torch.export()` 作为第一模型前端：先获取规范化 Core ATen graph，再转换为项目自己的 Canonical OperatorGraph；ONNX、StableHLO 和 Torch-MLIR 作为后续 adapter，不直接改变下游 IR。
- Compiler 与 runtime 分离：Compiler 生成 `CompiledProgram`/TISA Command 模板、逻辑 region、依赖、UnitMap 和地址表达式；Runtime 负责 shape/state binding、buffer allocation、physical address binding、command-buffer chunk 和提交事件。
- Runtime 与 device scheduler 分离：Runtime 动态决定任务何时进入设备；TISA device backend 在已提交窗口内决定每个 cycle 发射哪条 task。两者分别建模并分别统计收益。
- 当前 `ExecutionGraph -> SchedulerPolicy -> DiscreteEventBackend` 保留为默认 device backend；不能因为引入 ExecuTorch 或外部 simulator 而改变 Static/Dynamic 的共同输入。
- Backend 采用热插拔分层：`TimingProvider`、`EventBackend`、可选 `SystemBackend`。backend selection、timing source 和 calibration status 必须进入 manifest。
- Static/Dynamic 实验扩展为四种组合：`static runtime + static/dynamic device` 与 `dynamic runtime + static/dynamic device`，以区分软件提交收益和硬件 issue 收益。

### 开源 backend 的组合定位

| 项目 | 进入项目的层 | 结论 |
|---|---|---|
| SCALE-Sim | TimingProvider | 校准 MXU/systolic duration、II、带宽 stall，不替代 TISA scheduler |
| Timeloop/Accelergy | Mapping/traffic/energy | 生成 mapping 和 aggregate 参考，不生成 per-tile event trace |
| Ramulator2/DRAMSys | Memory timing provider | 提供 DRAM request completion，不接管 NPU task dependency |
| Gemmini/VTA | Hardware/ISA reference | 参考 queue、DMA、scratchpad、静态 pipeline 和 RTL timing，不直接作为通用 TISA backend |
| gem5/gem5-SALAM | Optional SystemBackend | 未来研究 CPU+NPU、runtime、DMA、内存和同步；需要自行实现 NPU device model |
| TileRT | Runtime reference | 借鉴 tile task、event、compute/I/O overlap，不作为 cycle-level NPU simulator |

没有一个项目同时提供可配置 NPU、通用 tile OOO、runtime、memory 和论文泳道图。因此当前项目保留 TISA device semantics，通过插件接入外部局部模型。

### 新的 IR/运行时边界

```text
Canonical OperatorGraph
    -> Schedule/Tiling IR
    -> TileGraph
    -> CompiledProgram / TISA Command
    -> RuntimeSubmission
    -> Device Backend / ExecutionGraph
```

`CompiledProgram` 不绑定物理地址和 issue policy；`RuntimeSubmission` 绑定实际 shape、persistent state、buffer base 和提交顺序；Device Backend 只消费提交结果，不回写 OperatorGraph/ScheduleSpec。

### 仍待验证

- ExecuTorch 当前版本导出图中需要支持的 Core ATen operator set 和动态 shape constraint 表达；
- `CompiledProgram` 地址表达式与真实 NPU ISA descriptor 的对应关系；
- runtime allocation/command queue 的开销参数来源；
- backend capability negotiation：TimingProvider 不足以表达 bank/port/conflict 时如何升级为 EventBackend；
- SCALE-Sim、Gemmini/Verilator 和 RTL trace 的 timing 对账粒度。

## 2026-08-21：重新核对 TISA 原文后的修正

### 论文原文确认

- 论文明确把 TISA 定义为 `TISA_Inst = (OpType, Operands, Attributes, UnitMap)`，Operand 为 `(TileShape, TileMem, AccessType)`，并另外定义 typed dependencies `Deps = (src, type, condition)`；`type` 包括 RAW/WAR/WAW。
- TISA 的粒度是 tile-level semantic instruction：比 kernel stream 更细，比 raw per-unit ISA instruction 更粗。论文明确写道 compiler 在 tile granularity 截止 lowering，不需要把每个 tile 继续展开成细粒度 ISA 指令。
- TISA 不是普通 compiler-only IR。论文称其为 hardware-consumed scheduling-semantics layer，并说明 Epoch 上存在 concrete binary encoding；它补充而不是替代 MXU、Vector、DMA 等 per-unit execution ISA。
- 论文中的 `tisa::load`、`tisa::load_transpose`、`tisa::gemm`、`tisa::softmax` 说明一个 TISA instruction 可以是一个语义明确的 tile operation；一个完整 tile 可能在 backend 内部进一步产生 DMA/MXU/ARU micro-events。
- 论文的 dynamic scheduler 逻辑上被称为 runtime scheduler，但实现目标是 AI-core 内的硬件 scheduler：论文给出 7--9 cycle dispatch budget、每个 unit 的 WQ/IQ/Fu/in-flight table，并报告 RTL synthesis。论文同时指出控制处理器上的 software runtime 会有 microsecond 级开销，无法承担 tile-level dispatch。
- 编译链是 `torchxla -> XLA/StableHLO -> MLIR Graph Compiler -> MLIR Fusion Compiler/custom TISA dialect -> TISA generator -> LLVM/backend-specific lowering`。TISA-CPU backend 用于功能验证，TISA-NPU backend 将 metadata 置入最终 binary 并由硬件 scheduler 消费。
- 论文选择 torchxla -> XLA/StableHLO 不只是为了导出 graph，也是为了让 `OpType` 对齐稳定的高层 semantic taxonomy。ExecuTorch 作为第一入口时必须保留 source module/composite provenance，不能只输出打散后的 Core ATen primitive。

### 对当前设计的影响

当前设计方向没有推翻，但存在一个必须修正的 IR gap：

```text
当前：TileInstance -> operator-specific lowering -> ExecutionTask -> scheduler
目标：TileInstance -> TISAInstruction -> device scheduler -> backend primitive ExecutionTask
```

具体问题：

1. `TileInstance` 目前只有 `operator_id`、coordinates、bounds、stage 和松散 attributes，缺少 `OpType`、Operand、TileMem scope、AccessType、UnitMap、typed Deps 和 partial-ready condition；它只能作为几何 tile，而不是 TISA scheduler 输入。
2. `ExecutionTask` 当前已经是 `load/matmul/store/reduce` primitive。若直接在这一层做 dynamic issue，会比论文的 tile-level scheduler 更细，并且在 primitive lowering 时丢失 operator semantics、resource affinity 和 typed dependency。
3. `BufferRegion` 已接近 TileMem，但仍缺少结构化 `base/scope`、symbolic address expression、operand grouping 和 partial region readiness；应作为 TISA Operand 的底层实现，而不是让 scheduler 从 tensor 名称猜地址。
4. `ExecutionTask.resource` 只有一个资源字符串，不能表达 `UnitMap=(unit, quantity, affinity)`；需要在 TISA 层表达合法 unit class、数量和 affinity，backend 再选择具体 instance。
5. `TileDependency`/`ExecutionTask.predecessors` 当前主要是 untyped string edges；需要增加 `RAW/WAR/WAW + condition`，并区分 compile-time dependency、TISA typed dependency 和 device-observed address hazard。
6. 当前 `TimingModel` 直接给 primitive task duration；TISA 对齐后需要额外建模 TISA dispatch/decode/scheduler overhead，以及 TISA instruction 到 backend primitive expansion 的边界。

### 不属于问题的部分

- 继续保留 `Model IR -> OperatorGraph -> ScheduleSpec` 是正确的；它对应论文 framework bridge、graph compiler 和 software-scheduled tile graph 的前半段。
- `LoweringRegistry` 仍然有价值，但它应先生成 TISA semantic instruction，再由 backend lowering registry 生成 `ExecutionTask`；不是简单删除所有算子专用 lowering。
- 当前 analytical event simulator、Static/Dynamic policy、MachineConfig 和 trace schema 可以继续复用，作为 TISA device backend 的 baseline implementation。
- ExecuTorch 仍然适合作为第一 frontend；它替代的是论文的 torchxla/StableHLO framework bridge，不替代 TISA dialect 或 device scheduler。
- 当前环境未安装 `torch`/`executorch`，因此真实 API 兼容性尚未验证；实现前需要在项目环境中固定版本并建立最小 export smoke test。

### 修正后的对齐架构

```text
ExecuTorch / StableHLO
        -> Canonical OperatorGraph
        -> Schedule/Tiling + fusion
        -> TileInstance (bounds/provenance)
        -> TISAProgram (semantic tile instructions)
        -> Runtime descriptor emission / binding
        -> Hardware-like TISA scheduler (WQ/IQ/Fu)
        -> Backend primitive expansion and timing
        -> Execution trace
```

## 2026-08-22：StableHLO adapter 现状

- 本节是 2026-08-22 的历史状态：当时 Python 3.12 有 PyTorch 2.9.1，但没有 StableHLO/MLIR bindings；官方 bindings 已在次日验证接入，`torch_xla` 仍未安装。
- 已在 frontend boundary 实现 `StableHLOAdapter`：优先接受 module assembly，另提供 textual MLIR/file/payload 入口；其输出直接是 `FrontendImport`，下游不感知 StableHLO node name。
- 第一版 textual subset 已覆盖 `func.func` 参数、tensor type、constant、单结果 elementwise/reduce/dot/softmax 和 return；StableHLO RMSNorm 算术链可复用同一 RMSNorm fusion、SchedulePlanner、TISA 和 backend。
- 该 adapter 是论文对齐的输入路径，不等价于完整 StableHLO/MLIR compiler。后续仍需真实 StableHLO object smoke、tuple result/native LayerNorm、layout 和动态 shape constraint 支持。

## 2026-08-22：真实 ATen 图与最小 Graph Compiler 规则

- PyTorch 2.9.1 的 `nn.Linear` 保留为单个 `aten.linear(x, weight[N,K], bias[N])`，不会自动拆成 mm/add；要进入通用 lowering，compiler 必须显式表达 RHS transpose 与 bias broadcast。
- rank-3 Linear 的 activation 带 batch/sequence 维，weight 仍是共享的二维 `[N,K]`；它不能套用“两侧 batch shape 必须一致”的普通 batched Matmul 校验，TISA/backend operand 必须显式标记 RHS batch broadcast。
- `nn.LayerNorm` 导出为单结果 `aten.layer_norm`，affine 参数作为 placeholder 输入；适合保留为一条 semantic LayerNorm，再在 backend payload 内展开 mean/variance/affine micro-steps。
- attention micrograph 导出为 `aten.transpose(K) -> aten.matmul(rank3) -> aten.softmax -> aten.matmul(rank3)`；单用途末两维 RHS transpose 可安全 fold 为 `load_transpose + batched_matmul`，无需物化独立 transpose tensor。
- rank-3 Matmul 的 batch 维必须进入 tile coordinates 和 BufferRegion；仅使用末两维 M/N 会产生能通过 shape-level验证但地址错误的 graph。
- Softmax 的 axis 常作为 positional argument；只读取 kwargs 会把 `dim=1` 错当成最后一维。region 构造也必须恢复 tensor 物理轴顺序，不能简单拼接 iteration dims 后再拼 reduction dim。
- 这些规则说明 Canonical OperatorGraph 之前必须有 framework-specific import，之后仍需要 graph compiler passes；`torch.export` 或 StableHLO 本身都不会直接生成可供 TISA scheduler 使用的 tile program。

## 2026-08-23：StableHLO round-trip 实现结论

- 论文形态的 `TorchExport -> StableHLO -> TISA` 可以先用 dependency-light textual emitter/parser 验证边界，不必在第一步把完整 OpenXLA/MLIR toolchain 嵌入 simulator。
- StableHLO 的 `dot_general` 维度必须转换为项目的固定矩阵语义：batch dimensions + M + N，K 为唯一 reduction；共享二维权重需要显式 `rhs_broadcast_batch`，否则 lowering 会错误地把权重当成逐 batch tensor。
- StableHLO primitive 链不是最终 semantic graph。Softmax 的 max/subtract/exp/sum/divide、LayerNorm 的 mean/variance chain 和 RMSNorm 的 square/sum/rsqrt chain必须经过严格 producer/consumer pattern fusion，才能复用已有 semantic lowering。
- round-trip 的 provenance、op id 和 frontend target 与 direct TorchExport 路径不同是正常现象；公平性应比较 semantic type/shape、tile count、TISA count、primitive task count 和 cycle，而不是要求 JSON hash 完全相同。
- 输出中应同时保留原始 `FrontendImport` 与 generated StableHLO module，避免把中间模块塞进 provenance 导致 graph JSON 膨胀，也便于定位 emitter/parser 的差异。
- textual subset 当前不应宣称“使用了完整 StableHLO compiler”：tuple result、复杂 region、layout、动态 shape constraint 和真实 XLA legalization 仍需后续接入标准 toolchain。

## 2026-08-23：官方 StableHLO 接入后的结论

- OpenXLA 官方 StableHLO wheel 的顶层 import 不是 `import stablehlo`，而是
  `from mlir.ir import Context, Module` 与 `import mlir.dialects.stablehlo`；解析前必须
  `stablehlo.register_dialect(context)`，验证使用 `module.operation.verify()`。
- “使用官方 StableHLO”至少包含 dialect registration、官方 parser/printer 和 verifier；
  仅输出带 `stablehlo.` 前缀的文本不足以证明 IR 合法。旧文本第一次真实验证即暴露出
  reduce reducer region、transpose syntax 和隐式 broadcast 问题。
- StableHLO reduce 会移除 reduction axes，PyTorch `keepdim=True` 必须由后续
  `broadcast_in_dim`/reshape 表达。Graph importer 可以在 fusion 识别阶段折叠这些 shape
  primitives，但不能在官方 IR 中伪造 keepdim reduce result type。
- 官方 wheel 负责 IR 基础设施，不负责 `torch.export -> StableHLO`。因此当前架构应明确
  分成 `PyTorch exporter/legalizer` 与 `OfficialStableHLO backend`；未来切换 torch-xla 或
  torch-mlir exporter 不应影响 OfficialStableHLOAdapter 后面的 Canonical Graph/TISA。
- 正式实验不能 silent fallback。`official` 缺依赖/校验失败必须报错；只有显式 `auto`
  才允许回退，并将 `stablehlo_fallback=true` 和原因写入 manifest。

## 2026-08-23：torch-xla exporter 实测

- `torch-xla==2.9.0` 的稳定 API 是
  `torch_xla.stablehlo.exported_program_to_stablehlo(ExportedProgram)`；返回对象同时提供
  `get_stablehlo_text()` 与 `get_stablehlo_bytecode()`。这与论文的 framework bridge
  形态直接对齐，不需要从 FX graph 手写 StableHLO 文本。
- 本机 `torch==2.9.1+cu128` 与 torch-xla 2.9.0 的 PJRT CPU backend 实测可用，尽管 patch
  版本并非完全一致；该组合必须通过 regression 固定，不能泛化为任意 PyTorch 版本兼容。
- torch-xla 导出的 Matmul 和 attention micrograph 使用官方 `dot_general`、reduce region、
  broadcast、transpose，可直接通过独立 OpenXLA StableHLO verifier 和当前 importer。
- torch-xla 会改变函数参数顺序并内联/重排常量，因此公平对比不能依赖 SSA 名称或参数
  顺序；应继续比较 semantic graph、tile、TISA、primitive task 和 cycle。
- 完整 attention block 暴露出下一层真实 compiler 工作：Linear 被 flatten 为 rank-2 dot
  再 reshape 回 rank-3，LayerNorm 变为多结果 `batch_norm_training` 加 affine chain。
  这证明“有官方 exporter”仍不等于“已经有完整 Graph Compiler”，下一步应实现 pattern
  recovery，而不是绕过 StableHLO 使用源 TorchExport graph。
- pattern recovery 实测可在不读取源 FX graph 的前提下恢复完整 attention block：四个
  flatten/dot/bias/unflatten Linear 恢复为 rank-3 BatchedMatmul 加 broadcast Add；未使用的
  `batch_norm_training` mean/variance 结果可以投影，但一旦被消费必须显式拒绝。
- torch-xla 2.9 对输入 `[1, 4, 8]` 的 LayerNorm 直接使用 feature index 1 的 batch norm；
  对 `[2, 4, 8]` 则先 reshape 为 `[1, 8, 8]`，再以 feature index 1 让每个 outer row
  独立沿 hidden 轴归一化。因此 recovery 必须同时验证 reshape 元素数、hidden 维、
  feature index、单用途链和 reshape-back shape，不能只按 op 名称融合。

## 2026-08-24：TISA scheduler 边界验证

- primitive DAG 合法不代表任意 coarse grouping 后的 TISA DAG 仍合法。同一 Softmax tile
  中把 `reduce_max/exp/reduce_sum/normalize` 仅按 Vector resource 合并，会让跨 reduction
  tile barrier 在 group graph 上同时产生双向依赖。第一版必须按 `resource + primitive`
  建立 TISA instruction；未来只有在证明 group contraction 不产生环时才能做 payload fusion。
- Static/Dynamic 公平对比现在可以共享完全相同的 `BackendArtifact`：scheduler 只选择
  `TISAInstruction`，每条指令 issue 后才在绑定 EU 上按本地拓扑顺序运行 payload，primitive
  不进入全局 ready window。
- 双层 trace 必须同时保留 instruction 和 payload：TISA lane 用于观察 descriptor issue、
  dependency/window/ROB 行为，primitive lane 用于解释 instruction duration 和具体 EU 占用；
  只画其中一层都无法区分 scheduler 决策与 backend timing。

## 2026-08-24：当前 TISA simulator 策略语义

- 阶段 9.9 已将 compiler 顺序改为 `TileGraph -> TISAProgram -> backend capability/payload lowering -> BackendArtifact`。`ExecutionGraph` 只由 analytical payload backend 生成，不能再反向决定 TISA 分组；backend owner edge 只需在 TISA dependency DAG 中可达即可。
- `simulate_tisa_artifact(policy="static_pipeline")` 的实际语义是 TISA program-order issue：可见窗口只保留最老的一条，遇到依赖、资源、tile window 或地址冲突时不绕过；它不使用 `StaticPipelineConfig`，该配置在 TISA backend 中会被拒绝。它因此是静态/in-order TISA baseline，不是论文完整的 static dual/triple stage reservation。
- `dynamic_ready_queue` 在 `dependency_window` 和 `ready_queue_depth` 截断后的 waiting window 中收集候选；`critical_path` 选择剩余 payload critical path 最大者，`oldest_first` 按 TISA program order 选择。它可以绕过窗口内阻塞的早期指令，并在不同 EU 空闲时同一 cycle 发射多条 TISA instruction。
- 当前 TISA payload 先按 primitive DAG 做本地拓扑排序，再将 duration 相加；父指令 issue 后 payload 在一个 EU instance 上 run-to-completion。TISA simulator 尚未使用 `issue_width`、`initiation_interval_cycles`、`pipeline_depth`，也没有真实 dispatch latency、partial-ready condition、memory bank/port 或 runtime command submission。
