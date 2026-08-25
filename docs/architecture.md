# 整体架构

## 总体流程图

```mermaid
flowchart TB
    subgraph Frontend[前端：真实 PyTorch 输入]
        A[PyTorch nn.Module] --> B[torch.export\nExportedProgram]
        B --> C[Torch-XLA\nATen -> StableHLO]
        C --> D[官方 StableHLO\nMLIR parse / verify]
    end

    subgraph Compiler[编译器：论文 GC / FC / TISA Generator]
        D --> E[GC\nGraph Compiler]
        E --> F[Software-scheduled\nSemantic TileGraph]
        F --> G[FC\nTISA Dialect]
        G --> H[TISA Generator\nTISAProgram]
        H --> I[CodegenBackend]
        I --> J[BackendArtifact\nTISA descriptor + payload]
    end

    subgraph Runtime[Runtime：提交与地址绑定]
        J --> K[RuntimeSubmission]
        K --> K1[physical address\ncommand chunk\ndescriptor arrival]
    end

    subgraph Device[Device：TISA 指令调度]
        K1 --> L[reception / WQ / IQ / ROB]
        L --> M{Device policy}
        M --> N[Static\n按 program order issue]
        M --> O[Dynamic\nready queue + OOO issue]
    end

    subgraph Backend[Backend：执行时序与事件]
        N --> P[EventBackend + TimingProvider]
        O --> P
        P --> Q[Execution-unit payload\nDMA / MXU / Vector]
        Q --> R[completion feedback]
        R -. 唤醒后继 TISA .-> L
    end

    P --> S[cycles / stalls / utilization]
    P --> T[swimlane SVG/PNG\nPerfetto JSON]
    J -. 同一份 compiled artifact .-> N
    J -. 同一份 compiled artifact .-> O
```

图中 `Static` 和 `Dynamic` 共享同一份 `BackendArtifact`；`RuntimeSubmission` 的提交策略与 device scheduler policy 是两个独立实验维度。

## 1. 设计约束

当前架构遵守五条约束：

1. 用户输入只能是 PyTorch `nn.Module`；
2. ATen 到 StableHLO 由 Torch-XLA 负责，不在项目内维护逐算子 emitter；
3. Static 和 Dynamic device scheduler 必须消费同一份 compiled artifact；
4. runtime software scheduling 与 TISA hardware scheduling 分层；
5. analytical、trace-calibrated 和未来 RTL/system backend 使用相同接口，但不得混淆精度声明。

因此生产流程只有一条：

```text
PyTorch nn.Module
  -> torch.export.ExportedProgram
  -> Torch-XLA StableHLO
  -> official StableHLO parse/verify
  -> Graph Compiler (GC)
  -> software-scheduled Semantic TileGraph
  -> Fusion Compiler (FC)
  -> TISA dialect
  -> TISA Generator
  -> TISAProgram
  -> BackendArtifact
  -> RuntimeSubmission
  -> TISA device simulation
```

单元测试可以直接构造某一层 IR，以隔离验证该层契约；用户前端不接受手写 graph、Canonical JSON 或 StableHLO 文件。

## 2. 主调用链

CLI 调用关系：

```text
npu_ooo.cli.main
  -> run_compile_model
  -> compile_torch_module
```

`compile_torch_module()` 位于 `src/npu_ooo/compiler/pipeline.py`，按顺序调用 Framework Bridge、GC、FC、TISA Generator 和 Backend。`compile_operator_graph()` 是 StableHLO 已导入后的公共入口，便于隔离测试各编译阶段。

## 3. 前端

### 3.1 PyTorch 捕获

```python
exported_program = torch.export.export(module, args, ...)
```

输出是 `torch.export.ExportedProgram`，包含 FX/ATen graph、graph signature、参数/buffer 描述和 shape constraint。example inputs 决定本次捕获的输入 rank、dtype 和静态 shape。

`TorchExportAdapter.from_exported_program()` 同时生成源图摘要，写入 `00_frontend/source_frontend_import.json`。这份图用于 provenance 和前后端语义对照，不是后续 StableHLO 编译的替代路径。

### 3.2 Torch-XLA legalization

```python
exported_program_to_stablehlo(exported_program, options)
```

Torch-XLA 负责将 ATen 语义转换为 StableHLO。项目保存可读 StableHLO、bytecode 大小/hash 和 Torch-XLA 版本。可读程序位于 `00_frontend/generated.mlir`。

### 3.3 官方 StableHLO 边界

`OfficialStableHLOAdapter` 使用 OpenXLA Python bindings：

```text
register StableHLO dialect
  -> Module.parse(text)
  -> module.operation.verify()
  -> canonical assembly
```

项目不再包含 `stablehlo_codegen.py` 或 `OfficialStableHLOGenerator`。官方 bindings 负责语法和 dialect 验证，项目只维护：

```text
StableHLO semantic family
  -> Canonical OperatorSpec
  -> backend capability key
```

当前 importer 仍有明确限制：官方 MLIR object 先投影到项目支持的可读 operation 子集，再由 semantic importer 建图。未注册 operation 会在 capability boundary 失败，不会静默降级。

## 4. Graph Compiler（GC）

GC 的输入是已经由官方 MLIR bindings 验证的 StableHLO projection；其第一步导入 Canonical OperatorGraph。恢复的信息包括：

```text
TensorSpec: shape、dtype、source kind
OperatorSpec: semantic type、inputs/outputs、iteration/reduction dims
DataEdge: producer、consumer、tensor
StableHLO provenance: source op、operand arity、capability key
```

当前 GC pass 顺序：

```text
CanonicalizeGraphPass
LinearDecompositionPass
RecoverStableHLOLayerNormPass
RecoverStableHLOFlattenedLinearPass
FoldTransposeIntoMatmulPass
LayerNormFusionPass
RMSNormFusionPass
SoftmaxFusionPass
```

Torch-XLA 可能把复合算子展开为 primitive 子图，recovery/fusion pass 依据图结构、shape 和常量恢复 GC 所需的语义边界，不能按模型名匹配。

GC 的最终产物是 `GCArtifact`，由以下内容组成：

```text
OperatorGraph
ScheduleSpec
Semantic TileGraph
fusion / residency / locality metadata
typed tile dependencies
initial software order
per-pass input/output graph snapshots
```

`GCArtifact` 的 Python dataclass 是论文 MLIR GC dialect 的语义代理，不声称复刻论文内部 pass 实现。

每个 GC pass 都保留独立的输入图、输出图和诊断，CLI 将它们写入
`01_gc/pass_dumps/<index>_<pass>.json`。这些 dump 用于解释图恢复和融合
发生在哪一个 pass，不参与后续调度。

### 4.1 Tiling、locality 与依赖

`SchedulePlanner` 当前调用统一的 `plan_uniform_tiles()` baseline：

```text
tile_size(dim) = min(CLI tile_size, resolved extent)
loop_order = iteration dims + reduction dims
stage_id = operator topological order
```

在 `MachineConfig` 可用时，planner 还生成 capacity-aware 的 residency intent：
优先将当前 operator 的输入/输出放入 root 的第一级 local memory，超出容量的
tensor 保留在 root memory。多 tile operator 会附带 `ping_pong` 计划，描述两个
local tile slot 的交替意图；它是 GC metadata，不是 runtime 的实际 buffer 分配。

每个 pass 的图快照和上述 memory intent 都是分析产物。它们不会改变
`TISAProgram` 的指令顺序，也不会替 device scheduler 做 issue 决策。

这是 GC 的初始软件 schedule，不是 static device scheduler。它决定 tile decomposition、初始 loop order 和可记录的 residency；最终 issue 顺序仍由 device scheduler 决定。

`build_tile_graph()` 根据 schedule 枚举 `TileInstance`，并保留 tile id、operator id、coordinates、边界和 semantic metadata。除跨算子 region edge 外，GC 还生成 reduction/state/accumulate 边，使软件 TileGraph 在进入 FC 前已经完整表达 tile 级合法性。

跨算子依赖使用 `logical_tensor_region_v1`：compiler 将 producer/consumer tile 投影到共享 tensor 的逻辑 region，只为重叠 region 建边。Matmul 的 M/N/K、broadcast elementwise、reduce/norm 和 full-tensor transform 都有显式映射；无法证明映射时才对该 operator edge 保守回退 all-to-all。`compile_statistics.json` 会记录回退边数和避免的无效依赖数。

当前 reshape/transpose 采用保守的 full-tensor schedule。reshape 映射为 DMA copy，transpose 映射为 DMA transpose；后续 stride-aware region planner 可以将其细化为 tile transform。

## 5. Fusion Compiler（FC）

FC 接收 `GCArtifact`，不再直接从普通 `OperatorGraph` 重新推导 tile。它将融合区域和 tile stage 专化为 TISA dialect operations，并为每条 scheduler-visible operation 附加：

```text
OpType / semantic operator
Operands: TileShape + TileMem + AccessType
typed Deps and readiness condition
UnitMap
fusion / reorder attributes
backend payload recipe
```

一个 Attention tile 的 FC 输出类似：

```text
tisa.load           <de>
tisa.load_transpose <de>
tisa.matmul         <me>
tisa.softmax        <ve>
tisa.matmul         <me>
tisa.store          <de>
```

每条 TISA operation 只绑定一种主要 EU。Softmax 的 `reduce_max/exp/reduce_sum/normalize` 属于 VE 内部 payload recipe，不再作为全局 scheduler 的独立 TISA instruction。

`TISADialectProgram` 是 FC 的 Python 语义代理，明确标注 `paper_stage=FC` 和 `dialect=tisa`。

## 6. TISA Generator 与 TISAProgram

`TISAGenerator` 将 `TISADialectProgram` 序列化为 `TISAProgram`。它不再做新的 tiling、fusion 或 backend primitive 推导，只保留 FC 已经确定的 descriptor 语义。

当前一个 semantic tile 的 TISA 结构为：

```text
load -> optional load_transpose -> matmul -> optional store
```

复合算子：

```text
load -> softmax -> store
backend payload: reduce_max -> exp -> reduce_sum -> normalize
```

这里的 `softmax` 是 scheduler-visible 的语义指令。当前 analytical backend
仍使用 materialized 的 row-wise `max/sum` lowering；这些 primitive 只在该
payload 内执行，不代表已经实现论文目标的 online Softmax state update。
后续切换到 online lowering 时，保持同一个 TISA 语义边界，只替换 payload
和对应的 GC state dependency。

每条 `TISAInstruction` 包含：

```text
tisa_id / tile_id / operator_id
op_type
TISAOperand(tile shape, TileMem, access type)
UnitMap
typed dependency（RAW/WAR/WAW/STATE/ACCUMULATE）
semantic metadata
payload_ref
```

TISA Generator 不查看 backend `ExecutionTask`，因此 TISA 契约独立于具体硬件 payload。`TISAInstruction` 的依赖 kind 允许 `RAW/WAR/WAW/STATE/ACCUMULATE/BUFFER_REUSE`，scheduler 当前统一按 source readiness 执行。

## 7. BackendArtifact

`CodegenBackend` 接收已经构造好的 TISAProgram，并为每条 TISA instruction 生成 backend-local payload：

```text
BackendArtifact {
  program: TISAProgram
  execution_graph: ExecutionGraph
  payloads: tisa_id -> ExecutionTask ids
}
```

全局 OOO window 中只允许出现 TISA instruction。`ExecutionTask` 是一条 TISA issue 后，在目标 execution unit 内部执行的步骤，不能重新进入全局 scheduler。

默认 `AnalyticalCodegenBackend` 通过 lowering registry 支持：

```text
matmul / batched_matmul / gemv
elementwise / residual_add
reduce
softmax
rmsnorm
layernorm
reshape / transpose
```

新硬件 backend 应实现同一 `CodegenBackend` contract，而不是在 CLI 中增加新分支。

## 8. Runtime

RuntimeSubmission 位于编译 artifact 与 device scheduler 之间，负责：

```text
logical tensor -> physical buffer address
TISA operand -> physical range
command chunk
descriptor available cycle
launch latency
synchronization cost
```

Runtime 的 `dynamic_ready_queue` 表示软件提交顺序可以绕过尚未到达的独立 descriptor。它不等于论文的 device-side OOO：

```text
runtime policy: 描述符何时到达设备
device policy: 已到达描述符何时 issue 到 execution unit
```

两层可通过 `--runtime-device-matrix` 形成四组合实验。

## 9. Device Scheduler

`schedule_tisa_program()` 消费 BackendArtifact、MachineConfig、RuntimeSubmission、SimulatorConfig、TimingProvider 和 EventBackend。

Static 和 Dynamic 共享同一 compiled artifact：

```text
static_pipeline:
  按 program order 和依赖约束 issue

dynamic_ready_queue:
  在 dependency window / ROB / ready queue 内，
  从已到达且依赖满足的 TISA instruction 中选择可 issue 项
```

可配置限制包括 instruction queue depth、ROB entries、dependency window、ready queue depth、max inflight tiles、address scoreboard 和 dynamic priority。

## 10. 可插拔后端

| 接口 | 输入/输出 | 当前实现 |
| --- | --- | --- |
| `CodegenBackend` | TISAProgram -> backend payload | `analytical` |
| `TimingProvider` | ExecutionTask -> duration/II | `analytical`、`timing_table`、`systolic_mxu_profile` |
| `EventBackend` | TISA + payload -> event execution | `analytical_event` |

配置化 `MachineConfig` 描述资源数量、memory、interconnect 和默认 timing。配置变化不应改变 IR schema。

## 11. 输出与可复现性

artifact 按 `00_frontend` 到 `07_trace` 分层。比较调度策略时必须固定：

```text
PyTorch module
example input shape/dtype
Torch-XLA/StableHLO version
tile size
MachineConfig
BackendArtifact
TimingProvider
RuntimeSubmission（除非实验变量就是 runtime）
```

`compile_statistics.json` 记录 per-operator tile/TISA/payload 数量、MAC、root-memory traffic 和依赖统计。`manifest.json` 记录 frontend path、工具版本、machine hash、backend、policy、TISA instruction count、cycle 和 calibration status。

## 12. 当前限制

- StableHLO semantic importer 只覆盖已注册 operation；
- Torch-XLA 复合模式恢复仍需扩大真实模型覆盖；
- tile planner 是统一启发式 baseline，尚无 cost model 或 auto-tuning；
- reshape/transpose 当前是 full-tensor DMA transform，尚未支持 stride-aware tile transform；
- analytical backend 不是 cycle-accurate RTL；
- 当前 MXU VCS 日志只有 descriptor-to-done 区间，不能直接作为 isolated Matmul compute latency；
- 真实 ResNet50、BERT、GPT-J、LLaMA2、DeepSeek block 尚未形成可复现实验集。

下一阶段见 [roadmap.md](roadmap.md)。
