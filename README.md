# NPU OOO Simulator

## 整体流程

```mermaid
flowchart LR
    A[PyTorch nn.Module] --> B[PyTorch 前端\ntorch.export + Torch-XLA + StableHLO]
    B --> C[GC\nStableHLO -> Semantic TileGraph]
    C --> C2[FC\nTileGraph -> TISA Dialect]
    C2 --> C3[TISA Generator\nDialect -> TISAProgram]
    C3 --> D[BackendArtifact]
    D --> E[Runtime\n地址绑定与提交]
    E --> F[Device scheduler\nStatic / Dynamic OOO]
    F --> G[Backend 仿真\n周期与执行事件]
    G --> H[周期、stall、泳道图、Perfetto]
```

这是一个用于研究 TISA 风格 NPU 动态调度的编译与仿真框架。项目当前只有一条生产前端路线：

```text
PyTorch nn.Module
  -> torch.export.ExportedProgram
  -> Torch-XLA
  -> 官方 StableHLO
  -> Graph Compiler (GC)
  -> software-scheduled Semantic TileGraph
  -> Fusion Compiler (FC)
  -> TISA Dialect
  -> TISA Generator
  -> TISAProgram
  -> BackendArtifact
  -> RuntimeSubmission
  -> static / dynamic TISA scheduler
  -> 周期结果、泳道图和 Perfetto trace
```

新增 workload 时只需要提供真实的 PyTorch `nn.Module`，不需要为算子手写 graph builder 或专用 lowering 入口。

## 当前能力

- 使用 `torch.export` 捕获 PyTorch 模型；
- 使用 Torch-XLA 完成 ATen 到 StableHLO 的 legalization；
- 使用 OpenXLA 官方 StableHLO bindings 执行 MLIR parse 和 verify；
- 将支持的 StableHLO semantic family 导入统一 OperatorGraph；
- 以论文 GC/FC 为边界执行图 canonicalization、复合算子恢复、切 tile、依赖构造和 TISA dialect 生成；
- 保存每个 GC pass 的输入/输出图，生成可解释的 residency 与 ping-pong intent；
- 支持多头 Attention 所需的 reshape、permute、scale、additive mask 和 batched Matmul；
- 使用逻辑 tensor region 生成跨算子 tile dependency，并输出 MAC/traffic/依赖统计；
- Static 和 Dynamic device policy 共享同一份 `TISAProgram/BackendArtifact`；
- runtime 单独负责物理地址、command chunk、descriptor availability 和提交开销；
- 固定窗口 KV-cache 支持 `RuntimeStateRegistry` 和多步 `RuntimeSequence`，可核对
  跨 invocation 的 state-complete 依赖与稳定物理地址；
- backend codegen、event engine、timing provider 均可替换；
- 输出 TISA/backend payload timing、SVG/PNG 泳道图和 Perfetto JSON。

当前默认时序仍是 analytical。除非显式加载来自 RTL 的校准 profile，否则结果不能称为 RTL cycle-accurate。

## 环境

正式前端需要同一个 Python 环境同时提供：

```text
Python 3.12
torch 2.9.1
torch-xla 2.9.0
OpenXLA StableHLO wheel
```

本机已验证的解释器是 `/usr/bin/python3.12`。安装细节见 [docs/install-stablehlo.md](docs/install-stablehlo.md)。

## 快速运行

编译并动态调度一个真实的两头 PyTorch Attention block：

```bash
cd /home/lora/OpenTPU/npu-ooo-simulator

PYTHONPATH=src /usr/bin/python3.12 -m npu_ooo.cli compile-model \
  --torch-module examples.torch_models:MultiHeadAttentionBlock \
  --input-shape 1,4,8 \
  --input-shape 1,1,4,4 \
  --tile-size 4 \
  --policy dynamic_ready_queue \
  --output-dir out/attention-dynamic
```

比较 static 和 dynamic 时，只改变 `--policy` 和输出目录：

```bash
PYTHONPATH=src /usr/bin/python3.12 -m npu_ooo.cli compile-model \
  --torch-module examples.torch_models:MultiHeadAttentionBlock \
  --input-shape 1,4,8 --input-shape 1,1,4,4 \
  --tile-size 4 \
  --policy static_pipeline \
  --output-dir out/attention-static
```

两次编译使用相同 shape、tile planner 和 backend。实验脚本若要求严格共享同一个已编译 artifact，可使用：

```text
--runtime-device-matrix
```

它在一次编译后运行 runtime static/dynamic 与 device static/dynamic 的四种组合。

当前也提供一个包含两次 RMSNorm、masked multi-head attention、residual 和 SwiGLU
MLP 的真实 PyTorch pre-norm decoder block：

```bash
PYTHONPATH=src /usr/bin/python3.12 -m npu_ooo.cli compile-model \
  --torch-module examples.torch_models:PreNormDecoderBlock \
  --input-shape 1,4,8 --input-shape 1,1,4,4 \
  --tile-size 4 \
  --policy dynamic_ready_queue \
  --output-dir out/decoder-dynamic
```

该通用示例暂不包含 RoPE 和 KV-cache；它们需要额外的 state/stride 语义，不应被普通
decoder block 的结果隐式代替。论文 benchmark 中的 `llama2-13b-oneblk` 另有显式
`rope_cos`/`rope_sin` 输入，当前已能恢复为非 opaque `rotary_embedding` region，并
将 Q/K 旋转路径的底层成员继续暴露给 TISA。KV-cache 当前支持固定 shape、unit-stride
的 `slice(cache) + concatenate(update)` 滑动窗口 contract，并可用
`RuntimeSequence` 仿真多步 decode；动态索引写入、真实 cache layout、跨请求 state
生命周期和完整 decode loop 仍未实现。

论文 benchmark 目录提供 `examples.paper_benchmarks.llama2:LLaMA2DecodeOneBlock` 作为
scaled one-token decode 输入。它用于验证真实 PyTorch -> Torch-XLA -> StableHLO -> TISA
和多步 state contract，不代表论文完整 LLaMA2-13B 的 hidden/head/cache 尺寸。

CLI 中可用 `--runtime-invocations N` 重复提交同一个 compiled artifact；配合
`--runtime-inter-invocation-gap C` 可显式加入 state 完成后的等待周期。多步结果额外
写入 `05_runtime/runtime_sequence.json`，汇总周期和 trace 仍写入 `06_simulation/`、
`07_trace/`。

## 添加 PyTorch 算子或模型

在任意可导入 Python 模块中定义真实的 `torch.nn.Module`。无参数构造的 module class 可以直接作为 CLI 入口：

```python
import torch

class MyOperator(torch.nn.Module):
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)
```

然后运行：

```bash
PYTHONPATH=src:/path/to/module /usr/bin/python3.12 -m npu_ooo.cli compile-model \
  --torch-module my_module:MyOperator \
  --input-shape 32,64 \
  --input-shape 64,128 \
  --output-dir out/my-operator
```

每个 `--input-shape` 对应一个 positional tensor input。CLI 会生成确定性的随机 example input；它们用于 `torch.export` 的 shape/dtype 捕获，不用于数值正确性评估。

若 Torch-XLA 产生了项目尚未注册的 StableHLO operation，编译会在 StableHLO semantic capability boundary 明确失败。扩展新算子时，应补充：

```text
StableHLO semantic capability
  -> Canonical op mapping 或 composite recovery pass
  -> TISA stage definition
  -> backend lowering capability
  -> 端到端 PyTorch 测试
```

不要新增按模型名或算子名分支的 CLI/builder 路线。

GC 默认使用 `--tile-size` 生成确定性 baseline；研究多个切分方案时可传入
`--tile-size-candidates 2,4,8`。planner 会基于 tile 数、估算计算周期、root traffic
和 local working-set overflow 选择一个候选，并在 `01_gc/schedule.json` 的
`candidate_costs`、`selected_tile_size` 中保留可审计的评分。该 cost model 只用于编译期
排序，不替代后端 timing provider，也不会让 static/dynamic 使用不同的 compiled artifact。

单条 StableHLO operation 的导入能力由 `StableHLOOpCapabilityRegistry` 管理；
Softmax、Norm 等多节点语义恢复由独立的 `SemanticFusionPatternRegistry` 管理。
后者只接受已经证明 shape、常量和数据流等价的图 pattern，不能用来吞掉未知
StableHLO operation 或作为静默 fallback。

Attention pattern 只添加非 opaque region metadata，并保留
`QK^T Matmul -> Softmax -> PV Matmul` 及其中间 transform 的独立 TISA；SwiGLU pattern
才会将已证明等价的 vector primitive chain 收敛为一个 `swiglu` TISA 边界，内部步骤
留在该指令的 backend payload 中。

## 输出目录

```text
out/<run>/
├── manifest.json
├── artifact_index.json
├── 00_frontend/
│   ├── source_frontend_import.json   # torch.export 源图摘要
│   ├── generated.mlir                # Torch-XLA StableHLO
│   ├── stablehlo_module.json         # verifier、版本和 provenance
│   └── frontend_import.json          # StableHLO 导入结果
├── 01_gc/
│   ├── canonical_graph.json       # GC 规范化后的算子图
│   ├── gc_artifact.json            # GC 完整阶段产物
│   ├── pass_dumps/                 # 每个 GC pass 的输入/输出图
│   ├── schedule.json               # 初始软件 schedule
│   ├── compile_statistics.json
│   ├── tile_graph.{json,dot}
│   └── operator_graph.{dot,svg}
├── 02_fc/
│   ├── tisa_dialect.json           # FC 输出的 TISA semantic ops
│   └── fc_diagnostics.json
├── 03_tisa/
│   ├── tisa_program.json
│   └── compiled_artifact.json
├── 04_backend/
│   ├── backend_artifact.json
│   ├── execution_graph.{json,dot}
│   └── machine.json
├── 05_runtime/
│   ├── runtime_submission.json
│   └── address_dependencies.json
├── 06_simulation/
│   ├── summary.json
│   ├── tasks.csv
│   └── tisa_instructions.csv
└── 07_trace/
    ├── swimlane.{svg,png}
    └── perfetto.json
```

推荐按目录编号依次检查。`generated.mlir` 是确认 Torch-XLA 输出的第一现场；`01_gc/pass_dumps/` 按顺序展示每个 GC pass 的输入图、输出图和诊断，`01_gc/gc_artifact.json` 展示最终融合、切分、初始顺序和依赖；`02_fc/tisa_dialect.json` 展示 FC 生成的语义 TISA op；`03_tisa/tisa_program.json` 是 device scheduler 的输入；`backend_artifact.json` 记录每条 TISA instruction 对应的后端 payload。复合算子的 reduce/exp 等内部步骤只在 backend payload 中出现。

当前 Softmax 的语义 TISA 已按 tile 粒度生成 `load -> softmax -> store`，默认使用
materialized row-wise lowering。可以通过 `--softmax-algorithm online` 启用分析版
online state payload：它保留同一个 `softmax` TISA 边界，将相邻 reduction tile 串成
`(max, sum)` 状态依赖链，便于比较调度周期。该实现不执行完整数值 online Softmax 的
rescale、最终归一化和输出 workspace，因此只能用于 scheduler/cycle 研究，不能称为
论文硬件或数值正确的 FlashAttention online 实现。

例如，使用 online 分析模型运行 Attention：

```bash
PYTHONPATH=src /usr/bin/python3.12 -m npu_ooo.cli compile-model \
  --torch-module examples.torch_models:MultiHeadAttentionBlock \
  --input-shape 1,4,8 --input-shape 1,1,4,4 \
  --tile-size 4 --softmax-algorithm online \
  --policy dynamic_ready_queue \
  --output-dir out/attention-online
```

运行产物的 `manifest.json` 会记录实际采用的 Softmax 算法；不指定参数时记录为
`materialized`。

## 调度边界

项目刻意区分三层决策：

```text
编译期：SchedulePlanner 决定 tile size 和 loop order
runtime：决定地址、command chunk 和 descriptor 到达时间
device scheduler：根据依赖、ROB、queue 和资源状态 issue TISA instruction
```

`--policy static_pipeline` 与 `--policy dynamic_ready_queue` 只改变最后一层。两者不会重新切 tile，也不会生成不同的 TISA 程序。

## Backend 与 RTL 校准

默认 registry：

```text
CodegenBackend: analytical
EventBackend: analytical_event
TimingProvider: analytical | timing_table | systolic_mxu_profile
```

导入 RTL trace 和 MXU VCS log 的方法见 [docs/rtl-calibration.md](docs/rtl-calibration.md)。`descriptor_issue_to_done` 与 isolated Matmul compute latency 是不同区间，provider 会拒绝错误映射。

## 代码导航

```text
src/npu_ooo/cli.py                       唯一用户入口和实验输出
src/npu_ooo/compiler/pipeline.py         线性 PyTorch-to-TISA 主流程
src/npu_ooo/frontend/bridge.py           torch.export 捕获与源图 provenance
src/npu_ooo/frontend/torch_xla_export.py Torch-XLA StableHLO 导出
src/npu_ooo/frontend/stablehlo_official.py 官方 verifier 和导入边界
src/npu_ooo/compiler/graph_compiler.py   论文 GC：图优化、切 tile、依赖
src/npu_ooo/compiler/fusion_patterns.py  GC 多节点语义恢复/融合 pattern registry
src/npu_ooo/compiler/fusion_compiler.py  论文 FC：TileGraph 到 TISA dialect
src/npu_ooo/compiler/tisa_generator.py   TISA dialect 到 virtual TISAProgram
src/npu_ooo/compiler/tisa_dialect.py     TISA 方言 stage/metadata 构造与语义 payload 绑定
src/npu_ooo/backend/                     可替换 backend 与 timing provider
src/npu_ooo/simulator/tisa.py            TISA device scheduler simulator
```

完整分层说明见 [docs/architecture.md](docs/architecture.md)，论文语义对齐见 [docs/tisa-alignment.md](docs/tisa-alignment.md)。

## 验证

```bash
PYTHONPATH=src /usr/bin/python3.12 -m unittest discover -s tests -v
PYTHONPATH=src /usr/bin/python3.12 -m compileall -q src tests examples
git diff --check
```

端到端测试真实调用 PyTorch、Torch-XLA 和官方 StableHLO；后端和 scheduler 单元测试可以直接构造其所属层的 IR，但这些 fixture 不是用户前端入口。
