# 研究决策记录

## 核心执行链

```text
PyTorch nn.Module
  -> torch.export
  -> Torch-XLA StableHLO
  -> official StableHLO verify/import
  -> Canonical OperatorGraph
  -> Schedule / TileGraph
  -> TISAProgram
  -> BackendArtifact
  -> RuntimeSubmission
  -> TISA device scheduler
  -> backend timing/event trace
```

用户入口使用真实 PyTorch module。StableHLO 由 Torch-XLA 生成，官方 bindings 负责
parse/verify，项目维护 semantic capability 到 Canonical/TISA/backend 的映射。

## 语义与执行层次

- `TileInstance` 表达切分范围；
- `TISAInstruction` 增加 OpType、TileShape、TileMem、AccessType、UnitMap 和 typed
  dependency；
- 全局 OOO scheduler 的输入是 TISAInstruction；
- `ExecutionTask` 表达一条 TISA issue 后在 execution unit 内执行的 backend payload；
- runtime 负责物理地址、command chunk、descriptor arrival 和同步；
- static/dynamic 使用同一 compiled artifact、MachineConfig、timing source 和 runtime
  submission，实验变量写入 manifest。

## Backend 定位

| Backend | 职责 |
| --- | --- |
| analytical | 可解释的 tile/event baseline |
| timing table | task/primitive duration 与 II 覆盖 |
| systolic MXU profile | 离线 MXU shape/duration/II 重放 |
| RTL completion importer | JSON/CSV/VCS log 转为 versioned profile |
| SCALE-Sim、Ramulator2/DRAMSys、RTL/Verilator | 后续局部硬件 timing |

## 公平比较规则

1. 调度策略实验固定 tile、地址、BackendArtifact 和 timing provider；
2. runtime policy 与 device policy 分开统计；
3. analytical、source-derived、RTL-observed 结果分组；
4. capability boundary 对每个 operation 给出名称、注册项和诊断；
5. fallback 由用户显式选择，并在 artifact 的 compatibility 字段记录。

## 关键语义结论

### LLaMA2 decode 与 state

`LLaMA2DecodeOneBlock` 使用 `x:[B,1,H]`、`cache:[B,heads,W,D]`、
`update:[B,heads,1,D]` 的固定窗口布局。cache 的 `slice + concatenate` 更新恢复为
两个带 `state_id/state_buffer` 的 `kv_cache_update`。RuntimeStateRegistry 提供稳定
persistent address，RuntimeSequence 以 `state_complete` 串联 invocation。动态 position、
paged cache、GQA/MQA layout 和跨 request ownership 属于后续 state capability。

### ResNet convolution/pooling

Torch-XLA 的 inference Conv2d 进入 `stablehlo.convolution`，BatchNorm 进入
`batch_norm_inference`，MaxPool/AvgPool 进入 `reduce_window`。Canonical importer
读取对称 padding、NCHW window 和 unit dilation。TileGraph 与 FC TileMem 使用一致的
kernel/window halo region，保证 backend root-memory handoff 与 TISA dependency 对齐。

### Dynamic shape

Torch-XLA dynamic export 生成 `get_dimension_size` 和 shape tensor。shape specialization
在官方 StableHLO 边界执行 constant propagation 与 dynamic-to-static legalization：

- dynamic broadcast：求值 shape dataflow 后改为 `broadcast_in_dim`；
- dynamic_slice：常量 start 按 clamp 规则改为 `slice`；
- dynamic_reshape：常量 shape 且元素总数守恒时改为 `reshape`。

运行时动态索引、动态更新和动态 layout 使用独立的 metadata、地址表达式和 state contract。

### Static broadcast 与 scalar

广播按输出域切 tile；`broadcast_dimensions` 将源 operand 投影到对应 region，singleton
轴读取 `[0:1]`。零秩 scalar 保留 shape `()`、constant_value、单元素 byte interval
和真实 elementwise input。reduce 的 init scalar 进入 `constant_args`。

### Dtype 与 multi-result

机器通过 `supported_dtypes` 与 `dtype_policy={strict,fallback}` 声明 dtype 能力。
已知 dtype 使用注册的字节宽度；未知 dtype进入 capability 诊断。Official StableHLO
projection 对多结果 operation 保留完整 result contract；`batch_norm_training`
recovery 使用已验证的 secondary result 规则。

### StableHLO capability 与 fusion registry

`StableHLOOpCapabilityRegistry` 处理单操作投影；`SemanticFusionPatternRegistry`
处理基于图结构、shape、常量和单消费者证明的复合语义。Attention 保留 QK、Softmax、
PV 等 scheduler-visible 成员，SwiGLU 将 vector primitive chain 绑定到一个 semantic
TISA payload。

## 当前实验标签

默认输出标签为 `TISA instruction-level analytical scheduling baseline`。RTL profile
的 source、interval、aggregation 和 calibration status 写入 manifest。泳道图同时展示
TISA lane 与 backend payload lane，便于定位 dependency wait、resource busy、
memory conflict 和 completion feedback。
