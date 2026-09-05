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

## 2026-08-31：GC typed dependency 语义

论文在 TISA 层定义：

```text
Deps = {(src, type, condition)}
type = RAW | WAR | WAW
```

`RAW` 表示写后读，`WAR` 表示读后写，`WAW` 表示写后写。论文通过 TISA operand 的
`TileMem` 区间、`scope` 和 `AccessType` 进行区间重叠分析；不重叠区域形成独立 tile，
`condition` 表示完整区域或部分区域的 ready 条件。`OpType` 和 `UnitMap` 负责语义
路由与资源选择，它们属于独立字段。

当前项目的 GC 输出使用：

```text
TileDependency(producer, consumer, tensor, kind)
kind = region_data | state | accumulate | buffer_reuse
```

其中 `region_data` 经 FC 映射为 `RAW`，`state` 和 `accumulate` 分别表达状态链与归约
顺序，映射为项目扩展的 `STATE` 和 `ACCUMULATE`。FC 为每条 TISA dependency 补充
`condition`，默认使用 `full_region_ready`，校准 backend 可以指定
`payload_ready:<task_id>`。

因此，当前 GC 边已经保留 producer、consumer、tensor region、hazard relation、ready
condition 和语义原因。FC、TISA、runtime address scoreboard 与 trace 复用同一 dependency
provenance，继续保持 region-aware tile overlap 和 scheduler-visible TISA 边界。

## 2026-08-31：GC typed dependency 第一版实现

`TileDependency` 现在显式保存 `hazard_kind`、producer/consumer logical region、
`condition` 和 `provenance`。`region_data` 建立 RAW 关系并携带 `full_region_ready`；
reduction/state chain 使用 STATE 或 ACCUMULATE，并携带 `state_complete` 或
`accumulate_ready`；buffer reuse 使用 BUFFER_REUSE 与释放条件。

FC 将上述字段原样投影到 `TISADependency`，同一 source/target 上的多条语义来源合并到
`provenance.sources`。这样 GC、FC、TISA、runtime address scoreboard 和 trace 使用同一
dependency provenance；`compile_statistics` 同时按 hazard 与 condition 汇总。

## 2026-08-31：symbolic shape binding 第一版

shape environment 现在由 frontend 统一校验：键使用 Python/StableHLO 兼容标识符，值为
正整数。Torch Export、StableHLO importer、shape specialization 和 Canonical resolve 使用
同一份 normalized mapping；symbolic tensor shape 可以在 specialization 中直接解析，
解析后的 shape、variant 和 environment 保存在 artifact provenance。

## 2026-09-02：dependency provenance 与动态索引

- Backend lowering 通过 `ExecutionTask.attributes["dependency_provenance"]` 保存
  predecessor 对应的 GC edge。metadata 同时包含 GC kind、paper hazard kind、logical
  region、ready condition 和来源；同 tile 的 primitive 顺序使用 `CONTROL`。
- ExecutionGraph trace、TISA trace、Perfetto 和 address scoreboard 读取同一份 metadata。
  address hazard 记录分为编译 provenance（能关联到 TISA dependency）和运行时物理区间
  provenance（`address_scoreboard`）。
- `DynamicIndexExpr` 是 Canonical/TISA 的符号索引 contract；它描述 source tensor、
  index operands、index rank、每轴 clamp bounds 和 StableHLO clamp rule。runtime 用
  `DynamicIndexBinding` 按 expression id 提供具体值并校验 rank。
- `dynamic_slice` 以 slice semantic family 进入 output-tile copy，并在 source TileMem
  中保留如 `arg0[clamp(arg1,0,6):+2,...]` 的符号地址表达式；当前 interval 使用完整
  source allocation，保证 analytical scoreboard 保守正确。
- `dynamic_update_slice` 进入 `kv_cache_update`，Canonical operator 保存
  `stateful/state_id/state_buffer` 与 DynamicIndexExpr；runtime state registry 继续
  管理持久 buffer，动态地址计算留待 stride-aware layout 阶段。

## 2026-09-02：阶段 2 实现审计

- StableHLO 标量索引使用 `i32`/`i64` 等元素类型。编译器当前只识别项目内部的
  `int32`/`int64` 别名，导致动态 slice 在 `compile_operator_graph()` 的 dtype 校验
  处提前失败。dtype 名称和字节宽度需要由一个共享模块统一解析。
- 动态 slice 的 TISA source operand 保留符号 address expression，runtime 使用
  `DynamicIndexBinding` 按 StableHLO clamp 规则解析窗口，再以 source tensor 的 dense 或
  显式 byte strides 计算 offset/span，并记录 resolved index/offset provenance。
- dynamic update slice 的 generic output 建立对 state buffer 的 persistent alias。TISA
  `state_region`、runtime operand binding 和 address scoreboard observation 使用同一 update
  window；完整 state allocation 仍由 `RuntimeStateRegistry` 管理。
- `DynamicIndexBinding` 保存 StableHLO 的原始有符号整数；`resolve_dynamic_index()` 输出
  clamp 后的非负合法起点，避免在 runtime boundary 提前改变索引语义。

## 2026-09-02：阶段 2 dynamic layout 与 transform

- StableHLO encoding 只有在暴露 byte strides、element strides 或 minor-to-major 等结构化
  信息时才解析为 concrete layout；裸 tag（如 `#row_major`）没有可验证定义时保留
  opaque/conservative region。
- logical region 的起点始终使用基础 tensor stride；slice step 只作用于跨度计算，避免
  将起始 offset 错误地乘以 slice step。
- transpose 固定使用 `output[d] = input[permutation[d]]`。TileGraph 在 output domain
  切分，transform lowering 按 permutation 生成 source region，使 source/output interval
  可以独立进入 dependency 与 scoreboard。
- `RuntimeLayoutBinding` 与 `DynamicIndexBinding` 一样按 invocation 提供具体值。编译产物
  保持不变，runtime 根据 layout 绑定重算 operand offset/span，并将 layout provenance
  保存到 submission/trace metadata。
- bank/port timing 使用 resolved physical address。小于 4096 个元素的 strided region
  展开实际 bank，大 region 使用 bounded fallback，避免 seqlen 增长导致仿真复杂度无界。
