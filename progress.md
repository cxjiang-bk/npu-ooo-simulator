# 当前进度

## 当前快照

项目已交付 Device scheduler 的逐周期微结构模型；控制开销的实际硬件校准和外部执行
timing backend 是后续工作。
生产链路和实验目录已经固定：

```text
PyTorch nn.Module
  -> torch.export
  -> Torch-XLA StableHLO
  -> official StableHLO verify/import
  -> GC / TileGraph
  -> FC / TISA dialect
  -> TISAProgram / BackendArtifact
  -> RuntimeSubmission
  -> static 或 dynamic device scheduler
  -> analytical / calibrated backend
  -> cycles、stall、swimlane、Perfetto
```

当前回归状态以本文件末尾的 scheduler 周期模型验收记录为准。

## 已交付能力

### 前端与编译

- 统一入口 `compile-and-sim --torch-module MODULE:CLASS`；分离执行使用 `compile` 和
  `simulate`；
- torch.export 源图 provenance、Torch-XLA StableHLO 导出和官方 MLIR parse/verify；
- StableHLO operation capability registry、semantic fusion/recovery registry；
- GC pass pipeline、固定点规范化、tile planner、candidate cost model；
- region-aware dependency、卷积/池化 halo、静态 broadcast、scalar region；
- TileMem 的 stride、layout、dtype metadata；
- 常量 dynamic broadcast、dynamic_slice、dynamic_reshape specialization；
- GC/FC/TISA 三阶段 artifact 与逐 pass dump；
- materialized/online Softmax payload 属性。

### 模型 benchmark

- 两头 multi-head Attention；
- pre-norm decoder block：RMSNorm、Attention、residual、SwiGLU/MLP；
- BERT、GPT-J、LLaMA2、DeepSeek dense one-block；
- BERT/GPT-J/LLaMA2/DeepSeek Transformer repetition 与 embedding/model shell；
- DeepSeek MoE proxy：router、top-k mask、expert GEMM、weighted dispatch/combine；
- DeepSeek one-token fixed-window decode 与多 request state replay；
- LLaMA2 RoPE、固定窗口 KV-cache、prefill/decode micro；
- ResNet bottleneck micro：Conv2D、BatchNorm inference、ReLU、MaxPool；
- registry 六个 case 与 paper-matrix 批处理。

### Runtime、scheduler 与 backend

- RuntimeSubmission 的地址绑定、command chunk、descriptor arrival；
- RuntimeStateRegistry、RuntimeSequence 和 state-complete invocation 链；
- static_pipeline 与 dynamic_ready_queue 共用同一 BackendArtifact；
- queue/ROB/window、UnitMap 资源、typed dependency、address scoreboard；
- 可选 memory bank/port structural-conflict model；
- analytical、timing table、systolic MXU profile 和 RTL completion importer；
- staged output、周期/stall 汇总、泳道图和 Perfetto trace。

## 当前能力边界

以下能力已定义为后续精确扩展项，并在编译或运行时保留清晰的语义入口：

- 更复杂 dynamic shape/index/layout dialect：扩展符号约束、paged layout 和可校准 memory
  timing；
- online Softmax 数值实现：扩展 rescale、最终 normalization 和 workspace 生命周期；
- 动态 top-k、token compaction/expert capacity、完整多层 decode cache；
- 论文 channel/hidden width、完整 stage topology 和数值等价 backend；
- 论文 WQ/IQ/Fu 容量、dispatch/wake-up/issue/completion 控制开销校准；
- SCALE-Sim、Ramulator2/DRAMSys、RTL/Verilator 和 system simulator backend。

当前默认结果标签为 TISA instruction-level analytical scheduling baseline。RTL profile
加载后，manifest 记录 source、interval、aggregation 和 calibration status。

## 关键里程碑

### 2026-08-29：Runtime state 与 LLaMA2 decode

固定窗口 KV-cache 通过 `state_id/state_buffer` 建立 persistent binding；两次 invocation
复用同一 BackendArtifact，由 `state_complete` 串联并合并事件。LLaMA2 decode one-block
生成 scaled micro workload，输出包含 runtime sequence artifact。

### 2026-08-29：ResNet capability

官方 StableHLO convolution、batch_norm_inference、reduce_window capability 接入
Canonical/GC/FC/backend。卷积和 pooling 的 halo region 进入 TileMem 与 dependency。

### 2026-08-29：GC cost model

SchedulePlanner 支持 tile candidate、计算/traffic/working-set cost，并记录
`candidate_costs` 与 `selected_tile_size`。

### 2026-08-30：shape、layout、dtype 语义

dynamic broadcast、常量 dynamic_slice、常量 dynamic_reshape specialization 接入官方
StableHLO verify；scalar tensor、layout encoding、dtype policy 和 multi-result boundary
进入 Canonical/TISA contract。

### 2026-08-30：论文矩阵

六个 registry case 使用统一 paper-matrix 入口。共享编译产物位于
`00_frontend` 到 `04_backend`，策略结果位于 `policy_matrix/05_runtime` 到
`07_trace`，根目录写入 `matrix_index.json`、`sweep.csv/json`。

## 下一步

按以下顺序推进：

1. scheduler 微结构和控制开销校准；
2. 外部 timing/memory/RTL backend；
3. source-derived 与 RTL-observed 论文矩阵；
4. 动态 MoE token 数据流和精确 full-model 扩展。

## 2026-08-31：进度盘点与执行顺序确认

项目已完成真实 PyTorch 到 TISA device simulator 的主链路，当前回归记录为 121 项通过。
编译、runtime、scheduler、backend、staged artifact 和论文 proxy matrix 均具备可复现入口。

GC typed dependency 已完成，下一项转入 symbolic shape、dynamic index 和 layout binding。

## 2026-08-31：GC typed dependency 第一版完成

- `TileDependency` 现在保存 `hazard_kind`、producer/consumer logical region、`condition`
  和 `provenance`，并提供统一校验与 JSON 表示；`region_data`、`state`、`accumulate`、
  `buffer_reuse` 分别映射到论文式 RAW、STATE、ACCUMULATE、BUFFER_REUSE 关系。
- `TISADependency` 继承 GC provenance，重复来源合并为 `sources`，因此同一条 TISA 边可
  审计 stage order、graph edge 和 reduction/state chain 的来源。
- `compile_statistics.json` 新增 `hazard_kind_counts` 与 `condition_counts`，泳道和
  Perfetto 继续使用同一 TISA dependency 数据。
- 新增 Matmul region、RAW provenance、Softmax STATE provenance 和统计汇总回归；全量
  回归为 119 项通过，`compileall` 与 `git diff --check` 通过。

下一项实现转入 symbolic shape、dynamic index 和 layout binding。

## 2026-08-31：symbolic shape binding 第一版完成

- 新增统一 `normalize_shape_environment()`，在 Torch Export、StableHLO importer 和
  shape specialization 边界校验符号标识符与正整数值。
- shape specialization 使用 Canonical graph 中的 symbolic tensor shape，结合
  `shape_environment` 求解动态 broadcast、dynamic slice 和 dynamic reshape 的输入/输出
  shape；绑定结果写入 frontend provenance 与 specialization artifact。
- 新增符号 shape specialization 与非法绑定回归；全量回归更新为 121 项通过，
  `compileall` 与 `git diff --check` 通过。

下一项继续扩展 dynamic index、dynamic layout 和 stride-aware transform。

## 2026-09-02：GC provenance 贯通 trace 与 address scoreboard

- `lower_mixed_graph()` 在每个 `ExecutionTask` 上保存 predecessor 到 GC `TileDependency`
  的映射，包含 kind、hazard kind、logical region、condition 和 provenance；同 tile 的
  backend stage edge 使用明确的 `CONTROL/payload_stage_complete` 语义。
- ExecutionGraph 的 `WAKE_UP`、`ISSUE`、`COMPLETE`、address scoreboard stall 和
  `address_hazards` 直接输出上述 metadata。TISA trace、Perfetto instant event、tasks.csv
  和 tisa_instructions.csv 使用相同字段。
- TISA address scoreboard 将逻辑/物理区间冲突统一表示为 predecessor、successor、
  RAW/WAR/WAW、tensor、memory、condition、provenance，并增加 `TISA_ADDRESS_BLOCK` 事件。

## 2026-09-02：dynamic index 正式 contract

- 新增 `DynamicIndexExpr` 与 `DynamicIndexBinding`。expression 保存 source tensor、
  index operands、index rank、StableHLO clamp rule、每轴 bounds、resolved values 和
  attributes；binding 保存 invocation 的具体 index 值并执行 rank 校验。
- StableHLO capability registry 接入 `dynamic_slice` 与 `dynamic_update_slice`。动态
  slice 在 Canonical/FC/TISA 中保留 metadata 和符号 address expression；动态 update
  slice 恢复为带 `stateful/state_id/state_buffer` 的 `kv_cache_update`。
- specialization 对运行时索引保留动态 operation，对常量索引继续执行 clamp 后静态化。
  runtime submission 保存并校验 DynamicIndexBinding；trace 和 metrics 输出 binding。
- transform lowering 接入动态 slice 的 output-tile copy，确保该 operation 可以进入
  GC -> FC -> TISA -> analytical backend 链路；未提供 runtime binding 时保留保守 logical
  payload 区间。

## 2026-09-02：阶段 1/2 继续推进

- 审计发现 StableHLO `i32` 索引在统一编译入口被 dtype policy 拒绝；下一步使用共享
  dtype registry 解决所有 StableHLO/内部别名和字节宽度。
- 动态地址解析设计固定为 `DynamicIndexExpr + DynamicIndexBinding + TensorSpec` 的
  三方契约：先 clamp，再按 dense 或显式 byte strides 求窗口 offset，最后执行容量校验。
- dynamic_update_slice 的 state region 将沿同一契约写入 TISA/runtime/scoreboard，保持
  state buffer 的 persistent allocation 与动态写入窗口分离。

## 2026-09-02：动态地址与 state window 交付

- 建立共享 dtype registry，统一 StableHLO `i32/i64/ui*` 与项目内部 dtype 别名，并由
  同一入口提供 capability 比较和存储字节宽度。
- `allocate_buffer_bindings()` 现在保存每个 buffer 的 resolved shape、byte strides 和
  layout。runtime 根据 `DynamicIndexExpr`、`DynamicIndexBinding` 和这些 layout metadata
  先执行 StableHLO clamp，再计算动态窗口的物理 offset/跨度并执行容量校验。
- dynamic slice source operand 使用窗口 shape；dynamic update slice 的 state output alias
  使用 update window。TISA、runtime operand binding 和 address scoreboard 都保留
  `dynamic_region`、resolved indices、resolved offset 和 provenance。
- `create_runtime_sequence()` 支持每个 invocation 的动态索引绑定；新增两项端到端回归。
  全量回归更新为 129 项通过，`compileall` 与 `git diff --check` 通过。
- runtime binding 接受有符号索引并在 resolver 中执行 StableHLO clamp；新增负索引回归。

阶段 2 的 dynamic index/state 与 layout/transform address contract 已完成；下一项转入完整
模型 repetition、DeepSeek MoE capability 和论文形状的 traffic 对账。

## 2026-09-02：阶段 2 dynamic layout 与 stride-aware transform 完成

- 新增统一 `ir.layout` resolver，解析显式 byte strides、结构化 StableHLO
  `strides/byte_strides`、`minor_to_major`、row-major/column-major；opaque encoding 保留
  conservative logical region。
- `BufferRegion` 和 `TileMem` 保存 stride 与 logical starts/shape；matmul、TISA、transform
  lowering、runtime allocation 共用同一套 offset/span/allocation 计算。
- transpose 按 permutation 映射 output tile 到 source region；static slice 合成 source
  stride 与 slice stride；reshape 记录 contiguous view 或 strided materialize copy。
- `RuntimeLayoutBinding` 支持 invocation-level shape、stride、layout、offset 绑定；runtime
  执行容量校验，bank/port scoreboard 对小型 strided region 使用 element-level bank mapping，
  大型 region 使用 bounded conservative fallback。
- 新增 layout、transpose、slice、runtime layout 回归；全量回归 134 项通过，`compileall`
  与 `git diff --check` 通过。

## 2026-09-02：动态参数入口文档化

- README 新增 compile/runtime/simulation 边界说明，明确 `compile-and-sim` 是端到端编排入口，
  但 `TISAProgram/BackendArtifact`、`RuntimeSubmission` 和 `SimulationResult` 在语义上
  仍然分层，可被独立 API 调用。
- README 增加 `position`、KV-cache index、`seqlen` 和动态 layout 的 binding 表，说明
  `--input-shape` 属于 compile-time specialization；当前动态 index/layout 通过 Python
  runtime API 传入，CLI 的 runtime 参数暂不自动映射到模型输入。

## 2026-09-04：compile 与 simulation 分离

- 新增 `compile` CLI，只执行 PyTorch -> torch.export -> Torch-XLA/StableHLO -> GC/FC ->
  TISA/backend，并将 `BackendArtifact`、Canonical Graph、TISA program、MachineConfig
  和阶段产物保存为可复用 compile package。
- 新增 `simulate --compile-dir` CLI。它从 `01_gc/canonical_graph.json`、
  `04_backend/backend_artifact.json` 和 `04_backend/machine.json` 恢复对象，不导入
  PyTorch、不重复编译，只执行 runtime buffer/address binding、dynamic index/layout
  解析、descriptor 提交和 device/backend timing。
- 新增 `compile-and-sim`，替代旧的 `compile-model`，保留原有一站式行为；旧命令显式
  拒绝，文档和测试已同步。
- 为 `TensorSpec`、`OperatorGraph`、`BufferRegion`、`ExecutionTask`、`ExecutionGraph`
  以及全部 TISA/BackendArtifact 对象增加严格 `from_dict()` 校验，确保跨进程/跨命令
  的 artifact 恢复后仍满足原有 IR contract。
- `simulate --runtime-config runtime.json` 支持每次 invocation 的
  `dynamic_indices`、`dynamic_layouts`、runtime policy、chunk、launch latency、
  descriptor availability 等配置；同一 compile package 可复用到不同 machine、timing
  provider、runtime policy 和 device policy。
- 用已有 decoder compile package 完成独立仿真 smoke：284 条 TISA、329 个 backend task、
  2953 cycles；新增 BackendArtifact JSON round-trip 回归。
- CLI 参数边界已拆分：`compile` 只暴露编译输入和 codegen 选项；`simulate` 暴露 runtime、
  device scheduler、机器覆盖和 timing/event backend；`compile-and-sim` 组合两组参数。
- 新增无前端依赖的 compile package -> `simulate` 集成回归，确认仿真入口从 JSON 恢复
  `OperatorGraph`、`BackendArtifact`、`MachineConfig`，并只生成 runtime、simulation、trace
  产物。定向 CLI/TISA 回归为 24 项通过。

## 2026-08-31：GC typed dependency 语义对齐与实现

- 对照论文 Section IV 与 V-A，确认 TISA 的形式化依赖为
  `Deps={(src,type,condition)}`，正式类型为 `RAW/WAR/WAW`；类型描述访问冲突关系，
  `condition` 描述所需区域何时 ready。
- 明确项目的两级映射：OperatorGraph `DataEdge` 经过逻辑 region 投影形成 GC
  `TileDependency(kind=region_data/state/accumulate)`，FC 再映射为
  `TISADependency(kind=RAW/STATE/ACCUMULATE/...)` 并补充 readiness condition。
- 确认 `region_data` 表示 producer/consumer 的逻辑数据区域关系，`state` 表示跨
  reduction tile 的状态链，`accumulate` 表示部分结果的归约顺序；这些是项目 GC 语义
  类别，论文正式 TISA 类型仍以 RAW/WAR/WAW 为核心。
- 完成 GC 边的 hazard relation、region、condition 和 provenance 字段，并让 FC 与
  device scheduler 审计每条依赖的来源和满足条件。

## 2026-09-04：模型与 benchmark proxy 覆盖完成

- `paper-matrix` 增加 `model_scope`、`layer_count`、`deepseek_mode`、`request_count` 和
  inter-request gap；非默认组合使用独立 profile 路径和编译 model id。
- `model_proxy` 为 Transformer 加入 token embedding、BERT position/type embedding、
  decoder causal mask、RoPE、重复 block 和 output head；ResNet proxy 加入 stem、重复
  bottleneck、global average 和 classifier。
- 新增 StableHLO embedding row-gather capability、official projection、TileGraph/TISA
  operand region 和 analytical gather payload。真实 Torch-XLA 产物使用扁平 `ui32`
  indices，已纳入 dtype/shape 契约。
- DeepSeek MoE proxy 在图内计算 router softmax，以 request top-k mask 驱动四个 expert，
  GC 标注 scheduler-visible `moe_dispatch` region；动态 top-k、token compaction 和 capacity
  保持为显式精确路由边界。
- DeepSeek decode 改为一 token + 两份 fixed-window KV state；`request_count` 复用同一
  BackendArtifact，stateful invocation 使用 `state_complete`，stateless invocation 顺序重放。
- reduce lowering 支持多个 iteration/reduction 维度，解决 batch/sequence router reduction
  与 spatial global reduction 的通用 region/partial accumulation 问题。
- BERT 两层/两请求真实 paper-matrix smoke：static 34053 cycles、dynamic 30999 cycles；
  这些数值属于 analytical proxy，只用于验证完整实验链和相对调度趋势。
- 本地回归发现 161 项，127 项执行通过、34 项因缺少 Torch-XLA/MLIR 跳过；9980X-new
  的完整前端环境执行全部 161 项并全部通过。两端 `compileall` 与 `git diff --check` 通过。

## 2026-09-06：Device scheduler 逐周期微结构模型

- 以 `af8b528` 为代码基线，新增可选择的 `cycle_event`，输入保持 BackendArtifact、
  RuntimeSubmission 和 MachineConfig。payload 使用现有 TimingProvider。
- 显式维护 reception、per-EU WQ/IQ/Fu、完成待反馈状态和 ROB；Fu 按 operand 条目占用，
  ROB 在 dispatch 分配、按 descriptor 提交顺序退休。ROB 规则属于项目设计假设。
- 固定每周期 `retire → complete → wakeup → issue → select → dispatch → receive`，
  参数化各阶段 width、dispatch/select/wakeup/completion/retire latency；物理完成、完成反馈
  和退休分别输出事件。
- 地址检查覆盖原始 program order 中所有较老未完成指令，包括尚未 issue 的生产者。
  修正 bank 跨界元素映射，并限制大连续区间扫描到 bank 数量。
- 独立统计队列满、Fu/tile 窗口、依赖、资源、地址、bank/port、完成带宽、退休反压及各阶段
  带宽；输出每周期容量和每条指令生命周期。Sequence 保留各 invocation 的阶段绝对周期。
- 新增 20 项调度专项测试，其中一项覆盖 20 组固定种子依赖图的两种策略；另增加真实
  Attention 编译包跨策略 hash 与依赖检查。CSV/Perfetto/standalone CLI 均有验收。
- 全量验证：本地发现 182 项，147 项执行通过、35 项因 Torch-XLA/MLIR 依赖缺失跳过；
  9980X-new 的独立测试副本执行全部 182 项并通过。新增模块 Ruff F 类检查、compileall
  和 git diff --check 通过。
- Attention CLI 实验：同一 127 条 TISA 编译包，address scoreboard 开启，默认 cycle
  pipeline 下 static=3874 cycles、dynamic=3613 cycles。各阶段产物保存于
  `out/scheduler-cycle-20260906/`（实验输出目录不纳入 Git）。
- 接口及设计边界见 `docs/device-scheduler.md`。scheduler 时序状态为 uncalibrated；
  实际 Epoch 7–9 cycle dispatch、partial-ready 子区域协议、在线优先级和多核路由需独立校准。

## 2026-09-07：目标存储规划与 Runtime binding 根因修复

- 根因确认：FC/TISA、backend payload 与 Runtime 原先分别维护逻辑 TileMem、局部访存和
  DRAM tensor allocation，没有共享 allocation identity。现由 CodegenBackend 统一输出
  最终目标 TISA、ExecutionTask payload 和 `MemoryPlan` v2；Runtime 只绑定地址空间基址。
- `MachineConfig.operation_placements` 显式描述 Matmul 的 operand role、目标 memory、EU 和
  transfer route。minimal 为 DRAM↔SRAM；lpu-like 为
  `GM→UB→LMB/RMB→MXU→PSB→UB→GM`，跨 GDMA/LDMA/MXU/ARU 的边界均是独立 TISA stage。
- `buffer_id/allocation_id` 已贯通 BufferRegion、TileMem、MemoryPlan、BufferBinding 和
  RuntimeOperandBinding。最终 TISA 对 payload 的 memory/access/range 做严格覆盖校验；
  scheduler 在无 Runtime 时按 allocation identity 检查，在有 submission 时按物理地址检查。
- 局部 tile 使用 packed span，外部 strided source 保留 bounding span；`valid_bytes` 独立表示
  有效数据/traffic。4×4 f32、row stride 32 的验收值为 64-byte valid、112-byte source
  span、64-byte packed local span。
- allocator 按 memory capacity/alignment 分配 slot，通过 RAW/WAR/WAW 与
  `BUFFER_REUSE/allocation_released` 保护复用；alias/view 保留独立 buffer identity 和显式
  alias。容量不足直接报告 memory、需求和容量。
- compile package 增加 `04_backend/memory_plan.json` 和 topology hash。simulate 可覆盖容量、
  带宽、latency、bank/port、unit count（仍需满足容量），改变 topology/placement/alignment
  时拒绝复用；旧 package 明确要求重新编译。
- 验证：本地 `151 passed, 35 skipped, 51 subtests passed`；9980X-new 正式
  PyTorch/Torch-XLA/StableHLO 环境运行 186 项，全部通过。正式 Attention 使用同一份
  lpu-like compile package 运行 cycle_event static/dynamic，均退休 21 条最终 TISA，
  compile package hash 同为 `edae32af27f3ca5820bbbca1450c74fef8297ed2e47f2d4b3435004e1b716237`；
  本次小图两种 device policy 均为 560 cycles，说明新增约束下没有可利用的额外 ready 并行。
  新产物位于 `out/target-memory-20260907/`。
- 兼容边界：Matmul 已具备专用 role placement；其他 operation 目前由 backend 通用
  root/local materialization 保证 TISA/payload/runtime 一致，尚未逐类加入专用多级 route。
  bank-aware placement、eviction/fragmentation、跨 core routing 和真实 memory timing 仍属后续。

## 2026-09-07：FC 与 Target Lowering 边界修正

- 上一版虽然统一了 MemoryPlan，但 FC 的 `_matmul_target_stages()` 已读取具体 route/EU，
  Matmul lowering 又用 `_route_stage_keys()` 独立重复分组。本轮删除两处配对逻辑，将物理
  route 展开完整后移到 CodegenBackend 内的 `TargetLowerer`。
- FC Matmul 对不同 target 固定为 abstract load/compute/store。Load/store 同时声明 source
  与 destination，TileMem 保存 operand role、symbolic buffer、Private/Local/Shared
  visibility、owner/domain；FC 不再出现 GM/UB/LMB/RMB/PSB 或 GDMA/LDMA。
- Generator 输出独立 `virtual_tisa_program.json`。`TargetPlan` 保存 abstract→target 映射、
  每跳 source/destination/engine/transform、target operands、依赖展开和 provenance；最终
  scheduler program 仍满足 `compiled.tisa_program == backend_artifact.program`。
- Matmul payload 逐条读取 TargetPlan 生成。其他已有 lowerer 的 region 必须绑定到规划 operand
  并通过覆盖校验；Softmax/Norm 等 backend-private scratch 在 allocation 前作为显式
  internal-resource declaration 加入 TargetPlan。
- 新增隔离测试：同一 GCArtifact/FC 输出在 minimal 与 lpu-like 完全一致，而 target plan
  分别生成 3 条和 5 条 target TISA；添加 MID 中间存储和 AUX_DMA engine 时只改变
  TargetPlan，FC 保持不变。另覆盖 abstract provenance、跨 hop 依赖及 TargetPlan JSON
  round-trip。
- 保留 MemoryPlan v2、容量/对齐、valid-bytes/span、alias/reuse hazard、Runtime binding、
  topology guard、dynamic index/KV state 和 static/dynamic 同包契约。新验收产物使用
  `out/fc-target-boundary-20260907/`，旧实验目录保留作历史对照。
- 该职责划分是本项目的实现选择：论文支持符号 TileMem 与 target-specific generation，
  但未公开 Epoch 内部 TargetPlan/allocation 的确切模块归属。
- 验证：本地 `155 passed, 35 skipped, 51 subtests passed`；9980X-new 正式前端环境
  `190 tests` 全部通过。真实 Attention 的 FC/virtual 指令数均为 14，lpu-like target
  lowering 展开为 21 条，static/dynamic 均退休 21 条并共享完全相同的 runtime buffers；
  compile package SHA256 均为
  `0fec557838af564a90fc9bf8b4a500d2463d44b7d83f4cd02eedd88b47997b7d`，当前小图两种
  policy 均为 560 cycles。
