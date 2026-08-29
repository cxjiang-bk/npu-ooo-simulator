# 当前进度

## 已完成

- 唯一生产入口：`compile-model --torch-module MODULE:CLASS_OR_FACTORY`；
- 完整前端链路：`PyTorch -> torch.export -> Torch-XLA -> official StableHLO -> Canonical IR`；
- StableHLO 官方 parse/verify 与 semantic capability boundary；
- 自动 graph pass、tile planner、TISA dialect semantic builder 和 analytical backend payload；
- TISA instruction 粒度 static/dynamic device scheduler；
- runtime 地址分配、command chunk、descriptor availability 和 runtime/device 四组合矩阵；
- 可配置 MachineConfig、codegen/timing/event registry；
- staged output：`00_frontend` 到 `07_trace`、manifest、artifact index、SVG/PNG/Perfetto；
- MXU RTL completion trace 和 VCS console log 的离线 profile importer；
- 删除手写 benchmark、直接 StableHLO/JSON 输入和旧 primitive scheduler 生产入口；
- 54 个回归测试、真实 attention CLI smoke 均通过。
- 首个两头 Attention block 已覆盖 head reshape/permute、scale、additive mask、Softmax、PV、output projection 和 residual；
- reshape/transpose 已生成 DMA-bound TISA transform，TileGraph 已使用逻辑 tensor region 建边；
- `compile_statistics.json` 已输出 per-operator tile/TISA/payload、MAC、root traffic 和 dependency 统计；
- 当前回归为 62 项；一次编译的四组合实验中 device static=2344、dynamic=2119 analytical cycles。
- TISA readiness 已增加 completion-boundary 解释器和显式 `payload_ready:<task_id>` partial-ready 原型；后者可在 source payload 完成前唤醒依赖，并输出 `TISA_PARTIAL_READY` trace 事件。
- 已接入可选 memory bank/port scoreboard：依据 `MachineConfig.memory_levels` 的 bank 和读写端口容量记录结构冲突；默认关闭，不改变既有 baseline。
- GC 的 LayerNorm recovery 已改为 fixed-point pass，可完整处理同一 module 中多个 Torch-XLA `batch_norm_training` 规范化节点。
- RMSNorm recovery 已支持 Torch-XLA 的 `power -> reduce -> rsqrt -> affine` 形式及中间 reshape，并在 backend payload 中建模 affine weight read。
- 首个真实 PyTorch pre-norm decoder block 已编译并完成 static/dynamic device 仿真，覆盖 RMSNorm、multi-head attention、masked Softmax、residual 和 SwiGLU/MLP；RoPE 与 KV-cache 尚未加入。

## 真实验证

```text
Python 3.12
torch 2.9.1
torch-xla 2.9.0
StableHLO 1.12.1...
multi-head attention: 48 tiles, 121 TISA instructions
minimal analytical device: static 2344, dynamic 2119 cycles
```

## 当前限制

- StableHLO semantic importer 仍只覆盖已注册 operation；
- tile planner 还是确定性启发式，没有 cost model；
- reshape/transpose 仍是 full-tensor transform，symbolic/dynamic shape 与 stride-aware layout 尚未完成；
- analytical event backend 不是 RTL cycle-accurate；
- 当前 MXU VCS log 主要提供 descriptor-to-completion 区间；
- GC 当前只生成 completion-boundary readiness；真实 partial-tile producer 语义仍需由 backend/calibration 端接入；
- memory bank scoreboard 目前是 analytical structural-conflict model，不是 cycle-accurate SRAM/DRAM backend；logical scope 且无 runtime physical binding 时不会强行猜测 bank；
- 论文规模的完整 ResNet50、BERT、GPT-J、LLaMA2、DeepSeek block 尚未形成可复现实验集；
  当前已具备 ResNet bottleneck、BERT/GPT-J/LLaMA2/DeepSeek dense one-block micro case。

## 下一步

保持模型到 TISA 为当前主线：补齐 symbolic/layout/cost-model 编译能力，并确认
DeepSeek dense 与 MoE 两条路径的 operation 边界；scheduler/backend 校准继续后置。

## 2026-08-27：阶段 1A

- 完成 `stablehlo.convert` capability/import 和 dtype conversion metadata；
- 完成缺失 StableHLO capability 的显式诊断；
- 新增 2 个 capability boundary regression tests；
- 全量回归：69 tests passed；BERT/GPT-J/LLaMA2 micro workload 已能从 PyTorch 生成 TISA；
- 补齐 StableHLO `f16/f32` 到 TISA/runtime 的 dtype-byte 规范化别名；
- 下一项：建立 Fusion Pattern Registry，随后补 Transformer/ResNet 模型语义。

## 2026-08-28：阶段 1B Fusion Pattern Registry

- 新增独立的 `SemanticFusionPatternRegistry`，与单操作
  `StableHLOOpCapabilityRegistry` 保持职责分离；
- 注册现有 LayerNorm recovery、LayerNorm、RMSNorm、Softmax 多节点 pattern；
- 默认 GC pipeline 由结构 pass 与 semantic pattern priority 确定性合并，八个 pass 的
  历史顺序、名称、fixed-point 与 dump 行为保持不变；
- 新增 5 项 registry 回归；全量回归为 78 tests passed；
- BERT、GPT-J、LLaMA2、DeepSeek prefill/decode micro workload 均重新生成有效 TISA；
  ResNet 仍按预期在 `stablehlo.convolution` capability boundary 显式失败；
- 下一项：实现 Attention region 与 SwiGLU pattern，并以 BERT/GPT-J one-block 回归验证。

## 2026-08-28：阶段 1B Attention/SwiGLU

- 新增 `recover_attention_region`：识别真实 Torch-XLA 图中的
  `QK matmul -> score transform -> Softmax -> probability transform -> PV matmul`，
  生成非 opaque region metadata，但保留每个成员为 scheduler-visible TISA；
- 新增 `fuse_swiglu`：将 `logistic -> silu multiply -> gate multiply` 恢复为一个
  `swiglu` semantic operator，projection Matmul 保持 region 外部；
- 新增 SwiGLU analytical lowering，compute payload 在同一 vector EU 内执行
  `logistic/silu_multiply/gate_multiply`；对 LLaMA2 Torch-XLA 的 `f32 -> f16 -> f32`
  round-trip 保留显式 `dtype_convert` payload，不丢失 dtype 语义；
- FC、lowering registry、backend capability registry 和 TileGraph metadata 已同步；
- MHA、pre-norm decoder、BERT/GPT-J/LLaMA2 micro 回归及 static/dynamic 仿真通过；
- BERT/GPT-J one-block 已有独立 Attention region、SwiGLU payload 和 artifact validity
  回归；全量测试为 79 tests passed；
- 下一项是开始 LLaMA2 RoPE/KV-cache 需求分析。

## 2026-08-29：阶段 1C 开始

- 固定窗口 KV-cache 已具备单次 `persistent_buffer_v1` contract，并能从真实
  PyTorch/Torch-XLA 图恢复 `kv_cache_update`；当前缺少跨 invocation 的 state 生命周期。
- 本阶段新增 `RuntimeStateRegistry`、`RuntimeSequence` 和 sequence simulator，目标是让
  第 N 次 submission 显式依赖第 N-1 次 state completion，同时保持单次 API 兼容。

## 2026-08-29：阶段 1C 完成

- `RuntimeStateRegistry` 同时保存完整 runtime buffer 集合与 persistent state 子集，按
  `state_id` 校验 alias 的地址、memory scope 和容量稳定性；persistent binding 不进入
  临时 buffer reuse。
- `RuntimeSequence` 为同一 `BackendArtifact` 创建多次 invocation，生成明确的
  `state_complete` 依赖边，并支持每步替换普通 input/output bindings。
- `simulate_tisa_sequence` / `schedule_tisa_sequence` 复用现有 EventBackend，平移合并
  invocation timing，发出 `STATE_RELEASE/STATE_WAIT/STATE_READY` 事件并统计 sequence
  总周期、间隔等待和 invocation 周期。
- 新增固定窗口 KV-cache 两步 decode 回归；全量回归为 92 tests passed。

## 2026-08-29：LLaMA2 decode workload

- 在 `examples/paper_benchmarks/llama2.py` 增加独立的 `LLaMA2DecodeOneBlock`，输入为
  one-token hidden、K/V cache、显式 RoPE cos/sin 与 attention mask，输出 hidden 和更新后
  的两个 cache；cache 更新严格使用 `slice + concatenate`，便于对齐当前 GC contract。
- `build_decode()` 不新增或伪造论文 Table IX 的测量行，只提供带 `phase=decode` 的 scaled
  micro workload；完整模型、真实 hidden/head/cache layout 仍不在范围内。
- 真实 Torch-XLA 编译恢复两个 `kv_cache_update`，并可构造两步 `RuntimeSequence`；CLI
  实测输出 307 TISA instructions、2 invocations、8851 analytical cycles，产物包含
  `05_runtime/runtime_sequence.json` 和合并泳道图。
- LLaMA2 decode 的 static/dynamic sequence 回归均通过；全量回归更新为 94 tests passed。

## 2026-08-29：ResNet Conv2D/BatchNorm/Pooling

- 官方 StableHLO 投影修复 `dense<scalar> : tensor<...>` 属性解析，避免把 MLIR
  类型维度误读成 padding 数据；`stablehlo.convolution` 的对称 padding 可正确展开。
- 新增 StableHLO `batch_norm_inference` 与 `reduce_window` capability、官方投影和
  Canonical importer；未覆盖的 grouped/layout/dynamic 变体仍显式失败。
- 新增 BatchNorm inference 与 NCHW max/sum pooling analytical lowering、TISA stage、
  ARU capability 和 region-aware halo dependency。Torch Export 会忽略未使用的零秩
  BatchNorm bookkeeping placeholder（例如 `num_batches_tracked`）。
- `ResNet50BottleneckWorkload` 现在包含 stem MaxPool、四个 inference BatchNorm、四个
  Conv2D、ReLU 与 residual；micro shape 可完整生成 TISA/backend artifact。
- 回归覆盖 padding、卷积 halo、BatchNorm/Pool 任务归属与 artifact validation；全量
  回归为 94 tests passed。

当前 ResNet 语义边界：静态 NCHW/OIHW、unit dilation、feature/batch group 为 1；
pooling 使用 N/C-preserving `reduce_window`，平均池化的除法仍由后续 elementwise
operation 表达。完整 ResNet50 的 stem 7x7、stride/downsample、全层 repetition 还未
形成论文规模实验矩阵。

## 2026-08-29：GC tile candidate cost model

- `SchedulePlanner` 从单一 heuristic baseline 升级为可审计的 `cost-model-v1`，支持
  `tile_size_candidates`；候选依据 tile 数、估算计算周期、root traffic 和 local
  working-set overflow 排序，tie-break 使用较小 tile size。
- `compile_torch_module`、`compile_operator_graph` 和 CLI 新增
  `--tile-size-candidates 2,4,8`；最终只选择一份 schedule，因此 static/dynamic 仍严格
  复用同一 TISA/backend artifact。
- `ScheduleSpec` 保存 `candidate_costs`、`selected_tile_size` 和模型版本，方便把编译
  选择与后端实际周期分开分析；该模型不是 RTL timing。
- 新增 planner regression；全量回归为 96 tests passed。

## 2026-08-29：dynamic shape 失败契约

- 真实 Torch-XLA dynamic export 会产生 `get_dimension_size`、
  `dynamic_broadcast_in_dim` 和 shape-tensor dataflow，不是把 `?` 替换为整数即可完成。
- compiler 现在在 official StableHLO import 之前报告需要 shape-specialization pass，并
  列出实际动态 operation；`shape_environment` 明确只负责 Canonical symbol resolution。
- 新增真实 dynamic `torch.export` regression；symbolic/dynamic shape 仍未标记完成。

## 2026-08-30：静态 layout/broadcast 增量开始

- 提交 `bd5bfd2 align planner units and batch norm tile operands`；全量 97 tests passed。
- 确认静态 `broadcast_in_dim` 当前仍被强制为 full-tensor transform，下一步对齐 GC、FC
  与 analytical backend 的逐 tile region 语义。
- 检查过程中一次命令使用了错误工作目录 `npu-ooo_simulator`；已改回项目实际目录，未产生修改。

## 2026-08-30：静态 broadcast 输出域 tiling

- `plan_uniform_tiles` 不再把静态 `broadcast_in_dim` 误标为 full-tensor transform；广播
  按输出域保留边界 tile，普通 reshape/transpose/KV-cache 仍保持单 tile。
- TileGraph 与 FC `TileMem` 使用 `broadcast_dimensions` 将输出 tile 映射到源向量切片；
  源 singleton 轴固定读取 `[0:1]`，最后边界 tile 不越界。
- analytical transform backend 为每个广播 tile 生成独立 copy payload，并通过 root
  region overlap 建立跨算子依赖；真实 PyTorch `bias + value` 回归覆盖 12 个 tile。
- 定向与全量回归：98 tests passed；本项尚未提交，下一步补 scalar operand 语义。
