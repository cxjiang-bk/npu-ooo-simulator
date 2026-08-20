# 实施路线图

## 总体策略

先建立一个小而完整、可手算验证的 2mm 闭环，再逐步增加算子、动态依赖和真实硬件细节。每个阶段都必须产生可独立验收的 artifact，避免先写完整 compiler 再发现 timing 语义不可解释。

## 阶段 0：契约冻结

目标：确定 MachineConfig、Model/Benchmark IR、Operator/Schedule/Tile/Execution IR、Trace 和 Experiment manifest 的 schema。

交付物：

- MachineConfig schema 与校验规则；
- Model/Benchmark、Operator/Schedule/Tile/Execution IR 字段定义；
- scheduler policy 接口；
- event/summary/manifest schema；
- 两个手算 pipeline 示例。

验收：给定一张 dual-stage/triple-stage DAG，能够仅根据文档人工推导预期时间线；所有字段都能解释论文图中的 iteration、stage、尾部 drain 和总周期。

## 阶段 1：MachineConfig、Model IR 与基础 Operator IR

目标：实现通用架构配置、Model/Benchmark IR 和 Operator Graph IR。

首批 profile：

- `minimal`: 单 DMA、单 MXU、简单 SRAM；
- `lpu-derived`: 从已有 LPU 参数导入，但不依赖其 RTL 运行环境；
- `wide-mxu`: 修改 MXU 数量/shape/II 的探索 profile。

验收：

- profile 可序列化、校验并生成稳定 hash；
- ModelSpec 支持 graph template、重复 block、shape environment 和 prefill/decode phase；
- Table IX 的每个 model/config/phase 组合可以表达为独立 BenchmarkCase；
- Operator Graph 保存 model/layer/template provenance；
- 非法 memory parent、path 和 resource 引用在仿真前失败；
- 修改 queue、bandwidth、MXU shape 不需要修改 simulator 代码。

## 阶段 2：2mm Tile Graph

目标：跑通 `Model/Benchmark -> Operator Graph -> Schedule -> Tile Instance -> Execution Graph`。

交付物：

- Matmul lowering；
- 2mm benchmark；
- tile region/address 计算；
- DMA/MXU primitive task；
- RAW producer-consumer 依赖。

验收：

- tile 数量、边界和依赖可手算；
- task ID 和拓扑顺序可重复；
- aggregate MAC 和 transfer bytes 与参考模型对账；
- compiler 输出 execution graph JSON，尚不要求运行动态调度。

## 阶段 3：Static Simulator 与泳道图

目标：实现确定性离散事件 simulator 和静态基线。

策略：

- Sequential；
- Static dual-stage pipeline；
- Static triple-stage pipeline。

交付物：

- event engine；
- resource/queue/II model；
- events CSV、summary JSON、Perfetto trace；
- 论文风格静态泳道图。

验收：

- 串行、双资源重叠、队列满、pipeline drain micro-test 与手算一致；
- 2mm 在不同 bandwidth/MXU profile 下周期变化可解释；
- 静态泳道能显示资源空泡和等待原因。

## 阶段 4：Dynamic/TISA-like Scheduler

目标：在相同 execution graph 上加入动态 ready queue 和 completion wake-up。

递进实现：

1. dependency-ready + resource-ready；
2. configurable out-of-order window；
3. oldest/critical-path/locality priority；
4. address-range RAW/WAR/WAW scoreboard；
5. queue/ROB/backpressure。

验收：

- 复现 sequential/static dual/dynamic dual/static triple/dynamic triple 五种图；
- Dynamic 不越过真实数据依赖；
- window=1 退化为近似 in-order；
- 扩大 window 的收益和代价能从 trace/queue 指标解释；
- Static/Dynamic 只切换 policy，不改变 graph 或 latency。

## 阶段 5：模型与算子覆盖扩展

目标：验证后端不是 2mm 专用 simulator，并覆盖论文 benchmark 所需的模型结构。

顺序：

1. Elementwise/Reduce（elementwise/residual-add、row-reduce 已完成 lowering 闭环）；
2. ResNet bottleneck：Conv2D/Norm/Activation/Residual/Pooling；
3. Softmax/LayerNorm/RMSNorm composite lowering（softmax、LayerNorm 和 RMSNorm 已展开 composite primitive）；
4. Decoder block：已先接入 `RMSNorm -> Matmul -> ResidualAdd` 混合 fragment；下一步扩展 QKV/Attention/MLP/RoPE/KV cache；
5. BERT/GPT-J/LLaMA2/DeepSeek benchmark templates；
6. Conv2D halo/layout 和 optional MoE routing 作为后续扩展。

验收：每种 P0 semantic operator 都有独立 lowering、micro-test 和至少一组 Static/Dynamic trace；模型层能实例化 CNN、encoder 和 decoder template；scheduler 中无模型或算子名称分支。

## 阶段 6：实验框架

目标：系统比较架构与调度参数。

实验矩阵：

```text
architecture profile
  x benchmark shape
  x tiling schedule / tile size
  x scheduler policy
  x queue/window setting
```

交付物：

- 一条命令运行对比实验；
- `sweep-two-mm` 批量扫描 architecture × tile size × policy × window × ROB；
- `sweep-workloads` 已能扫描多个算子/模型 fragment × architecture × tile size × policy × window × ROB；
- 每个 case 独立 manifest；
- 汇总 CSV/JSON；
- 单 workload CLI 已统一输出 SVG 和 PNG 泳道图；
- total cycle、speedup、utilization、stall、drain、buffer/queue peak；
- 可复现的泳道图目录。

## 阶段 7：外部模型与硬件校准

目标：逐步提升 timing model 的可信度。

- TileFlow/Timeloop 对账 mapping、traffic 和 aggregate trend；
- SCALE-Sim 校准 MXU dataflow 和 systolic timing；
- VTA/Gemmini 对照 queue/dependency 语义；
- Verilator/RTL trace 校准 unit latency、II、buffer port/bank conflict；
- 最终接入当前 NPU ISA/profile。

任何未经 RTL/hardware observation 校准的数字，manifest 中保持 `analytical` 或 `source-derived` 状态。

## 第一里程碑定义

第一里程碑完成条件：

```text
2mm benchmark
+ 2 个 architecture profile
+ sequential/static dual/dynamic dual
+ total cycle 和 stall breakdown
+ Perfetto trace 和 PNG swimlane
+ 手算 micro-test
+ 可复现实验 manifest
```

这是后续 Attention/TISA 复现的稳定底座。
