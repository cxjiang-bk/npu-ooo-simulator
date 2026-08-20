# 总体架构

## 1. 系统边界

本项目研究的是 NPU 上的 tile-level scheduling，而不是仅做 loop mapping，也不是第一步就复刻某一款 NPU 的 RTL。

```text
Model/Benchmark Case
      |
      v
Model IR / Graph Template
      |
      v
Operator Graph IR -------- Framework/StableHLO bridge
      |
      v
Schedule/Tiling IR -------- Mapping search (optional)
      |
      v
Tile Instance IR
      |
      v
Primitive Execution Graph <-------- MachineConfig
      |                                    |
      +----------> SchedulerPolicy <-------+
                         |
                         v
               Discrete-event Simulator
                         |
                         v
                 Trace + Cycle Summary
```

Model 层决定 workload 的拓扑、重复结构、shape environment 和 execution phase；Operator 层决定单个算子的数学语义；Mapping 层决定 tile 如何切分、循环顺序和驻留；execution 层决定具体 tile 在哪些资源上、何时执行。四层不能混为一谈。

论文中的 `Framework bridge -> Graph compiler -> Fusion compiler -> TISA generator -> backend` 对应本项目的 Model import、Operator Graph、Schedule/Tiling、Tile/TISA 和 simulator/backend。

## 1.1 Model IR：为什么需要这一层

只拥有 Operator Graph 不足以复现论文的 benchmark 表。以下信息不属于单个 operator：

- 模型家族、版本和 block template；
- block 重复次数、权重共享和跨层连接；
- batch、sequence length、image resolution、hidden/head dimensions；
- inference phase：`train`、`prefill`、`decode`；
- KV cache、causal mask、position state 和其他 runtime state；
- dtype、quantization、layout 和 benchmark warmup/repetition；
- optional routing，例如 MoE token dispatch 和 expert capacity。

因此 Model IR 不保存完整权重，也不把每一层无条件展开成巨型图，而是同时支持 `GraphTemplate` 和 `GraphInstance`：模板表达重复 block，实例化参数表达当前 benchmark case。

一个最小 Model IR 可以表示为：

```text
ModelSpec {
  name, family, version
  shape_env, dtype, layout
  execution_phase
  graph_templates, top_level_nodes
  persistent_state
}

BenchmarkCase {
  model_id
  evaluation_scope: one_block | layer | full_model
  batch, sequence_length, image_size
  phase: prefill | decode | train
  precision, quantization
  model_config_id, architecture_config_id
}
```

`ModelSpec` 经过实例化后才生成 Operator Graph IR。这样同一个 LLaMA block 可以复用于不同 sequence length、batch 和 prefill/decode case。

## 2. 五层 IR

### 2.1 Model/Benchmark IR

描述 workload 的模型级语义：

```text
model family and version
block templates and repetition
shape environment and execution phase
state/cache/parameter metadata
benchmark case and measurement protocol
```

首批 model family：

- `cnn_residual`：ResNet50；
- `encoder_transformer`：BERT-Base；
- `decoder_transformer`：GPT-J、LLaMA2；
- `decoder_reasoning`：DeepSeek-R1-16B；
- `moe_decoder`：作为可选扩展，不对 DeepSeek-R1 是否使用 MoE 做未经证实的假设。

### 2.2 Operator Graph IR

描述计算语义，不包含具体硬件：

```text
Tensor: name, shape, dtype, layout
Operator: type, inputs, outputs, iter dims, reduce dims, attributes
Edge: producer, consumer, tensor
```

首批 operator type：

- `matmul`；
- `elementwise`；
- `reduce`；
- `softmax`，先作为可展开的 composite op；
- Conv2D 只预留接口，后续再处理 halo、padding 和 layout。

### 2.3 Schedule/Tiling IR

描述编译期 mapping 决策：

```text
loop order
tile factors
temporal/spatial mapping
fusion/stage boundaries
tensor residency
buffer assignment
explicit child dependencies
```

这一层允许手写 schedule，也允许以后接 TileFlow/Timeloop 或其他 mapper。

### 2.4 Tile Instance IR

把 schedule 中的切分规则展开成实际运行实例：

```text
tile_id
operator_id
iteration coordinates
logical bounds
tile shape
input/output regions
stage_id
program order
```

`M=32` 是 schedule factor；`M=[64,96)` 才是 tile instance。边界 tile 必须保留实际 shape。

### 2.5 Primitive Execution Graph

一个 operator tile 根据 MachineConfig lower 成若干 primitive task：

```text
task_id
tile_id
opcode/primitive
resource_class
source/target buffers
read/write address ranges
latency request
predecessors
program-order tag
```

例如 Matmul tile 可以 lower 为：

```text
load-left -> load-right -> mxu-compute -> psum-update -> store-output
```

具体是否需要某一步由架构 profile 和 buffer residency 决定，而不是由 scheduler 猜测。

## 3. 参数化 MachineConfig

MachineConfig 是 TileFlow emitter、operator lowering 和 simulator 的共同事实来源。

### 3.1 Memory

每个 memory level 至少包含：

```text
name, parent, capacity_bytes
read/write bandwidth
read/write latency
read/write ports
bank count, bank width
alignment
```

不预设固定命名。以下两种都应该可表达：

```text
DRAM -> SRAM -> RF
GM -> UB -> LMB/RMB/PMB/PSB/ARB
```

### 3.2 Execution Unit

每类 unit 至少包含：

```text
name, count, supported_ops
queue_depth, issue_width
pipeline_depth
latency model
initiation_interval
```

`latency` 与 `initiation_interval` 必须分离，以表达流水化单元连续启动不同 tile 的能力。

### 3.3 Transfer Path

数据通路由配置描述：

```text
source, target, engine
channel_count
bandwidth, setup_latency
optional transform and transform_latency
overlap capability
```

不能在代码里固定 `_route(GM, UB) -> GDMA`。当前 LPU 路径只是一份 profile。

### 3.4 Scheduling Capacity

动态调度实验需要显式配置：

```text
global instruction queue depth
per-resource queue depth
ROB/out-of-order window entries
maximum in-flight tiles
dependency tracking entries
issue width
```

这些参数必须进入 experiment manifest，否则无法解释动态调度收益来自何处。

## 4. Operator Lowering Registry

每个算子 lowering 插件负责：

1. 计算 tile 的输入输出 region；
2. 根据 residency 生成 transfer；
3. 生成 compute/reduce primitive；
4. 生成 tile 内依赖；
5. 请求 MachineConfig latency model。

Scheduler 只消费统一的 ExecutionTask，不包含算子专用分支。

当前 registry 已支持 `matmul/batched_matmul/gemv`、`elementwise/residual_add`、`reduce`、`softmax`、`rmsnorm` 和 `layernorm`。`lower_mixed_graph` 对 heterogeneous graph 按拓扑逐算子调用插件，再将每个插件的任务合并为一个 ExecutionGraph：

```text
Operator A store(root, tensor T)
       -- DataEdge(T) + BufferRegion overlap -->
Operator B load(root, tensor T)
```

这是一版保守的跨算子 handoff，明确要求上游写回 root memory、下游从 root memory 读取。它适合先比较调度策略，避免 scheduler 猜测 tensor 地址；后续 mapping/residency 优化可以替换为 local-memory handoff 或真正的 fusion lowering，而不改动 policy 接口。

首批 lowering 顺序：

```text
Matmul -> 2mm -> Elementwise -> Reduce/Softmax -> Attention
```

## 5. 依赖模型

Execution Graph 同时保留三类依赖：

- 显式数据流依赖：producer tile 到 consumer tile；
- 编译期顺序约束：stage、barrier、固定 program order；
- 地址范围依赖：RAW、WAR、WAW。

第一版先根据精确 tensor region 建立依赖。TISA-like 动态后端再把地址/range metadata 放进 dependency window 或 scoreboard，在运行时唤醒 ready task。

## 6. SchedulerPolicy

所有 policy 接受同一个 Execution Graph 和 MachineConfig。

### 6.1 Sequential

按 iteration-major/program order 执行，前一个 iteration 完成后才推进下一个，作为最保守基线。

### 6.2 StaticPipeline

编译期确定：

```text
task order
resource reservation
stage offset
optional modulo initiation interval
```

支持 dual-stage 和 triple-stage，但 stage 名称来自 graph，不写死 `M0/S/M1`。`StaticPipelineConfig` 可用 `stage_offsets + initiation_interval_cycles + task.attributes[iteration]` 生成 reservation，也可用 `task_issue_cycles` 精确指定每条指令的 issue cycle。没有 reservation 配置时，`static_pipeline` 保持 deterministic program-order list scheduling，便于和旧基线对照。

### 6.3 DynamicReadyQueue

运行时维护 ready queue、resource queue 和 in-flight window。候选 priority 至少包括：

- oldest-first；
- critical-path-first；
- resource-locality-first；
- iteration-first，作为静态顺序近似对照。

当前 simulator 将可执行的动态启发式显式化为 `SimulatorConfig.dynamic_priority`：`critical_path`（默认）或 `oldest_first`。这使得 softmax 等多阶段 DAG 可以把“动态 ready queue 机制”和“具体优先级函数”作为两个独立实验维度报告。

TISA-like policy 现在支持可选 runtime range scoreboard、窗口大小和 completion wake-up；scoreboard 不改写编译期图，而是只追踪 active task 的地址范围，因此能在 trace 中区分编译期依赖与运行时 address stall。

## 7. Discrete-event Simulator

核心事件：

```text
ISSUE
START
COMPLETE
WAKE_UP
STALL_BEGIN / STALL_END
```

Simulator 维护：

- resource instance 与 next-issue time；
- resource queue occupancy；
- task dependency count；
- in-flight/ROB state；
- buffer occupancy 和 live allocation；
- event priority queue；
- deterministic tie-break order。

每个 task 的开始时间由依赖、资源、queue、initiation interval 和 buffer 状态共同决定。

当前实现将这一层拆成两个接口：

```text
SchedulerPolicy
  -> 从 visible ready queue 选择 task

TimingModel / EventBackend
  -> 计算 issue/start/complete
  -> 维护 unit queue、II、ROB、dependency window、tile window
  -> 在 COMPLETE 时唤醒后继 task
```

`AnalyticalTimingModel` 只消费 `ExecutionTask.duration_cycles` 和 `MachineConfig` 的 unit 默认值；它是可替换的 timing provider，不代表真实硬件。`SimulatorConfig` 可以覆盖 instruction queue、ROB、ready queue、dependency window 和 max in-flight tiles。

`--address-scoreboard` 启用 runtime range scoreboard：根据相同 `tensor/memory` 上的 `BufferRegion` 重叠关系，在 issue 前检查 active task 并产生 RAW/WAR/WAW stall；COMPLETE 后释放范围并继续调度。当前地址来自 canonical `ExecutionTask` metadata，尚不是从真实 TISA binary 动态解析出的硬件 scoreboard。

## 8. Trace 与结果

一次仿真至少输出：

```text
tasks.csv
summary.json
manifest.json
perfetto.json
swimlane.svg
operator_graph.json / operator_graph.svg
tile_graph.json
execution_graph.json
```

`events.csv` 使用 cycle-native 字段：

```text
task_id, tile_id, operator, iteration
resource, resource_instance
issue_cycle, start_cycle, end_cycle
wait_reason, predecessors
```

`trace.json` 使用 Chrome Trace Event `X` 事件，可直接在 Perfetto 中按 resource/resource instance 展示。

`summary.json` 至少包含：

```text
total_cycles
speedup_vs_baseline
resource_utilization
stall_cycles_by_reason
pipeline_drain_cycles
queue_occupancy_timeline
queue_peak_occupancy
buffer_peak_usage
completed_tile_count
```

## 9. 公平比较约束

Static 与 Dynamic 对比时，以下内容必须一致：

- workload 和 tensor layout；
- tile decomposition；
- buffer allocation；
- tensor address/range；
- dependency graph；
- latency model；
- hardware resource count；
- simulation seed 和 tie-break rule。

实验只切换 SchedulerPolicy。若同时改变 tile 或硬件，结果必须作为独立实验维度报告。

## 10. 校准边界

第一版是 architecture exploration model。验证分三层：

1. 手算 micro-case：验证 scheduler/simulator 语义；
2. TileFlow/SCALE-Sim：对账计算量、搬运量和 aggregate 趋势；
3. Verilator/RTL/hardware counters：校准 latency、queue、bank conflict 和 observed cycle。

只有第三层完成后，相关结果才可标记为 RTL-observed 或 cycle-accurate。
