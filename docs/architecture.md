# 总体架构

## 1. 系统边界

本项目研究的是 NPU 上的 tile-level scheduling，而不是仅做 loop mapping，也不是第一步就复刻某一款 NPU 的 RTL。

```text
Operator Graph
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

Mapping 层决定 tile 如何切分、循环顺序和驻留；execution 层决定具体 tile 在哪些资源上、何时执行。两层不能混为一谈。

## 2. 四层 IR

### 2.1 Operator Graph IR

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

### 2.2 Schedule/Tiling IR

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

### 2.3 Tile Instance IR

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

### 2.4 Primitive Execution Graph

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

支持 dual-stage 和 triple-stage，但 stage 名称来自 graph，不写死 `M0/S/M1`。

### 6.3 DynamicReadyQueue

运行时维护 ready queue、resource queue 和 in-flight window。候选 priority 至少包括：

- oldest-first；
- critical-path-first；
- resource-locality-first；
- iteration-first，作为静态顺序近似对照。

后续 TISA-like policy 加入 range scoreboard、窗口大小和 completion wake-up。

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

## 8. Trace 与结果

一次仿真至少输出：

```text
events.csv
summary.json
manifest.json
trace.json
swimlane.png
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
