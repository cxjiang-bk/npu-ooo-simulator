# Device scheduler 周期仿真运行指南

调度器的模块边界、论文 Figure 4 对应关系以及
`RuntimeSubmission → LoadedDeviceProgram → Reception FIFO → WQ/IQ → ExecutionBackend`
架构见 [Scheduler 架构与流程](../architecture/scheduler.md)。本文聚焦如何运行和解释
`cycle_event` 模型。

`cycle_event` 将 TISA 的接收、WQ、IQ、执行、完成反馈和退休建成逐周期状态机。
外层兼容入口仍接受 `BackendArtifact + RuntimeSubmission + MachineConfig`，但 scheduler
内核只接受 loader 产生的 `LoadedDeviceProgram` 与 `ExecutionBackend` 反馈；现有
`analytical_event` 保留为事件模型基线。两种模型共享公开 descriptor/feedback 语义。

论文依据是 Section V、Figure 4 和 Algorithm 1/2。论文公开的是语义结构和整体 dispatch
开销，下面的流水寄存边界和共享总线是本项目显式选择的研究参数。
结果记录 `scheduler_calibration_status=uncalibrated`；加载 RTL payload timing 时，
`timing_calibration_status` 单独记录其校准状态，整机结果继续标记为 analytical。

## 使用

先编译一次，再固定 package 与 runtime binding 比较策略：

```bash
PYTHONPATH=src python3.12 -m npu_ooo.cli compile \
  --torch-module examples.torch_models:MultiHeadAttentionBlock \
  --input-shape 1,4,8 --input-shape 1,1,4,4 --tile-size 4 \
  --output-dir out/attention-compile

PYTHONPATH=src python3.12 -m npu_ooo.cli simulate \
  --compile-dir out/attention-compile --event-backend cycle_event \
  --scheduler-config configs/scheduler/cycle_baseline.json \
  --address-scoreboard --policy dynamic_ready_queue \
  --output-dir out/attention-cycle-dynamic

PYTHONPATH=src python3.12 -m npu_ooo.cli simulate \
  --compile-dir out/attention-compile --event-backend cycle_event \
  --scheduler-config configs/scheduler/cycle_baseline.json \
  --address-scoreboard --policy static_streams \
  --output-dir out/attention-cycle-static
```

`compile-and-sim`、`paper-matrix` 和 RuntimeSequence 也接受 `cycle_event`。
`--scheduler-config` 读取完整 pipeline 对象，省略字段取默认值；未知字段或非法值报错。
未指定该文件时，使用 MachineConfig 的 `scheduler.pipeline`。旧 machine JSON 缺少该字段
时自动补齐默认值。`SimulatorConfig.pipeline` 提供相同的 Python API 覆盖入口。

## 状态与容量

| 结构 | 容量来源 | 分配与释放 |
| --- | --- | --- |
| Reception FIFO | `instruction_queue_depth` | receive 分配，dispatch 释放 |
| 每类 EU 的 WQ | `unit.queue_depth × unit.count` | dispatch 分配，select 释放 |
| 每类 EU 的 IQ | `min(pipeline.iq_entries, ready_queue_depth)` | select 分配，issue 释放 |
| 每类 EU 的语义表 Fu | `pipeline.inflight_entries`，按 operand 条目计数 | issue 分配，完成反馈接受后释放 |
| Dynamic retire record | completion-ready 指令记录 | complete 后按完成顺序 retire |
| Ordered retire ledger | `rob_entries` 适用于 `sequential/static_pipeline` | dispatch 追加，completed ledger head retire |
| Tile window | `max_inflight_tiles` | 首条子 stage issue 占用，整 tile 所有 TISA complete 释放 |
| EU 实例 | `unit.count` | 由 ExecutionBackend 独占；issue 接受后占用，物理执行结束后释放 |

`Fu` 保存活动指令身份，operand 的范围、访问类型和依赖通过 descriptor 查得。动态策略在
select 和 issue 边界将候选与目标 resource 的本地 Fu 比较，按绑定 memory/allocation、
byte range 和 READ/WRITE 方向检测保守 RAW/WAR/WAW SemanticConflict。
一条 TISA 绑定一个具体 EU 资源类别。scheduler 只调用 can-accept/issue；payload 内部
primitive 顺序、实例 busy/II、task trace 和 physical-done 都由 ExecutionBackend 实现，
不会在 scheduler 中维护第二份 EU 状态。`pipeline_depth` 在 analytical execution backend
中尚未启用同一实例的多 TISA 流水重叠。

`receive_width/dispatch_width/select_width/issue_width/completion_width/retire_width`
均为 Dynamic 全局每周期上限；issue 同时受 `unit.issue_width` 的资源类别上限约束。
`dependency_window` 是每个 WQ 的扫描窗口。IQ、Fu 在同一类 EU 实例间共享。
`static_streams` 另使用 `control_width`（默认 1）以及 `control_latency`、`wait_latency`、
`fence_latency`（默认均为 1）；三种 latency 可显式设 0 做隔离实验，但默认不假设免费同步。
Static dispatch 为每个逻辑 EU 独立维护 WQ，每周期向每个 EU 的 WQ 放入一条编译 stream
预期指令。

## 时钟边界

周期 C 的处理顺序固定为：

```text
retire → complete → wakeup → issue → select → dispatch → receive
```

逆向处理流水线保证新接收/新选出的条目经过寄存边界。issue 释放 IQ 后 select 可以补入。

| 事件 | 最早周期 |
| --- | --- |
| receive | `ceil(descriptor availability + runtime launch)`，受接收宽度/容量限制 |
| dispatch | receive 后一周期，受 WQ/dispatch width 限制 |
| wakeup | completion-tag 广播清空 pending mask 后，`max(dispatch + dispatch_latency, 各 dependency feedback + wakeup_latency)` |
| select | wakeup 当周期，受窗口、本地 Fu SemanticConflict、地址依赖、IQ 容量和 select width 限制 |
| issue | `select + select_latency`，重新验证 SemanticConflict，并受 EU/Fu/tile/memory 资源和 issue width 限制 |
| execution done | `ceil(issue + payload duration)` |
| complete | `execution done + completion_latency`，受 completion width 仲裁限制 |
| retire | Dynamic 为 `complete + retire_latency`，受 retire width 限制；ordered policies 采用 ledger 队首 |

默认 receive@0 的独立指令会 dispatch@1、select@2、issue@3。若 duration=2，则
done/complete@5、retire@6。其 RAW 消费者默认 wakeup@6、select@6、issue@7。
`dispatch_latency/select_latency/retire_latency` 至少为 1；`wakeup_latency` 与
`completion_latency` 可为 0。同一时刻的 trace 保留实际阶段处理顺序。

完成总线按 `(execution done cycle, descriptor order)` 仲裁。物理 EU 可以在反馈等待期间
再次执行。Fu 持有对应条目直到 complete；Dynamic retire record 在 completion-ready 时
推进，ordered policies 的 retirement ledger 持有 descriptor order record 直到 retire。
执行效果在 complete 阶段生效。

`TISA_COMPLETE` 广播完整 readiness condition，`TISA_PARTIAL_READY` 广播对应 partial
condition；广播只携带 completion identity/condition，不传输 payload 数据。全部 per-EU WQ
会 snoop tag 并清除匹配 bit。该互连是项目 correctness-first 选择，论文只规定 dependent
notification，没有公开 Epoch 的 broadcast/scoreboard 实现。

## 调度与依赖

- `static_pipeline` 消费 `StaticSchedulePlan`，按本次 runtime 已选择的提交顺序 admission；前一条 issue
  后下一条可在不同 EU 上重叠。计划显式记录 resource、dependency token 和 reservation。
  这是旧“全局下一条 + 跨 EU overlap”兼容基线，不声称复现作者未公开的静态排程器。
- `static_streams` 消费编译期 `StaticControlProgram`。descriptor 先进入共享 Reception FIFO，
  再按编译 stream 路由到各 EU WQ；每个 EU 只推进自己的 WQ 和 head command。wait/fence
  只阻塞当前流，set 等待真实 execution feedback，控制时序属于未校准项目假设。
- `dynamic_ready_queue` 只在已接收 descriptor 的各 WQ 有界窗口中选择 ready 条目。默认
  `oldest_first` 使用接收队列年龄；`compiler_hint` 使用 descriptor 显式 hint；
  `oracle_critical_path` 使用完整图，仅作为离线参考。旧名称 `critical_path` 是该 oracle 的
  兼容别名，不称为论文在线自适应算法。
- `sequential` 在上一条退休后才 issue 下一条。
- 所有显式依赖均等待对应 complete；`payload_ready:<task_id>` 使用指定 primitive 的
  完成边界，加 completion/wakeup latency 唤醒。partial-ready 通知是独立 sideband 原型，
  不消耗 full-completion 总线宽度，也不模拟逐子区域有效位。
- 编译器提供语义/复用依赖；target memory planner 与 runtime loader 都按不相交物理区间
  维护 last-writer/readers frontier：read 只等待最后 writer，write 等待最近 writer 或自
  上次 writer 以来的 reader 集合，不再连接全部历史冲突。runtime 地址绑定后将该 frontier
  转换为显式 RAW/WAR/WAW completion-token dependency；设备 `--address-scoreboard` 只比较
  已经接收、较老且未 complete 的 descriptor。未解析布局仍使用保守区间。
- `--memory-bank-scoreboard` 在整个物理 payload 执行期间保留 bank/port；这是保守的结构
  资源模型，逐次 SRAM/DRAM transaction timing 由后续 memory backend 提供。

若 runtime 的预排序会让消费者先于 loader 新增的 alias producer 到达，loader 会在设备
执行前明确拒绝，配置资源不足产生明确诊断；`max_cycles`
为有时间进展但长期无法完成的运行提供上限。

## 输出与统计

`summary.json` schema v2 默认只保存聚合指标与 timing：

- `instruction_pipeline`：每条指令的 received/dispatched/wakeup/selected/issued/done/
  completed/retired 周期；
- `wq_peak/iq_peak/fu_peak`：各级队列与 Fu 峰值；
- `rob_peak/retirement_backlog_peak/completed_retirement_backlog_peak`：
  `sequential/static_pipeline` 的 ordered retire ledger 指标；
- Dynamic 使用 completion-ready retire record，metrics、stall reason 和
  `queue_occupancy_timeline` 只包含 Dynamic 架构实际存在的状态；逐周期
  `queue_occupancy_timeline` 可在进程内结果中检查；
  默认 summary 保存聚合指标；
- `stall_cycles`：每种原因发生的周期数；同周期多条指令计 1；
- `stall_instruction_cycles`：按 `(instruction, reason, cycle)` 去重后计数；
- `compile_package_sha256`：本次消费的 BackendArtifact 规范 JSON hash；
- `trace.event_counts/stall_interval_count`：未嵌入 summary 的 trace 聚合计数；
- `offchip_read/write/total_bytes`：由 execution backend 按 payload 的 root-memory region
  统计，避免 scheduler 反向读取 task；
- `device_finish_cycle` 为最终 retire，`completion_finish_cycle` 为最终 complete，
  `retirement_drain_cycles` 为差值；`total_cycles` 包含 runtime synchronization。

Dynamic 原因包括 reception/WQ/IQ full、dependency wait、semantic conflict、FU busy、
Fu table full、tile window、address hazard、bank/port conflict、completion bandwidth、
retire backpressure 和各阶段带宽。`sequential/static_pipeline` 额外统计 `rob_full`。
多种阻塞可以在同一周期发生，因此各项 `stall_cycles` 不应相加解释为总运行周期。
WQ 中达到 dispatch_latency 边界的条目参与 dependency stall 统计；select 仅扫描窗口。
流水固定延迟单独由生命周期周期反映。

`tisa_instructions.csv` 增加 receive、dispatch、wakeup、select、execution_done、complete、
retire 列。连续的同一 `(instruction, reason, stage)` stall 压缩成一条
`TISA_STALL [start,end,duration]`；stall 只携带 reason、stage 和 dependency count，不复制
完整依赖。Perfetto 将其保存为 duration event；完整 bound dependency 仍在
`05_runtime/bound_device_program.json`，其他生命周期事件可携带依赖 provenance。
RuntimeSequence 将生命周期周期平移并按 invocation id 区分；stall 和退休数按全部
invocation 汇总。

主要验收位于 `tests/test_cycle_scheduler.py`：使用 1–6 条 TISA 的手算时间线覆盖依赖、
并行、窗口、所有主要反压、反馈/退休、部分就绪、物理到达、独立 CLI 和 sequence。

## GM 随机延迟

gm_latency_trace 是仿真阶段的 task-level GM 延迟模型。它为每个触及 GM 的 execution
task 记录一条固定的 extra_latency_cycles，仿真时将该值叠加到 analytical task duration。
其他 task 沿用原有 duration。模型以 extra_latency_probability 控制非理想事件比例，
再从均匀整数区间采样额外延迟；trace 是可重复回放的仿真输入。

先从已经完成的 compile package 生成 trace：

~~~bash
PYTHONPATH=src python3.12 -m npu_ooo.cli generate-gm-latency \
  --compile-dir out/attention-compile \
  --output out/gm-latency-seed42.json \
  --seed 42 --extra-latency-probability 0.1 \
  --min-extra-latency-cycles 1 --max-extra-latency-cycles 40
~~~

生成命令读取 compile package 并写出独立 JSON，trace 生命周期属于 simulation input。编译阶段
使用预期 task duration 和依赖生成 set/wait/fence，静态控制程序的命令顺序保持稳定。requests
以 task ID 为键；正常请求记录 extra_latency_cycles=0，非理想请求记录采样值，因此
Static Streams 与 Dynamic 使用相同请求时会得到相同的 intrinsic latency。

仿真时显式选择 provider，并让两种策略共享同一 trace：

~~~bash
for policy in static_streams dynamic_ready_queue; do
  PYTHONPATH=src python3.12 -m npu_ooo.cli simulate \
    --compile-dir out/attention-compile --event-backend cycle_event \
    --timing-provider gm_latency_trace \
    --timing-config out/gm-latency-seed42.json \
    --runtime-policy static --policy "$policy" \
    --output-dir "out/attention-gm-$policy"
done
~~~

trace JSON 保存 artifact_id、program_id、machine topology hash、采样分布、非理想请求数和每条请求的
GM 读写字节数，便于审计输入。当前模型覆盖 task 级额外延迟；DDR bank、channel、row buffer
和 refresh 状态机属于后续模型范围。

## 公开设备契约

```text
RuntimeSubmission + final TISA + MemoryPlan
  -> runtime.loader
  -> BoundTISADescriptor + DescriptorEnvelope + StaticSchedulePlan
  -> scheduler --IssueRequest--> ExecutionBackend
  <- scheduler <- ExecutionFeedback(execution_done / partial_ready)
```

completion token 包含 invocation id，重复调用保持独立。`execution_done` 是物理结束；
cycle scheduler 之后仍经过 completion latency/width 仲裁才标记 complete，再经 wakeup latency
唤醒依赖，Dynamic 按 completion-ready 顺序 retire，ordered policies 按 ledger 顺序 retire。
execution backend 拒绝 issue 时 scheduler 不产生假 issue。
模块边界测试禁止 event/cycle scheduler 读取 `BackendArtifact.execution_graph` 或 payload map。

## Scheduler profile

scheduler-config 为 cycle_event 提供两种 JSON 形式：

- Pipeline JSON 直接使用 SchedulerPipelineConfig 字段。[cycle baseline](../../configs/scheduler/cycle_baseline.json) 是标准模板，省略字段采用 pipeline 默认值。
- Profile JSON 包含 schema_version、profile、capacities 和可选 pipeline。capacities 仅调整 scheduler 仿真结构容量，编译包的机器拓扑保持一致；profile 省略 pipeline 时沿用编译包 machine.scheduler.pipeline，显式提供 pipeline 时采用该次运行的控制流水参数。

Profile 容量字段与仿真结构一一对应：

| 字段 | 仿真结构 |
| --- | --- |
| instruction_queue_depth | Reception FIFO 容量 |
| rob_entries | `sequential/static_pipeline` 的 ordered retire ledger 容量 |
| max_inflight_tiles | tile window 容量 |
| dependency_window | 固定 WQ 扫描窗口 |
| ready_queue_depth | IQ 容量上限 |

仓库提供三档容量 profile 用于受控敏感性分析：

- [dynamic_window_narrow](../../configs/scheduler/dynamic_window_narrow.json)：Reception 8、tile window 4、WQ window 4、IQ 8；文件保留 ordered policy 的兼容容量。
- [dynamic_window_baseline](../../configs/scheduler/dynamic_window_baseline.json)：Reception 32、tile window 8、WQ window 8、IQ 32；文件保留 ordered policy 的兼容容量。
- [dynamic_window_wide](../../configs/scheduler/dynamic_window_wide.json)：Reception 64、tile window 16、WQ window 16、IQ 64；文件保留 ordered policy 的兼容容量。

先完成一次 compile，再让所有 profile 消费同一编译包：

~~~bash
for profile in dynamic_window_narrow dynamic_window_baseline dynamic_window_wide; do
  PYTHONPATH=src python3.12 -m npu_ooo.cli simulate \
    --compile-dir out/attention-compile \
    --event-backend cycle_event \
    --scheduler-config configs/scheduler/$profile.json \
    --policy dynamic_ready_queue \
    --dynamic-priority oldest_first \
    --output-dir out/attention-$profile
done
~~~

每个容量字段按 CLI capacity > profile capacities > compile machine scheduler defaults 生效。profile 负责本次运行的 scheduler 容量和可选控制流水，编译包继续提供机器资源、placement、transfer path 和 payload timing，从而让 Static Streams 与 Dynamic 的参数对照保持同一硬件基础。
