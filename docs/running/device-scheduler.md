# Device scheduler 周期仿真运行指南

调度器的模块边界、论文 Figure 4 对应关系以及
`RuntimeSubmission → LoadedDeviceProgram → Reception FIFO → ROB/WQ → IQ → ExecutionBackend`
架构见 [Scheduler 架构与流程](../architecture/scheduler.md)。本文聚焦如何运行和解释
`cycle_event` 模型。

`cycle_event` 将 TISA 的接收、WQ、IQ、执行、完成反馈和退休建成逐周期状态机。
外层兼容入口仍接受 `BackendArtifact + RuntimeSubmission + MachineConfig`，但 scheduler
内核只接受 loader 产生的 `LoadedDeviceProgram` 与 `ExecutionBackend` 反馈；现有
`analytical_event` 保留为事件模型基线。两种模型共享公开 descriptor/feedback 语义。

论文依据是 Section V、Figure 4 和 Algorithm 1/2。论文公开的是语义结构和整体 dispatch
开销，下面的流水寄存边界、共享总线和 ROB 规则是本项目显式选择的研究参数。
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
  --address-scoreboard --policy static_pipeline \
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
| ROB | `rob_entries`，按 TISA 指令计数 | dispatch 分配，按 descriptor 提交顺序 retire |
| Tile window | `max_inflight_tiles` | 首条子 stage issue 占用，整 tile 所有 TISA complete 释放 |
| EU 实例 | `unit.count` | 由 ExecutionBackend 独占；issue 接受后占用，物理执行结束后释放 |

`Fu` 保存活动指令身份，operand 的范围、访问类型和依赖通过 descriptor 查得。
一条 TISA 绑定一个具体 EU 资源类别。scheduler 只调用 can-accept/issue；payload 内部
primitive 顺序、实例 busy/II、task trace 和 physical-done 都由 ExecutionBackend 实现，
不会在 scheduler 中维护第二份 EU 状态。`pipeline_depth` 在 analytical execution backend
中尚未启用同一实例的多 TISA 流水重叠。

`receive_width/dispatch_width/select_width/issue_width/completion_width/retire_width`
均为全局每周期上限；issue 同时受 `unit.issue_width` 的资源类别上限约束。
`dependency_window` 是每个 WQ 的扫描窗口。IQ、Fu 在同一类 EU 实例间共享。
`static_streams` 另使用 `control_width`（默认 1）以及 `control_latency`、`wait_latency`、
`fence_latency`（默认均为 1）；三种 latency 可显式设 0 做隔离实验，但默认不假设免费同步。

## 时钟边界

周期 C 的处理顺序固定为：

```text
retire → complete → wakeup → issue → select → dispatch → receive
```

逆向处理流水线保证新接收/新选出的条目经过寄存边界。容量释放可供同周期后面的阶段使用：
如 retire 释放 ROB 后 dispatch 可以补入，issue 释放 IQ 后 select 可以补入。

| 事件 | 最早周期 |
| --- | --- |
| receive | `ceil(descriptor availability + runtime launch)`，受接收宽度/容量限制 |
| dispatch | receive 后一周期，受 WQ/ROB/dispatch width 限制 |
| wakeup | `max(dispatch + dispatch_latency, 各 dependency feedback + wakeup_latency)` |
| select | wakeup 当周期，受窗口、地址依赖、IQ 容量和 select width 限制 |
| issue | `select + select_latency`，受 EU/Fu/tile/memory 资源和 issue width 限制 |
| execution done | `ceil(issue + payload duration)` |
| complete | `execution done + completion_latency`，受 completion width 仲裁限制 |
| retire | `complete + retire_latency`，受 ROB 队首和 retire width 限制 |

默认 receive@0 的独立指令会 dispatch@1、select@2、issue@3。若 duration=2，则
done/complete@5、retire@6。其 RAW 消费者默认 wakeup@6、select@6、issue@7。
`dispatch_latency/select_latency/retire_latency` 至少为 1；`wakeup_latency` 与
`completion_latency` 可为 0。同一时刻的 trace 保留实际阶段处理顺序。

完成总线按 `(execution done cycle, descriptor order)` 仲裁。物理 EU 可以在反馈等待期间
再次执行，而 Fu/ROB 持有对应条目直到 complete/retire。ROB 表示有序完成记账，不模拟
CPU 推测执行、异常回滚或将内存写入延迟至退休。

## 调度与依赖

- `static_pipeline` 消费 `StaticSchedulePlan`，按本次 runtime 已选择的提交顺序 admission；前一条 issue
  后下一条可在不同 EU 上重叠。计划显式记录 resource、dependency token 和 reservation。
  这是旧“全局下一条 + 跨 EU overlap”兼容基线，不声称复现作者未公开的静态排程器。
- `static_streams` 消费编译期 `StaticControlProgram`。每个 EU 流只推进自己的 head command；
  wait/fence 只阻塞当前流，set 等待真实 execution feedback。它不调用 Dynamic 的隐藏
  dependency-ready 检查来替编译器补 wait，控制时序属于未校准项目假设。
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
执行前明确拒绝，而不是让有限 ROB 死锁。配置资源不足同样产生明确诊断；`max_cycles`
为有时间进展但长期无法完成的运行提供上限。

## 输出与统计

`summary.json` schema v2 默认只保存聚合指标与 timing：

- `instruction_pipeline`：每条指令的 received/dispatched/wakeup/selected/issued/done/
  completed/retired 周期；
- `wq_peak/iq_peak/fu_peak/rob_peak`：队列和表项峰值；逐周期
  `queue_occupancy_timeline` 仍可在进程内结果中检查，但不再嵌入默认 summary；
- `stall_cycles`：每种原因发生的周期数；同周期多条指令计 1；
- `stall_instruction_cycles`：按 `(instruction, reason, cycle)` 去重后计数；
- `compile_package_sha256`：本次消费的 BackendArtifact 规范 JSON hash；
- `trace.event_counts/stall_interval_count`：未嵌入 summary 的 trace 聚合计数；
- `offchip_read/write/total_bytes`：由 execution backend 按 payload 的 root-memory region
  统计，避免 scheduler 反向读取 task；
- `device_finish_cycle` 为最终 retire，`completion_finish_cycle` 为最终 complete，
  `retirement_drain_cycles` 为差值；`total_cycles` 包含 runtime synchronization。

原因包括 reception/WQ/IQ/ROB full、dependency wait、FU busy、Fu table full、tile window、
address hazard、bank/port conflict、completion bandwidth、retire backpressure 和各阶段
带宽。多种阻塞可以在同一周期发生，因此各项 `stall_cycles` 不应相加解释为总运行周期。
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

## 公开设备契约

```text
RuntimeSubmission + final TISA + MemoryPlan
  -> runtime.loader
  -> BoundTISADescriptor + DescriptorEnvelope + StaticSchedulePlan
  -> scheduler --IssueRequest--> ExecutionBackend
  <- scheduler <- ExecutionFeedback(execution_done / partial_ready)
```

completion token 包含 invocation id，重复调用不会串扰。`execution_done` 是物理结束；
cycle scheduler 之后仍经过 completion latency/width 仲裁才标记 complete，再经 wakeup latency
唤醒依赖，最后按 ROB 规则 retire。execution backend 拒绝 issue 时 scheduler 不产生假 issue。
模块边界测试禁止 event/cycle scheduler 读取 `BackendArtifact.execution_graph` 或 payload map。
