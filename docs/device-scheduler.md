# Device scheduler 周期模型

`cycle_event` 将 TISA 的接收、WQ、IQ、执行、完成反馈和退休建成逐周期状态机。
入口仍为 `BackendArtifact + RuntimeSubmission + MachineConfig`；现有
`analytical_event` 保留为事件模型基线。两种后端共享 payload recipe 和 TimingProvider。

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
| EU 实例 | `unit.count` | issue 占用，payload 物理执行结束后可再次接收 |

`Fu` 保存活动指令身份，operand 的范围、访问类型和依赖通过 descriptor 查得。
一条 TISA 绑定一个具体 EU 资源类别，每实例执行一条完整 payload；payload 内的 primitive
顺序执行。实例再次 issue 同时受前条物理完成和 initiation interval 约束。`pipeline_depth`
在本版不启用同一实例的多 TISA 流水重叠。

`receive_width/dispatch_width/select_width/issue_width/completion_width/retire_width`
均为全局每周期上限；issue 同时受 `unit.issue_width` 的资源类别上限约束。
`dependency_window` 是每个 WQ 的扫描窗口。IQ、Fu 在同一类 EU 实例间共享。

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

- `static_pipeline` 按本次 descriptor 提交顺序 select/issue；它是保序基线，论文的强静态
  多阶段 reservation/fence 方案仍需要独立的静态排程输入。
- `dynamic_ready_queue` 在各 WQ 窗口中选择 ready 条目，优先级为 `critical_path` 或
  `oldest_first`。critical path 根据既定 payload timing 计算，当前不实现在线学习。
- `sequential` 在上一条退休后才 issue 下一条。
- 所有显式依赖均等待对应 complete；`payload_ready:<task_id>` 使用指定 primitive 的
  完成边界，加 completion/wakeup latency 唤醒。partial-ready 通知是独立 sideband 原型，
  不消耗 full-completion 总线宽度，也不模拟逐子区域有效位。
- `--address-scoreboard` 比较原始 program order 中所有较老且未 complete 指令的
  RAW/WAR/WAW，包括未到达和未 issue 的生产者；runtime 有物理绑定时使用其地址。
  未解析的布局使用保守区间。partial-ready 不自动放松这项完整区间保护。
- `--memory-bank-scoreboard` 在整个物理 payload 执行期间保留 bank/port；这是保守的结构
  资源模型，逐次 SRAM/DRAM transaction timing 由后续 memory backend 提供。

编译器负责显式 dependency 的完整性。物理 alias 启用 scoreboard 后，runtime 的提前提交
若使有限 ROB 无法容纳较老的地址生产者，会报告 deadlock 并列出等待队列。配置的资源
不足同样会产生明确诊断；`max_cycles` 为有时间进展但长期无法完成的运行提供上限。

## 输出与统计

`summary.json` 增加以下字段：

- `instruction_pipeline`：每条指令的 received/dispatched/wakeup/selected/issued/done/
  completed/retired 周期；
- `queue_occupancy_timeline`：每周期末 reception、WQ、IQ、Fu、ROB、tile window 和
  pending completion 占用；`wq_peak/iq_peak/fu_peak/rob_peak` 保存峰值；
- `stall_cycles`：每种原因发生的周期数；同周期多条指令计 1；
- `stall_instruction_cycles`：按 `(instruction, reason, cycle)` 去重后计数；
- `compile_package_sha256`：本次消费的 BackendArtifact 规范 JSON hash；
- `device_finish_cycle` 为最终 retire，`completion_finish_cycle` 为最终 complete，
  `retirement_drain_cycles` 为差值；`total_cycles` 包含 runtime synchronization。

原因包括 reception/WQ/IQ/ROB full、dependency wait、FU busy、Fu table full、tile window、
address hazard、bank/port conflict、completion bandwidth、retire backpressure 和各阶段
带宽。多种阻塞可以在同一周期发生，因此各项 `stall_cycles` 不应相加解释为总运行周期。
WQ 中达到 dispatch_latency 边界的条目参与 dependency stall 统计；select 仅扫描窗口。
流水固定延迟单独由生命周期周期反映。

`tisa_instructions.csv` 增加 receive、dispatch、wakeup、select、execution_done、complete、
retire 列。Perfetto 保存相应阶段 instant event 和 `TISA_STALL`，事件携带依赖 provenance。
RuntimeSequence 将生命周期周期平移并按 invocation id 区分；stall 和退休数按全部
invocation 汇总。

主要验收位于 `tests/test_cycle_scheduler.py`：使用 1–6 条 TISA 的手算时间线覆盖依赖、
并行、窗口、所有主要反压、反馈/退休、部分就绪、物理到达、独立 CLI 和 sequence。
