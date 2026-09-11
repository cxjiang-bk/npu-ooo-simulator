# 文档导航

文档按用途分目录，避免架构说明、运行命令和结果分析互相穿插。

## 架构与设计

- [总体架构](architecture/architecture.md)：从前端、GC/FC/TISA、Target Lowering 到 Runtime、Device Scheduler 和 Backend 的完整链路。
- [Scheduler 架构与流程](architecture/scheduler.md)：论文 `Reception Buffer → WQ → IQ → Exec` 与当前 `cycle.py` 实现的对应关系，包含 ROB、wakeup、select、issue 和 complete/retire 流程。
- [Scheduler 与论文对齐](architecture/scheduler-paper-alignment.md)：`SemanticConflict(I, Fu[u])`、per-EU Fu、全局跨 EU dependency notification，以及当前实现缺口。
- [TISA 论文语义对齐](architecture/tisa-alignment.md)：论文概念与项目 IR、TargetPlan、payload 的映射。
- [算子分类与覆盖](architecture/operator-taxonomy.md)：semantic operator、lowering primitive 和覆盖计划。
- [FlashAttention 与 Top-2 MoE](architecture/flash-attention-moe.md)：online softmax、内部 top-2 router、scheduler-visible 数据流及当前稀疏执行边界。
- [Target lowering 覆盖](architecture/target-lowering-coverage.md)：不同算子的目标 memory、route、EU 和 payload 能力。

## 运行与配置

- [环境安装](running/install-stablehlo.md)：PyTorch、Torch-XLA、StableHLO 安装和验证。
- [Workload 配置](running/workload-config.md)：通用 workload JSON、编译和仿真入口。
- [Device Scheduler 周期仿真](running/device-scheduler.md)：`cycle_event` 参数、命令、容量、延迟、stall 和输出指标。
- [编译期静态排程](running/static-scheduling.md)：`static_streams`、set/wait/fence 和静态控制程序。

## 分析与校准

- [结果分析](analysis/result-analysis.md)：气泡、等待链、资源利用率和 Static/Dynamic 对比。
- [可视化](analysis/visualization.md)：TISA 图、生命周期、buffer occupancy、swimlane 和 Perfetto。
- [RTL Completion Trace 校准](analysis/rtl-calibration.md)：RTL completion profile 的导入和 timing 校准契约。

## 项目管理

- [后续开发路线](project/roadmap.md)：当前能力、限制和后续工作。

README 中的命令行示例仍然是用户入口；本页用于按主题查找设计和运行说明。
