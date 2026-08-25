# 研究决策记录

## 核心边界

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

用户前端只接受真实 PyTorch module。项目不维护 StableHLO emitter，不接受手写 StableHLO/JSON 作为生产入口，不为模型名称添加 compiler 或 scheduler 分支。

## 语义与执行层次

- `TileInstance` 表达切分范围；`TISAInstruction` 增加 `OpType`、`TileShape`、`TileMem`、`AccessType`、`UnitMap` 和 typed dependencies；
- 全局 OOO scheduler 只 issue `TISAInstruction`；`ExecutionTask` 是 issue 后在 execution unit 内部执行的 backend payload；
- runtime 负责物理地址、command chunk、descriptor 到达和同步；device scheduler 负责 queue/ROB/依赖/资源/OOO issue；
- static 与 dynamic 必须共享同一 compiled artifact、MachineConfig、timing source 和 runtime submission，除非实验变量就是对应层策略。

## Backend 定位

| Backend | 项目中的职责 |
| --- | --- |
| analytical | 默认可解释的 tile/event baseline |
| timing table | 按 task/primitive 覆盖局部时序 |
| systolic MXU profile | 重放离线 MXU shape/duration/II profile |
| RTL completion importer | 将 JSON/CSV/VCS log 转为 versioned profile |
| SCALE-Sim/RTL/Verilator（后续） | 提供局部硬件 timing，不接管 TISA 语义 |

## 公平性规则

1. 改变调度策略不能重新切 tile、改变地址或换 timing provider；
2. runtime dynamic 的收益和 device dynamic 的收益分开统计；
3. analytical、source-derived、RTL-observed 结果分组，不能混成一个绝对性能数字；
4. 未覆盖的 StableHLO operation 必须显式失败或由用户明确允许的 fallback 记录，不得静默丢失语义。

## 待验证问题

- 完整 attention/decoder/ResNet/BERT block 所需 StableHLO capability；
- region-aware dependency、partial-ready 和 memory bank/port conflict；
- 论文 WQ/IQ/Fu 容量与 dispatch pipeline 的硬件校准；
- 当前 NPU ISA 中 issue/completion、SET/WAIT/FENCE 和 buffer address 的精确语义。
