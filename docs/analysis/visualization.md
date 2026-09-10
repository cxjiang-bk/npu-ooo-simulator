# Tile、TISA、生命周期与 Buffer 可视化

## 编译图

- `01_gc/tile_graph.dot`：按 operator cluster，节点显示 iteration coordinates、tile range、
  stage；边区分 data/state/accumulate/buffer reuse/control 及 hazard/condition。
- `08_analysis/tisa_graph.dot`：target TISA 的 op、EU、memory、abstract provenance、正确性边；
  同时以虚线显示 Static stream 和 set/wait/fence。
- `08_analysis/tisa_graph_dynamic.dot`：只显示 Dynamic 实际使用的共享正确性边，不把 Static
  control 画成有效约束。
- `08_analysis/program_hierarchy.json`：operator → tile → target TISA → payload 的逐级数据，
  大图 DOT 默认最多展开 256 条 TISA。

## 时间线

默认 `swimlane.svg/png` 只画 Runtime 和物理 EU payload，矩形标签保留 parent TISA；不再
重复画 TISA issue→complete 轨道。需要旧层级时打开 `swimlane-detailed.svg`。

`report.html` 是可独立打开的离线联合视图，共享一条 cycle 轴：

- 纵向主/次网格标出周期坐标，所有面板使用相同的起止周期；
- 上方：物理 EU 执行与 bubble；
- 中间：由生命周期事件重构的每类 EU 独立 `WQ[EU]`、`IQ[EU]` 面板，以及 ROB、
  completion 面板；
- 下方：各 memory 的 protected/retained occupancy。每个 memory 使用自身观测峰值作为
  纵轴上限，同时在左侧保留 allocated/capacity 数值，避免小幅占用被总容量压平；
- 点击物理区间可显示 parent TISA，表格和 JSON 可继续定位 blocker、slot 和 dependency。

## Buffer 口径

`05_runtime/buffer_lifecycle.json` 与 `buffer_occupancy.csv` 区分：

1. allocated：MemoryPlan 固定预留；
2. protected：正在写入或仍被消费者/同步保护，暂不可复用；默认曲线；
3. retained：数据已经产生且后续仍需要。

`BUFFER_PROTECT/VALID/RELEASE` 来自实际 issue、physical done 和最后消费者 completion。
allocation alias 按唯一 `allocation_id` 计一次；allocation/valid/padding bytes 分开。当前为
保守 allocation/TISA 粒度，不伪造逐字节填充或 DRAM transaction。Persistent state 在
invocation 内始终保留，并在 sequence 版本身份中携带 invocation。
