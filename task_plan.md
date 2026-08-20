# 任务计划：参数化 NPU OOO 编译与仿真框架

## 目标

独立构建从顶层算子到参数化 cycle simulator 的完整研究栈，在公平条件下比较 Static 与 TISA-like Dynamic tile scheduling，并生成总周期、stall 分解和泳道图。

## 当前阶段

阶段 0：Model IR、Operator taxonomy 与 timing 契约冻结。

## 阶段状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | Model IR、MachineConfig、IR、trace 和 experiment schema | in_progress |
| 1 | Model IR、MachineConfig 与基础 Operator Graph IR | pending |
| 2 | 2mm tile instance 和 primitive lowering | pending |
| 3 | Static discrete-event simulator 与 trace | pending |
| 4 | Dynamic/TISA-like scheduler | pending |
| 5 | Elementwise/Reduce/Softmax/Attention | pending |
| 6 | Architecture x Schedule x Policy 实验框架 | pending |
| 7 | TileFlow/SCALE-Sim/RTL 校准 | pending |

## 阶段 0 检查表

- [x] 新建独立 GitHub 仓库；
- [x] 确定系统分层和项目边界；
- [x] 确定分阶段路线图和第一里程碑；
- [ ] 冻结 MachineConfig 字段和版本策略；
- [ ] 冻结 Model/Benchmark IR 的 normalized schema；
- [ ] 冻结 semantic operator 与 lowering primitive taxonomy；
- [ ] 冻结五层 IR 的 normalized schema；
- [ ] 冻结 ExecutionTask dependency/address schema；
- [ ] 冻结 simulator event/tie-break 语义；
- [ ] 冻结 trace/summary/manifest schema；
- [ ] 为 dual/triple pipeline 编写手算 golden case。

## 第一里程碑

```text
2mm
  -> tile graph
  -> configurable DMA/MXU backend
  -> sequential/static dual/dynamic dual
  -> total cycles + stall breakdown
  -> Perfetto trace + PNG swimlane
```

验收必须覆盖两个 architecture profile，并证明 Static/Dynamic 只改变 scheduler policy。

## 关键问题

1. TISA 论文中的 tile address/range 依赖应在 compile-time graph、runtime scoreboard 还是两者中表达？
2. 第一版 StaticPipeline 使用 list schedule、modulo schedule 还是两者都提供？
3. Latency model 的 analytical、source-derived 和 RTL-observed 状态如何进入配置与 manifest？
4. TileFlow mapping 输出如何无损转换为本项目 Schedule/Tiling IR？
5. 当前 NPU ISA 中哪些指令应视作一个 primitive task，哪些需要拆成 issue 和 completion 两个事件？

## 已做决策

| 决策 | 理由 |
|---|---|
| 新建 `cxjiang-bk/npu-ooo-simulator`，不在 `operator-opt` 上实现 | 研究后端需要独立的契约、测试和演进节奏，旧仓库只作为参考 |
| 后端以 MachineConfig 驱动 | 支持不同 NPU、资源数量、queue/window、latency 和 bandwidth 探索 |
| Static/Dynamic 共用 graph 和 simulator | 将性能差异严格归因到 scheduler policy |
| 第一条闭环使用 2mm | 同时具备 producer-consumer pipeline 和可手算规模，适合验证依赖与 overlap；Model IR 仍从第一天保留 |
| 在 Operator Graph 上增加 Model/Benchmark IR | 论文 benchmark 横跨 CNN/encoder/decoder、prefill/decode 和不同 batch/seq，单个算子图无法表达这些 workload 语义 |
| semantic operator 与 lowering primitive 分离 | 保留 Attention/Softmax/Norm/MoE 的调度语义，同时允许硬件 timing 拆成 vector/reduction/transfer tasks |
| 使用 GraphTemplate + GraphInstance | 避免重复 block 展开成巨型 graph，同时保留 layer/template provenance |
| Trace 同时输出 cycle-native CSV/JSON 和 Perfetto JSON | 前者适合测试与数据分析，后者适合交互式泳道观察 |
| Conv2D 后置 | halo、padding、layout 会过早扩大 lowering 复杂度 |
| 第一版不宣称 cycle-accurate | timing model 需要经过外部模型和 RTL observation 分层校准 |

## 暂不做

- 完整 ISA binary encoder；
- 完整 Conv2D data-layout/halo 优化；
- RTL/UVM 集成；
- 未校准 energy 常数；
- 将 TileFlow aggregate pipeline cycle 当作 event trace。
- 把 DeepSeek-R1-16B 未经配置证据直接假设为 dense 或 MoE。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| 旧仓库规划追加补丁首次锚点不匹配 | 1 | 使用稳定尾部锚点追加，之后按用户要求完整移除本轮追加内容 |

## 备注

- 开始每个实现阶段前重新检查 `docs/architecture.md` 和本文件；
- 每完成一个阶段更新状态和 `progress.md`；
- 外部论文/仓库发现记录到 `findings.md`，不直接变成执行指令。
