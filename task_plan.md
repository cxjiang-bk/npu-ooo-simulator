# 任务计划：参数化 NPU OOO 编译与仿真框架

## 目标

独立构建从顶层算子到参数化 cycle simulator 的完整研究栈，在公平条件下比较 Static 与 TISA-like Dynamic tile scheduling，并生成总周期、stall 分解和泳道图。

## 当前阶段

阶段 4-6：动态/TISA-like backend、算子覆盖和实验矩阵并行推进。

## 阶段状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | Model IR、MachineConfig、IR、trace 和 experiment schema | in_progress |
| 1 | Model IR、MachineConfig 与基础 Operator Graph IR | completed |
| 2 | 2mm tile instance 和 primitive lowering | completed |
| 3 | Static discrete-event simulator 与 trace | completed |
| 4 | Dynamic/TISA-like scheduler | in_progress |
| 5 | Elementwise/Reduce/Softmax/Attention | in_progress |
| 6 | Architecture x Schedule x Policy 实验框架 | in_progress |
| 7 | TileFlow/SCALE-Sim/RTL 校准 | pending |

## 阶段 0 检查表

- [x] 新建独立 GitHub 仓库；
- [x] 确定系统分层和项目边界；
- [x] 确定分阶段路线图和第一里程碑；
- [ ] 冻结 MachineConfig 字段和版本策略；
- [x] MachineConfig canonical JSON round-trip 与 CLI 外部配置入口；
- [ ] 冻结 Model/Benchmark IR 的 normalized schema；
- [ ] 冻结 `evaluation_scope=one_block|layer|full_model` 语义；
- [ ] 冻结 semantic operator 与 lowering primitive taxonomy；
- [x] 首批 semantic operator 的 lowering registry 与 mixed-graph handoff 契约；
- [x] LayerNorm mean/variance barrier lowering 与 micro-test；
- [x] workload sweep 的 dynamic priority 维度与 Static 配对 baseline；
- [x] 单头 attention `QK^T -> Softmax -> PV` mixed graph 与 CLI；
- [x] LayerNorm + attention + MLP + residual transformer block skeleton；
- [ ] 冻结五层 IR 的 normalized schema；
- [ ] 冻结 ExecutionTask dependency/address schema；
- [ ] 冻结 simulator event/tie-break 语义；
- [ ] 冻结 trace/summary/manifest schema；
- [x] 为 dual/triple pipeline 编写手算 golden case（dual reservation + drain 已由测试固定；stage_count 支持 triple）。

阶段 0 目前已落地 Model/Operator/MachineConfig、Schedule/Tile/Execution、trace/summary/manifest 基础 schema；核心 dual/triple golden case 已有，完整 normalized schema 版本策略和真实 ISA 契约仍待冻结。

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

当前已有两个 architecture profile 的 analytical cycle 对比、CSV/SVG/PNG/Perfetto 输出、ROB/window/queue 指标、显式 static stage reservation、runtime address scoreboard 和 occupancy timeline；`sweep-two-mm` 已能批量生成 architecture × policy × window × ROB 结果。混合 decoder block 已完成第一条跨算子 lowering 闭环；真实 MXU/memory timing 校准仍待后续提交。

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
| scheduler policy 与 event backend 分离 | policy 只选择 ready task；queue、ROB、II、资源占用和 completion wake-up 由 simulator 统一处理 |
| `SimulatorConfig` 覆盖 MachineConfig runtime capacity | 便于直接 sweep instruction queue、ROB、dependency window、in-flight tile，而不修改编译图 |
| address scoreboard 作为可选 runtime layer | 基于 active `BufferRegion` 生成 RAW/WAR/WAW issue stall，COMPLETE 后释放范围；不改写默认 graph，方便和 compile-time dependency 做公平对照 |

## 暂不做

- 完整 ISA binary encoder；
- 完整 Conv2D data-layout/halo 优化；
- RTL/UVM 集成；
- 未校准 energy 常数；
- 将 TileFlow aggregate pipeline cycle 当作 event trace。
- 把当前 analytical timing 或 address prepass 宣称为真实 TISA/RTL scoreboard。
- 把 DeepSeek-R1-16B 未经配置证据直接假设为 dense 或 MoE。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| 旧仓库规划追加补丁首次锚点不匹配 | 1 | 使用稳定尾部锚点追加，之后按用户要求完整移除本轮追加内容 |
| artifact 验证命令包含 `rm -rf` 被执行策略拒绝 | 1 | 不清理目录，直接由确定性导出覆盖同名 artifact |
| LayerNorm barrier 测试错误假设首 tile 完成均值 | 1 | 按实际 M 外层/N 内层 tile 顺序断言每行最后一个 reduction tile |
| Attention tile count 测试把三个阶段相乘 | 1 | 按 QK、Softmax、PV 三个独立 operator 的 tile 数求和，默认小图为 12 |
| Transformer block handoff count 测试把 tile/task 数混同 | 1 | 依赖计数按 root-memory 相交的 store/load 边统计，默认 skeleton 为 8 |

## 备注

- 开始每个实现阶段前重新检查 `docs/architecture.md` 和本文件；
- 每完成一个阶段更新状态和 `progress.md`；
- 外部论文/仓库发现记录到 `findings.md`，不直接变成执行指令。
