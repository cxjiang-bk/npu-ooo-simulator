# 研究发现与决策

## 需求

- 从顶层算子建立独立编译与仿真栈；
- 后端架构参数可配置，不能固定为当前 NPU；
- 支持不同调度策略的泳道图和整体执行周期；
- 复现 TISA 论文中的 sequential、static/dynamic dual/triple staged pipeline；
- 旧 `operator-opt` 只作为只读参考。
- 论文 benchmark 需要 Model/Benchmark layer，不能只建立孤立 Operator Graph。

## 已确认事实

- `operator-opt` 已覆盖 Fusion IR、Tiling Tree、Attention/2mm TileFlow mapping 和 aggregate cost，但缺少 per-tile execution lowering 和事件仿真；
- 其 `LpuTaskGraph` 有 resource/duration/predecessor/start/finish，可作为粗粒度参考，不足以表达 queue、II、地址范围和 runtime wake-up；
- TileFlow 的 `Sequential/Pipeline` 主要通过 cycle sum/max 构造 aggregate estimate，不直接产生论文所需的 per-tile issue/completion trace；
- 当前可运行后端有较多 Attention 专用命名和路径规则，因此新项目必须采用 operator lowering registry；
- Static 与 Dynamic 公平比较必须共享 tile graph、buffer、地址、依赖、latency 和 hardware config。
- TISA 原文的 compiler stack 是 `Framework bridge -> Graph compiler -> Fusion compiler -> TISA generator -> backend`，并使用 StableHLO/MLIR 保留 operator semantics；这直接支持在 Operator Graph 上增加 Model/Benchmark IR。
- Table IX 的 benchmark case 具有不同 model family、dtype、batch、sequence/image shape 和 prefill/decode phase；每一行都应是独立 BenchmarkCase。
- 论文 TISA instruction 不是只含 opcode：`OpType + Operands(TileShape/TileMem/AccessType) + Attributes + UnitMap`；这要求 semantic operator context 在 lowering 后仍保留。

## 开源参考定位

| 项目 | 参考内容 | 不直接复用的部分 |
|---|---|---|
| TileFlow/Timeloop | mapping、tiling、memory traffic、aggregate cost | per-tile event simulation |
| TVM-VTA | LOAD/COMPUTE/STORE、dependency token、static pipeline | VTA 固定 ISA/三级 pipeline |
| Gemmini | queues、ROB、access/execute decoupling | RISC-V/RoCC 和 Gemmini 专用实现 |
| SCALE-Sim | MXU/systolic timing、bandwidth/stall | 多 execution-unit OOO scheduler |
| Perfetto | 多 lane event trace | scheduling semantics |

## 技术决策

| 决策 | 理由 |
|---|---|
| canonical MachineConfig 独立于 RTL parser | 支持手写探索 profile、RTL-derived profile 和未来其他 NPU |
| 四层 IR 明确分离 | 防止 schedule factor、runtime tile 和 hardware task 混淆 |
| 独立 ExecutionTask graph | 现有 aggregate task graph 无法承载 tile address/queue/event 语义 |
| 确定性离散事件 simulator | 便于手算验证、回归和可复现实验 |
| latency 与 initiation interval 分离 | 必须表达流水化执行单元的 overlap |
| 2mm 先行，Attention 后接 | 先验证核心机制，再增加 softmax/barrier/cache 生命周期 |
| Model IR 先于 Operator Graph 实例化 | 模型重复 block、运行 phase、KV cache、mask 和 benchmark shape 不属于单个算子 |
| semantic operator 与 primitive task 分离 | Dynamic scheduler 需要知道 `SOFTMAX`/`RMSNORM` 等语义，而 simulator 仍需计时 `reduce_max/exp/reduce_sum` 等 primitive |
| ScheduleSpec 明确保存 tile factor、loop order、residency 和 stage | 让 mapping 结果可序列化，并与实际边界 tile、runtime task 解耦 |
| TileInstance 的 coordinates 保存 tile index，bounds 保存实际 `[start, stop)` | 边界 tile 不会被错误地当成满 tile，后续地址/traffic 统计可直接复用 |
| ExecutionTask 显式携带 BufferRegion 和 predecessor | 统一承载 TISA operand 的 TileShape/TileMem/AccessType 语义，scheduler 不需要猜 tensor 地址或依赖 |
| Matmul lowering 对每个 K tile 建立累加链，最终 tile 才生成 store | 保留 partial-sum 生命周期，同时让跨算子 producer store -> consumer load 依赖可观察 |
| Policy 只改变 ready-task 选择；task graph 与 MachineConfig 作为共享输入 | 保证 Static/Dynamic 周期差异归因于调度策略，而不是重新切 tile 或更换 timing model |
| scheduler policy 与 event backend 分离 | policy 只选择 ready task；event backend 统一处理 issue/start/complete、queue、ROB、II、in-flight tile 和 completion wake-up |
| `SimulatorConfig` 覆盖 MachineConfig runtime capacity | 可以对 dependency window、ROB、instruction queue、ready queue 和 tile window 做实验 sweep |
| address scoreboard 作为可选 runtime layer | 基于 active `BufferRegion` 生成 RAW/WAR/WAW issue stall，COMPLETE 后释放范围；不改写默认 graph，方便和 compile-time dependency 做公平对照 |
| Elementwise/residual-add 先于 reduce/softmax 接入 | 它能验证多输入同形 tile、ARU primitive 和 producer-consumer store/load 依赖，同时不引入 reduction barrier 与指数近似等额外语义 |
| Dynamic priority 必须作为独立实验维度 | Softmax 的 ARU/DMA 竞争反例中，`window=8/ROB=8` 下 critical-path heuristic 为 4808 cycles，oldest-first 为 3784 cycles；动态机制本身不保证某个启发式总是占优 |
| Tile size 属于 mapping 实验维度，不属于 scheduler policy | 2mm `tile_size=16` 与 `32` 产生不同 tile/task graph 和周期，但每个 tile size 内 Static/Dynamic 仍共享完全相同的 lowered graph；sweep manifest 必须把 tile size 单独记录 |
| RMSNorm 可先建模为 sum-square barrier | `square -> reduce_sum_square -> rmsnorm` 保留跨 reduction tile 的生命周期和完成依赖；epsilon/scale 的数值语义留在 operator attributes，当前 analytical timing 不宣称数值精确 |

## 视觉发现

论文示意图的五条时间线可以统一为 iteration-specific stage DAG：

```text
Sequential
Static dual-stage
Dynamic dual-stage
Static triple-stage
Dynamic triple-stage
```

彩色块表示不同 iteration 的 stage task；虚线表示静态 stage/iteration 边界；尾部阴影 `E*` 表示 pipeline drain 或依赖/资源造成的结束差异。新 simulator 必须显式输出 task start/end 和 drain cycles，不能只返回一个总 Cycle。

## 待验证

- TISA 原文对 tile dependency table、地址范围、窗口和完成事件的精确定义；
- 当前 NPU ISA 的 issue/completion、SET/WAIT/FENCE 和 buffer address 语义；
- 哪些 latency 可从 RTL source 提取，哪些必须通过 waveform/hardware counter 校准；
- TileFlow mapping 到新 Schedule IR 的完整信息保真度。
- Model import 的 StableHLO/ONNX/PyTorch adapter 最小公共字段；
- DeepSeek-R1-16B benchmark 的实际 dense/MoE 配置与 KV-cache/attention 细节；
- ResNet50 inference 中 BatchNorm 是否已 fold 到 Conv；

## 资源

- 论文：`/home/lora/OpenTPU/ooo_research/Song 等 - Dynamic scheduling for AI accelerators via TISA.pdf`
- 参考仓库：`/home/lora/OpenTPU/operator-opt`，仅只读使用；
- 新项目：`https://github.com/cxjiang-bk/npu-ooo-simulator`。
