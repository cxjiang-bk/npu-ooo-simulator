# Target lowering 覆盖清单

本清单描述当前项目实现能力，不把内部职责划分解释为论文明确规定。FC 输出符号
transfer/compute；TargetPlan 决定具体 memory、route、engine、目标指令和 operand；payload
只能绑定已规划 operand，内部 scratch 必须在 allocation 前登记。

| Canonical op | FC / virtual TISA | target placement（minimal / lpu-like） | payload 与限制 |
| --- | --- | --- | --- |
| matmul / batched_matmul / gemv | lhs/rhs/output，abstract load/compute/store | 专用 role 配置：SRAM；或 LMB/RMB/PSB 与 GM↔UB 多跳 | 逐条 TargetInstructionPlan 直接生成；支持多 K、partial tile、transpose、累加 |
| elementwise / residual_add | input/output transfer + vector compute | SRAM / UB，DMA/GDMA + ARU | TargetPlan recipe adapter；保留 broadcast region |
| reduce | input/output + reduction state | SRAM / UB，DMA/GDMA + ARU | TargetPlan recipe adapter；支持多 iteration/reduction 维 |
| softmax | input/output + 单条 scheduler-visible composite compute | SRAM / UB，DMA/GDMA + ARU | 内部 reduce/exp/normalize 属同一 payload；scratch 显式加入 TargetPlan |
| rmsnorm / layernorm | input/output + composite vector compute | SRAM / UB，DMA/GDMA + ARU | 内部统计量 scratch 显式规划 |
| swiglu | input/output + composite vector compute | SRAM / UB，DMA/GDMA + ARU | logistic/silu/gate multiply 保持 instruction-local |
| kv_cache_update | state/input/output 与动态 region | SRAM / UB，DMA/GDMA + ARU | persistent state、动态 index/layout、alias 由 MemoryPlan/runtime 校验 |
| conv2d | tensor input/weight/output | SRAM / UB，DMA/GDMA + MXU | 显式 tensor-local 类；尚无 lpu-like 专用卷积多级 operand bank |
| batch_norm / pool | input/output region | SRAM / UB，DMA/GDMA + ARU | halo/统计 region 保留；analytical payload |
| reshape / transpose / slice | logical transform | root direct：DRAM+DMA / GM+GDMA | 显式 root-transfer 类；非连续 reshape 可物化 copy |
| embedding | table/index/output gather | root direct：DRAM+DMA / GM+GDMA | 显式 root-transfer 类；尚无专用 cache/local gather path |

所有行都要求唯一 target TISA owner、abstract→target provenance、TargetPlan operand 与 payload
region 一致、MemoryPlan 容量/对齐通过。Matmul 是直接计划驱动 codegen；其余算子复用已有
数学/region recipe，经 TargetPlan adapter 建立唯一 ownership。adapter 不允许重新选择
route/memory/EU，但复合算子尚未全部改写为逐 target instruction 的独立 codegen 类，这是
当前明确能力边界。

缺少 operation class、合法 route、engine 或 unit capability 时编译直接失败，不再选择“第一条
路径”或无条件 root/local fallback。当前 generic 非 direct class 限定为一跳；需要多跳时必须
新增算子专用 target lowerer。
