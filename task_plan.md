# 任务计划

## 目标

建立从真实 PyTorch 算子/模型到 TISA device simulator 的可配置研究栈，用同一份编译产物
比较 static 与 dynamic 调度，并输出可解释的周期、stall 和泳道图。

## 已完成基线

- [x] PyTorch nn.Module -> torch.export -> Torch-XLA -> official StableHLO -> Canonical IR；
- [x] GC canonicalization、semantic recovery、统一 tile planner、region/state dependency；
- [x] FC TISA 方言、TISA Generator、BackendArtifact；
- [x] RuntimeSubmission、物理地址绑定、runtime/device policy 四组合；
- [x] MachineConfig 与 codegen/timing/event backend registry；
- [x] staged artifact、manifest、泳道图、Perfetto trace；
- [x] RTL completion trace 和 VCS console log 的离线 profile importer；
- [x] Matmul、elementwise、reduce、Softmax、LayerNorm、RMSNorm、Attention、SwiGLU、
      RoPE、Conv2D、BatchNorm、pooling、reshape/transpose、static broadcast、scalar、
      dtype convert；
- [x] 固定窗口 KV-cache 与多步 RuntimeSequence；
- [x] 六个论文 benchmark 的 micro/representative proxy registry 与 paper-matrix 入口。

生产入口使用 compile-and-sim --torch-module MODULE:CLASS；分离流程使用 compile 和
simulate。测试 fixture 可以直接
构造所属层 IR，用于隔离验证接口契约。

## 阶段 1：模型与前端语义

### 已完成

- [x] 静态 Attention 与 pre-norm decoder；
- [x] BERT、GPT-J、LLaMA2、DeepSeek dense one-block；
- [x] LLaMA2 RoPE、固定窗口 KV-cache、prefill/decode micro；
- [x] ResNet bottleneck micro 的 Conv2D、BatchNorm inference、ReLU、pooling；
- [x] StableHLO operation capability registry 与 semantic fusion registry；
- [x] 常量 dynamic broadcast、dynamic_slice、dynamic_reshape specialization。

### 模型 proxy 扩展（已完成）

- [x] DeepSeek dense/MoE proxy capability 与显式精确路由边界；
- [x] embedding、position/type embedding、RoPE 和 causal mask；
- [x] Transformer/ResNet repetition、model proxy 与论文形状输入规模记录；
- [x] DeepSeek one-token fixed-window decode 与 request replay。

当前模型阶段交付的是 scheduler research proxy：hidden/channel 维度可缩放，模型组件、
层重复、phase、state 和请求数显式进入实验身份。动态 top-k、token compaction、expert
capacity、完整多层 decode cache 和精确论文拓扑进入后续数值/数据流扩展。

每个新增能力沿以下契约交付：

```text
StableHLO capability
  -> Canonical mapping / recovery
  -> TISA stage
  -> backend lowering
  -> PyTorch regression
```

## 阶段 2：编译与 TISA 正确性

### 已完成

- [x] PassManager、TileGraph、TISA descriptor 和 payload ownership；
- [x] region-aware dependency、卷积/池化 halo、broadcast、scalar region；
- [x] TileMem concrete stride、stride expression、layout metadata；
- [x] dtype policy、multi-result boundary 和 readiness condition；
- [x] candidate cost model、residency/ping-pong intent、per-pass dump；
- [x] materialized 与 online Softmax payload 属性。

### 动态地址与审计扩展（已完成）

- [x] symbolic shape 统一 binding（环境校验、Canonical resolve、shape specialization provenance）；
- [x] DynamicIndexExpr/Binding、dynamic_slice 和 dynamic_update_slice state metadata；
- [x] dynamic index -> physical offset/region resolution（clamp、dense/explicit stride、capacity check）；
- [x] dynamic update state window alias/address contract；
- [x] dynamic layout 和 stride-aware transform；
- [x] GC typed dependency 显式保存 hazard relation、logical region 和 readiness condition；
- [x] model proxy 的 component、parameter/input bytes 与编译 tile/MAC/traffic 统计入口。

阶段 1 的 trace/address provenance 已贯通：ExecutionGraph、TISA、Perfetto、CSV 和
address scoreboard 共享同一依赖来源；Matmul、broadcast、reduce、Conv2D、pooling 和
KV-cache 的专项验收继续随模型覆盖测试扩展。

阶段 1 的 trace/address provenance 和阶段 2 的 dynamic index/state address contract 已交付；
阶段 2 的 dynamic layout 与 stride-aware transform 已交付。

验收：固定 module、shape、tile、MachineConfig 和 backend 生成稳定 artifact hash；
static/dynamic 的差异来自 policy；小图数据可以逐项核对。

## 阶段 3：设备调度与后端

### 已完成

- [x] reception、queue、ROB/window、资源占用和 completion feedback analytical model；
- [x] typed RAW/WAR/WAW/STATE/ACCUMULATE、address scoreboard、partial-ready 原型；
- [x] memory bank/port structural-conflict model 与独立 stall 计数。
- [x] cycle_event：独立 reception/WQ/IQ/Fu/ROB 状态与逐 cycle 生命周期；
- [x] receive/dispatch/select/issue/completion/retire width 和显式 wakeup/控制延迟；
- [x] completion 与 retirement 分离、队列/Fu/tile 反压和逐周期 stall taxonomy；
- [x] 较老未 issue 地址冲突保护、hand-derived micro-tests 和同 artifact 策略比较。

### 目标存储映射根因修复（已完成）

- [x] `MachineConfig` 按 operation/operand role 定义 EU、目标 memory 与合法 transfer route；
- [x] minimal Matmul 映射到 DRAM/SRAM，lpu-like 映射到 LMB/RMB/PSB；
- [x] 多 EU 搬运在 FC 后的 Target Lowering 形成独立 TISA stage，最终 TISA 与
      BackendArtifact.program 一致；
- [x] `MemoryPlan` v2 保存 buffer/allocation、memory、alignment、layout/stride、slot、
      lifetime、alias 与 dependency-guarded reuse；
- [x] Runtime 只按编译计划绑定基址，旧 package 诊断与 MachineConfig topology guard；
- [x] 多 K/边界/strided Matmul、容量溢出、JSON round-trip、dynamic index/KV sequence、
      Attention 正式前端与 static/dynamic 同包验收。

### FC / Target Lowering 抽象边界（已完成）

- [x] FC Matmul 固定输出抽象 load/compute/store，不读取具体 memory、route hop 或 engine；
- [x] 抽象 transfer 显式声明 source/destination、operand role、符号 buffer 和
      Private/Local/Shared visibility/owner/domain；
- [x] TISA Generator 独立输出 virtual TISA；
- [x] `TargetPlan` 成为 memory placement、route/engine、target instruction/operand、
      dependency expansion 与 abstract→target provenance 的单一来源；
- [x] Matmul payload 直接从 TargetPlan 生成，删除 FC/lowering 双份 route grouping 和
      stage-key 配对；
- [x] backend 内部 scratch 在 allocation 前以 TargetPlan internal-resource declaration
      显式登记；
- [x] 同一 FC/virtual TISA 分别 lower 到 minimal 与 lpu-like，并覆盖 route/engine 变体。

### 进行中

- [ ] 用实际 scheduler RTL/profile 校准控制开销、在线仲裁及多核行为；
- [ ] SCALE-Sim/Ramulator2 类 MXU/memory timing；
- [ ] RTL/Verilator unit timing 与 system simulator adapter；
- [ ] backend capability、timing interval、calibration status 的统一声明。

## 阶段 4：论文实验矩阵

固定维度：

```text
model / shape / phase
  x tile candidate
  x runtime policy
  x device policy
  x MachineConfig
  x timing/event backend
```

- [x] paper-matrix 单次编译、共享 artifact 和 policy matrix；
- [x] case/variant staged output、matrix_index、sweep 汇总；
- [x] model-proxy repetition 与顺序 request-level RuntimeSequence；
- [ ] source-derived 与 RTL-observed 分组统计。

## 当前执行顺序

1. scheduler 微结构和控制开销校准；
2. 外部 timing/memory/RTL backend；
3. 论文规模 source-derived/RTL-observed 矩阵；
4. 动态 top-k/token compaction 与精确 full-model 数据流。

## 阶段 5：Compile-only 与独立仿真

### 已完成

- [x] 新增 `compile` compile-only package 入口；
- [x] 新增 `simulate --compile-dir` 独立仿真入口；
- [x] 一站式入口改名为 `compile-and-sim`；
- [x] 为跨命令恢复补齐 IR `from_dict()` 和 schema 校验；
- [x] runtime JSON 支持 dynamic index/layout 与 invocation 配置。

### 目标

- `compile` 只执行 PyTorch -> StableHLO -> GC/FC -> TISA/backend，并输出可持久化
  的 compile package；
- `simulate` 只读取 package，根据 invocation manifest 为编译期 MemoryPlan 绑定基址、
  dynamic index/layout，
  再选择 machine、runtime policy、device policy 和 timing backend；
- `compile-and-sim` 保留为一站式入口；`compile` 与 `simulate` 用于分离执行。

### 验收标准

- `BackendArtifact`、`ExecutionGraph`、TISA IR 可以从 JSON 严格恢复并通过原有 validate；
- compile package 不依赖 PyTorch 即可被 simulator 消费；
- 同一 package 使用不同 runtime manifest 和 scheduler 参数生成不同 simulation trace；
- dynamic index/layout 只影响 runtime binding 与地址/时序，不修改编译期 program；
- 端到端 CLI 与原有测试保持兼容。

当前验收状态：MemoryPlan v2 JSON package 独立仿真、拓扑校验和 staged output 已通过；
本地无前端依赖环境 151 项通过、35 项跳过；9980X-new 的 Torch-XLA/StableHLO 环境
186 项全部通过。

## 验证命令

```bash
PYTHONPATH=src /usr/bin/python3.12 -m unittest discover -s tests -v
PYTHONPATH=src /usr/bin/python3.12 -m compileall -q src tests examples
git diff --check
```
