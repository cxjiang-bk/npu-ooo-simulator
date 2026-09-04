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

### 进行中

- [ ] DeepSeek dense/MoE capability 清单；
- [ ] embedding、position embedding、causal mask 和 layout 变体；
- [ ] 完整模型 repetition 与论文形状 proxy 的输入规模验证。

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

### 进行中

- [x] symbolic shape 统一 binding（环境校验、Canonical resolve、shape specialization provenance）；
- [x] DynamicIndexExpr/Binding、dynamic_slice 和 dynamic_update_slice state metadata；
- [x] dynamic index -> physical offset/region resolution（clamp、dense/explicit stride、capacity check）；
- [x] dynamic update state window alias/address contract；
- [x] dynamic layout 和 stride-aware transform；
- [x] GC typed dependency 显式保存 hazard relation、logical region 和 readiness condition；
- [ ] 完整模型 proxy 的 tile/MAC/traffic 对账样例。

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

### 进行中

- [ ] 论文 WQ/IQ/Fu 容量、dispatch width、控制开销校准；
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
- [ ] full-model repetition、request-level runtime；
- [ ] source-derived 与 RTL-observed 分组统计。

## 当前执行顺序

1. compile/simulation 分离与可复用 artifact package；
2. DeepSeek 与完整模型 repetition；
3. scheduler 微结构和控制开销校准；
4. 外部 timing/memory/RTL backend；
5. 论文规模 source-derived/RTL-observed 矩阵。

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
- `simulate` 只读取 package，根据 invocation manifest 绑定 buffer、dynamic index/layout，
  再选择 machine、runtime policy、device policy 和 timing backend；
- `compile-and-sim` 保留为一站式入口；`compile` 与 `simulate` 用于分离执行。

### 验收标准

- `BackendArtifact`、`ExecutionGraph`、TISA IR 可以从 JSON 严格恢复并通过原有 validate；
- compile package 不依赖 PyTorch 即可被 simulator 消费；
- 同一 package 使用不同 runtime manifest 和 scheduler 参数生成不同 simulation trace；
- dynamic index/layout 只影响 runtime binding 与地址/时序，不修改编译期 program；
- 端到端 CLI 与原有测试保持兼容。

当前验收状态：JSON package 独立仿真已通过，CLI 参数边界和 staged simulation 输出已通过；
完整前端端到端测试需要安装官方 StableHLO 的 `mlir` Python binding。

## 验证命令

```bash
PYTHONPATH=src /usr/bin/python3.12 -m unittest discover -s tests -v
PYTHONPATH=src /usr/bin/python3.12 -m compileall -q src tests examples
git diff --check
```
