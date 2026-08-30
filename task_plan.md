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

生产入口统一使用 compile-model --torch-module MODULE:CLASS。测试 fixture 可以直接
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

- [ ] symbolic shape 统一 binding；
- [ ] dynamic index、dynamic layout 和 stride-aware transform；
- [ ] 完整模型 proxy 的 tile/MAC/traffic 对账样例。

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

1. symbolic shape、dynamic index、layout 语义；
2. DeepSeek 与完整模型 repetition；
3. scheduler 微结构和控制开销校准；
4. 外部 timing/memory/RTL backend；
5. 论文规模 source-derived/RTL-observed 矩阵。

## 验证命令

```bash
PYTHONPATH=src /usr/bin/python3.12 -m unittest discover -s tests -v
PYTHONPATH=src /usr/bin/python3.12 -m compileall -q src tests examples
git diff --check
```
