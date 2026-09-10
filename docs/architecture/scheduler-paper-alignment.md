# Scheduler 与 TISA 论文对齐说明

本文供后续 scheduler agent 和实现人员使用，专门说明当前
`cycle_event` scheduler 与 TISA 论文 Section IV/V、Figure 4、Algorithm 1/2 之间的
对应关系，尤其是 `Semantic Conflict Detection`、`Fu` 粒度和跨 EU 依赖。

本文不把论文中没有公开的数据结构或 RTL 细节扩展成“论文事实”。凡是当前项目的实现
选择，均明确标注为项目模型。

## 1. 结论先行

论文中的正确分层应理解为：

```text
全局 typed dependency readiness
        +
目标 EU 的本地 SemanticConflict(I, Fu[u])
        +
目标 EU 的结构资源检查
        ↓
issue
```

因此：

1. `Fu` 是每个调度 unit `u` 一个，即 `Fu[Tensor]`、`Fu[Vector]`、`Fu[DMA]`；不是所有 EU 共用一个 Fu。
2. 位于 `WQ[u]` 的候选在执行论文 Algorithm 2 时，主要检查目标 unit 的 `Fu[u]`。
3. 这不意味着候选只检查同 EU 依赖。`Deps` 的 source 是另一条 TISA instruction，依赖可以跨 EU。
4. Tensor producer 完成后，必须通过全局 completion/dependency notification 唤醒 Vector WQ 中的 consumer。
5. 当前代码已经有跨 EU 的显式 dependency 等待，但还没有真正实现 `SemanticConflict(I, Fu[u])`。

## 2. 论文明确给出的机制

论文定义 TISA 指令携带：

```text
OpType
Operands = TileShape + TileMem + AccessType
UnitMap = target unit / quantity / affinity
Deps = {(src, type, condition)}
```

论文 Figure 4 的主路径是：

```text
Reception Buffer
        ↓ semantic routing
WQ[Tensor] / WQ[Vector] / WQ[DMA]
        ↓ ready window + dependency resolution
IQ[Tensor] / IQ[Vector] / IQ[DMA]
        ↓
Exec[Tensor] / Exec[Vector] / Exec[DMA]
        ↓ completion feedback
dependent instructions wake up
```

论文明确说每个执行 unit `u` 维护自己的 in-flight semantic table：

```text
Fu = {
    operand index,
    start address,
    end address,
    access type,
    unit,
    instruction pointer
}
```

候选指令 `I` 被发往 unit `u` 时，执行：

```text
Hazard(I, Fu[u]) =
    exists r in Fu[u] such that SemanticConflict(I, r)
```

论文 Algorithm 2 的语义包括：

1. 不同 semantic scope 可以独立执行；
2. 不同地址区间可以独立执行；
3. 访问重叠时，结合 READ/WRITE 方向判断 RAW/WAR/WAW；
4. 结合 OpType compatibility 判断是否允许 overlap；
5. 如果不能安全重排，则阻塞候选。

论文还说明，指令完成后要：

```text
retire corresponding Fu entry
notify dependent instructions
update scheduling state
```

## 3. 论文没有详细说明、但算法成立所必需的部分

论文没有画出完整的 global dependency notification 结构，也没有说明依赖等待者是通过
全局表、completion broadcast、token scoreboard 还是 reverse dependency list 管理。

但从 `Deps = {(src, type, condition)}` 和“completion 后通知 dependent instructions”可以确定：

```text
跨 EU 依赖必须是全局可见的
```

例如：

```text
T: MXU 计算，写 PSB[P]
V: Vector/ARU 计算，读 PSB[P]

Deps(V) = { RAW(T, full_region_ready) }
```

运行过程应为：

```text
T → WQ[Tensor] → IQ[Tensor] → Exec[Tensor]
                                      |
                                      | completion(T)
                                      v
                               global dependency state
                                      |
                                      v
                              wake V in WQ[Vector]
                                      |
                                      v
                         SemanticConflict(V, Fu[Vector])
                                      |
                                      v
                            IQ[Vector] → Exec[Vector]
```

`V` 不需要检查 `Fu[Tensor]` 来等待 T；它通过自己的 typed dependency 等待 T 的完成。
但是，如果没有全局依赖状态，单纯检查 `Fu[Vector]` 会错误地让 V 提前读取 PSB[P]。

因此必须区分：

```text
跨 EU 数据依赖：global Deps / completion notification
本 EU 语义冲突：candidate vs Fu[u]
```

## 4. 当前项目的实际实现

### 4.1 Runtime 与 scheduler 输入

当前 `RuntimeSubmission` 先经过 loader，生成：

```text
LoadedDeviceProgram
  ├── BoundTISADescriptor
  │     TISA instruction、UnitMap、operand、Deps、completion token
  └── DescriptorEnvelope
        arrival cycle、chunk、descriptor order
```

`cycle.py` 只接收 `loaded` 和 `execution`：

```text
loaded    = scheduler-visible descriptors and arrival envelopes
execution = ExecutionBackend，管理 payload 和物理 EU instance
```

### 4.2 当前队列和 Fu 粒度

初始化代码为：

```python
self.wq = {name: [] for name in self.units}
self.iq = {name: [] for name in self.units}
self.fu = {name: set() for name in self.units}
```

这说明当前项目的 WQ、IQ、Fu 是按 `resource name` 划分的：

```text
WQ[MXU] / IQ[MXU] / Fu[MXU]
WQ[ARU] / IQ[ARU] / Fu[ARU]
WQ[DMA] / IQ[DMA] / Fu[DMA]
```

这与论文 Figure 4 的 unit-class 粒度一致。

但当前 Fu entry 只是：

```text
set(tisa_id)
```

它只用于计算：

```text
当前 resource 类别占用了多少 operand entry
```

它不是论文定义的 semantic table，还没有保存地址区间、access type、OpType 和重排属性。

### 4.3 当前 `wakeup()`

`wakeup()` 遍历 descriptor 的全部 dependencies，不限制 source 和 consumer 是否属于同一 EU：

```text
consumer ∈ WQ[Vector]
producer ∈ Fu[Tensor]
```

只要 consumer 的 dependency source 尚未 complete，consumer 就不能 wakeup。

因此，当前实现已经可以表达：

```text
MXU → ARU
ARU → DMA
DMA → MXU
```

但这属于预先生成的 explicit dependency readiness，不是 Semantic Conflict Detection。

### 4.4 当前 `_address_block()`

当前地址检查不查询 `Fu[resource]`，而是遍历所有更老、已经 receive 且尚未 complete 的 descriptor：

```text
older descriptors across all resources
```

它只判断：

```text
physical scope
allocation identity
byte range overlap
READ / WRITE / READ_WRITE
```

它没有判断：

```text
OpType compatibility
semantic scope compatibility
can_reorder_safely
reduction / atomic / psum special semantics
```

所以当前 `_address_block()` 是一个跨 EU 的物理地址 scoreboard，不是论文 Algorithm 2 的实现。

## 5. 当前实现与论文的差异表

| 机制 | 论文模型 | 当前项目 | 对齐结论 |
| --- | --- | --- | --- |
| Reception | Reception Buffer | `self.reception` | 基本一致 |
| Routing | 按 UnitMap 路由到 WQ[u] | dispatch 时写入 `wq[resource]` | 基本一致 |
| WQ/IQ | 每 unit 独立 WQ/IQ | 每 resource 独立 WQ/IQ | 基本一致 |
| Fu 粒度 | 每 unit 一个 `Fu[u]` | 每 resource 一个 Fu | 粒度基本一致 |
| Fu 内容 | semantic in-flight records | `set(tisa_id)` | 不一致 |
| 显式依赖 | typed Deps，可跨 EU | `wakeup()` 检查全部 dependencies | 基本一致 |
| completion 通知 | 完成后通知 dependent instructions | completion 后通过 `source.completed` 间接唤醒 | 语义近似，结构未显式建模 |
| SemanticConflict | candidate vs `Fu[u]` | 未实现 | 关键缺口 |
| 地址冲突 | 与 semantic compatibility 共同判断 | optional global address scoreboard | 不能替代论文机制 |
| 资源检查 | unit/resource availability | issue 阶段检查 | 基本一致 |
| 跨 EU address hazard | 依赖语义应全局可见 | global older-descriptor scan | 可作为项目保守扩展，但不是 Fu 检查 |

## 6. 应采用的对齐后架构

推荐把 scheduler 状态拆成三层：

```text
GlobalDependencyState
  dependency token → pending / ready
  dependency token → waiting instructions in any WQ

PerUnitFu
  Fu[unit] = semantic entries currently issued to that unit

StructuralResourceState
  EU instance、issue width、bank/port、tile window、Fu capacity
```

对每个候选 `I ∈ WQ[u]`，处理顺序应为：

```text
1. explicit dependencies ready?
       no  → stay in WQ[u]
       yes

2. SemanticConflict(I, Fu[u])?
       yes → stay in WQ[u]
       no

3. resource available(u)?
       no  → stay in WQ[u] or IQ[u]
       yes

4. promote to IQ[u] / issue
```

建议的逻辑结构：

```mermaid
flowchart LR
    R["Reception FIFO"] --> D["Semantic routing"]
    D --> WT["WQ[Tensor]"]
    D --> WV["WQ[Vector]"]
    D --> WD["WQ[DMA]"]

    WT --> DT["Deps ready + SemanticConflict(I, FuTensor)"]
    WV --> DV["Deps ready + SemanticConflict(I, FuVector)"]
    WD --> DD["Deps ready + SemanticConflict(I, FuDMA)"]

    DT --> IQT["IQ[Tensor]"]
    DV --> IQV["IQ[Vector]"]
    DD --> IQD["IQ[DMA]"]

    IQT --> ET["Exec[Tensor]"]
    IQV --> EV["Exec[Vector]"]
    IQD --> ED["Exec[DMA]"]

    ET --> F["Global completion / dependency notification"]
    EV --> F
    ED --> F
    F -. "wake dependents in any WQ" .-> WT
    F -. "wake dependents in any WQ" .-> WV
    F -. "wake dependents in any WQ" .-> WD
```

## 7. 关键实现原则

### 7.1 `wakeup()` 与 SemanticConflict 不要合并

二者处理的是不同问题：

```text
wakeup:
    producer 的 required completion/partial-ready 是否发生

SemanticConflict:
    当前 candidate 是否能与目标 EU 的 in-flight entry 并行
```

一个指令可能已经满足所有显式依赖，但仍然因为 `Fu[u]` 中的语义冲突不能 issue。
反过来，Fu[u] 没有冲突，也不能绕过尚未满足的跨 EU dependency。

### 7.2 SemanticConflict 应在 ready-window/select 阶段执行

论文 Algorithm 1 是从 WQ 选择 ready window 后执行 Algorithm 2，再把通过的指令提升到 IQ。
因此建议：

```text
WQ → ready window → dependency ready → SemanticConflict → IQ
```

由于 select 与 issue 之间存在 pipeline latency，issue 时可以再次验证一次，防止状态在两个
阶段之间发生变化。

### 7.3 跨 EU 依赖不能通过查询所有 Fu 替代

不建议把所有 `Fu[u]` 合并成一个全局 Fu，再让每个候选扫描全部 in-flight entries。这样会：

- 破坏论文的 per-unit decentralized 结构；
- 把不相关 EU 的 in-flight 状态引入本地冲突判断；
- 增加无意义的阻塞；
- 混淆“数据依赖”和“本地 semantic conflict”。

正确做法是：

```text
跨 EU：通过 global dependency token / waiter notification
本 EU：通过 Fu[u] 做 SemanticConflict
```

### 7.4 地址 scoreboard 与 SemanticConflict 的关系

当前 address scoreboard 可以保留，但应明确其角色：

```text
SemanticConflict:
    论文语义检查，基于 OpType、scope、TileMem、AccessType、reorder rule

Address scoreboard:
    runtime 物理地址和 allocation alias 的安全保护

Memory bank scoreboard:
    bank/port 结构冲突
```

如果 address scoreboard 产生的是硬 RAW/WAR/WAW dependency，它会在 wakeup 阶段阻塞指令；
这属于保守安全依赖。不能再声称它等价于论文的完整 SemanticConflict。

## 8. 跨 EU 验证案例

### Case A：MXU → Vector 的真实 RAW

```text
T: MXU write PSB[P]
V: Vector read PSB[P]
Deps(V) = RAW(T)
```

预期：

```text
T 可以进入 Fu[MXU]
V 留在 WQ[Vector]
T complete 后通知 V
V 再检查 Fu[Vector] 并 issue
```

错误实现：只查询 `Fu[Vector]`，导致 V 在 T 完成前执行。

### Case B：MXU 与 Vector 地址不相交

```text
T: MXU write PSB[P0]
V: Vector read PSB[P1]
P0 ∩ P1 = ∅
```

预期：T 和 V 可以同时执行，不应因为不同 EU 正在工作而互相阻塞。

### Case C：同一 Vector EU 的语义冲突

```text
V0: Vector write UB[A]
V1: Vector read UB[A]
V0 ∈ Fu[Vector]
V1 ∈ WQ[Vector]
```

预期：`SemanticConflict(V1, Fu[Vector])` 为 true，V1 等待 V0 完成或相应的安全边界。

### Case D：跨 EU 但无显式 dependency

```text
T: MXU write UB[A]
V: Vector read UB[A]
Deps(V) 缺失
```

预期不是让 V 直接执行，而是：

```text
编译/runtime dependency construction 必须补齐 RAW
```

SemanticConflict 不能替代缺失的 correctness dependency；否则候选可能根本无法被正确识别。

### Case E：WQ 中的 older producer 尚未 issue

```text
T: older DMA write A，仍在 WQ[DMA]
V: younger Vector read A，已经 ready
```

如果只查询 `Fu[Vector]`，V 看不到尚未 issue 的 T，可能错误越过 T。

因此，跨 EU 的正确性依赖必须在进入 WQ 前由 typed Deps / global semantic dependency state
表示，或者额外维护对 older waiting entries 的全局保护。仅依赖目标 EU 的 Fu 不足以覆盖这种情况。

## 9. 对当前代码的修改边界

后续实现应优先遵循以下边界：

1. 保留 `loaded` 作为 scheduler 输入，保留 `execution` 作为 payload/EU backend。
2. 保留 `wakeup()` 的跨 EU dependency readiness 语义。
3. 将 `self.fu[resource]` 从 `set(tid)` 扩展为包含 semantic operand/range 的 entry。
4. 在 `select()` 的 WQ ready-window 检查中增加 `semantic_conflict(candidate, fu[resource])`。
5. `issue()` 保留资源和 backend `can_accept()` 检查，并可重新验证 semantic conflict。
6. completion 时同时执行：释放 `Fu[resource]`、更新 completion token、通知所有 WQ 的等待者。
7. 不把所有 EU 的 Fu 合并为单一全局 Fu。
8. 不把当前 address scoreboard 直接命名为论文 Semantic Conflict Detection。
9. 对 runtime alias 生成的硬依赖保留 provenance，区分 `typed_dependency`、`runtime_alias` 和 `semantic_conflict`。

## 10. 最终对齐模型

```text
RuntimeSubmission
        ↓ loader
LoadedDeviceProgram
        ↓ receive
Reception FIFO
        ↓ semantic routing
WQ[u] for each execution unit
        ↓ global typed dependency readiness
ready candidate
        ↓ local SemanticConflict(candidate, Fu[u])
        ↓ structural resource checks
IQ[u]
        ↓ IssueRequest
ExecutionBackend / Exec[u]
        ↓ execution_done / partial_ready
        ├── remove semantic entry from Fu[u]
        ├── mark completion token ready
        └── notify dependent instructions in any WQ
```

一句话概括：

> 论文是“全局依赖唤醒 + per-EU Fu 语义冲突检查”；当前项目目前是“全局显式依赖唤醒 + 跨 EU 地址 scoreboard + per-EU Fu 容量统计”，尚未实现中间的 SemanticConflict 层。

相关代码入口：

- `src/npu_ooo/simulator/cycle.py:158`：WQ/IQ/Fu/ROB 状态初始化；
- `src/npu_ooo/simulator/cycle.py:368`：wakeup；
- `src/npu_ooo/simulator/cycle.py:393`：address scoreboard；
- `src/npu_ooo/simulator/cycle.py:414`：issue；
- `src/npu_ooo/simulator/cycle.py:514`：select；
- `src/npu_ooo/simulator/cycle.py:646`：逐周期主循环；
- `src/npu_ooo/runtime/loader.py:157`：runtime alias dependency；
- `src/npu_ooo/execution/analytical.py:188`：ExecutionBackend 接收和 EU instance 管理。
