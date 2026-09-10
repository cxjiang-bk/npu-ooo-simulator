# 编译期静态排程与同步

## 定义与论文边界

本项目将新的 `static_streams` 定义为：编译器对最终 target TISA 做资源约束 list
scheduling，生成每个 EU 实例的固定指令流，并显式插入 `set/wait/fence`。固定的是流内顺序
与同步关系，不是绝对执行周期。论文 Section III、VI 和 VIII-A 只公开 Static 使用编译期
重排、多阶段流水和 fence-based dependency management；事件编码、消费方式和控制成本没有
公开，以下内容均是项目实现假设，不声称是 Epoch RTL。

旧 `static_pipeline` 保持原行为：按 runtime submission 的全局下一条 select/issue，作为
兼容/reference 策略。新基线必须显式选择：

```bash
npu-ooo simulate --compile-dir out/model-compile \
  --event-backend cycle_event --policy static_streams \
  --output-dir out/model-static
```

## 编译位置与共享工作负载

```text
final target TISA + payload + MemoryPlan + correctness dependencies
  -> resource-constrained static list scheduling
  -> per-EU streams + set/wait/fence
  -> static_control_program.json
```

同步生成位于 target lowering 和 MemoryPlan 之后、runtime 之前。正文 TISA、payload、operand、
MemoryPlan 和正确性依赖只生成一次：

- Static 消费正文和 `StaticControlProgram`；
- Dynamic 只消费正文语义/地址/资源状态，`static_controls_consumed=false`；
- `shared_workload_hash` 必须相同；static/dynamic control hash 分开记录。

## 控制表示

`04_backend/static_control_program.json` 包含：

- `StaticInstructionStream(resource, instance)`：每个逻辑 EU 流；
- `issue`：提交共享正文中的一条 target TISA；
- `set`：等待 producer 的真实 feedback 后发布代次化事件；
- `wait`：等待一个数据事件，非消费式，允许多消费者；
- `fence`：等待一个或多个 allocation/buffer reuse 事件，作用域不是全局 barrier；
- iteration、stage、buffer slot/allocation、估计起止周期和来源 provenance。

事件运行时身份为：

```text
invocation_id :: compile_event_id :: generation
```

因此同一事件模板在不同 invocation、iteration 和 slot 数据版本之间不会误匹配。

## 执行语义

每个流只检查自己的 head command：

1. `issue` 等 descriptor 到达和 ExecutionBackend 接收；
2. `set` 等匹配的 physical/partial feedback，producer 刚提交不会置位；
3. `wait/fence` 未满足只阻塞当前流，其他流继续；
4. 事件不被 wait 消费；
5. invocation 在全部正文和尾部 set 完成后结束。

静态执行器不调用 Dynamic 的 dependency-ready 判定。编译阶段用共享依赖图作独立 oracle，
缺少 wait/fence 会让 artifact 校验失败。Runtime 新增的物理 alias 若不在编译依赖的传递闭包
中，`static_streams` 拒绝该绑定，避免 runtime 偷偷重新排程。

ExecutionBackend 仍唯一拥有 payload、EU busy、物理完成和内部 task trace；静态执行器不会
维护第二套冲突的 EU busy 状态。

## 控制成本

`SchedulerPipelineConfig` 新增：

| 字段 | 默认 | 含义 |
| --- | ---: | --- |
| `control_width` | 1 | 每周期最多启动的 set/wait/fence 数 |
| `control_latency` | 1 | set 控制处理延迟 |
| `wait_latency` | 1 | 已满足 wait 的处理延迟 |
| `fence_latency` | 1 | 已满足 fence 的处理延迟 |

默认非零，且标记为 uncalibrated project assumption。零成本仅用于隔离实验：

```bash
--scheduler-config configs/scheduler/static_control_zero_cost.json
```

另有 `static_control_slow.json` 和 `static_control_wide.json` 用于受控敏感性实验。

## 当前限制

- list scheduler 使用编译 payload 的声明时长，不读取未来运行 trace；
- 逻辑 stream instance 尚不约束 ExecutionBackend 的具体物理 instance 选择；当前内置机器
  的主要验证路径为每类单实例；
- control 时序、事件存储容量和 Epoch 编码未硬件校准；
- static 不包含 CPU 式推测、异常或回滚；
- buffer slot 采用 MemoryPlan 已有分配，排程不会另行改变 allocation。
