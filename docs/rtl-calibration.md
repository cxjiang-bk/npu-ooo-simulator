# RTL Completion Trace 校准契约

本项目当前的 `TISAProgram` 把 `load`、`matmul` 和 `store` 建模为不同的 TISA
stage。因而 RTL 的 full instruction `accept -> done` 周期不能直接写入现有
`matmul` primitive。阶段 11.4 引入一个离线、版本化的 completion trace importer，先把
RTL/仿真器观测转换为 `npu_ooo.systolic_mxu_profile.v1`，再由普通 TISA event backend
重放。

## 输入格式

JSON 输入必须是：

```json
{
  "format": "npu_ooo.rtl_completion_trace.v1",
  "metadata": {
    "source": "rtl/mxu/tb_mxu.sv",
    "git_revision": "<rtl revision>",
    "testbench_config": "<config id>",
    "clock_period_ns": 1.0,
    "calibration_status": "rtl-observed"
  },
  "records": [
    {
      "instruction_id": "mxu-0",
      "batch": 1,
      "m": 4,
      "n": 12,
      "k": 8,
      "descriptor_issue_cycle": 10,
      "compute_start_cycle": 14,
      "compute_done_cycle": 37,
      "psb_write_done_cycle": 41
    }
  ]
}
```

`batch` 可以省略，默认是 `1`。`descriptor_issue_cycle`、
`compute_start_cycle`、`compute_done_cycle` 和 `psb_write_done_cycle` 都是相对于同一
RTL 时钟的 cycle index。CSV 也支持同名的扁平字段，第一行必须是字段名。

仓库当前 MXU testbench (`rtl/unit_test/mxu/tb_mxu.sv`) 还提供了一条 console-log 适配路径。
它解析 `Prepared instruction`、`instruction accepted` 和 `Done Signal` 三类 `$display`，
并把 `K1` 按 `K1 * K0` 还原成 profile 的 `k`；`K0` 默认是 8，也可以通过参数覆盖：

```bash
PYTHONPATH=src python3 -m npu_ooo.cli import-rtl-log \
  --input /path/to/tb_mxu.run.log \
  --output /tmp/mxu-trace.json \
  --k-per-tile 8
PYTHONPATH=src python3 -m npu_ooo.cli import-rtl-trace \
  --input /tmp/mxu-trace.json \
  --output /tmp/mxu-rtl-profile.json \
  --interval descriptor_issue_to_done \
  --aggregation median
```

这条路径只得到 descriptor-to-completion interval，因为 testbench 没有打印
compute-start/matrix-array handshake。它适合先检查 shape、instruction acceptance 和
`done_if.vld` 对齐；生成的 profile 不能被当前 isolated `matmul` provider 消费，provider
会明确报错。要校准 isolated compute，testbench 必须额外导出 matrix-array input handshake
和 final compute output/PSB handoff，并将这些字段写入本契约。
parser 会跳过 `END instruction accepted` 和 `task_done=1` 的控制指令，并把跳过数量写入
trace metadata，避免控制指令打乱普通 Matmul 的 FIFO 配对。

每条记录至少要能满足所选 interval：

| interval | 起点 | 终点 | 映射边界 |
| --- | --- | --- | --- |
| `compute_start_to_compute_done` | `compute_start_cycle` | `compute_done_cycle` | 映射到当前 isolated `matmul` primitive，默认值 |
| `descriptor_issue_to_done` | `descriptor_issue_cycle` | `psb_write_done_cycle`，缺失时回退 `compute_done_cycle` | 映射 full descriptor-to-completion，必须在实验中明确其不是 isolated matmul |

当前 RTL 的 `done_if.out done` 由 MXU instruction manager 和 PSB write 链路共同影响。
因此只有第一种 interval 可以直接校准现有 `ExecutionTask(primitive="matmul")`。
第二种 interval 仍然可以导出 profile，但 profile metadata 会记录 interval；如果要把
它解释成一个原子 full-MXU payload，应先调整 `CodegenBackend` 的 payload 边界。

## 聚合规则

同一 `(batch, m, n, k)` 允许有多条观测。`import-rtl-trace` 支持：

- `max`：取最大 duration，适合保守容量估计；
- `median`：取中位数，适合稳定无异常的重复测量；
- `p95`：取排序后的 nearest-rank P95，适合保留尾延迟。

如果记录中显式提供 `initiation_interval_cycles`，II 使用同样的聚合方法。否则从所选
interval 的起始事件排序后计算相邻正差；只有一个样本时，II 回退为该 shape 的聚合
duration。这个回退是保守的、可审计的，不在 importer 中猜测流水线深度。

## 使用方式

```bash
PYTHONPATH=src:. python3 -m npu_ooo.cli import-rtl-trace \
  --input examples/rtl/mxu_completion_trace.json \
  --output /tmp/mxu-rtl-profile.json \
  --interval compute_start_to_compute_done \
  --aggregation median \
  --unmatched-matmul error
```

生成文件可以直接作为后续仿真的 timing config：

```bash
PYTHONPATH=src:. python3 -m npu_ooo.cli compile-model \
  --torch-module examples.torch_models:TwoMatmul \
  --input-shape 8,8 \
  --input-shape 8,8 \
  --input-shape 8,8 \
  --tile-size 4 \
  --timing-provider systolic_mxu_profile \
  --timing-config /tmp/mxu-rtl-profile.json \
  --output-dir out/matmul-rtl-profile
```

生成的 `systolic_mxu_profile.v1` 会保留 `trace_format`、`interval`、
`aggregation`、record/shape 数量、source 和 calibration status。未命中的 Matmul 默认
`error`，只有显式选择 `--unmatched-matmul analytical` 才允许混合回退。

## 当前边界

Importer 不读取 VCD/VPD/FSDB，也不调用 VCS、Verilator、Vivado 或 SCALE-Sim；它要求
上游 exporter 已经把观测整理为上述 schema。这样工具链、RTL revision、testbench 参数和
区间选择都能在 profile 中复现。后续真实 trace exporter 可以输出 JSON 或 CSV，而不需要
修改 scheduler、compiler 或 timing provider。
