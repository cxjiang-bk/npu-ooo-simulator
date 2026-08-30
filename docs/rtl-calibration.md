# RTL Completion Trace 校准契约

`TISAProgram` 把 `load`、`matmul`、`store` 表达为独立 TISA stage。RTL
观测先转换为版本化的 `npu_ooo.systolic_mxu_profile.v1`，再交给普通 TISA event
backend 重放。profile 的 interval 字段明确 execution boundary。

## 输入格式

JSON schema：

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

`batch` 默认值为 1。四个 cycle 字段使用同一 RTL 时钟的 cycle index。CSV 使用相同
字段名，第一行写入表头。

仓库的 MXU testbench `rtl/unit_test/mxu/tb_mxu.sv` 提供 console-log 适配器，解析
`Prepared instruction`、`instruction accepted` 和 `Done Signal`，并按
`K1 * K0` 还原 profile 的 k（`K0` 默认 8，可通过参数覆盖）：

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

console-log 适配器获得 descriptor-to-completion interval，适合检查 shape、instruction
acceptance 与 `done_if.vld` 对齐。isolated compute calibration 需要 testbench 导出
matrix-array input handshake 和 final compute output/PSB handoff，并写入同一 schema。
控制指令 `END instruction accepted` 与 `task_done=1` 进入 trace metadata 的 skip
计数。

## Interval 映射

| interval | 起点 | 终点 | payload 映射 |
| --- | --- | --- | --- |
| `compute_start_to_compute_done` | compute_start_cycle | compute_done_cycle | isolated `matmul` primitive |
| `descriptor_issue_to_done` | descriptor_issue_cycle | psb_write_done_cycle（缺省时 compute_done_cycle） | full descriptor-to-completion payload |

`done_if.out done` 同时反映 MXU instruction manager 与 PSB write 链路。profile metadata
保留 interval，使 provider 选择正确的 payload boundary。

## 聚合规则

同一 `(batch, m, n, k)` 可以包含多条观测。支持：

- `max`：保守容量估计；
- `median`：稳定重复测量；
- `p95`：尾延迟分析。

记录提供 `initiation_interval_cycles` 时，II 使用相同聚合；字段缺省时按起始事件排序
计算相邻正差；单样本使用 shape duration 作为 II，形成可审计的保守回退。

## 使用方式

```bash
PYTHONPATH=src:. python3 -m npu_ooo.cli import-rtl-trace \
  --input examples/rtl/mxu_completion_trace.json \
  --output /tmp/mxu-rtl-profile.json \
  --interval compute_start_to_compute_done \
  --aggregation median \
  --unmatched-matmul error
```

将生成 profile 用作 timing config：

```bash
PYTHONPATH=src:. python3 -m npu_ooo.cli compile-model \
  --torch-module examples.torch_models:TwoMatmul \
  --input-shape 8,8 --input-shape 8,8 --input-shape 8,8 \
  --tile-size 4 \
  --timing-provider systolic_mxu_profile \
  --timing-config /tmp/mxu-rtl-profile.json \
  --output-dir out/matmul-rtl-profile
```

profile 保留 `trace_format`、`interval`、`aggregation`、record/shape 数量、source
和 calibration status。shape 匹配缺口使用 `--unmatched-matmul analytical` 时进入
mixed calibration，并在 artifact 中记录。

## 工具边界

Importer 接收已整理的 JSON/CSV/VCS log，输出 versioned profile。VCD/VPD/FSDB、VCS、
Verilator、Vivado、SCALE-Sim 的执行由上游 exporter 或独立 backend 负责；scheduler、
compiler 和 timing provider 共享同一 profile schema。
