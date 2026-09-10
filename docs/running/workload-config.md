# 通用 PyTorch Workload 配置

`compile --config` 和 `compile-and-sim --config` 用同一份 schema-v1 JSON 构造
`Workload(module, args, kwargs)`，再统一调用 `compile_torch_module()`：

```text
JSON / Python workload factory
  -> Workload(module, args, kwargs, input_signature, provenance)
  -> torch.export
  -> Torch-XLA
  -> official StableHLO parse/verify
  -> GC / FC / TISA generator / target lowering / backend
```

该入口独立于 paper benchmark registry。它没有扩大编译器支持的 PyTorch operation 集合；
不支持的 operation 会在正式前端或后续 capability 检查中报错，不会替换成 proxy。

## 1. 可运行示例

仓库首先提供五份通用、固定 shape 配置：

| 文件 | 构造方式 | 主要输入 |
| --- | --- | --- |
| `configs/workloads/matmul.json` | 声明式模型 | 两个命名 float tensor |
| `configs/workloads/attention.json` | 声明式模型 | 命名 Q/K/V 和 additive mask |
| `configs/workloads/attention-causal-factory.json` | Python factory | Q/K/V 与 Python 构造的 causal mask |
| `configs/workloads/bert-multilayer.json` | 声明式模型 | 两层真实 block、int64 token/position/type IDs、float mask |
| `configs/workloads/llama-decode-fixed-window.json` | 声明式模型 | seqlen=1、固定 window=4 的 K/V cache、RoPE 和 mask |

论文模型 registry 的其余 workload 也有可独立运行的配置：

| 模型族 | One-block / bottleneck | 多层或 model proxy |
| --- | --- | --- |
| BERT | `bert-oneblock.json` | `bert-multilayer.json`（可配置两层 encoder shell） |
| ResNet-50 | `resnet50-bottleneck.json` | `resnet50-2block-model-proxy.json` |
| GPT-J | `gptj-prefill-oneblock.json` | `gptj-prefill-2layer-model-proxy.json` |
| LLaMA2 | `llama-prefill-oneblock.json`、`llama-decode-fixed-window.json` | `llama-prefill-2layer-model-proxy.json` |
| DeepSeek | `deepseek-prefill-dense-oneblock.json`、`deepseek-prefill-moe-oneblock.json` | `deepseek-prefill-2layer-model-proxy.json`、`deepseek-decode-dense-fixed-window.json`、`deepseek-decode-moe-fixed-window.json` |

文件名刻意保留 `oneblock`、`2layer-model-proxy` 和 `fixed-window`：前者只包含一个重复
计算块；后者真实构造两层及 embedding/head 等现有模型 shell，但仍不是论文完整层数和宽度；
decode 配置只有一个 block 的固定容量 cache。DeepSeek 当前沿用已有 builder 的
`bfloat16 -> float32` 仿真 fallback，并在 Workload provenance 中记录实际 dtype。

分别编译和仿真：

```bash
cd /home/jiangcx/LPU/npu-ooo-simulator

PYTHONPATH=src python3.12 -m npu_ooo.cli compile \
  --config configs/workloads/bert-multilayer.json \
  --output-dir out/bert-compile

PYTHONPATH=src python3.12 -m npu_ooo.cli simulate \
  --compile-dir out/bert-compile \
  --event-backend cycle_event \
  --policy dynamic_ready_queue \
  --output-dir out/bert-dynamic

PYTHONPATH=src python3.12 -m npu_ooo.cli compile-and-sim \
  --config configs/workloads/attention.json \
  --output-dir out/attention
```

独立 `simulate` 只读取 compile package，不会重新导入配置中的模型工厂，也不依赖 PyTorch
前端。比较 static/dynamic 时应复用同一个 compile package。

## 2. 顶层结构

```json
{
  "schema_version": 1,
  "seed": 0,
  "experiment_id": "my-static-workload",
  "model": {},
  "inputs": {},
  "compile": {},
  "runtime": {}
}
```

- `schema_version` 必须为 `1`。
- `seed` 默认为 `0`，在模型构造和输入生成之前同时设置 PyTorch/Python RNG。
- `experiment_id` 进入默认 `model_id` 和 provenance。
- `model + inputs` 表示声明式构造；也可以用顶层 `workload`，两种方式必须二选一。
- `compile` 只映射已有编译选项。
- `runtime` 只由 `compile-and-sim` 消费；`compile` 会校验并记录它，但不会生成 runtime
  submission。独立 `simulate` 继续使用自己的 `--runtime-config` 和 CLI 参数。

同一选项的优先级是：**显式 CLI > JSON > 默认值**。例如配置写
`"arch": "lpu-like"`，命令行 `--arch minimal` 会覆盖它；argparse 的默认
`minimal` 不会意外覆盖 JSON。`--torch-module`、`--input-shape`、`--input-dtype` 是一整套
旧简写输入源，不能与 `--config` 混用。

目标机器是一个整体选择：显式 CLI `--arch` 会替换 JSON 继承的 `machine_config`；若命令行
同时显式给出 `--arch` 和 `--machine-config`，则沿用既有规则，由完整 machine 文件优先。

## 3. 声明式模型与输入

### 3.1 模型构造

```json
"model": {
  "factory": "examples.configurable_models:ConfigurableBertProxy",
  "kwargs": {
    "num_layers": 2,
    "hidden_size": 8,
    "num_heads": 2,
    "intermediate_size": 16
  },
  "dtype": "float32"
}
```

`factory` 使用 `MODULE:QUALIFIED_NAME` 导入，不执行 `eval`。`kwargs` 传给模型构造器；
它们不是 `forward` 输入。factory 必须返回 `torch.nn.Module`。`dtype` 仅转换模型的浮点
参数和 buffer，允许 `float32`、`float16`、`bfloat16`；整数 buffer 不会被改成浮点。

层数和 hidden/head/intermediate size 是实际构造参数。示例 BERT proxy 的
`num_layers=2` 会创建两个独立参数化的 `EncoderLayer`，不是在统计结果上乘 2。不过它仍是
本项目的 scheduler research proxy，不是带预训练权重、完整算子和数值等价保证的 BERT。

### 3.2 Forward 输入

`inputs.args` 是 positional 参数数组，`inputs.kwargs` 是名称到输入节点的映射：

```json
"inputs": {
  "args": [],
  "kwargs": {
    "input_ids": {
      "kind": "tensor",
      "shape": [1, 4],
      "dtype": "int64",
      "init": {"kind": "randint", "low": 0, "high": 32}
    },
    "attention_mask": {
      "kind": "tensor",
      "shape": [1, 1, 4, 4],
      "dtype": "float32",
      "init": "zeros"
    }
  }
}
```

tensor dtype 各自独立，支持 `float32`、`float16`、`bfloat16`、`int64`、`int32`、
`int16`、`int8`、`uint8` 和 `bool`。token ID 应使用整数 dtype 和 `randint`；通用层不会
根据输入名称猜测 mask 语义，也不会把输入统一转换成 `model.dtype`。

shape 是编译时确定的正整数数组。`shape: []` 明确表示零秩 scalar tensor；JSON 中直接写
`1`、`0.5`、`true` 或 `null` 则分别是 Python 标量或 `None`，两者不会混淆。

支持的初始化方式：

| `init.kind` | 字段 | 约束 |
| --- | --- | --- |
| `zeros` / `ones` | 无 | 所有支持的 dtype |
| `randn` | 无 | 仅浮点 tensor |
| `randint` | `low`、`high` | 整数/bool tensor，且 `low < high` |
| `explicit` | `values` | 数据元素数必须与 shape 一致 |
| `arange` | 可选 `start`、`step` | 按元素数生成后 reshape，`step != 0` |
| `file` | `path` | 从 JSON 数组或 `{ "values": ... }` 读取显式数据 |

`file.path`、`compile.machine_config` 以及 `runtime` 中的 scheduler/timing/availability 文件
都相对于 workload JSON 所在目录解析，而不是当前 shell 目录。

### 3.3 嵌套输入

JSON 数组默认保留为 Python `list`；显式 tuple/list/dict 使用带 `kind` 的节点：

```json
{
  "kind": "tuple",
  "items": [
    {"kind": "list", "items": [1, 2]},
    {"kind": "dict", "items": {"enabled": true}}
  ]
}
```

对象节点必须有明确 `kind`，未知或无法表示的结构会报错，不会静默展平。加载器使用
`inspect.signature(module.forward).bind(...)` 检查 args/kwargs，并在
`input_signature.json` 中保存输入路径、forward 参数名、容器结构、tensor dtype/shape 和
初始化方式。当前 Torch-XLA 2.9 不接受带 example kwargs 的 ExportedProgram；编译入口会在
绑定签名后将普通 positional-or-keyword kwargs 等价位置化。显式 keyword-only 或 `**kwargs`
输入会给出明确诊断。

## 4. Python workload factory

复杂 mask、RoPE、嵌套 KV-cache 或特殊包装不必写成庞大的 JSON DSL。配置可以改用：

```json
"workload": {
  "factory": "examples.configurable_models:build_attention_workload",
  "kwargs": {
    "batch_size": 1,
    "num_heads": 2,
    "sequence_length": 4,
    "head_dim": 4,
    "dtype": "float32"
  }
}
```

factory 接收这些 JSON kwargs，并必须返回 `npu_ooo.frontend.Workload`：

```python
from npu_ooo.frontend import make_workload

def build_my_workload(sequence_length=4):
    module = MyModule().eval()
    inputs = (...)
    return make_workload(
        module,
        inputs,
        kwargs={"attention_mask": mask},
        experiment_id="my-workload",
        provenance={"mask": "causal_additive"},
    )
```

通用 compiler 不依赖 `PaperBenchmarkSpec`、case ID 或模型名称分支。`paper-matrix` 在更高层
保留论文行信息和实验组合，然后把 builder 结果包装成同一个 Workload 再调用
`compile_torch_module()`。

## 5. Compile 与 runtime 字段

`compile` 可使用：`arch`、`machine_config`、`tile_size`、`tile_size_candidates`、
`softmax_algorithm`、`onchip_handoff`、`codegen_backend` 和 `model_id`。含义与 README 的
现有 CLI 选项完全相同，没有第二套硬件配置。

`runtime` 可使用现有 `compile-and-sim` 选项对应的 snake_case 字段，包括
`event_backend`、`policy`、scheduler 容量、`address_scoreboard`、`runtime_policy`、
chunk/launch/synchronization、固定 state invocation 参数以及 timing/scheduler 文件。
它不新增 runtime dimension、动态 tile 或动态计算量语义。

解析器拒绝未知字段、非法 dtype/shape/range、显式数据大小不符、构造/forward 参数不匹配，
错误会尽量指向准确路径，例如 `inputs.kwargs.input_ids.dtype`。

## 6. 可复现产物与静态边界

每个 JSON 编译包在 `00_frontend/` 增加：

```text
workload.json                  # 实际 module、input signature 和 provenance
input_signature.json          # 完整 args/kwargs 输入树
resolved_workload_config.json # 配置路径/hash 及 CLI 覆盖后的实际 compile/runtime 配置
```

原有 StableHLO、GC/FC、TISA、TargetPlan、MemoryPlan、manifest 和 hash 语义保持不变。

当前所有 tensor extent 都在编译前确定。改变 seqlen、kvlen、cache capacity/window、batch、
hidden size 或会改变实际图/计算 shape 的构造参数，需要生成新配置并重新编译。
`dynamic_indices` 和 `dynamic_layouts` 只绑定已编译契约中的索引与布局，不能使一个编译包
适配任意序列长度。
