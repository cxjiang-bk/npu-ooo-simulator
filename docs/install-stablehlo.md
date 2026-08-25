# 安装 PyTorch、Torch-XLA 与 StableHLO

生产前端固定经过：

```text
torch.export -> Torch-XLA -> official StableHLO
```

没有 project exporter、textual fallback 或绕过 StableHLO 的直接路径。三个组件必须安装在同一个 Python 环境。

## 已验证版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.9.1 |
| Torch-XLA | 2.9.0 |
| StableHLO wheel | `1.12.1.1751868740+6f7b4ab8` |

本机使用：

```text
/usr/bin/python3.12
```

不要混用系统中其他 Python 的 `torch`、`torch_xla` 或 `mlir` 包。

## 安装

PyTorch 和 Torch-XLA：

```bash
/usr/bin/python3.12 -m pip install \
  'torch==2.9.1' \
  'torch-xla==2.9.0'
```

StableHLO 官方 wheel 发布在 OpenXLA StableHLO GitHub Releases。当前验证版本：

```bash
/usr/bin/python3.12 -m pip install \
  'https://github.com/openxla/stablehlo/releases/download/dev-wheels/stablehlo-1.12.1.1751868740%2B6f7b4ab8-cp312-cp312-linux_x86_64.whl'
```

如果上游 wheel URL 发生变化，应选择与 Python 3.12 和当前平台匹配的官方 wheel，并把新版本记录到实验 manifest。

## 验证依赖

```bash
/usr/bin/python3.12 - <<'PY'
from importlib.metadata import version

import torch
import torch_xla
from mlir.ir import Context, Module
import mlir.dialects.stablehlo as stablehlo

text = '''
module {
  func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
    return %x : tensor<2xf32>
  }
}
'''

with Context() as context:
    stablehlo.register_dialect(context)
    module = Module.parse(text)
    module.operation.verify()

print("torch", torch.__version__)
print("torch-xla", version("torch-xla"))
print("stablehlo", version("stablehlo"))
print("StableHLO parse/verify ok")
PY
```

## 验证完整前端

```bash
cd /home/lora/OpenTPU/npu-ooo-simulator

PYTHONPATH=src /usr/bin/python3.12 -m npu_ooo.cli compile-model \
  --torch-module examples.torch_models:AttentionMicrograph \
  --input-shape 1,4,8 \
  --input-shape 1,4,8 \
  --input-shape 1,4,8 \
  --tile-size 4 \
  --policy dynamic_ready_queue \
  --output-dir out/frontend-smoke
```

成功后检查：

```text
out/frontend-smoke/00_frontend/generated.mlir
out/frontend-smoke/00_frontend/stablehlo_module.json
out/frontend-smoke/03_tisa/tisa_program.json
out/frontend-smoke/manifest.json
```

`manifest.json` 应满足：

```json
{
  "frontend_path": "torch_export->torch_xla->official_stablehlo->canonical",
  "stablehlo_exporter": "torch-xla",
  "stablehlo_verified": true,
  "scheduler_target": "tisa"
}
```

Torch-XLA 不可用、版本不兼容或 StableHLO verify 失败时，编译会直接报错，不会切换到另一条前端。
