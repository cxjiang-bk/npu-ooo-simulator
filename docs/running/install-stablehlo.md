# 安装 PyTorch、Torch-XLA 与 StableHLO

生产前端采用：

```text
torch.export -> Torch-XLA -> official StableHLO
```

三个组件安装在同一个 Python 环境，保证版本和 MLIR bindings 一致。

## 已验证版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.9.1 |
| Torch-XLA | 2.9.0 |
| StableHLO wheel | `1.12.1.1751868740+6f7b4ab8` |

本机解释器：

```text
/usr/bin/python3.12
```

## 安装

```bash
/usr/bin/python3.12 -m pip install \
  'torch==2.9.1' \
  'torch-xla==2.9.0'
```

StableHLO 官方 wheel：

```bash
/usr/bin/python3.12 -m pip install \
  'https://github.com/openxla/stablehlo/releases/download/dev-wheels/stablehlo-1.12.1.1751868740%2B6f7b4ab8-cp312-cp312-linux_x86_64.whl'
```

上游 wheel 地址变化时，选择与 Python 3.12 和平台匹配的 OpenXLA 官方 wheel，并把版本
写入实验 manifest。

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

PYTHONPATH=src /usr/bin/python3.12 -m npu_ooo.cli compile-and-sim \
  --torch-module examples.torch_models:AttentionMicrograph \
  --input-shape 1,4,8 --input-shape 1,4,8 --input-shape 1,4,8 \
  --tile-size 4 --policy dynamic_ready_queue \
  --output-dir out/frontend-smoke
```

重点产物：

```text
out/frontend-smoke/00_frontend/generated.mlir
out/frontend-smoke/00_frontend/stablehlo_module.json
out/frontend-smoke/03_tisa/tisa_program.json
out/frontend-smoke/manifest.json
```

manifest 关键字段：

```json
{
  "frontend_path": "torch_export->torch_xla->official_stablehlo->canonical",
  "stablehlo_exporter": "torch-xla",
  "stablehlo_verified": true,
  "scheduler_target": "tisa"
}
```

Torch-XLA、StableHLO bindings 和 verify 结果共同决定前端编译状态；诊断信息写入 CLI
错误输出和 frontend artifact。
