# 安装官方 StableHLO

本项目的正式 StableHLO 路径使用 OpenXLA StableHLO 官方 wheel 提供的
`mlir.ir`、`mlir.dialects.stablehlo`、dialect 注册、parser 和 verifier。StableHLO
不是 PyPI 上的普通稳定包；官方 wheel 发布在 OpenXLA StableHLO GitHub Releases 的
`dev-wheels` release 中，并按 Python ABI 和平台区分。

## 当前已验证组合

| 项目 | 已验证值 |
| --- | --- |
| Python | CPython 3.12 / Linux x86_64 |
| PyTorch | 2.9.1 |
| StableHLO wheel | `1.12.1.1751868740+6f7b4ab8` |
| torch-xla | 2.9.0（可选 exporter） |
| 导入模块 | `mlir.ir`、`mlir.dialects.stablehlo` |
| verifier | `module.operation.verify()` |

当前机器的默认 `python3` 是 3.14，而已验证的官方 wheel 是 cp312，因此运行真实
PyTorch/StableHLO 路径时必须显式使用 `/usr/bin/python3.12`。

Debian 的 system Python 启用了 PEP 668，且当前没有 `python3.12-venv`。可以把 wheel
安装到用户 site-packages：

```bash
/usr/bin/python3.12 -m pip install \
  --target "$HOME/.local/lib/python3.12/site-packages" \
  'https://github.com/openxla/stablehlo/releases/download/dev-wheels/stablehlo-1.12.1.1751868740%2B6f7b4ab8-cp312-cp312-linux_x86_64.whl' \
  --upgrade
```

其他 Python/平台不要复用上述 URL。应从官方 release 选择匹配的
`cpXY-cpXY-{platform}.whl`，并在项目 smoke 通过后记录版本。

## 安装验证

```bash
/usr/bin/python3.12 - <<'PY'
from importlib.metadata import version
from mlir.ir import Context, Module
import mlir.dialects.stablehlo as stablehlo

with Context() as context:
    stablehlo.register_dialect(context)
    module = Module.parse("""
module {
  func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
    return %x : tensor<2xf32>
  }
}
""")
    module.operation.verify()

print("StableHLO", version("stablehlo"), "parse/verify ok")
PY
```

论文同类的 torch-xla exporter 可以安装为项目可选依赖。在标准虚拟环境中使用：

```bash
/usr/bin/python3.12 -m pip install -e '.[torch-xla]'
```

当前机器的 system Python 受 PEP 668 管理，已有 torch 及 torch-xla 的运行依赖，因此
本轮使用下列命令只安装已验证的 torch-xla wheel，避免覆盖系统 NumPy：

```bash
/usr/bin/python3.12 -m pip install \
  --target "$HOME/.local/lib/python3.12/site-packages" \
  'torch-xla==2.9.0' \
  --no-deps \
  --upgrade
```

## 编译验证

```bash
PYTHONPATH=src:. /usr/bin/python3.12 -m npu_ooo.cli compile-model \
  --torch-module examples.torch_models:attention_block \
  --input-shape 1,4,8 \
  --tile-size 4 \
  --arch minimal \
  --policy dynamic_ready_queue \
  --through-stablehlo \
  --stablehlo-backend official \
  --output-dir out/torch-attention-official-stablehlo
```

成功时 `manifest.json` 应包含：

```json
{
  "stablehlo_backend": "official",
  "stablehlo_verified": true,
  "stablehlo_producer": "project-stablehlo-legalizer",
  "stablehlo_verifier": "official-stablehlo-mlir",
  "stablehlo_fallback": false
}
```

`official` 是默认值，并在依赖缺失或 verifier 失败时直接报错。`auto` 会在官方绑定
不可用时显式回退到项目 textual subset，并把原因写入 manifest；`textual` 仅用于
dependency-light regression，不应作为论文结果的正式编译路径。

torch-xla exporter smoke：

```bash
PYTHONPATH=src:. PJRT_DEVICE=CPU /usr/bin/python3.12 -m npu_ooo.cli compile-model \
  --torch-module examples.torch_models:attention_micrograph \
  --input-shape 2,4,8 --input-shape 2,4,8 --input-shape 2,4,8 \
  --tile-size 2 \
  --through-stablehlo \
  --stablehlo-exporter torch-xla \
  --stablehlo-backend official \
  --output-dir out/torch-xla-attention
```

## 边界说明

官方 StableHLO wheel 提供 dialect、MLIR 对象模型、parser、printer 和 verifier，
不直接提供 PyTorch exporter。本项目当前链路是：

```text
torch.export
    -> 项目 FrontendImport / StableHLO legalization
    -> 官方 StableHLO assembly + verifier
    -> StableHLO importer
    -> Canonical OperatorGraph
```

因此 project exporter 路径保证中间 IR 是官方可解析、可验证的 StableHLO；此外
torch-xla exporter 已对 Matmul、attention micrograph 和完整 attention block 接通。
importer 已支持受约束的 Linear reshape folding 和 LayerNorm `batch_norm_training` recovery；
动态 shape、一般多结果消费和其他未覆盖 op 仍会在 frontend/compiler boundary 显式失败，
下游 Graph Compiler/TISA/backend 契约不需要因此改变。
