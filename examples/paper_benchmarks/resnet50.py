"""ResNet50 benchmark row and representative bottleneck workload."""

from __future__ import annotations

import torch

from npu_ooo.frontend import make_workload

from .types import PaperBenchmarkSpec, PaperBenchmarkWorkload


class ResNet50BottleneckWorkload(torch.nn.Module):
    """A genuine Conv2D residual bottleneck-shaped PyTorch workload."""

    def __init__(self, channels: int = 16, bottleneck_channels: int = 4) -> None:
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.conv1 = torch.nn.Conv2d(3, bottleneck_channels, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(bottleneck_channels, channels, 1, bias=False)
        self.shortcut = torch.nn.Conv2d(3, channels, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(bottleneck_channels)
        self.bn2 = torch.nn.BatchNorm2d(bottleneck_channels)
        self.bn3 = torch.nn.BatchNorm2d(channels)
        self.bn_shortcut = torch.nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        residual = self.bn_shortcut(self.shortcut(x))
        value = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        value = torch.nn.functional.relu(self.bn2(self.conv2(value)))
        value = self.bn3(self.conv3(value))
        return torch.nn.functional.relu(value + residual)


class ResidualBottleneckProxy(torch.nn.Module):
    """Shape-preserving residual bottleneck used by the repeated model proxy."""

    def __init__(self, channels: int = 16, bottleneck_channels: int = 4) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, bottleneck_channels, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.conv3 = torch.nn.Conv2d(bottleneck_channels, channels, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(bottleneck_channels)
        self.bn2 = torch.nn.BatchNorm2d(bottleneck_channels)
        self.bn3 = torch.nn.BatchNorm2d(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = torch.nn.functional.relu(self.bn1(self.conv1(value)))
        value = torch.nn.functional.relu(self.bn2(self.conv2(value)))
        value = self.bn3(self.conv3(value))
        return torch.nn.functional.relu(value + residual)


class ResNet50ModelProxy(torch.nn.Module):
    """Scaled stem, repeated bottlenecks, global average and classifier."""

    def __init__(self, layer_count: int, channels: int = 16) -> None:
        super().__init__()
        if layer_count <= 0:
            raise ValueError("layer_count must be positive")
        self.layer_count = layer_count
        self.stem = torch.nn.Conv2d(3, channels, 3, stride=2, padding=1, bias=False)
        self.stem_bn = torch.nn.BatchNorm2d(channels)
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.blocks = torch.nn.ModuleList(
            ResidualBottleneckProxy(channels) for _ in range(layer_count)
        )
        self.classifier = torch.nn.Linear(channels, 10)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.pool(torch.nn.functional.relu(self.stem_bn(self.stem(value))))
        for block in self.blocks:
            value = block(value)
        value = torch.mean(value, dim=(-2, -1))
        return self.classifier(value)


SPEC = PaperBenchmarkSpec(
    "resnet50", "ResNet50", "cnn_residual", "inference", "float16", 128, None, (224, 224),
    None, None, None, 6.2, 9.3, 1.50, "resnet_bottleneck",
    ("stem", "full_model_depth", "global_average_pool", "classification_head"),
    "full_model", 16,
)


def build(
    variant: str = "micro",
    dtype: torch.dtype | None = None,
    *,
    layer_count: int = 1,
    model_scope: str = "one_block",
    seed: int | None = 0,
) -> PaperBenchmarkWorkload:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if model_scope not in {"one_block", "model_proxy"}:
        raise ValueError("model_scope must be 'one_block' or 'model_proxy'")
    if model_scope == "one_block" and layer_count != 1:
        raise ValueError("ResNet layer_count requires model_scope='model_proxy'")
    if variant == "micro":
        batch, height, width = 1, 16, 16
    elif variant == "paper_shape":
        batch, height, width = SPEC.batch_size, *SPEC.image_size  # type: ignore[misc]
    else:
        raise ValueError("variant must be 'micro' or 'paper_shape'")
    requested_dtype = dtype or getattr(torch, SPEC.dtype)
    if seed is not None:
        torch.manual_seed(seed)
    module = (
        ResNet50BottleneckWorkload()
        if model_scope == "one_block"
        else ResNet50ModelProxy(layer_count)
    ).eval().to(dtype=requested_dtype)
    inputs = (torch.randn(batch, 3, height, width, dtype=requested_dtype),)
    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )
    return PaperBenchmarkWorkload(
        SPEC,
        make_workload(
            module,
            inputs,
            experiment_id=SPEC.case_id,
            provenance={"construction": "paper_benchmark_builder", "variant": variant},
        ),
        variant,
        {
            "paper_reference_only": True,
            "simulation_dimensions": "scaled" if variant == "micro" else "paper_image_shape_representative_bottleneck",
            "requested_dtype": str(requested_dtype).removeprefix("torch."),
            "simulation_dtype": str(requested_dtype).removeprefix("torch."),
            "dtype_fallback": False,
            "full_model_materialized": False,
            "materialized_scope": model_scope,
            "layer_count": layer_count,
            "model_depth_proxy": "one_block" if model_scope == "one_block" else "sequential_repeated_blocks",
            "paper_evaluation_scope": SPEC.paper_evaluation_scope,
            "reference_layer_count": SPEC.reference_layer_count,
            "remaining_features": (
                list(SPEC.unsupported_features)
                if model_scope == "one_block"
                else ["exact_stage_topology", "paper_channel_widths"]
            ),
            "model_components": {
                "stem": model_scope == "model_proxy",
                "residual_bottlenecks": layer_count,
                "global_average_pool": model_scope == "model_proxy",
                "classification_head": model_scope == "model_proxy",
            },
            "model_statistics": {
                "parameter_count": parameter_count,
                "parameter_bytes": parameter_bytes,
                "input_bytes": sum(value.numel() * value.element_size() for value in inputs),
                "input_tensor_count": len(inputs),
            },
            "compiler_route": "torch.export->torch-xla->official-stablehlo",
        },
    )
