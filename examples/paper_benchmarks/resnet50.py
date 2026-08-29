"""ResNet50 benchmark row and representative bottleneck workload."""

from __future__ import annotations

import torch

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


SPEC = PaperBenchmarkSpec(
    "resnet50", "ResNet50", "cnn_residual", "inference", "float16", 128, None, (224, 224),
    None, None, None, 6.2, 9.3, 1.50, "resnet_bottleneck",
    ("stablehlo.convolution", "stablehlo.batch_norm_inference", "pooling"),
)


def build(variant: str = "micro", dtype: torch.dtype | None = None) -> PaperBenchmarkWorkload:
    if variant == "micro":
        batch, height, width = 1, 16, 16
    elif variant == "paper_shape":
        batch, height, width = SPEC.batch_size, *SPEC.image_size  # type: ignore[misc]
    else:
        raise ValueError("variant must be 'micro' or 'paper_shape'")
    requested_dtype = dtype or getattr(torch, SPEC.dtype)
    torch.manual_seed(0)
    module = ResNet50BottleneckWorkload().eval().to(dtype=requested_dtype)
    inputs = (torch.randn(batch, 3, height, width, dtype=requested_dtype),)
    return PaperBenchmarkWorkload(
        SPEC, module, inputs, variant,
        {
            "paper_reference_only": True,
            "simulation_dimensions": "scaled" if variant == "micro" else "paper_image_shape_representative_bottleneck",
            "requested_dtype": str(requested_dtype).removeprefix("torch."),
            "simulation_dtype": str(requested_dtype).removeprefix("torch."),
            "dtype_fallback": False,
            "full_model_materialized": False,
            "compiler_route": "torch.export->torch-xla->official-stablehlo",
        },
    )
