from __future__ import annotations

from enum import Enum


class ModelFamily(str, Enum):
    """Optional model classification retained as frontend provenance."""

    SYNTHETIC = "synthetic"
    CNN_RESIDUAL = "cnn_residual"
    ENCODER_TRANSFORMER = "encoder_transformer"
    DECODER_TRANSFORMER = "decoder_transformer"
    DECODER_REASONING = "decoder_reasoning"
    MOE_DECODER = "moe_decoder"


__all__ = ["ModelFamily"]
