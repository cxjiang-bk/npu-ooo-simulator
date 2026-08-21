"""Small model factories used by tests and experiments."""

from .elementwise import build_elementwise_case, build_elementwise_model
from .decoder_block import build_decoder_block_case, build_decoder_block_model
from .attention import build_attention_case, build_attention_model
from .transformer_block import build_transformer_block_case, build_transformer_block_model
from .reduce import build_reduce_case, build_reduce_model
from .softmax import build_softmax_case, build_softmax_model
from .norm import (
    build_layernorm_case,
    build_layernorm_model,
    build_rmsnorm_case,
    build_rmsnorm_model,
)
from .two_mm import build_two_matmul_case, build_two_matmul_model
from .model_presets import MODEL_PRESETS, ModelPreset, available_model_presets, build_model_preset

__all__ = [
    "build_elementwise_case",
    "build_elementwise_model",
    "build_decoder_block_case",
    "build_decoder_block_model",
    "build_attention_case",
    "build_attention_model",
    "build_transformer_block_case",
    "build_transformer_block_model",
    "build_reduce_case",
    "build_reduce_model",
    "build_softmax_case",
    "build_softmax_model",
    "build_rmsnorm_case",
    "build_rmsnorm_model",
    "build_layernorm_case",
    "build_layernorm_model",
    "build_two_matmul_case",
    "build_two_matmul_model",
    "MODEL_PRESETS",
    "ModelPreset",
    "available_model_presets",
    "build_model_preset",
]
