from __future__ import annotations

"""Declared StableHLO operations accepted by the canonical importer.

torch-xla owns framework-to-StableHLO legalization.  This registry defines
the narrower contract owned by this project: which verified StableHLO
operations can be assigned a canonical semantic family, whether a compiler
recovery pass must consume them, and which backend capability they require.
"""

from dataclasses import dataclass

from npu_ooo.ir import SemanticOpType


@dataclass(frozen=True)
class StableHLOOpCapability:
    op_name: str
    semantic_family: str
    min_operands: int
    max_operands: int
    requires_recovery: bool = False
    backend_capability_key: str | None = None

    def supports_arity(self, operand_arity: int) -> bool:
        return self.min_operands <= operand_arity <= self.max_operands


def _pointwise(op_name: str, operand_arity: int) -> StableHLOOpCapability:
    return StableHLOOpCapability(
        op_name=f"stablehlo.{op_name}",
        semantic_family=SemanticOpType.ELEMENTWISE.value,
        min_operands=operand_arity,
        max_operands=operand_arity,
        backend_capability_key=f"pointwise.{op_name}",
    )


_CAPABILITIES = {
    capability.op_name: capability
    for capability in (
        *(
            _pointwise(name, 1)
            for name in (
                "abs",
                "convert",
                "cosine",
                "exponential",
                "log",
                "logistic",
                "negate",
                "rsqrt",
                "sine",
                "sqrt",
                "tanh",
            )
        ),
        *(
            _pointwise(name, 2)
            for name in (
                "add",
                "divide",
                "maximum",
                "minimum",
                "multiply",
                "power",
                "subtract",
            )
        ),
        StableHLOOpCapability(
            op_name="stablehlo.dot_general",
            semantic_family=SemanticOpType.MATMUL.value,
            min_operands=2,
            max_operands=2,
            backend_capability_key="matmul",
        ),
        StableHLOOpCapability(
            op_name="stablehlo.convolution",
            semantic_family=SemanticOpType.CONV2D.value,
            min_operands=2,
            max_operands=2,
            backend_capability_key="convolution",
        ),
        StableHLOOpCapability(
            op_name="stablehlo.reduce",
            semantic_family=SemanticOpType.REDUCE.value,
            min_operands=2,
            max_operands=2,
            backend_capability_key="reduce",
        ),
        StableHLOOpCapability(
            op_name="stablehlo.transpose",
            semantic_family=SemanticOpType.TRANSPOSE.value,
            min_operands=1,
            max_operands=1,
            requires_recovery=True,
        ),
        StableHLOOpCapability(
            op_name="stablehlo.reshape",
            semantic_family=SemanticOpType.RESHAPE.value,
            min_operands=1,
            max_operands=1,
            requires_recovery=True,
        ),
        StableHLOOpCapability(
            op_name="stablehlo.broadcast_in_dim",
            semantic_family=SemanticOpType.RESHAPE.value,
            min_operands=1,
            max_operands=1,
            requires_recovery=True,
        ),
        StableHLOOpCapability(
            op_name="stablehlo.slice",
            semantic_family="slice",
            min_operands=1,
            max_operands=1,
            requires_recovery=True,
            backend_capability_key="slice",
        ),
        StableHLOOpCapability(
            op_name="stablehlo.concatenate",
            semantic_family="concatenate",
            min_operands=2,
            max_operands=32,
            requires_recovery=True,
            backend_capability_key="concatenate",
        ),
        StableHLOOpCapability(
            op_name="stablehlo.batch_norm_training",
            semantic_family="stablehlo.batch_norm_training",
            min_operands=3,
            max_operands=3,
            requires_recovery=True,
        ),
        StableHLOOpCapability(
            op_name="stablehlo.batch_norm_inference",
            semantic_family=SemanticOpType.BATCH_NORM.value,
            min_operands=5,
            max_operands=5,
            backend_capability_key="batch_norm",
        ),
        StableHLOOpCapability(
            op_name="stablehlo.reduce_window",
            semantic_family=SemanticOpType.POOL.value,
            min_operands=2,
            max_operands=2,
            requires_recovery=True,
            backend_capability_key="pool",
        ),
        # These composite spellings are accepted by the dependency-light
        # textual route for compatibility.  Official StableHLO producers
        # normally emit primitive operation sequences instead.
        StableHLOOpCapability(
            op_name="stablehlo.softmax",
            semantic_family=SemanticOpType.SOFTMAX.value,
            min_operands=1,
            max_operands=1,
            backend_capability_key="softmax",
        ),
        StableHLOOpCapability(
            op_name="stablehlo.rms_norm",
            semantic_family=SemanticOpType.RMSNORM.value,
            min_operands=1,
            max_operands=1,
            backend_capability_key="rmsnorm",
        ),
        StableHLOOpCapability(
            op_name="stablehlo.layer_norm",
            semantic_family=SemanticOpType.LAYERNORM.value,
            min_operands=1,
            max_operands=3,
            backend_capability_key="layernorm",
        ),
    )
}

_ALIASES = {
    "stablehlo.dot": "stablehlo.dot_general",
    "stablehlo.mul": "stablehlo.multiply",
    "stablehlo.rmsnorm": "stablehlo.rms_norm",
    "stablehlo.layernorm": "stablehlo.layer_norm",
}


def normalize_stablehlo_op_name(op_name: str) -> str:
    normalized = op_name.strip().lower().replace("::", ".")
    if normalized.startswith("mhlo."):
        normalized = "stablehlo." + normalized.removeprefix("mhlo.")
    return _ALIASES.get(normalized, normalized)


def stablehlo_capability(op_name: str) -> StableHLOOpCapability | None:
    return _CAPABILITIES.get(normalize_stablehlo_op_name(op_name))


def registered_stablehlo_ops() -> tuple[str, ...]:
    return tuple(sorted(_CAPABILITIES))


__all__ = [
    "StableHLOOpCapability",
    "normalize_stablehlo_op_name",
    "registered_stablehlo_ops",
    "stablehlo_capability",
]
