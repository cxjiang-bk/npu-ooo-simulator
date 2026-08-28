from __future__ import annotations

"""Registry for multi-node semantic recovery and fusion patterns.

The StableHLO capability registry answers whether one StableHLO operation can
be imported.  This registry is deliberately a separate layer: its entries
recognize a proven multi-operation graph pattern and replace it with a
canonical semantic operator before tiling.  Keeping the two registries
separate makes unsupported operation diagnostics precise and leaves room for
future region-level patterns such as attention and SwiGLU.
"""

from dataclasses import dataclass
from typing import Iterable

from .passes import (
    AttentionRegionPass,
    GraphPass,
    LayerNormFusionPass,
    RecoverStableHLOLayerNormPass,
    RotaryEmbeddingRegionPass,
    RMSNormFusionPass,
    SoftmaxFusionPass,
    SwiGLUFusionPass,
)


@dataclass(frozen=True)
class SemanticFusionPattern:
    """One graph-level semantic recovery pattern exposed to GC."""

    name: str
    semantic_family: str
    graph_pass: GraphPass
    source: str = "python-graph-pattern"
    priority: int = 100

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("fusion pattern name must not be empty")
        if not self.semantic_family.strip():
            raise ValueError("fusion pattern semantic_family must not be empty")
        if not self.source.strip():
            raise ValueError("fusion pattern source must not be empty")
        if not getattr(self.graph_pass, "name", "").strip():
            raise ValueError("fusion pattern graph_pass must expose a non-empty name")
        if self.priority < 0:
            raise ValueError("fusion pattern priority must be non-negative")


class SemanticFusionPatternRegistry:
    """Deterministic registry for GC semantic fusion/recovery passes.

    Registration order is retained as a tie-breaker for equal priorities.
    ``create_passes`` exposes pass instances in registry order.  Current graph
    passes are stateless; stateful implementations should be registered as
    separate instances for separate compiler pipelines.
    """

    def __init__(self, patterns: Iterable[SemanticFusionPattern] = ()) -> None:
        self._patterns: dict[str, SemanticFusionPattern] = {}
        self._registration_order: list[str] = []
        for pattern in patterns:
            self.register(pattern)

    def register(self, pattern: SemanticFusionPattern) -> None:
        if pattern.name in self._patterns:
            raise ValueError(f"duplicate semantic fusion pattern: {pattern.name}")
        self._patterns[pattern.name] = pattern
        self._registration_order.append(pattern.name)

    def patterns(self) -> tuple[SemanticFusionPattern, ...]:
        order = {name: index for index, name in enumerate(self._registration_order)}
        return tuple(
            sorted(
                self._patterns.values(),
                key=lambda pattern: (pattern.priority, order[pattern.name]),
            )
        )

    def create_passes(self) -> tuple[GraphPass, ...]:
        return tuple(pattern.graph_pass for pattern in self.patterns())

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for pattern in self.patterns():
            if pattern.name != pattern.graph_pass.name:
                issues.append(
                    f"pattern '{pattern.name}' does not match graph pass "
                    f"name '{pattern.graph_pass.name}'"
                )
        return tuple(issues)


def default_semantic_fusion_registry() -> SemanticFusionPatternRegistry:
    """Return the paper-aligned default semantic pattern set.

    Priorities match the historical GC order.  The recovery pass runs before
    the generic LayerNorm fusion pass, while RMSNorm and Softmax follow the
    existing canonicalization and transpose passes.
    """

    return SemanticFusionPatternRegistry(
        (
            SemanticFusionPattern(
                name="recover_stablehlo_layernorm",
                semantic_family="layernorm",
                graph_pass=RecoverStableHLOLayerNormPass(),
                priority=20,
            ),
            SemanticFusionPattern(
                name="fuse_layernorm",
                semantic_family="layernorm",
                graph_pass=LayerNormFusionPass(),
                priority=50,
            ),
            SemanticFusionPattern(
                name="fuse_rmsnorm",
                semantic_family="rmsnorm",
                graph_pass=RMSNormFusionPass(),
                priority=60,
            ),
            SemanticFusionPattern(
                name="fuse_softmax",
                semantic_family="softmax",
                graph_pass=SoftmaxFusionPass(),
                priority=70,
            ),
            SemanticFusionPattern(
                name="recover_rotary_embedding",
                semantic_family="rotary_embedding",
                graph_pass=RotaryEmbeddingRegionPass(),
                priority=75,
            ),
            SemanticFusionPattern(
                name="recover_attention_region",
                semantic_family="attention",
                graph_pass=AttentionRegionPass(),
                priority=80,
            ),
            SemanticFusionPattern(
                name="fuse_swiglu",
                semantic_family="swiglu",
                graph_pass=SwiGLUFusionPass(),
                priority=90,
            ),
        )
    )


__all__ = [
    "SemanticFusionPattern",
    "SemanticFusionPatternRegistry",
    "default_semantic_fusion_registry",
]
