"""Typed dynamic-index contracts shared by compiler and runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


IndexValue = int | str


def _valid_symbol(value: str) -> bool:
    return bool(value) and value.replace("_", "a").isalnum()


@dataclass(frozen=True)
class DynamicIndexExpr:
    """A logical index expression that may be resolved at submission time.

    ``index_operands`` names scalar/tensor values in the canonical graph.
    ``clamp_bounds`` records the legal half-open window for each axis using
    concrete extents or shape symbols.  The expression remains symbolic until
    ``resolved_values`` is populated by specialization or runtime binding.
    """

    expression_id: str
    source_tensor: str
    index_operands: tuple[str, ...]
    index_rank: int | None = None
    clamp_rule: str = "stablehlo_dynamic_slice_clamp"
    clamp_bounds: tuple[tuple[IndexValue | None, IndexValue | None], ...] = ()
    resolved_values: tuple[int, ...] | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.expression_id or not self.source_tensor:
            issues.append("dynamic index expression identifiers must not be empty")
        if not self.index_operands:
            issues.append(f"dynamic index expression '{self.expression_id}' needs index operands")
        if not self.clamp_rule:
            issues.append(f"dynamic index expression '{self.expression_id}' clamp_rule must not be empty")
        expected_rank = self.index_rank or len(self.index_operands)
        if self.index_rank is not None and self.index_rank <= 0:
            issues.append(f"dynamic index expression '{self.expression_id}' index_rank must be positive")
        if self.clamp_bounds and len(self.clamp_bounds) != expected_rank:
            issues.append(
                f"dynamic index expression '{self.expression_id}' clamp_bounds rank must match index rank"
            )
        for index, operand in enumerate(self.index_operands):
            if not _valid_symbol(operand):
                issues.append(
                    f"dynamic index expression '{self.expression_id}' operand {index} is invalid"
                )
        for axis, bounds in enumerate(self.clamp_bounds):
            for value in bounds:
                if value is not None and not (
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0
                ) and not (isinstance(value, str) and _valid_symbol(value)):
                    issues.append(
                        f"dynamic index expression '{self.expression_id}' bound {axis} must be a non-negative integer or symbol"
                    )
        if self.resolved_values is not None:
            if len(self.resolved_values) != expected_rank:
                issues.append(
                    f"dynamic index expression '{self.expression_id}' resolved rank must match index rank"
                )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.resolved_values
            ):
                issues.append(
                    f"dynamic index expression '{self.expression_id}' resolved values must be non-negative integers"
                )
        return tuple(issues)

    def resolve(self, values: tuple[int, ...]) -> "DynamicIndexExpr":
        expected_rank = self.index_rank or len(self.index_operands)
        if len(values) != expected_rank:
            raise ValueError(
                f"dynamic index expression '{self.expression_id}' expects {expected_rank} values"
            )
        resolved = DynamicIndexExpr(
            expression_id=self.expression_id,
            source_tensor=self.source_tensor,
            index_operands=self.index_operands,
            index_rank=self.index_rank,
            clamp_rule=self.clamp_rule,
            clamp_bounds=self.clamp_bounds,
            resolved_values=tuple(values),
            attributes=dict(self.attributes),
        )
        issues = resolved.validate()
        if issues:
            raise ValueError("; ".join(issues))
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression_id": self.expression_id,
            "source_tensor": self.source_tensor,
            "index_operands": list(self.index_operands),
            "index_rank": self.index_rank,
            "clamp_rule": self.clamp_rule,
            "clamp_bounds": [list(bounds) for bounds in self.clamp_bounds],
            "resolved_values": list(self.resolved_values) if self.resolved_values is not None else None,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class DynamicIndexBinding:
    """Concrete runtime values for one :class:`DynamicIndexExpr`."""

    expression_id: str
    values: tuple[int, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.expression_id:
            issues.append("dynamic index binding expression_id must not be empty")
        if not self.values:
            issues.append(f"dynamic index binding '{self.expression_id}' needs values")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.values):
            issues.append(f"dynamic index binding '{self.expression_id}' values must be integers")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression_id": self.expression_id,
            "values": list(self.values),
            "attributes": dict(self.attributes),
        }


def resolve_dynamic_index(
    expression: Mapping[str, Any],
    binding: DynamicIndexBinding,
    source_shape: Sequence[int],
    window_shape: Sequence[int],
) -> tuple[int, ...]:
    """Resolve one invocation's indices using StableHLO window clamping.

    StableHLO dynamic slice and dynamic update slice use the same rule: each
    start is clamped to ``[0, source_extent - window_extent]``.  The
    expression metadata may provide an equivalent upper bound; the physical
    source shape remains authoritative for capacity safety.
    """

    expression_id = str(expression.get("expression_id", ""))
    if binding.expression_id != expression_id:
        raise ValueError(
            f"dynamic index binding '{binding.expression_id}' does not match '{expression_id}'"
        )
    rank = expression.get("index_rank")
    expected_rank = int(rank) if isinstance(rank, int) and rank > 0 else len(source_shape)
    if len(source_shape) != expected_rank or len(window_shape) != expected_rank:
        raise ValueError(
            f"dynamic index expression '{expression_id}' rank does not match source/window shape"
        )
    if len(binding.values) != expected_rank:
        raise ValueError(
            f"dynamic index binding '{binding.expression_id}' expects {expected_rank} values"
        )
    bounds = expression.get("clamp_bounds", ())
    resolved: list[int] = []
    for axis, (extent, window, value) in enumerate(
        zip(source_shape, window_shape, binding.values)
    ):
        if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
            raise ValueError(f"dynamic index source extent at axis {axis} must be positive")
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError(f"dynamic index window extent at axis {axis} must be positive")
        if window > extent:
            raise ValueError(
                f"dynamic index window extent {window} exceeds source extent {extent} at axis {axis}"
            )
        legal_upper = extent - window
        metadata_upper: int | None = None
        metadata_lower = 0
        if isinstance(bounds, (tuple, list)) and axis < len(bounds):
            bound = bounds[axis]
            if isinstance(bound, (tuple, list)) and bound:
                if isinstance(bound[0], int) and not isinstance(bound[0], bool):
                    metadata_lower = bound[0]
                if len(bound) > 1 and isinstance(bound[1], int) and not isinstance(bound[1], bool):
                    metadata_upper = bound[1]
        lower = max(0, metadata_lower)
        upper = legal_upper if metadata_upper is None else min(legal_upper, metadata_upper)
        if upper < lower:
            raise ValueError(
                f"dynamic index expression '{expression_id}' has invalid bounds at axis {axis}"
            )
        resolved.append(max(lower, min(int(value), upper)))
    return tuple(resolved)


__all__ = ["DynamicIndexBinding", "DynamicIndexExpr", "IndexValue", "resolve_dynamic_index"]
