from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .operator import OperatorGraph, OperatorSpec


@dataclass(frozen=True)
class TensorResidency:
    """Compile-time placement of a tensor in a named memory level."""

    tensor: str
    memory: str

    def to_dict(self) -> dict[str, str]:
        return {"tensor": self.tensor, "memory": self.memory}


@dataclass(frozen=True)
class OperatorSchedule:
    """Tiling and loop-order decisions for one semantic operator."""

    operator_id: str
    tile_sizes: tuple[tuple[str, int], ...]
    loop_order: tuple[str, ...] = ()
    residency: tuple[TensorResidency, ...] = ()
    stage_id: int = 0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def tile_size_map(self) -> dict[str, int]:
        return {name: value for name, value in self.tile_sizes}

    @property
    def residency_map(self) -> dict[str, str]:
        return {item.tensor: item.memory for item in self.residency}

    def tile_size(self, dimension: str) -> int:
        try:
            return self.tile_size_map[dimension]
        except KeyError as exc:
            raise KeyError(f"schedule '{self.operator_id}' has no tile size for '{dimension}'") from exc

    def memory_for(self, tensor: str, default: str) -> str:
        return self.residency_map.get(tensor, default)

    def validate(self, operator: OperatorSpec) -> tuple[str, ...]:
        issues: list[str] = []
        if self.operator_id != operator.op_id:
            issues.append(
                f"schedule id '{self.operator_id}' does not match operator '{operator.op_id}'"
            )
        if self.stage_id < 0:
            issues.append(f"schedule '{self.operator_id}' stage_id must be non-negative")
        dimension_extents = dict((*operator.iteration_dims, *operator.reduction_dims))
        if len(self.tile_size_map) != len(self.tile_sizes):
            issues.append(f"schedule '{self.operator_id}' tile dimensions must be unique")
        for name, size in self.tile_sizes:
            if name not in dimension_extents:
                issues.append(f"schedule '{self.operator_id}' references unknown dimension '{name}'")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                issues.append(f"schedule '{self.operator_id}' tile size for '{name}' must be positive")
            elif name in dimension_extents and isinstance(dimension_extents[name], int) and size > dimension_extents[name]:
                issues.append(
                    f"schedule '{self.operator_id}' tile size for '{name}' exceeds extent {dimension_extents[name]}"
                )
        expected_dims = set(dimension_extents)
        if set(self.tile_size_map) != expected_dims:
            missing = sorted(expected_dims - set(self.tile_size_map))
            extra = sorted(set(self.tile_size_map) - expected_dims)
            if missing:
                issues.append(f"schedule '{self.operator_id}' is missing tile dimensions {missing}")
            if extra:
                issues.append(f"schedule '{self.operator_id}' has extra tile dimensions {extra}")
        order = self.loop_order or tuple(name for name, _ in operator.iteration_dims + operator.reduction_dims)
        if len(set(order)) != len(order):
            issues.append(f"schedule '{self.operator_id}' loop_order must be unique")
        if set(order) != expected_dims:
            issues.append(f"schedule '{self.operator_id}' loop_order must cover all operator dimensions")
        if len({item.tensor for item in self.residency}) != len(self.residency):
            issues.append(f"schedule '{self.operator_id}' residency tensors must be unique")
        known_tensors = set((*operator.inputs, *operator.outputs))
        for item in self.residency:
            if item.tensor not in known_tensors:
                issues.append(f"schedule '{self.operator_id}' references unknown tensor '{item.tensor}'")
            if not item.memory:
                issues.append(f"schedule '{self.operator_id}' memory for '{item.tensor}' must not be empty")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "tile_sizes": {name: value for name, value in self.tile_sizes},
            "loop_order": list(self.loop_order),
            "residency": [item.to_dict() for item in self.residency],
            "stage_id": self.stage_id,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class ScheduleSpec:
    """A complete schedule for an OperatorGraph."""

    schedule_id: str
    operator_schedules: tuple[OperatorSchedule, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, graph: OperatorGraph) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.schedule_id:
            issues.append("schedule id must not be empty")
        operators = {operator.op_id: operator for operator in graph.operators}
        schedules = {schedule.operator_id: schedule for schedule in self.operator_schedules}
        if len(schedules) != len(self.operator_schedules):
            issues.append("operator schedules must be unique")
        for operator_id, schedule in schedules.items():
            operator = operators.get(operator_id)
            if operator is None:
                issues.append(f"schedule references unknown operator '{operator_id}'")
            else:
                issues.extend(schedule.validate(operator))
        missing = sorted(set(operators) - set(schedules))
        if missing:
            issues.append(f"schedule is missing operators {missing}")
        return tuple(issues)

    def for_operator(self, operator_id: str) -> OperatorSchedule:
        for schedule in self.operator_schedules:
            if schedule.operator_id == operator_id:
                return schedule
        raise KeyError(operator_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "operator_schedules": [schedule.to_dict() for schedule in self.operator_schedules],
            "attributes": dict(self.attributes),
        }


def default_two_matmul_schedule(graph: OperatorGraph) -> ScheduleSpec:
    """Return a small, explicit schedule used by the first end-to-end example."""

    schedules: list[OperatorSchedule] = []
    for operator in graph.operators:
        dims = dict((*operator.iteration_dims, *operator.reduction_dims))
        sizes = {name: min(32, extent) for name, extent in dims.items() if isinstance(extent, int)}
        if len(sizes) != len(dims):
            raise ValueError("default 2mm schedule requires a resolved graph")
        schedules.append(
            OperatorSchedule(
                operator_id=operator.op_id,
                tile_sizes=tuple((name, sizes[name]) for name in dims),
                loop_order=tuple(name for name, _ in operator.iteration_dims + operator.reduction_dims),
                stage_id=0 if operator.op_id == graph.topological_order()[0] else 1,
            )
        )
    result = ScheduleSpec("two_matmul_default", tuple(schedules), attributes={"source": "hand-written"})
    issues = result.validate(graph)
    if issues:
        raise ValueError("; ".join(issues))
    return result
