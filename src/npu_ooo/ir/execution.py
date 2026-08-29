from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class AccessType(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class BufferRegion:
    """A logical tensor range and its concrete placement for one task operand."""

    tensor: str
    memory: str
    shape: tuple[int, ...]
    starts: tuple[int, ...]
    dtype: str = "fp16"
    access: AccessType | str = AccessType.READ
    offset_bytes: int = 0
    size_bytes: int = 0
    layout: str = "dense"

    @property
    def normalized_access(self) -> str:
        return self.access.value if isinstance(self.access, AccessType) else str(self.access)

    @property
    def elements(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.tensor or not self.memory:
            issues.append("buffer region tensor and memory must not be empty")
        if len(self.shape) != len(self.starts):
            issues.append(f"buffer region '{self.tensor}' shape and starts must have equal rank")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.shape):
            issues.append(f"buffer region '{self.tensor}' shape values must be positive")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.starts):
            issues.append(f"buffer region '{self.tensor}' starts must be non-negative")
        if self.offset_bytes < 0:
            issues.append(f"buffer region '{self.tensor}' offset_bytes must be non-negative")
        if self.size_bytes < 0:
            issues.append(f"buffer region '{self.tensor}' size_bytes must be non-negative")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor": self.tensor,
            "memory": self.memory,
            "shape": list(self.shape),
            "starts": list(self.starts),
            "dtype": self.dtype,
            "access": self.normalized_access,
            "offset_bytes": self.offset_bytes,
            "size_bytes": self.size_bytes,
            "layout": self.layout,
        }


@dataclass(frozen=True)
class ExecutionTask:
    task_id: str
    tile_id: str
    operator_id: str
    primitive: str
    resource: str
    reads: tuple[BufferRegion, ...] = ()
    writes: tuple[BufferRegion, ...] = ()
    predecessors: tuple[str, ...] = ()
    duration_cycles: float | None = None
    initiation_interval_cycles: float | None = None
    stage_id: int = 0
    program_order: int = 0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.task_id or not self.tile_id or not self.operator_id:
            issues.append("execution task identifiers must not be empty")
        if not self.primitive or not self.resource:
            issues.append(f"execution task '{self.task_id}' primitive and resource must not be empty")
        if self.duration_cycles is not None and self.duration_cycles <= 0:
            issues.append(f"execution task '{self.task_id}' duration must be positive")
        if self.initiation_interval_cycles is not None and self.initiation_interval_cycles <= 0:
            issues.append(f"execution task '{self.task_id}' initiation interval must be positive")
        if self.stage_id < 0 or self.program_order < 0:
            issues.append(f"execution task '{self.task_id}' stage/order must be non-negative")
        for region in (*self.reads, *self.writes):
            issues.extend(region.validate())
        if len(set(self.predecessors)) != len(self.predecessors):
            issues.append(f"execution task '{self.task_id}' predecessors must be unique")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tile_id": self.tile_id,
            "operator_id": self.operator_id,
            "primitive": self.primitive,
            "resource": self.resource,
            "reads": [region.to_dict() for region in self.reads],
            "writes": [region.to_dict() for region in self.writes],
            "predecessors": list(self.predecessors),
            "duration_cycles": self.duration_cycles,
            "initiation_interval_cycles": self.initiation_interval_cycles,
            "stage_id": self.stage_id,
            "program_order": self.program_order,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class ExecutionGraph:
    graph_id: str
    tasks: tuple[ExecutionTask, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        tasks = {task.task_id: task for task in self.tasks}
        if len(tasks) != len(self.tasks):
            issues.append("execution task ids must be unique")
        for task in self.tasks:
            issues.extend(task.validate())
            for predecessor in task.predecessors:
                if predecessor not in tasks:
                    issues.append(f"task '{task.task_id}' references unknown predecessor '{predecessor}'")
        try:
            self.topological_order()
        except ValueError as exc:
            issues.append(str(exc))
        return tuple(issues)

    def topological_order(self) -> tuple[str, ...]:
        ids = [task.task_id for task in self.tasks]
        index = {task_id: position for position, task_id in enumerate(ids)}
        outgoing = {task_id: set() for task_id in ids}
        indegree = {task_id: 0 for task_id in ids}
        for task in self.tasks:
            for predecessor in task.predecessors:
                if predecessor not in outgoing:
                    raise ValueError("cannot topologically order execution graph with unknown predecessor")
                if task.task_id not in outgoing[predecessor]:
                    outgoing[predecessor].add(task.task_id)
                    indegree[task.task_id] += 1
        ready = sorted((task_id for task_id, degree in indegree.items() if degree == 0), key=index.__getitem__)
        result: list[str] = []
        while ready:
            current = ready.pop(0)
            result.append(current)
            for successor in sorted(outgoing[current], key=index.__getitem__):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort(key=index.__getitem__)
        if len(result) != len(ids):
            raise ValueError(f"execution graph '{self.graph_id}' contains a cycle")
        return tuple(result)

    def task(self, task_id: str) -> ExecutionTask:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "tasks": [task.to_dict() for task in self.tasks],
            "attributes": dict(self.attributes),
        }
