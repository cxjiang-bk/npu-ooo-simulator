from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping

from npu_ooo.ir import AccessType, BufferRegion, ExecutionGraph, ExecutionTask


class AddressHazardKind(str, Enum):
    RAW = "RAW"
    WAR = "WAR"
    WAW = "WAW"


@dataclass(frozen=True)
class AddressDependency:
    predecessor: str
    successor: str
    kind: AddressHazardKind
    tensor: str
    memory: str
    condition: str = "address_overlap"
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "predecessor": self.predecessor,
            "successor": self.successor,
            "kind": self.kind.value,
            "tensor": self.tensor,
            "memory": self.memory,
            "condition": self.condition,
            "provenance": dict(self.provenance or {}),
        }


@dataclass(frozen=True)
class AddressConflict:
    """A conflict between a ready task and an in-flight task."""

    predecessor: str
    successor: str
    kind: AddressHazardKind
    tensor: str
    memory: str
    condition: str = "address_overlap"
    provenance: dict[str, Any] | None = None

    def to_dependency(self) -> AddressDependency:
        return AddressDependency(
            predecessor=self.predecessor,
            successor=self.successor,
            kind=self.kind,
            tensor=self.tensor,
            memory=self.memory,
            condition=self.condition,
            provenance=dict(self.provenance or {}),
        )


class AddressScoreboard:
    """Runtime range scoreboard for in-flight execution tasks.

    The scoreboard intentionally tracks only active tasks. Compile-time graph
    edges remain the source of true dataflow; this layer models the additional
    RAW/WAR/WAW protection needed when independent instructions overlap in
    the same address space.
    """

    def __init__(self) -> None:
        self._active: dict[str, ExecutionTask] = {}

    @property
    def active_task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._active))

    def reserve(self, task: ExecutionTask) -> None:
        if task.task_id in self._active:
            raise ValueError(f"address scoreboard task '{task.task_id}' is already active")
        self._active[task.task_id] = task

    def release(self, task_id: str) -> None:
        if task_id not in self._active:
            raise ValueError(f"address scoreboard task '{task_id}' is not active")
        del self._active[task_id]

    def conflict(self, task: ExecutionTask) -> AddressConflict | None:
        for active_id in sorted(self._active):
            active = self._active[active_id]
            conflict = _conflict(active, task)
            if conflict is None:
                continue
            kind, region = conflict
            return AddressConflict(
                predecessor=active.task_id,
                successor=task.task_id,
                kind=kind,
                tensor=region.tensor,
                memory=region.memory,
                condition=_dependency_condition(task, active.task_id),
                provenance=_dependency_provenance(task, active.task_id),
            )
        return None


def _reads(region: BufferRegion) -> bool:
    access = region.normalized_access
    return access in {AccessType.READ.value, AccessType.READ_WRITE.value}


def _writes(region: BufferRegion) -> bool:
    access = region.normalized_access
    return access in {AccessType.WRITE.value, AccessType.READ_WRITE.value}


def _overlap(left: BufferRegion, right: BufferRegion) -> bool:
    if left.tensor != right.tensor or left.memory != right.memory or len(left.shape) != len(right.shape):
        return False
    return all(
        left_start + left_extent > right_start
        and right_start + right_extent > left_start
        for left_start, left_extent, right_start, right_extent in zip(
            left.starts, left.shape, right.starts, right.shape
        )
    )


def _conflict(left: ExecutionTask, right: ExecutionTask) -> tuple[AddressHazardKind, BufferRegion] | None:
    for left_region in (*left.reads, *left.writes):
        for right_region in (*right.reads, *right.writes):
            if not _overlap(left_region, right_region):
                continue
            if _writes(left_region) and _reads(right_region):
                return AddressHazardKind.RAW, left_region
            if _reads(left_region) and _writes(right_region):
                return AddressHazardKind.WAR, left_region
            if _writes(left_region) and _writes(right_region):
                return AddressHazardKind.WAW, left_region
    return None


def _dependency_entry(task: ExecutionTask, predecessor_id: str) -> dict[str, Any] | None:
    metadata = task.dependency_provenance.get(predecessor_id)
    return dict(metadata) if isinstance(metadata, Mapping) else None


def _dependency_condition(task: ExecutionTask, predecessor_id: str) -> str:
    entry = _dependency_entry(task, predecessor_id)
    if entry is None:
        return "address_overlap"
    edges = entry.get("edges", ())
    conditions = sorted(
        {
            str(edge.get("condition"))
            for edge in edges
            if isinstance(edge, dict) and edge.get("condition")
        }
    )
    return "+".join(conditions) if conditions else "address_overlap"


def _dependency_provenance(task: ExecutionTask, predecessor_id: str) -> dict[str, Any]:
    entry = _dependency_entry(task, predecessor_id)
    if entry is None:
        return {
            "source": "address_scoreboard",
            "relation": "runtime_overlap",
            "predecessor": predecessor_id,
            "successor": task.task_id,
        }
    return {
        "source": "address_scoreboard",
        "predecessor": predecessor_id,
        "successor": task.task_id,
        "dependency": entry,
    }


def add_address_dependencies(graph: ExecutionGraph) -> tuple[ExecutionGraph, tuple[AddressDependency, ...]]:
    """Add deterministic program-order hazards for overlapping buffer ranges.

    The pass is deliberately separate from semantic dataflow edges. It models
    the address-range part of a TISA scoreboard and can be disabled for a
    compile-time-only baseline.
    """

    tasks = sorted(graph.tasks, key=lambda task: (task.program_order, task.task_id))
    predecessors = {task.task_id: set(task.predecessors) for task in graph.tasks}
    topological = graph.topological_order()
    topological_index = {task_id: index for index, task_id in enumerate(topological)}

    def reaches(source: str, target: str) -> bool:
        pending = [source]
        visited: set[str] = set()
        successors: dict[str, list[str]] = {task.task_id: [] for task in graph.tasks}
        for task in graph.tasks:
            for predecessor in task.predecessors:
                successors[predecessor].append(task.task_id)
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if current == target:
                return True
            pending.extend(successors[current])
        return False

    dependencies: list[AddressDependency] = []
    for index, predecessor in enumerate(tasks):
        for successor in tasks[index + 1 :]:
            conflict = _conflict(predecessor, successor)
            if conflict is None:
                continue
            kind, region = conflict
            # Program order is the intended scoreboard order. If the semantic
            # graph explicitly orders the pair in the opposite direction, do
            # not create a cycle merely to add a redundant address edge.
            if topological_index[successor.task_id] < topological_index[predecessor.task_id]:
                continue
            if not reaches(predecessor.task_id, successor.task_id):
                predecessors[successor.task_id].add(predecessor.task_id)
                dependencies.append(
                    AddressDependency(
                        predecessor=predecessor.task_id,
                        successor=successor.task_id,
                        kind=kind,
                        tensor=region.tensor,
                        memory=region.memory,
                        condition=_dependency_condition(successor, predecessor.task_id),
                        provenance=_dependency_provenance(successor, predecessor.task_id),
                    )
                )

    if not dependencies:
        return graph, ()
    updated_tasks = tuple(
        replace(task, predecessors=tuple(sorted(predecessors[task.task_id])))
        for task in graph.tasks
    )
    updated = ExecutionGraph(
        graph_id=graph.graph_id,
        tasks=updated_tasks,
        attributes={
            **graph.attributes,
            "address_scoreboard": True,
            "address_dependency_count": len(dependencies),
        },
    )
    issues = updated.validate()
    if issues:
        raise ValueError("address dependency augmentation invalidated execution graph: " + "; ".join(issues))
    return updated, tuple(dependencies)
