from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "predecessor": self.predecessor,
            "successor": self.successor,
            "kind": self.kind.value,
            "tensor": self.tensor,
            "memory": self.memory,
        }


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
