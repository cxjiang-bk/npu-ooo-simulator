from __future__ import annotations

"""Automatic schedule planning boundary.

The current implementation intentionally reuses the project's deterministic
32-element heuristic.  Keeping it behind a planner object gives later passes a
stable place to add architecture-aware cost models without restoring
benchmark-specific ``default_*_schedule`` calls to the frontend path.
"""

from dataclasses import replace
import math

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import OperatorGraph, ScheduleSpec, TensorResidency, TensorSpec, plan_uniform_tiles


_DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int32": 4,
    "float32": 4,
    "fp32": 4,
    "int64": 8,
    "float64": 8,
    "fp64": 8,
}


def _dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(str(dtype).lower().replace("torch.", ""), 2)


def _memory_plan(machine: MachineConfig) -> tuple[str, str, int | None]:
    roots = sorted(level.name for level in machine.memory_levels if level.parent is None)
    if len(roots) != 1:
        raise ValueError("schedule planning requires exactly one root memory level")
    root = roots[0]
    local_candidates = sorted(
        path.target
        for path in machine.transfer_paths
        if path.source == root
    )
    if not local_candidates:
        return root, root, None
    local = local_candidates[0]
    return root, local, machine.memory(local).capacity_bytes


def _tensor_bytes(tensor: TensorSpec) -> int | None:
    if any(not isinstance(value, int) or value <= 0 for value in tensor.shape):
        return None
    return math.prod(tensor.shape) * _dtype_bytes(tensor.dtype)


def _attach_machine_metadata(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> ScheduleSpec:
    """Add deterministic residency and ping-pong intent to each operator schedule."""

    root_memory, local_memory, local_capacity = _memory_plan(machine)
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    updated = []
    for operator_schedule in schedule.operator_schedules:
        operator = next(item for item in graph.operators if item.op_id == operator_schedule.operator_id)
        used_bytes = 0
        residency = []
        overflow: list[str] = []
        residency_bytes: dict[str, int] = {}
        seen_tensors: set[str] = set()
        for tensor_name in (*operator.inputs, *operator.outputs):
            if tensor_name in seen_tensors:
                continue
            seen_tensors.add(tensor_name)
            size_bytes = _tensor_bytes(tensors[tensor_name])
            if (
                local_capacity is None
                or (
                    size_bytes is not None
                    and used_bytes + size_bytes <= local_capacity
                )
            ):
                residency.append((tensor_name, local_memory))
                residency_bytes[tensor_name] = size_bytes
                used_bytes += size_bytes
            else:
                overflow.append(tensor_name)

        tile_count = 1
        for dimension, extent in (*operator.iteration_dims, *operator.reduction_dims):
            tile_count *= math.ceil(extent / operator_schedule.tile_size(dimension))
        ping_pong_enabled = tile_count > 1 and local_memory != root_memory
        ping_pong = {
            "enabled": ping_pong_enabled,
            "buffer_count": 2 if ping_pong_enabled else 1,
            "scope": local_memory,
            "tile_count": tile_count,
            "planned_only": True,
            "policy": "alternate_tile_slots" if ping_pong_enabled else "single_slot",
        }
        updated.append(
            replace(
                operator_schedule,
                residency=tuple(
                    TensorResidency(
                        tensor=tensor_name,
                        memory=memory,
                    )
                    for tensor_name, memory in residency
                ),
                attributes={
                    **dict(operator_schedule.attributes),
                    "residency_policy": "capacity_aware_first_fit",
                    "residency_root_memory": root_memory,
                    "residency_local_memory": local_memory,
                    "residency_capacity_bytes": local_capacity,
                    "residency_bytes": residency_bytes,
                    "residency_overflow_tensors": overflow,
                    "ping_pong": ping_pong,
                },
            )
        )
    return replace(
        schedule,
        operator_schedules=tuple(updated),
        attributes={
            **dict(schedule.attributes),
            "machine_id": machine.config_id,
            "residency_policy": "capacity_aware_first_fit",
            "ping_pong_policy": "alternate_tile_slots",
        },
    )


class SchedulePlanner:
    """Plan a resolved graph with a deterministic, shape-aware heuristic."""

    name = "heuristic-v1"

    def plan(
        self,
        graph: OperatorGraph,
        *,
        tile_size: int = 32,
        machine: MachineConfig | None = None,
    ) -> ScheduleSpec:
        schedule = plan_uniform_tiles(graph, tile_size=tile_size)
        if machine is not None:
            schedule = _attach_machine_metadata(graph, schedule, machine)
        return replace(
            schedule,
            attributes={
                **dict(schedule.attributes),
                "source": "automatic-planner",
                "planner": self.name,
            },
        )


def default_schedule_planner() -> SchedulePlanner:
    return SchedulePlanner()


__all__ = ["SchedulePlanner", "default_schedule_planner"]
