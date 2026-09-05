from __future__ import annotations

"""Automatic schedule planning boundary.

The current implementation intentionally reuses the project's deterministic
32-element heuristic.  Keeping it behind a planner object gives later passes a
stable place to add architecture-aware cost models without restoring
benchmark-specific ``default_*_schedule`` calls to the frontend path.
"""

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    OperatorGraph,
    ScheduleSpec,
    TensorResidency,
    TensorSpec,
    dtype_bytes,
    plan_uniform_tiles,
)


def _dtype_bytes(dtype: str) -> int:
    return dtype_bytes(dtype, default=2)


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
    """Plan a resolved graph with a deterministic, explainable cost model."""

    name = "cost-model-v1"

    @staticmethod
    def _unit_for_operator(operator_type: str, machine: MachineConfig):
        for unit in machine.execution_units:
            if operator_type in unit.supported_ops:
                return unit
        if operator_type in {"reshape", "transpose"}:
            for unit in machine.execution_units:
                if "copy" in unit.supported_ops or "transpose" in unit.supported_ops:
                    return unit
        return None

    @staticmethod
    def estimate_cost(
        graph: OperatorGraph,
        schedule: ScheduleSpec,
        machine: MachineConfig,
    ) -> dict[str, Any]:
        """Estimate one schedule using only graph shape and MachineConfig.

        This is deliberately a ranking model, not a device timing model.  It
        accounts for the number of tiles, tile-local compute work, root-memory
        traffic and local-memory working-set overflow.  The selected score is
        retained in the schedule so candidate decisions can be audited later.
        """

        roots = [level for level in machine.memory_levels if level.parent is None]
        if len(roots) != 1:
            raise ValueError("cost model requires exactly one root memory level")
        root = roots[0]
        local = machine.memory(_memory_plan(machine)[1])
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        tile_count = 0
        estimated_compute = 0.0
        estimated_root_bytes = 0
        local_working_set = 0
        for operator in graph.operators:
            operator_schedule = schedule.for_operator(operator.op_id)
            operator_tile_count = 1
            for dimension, extent in (*operator.iteration_dims, *operator.reduction_dims):
                if not isinstance(extent, int):
                    raise ValueError("cost model requires a resolved graph")
                operator_tile_count *= math.ceil(
                    extent / operator_schedule.tile_size(dimension)
                )
            tile_count += operator_tile_count
            tile_elements = 1
            for dimension, _extent in (*operator.iteration_dims, *operator.reduction_dims):
                tile_elements *= operator_schedule.tile_size(dimension)
            unit = SchedulePlanner._unit_for_operator(operator.normalized_type, machine)
            if unit is not None:
                rate = unit.attributes.get(
                    "macs_per_cycle"
                    if operator.normalized_type in {"matmul", "batched_matmul", "gemv", "conv2d"}
                    else "elements_per_cycle",
                    unit.attributes.get("lanes", 1),
                )
                rate = float(rate) if isinstance(rate, (int, float)) and rate > 0 else 1.0
                estimated_compute += operator_tile_count * (
                    float(unit.latency_cycles) + math.ceil(tile_elements / rate)
                )
            tensor_bytes = sum(
                _tensor_bytes(tensors[name]) or 0
                for name in dict.fromkeys((*operator.inputs, *operator.outputs))
            )
            estimated_root_bytes += tensor_bytes * operator_tile_count
            local_working_set = max(local_working_set, tensor_bytes)
        overflow_bytes = max(
            0,
            local_working_set - int(local.capacity_bytes)
            if local.capacity_bytes is not None
            else 0,
        )
        bandwidth = float(root.read_bandwidth_bytes_per_cycle + root.write_bandwidth_bytes_per_cycle)
        traffic_cycles = estimated_root_bytes / max(1.0, bandwidth)
        overflow_penalty = (
            overflow_bytes / max(1, int(local.capacity_bytes)) * 100.0
            if local.capacity_bytes is not None
            else 0.0
        )
        score = estimated_compute + traffic_cycles + overflow_penalty
        return {
            "score": round(score, 6),
            "estimated_compute_cycles": round(estimated_compute, 6),
            "estimated_traffic_cycles": round(traffic_cycles, 6),
            "estimated_root_bytes": estimated_root_bytes,
            "tile_count": tile_count,
            "local_working_set_bytes": local_working_set,
            "local_overflow_bytes": overflow_bytes,
            "model": "tile_count+compute+root_traffic+local_overflow_v1",
        }

    def plan(
        self,
        graph: OperatorGraph,
        *,
        tile_size: int = 32,
        machine: MachineConfig | None = None,
        tile_size_candidates: Sequence[int] | None = None,
    ) -> ScheduleSpec:
        candidates = tuple(tile_size_candidates or (tile_size,))
        if not candidates:
            raise ValueError("tile_size_candidates must contain at least one value")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in candidates):
            raise ValueError("tile_size_candidates must contain positive integers")
        if machine is not None and len(candidates) > 1:
            scored = []
            for candidate in sorted(set(candidates)):
                candidate_schedule = plan_uniform_tiles(graph, tile_size=candidate)
                candidate_schedule = _attach_machine_metadata(graph, candidate_schedule, machine)
                scored.append((self.estimate_cost(graph, candidate_schedule, machine), candidate, candidate_schedule))
            _cost, selected_tile_size, schedule = min(scored, key=lambda item: (item[0]["score"], item[1]))
            schedule = replace(
                schedule,
                attributes={
                    **dict(schedule.attributes),
                    "tile_size_candidates": list(sorted(set(candidates))),
                    "selected_tile_size": selected_tile_size,
                    "candidate_costs": {
                        str(candidate): cost
                        for cost, candidate, _candidate_schedule in scored
                    },
                },
            )
        else:
            schedule = plan_uniform_tiles(graph, tile_size=tile_size)
        if machine is not None:
            if len(candidates) == 1:
                schedule = _attach_machine_metadata(graph, schedule, machine)
            cost = self.estimate_cost(graph, schedule, machine)
            schedule = replace(
                schedule,
                attributes={
                    **dict(schedule.attributes),
                    "selected_tile_size": int(dict(schedule.attributes).get("tile_size", tile_size)),
                    "candidate_costs": {
                        str(dict(schedule.attributes).get("tile_size", tile_size)): cost
                    }
                    if len(candidates) == 1
                    else dict(schedule.attributes).get("candidate_costs", {}),
                },
            )
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
