from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    AccessType,
    BufferRegion,
    ExecutionGraph,
    ExecutionTask,
    OperatorGraph,
    OperatorSpec,
    ScheduleSpec,
    TileGraph,
    TileInstance,
    build_tile_graph,
    default_two_matmul_schedule,
)
from npu_ooo.ir.model import ModelInstance


_DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}


@dataclass(frozen=True)
class LoweringResult:
    tile_graph: TileGraph
    execution_graph: ExecutionGraph
    statistics: dict[str, int | float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_graph": self.tile_graph.to_dict(),
            "execution_graph": self.execution_graph.to_dict(),
            "statistics": dict(self.statistics),
        }


def dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(dtype.lower(), 2)


def _root_memory(machine: MachineConfig) -> str:
    roots = sorted(level.name for level in machine.memory_levels if level.parent is None)
    if len(roots) != 1:
        raise ValueError("matmul lowering requires exactly one root memory level")
    return roots[0]


def _local_memory(machine: MachineConfig, root: str) -> str:
    candidates = sorted(
        path.target for path in machine.transfer_paths if path.source == root
    )
    if not candidates:
        raise ValueError(f"machine '{machine.config_id}' has no transfer path from root memory '{root}'")
    return candidates[0]


def _path(machine: MachineConfig, source: str, target: str):
    for path in machine.transfer_paths:
        if path.source == source and path.target == target:
            return path
    raise ValueError(
        f"machine '{machine.config_id}' has no transfer path {source}->{target}"
    )


def _unit_for(machine: MachineConfig, operation: str):
    for unit in machine.execution_units:
        if operation in unit.supported_ops:
            return unit
    raise ValueError(f"machine '{machine.config_id}' has no unit supporting '{operation}'")


def _transfer_timing(machine: MachineConfig, source: str, target: str, size_bytes: int) -> tuple[float, float, str]:
    path = _path(machine, source, target)
    unit = machine.unit(path.engine)
    duration = (
        path.setup_latency_cycles
        + path.transform_latency_cycles
        + math.ceil(size_bytes / path.bandwidth_bytes_per_cycle)
        + unit.latency_cycles
    )
    return float(duration), float(unit.initiation_interval_cycles), path.engine


def _compute_timing(machine: MachineConfig, output_shape: tuple[int, int], reduction: int) -> tuple[float, float, str]:
    unit = _unit_for(machine, "matmul")
    macs = output_shape[0] * output_shape[1] * reduction
    configured_rate = unit.attributes.get("macs_per_cycle")
    if isinstance(configured_rate, (int, float)) and configured_rate > 0:
        macs_per_cycle = float(configured_rate)
    else:
        rows = unit.attributes.get("rows", 16)
        cols = unit.attributes.get("cols", 16)
        depth = unit.attributes.get("k", 1)
        inferred = rows * cols * max(1, depth) * max(1, unit.issue_width)
        macs_per_cycle = float(inferred)
    duration = unit.latency_cycles + math.ceil(macs / macs_per_cycle)
    return float(duration), float(unit.initiation_interval_cycles), unit.name


def _region(
    tensor,
    memory: str,
    starts: tuple[int, ...],
    shape: tuple[int, ...],
    access: AccessType,
) -> BufferRegion:
    element_size = dtype_bytes(tensor.dtype)
    strides: list[int] = []
    stride = 1
    for extent in reversed(tensor.shape):
        strides.append(stride)
        stride *= extent
    strides.reverse()
    offset_elements = sum(start * stride_value for start, stride_value in zip(starts, strides))
    elements = math.prod(shape)
    return BufferRegion(
        tensor=tensor.name,
        memory=memory,
        shape=shape,
        starts=starts,
        dtype=tensor.dtype,
        access=access,
        offset_bytes=offset_elements * element_size,
        size_bytes=elements * element_size,
        layout=tensor.layout,
    )


def _regions_overlap(left: BufferRegion, right: BufferRegion) -> bool:
    if left.tensor != right.tensor or len(left.starts) != len(right.starts):
        return False
    for left_start, left_extent, right_start, right_extent in zip(
        left.starts, left.shape, right.starts, right.shape
    ):
        if left_start + left_extent <= right_start or right_start + right_extent <= left_start:
            return False
    return True


def _matmul_regions(operator: OperatorSpec, tensors: dict[str, Any], tile: TileInstance):
    if len(operator.inputs) < 2 or len(operator.outputs) != 1:
        raise ValueError(f"matmul operator '{operator.op_id}' must have two inputs and one output")
    iteration = tuple(name for name, _ in operator.iteration_dims)
    reduction = tuple(name for name, _ in operator.reduction_dims)
    if len(iteration) != 2 or len(reduction) != 1:
        raise ValueError(f"matmul operator '{operator.op_id}' requires two iteration and one reduction dimension")
    out0, out1 = iteration
    red = reduction[0]
    bounds = tile.bound_map
    out_starts = (bounds[out0][0], bounds[out1][0])
    out_shape = (bounds[out0][1] - bounds[out0][0], bounds[out1][1] - bounds[out1][0])
    red_start, red_stop = bounds[red]
    red_shape = red_stop - red_start
    left = tensors[operator.inputs[0]]
    right = tensors[operator.inputs[1]]
    output = tensors[operator.outputs[0]]
    left_region = _region(left, "__memory__", (out_starts[0], red_start), (out_shape[0], red_shape), AccessType.READ)
    right_region = _region(right, "__memory__", (red_start, out_starts[1]), (red_shape, out_shape[1]), AccessType.READ)
    output_region = _region(output, "__memory__", out_starts, out_shape, AccessType.READ_WRITE)
    return left_region, right_region, output_region, red_shape


def lower_matmul_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> LoweringResult:
    """Lower every matmul in a resolved graph into load/compute/store tasks."""

    graph_issues = graph.validate()
    schedule_issues = schedule.validate(graph)
    machine_issues = machine.validate()
    if graph_issues or schedule_issues or machine_issues:
        raise ValueError("; ".join((*graph_issues, *schedule_issues, *machine_issues)))
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    root = _root_memory(machine)
    local = _local_memory(machine, root)
    tasks: list[ExecutionTask] = []
    pending_predecessors: dict[str, set[str]] = {}
    producer_stores: dict[str, list[tuple[BufferRegion, str]]] = {}
    compute_by_output: dict[tuple[str, tuple[int, ...]], list[tuple[int, str]]] = {}
    task_order = 0
    total_macs = 0
    total_transfer_bytes = 0

    for operator_id in graph.topological_order():
        operator = next(operator for operator in graph.operators if operator.op_id == operator_id)
        if operator.normalized_type not in {"matmul", "batched_matmul", "gemv"}:
            raise NotImplementedError(f"no matmul lowering for operator type '{operator.normalized_type}'")
        op_schedule = schedule.for_operator(operator_id)
        tiles = []
        # Reuse the canonical tile expansion order from TileGraph.
        from npu_ooo.ir.tile import enumerate_operator_tiles

        tiles.extend(enumerate_operator_tiles(operator, op_schedule))
        reduction_name = operator.reduction_dims[0][0]
        output_dims = tuple(name for name, _ in operator.iteration_dims)
        for tile in tiles:
            left, right, output, reduction_shape = _matmul_regions(operator, tensors, tile)
            left_global = BufferRegion(**{**left.__dict__, "memory": root})
            right_global = BufferRegion(**{**right.__dict__, "memory": root})
            left_local = BufferRegion(**{**left.__dict__, "memory": local})
            right_local = BufferRegion(**{**right.__dict__, "memory": local})
            output_local = BufferRegion(**{**output.__dict__, "memory": local})
            output_global = BufferRegion(**{**output.__dict__, "memory": root, "access": AccessType.WRITE})
            tile_prefix = tile.tile_id
            load_a_id = f"{tile_prefix}.load_a"
            load_b_id = f"{tile_prefix}.load_b"
            compute_id = f"{tile_prefix}.mxu"
            load_a_duration, load_a_ii, load_a_unit = _transfer_timing(machine, root, local, left.size_bytes)
            load_b_duration, load_b_ii, load_b_unit = _transfer_timing(machine, root, local, right.size_bytes)
            compute_duration, compute_ii, compute_unit = _compute_timing(
                machine, output.shape, reduction_shape
            )
            load_a_preds = {
                store_id
                for region, store_id in producer_stores.get(left.tensor, [])
                if _regions_overlap(region, left_global)
            }
            load_b_preds = {
                store_id
                for region, store_id in producer_stores.get(right.tensor, [])
                if _regions_overlap(region, right_global)
            }
            tasks.extend(
                (
                    ExecutionTask(
                        task_id=load_a_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="load",
                        resource=load_a_unit,
                        reads=(left_global,),
                        writes=(left_local,),
                        predecessors=tuple(sorted(load_a_preds)),
                        duration_cycles=load_a_duration,
                        initiation_interval_cycles=load_a_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"operand": "lhs", "iteration": tile.ordinal},
                    ),
                    ExecutionTask(
                        task_id=load_b_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="load",
                        resource=load_b_unit,
                        reads=(right_global,),
                        writes=(right_local,),
                        predecessors=tuple(sorted(load_b_preds)),
                        duration_cycles=load_b_duration,
                        initiation_interval_cycles=load_b_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order + 1,
                        attributes={"operand": "rhs", "iteration": tile.ordinal},
                    ),
                )
            )
            task_order += 2
            output_key = (operator_id, tuple(tile.bound_map[name][0] for name in output_dims))
            previous = compute_by_output.setdefault(output_key, [])
            predecessors = {load_a_id, load_b_id}
            if previous:
                predecessors.add(max(previous, key=lambda item: item[0])[1])
            tasks.append(
                ExecutionTask(
                    task_id=compute_id,
                    tile_id=tile.tile_id,
                    operator_id=operator_id,
                    primitive="matmul",
                    resource=compute_unit,
                    reads=(left_local, right_local),
                    writes=(output_local,),
                    predecessors=tuple(sorted(predecessors)),
                    duration_cycles=compute_duration,
                    initiation_interval_cycles=compute_ii,
                    stage_id=tile.stage_id,
                    program_order=task_order,
                    attributes={
                        "m_tile": output.shape[0],
                        "n_tile": output.shape[1],
                        "k_tile": reduction_shape,
                        "macs": output.shape[0] * output.shape[1] * reduction_shape,
                        "iteration": tile.ordinal,
                    },
                )
            )
            task_order += 1
            previous.append((tile.bound_map[reduction_name][0], compute_id))
            if tile.bound_map[reduction_name][1] == dict(operator.reduction_dims)[reduction_name]:
                store_id = f"{tile_prefix}.store"
                store_duration, store_ii, store_unit = _transfer_timing(machine, local, root, output.size_bytes)
                tasks.append(
                    ExecutionTask(
                        task_id=store_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="store",
                        resource=store_unit,
                        reads=(output_local,),
                        writes=(output_global,),
                        predecessors=(compute_id,),
                        duration_cycles=store_duration,
                        initiation_interval_cycles=store_ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"final_reduction_tile": True, "iteration": tile.ordinal},
                    )
                )
                task_order += 1
                producer_stores.setdefault(output.tensor, []).append((output_global, store_id))
            total_macs += output.shape[0] * output.shape[1] * reduction_shape
            total_transfer_bytes += left.size_bytes + right.size_bytes
            if tile.bound_map[reduction_name][1] == dict(operator.reduction_dims)[reduction_name]:
                total_transfer_bytes += output.size_bytes

    execution = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=tuple(tasks),
        attributes={"source": "matmul-lowering", "root_memory": root, "local_memory": local},
    )
    issues = execution.validate()
    if issues:
        raise ValueError("; ".join(issues))
    tile_graph = build_tile_graph(graph, schedule)
    return LoweringResult(
        tile_graph=tile_graph,
        execution_graph=execution,
        statistics={
            "tile_count": len(tile_graph.tiles),
            "task_count": len(execution.tasks),
            "macs": total_macs,
            "transfer_bytes": total_transfer_bytes,
        },
    )


def lower_two_matmul(model: ModelInstance, machine: MachineConfig, schedule: ScheduleSpec | None = None) -> LoweringResult:
    schedule = schedule or default_two_matmul_schedule(model.graph)
    return lower_matmul_graph(model.graph, schedule, machine)
