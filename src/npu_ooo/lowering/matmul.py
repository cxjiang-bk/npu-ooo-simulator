from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

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
    dtype_bytes as shared_dtype_bytes,
    tensor_layout,
)


@dataclass(frozen=True)
class LoweringResult:
    tile_graph: TileGraph
    execution_graph: ExecutionGraph
    statistics: dict[str, int | float]
    payloads: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_graph": self.tile_graph.to_dict(),
            "execution_graph": self.execution_graph.to_dict(),
            "statistics": dict(self.statistics),
            "payloads": {key: list(value) for key, value in self.payloads.items()},
        }


def dtype_bytes(dtype: str) -> int:
    return shared_dtype_bytes(dtype, default=2)


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


def _compute_timing(machine: MachineConfig, output_shape: tuple[int, ...], reduction: int) -> tuple[float, float, str]:
    try:
        unit = machine.unit(machine.placement("matmul").unit)
    except KeyError:
        unit = _unit_for(machine, "matmul")
    macs = math.prod(output_shape) * reduction
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
    layout_info = tensor_layout(tensor)
    interval = layout_info.interval(starts, shape)
    if interval is None:
        # Unknown physical encodings cannot be mapped to a precise interval;
        # use the complete allocation as a conservative scoreboard range.
        offset_bytes = 0
        size_bytes = layout_info.allocation_size_bytes
    else:
        offset_bytes, size_bytes = interval
    return BufferRegion(
        tensor=tensor.name,
        memory=memory,
        shape=shape,
        starts=starts,
        dtype=tensor.dtype,
        access=access,
        offset_bytes=offset_bytes,
        size_bytes=size_bytes,
        layout=layout_info.layout,
        strides_bytes=layout_info.strides_bytes,
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
    if len(operator.inputs) > 2:
        raise ValueError(f"matmul operator '{operator.op_id}' must be decomposed to two inputs")
    iteration = tuple(name for name, _ in operator.iteration_dims)
    reduction = tuple(name for name, _ in operator.reduction_dims)
    if len(iteration) < 2 or len(reduction) != 1:
        raise ValueError(
            f"matmul operator '{operator.op_id}' requires output iteration dimensions and one reduction dimension"
        )
    batch_dimensions = iteration[:-2]
    out0, out1 = iteration[-2:]
    red = reduction[0]
    bounds = tile.bound_map
    batch_starts = tuple(bounds[name][0] for name in batch_dimensions)
    batch_shape = tuple(bounds[name][1] - bounds[name][0] for name in batch_dimensions)
    out_starts = (*batch_starts, bounds[out0][0], bounds[out1][0])
    out_shape = (
        *batch_shape,
        bounds[out0][1] - bounds[out0][0],
        bounds[out1][1] - bounds[out1][0],
    )
    red_start, red_stop = bounds[red]
    red_shape = red_stop - red_start
    left = tensors[operator.inputs[0]]
    right = tensors[operator.inputs[1]]
    output = tensors[operator.outputs[0]]
    left_region = _region(
        left,
        "__memory__",
        (*batch_starts, bounds[out0][0], red_start),
        (*batch_shape, bounds[out0][1] - bounds[out0][0], red_shape),
        AccessType.READ,
    )
    rhs_transposed = bool(operator.attributes.get("rhs_transposed", False))
    rhs_broadcast_batch = bool(operator.attributes.get("rhs_broadcast_batch", False))
    right_batch_starts = () if rhs_broadcast_batch else batch_starts
    right_batch_shape = () if rhs_broadcast_batch else batch_shape
    if rhs_transposed:
        right_starts = (*right_batch_starts, bounds[out1][0], red_start)
        right_shape = (*right_batch_shape, bounds[out1][1] - bounds[out1][0], red_shape)
    else:
        right_starts = (*right_batch_starts, red_start, bounds[out1][0])
        right_shape = (*right_batch_shape, red_shape, bounds[out1][1] - bounds[out1][0])
    right_region = _region(right, "__memory__", right_starts, right_shape, AccessType.READ)
    output_region = _region(output, "__memory__", out_starts, out_shape, AccessType.READ_WRITE)
    return left_region, right_region, output_region, red_shape


def _operand_region(
    operand: Any,
    tensor: Any,
    access: AccessType,
) -> BufferRegion:
    memory = operand.tile_mem
    if memory.memory_space is None or memory.buffer_id is None:
        raise ValueError(f"target operand '{operand.name}' is not materialized")
    if memory.logical_starts is None or memory.logical_shape is None:
        starts = (0,) * len(operand.tile_shape)
        shape = operand.tile_shape
    else:
        starts = memory.logical_starts
        shape = memory.logical_shape
    valid_bytes = memory.valid_bytes or math.prod(shape) * dtype_bytes(tensor.dtype)
    return BufferRegion(
        tensor=tensor.name,
        memory=memory.memory_space,
        shape=shape,
        starts=starts,
        dtype=tensor.dtype,
        access=access,
        offset_bytes=memory.offset_bytes or 0,
        size_bytes=memory.size_bytes or valid_bytes,
        layout=memory.layout,
        strides_bytes=memory.strides_bytes,
        buffer_id=memory.buffer_id,
        valid_bytes=valid_bytes,
        attributes={
            "operand_role": memory.role,
            "symbolic_buffer_id": memory.symbolic_buffer_id,
            "target_plan_operand": operand.name,
        },
    )


def lower_matmul_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
    *,
    target_plan: Any,
) -> LoweringResult:
    """Generate Matmul payload tasks directly from ``TargetPlan``."""

    graph_issues = graph.validate()
    schedule_issues = schedule.validate(graph)
    machine_issues = machine.validate()
    if graph_issues or schedule_issues or machine_issues:
        raise ValueError("; ".join((*graph_issues, *schedule_issues, *machine_issues)))
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    operators = {operator.op_id: operator for operator in graph.operators}
    target_instructions = tuple(
        item.instruction
        for item in target_plan.instructions
        if item.instruction.operator_id in operators
    )
    by_target: dict[str, list[str]] = {}
    tasks: list[ExecutionTask] = []
    transfer_bytes = 0
    total_macs = 0
    program_order = 0

    def predecessor_tasks(instruction: Any) -> tuple[str, ...]:
        return tuple(
            task_id
            for dependency in instruction.dependencies
            for task_id in by_target.get(dependency.source, ())
        )

    for instruction in target_instructions:
        operator = operators[instruction.operator_id]
        if operator.normalized_type not in {"matmul", "batched_matmul", "gemv"}:
            continue
        operands = {operand.name: operand for operand in instruction.operands}
        task_ids: list[str] = []
        transfer_pairs = instruction.attributes.get("transfer_pairs", ())
        if transfer_pairs:
            for index, pair in enumerate(transfer_pairs):
                source = operands[str(pair["source_operand"])]
                destination = operands[str(pair["destination_operand"])]
                source_region = _operand_region(
                    source, tensors[source.tile_mem.tensor], AccessType.READ
                )
                destination_region = _operand_region(
                    destination,
                    tensors[destination.tile_mem.tensor],
                    AccessType.WRITE,
                )
                size = int(pair["valid_bytes"])
                duration, interval, resource = _transfer_timing(
                    machine,
                    source.tile_mem.memory_space,
                    destination.tile_mem.memory_space,
                    size,
                )
                task_id = f"{instruction.tisa_id}.payload{index:02d}"
                task_ids.append(task_id)
                tasks.append(
                    ExecutionTask(
                        task_id=task_id,
                        tile_id=instruction.tile_id,
                        operator_id=instruction.operator_id,
                        primitive=str(pair.get("transform") or instruction.op_type),
                        resource=resource,
                        reads=(source_region,),
                        writes=(destination_region,),
                        predecessors=predecessor_tasks(instruction),
                        duration_cycles=duration,
                        initiation_interval_cycles=interval,
                        stage_id=int(instruction.attributes.get("stage_id", 0)),
                        program_order=program_order,
                        attributes={
                            "target_tisa_id": instruction.tisa_id,
                            "abstract_tisa_id": instruction.attributes.get("abstract_tisa_id"),
                            "operand": pair["role"],
                            "route_hop": instruction.attributes.get("route_hop"),
                            "transfer_bytes": size,
                        },
                    )
                )
                program_order += 1
                transfer_bytes += size
        else:
            reads = tuple(
                _operand_region(
                    operand,
                    tensors[operand.tile_mem.tensor],
                    AccessType.READ_WRITE
                    if operand.normalized_access == AccessType.READ_WRITE.value
                    else AccessType.READ,
                )
                for operand in instruction.operands
                if operand.normalized_access
                in {AccessType.READ.value, AccessType.READ_WRITE.value}
            )
            writes = tuple(
                _operand_region(
                    operand,
                    tensors[operand.tile_mem.tensor],
                    AccessType.READ_WRITE
                    if operand.normalized_access == AccessType.READ_WRITE.value
                    else AccessType.WRITE,
                )
                for operand in instruction.operands
                if operand.normalized_access
                in {AccessType.WRITE.value, AccessType.READ_WRITE.value}
            )
            output = next(region for region in writes if region.tensor in operator.outputs)
            reduction = int(instruction.attributes["k_tile"])
            duration, interval, resource = _compute_timing(
                machine, output.shape, reduction
            )
            task_id = f"{instruction.tisa_id}.payload00"
            task_ids.append(task_id)
            tasks.append(
                ExecutionTask(
                    task_id=task_id,
                    tile_id=instruction.tile_id,
                    operator_id=instruction.operator_id,
                    primitive="matmul",
                    resource=resource,
                    reads=reads,
                    writes=writes,
                    predecessors=predecessor_tasks(instruction),
                    duration_cycles=duration,
                    initiation_interval_cycles=interval,
                    program_order=program_order,
                    attributes={
                        "target_tisa_id": instruction.tisa_id,
                        "abstract_tisa_id": instruction.attributes.get("abstract_tisa_id"),
                        "m_tile": output.shape[-2],
                        "n_tile": output.shape[-1],
                        "k_tile": reduction,
                        "macs": math.prod(output.shape) * reduction,
                        "partial_sum_access": (
                            "read_modify_write"
                            if any(
                                item.normalized_access == AccessType.READ_WRITE.value
                                for item in instruction.operands
                                if item.tile_mem.role == "output"
                            )
                            else "write"
                        ),
                    },
                )
            )
            program_order += 1
            total_macs += math.prod(output.shape) * reduction
        by_target[instruction.tisa_id] = task_ids

    execution = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=tuple(tasks),
        attributes={
            "source": "target-plan-matmul-payload",
            "target_plan_id": target_plan.plan_id,
        },
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
            "transfer_bytes": transfer_bytes,
        },
        payloads={
            target_id: tuple(task_ids)
            for target_id, task_ids in by_target.items()
        },
    )
