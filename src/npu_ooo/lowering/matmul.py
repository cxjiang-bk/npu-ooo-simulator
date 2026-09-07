from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_graph": self.tile_graph.to_dict(),
            "execution_graph": self.execution_graph.to_dict(),
            "statistics": dict(self.statistics),
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


def _valid_bytes(region: BufferRegion) -> int:
    return region.elements * dtype_bytes(region.dtype)


def _packed_strides(shape: tuple[int, ...], element_size: int) -> tuple[int, ...]:
    stride = element_size
    result: list[int] = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def _placed_region(
    region: BufferRegion,
    *,
    memory: str,
    buffer_id: str,
    access: AccessType,
    role: str,
    slot: int | None,
    packed: bool,
) -> BufferRegion:
    valid_bytes = _valid_bytes(region)
    return BufferRegion(
        tensor=region.tensor,
        memory=memory,
        shape=region.shape,
        starts=region.starts,
        dtype=region.dtype,
        access=access,
        offset_bytes=0 if packed else region.offset_bytes,
        size_bytes=valid_bytes if packed else region.size_bytes,
        layout="packed" if packed else region.layout,
        strides_bytes=(
            _packed_strides(region.shape, dtype_bytes(region.dtype))
            if packed
            else region.strides_bytes
        ),
        buffer_id=buffer_id,
        valid_bytes=valid_bytes,
        attributes={
            "operand_role": role,
            "slot": slot,
            "range_semantics": "packed_valid_bytes" if packed else "source_bounding_span",
        },
    )


def _route_stage_keys(
    machine: MachineConfig,
    routes: Mapping[str, tuple[str, ...]],
    direction: str,
) -> dict[tuple[str, int], str]:
    """Mirror the FC route grouping without coupling lowering to FC tasks."""

    result: dict[tuple[str, int], str] = {}
    maximum_hops = max((len(route) - 1 for route in routes.values()), default=0)
    for hop in range(maximum_hops):
        engines = sorted(
            {
                _path(machine, route[hop], route[hop + 1]).engine
                for route in routes.values()
                if hop + 1 < len(route)
            }
        )
        engine_index = {engine: index for index, engine in enumerate(engines)}
        for role, route in routes.items():
            if hop + 1 >= len(route):
                continue
            engine = _path(machine, route[hop], route[hop + 1]).engine
            result[(role, hop)] = (
                f"{direction}_hop_{hop:02d}_{engine_index[engine]:02d}"
            )
    return result


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
    try:
        placement = machine.placement("matmul")
        role_placements = {
            role: placement.operand(role) for role in ("lhs", "rhs", "output")
        }
    except KeyError as exc:
        raise ValueError(
            f"machine '{machine.config_id}' has no explicit matmul operand placement"
        ) from exc
    tasks: list[ExecutionTask] = []
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
        ping_pong = op_schedule.attributes.get("ping_pong", {})
        slot_count = int(ping_pong.get("buffer_count", 1)) if isinstance(ping_pong, Mapping) else 1
        slot_count = max(1, slot_count)
        output_keys = sorted(
            {
                tuple(
                    tile.bound_map[name][0]
                    for name, _extent in operator.iteration_dims
                )
                for tile in tiles
            }
        )
        output_slots = {
            key: index % slot_count for index, key in enumerate(output_keys)
        }
        input_routes = {
            role: role_placements[role].route for role in ("lhs", "rhs")
        }
        output_routes = {"output": role_placements["output"].route}
        input_stage_keys = _route_stage_keys(machine, input_routes, "input")
        output_stage_keys = _route_stage_keys(machine, output_routes, "output")
        reduction_name = operator.reduction_dims[0][0]
        output_dims = tuple(name for name, _ in operator.iteration_dims)
        for tile in tiles:
            left, right, output, reduction_shape = _matmul_regions(operator, tensors, tile)
            tile_prefix = tile.tile_id
            input_slot = tile.ordinal % slot_count
            output_key = (operator_id, tuple(tile.bound_map[name][0] for name in output_dims))
            output_slot = output_slots[output_key[1]]
            base_regions = {"lhs": left, "rhs": right, "output": output}

            final_input_tasks: list[str] = []
            for role in ("lhs", "rhs"):
                route = role_placements[role].route
                source_region = _placed_region(
                    base_regions[role],
                    memory=route[0],
                    buffer_id=f"{base_regions[role].tensor}@{route[0]}",
                    access=AccessType.READ,
                    role=role,
                    slot=None,
                    packed=False,
                )
                predecessors = {
                    store_id
                    for produced_region, store_id in producer_stores.get(source_region.tensor, [])
                    if _regions_overlap(produced_region, source_region)
                }
                previous_id: str | None = None
                for hop, (source, target) in enumerate(zip(route, route[1:])):
                    destination = _placed_region(
                        base_regions[role],
                        memory=target,
                        buffer_id=f"{operator_id}.{role}.slot{input_slot}@{target}",
                        access=AccessType.WRITE,
                        role=role,
                        slot=input_slot,
                        packed=role_placements[role].layout == "packed",
                    )
                    duration, interval, unit = _transfer_timing(
                        machine, source, target, _valid_bytes(base_regions[role])
                    )
                    path = _path(machine, source, target)
                    task_id = f"{tile_prefix}.{role}.hop{hop:02d}"
                    task_predecessors = set(predecessors if hop == 0 else ())
                    if previous_id is not None:
                        task_predecessors.add(previous_id)
                    tasks.append(
                        ExecutionTask(
                            task_id=task_id,
                            tile_id=tile.tile_id,
                            operator_id=operator_id,
                            primitive=str(path.transform or "load"),
                            resource=unit,
                            reads=(source_region,),
                            writes=(destination,),
                            predecessors=tuple(sorted(task_predecessors)),
                            duration_cycles=duration,
                            initiation_interval_cycles=interval,
                            stage_id=tile.stage_id,
                            program_order=task_order,
                            attributes={
                                "operand": role,
                                "iteration": tile.ordinal,
                                "route_hop": hop,
                                "route": list(route),
                                "tisa_stage_key": input_stage_keys[(role, hop)],
                                "transfer_bytes": _valid_bytes(base_regions[role]),
                            },
                        )
                    )
                    task_order += 1
                    total_transfer_bytes += _valid_bytes(base_regions[role])
                    source_region = BufferRegion(
                        **{**destination.__dict__, "access": AccessType.READ}
                    )
                    previous_id = task_id
                if previous_id is not None:
                    final_input_tasks.append(previous_id)

            compute_id = f"{tile_prefix}.mxu"
            compute_duration, compute_ii, compute_unit = _compute_timing(
                machine, output.shape, reduction_shape
            )
            previous = compute_by_output.setdefault(output_key, [])
            predecessors = set(final_input_tasks)
            if previous:
                predecessors.add(max(previous, key=lambda item: item[0])[1])
            lhs_compute = _placed_region(
                left,
                memory=role_placements["lhs"].memory,
                buffer_id=f"{operator_id}.lhs.slot{input_slot}@{role_placements['lhs'].memory}",
                access=AccessType.READ,
                role="lhs",
                slot=input_slot,
                packed=role_placements["lhs"].layout == "packed",
            )
            rhs_compute = _placed_region(
                right,
                memory=role_placements["rhs"].memory,
                buffer_id=f"{operator_id}.rhs.slot{input_slot}@{role_placements['rhs'].memory}",
                access=AccessType.READ,
                role="rhs",
                slot=input_slot,
                packed=role_placements["rhs"].layout == "packed",
            )
            output_buffer_id = (
                f"{operator_id}.output.slot{output_slot}@{role_placements['output'].memory}"
            )
            output_read = _placed_region(
                output,
                memory=role_placements["output"].memory,
                buffer_id=output_buffer_id,
                access=AccessType.READ,
                role="output",
                slot=output_slot,
                packed=role_placements["output"].layout == "packed",
            )
            output_write = BufferRegion(
                **{**output_read.__dict__, "access": AccessType.WRITE}
            )
            first_partial = not previous
            tasks.append(
                ExecutionTask(
                    task_id=compute_id,
                    tile_id=tile.tile_id,
                    operator_id=operator_id,
                    primitive="matmul",
                    resource=compute_unit,
                    reads=(lhs_compute, rhs_compute, *((output_read,) if not first_partial else ())),
                    writes=(output_write,),
                    predecessors=tuple(sorted(predecessors)),
                    duration_cycles=compute_duration,
                    initiation_interval_cycles=compute_ii,
                    stage_id=tile.stage_id,
                    program_order=task_order,
                    attributes={
                        "batch_tile": list(output.shape[:-2]),
                        "m_tile": output.shape[-2],
                        "n_tile": output.shape[-1],
                        "k_tile": reduction_shape,
                        "macs": math.prod(output.shape) * reduction_shape,
                        "rhs_transposed": bool(operator.attributes.get("rhs_transposed", False)),
                        "rhs_broadcast_batch": bool(
                            operator.attributes.get("rhs_broadcast_batch", False)
                        ),
                        "iteration": tile.ordinal,
                        "partial_sum_access": "write" if first_partial else "read_modify_write",
                        "tisa_stage_key": "compute",
                    },
                )
            )
            task_order += 1
            previous.append((tile.bound_map[reduction_name][0], compute_id))
            if tile.bound_map[reduction_name][1] == dict(operator.reduction_dims)[reduction_name]:
                route = role_placements["output"].route
                source_region = BufferRegion(
                    **{**output_write.__dict__, "access": AccessType.READ}
                )
                previous_id = compute_id
                for hop, (source, target) in enumerate(zip(route, route[1:])):
                    is_root_target = target == root
                    destination = _placed_region(
                        output,
                        memory=target,
                        buffer_id=(
                            f"{output.tensor}@{target}"
                            if is_root_target
                            else f"{operator_id}.output.slot{output_slot}@{target}"
                        ),
                        access=AccessType.WRITE,
                        role="output",
                        slot=None if is_root_target else output_slot,
                        packed=(
                            False
                            if is_root_target
                            else role_placements["output"].layout == "packed"
                        ),
                    )
                    duration, interval, unit = _transfer_timing(
                        machine, source, target, _valid_bytes(output)
                    )
                    path = _path(machine, source, target)
                    task_id = f"{tile_prefix}.output.hop{hop:02d}"
                    tasks.append(
                        ExecutionTask(
                            task_id=task_id,
                            tile_id=tile.tile_id,
                            operator_id=operator_id,
                            primitive=str(path.transform or "store"),
                            resource=unit,
                            reads=(source_region,),
                            writes=(destination,),
                            predecessors=(previous_id,),
                            duration_cycles=duration,
                            initiation_interval_cycles=interval,
                            stage_id=tile.stage_id,
                            program_order=task_order,
                            attributes={
                                "operand": "output",
                                "iteration": tile.ordinal,
                                "route_hop": hop,
                                "route": list(route),
                                "final_reduction_tile": True,
                                "tisa_stage_key": output_stage_keys[("output", hop)],
                                "transfer_bytes": _valid_bytes(output),
                            },
                        )
                    )
                    task_order += 1
                    total_transfer_bytes += _valid_bytes(output)
                    source_region = BufferRegion(
                        **{**destination.__dict__, "access": AccessType.READ}
                    )
                    previous_id = task_id
                producer_stores.setdefault(output.tensor, []).append(
                    (source_region, previous_id)
                )
            total_macs += math.prod(output.shape) * reduction_shape

    execution = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=tuple(tasks),
        attributes={
            "source": "matmul-target-lowering",
            "root_memory": root,
            "operand_placement": {
                role: role_placements[role].to_dict()
                for role in ("lhs", "rhs", "output")
            },
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
            "transfer_bytes": total_transfer_bytes,
        },
    )
