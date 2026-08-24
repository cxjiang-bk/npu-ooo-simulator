from __future__ import annotations

import math
from typing import Any

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    AccessType,
    BufferRegion,
    ExecutionGraph,
    ExecutionTask,
    OperatorGraph,
    ScheduleSpec,
    TileGraph,
    TileInstance,
    build_tile_graph,
)
from npu_ooo.ir.model import ModelInstance

from .matmul import (
    LoweringResult,
    _local_memory,
    _path,
    _region,
    _root_memory,
    _transfer_timing,
    _unit_for,
    _regions_overlap,
    dtype_bytes,
)


def _elementwise_timing(machine: MachineConfig, elements: int) -> tuple[float, float, str]:
    unit = _unit_for(machine, "elementwise")
    configured_rate = unit.attributes.get("elements_per_cycle", unit.attributes.get("lanes", 1))
    rate = float(configured_rate) if isinstance(configured_rate, (int, float)) and configured_rate > 0 else 1.0
    duration = unit.latency_cycles + math.ceil(elements / rate)
    return float(duration), float(unit.initiation_interval_cycles), unit.name


def _pointwise_identity(operator: Any) -> tuple[str, str, str]:
    frontend_target = str(operator.attributes.get("frontend_target", ""))
    semantic_op = str(operator.attributes.get("semantic_op", ""))
    if not semantic_op:
        normalized_target = frontend_target.lower().replace("::", ".")
        if normalized_target.startswith(("stablehlo.", "mhlo.")):
            semantic_op = normalized_target.split(".", 1)[1]
        elif normalized_target.startswith("aten."):
            semantic_op = normalized_target.removeprefix("aten.").split(".", 1)[0]
        else:
            semantic_op = operator.normalized_type
    timing_key = str(
        operator.attributes.get("backend_capability_key")
        or operator.attributes.get("timing_key")
        or f"pointwise.{semantic_op}"
    )
    return semantic_op, frontend_target, timing_key


def _output_region(operator, tensors: dict[str, Any], tile: TileInstance, memory: str, access: AccessType) -> BufferRegion:
    output = tensors[operator.outputs[0]]
    dimensions = tuple(name for name, _ in operator.iteration_dims)
    bounds = tile.bound_map
    starts = tuple(bounds[name][0] for name in dimensions)
    shape = tuple(bounds[name][1] - bounds[name][0] for name in dimensions)
    return _region(output, memory, starts, shape, access)


def _broadcast_input_region(
    input_tensor: Any,
    output_shape: tuple[int, ...],
    dimensions: tuple[str, ...],
    tile: TileInstance,
    memory: str,
) -> BufferRegion:
    input_shape = tuple(input_tensor.shape)
    if len(input_shape) > len(output_shape):
        raise ValueError(f"input '{input_tensor.name}' rank exceeds elementwise output rank")
    padded_shape = (1,) * (len(output_shape) - len(input_shape)) + input_shape
    starts: list[int] = []
    shape: list[int] = []
    for axis, (input_extent, output_extent) in enumerate(zip(padded_shape, output_shape)):
        if input_extent not in {1, output_extent}:
            raise ValueError(
                f"input '{input_tensor.name}' shape {input_shape} cannot broadcast to {output_shape}"
            )
        if axis < len(output_shape) - len(input_shape):
            continue
        if input_extent == 1:
            starts.append(0)
            shape.append(1)
        else:
            start, stop = tile.bound_map[dimensions[axis]]
            starts.append(start)
            shape.append(stop - start)
    return _region(input_tensor, memory, tuple(starts), tuple(shape), AccessType.READ)


def lower_elementwise_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> LoweringResult:
    """Lower elementwise/residual-add operators into load -> ARU -> store tasks."""

    graph_issues = graph.validate()
    schedule_issues = schedule.validate(graph)
    machine_issues = machine.validate()
    if graph_issues or schedule_issues or machine_issues:
        raise ValueError("; ".join((*graph_issues, *schedule_issues, *machine_issues)))
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    root = _root_memory(machine)
    local = _local_memory(machine, root)
    tasks: list[ExecutionTask] = []
    producer_stores: dict[str, list[tuple[BufferRegion, str]]] = {}
    task_order = 0
    transfer_bytes = 0
    elements = 0

    for operator_id in graph.topological_order():
        operator = next(operator for operator in graph.operators if operator.op_id == operator_id)
        if operator.normalized_type not in {"elementwise", "residual_add"}:
            raise NotImplementedError(
                f"elementwise lowering does not support operator type '{operator.normalized_type}'"
            )
        if len(operator.outputs) != 1 or not operator.inputs:
            raise ValueError(f"elementwise operator '{operator.op_id}' requires inputs and one output")
        semantic_op, frontend_target, timing_key = _pointwise_identity(operator)
        dimensions = tuple(name for name, _ in operator.iteration_dims)
        if not dimensions or operator.reduction_dims:
            raise ValueError(f"elementwise operator '{operator.op_id}' requires iteration dimensions only")
        output_shape = tuple(tensors[operator.outputs[0]].shape)
        for input_name in operator.inputs:
            input_shape = tuple(tensors[input_name].shape)
            padded_shape = (1,) * (len(output_shape) - len(input_shape)) + input_shape
            if len(input_shape) > len(output_shape) or any(
                input_extent not in {1, output_extent}
                for input_extent, output_extent in zip(padded_shape, output_shape)
            ):
                raise ValueError(
                    f"elementwise operator '{operator.op_id}' input shape {input_shape} "
                    f"cannot broadcast to output shape {output_shape}"
                )
        from npu_ooo.ir.tile import enumerate_operator_tiles

        tiles = enumerate_operator_tiles(operator, schedule.for_operator(operator_id))
        output_tensor = tensors[operator.outputs[0]]
        for tile in tiles:
            output_global = _output_region(operator, tensors, tile, root, AccessType.WRITE)
            output_local = _output_region(operator, tensors, tile, local, AccessType.WRITE)
            output_local_read = _output_region(operator, tensors, tile, local, AccessType.READ)
            load_ids: list[str] = []
            load_regions: list[BufferRegion] = []
            for input_index, input_name in enumerate(operator.inputs):
                input_tensor = tensors[input_name]
                input_global = _broadcast_input_region(
                    input_tensor, output_shape, dimensions, tile, root
                )
                input_local = _broadcast_input_region(
                    input_tensor, output_shape, dimensions, tile, local
                )
                load_id = f"{tile.tile_id}.load_{input_index}"
                predecessors = {
                    store_id
                    for region, store_id in producer_stores.get(input_name, [])
                    if _regions_overlap(region, input_global)
                }
                duration, ii, unit = _transfer_timing(machine, root, local, input_global.size_bytes)
                tasks.append(
                    ExecutionTask(
                        task_id=load_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="load",
                        resource=unit,
                        reads=(input_global,),
                        writes=(BufferRegion(**{**input_local.__dict__, "access": AccessType.WRITE}),),
                        predecessors=tuple(sorted(predecessors)),
                        duration_cycles=duration,
                        initiation_interval_cycles=ii,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"operand": input_index, "iteration": tile.ordinal},
                    )
                )
                task_order += 1
                load_ids.append(load_id)
                load_regions.append(input_local)
                transfer_bytes += input_global.size_bytes

            tile_elements = math.prod(output_local.shape)
            compute_duration, compute_ii, compute_unit = _elementwise_timing(machine, tile_elements)
            compute_id = f"{tile.tile_id}.elementwise"
            tasks.append(
                ExecutionTask(
                    task_id=compute_id,
                    tile_id=tile.tile_id,
                    operator_id=operator_id,
                    primitive="elementwise",
                    resource=compute_unit,
                    reads=tuple(load_regions),
                    writes=(output_local,),
                    predecessors=tuple(load_ids),
                    duration_cycles=compute_duration,
                    initiation_interval_cycles=compute_ii,
                    stage_id=tile.stage_id,
                    program_order=task_order,
                    attributes={
                        "elements": tile_elements,
                        "operand_arity": int(
                            operator.attributes.get("operand_arity", len(operator.inputs))
                        ),
                        "input_count": len(operator.inputs),
                        "iteration": tile.ordinal,
                        "semantic_family": "elementwise",
                        "semantic_op": semantic_op,
                        "frontend_target": frontend_target,
                        "timing_key": timing_key,
                    },
                )
            )
            task_order += 1
            store_id = f"{tile.tile_id}.store"
            store_duration, store_ii, store_unit = _transfer_timing(machine, local, root, output_local.size_bytes)
            tasks.append(
                ExecutionTask(
                    task_id=store_id,
                    tile_id=tile.tile_id,
                    operator_id=operator_id,
                    primitive="store",
                    resource=store_unit,
                    reads=(output_local_read,),
                    writes=(output_global,),
                    predecessors=(compute_id,),
                    duration_cycles=store_duration,
                    initiation_interval_cycles=store_ii,
                    stage_id=tile.stage_id,
                    program_order=task_order,
                    attributes={"iteration": tile.ordinal},
                )
            )
            task_order += 1
            producer_stores.setdefault(output_tensor.name, []).append((output_global, store_id))
            transfer_bytes += output_local.size_bytes
            elements += tile_elements

    execution = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=tuple(tasks),
        attributes={"source": "elementwise-lowering", "root_memory": root, "local_memory": local},
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
            "elements": elements,
            "transfer_bytes": transfer_bytes,
        },
    )


def lower_elementwise(model: ModelInstance, machine: MachineConfig, schedule: ScheduleSpec) -> LoweringResult:
    return lower_elementwise_graph(model.graph, schedule, machine)
