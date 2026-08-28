from __future__ import annotations

"""Analytical backend payload for the semantic SwiGLU tile operation."""

from dataclasses import replace
import math

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    AccessType,
    BufferRegion,
    ExecutionGraph,
    ExecutionTask,
    OperatorGraph,
    ScheduleSpec,
    build_tile_graph,
)

from .elementwise import _elementwise_timing
from .matmul import (
    LoweringResult,
    _local_memory,
    _region,
    _root_memory,
    _transfer_timing,
    dtype_bytes,
)


def _temporary_region(
    tensor: str,
    template: BufferRegion,
    access: AccessType,
    *,
    dtype: str | None = None,
) -> BufferRegion:
    selected_dtype = dtype or template.dtype
    return BufferRegion(
        tensor=tensor,
        memory=template.memory,
        shape=template.shape,
        starts=template.starts,
        dtype=selected_dtype,
        access=access,
        offset_bytes=0,
        size_bytes=math.prod(template.shape) * dtype_bytes(selected_dtype),
        layout=template.layout,
    )


def lower_swiglu_graph(
    graph: OperatorGraph,
    schedule: ScheduleSpec,
    machine: MachineConfig,
) -> LoweringResult:
    """Lower SwiGLU into loads, three vector primitives, and one store."""

    graph_issues = graph.validate()
    schedule_issues = schedule.validate(graph)
    machine_issues = machine.validate()
    if graph_issues or schedule_issues or machine_issues:
        raise ValueError("; ".join((*graph_issues, *schedule_issues, *machine_issues)))

    tensors = {tensor.name: tensor for tensor in graph.tensors}
    root = _root_memory(machine)
    local = _local_memory(machine, root)
    tile_graph = build_tile_graph(graph, schedule)
    tasks: list[ExecutionTask] = []
    task_order = 0
    elements = 0
    transfer_bytes = 0

    for operator_id in graph.topological_order():
        operator = next(item for item in graph.operators if item.op_id == operator_id)
        if operator.normalized_type != "swiglu":
            raise NotImplementedError(
                f"SwiGLU lowering does not support '{operator.normalized_type}'"
            )
        if len(operator.inputs) != 2 or len(operator.outputs) != 1:
            raise ValueError(
                f"SwiGLU operator '{operator.op_id}' requires gate/up inputs and one output"
            )
        gate_tensor = tensors[operator.inputs[0]]
        up_tensor = tensors[operator.inputs[1]]
        output_tensor = tensors[operator.outputs[0]]
        if not (
            tuple(gate_tensor.shape)
            == tuple(up_tensor.shape)
            == tuple(output_tensor.shape)
        ):
            raise ValueError(
                f"SwiGLU operator '{operator.op_id}' gate/up/output shapes must match"
            )
        conversion_steps = tuple(operator.attributes.get("conversion_steps", ()))
        active_dtype = gate_tensor.dtype
        for index, step in enumerate(conversion_steps):
            if not isinstance(step, dict):
                raise ValueError(
                    f"SwiGLU operator '{operator.op_id}' conversion step {index} must be a mapping"
                )
            source_dtype = str(step.get("source_dtype", ""))
            target_dtype = str(step.get("target_dtype", ""))
            if source_dtype != active_dtype or not target_dtype:
                raise ValueError(
                    f"SwiGLU operator '{operator.op_id}' has a discontinuous conversion chain"
                )
            active_dtype = target_dtype
        if active_dtype != output_tensor.dtype:
            raise ValueError(
                f"SwiGLU operator '{operator.op_id}' conversion chain does not reach output dtype"
            )

        for tile in (
            item for item in tile_graph.tiles if item.operator_id == operator_id
        ):
            dimensions = tuple(name for name, _ in operator.iteration_dims)
            starts = tuple(tile.bound_map[name][0] for name in dimensions)
            shape = tuple(tile.extent(name) for name in dimensions)
            local_inputs: list[BufferRegion] = []
            load_ids: list[str] = []
            for operand_index, tensor in enumerate((gate_tensor, up_tensor)):
                global_region = _region(tensor, root, starts, shape, AccessType.READ)
                local_region = _region(tensor, local, starts, shape, AccessType.READ)
                duration, initiation_interval, resource = _transfer_timing(
                    machine, root, local, global_region.size_bytes
                )
                load_id = f"{tile.tile_id}.load_{operand_index}"
                tasks.append(
                    ExecutionTask(
                        task_id=load_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="load",
                        resource=resource,
                        reads=(global_region,),
                        writes=(replace(local_region, access=AccessType.WRITE),),
                        duration_cycles=duration,
                        initiation_interval_cycles=initiation_interval,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={"operand": operand_index, "iteration": tile.ordinal},
                    )
                )
                task_order += 1
                load_ids.append(load_id)
                local_inputs.append(local_region)
                transfer_bytes += global_region.size_bytes

            tile_elements = math.prod(shape)
            duration, initiation_interval, resource = _elementwise_timing(
                machine, tile_elements
            )
            logistic_write = _temporary_region(
                f"{operator_id}.logistic", local_inputs[0], AccessType.WRITE
            )
            logistic_id = f"{tile.tile_id}.logistic"
            tasks.append(
                ExecutionTask(
                    task_id=logistic_id,
                    tile_id=tile.tile_id,
                    operator_id=operator_id,
                    primitive="logistic",
                    resource=resource,
                    reads=(local_inputs[0],),
                    writes=(logistic_write,),
                    predecessors=(load_ids[0],),
                    duration_cycles=duration,
                    initiation_interval_cycles=initiation_interval,
                    stage_id=tile.stage_id,
                    program_order=task_order,
                    attributes={"semantic_family": "swiglu", "iteration": tile.ordinal},
                )
            )
            task_order += 1

            activated_write = _temporary_region(
                f"{operator_id}.activated_gate", local_inputs[0], AccessType.WRITE
            )
            silu_id = f"{tile.tile_id}.silu_multiply"
            tasks.append(
                ExecutionTask(
                    task_id=silu_id,
                    tile_id=tile.tile_id,
                    operator_id=operator_id,
                    primitive="silu_multiply",
                    resource=resource,
                    reads=(
                        local_inputs[0],
                        replace(logistic_write, access=AccessType.READ),
                    ),
                    writes=(activated_write,),
                    predecessors=(load_ids[0], logistic_id),
                    duration_cycles=duration,
                    initiation_interval_cycles=initiation_interval,
                    stage_id=tile.stage_id,
                    program_order=task_order,
                    attributes={"semantic_family": "swiglu", "iteration": tile.ordinal},
                )
            )
            task_order += 1

            active_region = replace(activated_write, access=AccessType.READ)
            active_predecessor = silu_id
            for conversion_index, step in enumerate(conversion_steps):
                target_dtype = str(step["target_dtype"])
                converted_write = _temporary_region(
                    f"{operator_id}.conversion_{conversion_index}",
                    active_region,
                    AccessType.WRITE,
                    dtype=target_dtype,
                )
                conversion_id = (
                    f"{tile.tile_id}.dtype_convert_{conversion_index}"
                )
                tasks.append(
                    ExecutionTask(
                        task_id=conversion_id,
                        tile_id=tile.tile_id,
                        operator_id=operator_id,
                        primitive="dtype_convert",
                        resource=resource,
                        reads=(active_region,),
                        writes=(converted_write,),
                        predecessors=(active_predecessor,),
                        duration_cycles=duration,
                        initiation_interval_cycles=initiation_interval,
                        stage_id=tile.stage_id,
                        program_order=task_order,
                        attributes={
                            "semantic_family": "swiglu",
                            "source_dtype": step["source_dtype"],
                            "target_dtype": target_dtype,
                            "iteration": tile.ordinal,
                        },
                    )
                )
                task_order += 1
                active_region = replace(converted_write, access=AccessType.READ)
                active_predecessor = conversion_id

            output_local = _region(
                output_tensor, local, starts, shape, AccessType.WRITE
            )
            gate_id = f"{tile.tile_id}.gate_multiply"
            tasks.append(
                ExecutionTask(
                    task_id=gate_id,
                    tile_id=tile.tile_id,
                    operator_id=operator_id,
                    primitive="gate_multiply",
                    resource=resource,
                    reads=(
                        active_region,
                        local_inputs[1],
                    ),
                    writes=(output_local,),
                    predecessors=(active_predecessor, load_ids[1]),
                    duration_cycles=duration,
                    initiation_interval_cycles=initiation_interval,
                    stage_id=tile.stage_id,
                    program_order=task_order,
                    attributes={"semantic_family": "swiglu", "iteration": tile.ordinal},
                )
            )
            task_order += 1

            output_global = _region(
                output_tensor, root, starts, shape, AccessType.WRITE
            )
            store_duration, store_ii, store_resource = _transfer_timing(
                machine, local, root, output_global.size_bytes
            )
            tasks.append(
                ExecutionTask(
                    task_id=f"{tile.tile_id}.store",
                    tile_id=tile.tile_id,
                    operator_id=operator_id,
                    primitive="store",
                    resource=store_resource,
                    reads=(replace(output_local, access=AccessType.READ),),
                    writes=(output_global,),
                    predecessors=(gate_id,),
                    duration_cycles=store_duration,
                    initiation_interval_cycles=store_ii,
                    stage_id=tile.stage_id,
                    program_order=task_order,
                    attributes={"iteration": tile.ordinal},
                )
            )
            task_order += 1
            elements += tile_elements
            transfer_bytes += output_global.size_bytes

    execution = ExecutionGraph(
        graph_id=f"{graph.graph_id}.execution",
        tasks=tuple(tasks),
        attributes={
            "source": "swiglu-lowering",
            "root_memory": root,
            "local_memory": local,
        },
    )
    issues = execution.validate()
    if issues:
        raise ValueError("; ".join(issues))
    return LoweringResult(
        tile_graph=tile_graph,
        execution_graph=execution,
        statistics={
            "tile_count": len(tile_graph.tiles),
            "task_count": len(execution.tasks),
            "elements": elements,
            "transfer_bytes": transfer_bytes,
            "composite_stage_count": 3,
        },
    )


__all__ = ["lower_swiglu_graph"]
