from __future__ import annotations

"""TISA dialect construction and analytical payload binding.

The semantic builder in this module deliberately does not inspect
``ExecutionTask``.  It derives scheduler-visible stages from the canonical
operator, schedule and tile graph.  A backend is then responsible for
materializing those stages as a payload and proving that every generated
primitive belongs to exactly one stage.
"""

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    AccessType,
    BackendArtifact,
    ExecutionTask,
    OperatorGraph,
    ScheduleSpec,
    TISADependency,
    TISAInstruction,
    TISAProgram,
    TISAOperand,
    TileGraph,
    TileInstance,
    TileMem,
    UnitMap,
    dtype_bytes,
    tensor_layout,
)
from npu_ooo.lowering import LoweringRegistry, default_lowering_registry, lower_mixed_graph


@dataclass(frozen=True)
class TISAStage:
    """Backend-independent stage contract for one semantic tile."""

    key: str
    primitive: str
    unit_map: UnitMap
    ordinal: int
    attributes: Mapping[str, Any]
    payload_primitives: tuple[str, ...] = ()


def _tile_region_to_dict(
    region: tuple[tuple[int, ...], tuple[int, ...]] | None,
) -> dict[str, list[int]] | None:
    """Serialize a GC logical region for TISA dependency provenance."""

    if region is None:
        return None
    starts, shape = region
    return {"starts": list(starts), "shape": list(shape)}


def _unit_map(primitive: str) -> UnitMap:
    if primitive in {"load", "load_transpose", "store", "copy", "transpose"}:
        return UnitMap("dma", affinity="data")
    if primitive in {"matmul", "conv2d"}:
        return UnitMap("tensor", affinity="matrix")
    return UnitMap("vector", affinity="vector")


def _operator_tiles(tile_graph: TileGraph, operator_id: str) -> tuple[TileInstance, ...]:
    return tuple(
        sorted(
            (tile for tile in tile_graph.tiles if tile.operator_id == operator_id),
            key=lambda tile: (tile.ordinal, tile.tile_id),
        )
    )


def _reduction_name(operator: Any) -> str | None:
    return operator.reduction_dims[0][0] if operator.reduction_dims else None


def _row_key(tile: TileInstance, operator: Any) -> tuple[int, ...]:
    reduction = _reduction_name(operator)
    return tuple(
        tile.bound_map[name][0]
        for name, _ in operator.iteration_dims
        if name != reduction
    )


def _is_first_reduction_tile(tile: TileInstance, operator: Any) -> bool:
    reduction = _reduction_name(operator)
    if reduction is None:
        return True
    return tile.bound_map[reduction][0] == 0


def _is_last_reduction_tile(tile: TileInstance, operator: Any) -> bool:
    reduction = _reduction_name(operator)
    if reduction is None:
        return True
    extent = dict(operator.reduction_dims)[reduction]
    return tile.bound_map[reduction][1] == extent


def _stages_for_tile(operator: Any, tile: TileInstance) -> tuple[TISAStage, ...]:
    op_type = operator.normalized_type
    stages: list[tuple[str, str]] = []
    if op_type in {"matmul", "batched_matmul", "gemv"}:
        stages.append(("load", "load"))
        if operator.attributes.get("rhs_transposed"):
            stages.append(("load_transpose", "load_transpose"))
        stages.append(("compute", "matmul"))
        if _is_last_reduction_tile(tile, operator):
            stages.append(("store", "store"))
    elif op_type == "conv2d":
        stages = [("load", "load"), ("compute", "conv2d")]
        if _is_last_reduction_tile(tile, operator):
            stages.append(("store", "store"))
    elif op_type == "batch_norm":
        stages = [("load", "load"), ("compute", "batch_norm"), ("store", "store")]
    elif op_type == "pool":
        stages = [("load", "load"), ("compute", "pool"), ("store", "store")]
    elif op_type in {"elementwise", "residual_add"}:
        stages = [("load", "load"), ("compute", "elementwise"), ("store", "store")]
    elif op_type == "reduce":
        stages = [("load", "load"), ("compute", "reduce")]
        if _is_last_reduction_tile(tile, operator):
            stages.append(("store", "store"))
    elif op_type == "softmax":
        stages = [("load", "load"), ("compute", "softmax"), ("store", "store")]
    elif op_type == "rmsnorm":
        stages = [("load", "load"), ("compute", "rmsnorm"), ("store", "store")]
    elif op_type == "layernorm":
        stages = [("load", "load"), ("compute", "layernorm"), ("store", "store")]
    elif op_type == "swiglu":
        stages = [("load", "load"), ("compute", "swiglu"), ("store", "store")]
    elif op_type == "kv_cache_update":
        stages = [("load", "load"), ("compute", "kv_cache_update"), ("store", "store")]
    elif op_type in {"reshape", "transpose", "slice"}:
        primitive = "copy" if op_type == "reshape" else "transpose"
        if op_type == "slice":
            primitive = "copy"
        stages = [("transform", primitive)]
    else:
        raise NotImplementedError(
            f"FC semantic builder does not support operator '{op_type}'"
        )
    composite_payloads = {
        "softmax": (
            ("online_update",)
            if operator.attributes.get("softmax_algorithm", "materialized") == "online"
            else ("reduce_max", "exp", "reduce_sum", "normalize")
        ),
        "rmsnorm": ("square", "reduce_sum_square", "rmsnorm"),
        "layernorm": (
            "reduce_sum",
            "layernorm_mean",
            "center",
            "reduce_sum_square",
            "layernorm",
        ),
        "swiglu": (
            "logistic",
            "silu_multiply",
            *(
                "dtype_convert"
                for _step in operator.attributes.get("conversion_steps", ())
            ),
            "gate_multiply",
        ),
        "kv_cache_update": ("kv_cache_update",),
    }

    def payload_for_stage(key: str, primitive: str) -> tuple[str, ...]:
        if key == "load":
            return ("load", "load_transpose", "copy", "transpose")
        if key == "store":
            return ("store",)
        return composite_payloads.get(op_type, (primitive,))

    # Stateful semantic operators carry their state contract on every stage.
    # This keeps the scheduler-visible TISA descriptor self-contained while
    # allowing the backend payload to remain an implementation detail.
    state_attributes = {
        key: operator.attributes[key]
        for key in (
            "stateful",
            "state_id",
            "state_buffer",
            "cache_axis",
            "cache_window",
            "update_length",
            "slice_start",
            "state_transition",
            "dynamic_index",
            "dynamic_index_operands",
            "state_update",
        )
        if key in operator.attributes
    }
    dynamic_index = operator.attributes.get("dynamic_index")
    if (
        op_type == "kv_cache_update"
        and operator.attributes.get("state_update")
        and isinstance(dynamic_index, Mapping)
    ):
        state_attributes["state_region"] = {
            "tensor": operator.attributes.get("state_buffer"),
            "dynamic_index": dict(dynamic_index),
            "window_shape": list(
                dynamic_index.get("attributes", {}).get("update_shape", ())
                if isinstance(dynamic_index.get("attributes", {}), Mapping)
                else ()
            ),
            "address_semantics": "dynamic_index_window",
        }

    return tuple(
        TISAStage(
            key=key,
            primitive=primitive,
            unit_map=_unit_map(primitive),
            ordinal=index,
            attributes={
                "tisa_stage": key,
                "primitive": primitive,
                "semantic_op": op_type,
                "payload_primitives": list(payload_for_stage(key, primitive)),
                **state_attributes,
                **(
                    {
                        "softmax_algorithm": operator.attributes.get(
                            "softmax_algorithm", "materialized"
                        )
                    }
                    if op_type == "softmax"
                    else {}
                ),
            },
            payload_primitives=payload_for_stage(key, primitive),
        )
        for index, (key, primitive) in enumerate(stages)
    )


def _readiness_condition(stage: TISAStage) -> str:
    """Name the semantic event that makes a stage eligible to issue."""

    if stage.key == "load":
        return "input_region_ready"
    if stage.key == "store":
        return "output_region_ready"
    if stage.key == "transform":
        return "full_region_ready"
    if stage.primitive in {"matmul", "batched_matmul", "gemv", "conv2d"}:
        return "operand_regions_ready"
    if stage.primitive in {
        "softmax",
        "rmsnorm",
        "layernorm",
        "swiglu",
        "reduce",
        "kv_cache_update",
        "batch_norm",
        "pool",
    }:
        return "semantic_tile_ready"
    return "operand_regions_ready"


def _dtype_bytes(dtype: str) -> int:
    return dtype_bytes(dtype, default=2)


def _resolved_tensor_shape(tensor: Any) -> tuple[int, ...] | None:
    shape = tuple(tensor.shape)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
        return None
    return shape


def _dense_region(
    tensor: Any,
    starts: tuple[int, ...],
    shape: tuple[int, ...],
) -> tuple[int, int] | None:
    full_shape = _resolved_tensor_shape(tensor)
    if full_shape is None or len(full_shape) != len(starts) or len(shape) != len(full_shape):
        return None
    if any(start < 0 or extent <= 0 or start + extent > limit for start, extent, limit in zip(starts, shape, full_shape)):
        return None
    return tensor_layout(tensor).interval(starts, shape)


def _address_expression(
    tensor_name: str,
    geometry: tuple[tuple[int, ...], tuple[int, ...]] | None,
) -> str | None:
    """Render a resolved logical tensor slice for TISA inspection."""

    if geometry is None:
        return None
    starts, shape = geometry
    slices = ", ".join(
        f"{start}:{start + extent}" for start, extent in zip(starts, shape)
    )
    return f"{tensor_name}[{slices}]"


def _dynamic_address_expression(operator: Any, tensor_name: str, geometry: Any) -> str | None:
    """Render a symbolic address for dynamic slice/state regions."""

    metadata = operator.attributes.get("dynamic_index")
    if not isinstance(metadata, Mapping):
        return _address_expression(tensor_name, geometry)
    if operator.normalized_type == "kv_cache_update":
        state_buffer = str(operator.attributes.get("state_buffer", ""))
        is_state_write = tensor_name == operator.outputs[0]
        if not is_state_write or not state_buffer:
            return _address_expression(tensor_name, geometry)
    elif operator.normalized_type != "slice" or tensor_name != operator.inputs[0]:
        return _address_expression(tensor_name, geometry)
    operands = tuple(str(item) for item in metadata.get("index_operands", ()))
    bounds = metadata.get("clamp_bounds", ())
    sizes = (
        operator.attributes.get("slice_sizes", ())
        if operator.normalized_type == "slice"
        else metadata.get("attributes", {}).get("update_shape", ())
    )
    if not operands or not bounds or len(operands) != len(bounds):
        return _address_expression(tensor_name, geometry)
    slices: list[str] = []
    for index, (operand, bound) in enumerate(zip(operands, bounds)):
        lower = bound[0] if isinstance(bound, (tuple, list)) and bound else 0
        upper = bound[1] if isinstance(bound, (tuple, list)) and len(bound) > 1 else None
        size = sizes[index] if index < len(sizes) else "?"
        upper_text = str(upper) if upper is not None else "extent"
        slices.append(f"clamp({operand},{lower},{upper_text}):+{size}")
    return f"{tensor_name}[{', '.join(slices)}]"


def _tensor_strides_bytes(tensor: Any) -> tuple[int, ...] | None:
    """Return concrete byte strides, preserving an explicit tensor layout."""

    if _resolved_tensor_shape(tensor) is None:
        return None
    return tensor_layout(tensor).strides_bytes


def _tensor_layout(tensor: Any) -> str:
    if _resolved_tensor_shape(tensor) is None:
        return str(getattr(tensor, "layout", "dense"))
    return tensor_layout(tensor).layout


def _stride_expression(strides_bytes: tuple[int, ...] | None) -> str | None:
    if strides_bytes is None:
        return None
    terms = [
        f"i{index}*{stride}"
        for index, stride in enumerate(strides_bytes)
        if stride
    ]
    return " + ".join(terms) if terms else "0"


def _tile_bounds(tile: TileInstance) -> dict[str, tuple[int, int]]:
    return tile.bound_map


def _dim_geometry(bounds: Mapping[str, tuple[int, int]], dimensions: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(bounds[name][0] for name in dimensions),
        tuple(bounds[name][1] - bounds[name][0] for name in dimensions),
    )


def _operand_geometry(
    operator: Any,
    tile: TileInstance,
    tensor: Any,
    tensors: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return dense tensor starts/shape for one semantic tile operand."""

    bounds = _tile_bounds(tile)
    iteration = tuple(name for name, _ in operator.iteration_dims)
    reduction = tuple(name for name, _ in operator.reduction_dims)
    op_type = operator.normalized_type
    name = tensor.name
    if op_type == "reshape" and (
        operator.attributes.get("broadcast")
        or operator.attributes.get("stablehlo_op") == "stablehlo.broadcast_in_dim"
    ):
        dimensions = operator.attributes.get("broadcast_dimensions")
        if not isinstance(dimensions, (tuple, list)):
            return None
        if name == operator.outputs[0]:
            return _dim_geometry(bounds, iteration)
        if name != operator.inputs[0]:
            return None
        if len(dimensions) != len(tensor.shape):
            return None
        starts: list[int] = []
        shape: list[int] = []
        for source_axis, output_axis in enumerate(dimensions):
            if not isinstance(output_axis, int) or output_axis < 0 or output_axis >= len(iteration):
                return None
            source_extent = int(tensor.shape[source_axis])
            output_start, output_stop = bounds[iteration[output_axis]]
            output_extent = output_stop - output_start
            if source_extent == 1:
                starts.append(0)
                shape.append(1)
            elif source_extent == output_extent or output_stop <= source_extent:
                starts.append(output_start)
                shape.append(output_extent)
            else:
                return None
        return tuple(starts), tuple(shape)
    if op_type == "slice":
        full_shape = _resolved_tensor_shape(tensor)
        if full_shape is None:
            return None
        if name == operator.outputs[0]:
            return _dim_geometry(bounds, iteration)
        if name == operator.inputs[0]:
            dynamic_index = operator.attributes.get("dynamic_index")
            if isinstance(dynamic_index, Mapping):
                sizes = operator.attributes.get("slice_sizes")
                if isinstance(sizes, (tuple, list)) and len(sizes) == len(full_shape):
                    return (0,) * len(full_shape), tuple(int(value) for value in sizes)
            starts = operator.attributes.get("slice_starts")
            limits = operator.attributes.get("slice_limits")
            if (
                isinstance(starts, (tuple, list))
                and isinstance(limits, (tuple, list))
                and len(starts) == len(full_shape)
                and len(limits) == len(full_shape)
            ):
                return (
                    tuple(int(value) for value in starts),
                    tuple(int(limit) - int(start) for start, limit in zip(starts, limits)),
                )
            return (0,) * len(full_shape), full_shape
        return None
    if op_type in {"matmul", "batched_matmul", "gemv"}:
        if len(operator.inputs) < 2 or not operator.outputs:
            return None
        batch = iteration[:-2]
        out0, out1 = iteration[-2:]
        red = reduction[0] if len(reduction) == 1 else None
        if red is None:
            return None
        batch_starts, batch_shape = _dim_geometry(bounds, batch)
        out0_start, out0_stop = bounds[out0]
        out1_start, out1_stop = bounds[out1]
        red_start, red_stop = bounds[red]
        out0_shape = out0_stop - out0_start
        out1_shape = out1_stop - out1_start
        red_shape = red_stop - red_start
        if name == operator.inputs[0]:
            return (*batch_starts, out0_start, red_start), (*batch_shape, out0_shape, red_shape)
        if name == operator.inputs[1]:
            broadcast_batch = bool(operator.attributes.get("rhs_broadcast_batch", False))
            rhs_starts = () if broadcast_batch else batch_starts
            rhs_shape = () if broadcast_batch else batch_shape
            if operator.attributes.get("rhs_transposed"):
                return (*rhs_starts, out1_start, red_start), (*rhs_shape, out1_shape, red_shape)
            return (*rhs_starts, red_start, out1_start), (*rhs_shape, red_shape, out1_shape)
        if name == operator.outputs[0]:
            return (*batch_starts, out0_start, out1_start), (*batch_shape, out0_shape, out1_shape)
        return None

    if op_type == "conv2d":
        if len(operator.inputs) < 2 or len(operator.outputs) != 1:
            return None
        if tuple(name for name, _ in operator.iteration_dims) != ("N", "O", "OH", "OW"):
            return None
        if len(operator.reduction_dims) != 1:
            return None
        n_start, n_stop = bounds["N"]
        o_start, o_stop = bounds["O"]
        oh_start, oh_stop = bounds["OH"]
        ow_start, ow_stop = bounds["OW"]
        k_start, k_stop = bounds[operator.reduction_dims[0][0]]
        input_tensor = tensors[operator.inputs[0]]
        weight_tensor = tensors[operator.inputs[1]]
        channels = int(input_tensor.shape[1])
        kernel_h, kernel_w = (int(value) for value in operator.attributes.get("kernel_shape", (1, 1)))
        input_h, input_w = int(input_tensor.shape[2]), int(input_tensor.shape[3])
        stride_h, stride_w = (int(value) for value in operator.attributes.get("window_strides", (1, 1)))
        pad = tuple(int(value) for value in operator.attributes.get("padding", (0, 0, 0, 0)))
        pad_top, _pad_bottom, pad_left, _pad_right = pad
        if name == operator.outputs[0]:
            return (n_start, o_start, oh_start, ow_start), (n_stop - n_start, o_stop - o_start, oh_stop - oh_start, ow_stop - ow_start)
        if name == operator.inputs[1]:
            return (0, 0, 0, 0), tuple(int(value) for value in weight_tensor.shape)
        if name == operator.inputs[0]:
            # A flattened K tile maps to channel/kernel coordinates.  Use a
            # conservative halo region so dependency construction never
            # under-approximates a convolution window.
            input_oh = max(0, oh_start * stride_h - pad_top)
            input_ow = max(0, ow_start * stride_w - pad_left)
            input_h_extent = min(input_h - input_oh, (oh_stop - oh_start - 1) * stride_h + kernel_h)
            input_w_extent = min(input_w - input_ow, (ow_stop - ow_start - 1) * stride_w + kernel_w)
            return (n_start, 0, input_oh, input_ow), (
                n_stop - n_start,
                channels,
                max(1, input_h_extent),
                max(1, input_w_extent),
            )
        return None

    if op_type == "pool":
        if len(operator.inputs) != 1 or len(operator.outputs) != 1:
            return None
        iteration = tuple(name for name, _ in operator.iteration_dims)
        if iteration != ("d0", "d1", "d2", "d3") or len(tensor.shape) != 4:
            return None
        bounds = tile.bound_map
        starts = tuple(bounds[name][0] for name in iteration)
        output_shape = tuple(bounds[name][1] - bounds[name][0] for name in iteration)
        if name == operator.outputs[0]:
            return starts, output_shape
        if name != operator.inputs[0]:
            return None
        window = tuple(int(value) for value in operator.attributes.get("window_dimensions", (1, 1, 1, 1)))
        stride = tuple(int(value) for value in operator.attributes.get("window_strides", (1, 1, 1, 1)))
        padding = tuple(int(value) for value in operator.attributes.get("padding", (0,) * 8))
        if len(window) != 4 or len(stride) != 4 or len(padding) != 8:
            return None
        input_start_h = max(0, starts[2] * stride[2] - padding[4])
        input_start_w = max(0, starts[3] * stride[3] - padding[6])
        input_h_extent = min(
            int(tensor.shape[2]) - input_start_h,
            (output_shape[2] - 1) * stride[2] + window[2],
        )
        input_w_extent = min(
            int(tensor.shape[3]) - input_start_w,
            (output_shape[3] - 1) * stride[3] + window[3],
        )
        return (
            (starts[0], starts[1], input_start_h, input_start_w),
            (output_shape[0], output_shape[1], max(1, input_h_extent), max(1, input_w_extent)),
        )

    if op_type == "batch_norm":
        if len(operator.inputs) != 5 or len(operator.outputs) != 1:
            return None
        if len(tensor.shape) == 4:
            output_dims = tuple(name for name, _ in operator.iteration_dims)
            if output_dims != ("d0", "d1", "d2", "d3"):
                return None
            starts, shape = _dim_geometry(bounds, output_dims)
            if name in {operator.inputs[0], operator.outputs[0]}:
                return starts, shape
        if len(tensor.shape) == 1 and name in operator.inputs[1:]:
            feature_index = int(operator.attributes.get("feature_index", 1))
            if feature_index < 0 or feature_index >= len(operator.iteration_dims):
                return None
            feature_name = tuple(name for name, _ in operator.iteration_dims)[feature_index]
            start, stop = bounds[feature_name]
            return (start,), (stop - start,)
        return None

    if op_type in {"elementwise", "residual_add"}:
        output = operator.outputs[0]
        output_dims = iteration
        output_starts, output_shape = _dim_geometry(bounds, output_dims)
        output_tensor = tensor if name == output else None
        if output_tensor is not None:
            return output_starts, output_shape
        full_shape = _resolved_tensor_shape(tensor)
        if full_shape is None or len(full_shape) > len(output_shape):
            return None
        leading = len(output_shape) - len(full_shape)
        starts: list[int] = []
        shape: list[int] = []
        for axis, extent in enumerate(full_shape):
            output_axis = leading + axis
            if extent == 1:
                starts.append(0)
                shape.append(1)
            else:
                starts.append(output_starts[output_axis])
                shape.append(output_shape[output_axis])
        return tuple(starts), tuple(shape)

    if op_type == "kv_cache_update":
        full_shape = _resolved_tensor_shape(tensor)
        if full_shape is None:
            return None
        update_name = operator.attributes.get("update_tensor")
        dynamic_index = operator.attributes.get("dynamic_index")
        if (
            isinstance(dynamic_index, Mapping)
            and tensor.name == update_name
            and len(operator.inputs) > 1
        ):
            update_tensor = tensors[operator.inputs[1]]
            update_shape = _resolved_tensor_shape(update_tensor)
            if update_shape is not None:
                return (0,) * len(update_shape), update_shape
        return (0,) * len(full_shape), full_shape

    if op_type in {"reshape", "transpose"}:
        full_shape = _resolved_tensor_shape(tensor)
        if full_shape is None:
            return None
        return (0,) * len(full_shape), full_shape

    all_dimensions = (*iteration, *reduction)
    if name in operator.outputs and op_type == "reduce":
        return _dim_geometry(bounds, iteration)
    tensor_shape = _resolved_tensor_shape(tensor)
    if tensor_shape is None:
        return None
    if len(tensor_shape) == len(all_dimensions):
        return _dim_geometry(bounds, all_dimensions)
    if len(tensor_shape) == len(iteration):
        return _dim_geometry(bounds, iteration)
    if reduction and len(tensor_shape) == len(reduction):
        return _dim_geometry(bounds, reduction)
    return None


def _stage_operands(
    operator: Any,
    tile: TileInstance,
    stage: TISAStage,
    tensors: Mapping[str, Any],
) -> tuple[TISAOperand, ...]:
    is_load = stage.primitive in {"load", "load_transpose"}
    is_store = stage.primitive == "store"
    names: list[tuple[str, AccessType]] = []
    if not is_store:
        names.extend((name, AccessType.READ) for name in operator.inputs)
    if not is_load:
        names.extend((name, AccessType.WRITE) for name in operator.outputs)
    if not names:
        names.append((operator.outputs[0], AccessType.WRITE))
    operands: list[TISAOperand] = []
    seen: set[tuple[str, str]] = set()
    for index, (name, access) in enumerate(names):
        key = (name, access.value)
        if key in seen:
            continue
        seen.add(key)
        tensor = tensors[name]
        geometry = _operand_geometry(operator, tile, tensor, tensors)
        operand_shape = geometry[1] if geometry is not None else tuple(
            stop - start for _name, start, stop in tile.bounds
        ) or (1,)
        dense_region = _dense_region(tensor, *geometry) if geometry is not None else None
        offset_bytes = dense_region[0] if dense_region is not None else None
        size_bytes = dense_region[1] if dense_region is not None else None
        strides_bytes = _tensor_strides_bytes(tensor)
        operands.append(
            TISAOperand(
                name=f"{name}:{access.value}:{index}",
                tile_shape=operand_shape,
                tile_mem=TileMem(
                    base=name,
                    scope="logical",
                    tensor=name,
                    offset_bytes=offset_bytes,
                    size_bytes=size_bytes,
                    address_expr=_dynamic_address_expression(operator, name, geometry),
                    strides_bytes=strides_bytes,
                    stride_expr=_stride_expression(strides_bytes),
                    layout=_tensor_layout(tensor),
                    logical_starts=geometry[0] if geometry is not None else None,
                    logical_shape=geometry[1] if geometry is not None else None,
                ),
                access_type=access,
            )
        )
    return tuple(operands)


class TISASemanticBuilder:
    """Build TISA descriptors from semantic graph and tile information only."""

    def build(
        self,
        graph: OperatorGraph,
        schedule: ScheduleSpec,
        tile_graph: TileGraph,
        machine: MachineConfig,
        *,
        program_id: str,
    ) -> TISAProgram:
        graph_issues = graph.validate()
        schedule_issues = schedule.validate(graph)
        tile_issues = tile_graph.validate()
        machine_issues = machine.validate()
        if graph_issues or schedule_issues or tile_issues or machine_issues:
            raise ValueError(
                "; ".join((*graph_issues, *schedule_issues, *tile_issues, *machine_issues))
            )
        operators = {operator.op_id: operator for operator in graph.operators}
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        tiles = {tile.tile_id: tile for tile in tile_graph.tiles}
        stage_map: dict[tuple[str, str], TISAStage] = {}
        instructions: list[TISAInstruction] = []
        source_order: dict[str, int] = {}
        for order, tile_id in enumerate(tile_graph.topological_order()):
            tile = tiles[tile_id]
            operator = operators[tile.operator_id]
            for stage in _stages_for_tile(operator, tile):
                tisa_id = f"tisa.{tile_id}.s{stage.ordinal:02d}"
                stage_map[(tile_id, stage.key)] = stage
                source_order[tisa_id] = order * 100 + stage.ordinal
                instructions.append(
                    TISAInstruction(
                        tisa_id=tisa_id,
                        tile_id=tile_id,
                        operator_id=operator.op_id,
                        # ``op_type`` names the scheduler-visible stage.  The
                        # semantic operator family remains explicit in the
                        # attributes so backend-independent analyses retain
                        # the composite identity (e.g. softmax/rmsnorm).
                        op_type=stage.primitive,
                        operands=_stage_operands(operator, tile, stage, tensors),
                        unit_map=stage.unit_map,
                        attributes={
                            **dict(tile.attributes),
                            **dict(stage.attributes),
                            "semantic_boundary": "tile",
                            "semantic_tile_id": tile_id,
                            "semantic_op_type": operator.normalized_type,
                            "paper_stage": "FC",
                            "scheduler_visible": True,
                            "readiness_condition": _readiness_condition(stage),
                            "source_program_order": source_order[tisa_id],
                        },
                        payload_ref=f"payload:{tisa_id}",
                    )
                )

        by_id = {instruction.tisa_id: instruction for instruction in instructions}
        dependencies: dict[str, dict[str, tuple[str, str, Mapping[str, Any]]]] = {
            tisa_id: {} for tisa_id in by_id
        }

        def add_dependency(
            target: str,
            source: str,
            kind: str = "RAW",
            *,
            condition: str | None = None,
            provenance: Mapping[str, Any] | None = None,
        ) -> None:
            if target == source:
                return
            current = dependencies[target].get(source)
            normalized_condition = str(
                condition
                or by_id[target].attributes.get("readiness_condition", "full_region_ready")
            )
            candidate = (kind, normalized_condition, dict(provenance or {}))
            if current is None or (current[0] != "RAW" and kind == "RAW"):
                dependencies[target][source] = candidate
                return
            if current[0] == kind and current[1] == normalized_condition and provenance:
                merged = dict(current[2])
                source_names = {
                    str(name)
                    for name in (
                        merged.get("source"),
                        *merged.get("sources", ()),
                        provenance.get("source"),
                        *provenance.get("sources", ()),
                    )
                    if name
                }
                if source_names:
                    merged["sources"] = sorted(source_names)
                for key, value in provenance.items():
                    if key not in merged:
                        merged[key] = value
                dependencies[target][source] = (current[0], current[1], merged)

        def instruction_id(tile_id: str, key: str) -> str:
            try:
                stage = stage_map[(tile_id, key)]
            except KeyError as exc:
                raise ValueError(f"missing TISA stage '{key}' for tile '{tile_id}'") from exc
            return f"tisa.{tile_id}.s{stage.ordinal:02d}"

        for tile in tile_graph.tiles:
            stages = _stages_for_tile(operators[tile.operator_id], tile)
            for previous, current in zip(stages, stages[1:]):
                add_dependency(
                    instruction_id(tile.tile_id, current.key),
                    instruction_id(tile.tile_id, previous.key),
                    provenance={
                        "source": "tile_stage_order",
                        "tile_id": tile.tile_id,
                        "from_stage": previous.key,
                        "to_stage": current.key,
                    },
                )

        # Graph edges are represented at the semantic tile boundary.  The
        # producer's terminal stage is sufficient because backend payloads are
        # bound only after the descriptor program has been built.
        for dependency in tile_graph.dependencies:
            producer = tiles[dependency.producer]
            consumer = tiles[dependency.consumer]
            producer_stages = _stages_for_tile(operators[producer.operator_id], producer)
            consumer_stages = _stages_for_tile(operators[consumer.operator_id], consumer)
            if dependency.kind in {"state", "accumulate"}:
                producer_stage = next(
                    (stage for stage in producer_stages if stage.key == "compute"),
                    producer_stages[-1],
                )
                consumer_stage = next(
                    (stage for stage in consumer_stages if stage.key == "compute"),
                    consumer_stages[0],
                )
            elif dependency.kind == "buffer_reuse":
                producer_stage = producer_stages[-1]
                consumer_stage = consumer_stages[0]
            else:
                producer_stage = producer_stages[-1]
                consumer_stage = consumer_stages[0]
            add_dependency(
                instruction_id(consumer.tile_id, consumer_stage.key),
                instruction_id(producer.tile_id, producer_stage.key),
                dependency.hazard_kind,
                condition=dependency.condition,
                provenance={
                    "source": "gc_tile_dependency",
                    "tile_dependency_kind": dependency.kind,
                    "tensor": dependency.tensor,
                    "producer_tile": dependency.producer,
                    "consumer_tile": dependency.consumer,
                    "producer_region": _tile_region_to_dict(dependency.producer_region),
                    "consumer_region": _tile_region_to_dict(dependency.consumer_region),
                    **dict(dependency.provenance),
                },
            )

        # Reduction barriers and matrix partial accumulation are semantic
        # dependencies, not artifacts of a particular primitive graph.
        by_operator: dict[str, list[TileInstance]] = {}
        for tile in tile_graph.tiles:
            by_operator.setdefault(tile.operator_id, []).append(tile)
        for operator_id, op_tiles in by_operator.items():
            operator = operators[operator_id]
            op_type = operator.normalized_type
            rows: dict[tuple[int, ...], list[TileInstance]] = {}
            for tile in op_tiles:
                rows.setdefault(_row_key(tile, operator), []).append(tile)
            reduction = _reduction_name(operator)
            for row_tiles in rows.values():
                ordered = sorted(
                    row_tiles,
                    key=lambda tile: tile.bound_map[reduction][0] if reduction else tile.ordinal,
                )
                if op_type in {"matmul", "batched_matmul", "gemv", "conv2d"} and reduction:
                    output_dims = tuple(name for name, _ in operator.iteration_dims if name != reduction)
                    groups: dict[tuple[int, ...], list[TileInstance]] = {}
                    for tile in ordered:
                        groups.setdefault(
                            tuple(tile.bound_map[name][0] for name in output_dims), []
                        ).append(tile)
                    for output_tiles in groups.values():
                        output_tiles.sort(key=lambda tile: tile.bound_map[reduction][0])
                        for previous, current in zip(output_tiles, output_tiles[1:]):
                            add_dependency(
                                instruction_id(current.tile_id, "compute"),
                                instruction_id(previous.tile_id, "compute"),
                                "ACCUMULATE",
                                condition="accumulate_ready",
                                provenance={
                                    "source": "matmul_partial_accumulate",
                                    "operator_id": operator_id,
                                },
                            )
                barrier_stage = {
                    "reduce": "compute",
                    "rmsnorm": "compute",
                    "layernorm": "compute",
                }.get(op_type)
                if barrier_stage and reduction:
                    for previous, current in zip(ordered, ordered[1:]):
                        add_dependency(
                            instruction_id(current.tile_id, barrier_stage),
                            instruction_id(previous.tile_id, barrier_stage),
                                "ACCUMULATE" if op_type in {"matmul", "batched_matmul", "gemv", "conv2d"} else "STATE",
                                condition=(
                                    "accumulate_ready"
                                    if op_type in {"matmul", "batched_matmul", "gemv", "conv2d"}
                                    else "state_complete"
                                ),
                                provenance={
                                    "source": "reduction_order",
                                    "operator_id": operator_id,
                                    "stage": barrier_stage,
                                },
                        )
                if op_type == "softmax" and reduction:
                    if operator.attributes.get("softmax_algorithm", "materialized") == "online":
                        for previous, current in zip(ordered, ordered[1:]):
                            add_dependency(
                                instruction_id(current.tile_id, "compute"),
                                instruction_id(previous.tile_id, "compute"),
                                "STATE",
                                condition="state_complete",
                                provenance={
                                    "source": "softmax_online_state_chain",
                                    "operator_id": operator_id,
                                },
                            )
                    else:
                        # Materialized softmax computes the final row max/sum
                        # before all exp/normalize payloads.  The semantic FC
                        # op therefore depends on the final reduction tile.
                        final = ordered[-1]
                        for tile in ordered[:-1]:
                            add_dependency(
                                instruction_id(tile.tile_id, "compute"),
                                instruction_id(final.tile_id, "compute"),
                                "STATE",
                                condition="state_complete",
                                provenance={
                                    "source": "softmax_materialized_reduction",
                                    "operator_id": operator_id,
                                },
                            )

        # Stable topological order is part of the descriptor contract.
        successors = {tisa_id: [] for tisa_id in by_id}
        indegree = {tisa_id: 0 for tisa_id in by_id}
        for target, sources in dependencies.items():
            for source in sources:
                successors[source].append(target)
                indegree[target] += 1
        ready = sorted(
            (tisa_id for tisa_id, degree in indegree.items() if degree == 0),
            key=lambda tisa_id: (source_order[tisa_id], tisa_id),
        )
        ordered: list[TISAInstruction] = []
        while ready:
            current = ready.pop(0)
            instruction = by_id[current]
            ordered.append(
                replace(
                    instruction,
                    dependencies=tuple(
                        TISADependency(
                            source=source,
                            kind=kind,
                            condition=condition,
                            provenance=provenance,
                        )
                        for source, (kind, condition, provenance) in sorted(
                            dependencies[current].items()
                        )
                    ),
                    attributes={
                        **dict(instruction.attributes),
                        "program_order": len(ordered),
                    },
                )
            )
            for successor in successors[current]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort(key=lambda tisa_id: (source_order[tisa_id], tisa_id))
        if len(ordered) != len(instructions):
            raise ValueError("FC semantic dependency graph contains a cycle")
        program = TISAProgram(
            program_id=program_id,
            instructions=tuple(ordered),
            attributes={
                "source": "tile-graph-semantic-builder",
                "scheduler_granularity": "tisa_tile",
                "codegen_direction": "tilegraph->tisa->backend-payload",
            },
        )
        issues = program.validate()
        if issues:
            raise ValueError("FC program construction failed: " + "; ".join(issues))
        return program


class AnalyticalBackendCodegen:
    """Materialize an already-built TISA program as analytical payloads."""

    def lower(
        self,
        graph: OperatorGraph,
        schedule: ScheduleSpec,
        tile_graph: TileGraph,
        machine: MachineConfig,
        *,
        program: TISAProgram,
        registry: LoweringRegistry | None = None,
    ) -> BackendArtifact:
        lowering = lower_mixed_graph(
            graph,
            schedule,
            machine,
            registry=registry or default_lowering_registry(),
            tile_graph=tile_graph,
        )
        tasks_by_stage: dict[tuple[str, str], list[ExecutionTask]] = {}
        for task in lowering.execution_graph.tasks:
            tasks_by_stage.setdefault((task.tile_id, task.primitive), []).append(task)
        payloads: dict[str, tuple[str, ...]] = {}
        consumed: set[str] = set()
        original_tasks = {task.task_id: task for task in lowering.execution_graph.tasks}

        composite_ops = {"softmax", "rmsnorm", "layernorm", "swiglu", "kv_cache_update"}
        for instruction in program.instructions:
            stage = str(instruction.attributes.get("tisa_stage", ""))
            semantic_op = str(instruction.attributes.get("semantic_op_type", ""))
            if semantic_op in composite_ops:
                if stage == "load":
                    primitive_names = {"load", "load_transpose", "copy", "transpose"}
                elif stage == "store":
                    primitive_names = {"store"}
                else:
                    primitive_names = set(
                        instruction.attributes.get("payload_primitives", ())
                    )
                candidates = tuple(
                    task
                    for task in lowering.execution_graph.tasks
                    if task.tile_id == instruction.tile_id
                    and task.primitive in primitive_names
                    and task.task_id not in consumed
                )
                if not candidates:
                    raise ValueError(
                        f"analytical backend has no composite payload for TISA instruction "
                        f"'{instruction.tisa_id}' stage '{stage}'"
                    )
                resources = {task.resource for task in candidates}
                if len(resources) != 1:
                    raise ValueError(
                        f"composite TISA payload '{instruction.tisa_id}' spans resources: "
                        + ", ".join(sorted(resources))
                    )
                # Keep the detailed primitive DAG in BackendArtifact.  It is
                # expanded only after the semantic TISA instruction issues;
                # the primitive tasks never enter the global ready queue.
                payloads[instruction.tisa_id] = tuple(task.task_id for task in candidates)
                consumed.update(task.task_id for task in candidates)
                continue

            primitive = str(instruction.attributes.get("primitive", ""))
            candidates = tuple(tasks_by_stage.get((instruction.tile_id, primitive), ()))
            if not candidates:
                raise ValueError(
                    f"analytical backend has no payload for TISA instruction '{instruction.tisa_id}' "
                    f"stage '{primitive}' on tile '{instruction.tile_id}'"
                )
            task_ids = tuple(task.task_id for task in candidates)
            payloads[instruction.tisa_id] = task_ids
            consumed.update(task_ids)

        all_tasks = set(original_tasks)
        unbound = sorted(all_tasks - consumed)
        if unbound:
            raise ValueError(
                "analytical backend generated primitive tasks without TISA ownership: "
                + ", ".join(unbound[:8])
            )

        execution_graph = lowering.execution_graph
        artifact = BackendArtifact(
            artifact_id=f"{graph.graph_id}.analytical-tisa-dialect",
            program=program,
            execution_graph=execution_graph,
            payloads=payloads,
            backend="analytical",
            attributes={
                "calibration_status": "analytical",
                "codegen_direction": "tilegraph->tisa->analytical-payload",
                "lowering_statistics": lowering.statistics,
            },
        )
        issues = artifact.validate()
        if issues:
            raise ValueError("FC backend artifact is invalid: " + "; ".join(issues))
        return artifact
