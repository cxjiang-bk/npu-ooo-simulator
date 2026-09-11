from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Mapping

from .operator import OperatorGraph, OperatorSpec
from .schedule import OperatorSchedule, ScheduleSpec


@dataclass(frozen=True)
class TileInstance:
    tile_id: str
    operator_id: str
    ordinal: int
    coordinates: tuple[tuple[str, int], ...]
    bounds: tuple[tuple[str, int, int], ...]
    stage_id: int = 0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def bound_map(self) -> dict[str, tuple[int, int]]:
        return {name: (start, stop) for name, start, stop in self.bounds}

    @property
    def coordinate_map(self) -> dict[str, int]:
        return dict(self.coordinates)

    def extent(self, dimension: str) -> int:
        start, stop = self.bound_map[dimension]
        return stop - start

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "operator_id": self.operator_id,
            "ordinal": self.ordinal,
            "coordinates": {name: index for name, index in self.coordinates},
            "bounds": {name: [start, stop] for name, start, stop in self.bounds},
            "stage_id": self.stage_id,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class TileDependency:
    """A dependency between two semantic tiles.

    ``kind`` names the GC semantic reason for the edge.  ``hazard_kind`` is
    the paper-facing access relation consumed by FC/TISA.  Region coordinates
    stay in logical tensor space so the edge can be audited before runtime
    address binding.
    """

    producer: str
    consumer: str
    tensor: str | None = None
    kind: str = "region_data"
    hazard_kind: str = "RAW"
    producer_region: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    consumer_region: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    producer_regions: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    consumer_regions: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    condition: str = "full_region_ready"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def relation(self) -> str:
        """Return the normalized paper hazard relation."""

        return self.hazard_kind

    @staticmethod
    def _validate_region(
        region: tuple[tuple[int, ...], tuple[int, ...]] | None,
        *,
        label: str,
    ) -> tuple[str, ...]:
        if region is None:
            return ()
        starts, shape = region
        issues: list[str] = []
        if len(starts) != len(shape):
            issues.append(f"tile dependency {label} region starts and shape must have equal rank")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in starts
        ):
            issues.append(f"tile dependency {label} region starts must be non-negative integers")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape
        ):
            issues.append(f"tile dependency {label} region shape must contain positive integers")
        return tuple(issues)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.producer or not self.consumer:
            issues.append("tile dependency producer and consumer must not be empty")
        if self.producer == self.consumer:
            issues.append(f"tile dependency '{self.producer}' cannot depend on itself")
        if self.kind not in {
            "data",
            "region_data",
            "state",
            "accumulate",
            "buffer_reuse",
            "control",
        }:
            issues.append(f"tile dependency kind '{self.kind}' is unsupported")
        if self.hazard_kind not in {
            "RAW",
            "WAR",
            "WAW",
            "STATE",
            "ACCUMULATE",
            "BUFFER_REUSE",
            "CONTROL",
        }:
            issues.append(f"tile dependency hazard kind '{self.hazard_kind}' is unsupported")
        if not self.condition:
            issues.append("tile dependency condition must not be empty")
        issues.extend(self._validate_region(self.producer_region, label="producer"))
        issues.extend(self._validate_region(self.consumer_region, label="consumer"))
        for index, region in enumerate(self.producer_regions):
            issues.extend(self._validate_region(region, label=f"producer[{index}]"))
        for index, region in enumerate(self.consumer_regions):
            issues.extend(self._validate_region(region, label=f"consumer[{index}]"))
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "consumer": self.consumer,
            "tensor": self.tensor,
            "kind": self.kind,
            "hazard_kind": self.hazard_kind,
            "relation": self.relation,
            "producer_regions": [
                self._region_to_dict(region) for region in self.producer_regions
            ],
            "consumer_regions": [
                self._region_to_dict(region) for region in self.consumer_regions
            ],
            "producer_region": self._region_to_dict(self.producer_region),
            "consumer_region": self._region_to_dict(self.consumer_region),
            "condition": self.condition,
            "provenance": dict(self.provenance),
        }

    @staticmethod
    def _region_to_dict(
        region: tuple[tuple[int, ...], tuple[int, ...]] | None,
    ) -> dict[str, list[int]] | None:
        if region is None:
            return None
        starts, shape = region
        return {"starts": list(starts), "shape": list(shape)}


@dataclass(frozen=True)
class TileGraph:
    graph_id: str
    tiles: tuple[TileInstance, ...]
    dependencies: tuple[TileDependency, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        tile_ids = {tile.tile_id for tile in self.tiles}
        if len(tile_ids) != len(self.tiles):
            issues.append("tile ids must be unique")
        for dependency in self.dependencies:
            issues.extend(dependency.validate())
            if dependency.producer not in tile_ids:
                issues.append(f"tile dependency references unknown producer '{dependency.producer}'")
            if dependency.consumer not in tile_ids:
                issues.append(f"tile dependency references unknown consumer '{dependency.consumer}'")
            if dependency.producer == dependency.consumer:
                issues.append(f"tile '{dependency.producer}' cannot depend on itself")
        try:
            self.topological_order()
        except ValueError as exc:
            issues.append(str(exc))
        return tuple(issues)

    def topological_order(self) -> tuple[str, ...]:
        ids = [tile.tile_id for tile in self.tiles]
        order_index = {tile_id: index for index, tile_id in enumerate(ids)}
        outgoing = {tile_id: set() for tile_id in ids}
        indegree = {tile_id: 0 for tile_id in ids}
        for dependency in self.dependencies:
            if dependency.producer not in outgoing or dependency.consumer not in indegree:
                raise ValueError("cannot topologically order tile graph with unknown dependency endpoint")
            if dependency.consumer not in outgoing[dependency.producer]:
                outgoing[dependency.producer].add(dependency.consumer)
                indegree[dependency.consumer] += 1
        ready = sorted((tile_id for tile_id, degree in indegree.items() if degree == 0), key=order_index.__getitem__)
        result: list[str] = []
        while ready:
            current = ready.pop(0)
            result.append(current)
            for successor in sorted(outgoing[current], key=order_index.__getitem__):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort(key=order_index.__getitem__)
        if len(result) != len(ids):
            raise ValueError(f"tile graph '{self.graph_id}' contains a cycle")
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "tiles": [tile.to_dict() for tile in self.tiles],
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
            "attributes": dict(self.attributes),
        }


def enumerate_operator_tiles(operator: OperatorSpec, schedule: OperatorSchedule) -> tuple[TileInstance, ...]:
    """Expand a resolved operator schedule, retaining partial boundary tiles."""

    dimensions = tuple(schedule.loop_order or tuple(name for name, _ in operator.iteration_dims + operator.reduction_dims))
    extents = dict((*operator.iteration_dims, *operator.reduction_dims))
    ranges: list[tuple[tuple[int, int], ...]] = []
    for dimension in dimensions:
        extent = extents[dimension]
        tile_size = schedule.tile_size(dimension)
        ranges.append(tuple((start, min(start + tile_size, extent)) for start in range(0, extent, tile_size)))
    tiles: list[TileInstance] = []
    for ordinal, selected in enumerate(product(*ranges)):
        coordinates = tuple((dimension, index) for index, dimension in enumerate(dimensions))
        # Coordinates are tile indices, while bounds retain the concrete tensor ranges.
        coordinates = tuple(
            (dimension, start // schedule.tile_size(dimension))
            for dimension, (start, _stop) in zip(dimensions, selected)
        )
        bounds = tuple((dimension, start, stop) for dimension, (start, stop) in zip(dimensions, selected))
        semantic_attributes = {
            key: value
            for key, value in (
                (
                    "semantic_family",
                    operator.attributes.get("semantic_family", operator.normalized_type),
                ),
                ("semantic_op", operator.attributes.get("semantic_op")),
                ("stablehlo_op", operator.attributes.get("stablehlo_op")),
                ("operand_arity", operator.attributes.get("operand_arity")),
                (
                    "backend_capability_key",
                    operator.attributes.get("backend_capability_key"),
                ),
                ("semantic_region_id", operator.attributes.get("semantic_region_id")),
                (
                    "semantic_region_family",
                    operator.attributes.get("semantic_region_family"),
                ),
                ("semantic_region_role", operator.attributes.get("semantic_region_role")),
                (
                    "semantic_region_opaque",
                    operator.attributes.get("semantic_region_opaque"),
                ),
                ("rotary_algorithm", operator.attributes.get("rotary_algorithm")),
                ("rotary_embedding", operator.attributes.get("rotary_embedding")),
                ("moe_routing_contract", operator.attributes.get("moe_routing_contract")),
                ("moe_dispatch_contract", operator.attributes.get("moe_dispatch_contract")),
                ("stateful", operator.attributes.get("stateful")),
                ("state_id", operator.attributes.get("state_id")),
                ("state_buffer", operator.attributes.get("state_buffer")),
                ("state_update", operator.attributes.get("state_update")),
                ("dynamic_index", operator.attributes.get("dynamic_index")),
            )
            if value is not None and value != ""
        }
        if (
            operator.attributes.get("state_update")
            and isinstance(operator.attributes.get("dynamic_index"), Mapping)
        ):
            dynamic_index = operator.attributes["dynamic_index"]
            metadata = dynamic_index.get("attributes", {})
            semantic_attributes["state_region"] = {
                "tensor": operator.attributes.get("state_buffer"),
                "dynamic_index": dict(dynamic_index),
                "window_shape": list(
                    metadata.get("update_shape", ())
                    if isinstance(metadata, Mapping)
                    else ()
                ),
                "address_semantics": "dynamic_index_window",
            }
        tiles.append(
            TileInstance(
                tile_id=f"{operator.op_id}.t{ordinal:04d}",
                operator_id=operator.op_id,
                ordinal=ordinal,
                coordinates=coordinates,
                bounds=bounds,
                stage_id=schedule.stage_id,
                attributes=semantic_attributes,
            )
        )
    return tuple(tiles)


def _dimension_region(
    tile: TileInstance,
    dimensions: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(tile.bound_map[name][0] for name in dimensions),
        tuple(tile.extent(name) for name in dimensions),
    )


LogicalRegion = tuple[tuple[int, ...], tuple[int, ...]]

def _shape_elements(shape):
    result = 1
    for extent in shape:
        result *= int(extent)
    return result

def _linear_to_coords(shape, index):
    if index < 0 or index >= _shape_elements(shape):
        raise ValueError(f"linear index {index} is outside shape {shape}")
    if not shape:
        return ()
    coordinates = [0] * len(shape)
    remaining = index
    for axis in range(len(shape) - 1, -1, -1):
        coordinates[axis] = remaining % int(shape[axis])
        remaining //= int(shape[axis])
    return tuple(coordinates)

def _tile_row_intervals(tile, dimensions, tensor_shape):
    starts, extents = _dimension_region(tile, dimensions)
    if len(starts) != len(tensor_shape):
        raise ValueError(f"tile dimensions rank {len(starts)} does not match tensor rank {len(tensor_shape)}")
    if not tensor_shape:
        return (((), 0, 1),)
    prefix_ranges = [range(starts[axis], starts[axis] + extents[axis]) for axis in range(len(tensor_shape) - 1)]
    rows = []
    for prefix in product(*prefix_ranges) if prefix_ranges else [()]:
        coordinate = (*prefix, starts[-1])
        linear_start = sum(int(value) * _shape_elements(tensor_shape[axis + 1 :]) for axis, value in enumerate(coordinate))
        rows.append((coordinate, linear_start, extents[-1]))
    return tuple(rows)

def reshape_tile_mappings(tile, dimensions, input_shape, output_shape):
    if _shape_elements(input_shape) != _shape_elements(output_shape):
        raise ValueError(f"reshape element count mismatch: {input_shape} -> {output_shape}")
    mappings = []
    for output_coordinate, linear_start, row_length in _tile_row_intervals(tile, dimensions, output_shape):
        consumed = 0
        while consumed < row_length:
            source_coordinate = _linear_to_coords(input_shape, linear_start + consumed)
            source_available = input_shape[-1] - source_coordinate[-1] if input_shape else 1
            length = min(row_length - consumed, source_available)
            output_start = (*output_coordinate[:-1], output_coordinate[-1] + consumed) if output_shape else ()
            output_segment_shape = (*((1,) * (len(output_shape) - 1)), length) if output_shape else ()
            input_segment_shape = (*((1,) * (len(input_shape) - 1)), length) if input_shape else ()
            mappings.append(((output_start, output_segment_shape), (source_coordinate, input_segment_shape)))
            consumed += length
    return tuple(mappings)

def slice_tile_mapping(tile, dimensions, starts, strides):
    output_region = _dimension_region(tile, dimensions)
    output_starts, output_shape = output_region
    input_starts = tuple(int(start) + int(offset) * int(stride) for start, offset, stride in zip(starts, output_starts, strides))
    return output_region, (input_starts, output_shape)

def concatenate_tile_mappings(tile, dimensions, input_shapes, output_shape, dimension):
    if dimension < 0:
        dimension += len(output_shape)
    if dimension < 0 or dimension >= len(output_shape):
        raise ValueError(f"concatenate dimension {dimension} is out of range")
    output_starts, output_extents = _dimension_region(tile, dimensions)
    output_begin, output_end = output_starts[dimension], output_starts[dimension] + output_extents[dimension]
    mappings, cursor = [], 0
    for input_index, shape in enumerate(input_shapes):
        input_begin, input_end = cursor, cursor + shape[dimension]
        cursor = input_end
        overlap_begin, overlap_end = max(output_begin, input_begin), min(output_end, input_end)
        if overlap_begin >= overlap_end:
            continue
        output_start, output_extent = list(output_starts), list(output_extents)
        output_start[dimension], output_extent[dimension] = overlap_begin, overlap_end - overlap_begin
        input_start, input_extent = list(output_starts), list(output_extents)
        input_start[dimension], input_extent[dimension] = overlap_begin - input_begin, overlap_end - overlap_begin
        mappings.append((input_index, (tuple(output_start), tuple(output_extent)), (tuple(input_start), tuple(input_extent))))
    if not mappings:
        raise ValueError("concatenate tile does not intersect any input")
    return tuple(mappings)

def _bounding_region(regions):
    if not regions:
        return None
    rank = len(regions[0][0])
    starts = tuple(min(region[0][axis] for region in regions) for axis in range(rank))
    stops = tuple(max(region[0][axis] + region[1][axis] for region in regions) for axis in range(rank))
    return starts, tuple(stop - start for start, stop in zip(starts, stops))

def _tile_tensor_regions(operator, tile, tensor_shape, tensor_name, *, tensor_shapes=None):
    iteration = tuple(name for name, _extent in operator.iteration_dims)
    op_type = operator.normalized_type
    shapes = tensor_shapes or {}
    output_shape = tuple(shapes.get(operator.outputs[0], tensor_shape))
    if op_type == "reshape" and (operator.attributes.get("broadcast") or operator.attributes.get("stablehlo_op") == "stablehlo.broadcast_in_dim"):
        if tensor_name == operator.outputs[0]:
            return (_dimension_region(tile, iteration),)
        if tensor_name != operator.inputs[0]:
            return None
        dimensions = operator.attributes.get("broadcast_dimensions")
        if not isinstance(dimensions, (tuple, list)):
            return None
        source_shape = tuple(shapes.get(operator.inputs[0], tensor_shape))
        if len(dimensions) != len(source_shape):
            return None
        starts, shape = [], []
        for source_axis, output_axis in enumerate(dimensions):
            if not isinstance(output_axis, int) or output_axis < 0 or output_axis >= len(iteration):
                return None
            source_extent = source_shape[source_axis]
            output_start, output_stop = tile.bound_map[iteration[output_axis]]
            if source_extent == 1:
                starts.append(0); shape.append(1)
            elif output_stop <= source_extent:
                starts.append(output_start); shape.append(output_stop - output_start)
            else:
                return None
        return ((tuple(starts), tuple(shape)),)
    if op_type == "reshape":
        if tensor_name not in {operator.inputs[0], operator.outputs[0]}:
            return None
        input_shape = tuple(shapes.get(operator.inputs[0], tensor_shape))
        mappings = reshape_tile_mappings(tile, iteration, input_shape, output_shape)
        if tensor_name == operator.outputs[0]:
            return tuple(output_region for output_region, _input_region in mappings)
        return tuple(input_region for _output_region, input_region in mappings)
    if op_type == "slice":
        if tensor_name == operator.outputs[0]:
            return (_dimension_region(tile, iteration),)
        if tensor_name != operator.inputs[0]:
            return None
        dynamic_index = operator.attributes.get("dynamic_index")
        if isinstance(dynamic_index, Mapping):
            sizes = operator.attributes.get("slice_sizes")
            if isinstance(sizes, (tuple, list)) and len(sizes) == len(tensor_shape):
                return (((0,) * len(tensor_shape), tuple(int(value) for value in sizes)),)
            return (((0,) * len(tensor_shape), tensor_shape),)
        starts = operator.attributes.get("slice_starts")
        strides = operator.attributes.get("slice_strides")
        if isinstance(starts, (tuple, list)) and isinstance(strides, (tuple, list)) and len(starts) == len(tensor_shape) and len(strides) == len(tensor_shape):
            return (slice_tile_mapping(tile, iteration, tuple(int(value) for value in starts), tuple(int(value) for value in strides))[1],)
        return (((0,) * len(tensor_shape), tensor_shape),)
    if op_type == "concatenate":
        if tensor_name not in (*operator.inputs, *operator.outputs):
            return None
        dimension = operator.attributes.get("concatenate_dimension")
        if not isinstance(dimension, int):
            return None
        input_shapes = tuple(tuple(shapes.get(name, tensor_shape)) for name in operator.inputs)
        mappings = concatenate_tile_mappings(
            tile, iteration, input_shapes, output_shape, int(dimension)
        )
        if tensor_name == operator.outputs[0]:
            return tuple(output_region for _index, output_region, _input_region in mappings)
        input_index = operator.inputs.index(tensor_name)
        return tuple(
            input_region
            for index, _output_region, input_region in mappings
            if index == input_index
        )
    if op_type == "transpose":
        if tensor_name == operator.outputs[0]:
            return (_dimension_region(tile, iteration),)
        if tensor_name != operator.inputs[0]:
            return None
        permutation = operator.attributes.get("transpose_dims")
        if not isinstance(permutation, (tuple, list)):
            return None
        output_starts, output_shape_tile = _dimension_region(tile, iteration)
        source_starts, source_shape = [0] * len(permutation), [0] * len(permutation)
        for output_axis, source_axis in enumerate(permutation):
            source_starts[int(source_axis)], source_shape[int(source_axis)] = output_starts[output_axis], output_shape_tile[output_axis]
        return ((tuple(source_starts), tuple(source_shape)),)
    # Only transform operators have multi-region projections here. Returning
    # None for compute operators preserves their specialized projections
    # in the legacy single-region fallback below.
    return None

def _tile_tensor_region(
    operator: OperatorSpec,
    tile: TileInstance,
    tensor_shape: tuple[int, ...],
    tensor_name: str,
    *,
    tensor_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Project one operator tile onto a logical tensor region."""

    mapped = _tile_tensor_regions(
        operator, tile, tensor_shape, tensor_name, tensor_shapes=tensor_shapes
    )
    if mapped is not None:
        if mapped and isinstance(mapped[0], tuple) and mapped[0] and isinstance(mapped[0][0], int):
            mapped = (mapped,)
        return _bounding_region(tuple(mapped))

    iteration = tuple(name for name, _extent in operator.iteration_dims)
    reduction = tuple(name for name, _extent in operator.reduction_dims)
    op_type = operator.normalized_type

    if op_type == "reshape" and (
        operator.attributes.get("broadcast")
        or operator.attributes.get("stablehlo_op") == "stablehlo.broadcast_in_dim"
    ):
        if tensor_name == operator.outputs[0]:
            return _dimension_region(tile, iteration)
        if tensor_name != operator.inputs[0]:
            return None
        dimensions = operator.attributes.get("broadcast_dimensions")
        if not isinstance(dimensions, (tuple, list)):
            return None
        source_shape = tuple(tensor_shape)
        if len(dimensions) != len(source_shape):
            return None
        output_bounds = tile.bound_map
        starts: list[int] = []
        shape: list[int] = []
        for source_axis, output_axis in enumerate(dimensions):
            if not isinstance(output_axis, int) or output_axis < 0 or output_axis >= len(iteration):
                return None
            source_extent = source_shape[source_axis]
            output_start, output_stop = output_bounds[iteration[output_axis]]
            if source_extent == 1:
                starts.append(0)
                shape.append(1)
            elif source_extent == output_stop - output_start or source_extent > 1:
                if output_stop > source_extent:
                    return None
                starts.append(output_start)
                shape.append(output_stop - output_start)
            else:
                return None
        return tuple(starts), tuple(shape)

    if op_type == "slice":
        if tensor_name == operator.outputs[0]:
            return _dimension_region(tile, iteration)
        if tensor_name != operator.inputs[0]:
            return None
        dynamic_index = operator.attributes.get("dynamic_index")
        if isinstance(dynamic_index, Mapping):
            sizes = operator.attributes.get("slice_sizes")
            if isinstance(sizes, (tuple, list)) and len(sizes) == len(tensor_shape):
                return (0,) * len(tensor_shape), tuple(int(value) for value in sizes)
        starts = operator.attributes.get("slice_starts")
        limits = operator.attributes.get("slice_limits")
        if (
            isinstance(starts, (tuple, list))
            and isinstance(limits, (tuple, list))
            and len(starts) == len(tensor_shape)
            and len(limits) == len(tensor_shape)
        ):
            return (
                tuple(int(value) for value in starts),
                tuple(int(limit) - int(start) for start, limit in zip(starts, limits)),
            )
        return (0,) * len(tensor_shape), tensor_shape

    if op_type == "embedding":
        if len(operator.inputs) != 2 or len(operator.outputs) != 1:
            return None
        if tensor_name == operator.outputs[0]:
            return _dimension_region(tile, iteration)
        if tensor_name == operator.inputs[0]:
            return (0,) * len(tensor_shape), tensor_shape
        if tensor_name == operator.inputs[1]:
            output_starts, output_shape = _dimension_region(tile, iteration)
            if len(tensor_shape) > len(output_shape):
                return None
            return output_starts[: len(tensor_shape)], output_shape[: len(tensor_shape)]
        return None

    if op_type == "kv_cache_update":
        update_name = operator.attributes.get("update_tensor")
        dynamic_index = operator.attributes.get("dynamic_index")
        if (
            isinstance(dynamic_index, Mapping)
            and tensor_name == update_name
            and len(operator.inputs) > 1
        ):
            metadata = dynamic_index.get("attributes", {})
            raw_shape = metadata.get("update_shape") if isinstance(metadata, Mapping) else None
            if isinstance(raw_shape, (tuple, list)) and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in raw_shape
            ):
                update_shape = tuple(raw_shape)
                return (0,) * len(update_shape), update_shape
        return (0,) * len(tensor_shape), tensor_shape

    if op_type in {"reshape", "transpose"}:
        return (0,) * len(tensor_shape), tensor_shape

    if op_type in {"matmul", "batched_matmul", "gemv"}:
        if len(iteration) < 2 or len(reduction) != 1:
            return None
        batch = iteration[:-2]
        row, column = iteration[-2:]
        inner = reduction[0]
        batch_starts, batch_shape = _dimension_region(tile, batch)
        row_start, column_start, inner_start = (
            tile.bound_map[row][0],
            tile.bound_map[column][0],
            tile.bound_map[inner][0],
        )
        row_shape, column_shape, inner_shape = (
            tile.extent(row),
            tile.extent(column),
            tile.extent(inner),
        )
        if tensor_name == operator.outputs[0]:
            return (
                (*batch_starts, row_start, column_start),
                (*batch_shape, row_shape, column_shape),
            )
        if tensor_name == operator.inputs[0]:
            return (
                (*batch_starts, row_start, inner_start),
                (*batch_shape, row_shape, inner_shape),
            )
        if len(operator.inputs) > 1 and tensor_name == operator.inputs[1]:
            broadcast_batch = bool(operator.attributes.get("rhs_broadcast_batch", False))
            rhs_starts = () if broadcast_batch else batch_starts
            rhs_shape = () if broadcast_batch else batch_shape
            if operator.attributes.get("rhs_transposed"):
                return (
                    (*rhs_starts, column_start, inner_start),
                    (*rhs_shape, column_shape, inner_shape),
                )
            return (
                (*rhs_starts, inner_start, column_start),
                (*rhs_shape, inner_shape, column_shape),
            )
        return None

    if op_type == "conv2d":
        # The output tile is indexed by N/O/OH/OW while the input operand
        # carries a spatial halo for the kernel window.  Keeping this
        # projection in TileGraph aligned with the analytical lowering is
        # required for exact cross-operator readiness dependencies.
        if (
            len(operator.inputs) < 2
            or len(operator.outputs) != 1
            or tuple(iteration) != ("N", "O", "OH", "OW")
            or len(reduction) != 1
            or len(tensor_shape) != 4
        ):
            return None
        bounds = tile.bound_map
        n_start, n_stop = bounds["N"]
        o_start, o_stop = bounds["O"]
        oh_start, oh_stop = bounds["OH"]
        ow_start, ow_stop = bounds["OW"]
        kernel_h, kernel_w = tuple(
            int(value)
            for value in operator.attributes.get("kernel_shape", (1, 1))
        )
        stride_h, stride_w = tuple(
            int(value)
            for value in operator.attributes.get("window_strides", (1, 1))
        )
        pad_top, _pad_bottom, pad_left, _pad_right = tuple(
            int(value) for value in operator.attributes.get("padding", (0, 0, 0, 0))
        )
        if tensor_name == operator.outputs[0]:
            return (
                (n_start, o_start, oh_start, ow_start),
                (n_stop - n_start, o_stop - o_start, oh_stop - oh_start, ow_stop - ow_start),
            )
        if tensor_name == operator.inputs[1]:
            # Weight tiles are currently materialized as a full OIHW tensor by
            # the analytical backend; retain that conservative region here.
            return (0, 0, 0, 0), tensor_shape
        if tensor_name == operator.inputs[0]:
            input_start_h = max(0, oh_start * stride_h - pad_top)
            input_start_w = max(0, ow_start * stride_w - pad_left)
            input_h = tensor_shape[2]
            input_w = tensor_shape[3]
            input_h_extent = min(
                input_h - input_start_h,
                (oh_stop - oh_start - 1) * stride_h + kernel_h,
            )
            input_w_extent = min(
                input_w - input_start_w,
                (ow_stop - ow_start - 1) * stride_w + kernel_w,
            )
            return (
                (n_start, 0, input_start_h, input_start_w),
                (
                    n_stop - n_start,
                    int(dict(operator.attributes).get("input_channels", tensor_shape[1])),
                    max(1, input_h_extent),
                    max(1, input_w_extent),
                ),
            )
        return None

    if op_type == "pool":
        if len(operator.inputs) != 1 or len(operator.outputs) != 1 or len(tensor_shape) != 4:
            return None
        if tuple(iteration) != ("d0", "d1", "d2", "d3"):
            return None
        bounds = tile.bound_map
        starts = tuple(bounds[name][0] for name in iteration)
        output_shape = tuple(bounds[name][1] - bounds[name][0] for name in iteration)
        if tensor_name == operator.outputs[0]:
            return starts, output_shape
        if tensor_name != operator.inputs[0]:
            return None
        window = tuple(int(value) for value in operator.attributes.get("window_dimensions", (1, 1, 1, 1)))
        stride = tuple(int(value) for value in operator.attributes.get("window_strides", (1, 1, 1, 1)))
        padding = tuple(int(value) for value in operator.attributes.get("padding", (0,) * 8))
        if len(window) != 4 or len(stride) != 4 or len(padding) != 8:
            return None
        input_start_h = max(0, starts[2] * stride[2] - padding[4])
        input_start_w = max(0, starts[3] * stride[3] - padding[6])
        input_h_extent = min(
            int(tensor_shape[2]) - input_start_h,
            (output_shape[2] - 1) * stride[2] + window[2],
        )
        input_w_extent = min(
            int(tensor_shape[3]) - input_start_w,
            (output_shape[3] - 1) * stride[3] + window[3],
        )
        return (
            (starts[0], starts[1], input_start_h, input_start_w),
            (output_shape[0], output_shape[1], max(1, input_h_extent), max(1, input_w_extent)),
        )

    if op_type in {"elementwise", "residual_add"}:
        output_starts, output_shape = _dimension_region(tile, iteration)
        if tensor_name == operator.outputs[0]:
            return output_starts, output_shape
        if len(tensor_shape) > len(output_shape):
            return None
        leading = len(output_shape) - len(tensor_shape)
        starts: list[int] = []
        shape: list[int] = []
        for axis, extent in enumerate(tensor_shape):
            output_axis = leading + axis
            if extent == 1:
                starts.append(0)
                shape.append(1)
            else:
                starts.append(output_starts[output_axis])
                shape.append(output_shape[output_axis])
        return tuple(starts), tuple(shape)

    dimensions = (*iteration, *reduction)
    if tensor_name in operator.outputs and op_type == "reduce":
        return _dimension_region(tile, iteration)
    if len(tensor_shape) == len(dimensions):
        return _dimension_region(tile, dimensions)
    if len(tensor_shape) == len(iteration):
        return _dimension_region(tile, iteration)
    if reduction and len(tensor_shape) == len(reduction):
        return _dimension_region(tile, reduction)
    return None


def _regions_overlap(
    left: tuple[tuple[int, ...], tuple[int, ...]],
    right: tuple[tuple[int, ...], tuple[int, ...]],
) -> bool:
    left_starts, left_shape = left
    right_starts, right_shape = right
    if len(left_starts) != len(right_starts):
        return False
    return all(
        left_start < right_start + right_extent
        and right_start < left_start + left_extent
        for left_start, left_extent, right_start, right_extent in zip(
            left_starts,
            left_shape,
            right_starts,
            right_shape,
        )
    )


def build_tile_graph(graph: OperatorGraph, schedule: ScheduleSpec) -> TileGraph:
    issues = schedule.validate(graph)
    if issues:
        raise ValueError("; ".join(issues))
    tiles: list[TileInstance] = []
    by_operator: dict[str, tuple[TileInstance, ...]] = {}
    for operator in graph.operators:
        current = enumerate_operator_tiles(operator, schedule.for_operator(operator.op_id))
        by_operator[operator.op_id] = current
        tiles.extend(current)
    operators = {operator.op_id: operator for operator in graph.operators}
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    tensor_shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
    dependencies: list[TileDependency] = []
    conservative_edge_count = 0
    avoided_all_to_all_dependencies = 0
    for edge in graph.edges:
        producer_tiles = by_operator[edge.producer]
        consumer_tiles = by_operator[edge.consumer]
        tensor_shape = tuple(tensors[edge.tensor].shape)
        def edge_regions(operator, tile):
            mapped = _tile_tensor_regions(
                operator,
                tile,
                tensor_shape,
                edge.tensor,
                tensor_shapes=tensor_shapes,
            )
            if mapped is not None:
                return tuple(mapped)
            fallback = _tile_tensor_region(
                operator,
                tile,
                tensor_shape,
                edge.tensor,
                tensor_shapes=tensor_shapes,
            )
            return (fallback,) if fallback is not None else None

        producer_regions = {
            tile.tile_id: edge_regions(operators[edge.producer], tile)
            for tile in producer_tiles
        }
        consumer_regions = {
            tile.tile_id: edge_regions(operators[edge.consumer], tile)
            for tile in consumer_tiles
        }
        full_count = len(producer_tiles) * len(consumer_tiles)
        unresolved = any(
            regions is None
            for regions in (*producer_regions.values(), *consumer_regions.values())
        )
        if unresolved:
            selected = tuple(
                (producer, consumer)
                for producer in producer_tiles
                for consumer in consumer_tiles
            )
            conservative_edge_count += 1
        else:
            # Empty region tuples mean that a tiled transform does not touch
            # this edge on that tile (for example, a concat tile entirely
            # belonging to another input).  Such a tile needs no data edge.
            selected = tuple(
                (producer, consumer)
                for producer in producer_tiles
                for consumer in consumer_tiles
                if producer_regions[producer.tile_id]
                and consumer_regions[consumer.tile_id]
                and any(
                    _regions_overlap(producer_region, consumer_region)
                    for producer_region in producer_regions[producer.tile_id]
                    for consumer_region in consumer_regions[consumer.tile_id]
                )
            )
            active_consumers = {
                tile.tile_id
                for tile in consumer_tiles
                if consumer_regions[tile.tile_id]
            }
            consumers_with_producer = {consumer.tile_id for _producer, consumer in selected}
            if consumers_with_producer != active_consumers:
                selected = tuple(
                    (producer, consumer)
                    for producer in producer_tiles
                    for consumer in consumer_tiles
                )
                conservative_edge_count += 1
        avoided_all_to_all_dependencies += full_count - len(selected)
        dependencies.extend(
            TileDependency(
                producer=producer.tile_id,
                consumer=consumer.tile_id,
                tensor=edge.tensor,
                kind="region_data",
                hazard_kind="RAW",
                producer_region=_bounding_region(producer_regions[producer.tile_id] or ()),
                consumer_region=_bounding_region(consumer_regions[consumer.tile_id] or ()),
                producer_regions=tuple(producer_regions[producer.tile_id] or ()),
                consumer_regions=tuple(consumer_regions[consumer.tile_id] or ()),
                condition="full_region_ready",
                provenance={
                    "source": "operator_graph_edge",
                    "producer_operator": edge.producer,
                    "consumer_operator": edge.consumer,
                    "dependency_model": "logical_tensor_region_v2",
                    "producer_regions": [
                        {"starts": list(region[0]), "shape": list(region[1])}
                        for region in (producer_regions[producer.tile_id] or ())
                    ],
                    "consumer_regions": [
                        {"starts": list(region[0]), "shape": list(region[1])}
                        for region in (consumer_regions[consumer.tile_id] or ())
                    ],
                },
            )
            for producer, consumer in selected
        )

    # GC owns reduction/state ordering.  Keeping these edges in the
    # TileGraph makes the software-scheduled output complete before FC lowers
    # it to TISA dependencies.  Independent rows remain free to overlap.
    for operator in graph.operators:
        if not operator.reduction_dims:
            continue
        operator_tiles = by_operator[operator.op_id]
        reduction_names = tuple(name for name, _ in operator.reduction_dims)
        iteration_names = tuple(name for name, _ in operator.iteration_dims)
        rows: dict[tuple[int, ...], list[TileInstance]] = {}
        for tile in operator_tiles:
            rows.setdefault(
                tuple(tile.bound_map[name][0] for name in iteration_names),
                [],
            ).append(tile)
        if (
            operator.normalized_type == "softmax"
            and operator.attributes.get("softmax_algorithm", "materialized") != "online"
        ):
            # Materialized row-wise Softmax finalizes max/sum inside its
            # composite payload. Online mode instead keeps this GC state chain.
            continue
        kind = (
            "accumulate"
            if operator.normalized_type in {"matmul", "batched_matmul", "gemv", "conv2d"}
            else "state"
        )
        for row_tiles in rows.values():
            ordered = sorted(
                row_tiles,
                key=lambda tile: tuple(tile.bound_map[name][0] for name in reduction_names),
            )
            dependencies.extend(
                TileDependency(
                    producer=producer.tile_id,
                    consumer=consumer.tile_id,
                    tensor=None,
                    kind=kind,
                    hazard_kind={
                        "state": "STATE",
                        "accumulate": "ACCUMULATE",
                    }[kind],
                    condition=(
                        "state_complete"
                        if kind == "state"
                        else "accumulate_ready"
                    ),
                    provenance={
                        "source": "reduction_order",
                        "operator_id": operator.op_id,
                        "reduction_dims": [name for name, _ in operator.reduction_dims],
                    },
                )
                for producer, consumer in zip(ordered, ordered[1:])
            )
    result = TileGraph(
        graph.graph_id,
        tuple(tiles),
        tuple(dependencies),
        attributes={
            "dependency_model": "logical_tensor_region_v2",
            "conservative_edge_count": conservative_edge_count,
            "avoided_all_to_all_dependencies": avoided_all_to_all_dependencies,
        },
    )
    issues = result.validate()
    if issues:
        raise ValueError("; ".join(issues))
    return result
