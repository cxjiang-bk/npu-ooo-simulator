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
    producer: str
    consumer: str
    tensor: str | None = None
    kind: str = "data"

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "consumer": self.consumer,
            "tensor": self.tensor,
            "kind": self.kind,
        }


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
            )
            if value not in {None, ""}
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


def _tile_tensor_region(
    operator: OperatorSpec,
    tile: TileInstance,
    tensor_shape: tuple[int, ...],
    tensor_name: str,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Project one operator tile onto a logical tensor region."""

    iteration = tuple(name for name, _extent in operator.iteration_dims)
    reduction = tuple(name for name, _extent in operator.reduction_dims)
    op_type = operator.normalized_type

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
    dependencies: list[TileDependency] = []
    conservative_edge_count = 0
    avoided_all_to_all_dependencies = 0
    for edge in graph.edges:
        producer_tiles = by_operator[edge.producer]
        consumer_tiles = by_operator[edge.consumer]
        tensor_shape = tuple(tensors[edge.tensor].shape)
        producer_regions = {
            tile.tile_id: _tile_tensor_region(
                operators[edge.producer], tile, tensor_shape, edge.tensor
            )
            for tile in producer_tiles
        }
        consumer_regions = {
            tile.tile_id: _tile_tensor_region(
                operators[edge.consumer], tile, tensor_shape, edge.tensor
            )
            for tile in consumer_tiles
        }
        full_count = len(producer_tiles) * len(consumer_tiles)
        if any(region is None for region in (*producer_regions.values(), *consumer_regions.values())):
            selected = tuple(
                (producer, consumer)
                for producer in producer_tiles
                for consumer in consumer_tiles
            )
            conservative_edge_count += 1
        else:
            selected = tuple(
                (producer, consumer)
                for producer in producer_tiles
                for consumer in consumer_tiles
                if _regions_overlap(
                    producer_regions[producer.tile_id],  # type: ignore[arg-type]
                    consumer_regions[consumer.tile_id],  # type: ignore[arg-type]
                )
            )
            consumers_with_producer = {consumer.tile_id for _producer, consumer in selected}
            if consumers_with_producer != {tile.tile_id for tile in consumer_tiles}:
                selected = tuple(
                    (producer, consumer)
                    for producer in producer_tiles
                    for consumer in consumer_tiles
                )
                conservative_edge_count += 1
        avoided_all_to_all_dependencies += full_count - len(selected)
        dependencies.extend(
            TileDependency(producer.tile_id, consumer.tile_id, edge.tensor, "region_data")
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
            if operator.normalized_type in {"matmul", "batched_matmul", "gemv"}
            else "state"
        )
        for row_tiles in rows.values():
            ordered = sorted(
                row_tiles,
                key=lambda tile: tuple(tile.bound_map[name][0] for name in reduction_names),
            )
            dependencies.extend(
                TileDependency(
                    producer.tile_id,
                    consumer.tile_id,
                    None,
                    kind,
                )
                for producer, consumer in zip(ordered, ordered[1:])
            )
    result = TileGraph(
        graph.graph_id,
        tuple(tiles),
        tuple(dependencies),
        attributes={
            "dependency_model": "logical_tensor_region_v1",
            "conservative_edge_count": conservative_edge_count,
            "avoided_all_to_all_dependencies": avoided_all_to_all_dependencies,
        },
    )
    issues = result.validate()
    if issues:
        raise ValueError("; ".join(issues))
    return result
