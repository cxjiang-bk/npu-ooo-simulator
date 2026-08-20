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
        tiles.append(
            TileInstance(
                tile_id=f"{operator.op_id}.t{ordinal:04d}",
                operator_id=operator.op_id,
                ordinal=ordinal,
                coordinates=coordinates,
                bounds=bounds,
                stage_id=schedule.stage_id,
            )
        )
    return tuple(tiles)


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
    dependencies: list[TileDependency] = []
    for edge in graph.edges:
        for producer in by_operator[edge.producer]:
            for consumer in by_operator[edge.consumer]:
                dependencies.append(TileDependency(producer.tile_id, consumer.tile_id, edge.tensor))
    result = TileGraph(graph.graph_id, tuple(tiles), tuple(dependencies))
    issues = result.validate()
    if issues:
        raise ValueError("; ".join(issues))
    return result
