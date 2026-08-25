from __future__ import annotations

"""Paper-aligned Graph Compiler (GC) boundary.

The paper uses an MLIR implementation.  This project keeps the same semantic
boundary with Python IR objects so that every intermediate result remains easy
to inspect and validate.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import OperatorGraph, ScheduleSpec, TileGraph, build_tile_graph

from .passes import PassDiagnostic, PassSnapshot, default_pass_manager
from .planner import default_schedule_planner


@dataclass(frozen=True)
class GCArtifact:
    """Software-scheduled TileGraph emitted by the Graph Compiler."""

    graph: OperatorGraph
    schedule: ScheduleSpec
    tile_graph: TileGraph
    diagnostics: tuple[PassDiagnostic, ...] = ()
    pass_dumps: tuple[PassSnapshot, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, machine: MachineConfig | None = None) -> tuple[str, ...]:
        issues = list(self.graph.validate())
        issues.extend(self.schedule.validate(self.graph))
        issues.extend(self.tile_graph.validate())
        if machine is not None:
            issues.extend(machine.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_stage": "GC",
            "implementation": "python-semantic-proxy",
            "graph": self.graph.to_dict(),
            "schedule": self.schedule.to_dict(),
            "tile_graph": self.tile_graph.to_dict(),
            "diagnostics": [
                {"level": item.level, "pass": item.pass_name, "message": item.message}
                for item in self.diagnostics
            ],
            "pass_dumps": [item.to_dict() for item in self.pass_dumps],
            "attributes": dict(self.attributes),
        }


class GraphCompiler:
    """Run canonicalization, recovery, tiling and dependency construction."""

    name = "graph-compiler-python-v1"

    def compile(
        self,
        graph: OperatorGraph,
        machine: MachineConfig,
        *,
        tile_size: int = 32,
    ) -> GCArtifact:
        graph_issues = graph.validate()
        machine_issues = machine.validate()
        if graph_issues or machine_issues:
            raise ValueError("; ".join((*graph_issues, *machine_issues)))

        softmax_algorithm = machine.attributes.get("softmax_algorithm")
        if softmax_algorithm is not None and softmax_algorithm not in {"materialized", "online"}:
            raise ValueError(
                "machine attribute 'softmax_algorithm' must be 'materialized' or 'online'"
            )
        pass_result = default_pass_manager().run(graph)
        canonical_graph = pass_result.graph
        if softmax_algorithm is not None:
            canonical_graph = replace(
                canonical_graph,
                operators=tuple(
                    replace(
                        operator,
                        attributes={
                            **dict(operator.attributes),
                            "softmax_algorithm": softmax_algorithm,
                        },
                    )
                    if operator.normalized_type == "softmax"
                    else operator
                    for operator in canonical_graph.operators
                ),
            )
        schedule = default_schedule_planner().plan(
            canonical_graph,
            tile_size=tile_size,
            machine=machine,
        )
        tile_graph = build_tile_graph(canonical_graph, schedule)
        schedule = replace(
            schedule,
            attributes={
                **dict(schedule.attributes),
                "paper_stage": "GC",
                "software_scheduled": True,
            },
        )
        tile_graph = replace(
            tile_graph,
            attributes={
                **dict(tile_graph.attributes),
                "paper_stage": "GC",
                "tile_graph_kind": "software_scheduled_semantic_tile_graph",
                "dependency_kinds": ["region_data", "state", "accumulate", "buffer_reuse"],
                "initial_order": "tile_graph_topological_order",
                "residency_plan": "schedule.residency",
                "ping_pong_plan": "schedule.operator_schedules[].attributes.ping_pong",
                "tile_to_core_assignment": "compiler_attributes_or_machine_default",
            },
        )
        artifact = GCArtifact(
            graph=canonical_graph,
            schedule=schedule,
            tile_graph=tile_graph,
            diagnostics=pass_result.diagnostics,
            pass_dumps=pass_result.snapshots,
            attributes={
                "paper_stage": "GC",
                "implementation": "python-semantic-proxy",
                "compiler": self.name,
                "input_contract": "verified StableHLO projection",
                "output_contract": "software-scheduled semantic TileGraph",
                "fusion": "graph-pattern recovery",
                "tiling": "automatic shape-aware baseline",
                "locality": "schedule residency metadata",
                "dependency_model": tile_graph.attributes.get("dependency_model"),
                "pass_count": len(pass_result.snapshots),
            },
        )
        issues = artifact.validate(machine)
        if issues:
            raise ValueError("Graph Compiler produced an invalid artifact: " + "; ".join(issues))
        return artifact


def default_graph_compiler() -> GraphCompiler:
    return GraphCompiler()


__all__ = ["GCArtifact", "GraphCompiler", "default_graph_compiler"]
