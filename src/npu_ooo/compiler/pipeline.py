from __future__ import annotations

"""The first unified graph-to-backend compiler pipeline.

This module intentionally keeps passes small and inspectable.  It is the
bridge from the canonical frontend graph to today's analytical backend; later
passes can replace the default schedule planner or backend code generator
without changing the frontend contract.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.frontend import FrontendImport
from npu_ooo.ir import (
    AccessType,
    BackendArtifact,
    BufferRegion,
    ExecutionGraph,
    OperatorGraph,
    ScheduleSpec,
    TISADependency,
    TISAInstruction,
    TISAOperand,
    TISAProgram,
    TileGraph,
    TileMem,
    UnitMap,
    default_mixed_schedule,
)
from npu_ooo.ir.model import ModelInstance
from npu_ooo.lowering import LoweringRegistry, default_lowering_registry, lower_mixed_graph


@dataclass(frozen=True)
class CompilerDiagnostic:
    """A stable pass diagnostic suitable for CLI and manifest output."""

    level: str
    pass_name: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "pass": self.pass_name, "message": self.message}


@dataclass(frozen=True)
class CompiledArtifact:
    """All inspectable artifacts produced by one compiler invocation."""

    frontend: FrontendImport
    graph: OperatorGraph
    schedule: ScheduleSpec
    tile_graph: TileGraph
    tisa_program: TISAProgram
    backend_artifact: BackendArtifact
    diagnostics: tuple[CompilerDiagnostic, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        issues.extend(self.frontend.validate())
        issues.extend(self.graph.validate())
        issues.extend(self.schedule.validate(self.graph))
        issues.extend(self.tile_graph.validate())
        issues.extend(self.tisa_program.validate())
        issues.extend(self.backend_artifact.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frontend": self.frontend.to_dict(),
            "graph": self.graph.to_dict(),
            "schedule": self.schedule.to_dict(),
            "tile_graph": self.tile_graph.to_dict(),
            "tisa_program": self.tisa_program.to_dict(),
            "backend_artifact": self.backend_artifact.to_dict(),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "attributes": dict(self.attributes),
        }


def _unit_for_operator(op_type: str) -> UnitMap:
    if op_type in {"matmul", "batched_matmul", "gemv"}:
        return UnitMap("tensor", affinity="matrix")
    if op_type in {"softmax", "reduce", "layernorm", "rmsnorm"}:
        return UnitMap("vector", affinity="reduction")
    if op_type in {"elementwise", "residual_add"}:
        return UnitMap("vector", affinity="alu")
    if op_type in {"reshape", "transpose"}:
        return UnitMap("dma", affinity="view")
    return UnitMap("scalar")


def _unit_for_resource(resource: str) -> UnitMap:
    """Map a backend primitive resource to its TISA-visible EU class."""

    normalized = resource.lower()
    if normalized in {"dma", "de", "copy"}:
        return UnitMap("dma", affinity="data")
    if normalized in {"mxu", "me", "tensor", "matrix"}:
        return UnitMap("tensor", affinity="matrix")
    if normalized in {"aru", "ve", "vector", "vu"}:
        return UnitMap("vector", affinity="vector")
    return UnitMap(normalized)


def _semantic_group_type(operator_type: str, tasks: tuple[Any, ...]) -> str:
    primitives = {str(task.primitive) for task in tasks}
    if primitives <= {"load", "load_transpose"}:
        return "load"
    if primitives <= {"store"}:
        return "store"
    return operator_type


def _region_key(region: BufferRegion) -> tuple[Any, ...]:
    return (
        region.tensor,
        region.memory,
        region.shape,
        region.starts,
        region.dtype,
        region.normalized_access,
        region.offset_bytes,
        region.size_bytes,
        region.layout,
    )


def _operand_from_region(region: BufferRegion, index: int) -> TISAOperand:
    return TISAOperand(
        name=f"{region.tensor}:{region.normalized_access}:{index}",
        tile_shape=tuple(region.shape),
        tile_mem=TileMem(
            base=region.tensor,
            scope=region.memory,
            tensor=region.tensor,
            offset_bytes=region.offset_bytes,
            size_bytes=region.size_bytes,
        ),
        access_type=region.access,
    )


def _dependency_kind(source: Any, target: Any) -> str:
    source_writes = tuple(source.writes)
    source_reads = tuple(source.reads)
    target_writes = tuple(target.writes)
    target_reads = tuple(target.reads)
    if source_writes and target_reads:
        return "RAW"
    if source_writes and target_writes:
        return "WAW"
    if source_reads and target_writes:
        return "WAR"
    return "RAW"


def _build_tisa_program(
    graph: OperatorGraph,
    execution_graph: ExecutionGraph,
    *,
    program_id: str,
) -> tuple[TISAProgram, Mapping[str, tuple[str, ...]]]:
    operators = {operator.op_id: operator for operator in graph.operators}
    by_tile: dict[str, list[Any]] = {}
    tile_order: dict[str, int] = {}
    for task in execution_graph.tasks:
        by_tile.setdefault(task.tile_id, []).append(task)
        tile_order[task.tile_id] = min(tile_order.get(task.tile_id, task.program_order), task.program_order)
    # A semantic tile may contain a DMA prologue, one compute payload and a
    # DMA epilogue.  Split these into scheduler-visible instructions whenever
    # the EU resource changes, while keeping the same tile provenance.
    groups_by_tile: dict[str, list[tuple[Any, ...]]] = {}
    task_to_tisa: dict[str, str] = {}
    for tile_id, tasks in by_tile.items():
        ordered = sorted(tasks, key=lambda task: task.program_order)
        groups: list[list[Any]] = []
        for task in ordered:
            if not groups or groups[-1][-1].resource != task.resource:
                groups.append([task])
            else:
                groups[-1].append(task)
        frozen_groups = [tuple(group) for group in groups]
        groups_by_tile[tile_id] = frozen_groups
        for group_index, group in enumerate(frozen_groups):
            group_id = f"tisa.{tile_id}.u{group_index:02d}"
            for task in group:
                task_to_tisa[task.task_id] = group_id
    instructions: list[TISAInstruction] = []
    payloads: dict[str, tuple[str, ...]] = {}
    for tile_id, groups in sorted(groups_by_tile.items(), key=lambda item: tile_order[item[0]]):
        for group_index, tasks in enumerate(groups):
            first = tasks[0]
            operator = operators[first.operator_id]
            regions: list[BufferRegion] = []
            for task in tasks:
                for region in (*task.reads, *task.writes):
                    if _region_key(region) not in {_region_key(item) for item in regions}:
                        regions.append(region)
            dependencies: dict[str, str] = {}
            for task in tasks:
                for predecessor_id in task.predecessors:
                    predecessor = execution_graph.task(predecessor_id)
                    source_tisa = task_to_tisa[predecessor_id]
                    tisa_id = f"tisa.{tile_id}.u{group_index:02d}"
                    if source_tisa == tisa_id:
                        continue
                    kind = _dependency_kind(predecessor, task)
                    previous = dependencies.get(source_tisa)
                    # RAW is the strongest readiness condition when several
                    # primitive edges connect the same pair of instructions.
                    if previous is None or (previous != "RAW" and kind == "RAW"):
                        dependencies[source_tisa] = kind
            tisa_id = f"tisa.{tile_id}.u{group_index:02d}"
            instruction = TISAInstruction(
                tisa_id=tisa_id,
                tile_id=tile_id,
                operator_id=first.operator_id,
                op_type=_semantic_group_type(operator.normalized_type, tasks),
                operands=tuple(_operand_from_region(region, index) for index, region in enumerate(regions)),
                unit_map=_unit_for_resource(first.resource),
                dependencies=tuple(
                    TISADependency(source=source, kind=kind)
                    for source, kind in sorted(dependencies.items())
                ),
                attributes={
                    "program_order": first.program_order,
                    "primitive_count": len(tasks),
                    "primitive_resources": sorted({task.resource for task in tasks}),
                    "semantic_boundary": "tile",
                    "semantic_tile_id": tile_id,
                    "tile_group_index": group_index,
                },
                payload_ref=f"payload:{tisa_id}",
            )
            instructions.append(instruction)
            payloads[tisa_id] = tuple(task.task_id for task in tasks)
    program = TISAProgram(
        program_id=program_id,
        instructions=tuple(instructions),
        attributes={"source": "canonical-operator-graph", "scheduler_granularity": "tisa_tile"},
    )
    issues = program.validate()
    if issues:
        raise ValueError("TISA program construction failed: " + "; ".join(issues))
    return program, payloads


def compile_operator_graph(
    graph: OperatorGraph,
    machine: MachineConfig,
    *,
    shape_environment: Mapping[str, int] | None = None,
    model_id: str | None = None,
    frontend: FrontendImport | None = None,
    tile_size: int = 32,
    registry: LoweringRegistry | None = None,
) -> CompiledArtifact:
    """Run normalization, default tiling, semantic lowering and backend codegen."""

    graph = graph.resolve(dict(shape_environment or {})) if shape_environment else graph
    graph_issues = graph.validate()
    if graph_issues:
        raise ValueError("compiler input graph is invalid: " + "; ".join(graph_issues))
    machine_issues = machine.validate()
    if machine_issues:
        raise ValueError("compiler machine is invalid: " + "; ".join(machine_issues))
    if frontend is None:
        from npu_ooo.frontend import import_operator_graph

        frontend = import_operator_graph(graph, model_id=model_id)
    schedule = default_mixed_schedule(graph, tile_size=tile_size)
    lowering = lower_mixed_graph(graph, schedule, machine, registry=registry or default_lowering_registry())
    program, payloads = _build_tisa_program(
        graph,
        lowering.execution_graph,
        program_id=f"{graph.graph_id}.tisa",
    )
    artifact = BackendArtifact(
        artifact_id=f"{graph.graph_id}.analytical",
        program=program,
        execution_graph=lowering.execution_graph,
        payloads=payloads,
        backend="analytical",
        attributes={"calibration_status": "analytical", "lowering_statistics": lowering.statistics},
    )
    result = CompiledArtifact(
        frontend=frontend,
        graph=graph,
        schedule=schedule,
        tile_graph=lowering.tile_graph,
        tisa_program=program,
        backend_artifact=artifact,
        diagnostics=(
            CompilerDiagnostic("info", "frontend", f"imported via {frontend.frontend}"),
            CompilerDiagnostic("info", "tiling", f"generated {len(lowering.tile_graph.tiles)} tile instances"),
            CompilerDiagnostic("info", "tisa_codegen", f"generated {len(program.instructions)} TISA instructions"),
        ),
        attributes={
            "compiler_pipeline": "frontend->canonical->schedule->tile->tisa->analytical-backend",
            "model_id": model_id or graph.graph_id,
        },
    )
    issues = result.validate()
    if issues:
        raise ValueError("compiled artifact validation failed: " + "; ".join(issues))
    return result


def compile_frontend_import(
    imported: FrontendImport,
    machine: MachineConfig,
    *,
    tile_size: int = 32,
    registry: LoweringRegistry | None = None,
) -> CompiledArtifact:
    issues = imported.validate()
    if issues:
        raise ValueError("frontend import is invalid: " + "; ".join(issues))
    graph = imported.graph.resolve(dict(imported.shape_environment)) if imported.shape_environment else imported.graph
    return compile_operator_graph(
        graph,
        machine,
        frontend=imported,
        tile_size=tile_size,
        registry=registry,
    )


def compile_model_instance(
    instance: ModelInstance,
    machine: MachineConfig,
    *,
    tile_size: int = 32,
    registry: LoweringRegistry | None = None,
) -> CompiledArtifact:
    """Compile an instantiated Model IR through the same frontend boundary."""

    from npu_ooo.frontend import import_operator_graph

    imported = import_operator_graph(
        instance.graph,
        model_id=instance.model_id,
        variant=instance.model_variant,
    )
    imported = FrontendImport(
        graph=imported.graph,
        model_id=imported.model_id,
        variant=imported.variant,
        shape_environment={},
        frontend=imported.frontend,
        provenance={
            **dict(imported.provenance),
            **dict(instance.provenance),
            "case_id": instance.case_id,
            "template_id": instance.template_id,
        },
    )
    return compile_frontend_import(
        imported,
        machine,
        tile_size=tile_size,
        registry=registry,
    )
