from __future__ import annotations

"""The first unified graph-to-backend compiler pipeline.

This module intentionally keeps passes small and inspectable.  It is the
bridge from the canonical frontend graph to today's analytical backend; later
passes can replace the default schedule planner or backend code generator
without changing the frontend contract.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from npu_ooo.arch import MachineConfig
from npu_ooo.frontend import FrontendImport, OfficialStableHLOModule, StableHLOModule
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
)
from npu_ooo.ir.model import ModelInstance
from npu_ooo.lowering import LoweringRegistry, default_lowering_registry, lower_mixed_graph

from .passes import default_pass_manager
from .planner import default_schedule_planner


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
    stablehlo: StableHLOModule | OfficialStableHLOModule | None = None
    source_frontend: FrontendImport | None = None
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
        if self.stablehlo is not None:
            issues.extend(self.stablehlo.validate())
        if self.source_frontend is not None:
            issues.extend(self.source_frontend.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frontend": self.frontend.to_dict(),
            "graph": self.graph.to_dict(),
            "schedule": self.schedule.to_dict(),
            "tile_graph": self.tile_graph.to_dict(),
            "tisa_program": self.tisa_program.to_dict(),
            "backend_artifact": self.backend_artifact.to_dict(),
            "stablehlo": self.stablehlo.to_dict() if self.stablehlo is not None else None,
            "source_frontend": self.source_frontend.to_dict() if self.source_frontend is not None else None,
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
    if "load_transpose" in primitives:
        return "load_transpose"
    if primitives <= {"load", "load_transpose"}:
        return "load"
    if primitives <= {"store"}:
        return "store"
    return operator_type


def _task_group_key(task: Any) -> tuple[str, str]:
    primitive = str(task.primitive)
    if primitive in {"load", "load_transpose", "store"}:
        return str(task.resource), primitive
    return str(task.resource), "compute"


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
            if not groups or _task_group_key(groups[-1][-1]) != _task_group_key(task):
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
    pass_result = default_pass_manager().run(graph)
    graph = pass_result.graph
    frontend = replace(
        frontend,
        graph=graph,
        provenance={
            **dict(frontend.provenance),
            "compiler_passes": [item.pass_name for item in pass_result.diagnostics],
        },
    )
    schedule = default_schedule_planner().plan(graph, tile_size=tile_size)
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
            CompilerDiagnostic(
                "info",
                "frontend",
                "imported via "
                + (
                    frontend.frontend.value
                    if hasattr(frontend.frontend, "value")
                    else str(frontend.frontend)
                ),
            ),
            *(
                CompilerDiagnostic(item.level, item.pass_name, item.message)
                for item in pass_result.diagnostics
            ),
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


def compile_frontend_import_through_stablehlo(
    imported: FrontendImport,
    machine: MachineConfig,
    *,
    tile_size: int = 32,
    registry: LoweringRegistry | None = None,
    stablehlo_variant: str = "stablehlo-generated-v0",
    stablehlo_backend: str = "official",
) -> CompiledArtifact:
    """Run the paper-shaped ``Frontend -> StableHLO -> TISA`` route.

    ``official`` emits valid StableHLO and requires the OpenXLA Python wheel
    for MLIR parse/verify. ``textual`` retains the old dependency-light subset
    for regression tests. ``auto`` uses official bindings when available and
    records an explicit fallback otherwise.
    """

    from dataclasses import replace
    from npu_ooo.frontend import (
        OfficialStableHLOAdapter,
        OfficialStableHLOGenerator,
        StableHLOAdapter,
        StableHLOGenerator,
        official_stablehlo_available,
    )

    source_issues = imported.validate()
    if source_issues:
        raise ValueError("frontend import is invalid: " + "; ".join(source_issues))
    if stablehlo_backend not in {"official", "textual", "auto"}:
        raise ValueError("stablehlo_backend must be one of: official, textual, auto")
    selected_backend = stablehlo_backend
    fallback_reason: str | None = None
    if selected_backend == "auto":
        if official_stablehlo_available():
            selected_backend = "official"
        else:
            selected_backend = "textual"
            fallback_reason = "official StableHLO bindings are unavailable"
    if selected_backend == "official":
        variant = "stablehlo-official-v1" if stablehlo_variant == "stablehlo-generated-v0" else stablehlo_variant
        stablehlo = OfficialStableHLOGenerator(variant=variant).generate(imported)
        stable_import = OfficialStableHLOAdapter.import_text(
            stablehlo.text,
            model_id=imported.model_id,
            variant=stablehlo.variant,
            shape_environment=imported.shape_environment,
        )
    else:
        stablehlo = StableHLOGenerator(variant=stablehlo_variant).generate(imported)
        stable_import = StableHLOAdapter.from_text(
            stablehlo.text,
            model_id=imported.model_id,
            variant=stablehlo.variant,
            shape_environment=imported.shape_environment,
        )
    stable_import = replace(
        stable_import,
        family=imported.family,
        provenance={
            **dict(stable_import.provenance),
            "source_frontend": (
                imported.frontend.value
                if hasattr(imported.frontend, "value")
                else str(imported.frontend)
            ),
            "generated_stablehlo_variant": stablehlo.variant,
            "generated_stablehlo_provenance": dict(stablehlo.provenance),
        },
    )
    compiled = compile_frontend_import(
        stable_import,
        machine,
        tile_size=tile_size,
        registry=registry,
    )
    return replace(
        compiled,
        stablehlo=stablehlo,
        source_frontend=imported,
        attributes={
            **dict(compiled.attributes),
            "frontend_path": f"torch_export->stablehlo_{selected_backend}->stablehlo_import->canonical",
            "stablehlo_variant": stablehlo.variant,
            "stablehlo_backend": selected_backend,
            "stablehlo_exporter": "project",
            "stablehlo_exporter_version": None,
            "stablehlo_verified": bool(getattr(stablehlo, "verified", False)),
            "stablehlo_producer": getattr(stablehlo, "producer", "project-textual-generator"),
            "stablehlo_verifier": getattr(stablehlo, "verifier", None),
            "stablehlo_version": getattr(stablehlo, "stablehlo_version", None),
            "stablehlo_fallback": fallback_reason is not None,
            "stablehlo_fallback_reason": fallback_reason,
        },
    )


def compile_torch_exported_program(
    exported_program: Any,
    machine: MachineConfig,
    *,
    model_id: str = "torch_export_model",
    variant: str = "torch-export-v0",
    shape_environment: Mapping[str, int] | None = None,
    tile_size: int = 32,
    registry: LoweringRegistry | None = None,
) -> CompiledArtifact:
    """Compile a ``torch.export`` program without exposing FX downstream."""

    from npu_ooo.frontend import TorchExportAdapter

    imported = TorchExportAdapter.from_exported_program(
        exported_program,
        model_id=model_id,
        variant=variant,
        shape_environment=shape_environment,
    )
    return compile_frontend_import(
        imported,
        machine,
        tile_size=tile_size,
        registry=registry,
    )


def compile_torch_exported_program_through_stablehlo(
    exported_program: Any,
    machine: MachineConfig,
    *,
    model_id: str = "torch_export_model",
    variant: str = "torch-export-v0",
    shape_environment: Mapping[str, int] | None = None,
    tile_size: int = 32,
    registry: LoweringRegistry | None = None,
    stablehlo_variant: str = "stablehlo-generated-v0",
    stablehlo_backend: str = "official",
    stablehlo_exporter: str = "project",
) -> CompiledArtifact:
    from npu_ooo.frontend import TorchExportAdapter, TorchXLAStableHLOExporter

    imported = TorchExportAdapter.from_exported_program(
        exported_program,
        model_id=model_id,
        variant=variant,
        shape_environment=shape_environment,
    )
    if stablehlo_exporter == "project":
        return compile_frontend_import_through_stablehlo(
            imported,
            machine,
            tile_size=tile_size,
            registry=registry,
            stablehlo_variant=stablehlo_variant,
            stablehlo_backend=stablehlo_backend,
        )
    if stablehlo_exporter != "torch-xla":
        raise ValueError("stablehlo_exporter must be one of: project, torch-xla")
    if stablehlo_backend != "official":
        raise ValueError("torch-xla exporter requires stablehlo_backend='official'")

    xla_variant = (
        "stablehlo-torch-xla-v1"
        if stablehlo_variant == "stablehlo-generated-v0"
        else stablehlo_variant
    )
    stablehlo = TorchXLAStableHLOExporter.export_program(
        exported_program,
        model_id=model_id,
        variant=xla_variant,
    )
    compiled = compile_stablehlo_text(
        stablehlo.text,
        machine,
        model_id=model_id,
        variant=stablehlo.variant,
        shape_environment=shape_environment,
        tile_size=tile_size,
        registry=registry,
        stablehlo_backend="official",
    )
    stable_frontend = replace(
        compiled.frontend,
        family=imported.family,
        provenance={
            **dict(compiled.frontend.provenance),
            "source_frontend": imported.frontend.value
            if hasattr(imported.frontend, "value")
            else str(imported.frontend),
            "stablehlo_exporter": "torch-xla",
            "stablehlo_exporter_version": stablehlo.provenance.get("exporter_version"),
        },
    )
    return replace(
        compiled,
        frontend=stable_frontend,
        stablehlo=stablehlo,
        source_frontend=imported,
        attributes={
            **dict(compiled.attributes),
            "frontend_path": "torch_export->torch_xla->official_stablehlo->canonical",
            "stablehlo_variant": stablehlo.variant,
            "stablehlo_backend": "official",
            "stablehlo_exporter": "torch-xla",
            "stablehlo_exporter_version": stablehlo.provenance.get("exporter_version"),
            "stablehlo_verified": True,
            "stablehlo_producer": stablehlo.producer,
            "stablehlo_verifier": stablehlo.verifier,
            "stablehlo_version": stablehlo.stablehlo_version,
            "stablehlo_fallback": False,
            "stablehlo_fallback_reason": None,
        },
    )


def compile_torch_module(
    module: Any,
    args: Sequence[Any],
    machine: MachineConfig,
    *,
    kwargs: Mapping[str, Any] | None = None,
    dynamic_shapes: Mapping[str, Any] | None = None,
    model_id: str = "torch_export_model",
    variant: str = "torch-export-v0",
    shape_environment: Mapping[str, int] | None = None,
    tile_size: int = 32,
    registry: LoweringRegistry | None = None,
) -> CompiledArtifact:
    """Capture and compile a real PyTorch module through the frontend boundary."""

    from npu_ooo.frontend import TorchExportAdapter

    imported = TorchExportAdapter.export_module(
        module,
        args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        model_id=model_id,
        variant=variant,
        shape_environment=shape_environment,
    )
    return compile_frontend_import(
        imported,
        machine,
        tile_size=tile_size,
        registry=registry,
    )


def compile_torch_module_through_stablehlo(
    module: Any,
    args: Sequence[Any],
    machine: MachineConfig,
    *,
    kwargs: Mapping[str, Any] | None = None,
    dynamic_shapes: Mapping[str, Any] | None = None,
    model_id: str = "torch_export_model",
    variant: str = "torch-export-v0",
    shape_environment: Mapping[str, int] | None = None,
    tile_size: int = 32,
    registry: LoweringRegistry | None = None,
    stablehlo_variant: str = "stablehlo-generated-v0",
    stablehlo_backend: str = "official",
    stablehlo_exporter: str = "project",
) -> CompiledArtifact:
    from npu_ooo.frontend import TorchExportAdapter

    exported_program = TorchExportAdapter.capture_module(
        module,
        args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    return compile_torch_exported_program_through_stablehlo(
        exported_program,
        machine,
        model_id=model_id,
        variant=variant,
        shape_environment=shape_environment,
        tile_size=tile_size,
        registry=registry,
        stablehlo_variant=stablehlo_variant,
        stablehlo_backend=stablehlo_backend,
        stablehlo_exporter=stablehlo_exporter,
    )


def compile_stablehlo_text(
    text: str,
    machine: MachineConfig,
    *,
    model_id: str = "stablehlo_model",
    variant: str = "stablehlo-v0",
    shape_environment: Mapping[str, int] | None = None,
    tile_size: int = 32,
    registry: LoweringRegistry | None = None,
    stablehlo_backend: str = "official",
) -> CompiledArtifact:
    """Compile StableHLO, using official MLIR verification by default."""

    from npu_ooo.frontend import OfficialStableHLOAdapter, StableHLOAdapter, official_stablehlo_available

    selected = stablehlo_backend
    fallback_reason: str | None = None
    if selected == "auto":
        if official_stablehlo_available():
            selected = "official"
        else:
            selected = "textual"
            fallback_reason = "official StableHLO bindings are unavailable"
    if selected == "official":
        imported = OfficialStableHLOAdapter.import_text(
            text,
            model_id=model_id,
            variant=variant,
            shape_environment=shape_environment,
        )
    elif selected == "textual":
        imported = StableHLOAdapter.from_text(
            text,
            model_id=model_id,
            variant=variant,
            shape_environment=shape_environment,
        )
    else:
        raise ValueError("stablehlo_backend must be one of: official, textual, auto")
    compiled = compile_frontend_import(
        imported,
        machine,
        tile_size=tile_size,
        registry=registry,
    )
    return replace(
        compiled,
        attributes={
            **dict(compiled.attributes),
            "frontend_path": f"stablehlo_{selected}->canonical",
            "stablehlo_variant": variant,
            "stablehlo_backend": selected,
            "stablehlo_exporter": None,
            "stablehlo_exporter_version": None,
            "stablehlo_verified": selected == "official",
            "stablehlo_producer": "external-stablehlo",
            "stablehlo_verifier": "official-stablehlo-mlir" if selected == "official" else None,
            "stablehlo_version": imported.provenance.get("stablehlo_version"),
            "stablehlo_fallback": fallback_reason is not None,
            "stablehlo_fallback_reason": fallback_reason,
        },
    )


def compile_stablehlo_file(
    path: str,
    machine: MachineConfig,
    *,
    model_id: str = "stablehlo_model",
    variant: str = "stablehlo-v0",
    shape_environment: Mapping[str, int] | None = None,
    tile_size: int = 32,
    registry: LoweringRegistry | None = None,
    stablehlo_backend: str = "official",
) -> CompiledArtifact:
    """Compile a StableHLO MLIR file with the selected binding backend."""

    from pathlib import Path

    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"StableHLO file does not exist: {path}") from exc
    return compile_stablehlo_text(
        text,
        machine,
        model_id=model_id,
        variant=variant,
        shape_environment=shape_environment,
        tile_size=tile_size,
        registry=registry,
        stablehlo_backend=stablehlo_backend,
    )


def compile_stablehlo_module(
    module: Any,
    machine: MachineConfig,
    *,
    model_id: str = "stablehlo_model",
    variant: str = "stablehlo-v0",
    shape_environment: Mapping[str, int] | None = None,
    tile_size: int = 32,
    registry: LoweringRegistry | None = None,
    stablehlo_backend: str = "official",
) -> CompiledArtifact:
    """Compile an MLIR module-like object by reading its assembly."""

    operation = getattr(module, "operation", module)
    text: str | None = module if isinstance(module, str) else None
    if text is None:
        for method_name in ("get_asm", "to_string", "as_text"):
            method = getattr(operation, method_name, None)
            if callable(method):
                text = str(method())
                break
    if text is None:
        text = str(module)
    return compile_stablehlo_text(
        text,
        machine,
        model_id=model_id,
        variant=variant,
        shape_environment=shape_environment,
        tile_size=tile_size,
        registry=registry,
        stablehlo_backend=stablehlo_backend,
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
