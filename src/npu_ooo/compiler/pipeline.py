from __future__ import annotations

"""PyTorch-to-TISA compiler pipeline.

There is one production frontend route:

``PyTorch -> torch.export -> torch-xla -> official StableHLO -> TISA``.

The public entry point keeps that route linear and visible.  The canonical
graph compiler remains a separate reusable phase because backend and pass
unit tests need to start after the framework boundary.
"""

from dataclasses import dataclass, field, replace
import re
from typing import Any, Mapping, Sequence

from npu_ooo.arch import MachineConfig
from npu_ooo.backend import CodegenBackend, default_codegen_backend_registry
from npu_ooo.frontend import FrontendImport, FrontendImportError, OfficialStableHLOModule
from npu_ooo.ir import (
    BackendArtifact,
    OperatorGraph,
    ScheduleSpec,
    TISAProgram,
    TileGraph,
)
from npu_ooo.lowering import LoweringRegistry, default_lowering_registry

from .fusion_compiler import TISADialectProgram, default_fusion_compiler
from .graph_compiler import GCArtifact, default_graph_compiler
from .statistics import build_compile_statistics
from .tisa_generator import default_tisa_generator


_KNOWN_DTYPES = frozenset(
    {
        "bool",
        "pred",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "fp16",
        "f16",
        "bfloat16",
        "bf16",
        "float32",
        "fp32",
        "f32",
        "float64",
        "fp64",
        "f64",
    }
)


def _normalize_dtype(dtype: Any) -> str:
    return str(dtype).lower().replace("torch.", "")


def _dtype_compatibility(graph: OperatorGraph, machine: MachineConfig) -> dict[str, Any]:
    """Validate graph dtypes against an optional machine capability contract."""

    attributes = machine.attributes
    policy = str(attributes.get("dtype_policy", "strict")).lower()
    if policy not in {"strict", "fallback"}:
        raise ValueError("machine attribute 'dtype_policy' must be 'strict' or 'fallback'")
    graph_dtypes = tuple(sorted({_normalize_dtype(tensor.dtype) for tensor in graph.tensors}))
    unknown = tuple(dtype for dtype in graph_dtypes if dtype not in _KNOWN_DTYPES)
    if unknown:
        raise ValueError(
            "graph contains unknown dtype(s): "
            + ", ".join(unknown)
            + "; register byte width and backend capability before compiling"
        )
    declared = attributes.get("supported_dtypes")
    if declared is None:
        supported = tuple(sorted(_KNOWN_DTYPES))
    elif isinstance(declared, (tuple, list, set, frozenset)) and all(
        isinstance(item, str) and item.strip() for item in declared
    ):
        supported = tuple(sorted({_normalize_dtype(item) for item in declared}))
        unknown_supported = tuple(dtype for dtype in supported if dtype not in _KNOWN_DTYPES)
        if unknown_supported:
            raise ValueError(
                "machine attribute 'supported_dtypes' contains unknown dtype(s): "
                + ", ".join(unknown_supported)
            )
    else:
        raise ValueError("machine attribute 'supported_dtypes' must be a sequence of dtype names")
    unsupported = tuple(dtype for dtype in graph_dtypes if dtype not in supported)
    if unsupported and policy == "strict":
        raise ValueError(
            f"machine '{machine.config_id}' does not natively support dtype(s): "
            + ", ".join(unsupported)
            + "; set dtype_policy='fallback' to run an explicitly marked analytical fallback"
        )
    return {
        "policy": policy,
        "graph_dtypes": list(graph_dtypes),
        "supported_dtypes": list(supported),
        "unsupported_dtypes": list(unsupported),
        "status": "fallback" if unsupported else "native",
    }


@dataclass(frozen=True)
class CompilerDiagnostic:
    """One stable compiler diagnostic for CLI and manifest output."""

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
    stablehlo: OfficialStableHLOModule
    source_frontend: FrontendImport
    gc_artifact: GCArtifact | None = None
    tisa_dialect: TISADialectProgram | None = None
    diagnostics: tuple[CompilerDiagnostic, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        issues.extend(self.frontend.validate())
        issues.extend(self.source_frontend.validate())
        issues.extend(self.stablehlo.validate())
        issues.extend(self.graph.validate())
        issues.extend(self.schedule.validate(self.graph))
        issues.extend(self.tile_graph.validate())
        issues.extend(self.tisa_program.validate())
        issues.extend(self.backend_artifact.validate())
        if self.gc_artifact is not None:
            issues.extend(self.gc_artifact.validate())
        if self.tisa_dialect is not None:
            issues.extend(self.tisa_dialect.validate())
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frontend": self.frontend.to_dict(),
            "source_frontend": self.source_frontend.to_dict(),
            "stablehlo": self.stablehlo.to_dict(),
            "graph": self.graph.to_dict(),
            "schedule": self.schedule.to_dict(),
            "tile_graph": self.tile_graph.to_dict(),
            "tisa_program": self.tisa_program.to_dict(),
            "backend_artifact": self.backend_artifact.to_dict(),
            "gc_artifact": self.gc_artifact.to_dict() if self.gc_artifact else None,
            "tisa_dialect": self.tisa_dialect.to_dict() if self.tisa_dialect else None,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "attributes": dict(self.attributes),
        }


def compile_operator_graph(
    graph: OperatorGraph,
    machine: MachineConfig,
    *,
    frontend: FrontendImport,
    source_frontend: FrontendImport,
    stablehlo: OfficialStableHLOModule,
    tile_size: int = 32,
    tile_size_candidates: Sequence[int] | None = None,
    registry: LoweringRegistry | None = None,
    codegen_backend: CodegenBackend | None = None,
) -> CompiledArtifact:
    """Compile an imported StableHLO graph from Canonical IR through backend payloads."""

    graph_issues = graph.validate()
    if graph_issues:
        raise ValueError("compiler input graph is invalid: " + "; ".join(graph_issues))
    machine_issues = machine.validate()
    if machine_issues:
        raise ValueError("compiler machine is invalid: " + "; ".join(machine_issues))
    dtype_compatibility = _dtype_compatibility(graph, machine)

    gc_artifact = default_graph_compiler().compile(
        graph,
        machine,
        tile_size=tile_size,
        tile_size_candidates=tile_size_candidates,
    )
    graph = gc_artifact.graph
    frontend = replace(
        frontend,
        graph=graph,
        provenance={
            **dict(frontend.provenance),
            "compiler_passes": [item.pass_name for item in gc_artifact.diagnostics],
            "graph_compiler": gc_artifact.attributes.get("compiler"),
        },
    )

    schedule = gc_artifact.schedule
    tile_graph = gc_artifact.tile_graph
    tisa_dialect = default_fusion_compiler().compile(gc_artifact, machine)
    program = default_tisa_generator().generate(tisa_dialect)

    selected_codegen = codegen_backend or default_codegen_backend_registry().create(
        "analytical",
        lowering_registry=registry or default_lowering_registry(),
    )
    backend_artifact = selected_codegen.lower(
        graph,
        schedule,
        tile_graph,
        machine,
        program=program,
    )
    compile_statistics = build_compile_statistics(
        graph,
        tile_graph,
        program,
        backend_artifact,
        machine,
    )

    result = CompiledArtifact(
        frontend=frontend,
        source_frontend=source_frontend,
        stablehlo=stablehlo,
        graph=graph,
        schedule=schedule,
        tile_graph=tile_graph,
        tisa_program=program,
        backend_artifact=backend_artifact,
        gc_artifact=gc_artifact,
        tisa_dialect=tisa_dialect,
        diagnostics=(
            CompilerDiagnostic("info", "torch.export", "captured PyTorch module"),
            CompilerDiagnostic("info", "torch-xla", "exported official StableHLO"),
            *(CompilerDiagnostic(item.level, item.pass_name, item.message) for item in gc_artifact.diagnostics),
            CompilerDiagnostic("info", "tiling", f"generated {len(tile_graph.tiles)} tile instances"),
            CompilerDiagnostic("info", "tisa", f"generated {len(program.instructions)} TISA instructions"),
        ),
        attributes={
            "compiler_pipeline": "pytorch->torch.export->torch-xla->stablehlo->GC->FC->tisa-generator->backend",
            "compiler_stages": [
                "framework_bridge",
                "graph_compiler",
                "fusion_compiler",
                "tisa_generator",
                "backend",
            ],
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
            "dtype_compatibility": dtype_compatibility,
            "codegen_direction": backend_artifact.attributes.get(
                "codegen_direction", "tilegraph->tisa->backend-payload"
            ),
            "codegen_backend": selected_codegen.name,
            "codegen_backend_capabilities": selected_codegen.capabilities.to_dict(),
            "model_id": source_frontend.model_id,
            "compile_statistics": compile_statistics,
        },
    )
    issues = result.validate()
    if issues:
        raise ValueError("compiled artifact validation failed: " + "; ".join(issues))
    return result


def compile_torch_module(
    module: Any,
    args: Sequence[Any],
    machine: MachineConfig,
    *,
    kwargs: Mapping[str, Any] | None = None,
    dynamic_shapes: Mapping[str, Any] | None = None,
    model_id: str = "torch_model",
    shape_environment: Mapping[str, int] | None = None,
    tile_size: int = 32,
    tile_size_candidates: Sequence[int] | None = None,
    registry: LoweringRegistry | None = None,
    codegen_backend: CodegenBackend | None = None,
) -> CompiledArtifact:
    """Compile one PyTorch module through the paper-aligned frontend route."""

    from npu_ooo.frontend import (
        OfficialStableHLOAdapter,
        TorchExportAdapter,
        TorchXLAStableHLOExporter,
    )

    # 1. Capture Python execution as a stable ATen/FX program.
    exported_program = TorchExportAdapter.capture_module(
        module,
        args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    source_frontend = TorchExportAdapter.from_exported_program(
        exported_program,
        model_id=model_id,
        variant="torch-export-v1",
        shape_environment=shape_environment,
    )

    # 2. Let torch-xla own ATen-to-StableHLO legalization.
    stablehlo = TorchXLAStableHLOExporter.export_program(
        exported_program,
        model_id=model_id,
        variant="stablehlo-torch-xla-v1",
    )
    dynamic_operations = tuple(
        sorted(
            set(
                re.findall(
                    r"stablehlo\.(?:get_dimension_size|dynamic_[A-Za-z_][\w.]*)",
                    stablehlo.text,
                )
            )
        )
    )
    if dynamic_operations:
        raise FrontendImportError(
            "dynamic StableHLO requires a shape-specialization pass before Canonical "
            "import; shape_environment resolves Canonical symbols but does not rewrite "
            "StableHLO shape-tensor subgraphs. Dynamic operations: "
            + ", ".join(dynamic_operations)
        )

    # 3. Verify with official MLIR bindings and import supported semantics.
    stable_frontend = OfficialStableHLOAdapter.import_text(
        stablehlo.text,
        model_id=model_id,
        variant=stablehlo.variant,
        shape_environment=shape_environment,
    )
    stable_frontend = replace(
        stable_frontend,
        family=source_frontend.family,
        provenance={
            **dict(stable_frontend.provenance),
            "source_frontend": "torch.export",
            "stablehlo_exporter": "torch-xla",
            "stablehlo_exporter_version": stablehlo.provenance.get("exporter_version"),
        },
    )
    graph = (
        stable_frontend.graph.resolve(dict(stable_frontend.shape_environment))
        if stable_frontend.shape_environment
        else stable_frontend.graph
    )

    # 4. Run the reusable Canonical IR -> TISA -> backend phase.
    return compile_operator_graph(
        graph,
        machine,
        frontend=stable_frontend,
        source_frontend=source_frontend,
        stablehlo=stablehlo,
        tile_size=tile_size,
        tile_size_candidates=tile_size_candidates,
        registry=registry,
        codegen_backend=codegen_backend,
    )


__all__ = [
    "CompilerDiagnostic",
    "CompiledArtifact",
    "compile_operator_graph",
    "compile_torch_module",
]
