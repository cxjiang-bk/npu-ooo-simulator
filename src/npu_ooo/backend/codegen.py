"""Code-generation backend adapters for TISA payload materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import (
    BackendArtifact,
    OperatorGraph,
    ScheduleSpec,
    TISAProgram,
    TileGraph,
)

from .contracts import BackendCapabilities, validate_backend_capability
from .registry import analytical_capabilities

if TYPE_CHECKING:
    from npu_ooo.lowering import LoweringRegistry


@dataclass(frozen=True)
class AnalyticalCodegenBackend:
    """Adapt the existing analytical payload lowerer to CodegenBackend."""

    name: str = "analytical"
    lowering_registry: LoweringRegistry | None = None
    capabilities: BackendCapabilities = field(
        default_factory=lambda: BackendCapabilities(
            backend="analytical",
            supported_primitives=analytical_capabilities().supported_primitives,
            calibration_status="analytical",
            attributes={
                "payload": "ExecutionGraph",
                "codegen_direction": "tilegraph->tisa->analytical-payload",
            },
        )
    )

    def lower(
        self,
        graph: OperatorGraph,
        schedule: ScheduleSpec,
        tile_graph: TileGraph,
        machine: MachineConfig,
        *,
        program: TISAProgram,
    ) -> BackendArtifact:
        # Keep the adapter outside compiler internals and load the legacy
        # implementation only when this backend is selected.
        from npu_ooo.compiler.tisa_first import AnalyticalBackendCodegen

        result = AnalyticalBackendCodegen().lower(
            graph,
            schedule,
            tile_graph,
            machine,
            program=program,
            registry=self.lowering_registry,
        )
        issues = validate_backend_capability(result.artifact, machine, self.capabilities)
        if issues:
            raise ValueError(
                "codegen backend capability validation failed: " + "; ".join(issues)
            )
        return result.artifact


__all__ = ["AnalyticalCodegenBackend"]
