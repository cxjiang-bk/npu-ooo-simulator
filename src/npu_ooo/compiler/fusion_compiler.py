from __future__ import annotations

"""Paper-aligned Fusion Compiler (FC) boundary."""

from dataclasses import dataclass, replace
from typing import Any, Mapping

from npu_ooo.arch import MachineConfig
from npu_ooo.ir import TISAProgram

from .graph_compiler import GCArtifact
from .tisa_first import TISASemanticBuilder


@dataclass(frozen=True)
class TISADialectProgram:
    """TISA dialect proxy emitted by FC before binary generation."""

    program: TISAProgram
    gc_artifact_id: str
    attributes: Mapping[str, Any]

    def validate(self) -> tuple[str, ...]:
        return self.program.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_stage": "FC",
            "dialect": "tisa",
            "implementation": "python-semantic-proxy",
            "program": self.program.to_dict(),
            "attributes": dict(self.attributes),
        }


class FusionCompiler:
    """Specialize a GC TileGraph into scheduler-visible TISA dialect ops."""

    name = "fusion-compiler-python-v1"

    def compile(self, artifact: GCArtifact, machine: MachineConfig) -> TISADialectProgram:
        issues = artifact.validate(machine)
        if issues:
            raise ValueError("FC input GC artifact is invalid: " + "; ".join(issues))
        program = TISASemanticBuilder().build(
            artifact.graph,
            artifact.schedule,
            artifact.tile_graph,
            machine,
            program_id=f"{artifact.graph.graph_id}.tisa-dialect",
        )
        program = replace(
            program,
            attributes={
                **dict(program.attributes),
                "paper_stage": "FC",
                "dialect": "tisa",
                "implementation": "python-semantic-proxy",
                "fusion_regions": "operator-and-tile semantic boundaries",
                "metadata_contract": [
                    "OpType",
                    "Operands",
                    "TileMem",
                    "TileMem.strides_bytes",
                    "TileMem.stride_expr",
                    "TileMem.layout",
                    "AccessType",
                    "Deps",
                    "UnitMap",
                ],
            },
        )
        result = TISADialectProgram(
            program=program,
            gc_artifact_id=artifact.graph.graph_id,
            attributes={
                "paper_stage": "FC",
                "compiler": self.name,
                "input_contract": "software-scheduled semantic TileGraph",
                "output_contract": "TISA dialect operations",
            },
        )
        result_issues = result.validate()
        if result_issues:
            raise ValueError("Fusion Compiler produced an invalid dialect: " + "; ".join(result_issues))
        return result


def default_fusion_compiler() -> FusionCompiler:
    return FusionCompiler()


__all__ = ["FusionCompiler", "TISADialectProgram", "default_fusion_compiler"]
