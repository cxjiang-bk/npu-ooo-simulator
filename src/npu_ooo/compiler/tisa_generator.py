from __future__ import annotations

"""TISA dialect to virtual TISA program generation."""

from dataclasses import replace

from npu_ooo.ir import TISAProgram

from .fusion_compiler import TISADialectProgram


class TISAGenerator:
    """Serialize FC operations while retaining the TISA semantic contract."""

    name = "tisa-generator-python-v1"

    def generate(self, dialect: TISADialectProgram) -> TISAProgram:
        issues = dialect.validate()
        if issues:
            raise ValueError("TISA dialect is invalid: " + "; ".join(issues))
        return replace(
            dialect.program,
            program_id=dialect.program.program_id.replace(".tisa-dialect", ".tisa"),
            attributes={
                **dict(dialect.program.attributes),
                "paper_stage": "TISA_GENERATOR",
                "virtual_isa": "tisa-v1",
                "source_dialect": "tisa",
                "implementation": "python-semantic-proxy",
            },
        )


def default_tisa_generator() -> TISAGenerator:
    return TISAGenerator()


__all__ = ["TISAGenerator", "default_tisa_generator"]
