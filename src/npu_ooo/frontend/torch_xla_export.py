from __future__ import annotations

"""Optional torch-xla framework bridge for official StableHLO export."""

from dataclasses import replace
import hashlib
import importlib.metadata
from typing import Any

from .bridge import FrontendImportError
from .stablehlo_official import OfficialStableHLOAdapter, OfficialStableHLOModule


def torch_xla_available() -> bool:
    try:
        import torch_xla.stablehlo  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return False
    return True


def torch_xla_version() -> str | None:
    try:
        return importlib.metadata.version("torch-xla")
    except importlib.metadata.PackageNotFoundError:
        return None


class TorchXLAStableHLOExporter:
    """Export ``torch.export.ExportedProgram`` through torch-xla's bridge."""

    @classmethod
    def export_program(
        cls,
        exported_program: Any,
        *,
        model_id: str = "torch_xla_model",
        variant: str = "stablehlo-torch-xla-v1",
    ) -> OfficialStableHLOModule:
        try:
            from torch_xla.stablehlo import (
                StableHLOExportOptions,
                exported_program_to_stablehlo,
            )
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            raise FrontendImportError(
                "torch-xla StableHLO exporter is unavailable; install the verified "
                "torch-xla package (see docs/install-stablehlo.md)"
            ) from exc
        try:
            options = StableHLOExportOptions(include_human_readable_text=True)
            exported = exported_program_to_stablehlo(exported_program, options)
            text = exported.get_stablehlo_text()
            bytecode = exported.get_stablehlo_bytecode()
        except Exception as exc:
            raise FrontendImportError(f"torch-xla StableHLO export failed: {exc}") from exc
        if not isinstance(text, str) or not text.strip():
            raise FrontendImportError("torch-xla did not return human-readable StableHLO")
        verified = OfficialStableHLOAdapter.parse_text(
            text,
            model_id=model_id,
            variant=variant,
        )
        bytecode_bytes = bytes(bytecode)
        return replace(
            verified,
            producer="torch-xla",
            provenance={
                **dict(verified.provenance),
                "exporter": "torch-xla",
                "exporter_version": torch_xla_version(),
                "bytecode_size": len(bytecode_bytes),
                "bytecode_sha256": hashlib.sha256(bytecode_bytes).hexdigest(),
            },
        )


__all__ = [
    "TorchXLAStableHLOExporter",
    "torch_xla_available",
    "torch_xla_version",
]
