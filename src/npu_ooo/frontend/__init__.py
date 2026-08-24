"""Framework bridges that converge on the project's canonical operator IR.

The adapters are intentionally dependency-light.  ``torch`` is imported only
when ``TorchExportAdapter`` is used, and StableHLO can be supplied as textual
MLIR or as a module-like object, so the analytical simulator remains
installable without the optional framework packages.
"""

from .bridge import (
    FrontendImport,
    FrontendImportError,
    JsonGraphAdapter,
    TorchExportAdapter,
    import_operator_graph,
)
from .stablehlo import StableHLOAdapter
from .stablehlo_codegen import StableHLOGenerator, StableHLOModule, generate_stablehlo
from .stablehlo_official import (
    OfficialStableHLOAdapter,
    OfficialStableHLOGenerator,
    OfficialStableHLOModule,
    official_stablehlo_available,
    official_stablehlo_version,
)
from .torch_xla_export import (
    TorchXLAStableHLOExporter,
    torch_xla_available,
    torch_xla_version,
)

__all__ = [
    "FrontendImport",
    "FrontendImportError",
    "JsonGraphAdapter",
    "TorchExportAdapter",
    "import_operator_graph",
    "StableHLOAdapter",
    "StableHLOGenerator",
    "StableHLOModule",
    "generate_stablehlo",
    "OfficialStableHLOAdapter",
    "OfficialStableHLOGenerator",
    "OfficialStableHLOModule",
    "official_stablehlo_available",
    "official_stablehlo_version",
    "TorchXLAStableHLOExporter",
    "torch_xla_available",
    "torch_xla_version",
]
