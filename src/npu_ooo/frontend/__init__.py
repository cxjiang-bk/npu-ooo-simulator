"""Framework bridges that converge on the project's canonical operator IR.

The adapters are intentionally dependency-light.  ``torch`` is imported only
when ``TorchExportAdapter`` is used, so the analytical simulator remains
installable in environments without PyTorch/ExecuTorch.
"""

from .bridge import (
    FrontendImport,
    FrontendImportError,
    JsonGraphAdapter,
    TorchExportAdapter,
    import_operator_graph,
)

__all__ = [
    "FrontendImport",
    "FrontendImportError",
    "JsonGraphAdapter",
    "TorchExportAdapter",
    "import_operator_graph",
]
