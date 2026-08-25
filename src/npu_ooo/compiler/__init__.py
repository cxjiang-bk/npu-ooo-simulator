"""Framework-independent compiler pipeline."""

from .pipeline import (
    CompilerDiagnostic,
    CompiledArtifact,
    compile_operator_graph,
    compile_torch_module,
)
from .passes import (
    CanonicalizeGraphPass,
    FoldTransposeIntoMatmulPass,
    LayerNormFusionPass,
    LinearDecompositionPass,
    PassDiagnostic,
    PassManager,
    PassResult,
    RecoverStableHLOFlattenedLinearPass,
    RecoverStableHLOLayerNormPass,
    RMSNormFusionPass,
    SoftmaxFusionPass,
    default_pass_manager,
)
from .planner import SchedulePlanner, default_schedule_planner
from .tisa_first import (
    TISASemanticBuilder,
    TISAStage,
)

__all__ = [
    "CompilerDiagnostic",
    "CompiledArtifact",
    "compile_operator_graph",
    "compile_torch_module",
    "CanonicalizeGraphPass",
    "FoldTransposeIntoMatmulPass",
    "LayerNormFusionPass",
    "LinearDecompositionPass",
    "PassDiagnostic",
    "PassManager",
    "PassResult",
    "RecoverStableHLOFlattenedLinearPass",
    "RecoverStableHLOLayerNormPass",
    "RMSNormFusionPass",
    "SoftmaxFusionPass",
    "default_pass_manager",
    "SchedulePlanner",
    "default_schedule_planner",
    "TISASemanticBuilder",
    "TISAStage",
]
