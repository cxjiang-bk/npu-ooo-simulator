"""Framework-independent compiler pipeline."""

from .pipeline import (
    CompilerDiagnostic,
    CompiledArtifact,
    compile_operator_graph,
    compile_torch_module,
)
from .passes import (
    AttentionRegionPass,
    CanonicalizeGraphPass,
    FoldTransposeIntoMatmulPass,
    LayerNormFusionPass,
    LinearDecompositionPass,
    PassDiagnostic,
    PassManager,
    PassSnapshot,
    PassResult,
    RecoverStableHLOFlattenedLinearPass,
    RecoverStableHLOLayerNormPass,
    RecoverStableHLOKVCachePass,
    RotaryEmbeddingRegionPass,
    RMSNormFusionPass,
    SoftmaxFusionPass,
    SwiGLUFusionPass,
    default_pass_manager,
)
from .planner import SchedulePlanner, default_schedule_planner
from .statistics import build_compile_statistics
from .fusion_compiler import FusionCompiler, TISADialectProgram, default_fusion_compiler
from .graph_compiler import GCArtifact, GraphCompiler, default_graph_compiler
from .fusion_patterns import (
    SemanticFusionPattern,
    SemanticFusionPatternRegistry,
    default_semantic_fusion_registry,
)
from .tisa_generator import TISAGenerator, default_tisa_generator
from .tisa_dialect import (
    TISASemanticBuilder,
    TISAStage,
)

__all__ = [
    "CompilerDiagnostic",
    "CompiledArtifact",
    "compile_operator_graph",
    "compile_torch_module",
    "CanonicalizeGraphPass",
    "AttentionRegionPass",
    "FoldTransposeIntoMatmulPass",
    "LayerNormFusionPass",
    "LinearDecompositionPass",
    "PassDiagnostic",
    "PassManager",
    "PassSnapshot",
    "PassResult",
    "RecoverStableHLOFlattenedLinearPass",
    "RecoverStableHLOLayerNormPass",
    "RecoverStableHLOKVCachePass",
    "RotaryEmbeddingRegionPass",
    "RMSNormFusionPass",
    "SoftmaxFusionPass",
    "SwiGLUFusionPass",
    "default_pass_manager",
    "SchedulePlanner",
    "default_schedule_planner",
    "build_compile_statistics",
    "GCArtifact",
    "GraphCompiler",
    "default_graph_compiler",
    "SemanticFusionPattern",
    "SemanticFusionPatternRegistry",
    "default_semantic_fusion_registry",
    "FusionCompiler",
    "TISADialectProgram",
    "default_fusion_compiler",
    "TISAGenerator",
    "default_tisa_generator",
    "TISASemanticBuilder",
    "TISAStage",
]
