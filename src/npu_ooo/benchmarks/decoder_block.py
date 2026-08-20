from __future__ import annotations

from npu_ooo.ir import (
    BenchmarkCase,
    DataEdge,
    EvaluationScope,
    ExecutionPhase,
    GraphTemplate,
    ModelFamily,
    ModelSpec,
    OperatorGraph,
    OperatorSpec,
    SemanticOpType,
    TensorSpec,
)


def build_decoder_block_model(*, tokens: int = 64, hidden: int = 96) -> ModelSpec:
    """Build a small heterogeneous decoder fragment for scheduling experiments."""

    graph = OperatorGraph(
        graph_id="decoder_block_micro_graph",
        tensors=(
            TensorSpec("X", ("M", "K")),
            TensorSpec("X_norm", ("M", "K")),
            TensorSpec("W_proj", ("K", "N")),
            TensorSpec("Projection", ("M", "N")),
            TensorSpec("Y", ("M", "N")),
        ),
        operators=(
            OperatorSpec(
                op_id="rmsnorm0",
                op_type=SemanticOpType.RMSNORM,
                inputs=("X",),
                outputs=("X_norm",),
                iteration_dims=(("M", "M"),),
                reduction_dims=(("K", "K"),),
                attributes={"axis": "K", "epsilon": 1e-5},
                provenance={"block_stage": "pre_norm"},
            ),
            OperatorSpec(
                op_id="projection0",
                op_type=SemanticOpType.MATMUL,
                inputs=("X_norm", "W_proj"),
                outputs=("Projection",),
                iteration_dims=(("M", "M"), ("N", "N")),
                reduction_dims=(("K", "K"),),
                provenance={"block_stage": "projection"},
            ),
            OperatorSpec(
                op_id="residual0",
                op_type=SemanticOpType.RESIDUAL_ADD,
                inputs=("Projection", "X"),
                outputs=("Y",),
                iteration_dims=(("M", "M"), ("N", "N")),
                provenance={"block_stage": "residual"},
            ),
        ),
        edges=(
            DataEdge("rmsnorm0", "projection0", "X_norm"),
            DataEdge("projection0", "residual0", "Projection"),
        ),
        attributes={
            "pattern": "pre_norm_projection_residual",
            "scope": "decoder_block_fragment",
        },
    )
    return ModelSpec(
        model_id="decoder_block_micro",
        family=ModelFamily.DECODER_TRANSFORMER,
        variant="pre-norm-projection-residual-v0",
        shape_symbols=(("M", tokens), ("K", hidden), ("N", hidden)),
        templates=(
            GraphTemplate(
                "decoder_block_micro",
                graph,
                parameters=("M", "K", "N"),
            ),
        ),
        top_level_template="decoder_block_micro",
        dtype_policy="fp16",
        attributes={"operator_count": 3, "weights": "shape_only"},
    )


def build_decoder_block_case(
    *,
    tokens: int = 64,
    architecture_profile: str = "minimal",
    scheduler_profile: str = "sequential",
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id="decoder_block_micro_default",
        model_id="decoder_block_micro",
        evaluation_scope=EvaluationScope.ONE_BLOCK,
        phase=ExecutionPhase.PREFILL,
        batch=1,
        sequence_length=tokens,
        dtype="fp16",
        architecture_profile=architecture_profile,
        scheduler_profile=scheduler_profile,
        attributes={"scope_note": "RMSNorm -> Matmul -> ResidualAdd fragment"},
    )
