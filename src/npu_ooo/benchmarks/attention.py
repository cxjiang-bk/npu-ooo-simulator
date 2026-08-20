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


def build_attention_model(
    *,
    query_tokens: int = 64,
    key_tokens: int = 64,
    head_dim: int = 32,
) -> ModelSpec:
    """Build a single-head attention score/softmax/value fragment."""

    graph = OperatorGraph(
        graph_id="single_head_attention_graph",
        tensors=(
            TensorSpec("Q", ("M", "D")),
            TensorSpec("K_T", ("D", "S"), attributes={"transposed": True}),
            TensorSpec("Scores", ("M", "S"), attributes={"intermediate": True}),
            TensorSpec("P", ("M", "S"), attributes={"intermediate": True}),
            TensorSpec("V", ("S", "D")),
            TensorSpec("Context", ("M", "D")),
        ),
        operators=(
            OperatorSpec(
                op_id="attention_scores",
                op_type=SemanticOpType.MATMUL,
                inputs=("Q", "K_T"),
                outputs=("Scores",),
                iteration_dims=(("M", "M"), ("S", "S")),
                reduction_dims=(("D", "D"),),
                attributes={"scale": "1/sqrt(D)"},
                provenance={"attention_stage": "qk_scores"},
            ),
            OperatorSpec(
                op_id="attention_softmax",
                op_type=SemanticOpType.SOFTMAX,
                inputs=("Scores",),
                outputs=("P",),
                iteration_dims=(("M", "M"),),
                reduction_dims=(("S", "S"),),
                attributes={"axis": "S", "causal": False},
                provenance={"attention_stage": "score_normalization"},
            ),
            OperatorSpec(
                op_id="attention_context",
                op_type=SemanticOpType.MATMUL,
                inputs=("P", "V"),
                outputs=("Context",),
                iteration_dims=(("M", "M"), ("D", "D")),
                reduction_dims=(("S", "S"),),
                provenance={"attention_stage": "pv_context"},
            ),
        ),
        edges=(
            DataEdge("attention_scores", "attention_softmax", "Scores"),
            DataEdge("attention_softmax", "attention_context", "P"),
        ),
        attributes={
            "pattern": "single_head_qk_softmax_pv",
            "causal_mask": False,
            "kv_cache": False,
        },
    )
    return ModelSpec(
        model_id="single_head_attention",
        family=ModelFamily.DECODER_TRANSFORMER,
        variant="qk-softmax-pv-v0",
        shape_symbols=(("M", query_tokens), ("S", key_tokens), ("D", head_dim)),
        templates=(GraphTemplate("single_head_attention", graph, parameters=("M", "S", "D")),),
        top_level_template="single_head_attention",
        dtype_policy="fp16",
        attributes={"scope_note": "single head, no mask/cache"},
    )


def build_attention_case(
    *,
    query_tokens: int = 64,
    architecture_profile: str = "minimal",
    scheduler_profile: str = "sequential",
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id="single_head_attention_default",
        model_id="single_head_attention",
        evaluation_scope=EvaluationScope.ONE_BLOCK,
        phase=ExecutionPhase.PREFILL,
        batch=1,
        sequence_length=query_tokens,
        dtype="fp16",
        architecture_profile=architecture_profile,
        scheduler_profile=scheduler_profile,
        attributes={"scope_note": "QK^T -> Softmax -> PV"},
    )
