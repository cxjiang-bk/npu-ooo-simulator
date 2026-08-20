from __future__ import annotations

from npu_ooo.ir import (
    BenchmarkCase,
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


def build_softmax_model(*, rows: int = 128, cols: int = 96) -> ModelSpec:
    graph = OperatorGraph(
        graph_id="softmax_graph",
        tensors=(
            TensorSpec("X", ("M", "N")),
            TensorSpec("Y", ("M", "N")),
        ),
        operators=(
            OperatorSpec(
                op_id="softmax0",
                op_type=SemanticOpType.SOFTMAX,
                inputs=("X",),
                outputs=("Y",),
                iteration_dims=(("M", "M"),),
                reduction_dims=(("N", "N"),),
                attributes={"axis": "N"},
                provenance={"template_op": "softmax"},
            ),
        ),
        attributes={"pattern": "row_softmax"},
    )
    return ModelSpec(
        model_id="row_softmax",
        family=ModelFamily.SYNTHETIC,
        variant="row-softmax-v0",
        shape_symbols=(("M", rows), ("N", cols)),
        templates=(GraphTemplate("row_softmax", graph, parameters=("M", "N")),),
        top_level_template="row_softmax",
        dtype_policy="fp16",
    )


def build_softmax_case(
    *,
    architecture_profile: str = "minimal",
    scheduler_profile: str = "sequential",
    evaluation_scope: EvaluationScope = EvaluationScope.FULL_MODEL,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id="row_softmax_default",
        model_id="row_softmax",
        evaluation_scope=evaluation_scope,
        phase=ExecutionPhase.INFERENCE,
        batch=1,
        dtype="fp16",
        architecture_profile=architecture_profile,
        scheduler_profile=scheduler_profile,
    )
