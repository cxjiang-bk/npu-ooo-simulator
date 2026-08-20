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


def build_elementwise_model(*, rows: int = 128, cols: int = 96) -> ModelSpec:
    graph = OperatorGraph(
        graph_id="elementwise_add_graph",
        tensors=(
            TensorSpec("A", ("M", "N")),
            TensorSpec("B", ("M", "N")),
            TensorSpec("C", ("M", "N")),
        ),
        operators=(
            OperatorSpec(
                op_id="add0",
                op_type=SemanticOpType.RESIDUAL_ADD,
                inputs=("A", "B"),
                outputs=("C",),
                iteration_dims=(("M", "M"), ("N", "N")),
                provenance={"template_op": "residual_add"},
            ),
        ),
        attributes={"pattern": "elementwise_add"},
    )
    return ModelSpec(
        model_id="elementwise_add",
        family=ModelFamily.SYNTHETIC,
        variant="elementwise-add-v0",
        shape_symbols=(("M", rows), ("N", cols)),
        templates=(GraphTemplate("elementwise_add", graph, parameters=("M", "N")),),
        top_level_template="elementwise_add",
        dtype_policy="fp16",
    )


def build_elementwise_case(
    *,
    architecture_profile: str = "minimal",
    scheduler_profile: str = "sequential",
    evaluation_scope: EvaluationScope = EvaluationScope.FULL_MODEL,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id="elementwise_add_default",
        model_id="elementwise_add",
        evaluation_scope=evaluation_scope,
        phase=ExecutionPhase.INFERENCE,
        batch=1,
        dtype="fp16",
        architecture_profile=architecture_profile,
        scheduler_profile=scheduler_profile,
    )
