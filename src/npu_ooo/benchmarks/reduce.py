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


def build_reduce_model(*, rows: int = 128, cols: int = 96) -> ModelSpec:
    graph = OperatorGraph(
        graph_id="reduce_graph",
        tensors=(
            TensorSpec("X", ("M", "N")),
            TensorSpec("Y", ("M",)),
        ),
        operators=(
            OperatorSpec(
                op_id="reduce0",
                op_type=SemanticOpType.REDUCE,
                inputs=("X",),
                outputs=("Y",),
                iteration_dims=(("M", "M"),),
                reduction_dims=(("N", "N"),),
                attributes={"reduction": "sum"},
                provenance={"template_op": "reduce_sum"},
            ),
        ),
        attributes={"pattern": "row_reduce"},
    )
    return ModelSpec(
        model_id="row_reduce",
        family=ModelFamily.SYNTHETIC,
        variant="row-reduce-v0",
        shape_symbols=(("M", rows), ("N", cols)),
        templates=(GraphTemplate("row_reduce", graph, parameters=("M", "N")),),
        top_level_template="row_reduce",
        dtype_policy="fp16",
    )


def build_reduce_case(
    *,
    architecture_profile: str = "minimal",
    scheduler_profile: str = "sequential",
    evaluation_scope: EvaluationScope = EvaluationScope.FULL_MODEL,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id="row_reduce_default",
        model_id="row_reduce",
        evaluation_scope=evaluation_scope,
        phase=ExecutionPhase.INFERENCE,
        batch=1,
        dtype="fp16",
        architecture_profile=architecture_profile,
        scheduler_profile=scheduler_profile,
    )
