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


def build_rmsnorm_model(*, rows: int = 128, cols: int = 96) -> ModelSpec:
    graph = OperatorGraph(
        graph_id="rmsnorm_graph",
        tensors=(TensorSpec("X", ("M", "N")), TensorSpec("Y", ("M", "N"))),
        operators=(
            OperatorSpec(
                op_id="rmsnorm0",
                op_type=SemanticOpType.RMSNORM,
                inputs=("X",),
                outputs=("Y",),
                iteration_dims=(("M", "M"),),
                reduction_dims=(("N", "N"),),
                attributes={"axis": "N", "epsilon": 1e-5},
                provenance={"template_op": "rmsnorm"},
            ),
        ),
        attributes={"pattern": "row_rmsnorm"},
    )
    return ModelSpec(
        model_id="row_rmsnorm",
        family=ModelFamily.DECODER_TRANSFORMER,
        variant="row-rmsnorm-v0",
        shape_symbols=(("M", rows), ("N", cols)),
        templates=(GraphTemplate("row_rmsnorm", graph, parameters=("M", "N")),),
        top_level_template="row_rmsnorm",
        dtype_policy="fp16",
    )


def build_rmsnorm_case(
    *,
    architecture_profile: str = "minimal",
    scheduler_profile: str = "sequential",
    evaluation_scope: EvaluationScope = EvaluationScope.ONE_BLOCK,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id="row_rmsnorm_default",
        model_id="row_rmsnorm",
        evaluation_scope=evaluation_scope,
        phase=ExecutionPhase.INFERENCE,
        batch=1,
        dtype="fp16",
        architecture_profile=architecture_profile,
        scheduler_profile=scheduler_profile,
    )
