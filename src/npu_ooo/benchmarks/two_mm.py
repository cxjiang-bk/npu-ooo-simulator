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


def build_two_matmul_model() -> ModelSpec:
    graph = OperatorGraph(
        graph_id="two_matmul_graph",
        tensors=(
            TensorSpec("A", ("M", "K")),
            TensorSpec("B", ("K", "L")),
            TensorSpec("C", ("M", "L"), attributes={"intermediate": True}),
            TensorSpec("D", ("L", "N")),
            TensorSpec("E", ("M", "N")),
        ),
        operators=(
            OperatorSpec(
                op_id="gemm0",
                op_type=SemanticOpType.MATMUL,
                inputs=("A", "B"),
                outputs=("C",),
                iteration_dims=(("M", "M"), ("L", "L")),
                reduction_dims=(("K", "K"),),
                provenance={"template_op": "gemm0"},
            ),
            OperatorSpec(
                op_id="gemm1",
                op_type=SemanticOpType.MATMUL,
                inputs=("C", "D"),
                outputs=("E",),
                iteration_dims=(("M", "M"), ("N", "N")),
                reduction_dims=(("L", "L"),),
                provenance={"template_op": "gemm1"},
            ),
        ),
        edges=(DataEdge("gemm0", "gemm1", "C"),),
        attributes={"pattern": "matmul_matmul"},
    )
    return ModelSpec(
        model_id="two_matmul",
        family=ModelFamily.SYNTHETIC,
        variant="2mm-v0",
        shape_symbols=(("M", 128), ("K", 64), ("L", 96), ("N", 80)),
        templates=(GraphTemplate("two_matmul", graph, parameters=("M", "K", "L", "N")),),
        top_level_template="two_matmul",
        dtype_policy="fp16",
    )


def build_two_matmul_case(
    *,
    architecture_profile: str = "minimal",
    scheduler_profile: str = "sequential",
    evaluation_scope: EvaluationScope = EvaluationScope.FULL_MODEL,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id="two_matmul_default",
        model_id="two_matmul",
        evaluation_scope=evaluation_scope,
        phase=ExecutionPhase.INFERENCE,
        batch=1,
        dtype="fp16",
        architecture_profile=architecture_profile,
        scheduler_profile=scheduler_profile,
    )
