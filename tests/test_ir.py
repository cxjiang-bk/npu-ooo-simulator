import unittest

from npu_ooo.benchmarks import build_two_matmul_case, build_two_matmul_model
from npu_ooo.ir import EvaluationScope


class ModelIrTest(unittest.TestCase):
    def test_two_matmul_model_instantiates_symbolic_shapes(self) -> None:
        model = build_two_matmul_model()
        case = build_two_matmul_case()

        self.assertEqual(model.validate(), ())
        instance = model.instantiate(case)

        self.assertEqual(instance.validate(), ())
        self.assertEqual(instance.graph.topological_order(), ("gemm0", "gemm1"))
        self.assertEqual(
            {tensor.name: tensor.shape for tensor in instance.graph.tensors}["C"],
            (128, 96),
        )
        self.assertEqual(instance.provenance["evaluation_scope"], "full_model")

    def test_case_shape_override_is_independent_from_model_defaults(self) -> None:
        model = build_two_matmul_model()
        case = build_two_matmul_case().__class__(
            **{
                **build_two_matmul_case().to_dict(),
                "evaluation_scope": EvaluationScope.ONE_BLOCK,
                "shape_overrides": (("M", 32), ("N", 40)),
            }
        )

        instance = model.instantiate(case)
        tensors = {tensor.name: tensor for tensor in instance.graph.tensors}
        self.assertEqual(tensors["A"].shape, (32, 64))
        self.assertEqual(tensors["E"].shape, (32, 40))
        self.assertEqual(instance.provenance["evaluation_scope"], "one_block")

    def test_invalid_edge_and_cycle_are_rejected(self) -> None:
        model = build_two_matmul_model()
        graph = model.templates[0].graph
        invalid = graph.__class__(
            graph_id=graph.graph_id,
            tensors=graph.tensors,
            operators=graph.operators,
            edges=(
                graph.edges[0],
                graph.edges[0].__class__("gemm1", "gemm0", "C"),
            ),
        )
        self.assertTrue(any("cycle" in issue for issue in invalid.validate()))


if __name__ == "__main__":
    unittest.main()
