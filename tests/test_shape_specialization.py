import unittest

from npu_ooo.frontend import FrontendImportError
from npu_ooo.frontend.shape_specialization import specialize_stablehlo
from npu_ooo.ir import OperatorGraph, TensorSpec


class ShapeSpecializationTest(unittest.TestCase):
    def _graph(self) -> OperatorGraph:
        return OperatorGraph(
            graph_id="dynamic-shape-test",
            tensors=(TensorSpec("value", (2, 4), dtype="f16"), TensorSpec("out", (2, 4), dtype="f16")),
            operators=(),
            attributes={"graph_inputs": ["value"], "graph_outputs": ["out"]},
        )

    def test_shape_tensor_dataflow_is_evaluated_and_dead_ops_removed(self) -> None:
        text = """
        module {
          func.func @main(%arg0: tensor<?x4xf16>) -> tensor<?x4xf16> {
            %shape = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x4xf16>) -> tensor<i32>
            %shape_vec = stablehlo.reshape %shape : (tensor<i32>) -> tensor<1xi32>
            %width = stablehlo.constant dense<4> : tensor<1xi32>
            %target = stablehlo.concatenate %shape_vec, %width, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
            %value = stablehlo.dynamic_broadcast_in_dim %arg0, %target, dims = [0, 1] : (tensor<?x4xf16>, tensor<2xi32>) -> tensor<?x4xf16>
            return %value : tensor<?x4xf16>
          }
        }
        """
        result = specialize_stablehlo(text, self._graph(), shape_environment={"batch": 2})

        self.assertIn("stablehlo.broadcast_in_dim", result.text)
        self.assertNotIn("stablehlo.dynamic_broadcast_in_dim", result.text)
        self.assertNotIn("stablehlo.get_dimension_size", result.text)
        self.assertEqual(
            set(result.dynamic_operations),
            {"stablehlo.dynamic_broadcast_in_dim", "stablehlo.get_dimension_size"},
        )
        self.assertTrue(result.removed_shape_operations)

    def test_unsupported_dynamic_operation_fails_explicitly(self) -> None:
        text = """
        module {
          func.func @main(%arg0: tensor<?xf16>) -> tensor<?xf16> {
            %0 = stablehlo.dynamic_iota %arg0 : (tensor<?xf16>) -> tensor<?xf16>
            return %0 : tensor<?xf16>
          }
        }
        """
        with self.assertRaisesRegex(FrontendImportError, "does not support dynamic operation"):
            specialize_stablehlo(text, self._graph())


if __name__ == "__main__":
    unittest.main()
