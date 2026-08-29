import unittest

from npu_ooo.frontend import FrontendImportError, OfficialStableHLOAdapter
from npu_ooo.frontend.shape_specialization import specialize_stablehlo
from npu_ooo.ir import OperatorGraph, TensorSpec


class ShapeSpecializationTest(unittest.TestCase):
    def _graph(
        self,
        value_shape: tuple[int, ...] = (2, 4),
        output_shape: tuple[int, ...] = (2, 4),
    ) -> OperatorGraph:
        return OperatorGraph(
            graph_id="dynamic-shape-test",
            tensors=(
                TensorSpec("value", value_shape, dtype="f16"),
                TensorSpec("out", output_shape, dtype="f16"),
            ),
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

    def test_dynamic_update_slice_remains_an_explicit_boundary(self) -> None:
        text = """
        module {
          func.func @main(%arg0: tensor<4x4xf16>, %arg1: tensor<1x4xf16>, %start: tensor<i32>) -> tensor<4x4xf16> {
            %value = stablehlo.dynamic_update_slice %arg0, %arg1, %start, %start : (tensor<4x4xf16>, tensor<1x4xf16>, tensor<i32>, tensor<i32>) -> tensor<4x4xf16>
            return %value : tensor<4x4xf16>
          }
        }
        """
        with self.assertRaisesRegex(FrontendImportError, "does not support dynamic operation"):
            specialize_stablehlo(text, self._graph((4, 4), (4, 4)))

    def test_constant_dynamic_slice_is_clamped_and_staticized(self) -> None:
        text = """
        module {
          func.func @main(%arg0: tensor<?x4xf16>) -> tensor<2x3xf16> {
            %start0 = stablehlo.constant dense<99> : tensor<i32>
            %start1 = stablehlo.constant dense<1> : tensor<i32>
            %value = stablehlo.dynamic_slice %arg0, %start0, %start1 sizes = [2, 3] : (tensor<?x4xf16>, tensor<i32>, tensor<i32>) -> tensor<2x3xf16>
            return %value : tensor<2x3xf16>
          }
        }
        """
        result = specialize_stablehlo(text, self._graph())

        self.assertIn(
            "stablehlo.slice %arg0 [0:2, 1:4]",
            result.text,
        )
        self.assertNotIn("stablehlo.dynamic_slice", result.text)
        self.assertNotIn("%start0", result.text)
        self.assertNotIn("%start1", result.text)
        self.assertIn("stablehlo.dynamic_slice", result.dynamic_operations)
        self.assertTrue(OfficialStableHLOAdapter.parse_text(result.text).verified)

    def test_constant_dynamic_reshape_is_staticized(self) -> None:
        text = """
        module {
          func.func @main(%arg0: tensor<2x6xf16>) -> tensor<3x4xf16> {
            %shape = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
            %value = stablehlo.dynamic_reshape %arg0, %shape : (tensor<2x6xf16>, tensor<2xi64>) -> tensor<3x4xf16>
            return %value : tensor<3x4xf16>
          }
        }
        """
        result = specialize_stablehlo(text, self._graph((2, 6), (3, 4)))

        self.assertIn(
            "stablehlo.reshape %arg0 : (tensor<2x6xf16>) -> tensor<3x4xf16>",
            result.text,
        )
        self.assertNotIn("stablehlo.dynamic_reshape", result.text)
        self.assertNotIn("%shape", result.text)
        self.assertIn("stablehlo.dynamic_reshape", result.dynamic_operations)
        self.assertTrue(OfficialStableHLOAdapter.parse_text(result.text).verified)

    def test_dynamic_reshape_rejects_element_count_mismatch(self) -> None:
        text = """
        module {
          func.func @main(%arg0: tensor<2x6xf16>) -> tensor<5x3xf16> {
            %shape = stablehlo.constant dense<[5, 3]> : tensor<2xi64>
            %value = stablehlo.dynamic_reshape %arg0, %shape : (tensor<2x6xf16>, tensor<2xi64>) -> tensor<5x3xf16>
            return %value : tensor<5x3xf16>
          }
        }
        """
        with self.assertRaisesRegex(FrontendImportError, "changes element count"):
            specialize_stablehlo(text, self._graph((2, 6), (5, 3)))

    def test_nonconstant_dynamic_slice_fails_explicitly(self) -> None:
        text = """
        module {
          func.func @main(%arg0: tensor<?x4xf16>, %start: tensor<i32>) -> tensor<2x3xf16> {
            %value = stablehlo.dynamic_slice %arg0, %start, %start sizes = [2, 3] : (tensor<?x4xf16>, tensor<i32>, tensor<i32>) -> tensor<2x3xf16>
            return %value : tensor<2x3xf16>
          }
        }
        """
        with self.assertRaisesRegex(FrontendImportError, "constant start indices"):
            specialize_stablehlo(text, self._graph())

    def test_dynamic_slice_negative_start_is_clamped(self) -> None:
        text = """
        module {
          func.func @main(%arg0: tensor<?x4xf16>) -> tensor<1x2xf16> {
            %start0 = stablehlo.constant dense<-3> : tensor<i32>
            %start1 = stablehlo.constant dense<0> : tensor<i32>
            %value = stablehlo.dynamic_slice %arg0, %start0, %start1 sizes = [1, 2] : (tensor<?x4xf16>, tensor<i32>, tensor<i32>) -> tensor<1x2xf16>
            return %value : tensor<1x2xf16>
          }
        }
        """
        result = specialize_stablehlo(text, self._graph())
        self.assertIn("stablehlo.slice %arg0 [0:1, 0:2]", result.text)

    def test_dynamic_slice_rejects_size_rank_mismatch(self) -> None:
        text = """
        module {
          func.func @main(%arg0: tensor<?x4xf16>) -> tensor<2x3xf16> {
            %start0 = stablehlo.constant dense<0> : tensor<i32>
            %start1 = stablehlo.constant dense<0> : tensor<i32>
            %value = stablehlo.dynamic_slice %arg0, %start0, %start1 sizes = [2] : (tensor<?x4xf16>, tensor<i32>, tensor<i32>) -> tensor<2x3xf16>
            return %value : tensor<2x3xf16>
          }
        }
        """
        with self.assertRaisesRegex(FrontendImportError, "rank does not match"):
            specialize_stablehlo(text, self._graph())


if __name__ == "__main__":
    unittest.main()
