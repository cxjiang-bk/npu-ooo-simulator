import unittest

from npu_ooo.frontend import FrontendImportError
from npu_ooo.frontend.stablehlo import StableHLOAdapter


class StableHLOCapabilityBoundaryTest(unittest.TestCase):
    def test_broadcast_in_dim_preserves_shape_and_dimension_mapping(self) -> None:
        imported = StableHLOAdapter.from_text(
            """
            module {
              func.func @main(%arg0: tensor<8xf32>) -> tensor<1x4x8xf32> {
                %0 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<8xf32>) -> tensor<1x4x8xf32>
                return %0 : tensor<1x4x8xf32>
              }
            }
            """
        )

        operator = imported.graph.operators[0]
        tensors = {tensor.name: tensor for tensor in imported.graph.tensors}
        self.assertEqual(operator.normalized_type, "reshape")
        self.assertEqual(tensors[operator.inputs[0]].shape, (8,))
        self.assertEqual(tensors[operator.outputs[0]].shape, (1, 4, 8))
        self.assertTrue(operator.attributes["broadcast"])
        self.assertEqual(operator.attributes["broadcast_dimensions"], [2])

    def test_broadcast_in_dim_rejects_incompatible_extent(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot broadcast"):
            StableHLOAdapter.from_text(
                """
                module {
                  func.func @main(%arg0: tensor<7xf32>) -> tensor<1x4x8xf32> {
                    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<7xf32>) -> tensor<1x4x8xf32>
                    return %0 : tensor<1x4x8xf32>
                  }
                }
                """
            )

    def test_convert_is_imported_with_dtype_semantics(self) -> None:
        imported = StableHLOAdapter.from_text(
            """
            module {
              func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf16> {
                %0 = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xf16>
                return %0 : tensor<4xf16>
              }
            }
            """
        )

        self.assertEqual(len(imported.graph.operators), 1)
        operator = imported.graph.operators[0]
        self.assertEqual(operator.normalized_type, "elementwise")
        self.assertEqual(operator.attributes["source_dtype"], "f32")
        self.assertEqual(operator.attributes["target_dtype"], "f16")
        self.assertEqual(operator.attributes["conversion_kind"], "dtype_cast")

    def test_scalar_constant_remains_rank_zero_operand(self) -> None:
        imported = StableHLOAdapter.from_text(
            """
            module {
              func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
                %c = stablehlo.constant dense<1.0> : tensor<f32>
                %0 = stablehlo.add %arg0, %c : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                return %0 : tensor<2x3xf32>
              }
            }
            """
        )

        tensors = {tensor.name: tensor for tensor in imported.graph.tensors}
        operator = imported.graph.operators[0]
        self.assertEqual(tensors["c"].shape, ())
        self.assertEqual(tensors["c"].attributes["constant_value"], 1.0)
        self.assertEqual(operator.inputs, ("arg0", "c"))

    def test_unknown_operation_reports_missing_capability_and_known_set(self) -> None:
        with self.assertRaisesRegex(
            FrontendImportError,
            r"missing StableHLO capability.*stablehlo.custom_call.*Known operations:.*stablehlo.dot_general",
        ):
            StableHLOAdapter.from_text(
                """
                module {
                  func.func @main(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
                    %0 = stablehlo.custom_call %arg0 {call_target_name = "unsupported"} : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
                    return %0 : tensor<1x3x4x4xf32>
                  }
                }
                """
            )


if __name__ == "__main__":
    unittest.main()
