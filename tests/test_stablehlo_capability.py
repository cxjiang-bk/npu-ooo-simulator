import unittest

from npu_ooo.frontend import FrontendImportError
from npu_ooo.frontend.stablehlo import StableHLOAdapter


class StableHLOCapabilityBoundaryTest(unittest.TestCase):
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

    def test_unknown_operation_reports_missing_capability_and_known_set(self) -> None:
        with self.assertRaisesRegex(
            FrontendImportError,
            r"missing StableHLO capability.*stablehlo.convolution.*Known operations:.*stablehlo.dot_general",
        ):
            StableHLOAdapter.from_text(
                """
                module {
                  func.func @main(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
                    %0 = stablehlo.convolution %arg0, %arg0 : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
                    return %0 : tensor<1x3x4x4xf32>
                  }
                }
                """
            )


if __name__ == "__main__":
    unittest.main()
