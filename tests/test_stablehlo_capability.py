import unittest
from types import SimpleNamespace

from npu_ooo.frontend import FrontendImportError
from npu_ooo.frontend.stablehlo import StableHLOAdapter
from npu_ooo.frontend.stablehlo_official import _project_module


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

    def test_tensor_encoding_is_preserved_as_layout_metadata(self) -> None:
        imported = StableHLOAdapter.from_text(
            """
            module {
              func.func @main(%arg0: tensor<2x3xf32, #row_major>) -> tensor<2x3xf32, #row_major> {
                %0 = stablehlo.negate %arg0 : (tensor<2x3xf32, #row_major>) -> tensor<2x3xf32, #row_major>
                return %0 : tensor<2x3xf32, #row_major>
              }
            }
            """
        )

        tensors = {tensor.name: tensor for tensor in imported.graph.tensors}
        self.assertEqual(tensors["arg0"].shape, (2, 3))
        self.assertEqual(tensors["arg0"].attributes["layout_source"], "stablehlo_encoding")
        self.assertEqual(tensors["arg0"].attributes["layout_encoding"], "#row_major")
        self.assertEqual(tensors["0"].attributes["layout_encoding"], "#row_major")

    def test_official_projection_rejects_multi_result_operation_explicitly(self) -> None:
        class Value:
            def __init__(self, type_name):
                self.type = type_name

        class Operation:
            def __init__(self, name, *, results=(), operands=()):
                self.name = name
                self.operation = self
                self.results = list(results)
                self.operands = list(operands)
                self.attributes = {}
                self.regions = []

        arguments = [Value("tensor<2x3xf32>")]
        first = Value("tensor<2x3xf32>")
        second = Value("tensor<2x3xf32>")
        multi = Operation("stablehlo.fake_multi", results=(first, second), operands=arguments)
        returned = Operation("func.return", operands=(first,))
        block = SimpleNamespace(arguments=arguments)
        block.__iter__ = lambda self: iter((multi, returned))
        # A tiny iterable block object is enough for the projection boundary.
        class Block:
            def __init__(self, arguments, operations):
                self.arguments = arguments
                self._operations = operations

            def __iter__(self):
                return iter(self._operations)

        function_block = Block(arguments, (multi, returned))
        function = Operation("func.func")
        function.regions = [SimpleNamespace(blocks=[function_block])]
        function.attributes = {"function_type": "(tensor<2x3xf32>) -> tensor<2x3xf32>"}
        module_block = Block([], (function,))
        module = SimpleNamespace(
            operation=SimpleNamespace(regions=[SimpleNamespace(blocks=[module_block])])
        )

        with self.assertRaisesRegex(FrontendImportError, "multi-result operation.*2 results"):
            _project_module(module)

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
