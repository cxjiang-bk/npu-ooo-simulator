import importlib.util
import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.compiler import compile_torch_module
from npu_ooo.frontend import official_stablehlo_available, torch_xla_available


FRONTEND_AVAILABLE = bool(
    importlib.util.find_spec("torch")
    and torch_xla_available()
    and official_stablehlo_available()
)


@unittest.skipUnless(FRONTEND_AVAILABLE, "requires PyTorch, Torch-XLA and official StableHLO")
class PyTorchFrontendTest(unittest.TestCase):
    def test_two_matmul_follows_the_only_frontend_route(self) -> None:
        import torch

        from examples.torch_models import TwoMatmul

        compiled = compile_torch_module(
            TwoMatmul(),
            (torch.randn(8, 8), torch.randn(8, 8), torch.randn(8, 8)),
            minimal_machine_config(),
            model_id="two-matmul",
            tile_size=4,
        )

        self.assertEqual(
            compiled.attributes["frontend_path"],
            "torch_export->torch_xla->official_stablehlo->canonical",
        )
        self.assertEqual(compiled.source_frontend.frontend.value, "torch.export")
        self.assertEqual(compiled.frontend.frontend.value, "stablehlo")
        self.assertEqual(compiled.stablehlo.producer, "torch-xla")
        self.assertTrue(compiled.stablehlo.verified)
        self.assertEqual([operator.op_type for operator in compiled.graph.operators], ["matmul", "matmul"])
        self.assertTrue(compiled.tisa_program.instructions)
        self.assertEqual(compiled.validate(), ())

    def test_attention_preserves_stablehlo_and_tisa_dependencies(self) -> None:
        import torch

        from examples.torch_models import AttentionMicrograph

        shape = (1, 4, 8)
        compiled = compile_torch_module(
            AttentionMicrograph(),
            tuple(torch.randn(*shape) for _ in range(3)),
            minimal_machine_config(),
            model_id="attention",
            tile_size=4,
        )

        self.assertIn("stablehlo.dot_general", compiled.stablehlo.text)
        self.assertIn("stablehlo.reduce", compiled.stablehlo.text)
        self.assertTrue(any(instruction.dependencies for instruction in compiled.tisa_program.instructions))
        primitives = {instruction.op_type for instruction in compiled.tisa_program.instructions}
        self.assertTrue({"matmul", "reduce_max", "exp", "normalize"}.issubset(primitives))

    def test_public_compiler_api_has_no_legacy_frontend_wrappers(self) -> None:
        import npu_ooo.compiler as compiler

        legacy = {
            "compile_frontend_import",
            "compile_model_instance",
            "compile_stablehlo_file",
            "compile_stablehlo_text",
            "compile_torch_module_through_stablehlo",
        }
        self.assertTrue(legacy.isdisjoint(compiler.__all__))


if __name__ == "__main__":
    unittest.main()
