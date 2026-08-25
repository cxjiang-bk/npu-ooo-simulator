from collections import Counter
from dataclasses import replace
import importlib.util
import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.compiler import compile_torch_module
from npu_ooo.frontend import official_stablehlo_available, torch_xla_available
from npu_ooo.scheduler import SchedulerPolicy, schedule_tisa_program


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
        self.assertEqual(
            compiled.attributes["compiler_stages"],
            ["framework_bridge", "graph_compiler", "fusion_compiler", "tisa_generator", "backend"],
        )
        self.assertEqual(compiled.source_frontend.frontend.value, "torch.export")
        self.assertEqual(compiled.frontend.frontend.value, "stablehlo")
        self.assertEqual(compiled.stablehlo.producer, "torch-xla")
        self.assertTrue(compiled.stablehlo.verified)
        self.assertEqual([operator.op_type for operator in compiled.graph.operators], ["matmul", "matmul"])
        dependency_counts = Counter(
            dependency.kind for dependency in compiled.tile_graph.dependencies
        )
        self.assertEqual(
            dependency_counts,
            {"region_data": 16, "accumulate": 8},
        )
        self.assertEqual(
            compiled.tile_graph.attributes["avoided_all_to_all_dependencies"],
            48,
        )
        self.assertTrue(compiled.tisa_program.instructions)
        address_expressions = [
            operand.tile_mem.address_expr
            for instruction in compiled.tisa_program.instructions
            for operand in instruction.operands
        ]
        self.assertTrue(address_expressions)
        self.assertTrue(all(expression for expression in address_expressions))
        stride_metadata = [
            operand.tile_mem
            for instruction in compiled.tisa_program.instructions
            for operand in instruction.operands
        ]
        self.assertTrue(all(item.strides_bytes for item in stride_metadata))
        self.assertTrue(all(item.stride_expr for item in stride_metadata))
        self.assertTrue(all(item.layout == "dense" for item in stride_metadata))
        self.assertIsNotNone(compiled.gc_artifact)
        self.assertIsNotNone(compiled.tisa_dialect)
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
        self.assertTrue({"matmul", "softmax"}.issubset(primitives))
        payload_primitives = {
            primitive
            for instruction in compiled.tisa_program.instructions
            for primitive in instruction.attributes.get("payload_primitives", ())
        }
        self.assertTrue(
            {"reduce_max", "exp", "reduce_sum", "normalize"}.issubset(payload_primitives)
        )

    def test_silu_uses_stablehlo_logistic_capability(self) -> None:
        import torch

        class SiLU(torch.nn.Module):
            def forward(self, value):
                return torch.nn.functional.silu(value)

        compiled = compile_torch_module(
            SiLU(),
            (torch.randn(1, 4, 8),),
            minimal_machine_config(),
            model_id="silu",
            tile_size=4,
        )

        self.assertIn(
            "stablehlo.logistic",
            {operator.attributes.get("frontend_target") for operator in compiled.graph.operators},
        )
        self.assertEqual(compiled.validate(), ())

    def test_attention_online_configuration_survives_softmax_recovery(self) -> None:
        import torch

        from examples.torch_models import AttentionMicrograph

        shape = (1, 4, 8)
        machine = replace(
            minimal_machine_config(),
            attributes={"softmax_algorithm": "online", "calibration_status": "analytical"},
        )
        compiled = compile_torch_module(
            AttentionMicrograph(),
            tuple(torch.randn(*shape) for _ in range(3)),
            machine,
            model_id="attention-online",
            tile_size=4,
        )

        softmax = [
            instruction
            for instruction in compiled.tisa_program.instructions
            if instruction.op_type == "softmax"
        ]
        self.assertTrue(softmax)
        self.assertTrue(
            all(item.attributes["softmax_algorithm"] == "online" for item in softmax)
        )
        self.assertTrue(
            all(item.attributes["payload_primitives"] == ["online_update"] for item in softmax)
        )
        self.assertIn(
            "online_update",
            {task.primitive for task in compiled.backend_artifact.execution_graph.tasks},
        )
        self.assertEqual(compiled.validate(), ())

    def test_multi_head_attention_compiles_transforms_and_shares_static_dynamic_artifact(self) -> None:
        import torch

        from examples.torch_models import MultiHeadAttentionBlock

        compiled = compile_torch_module(
            MultiHeadAttentionBlock(),
            (torch.randn(1, 4, 8), torch.zeros(1, 1, 4, 4)),
            minimal_machine_config(),
            model_id="multi-head-attention",
            tile_size=4,
        )

        operator_types = {operator.op_type for operator in compiled.graph.operators}
        self.assertTrue(
            {"reshape", "transpose", "batched_matmul", "softmax"}.issubset(operator_types)
        )
        primitives = {task.primitive for task in compiled.backend_artifact.execution_graph.tasks}
        self.assertTrue({"copy", "transpose"}.issubset(primitives))
        self.assertEqual(
            compiled.tile_graph.attributes["dependency_model"],
            "logical_tensor_region_v1",
        )
        self.assertGreater(
            compiled.tile_graph.attributes["avoided_all_to_all_dependencies"],
            0,
        )
        statistics = compiled.attributes["compile_statistics"]
        self.assertEqual(statistics["summary"]["tile_count"], len(compiled.tile_graph.tiles))
        self.assertGreater(statistics["summary"]["macs"], 0)

        artifact_before = compiled.backend_artifact.to_dict()
        static_result = schedule_tisa_program(
            compiled.backend_artifact,
            minimal_machine_config(),
            SchedulerPolicy.STATIC_PIPELINE,
        )
        dynamic_result = schedule_tisa_program(
            compiled.backend_artifact,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )
        self.assertEqual(compiled.backend_artifact.to_dict(), artifact_before)
        self.assertGreater(static_result.total_cycles, 0)
        self.assertGreater(dynamic_result.total_cycles, 0)

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
