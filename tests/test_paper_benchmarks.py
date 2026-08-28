import importlib.util
import unittest

from examples.paper_benchmarks import build_paper_benchmark, paper_benchmark_specs
from npu_ooo.arch import minimal_machine_config
from npu_ooo.compiler import compile_torch_module
from npu_ooo.frontend import official_stablehlo_available, torch_xla_available


FRONTEND_AVAILABLE = bool(
    importlib.util.find_spec("torch")
    and torch_xla_available()
    and official_stablehlo_available()
)


class PaperBenchmarkRegistryTest(unittest.TestCase):
    def test_table_ix_registry_contains_six_rows_in_paper_order(self) -> None:
        specs = paper_benchmark_specs()
        self.assertEqual(
            [spec.case_id for spec in specs],
            [
                "resnet50",
                "bert-base",
                "gpt-j-6b-oneblk",
                "llama2-13b-oneblk",
                "deepseek-r1-16b-prefill",
                "deepseek-r1-16b-decode",
            ],
        )
        self.assertEqual(specs[0].reference_a100_ms, 9.3)
        self.assertEqual(specs[-1].phase, "decode")

    def test_each_row_builds_an_independent_real_pytorch_workload(self) -> None:
        for spec in paper_benchmark_specs():
            workload = build_paper_benchmark(spec.case_id, variant="micro")
            self.assertEqual(workload.spec, spec)
            self.assertTrue(workload.module.training is False)
            self.assertTrue(workload.inputs)
            self.assertTrue(all(value.numel() > 0 for value in workload.inputs))


@unittest.skipUnless(FRONTEND_AVAILABLE, "requires PyTorch, Torch-XLA and official StableHLO")
class PaperBenchmarkFrontendTest(unittest.TestCase):
    def test_transformer_rows_reach_tisa(self) -> None:
        for case_id in ("bert-base", "gpt-j-6b-oneblk", "llama2-13b-oneblk"):
            workload = build_paper_benchmark(case_id, variant="micro")
            compiled = compile_torch_module(
                workload.module,
                workload.inputs,
                minimal_machine_config(),
                model_id=case_id,
                tile_size=4,
            )
            self.assertTrue(compiled.tisa_program.instructions)
            self.assertEqual(compiled.validate(), ())

    def test_resnet_reports_missing_convolution_capability(self) -> None:
        workload = build_paper_benchmark("resnet50", variant="micro")
        with self.assertRaisesRegex(ValueError, "missing StableHLO capability.*stablehlo.convolution"):
            compile_torch_module(
                workload.module,
                workload.inputs,
                minimal_machine_config(),
                model_id="resnet50",
                tile_size=4,
            )


if __name__ == "__main__":
    unittest.main()
