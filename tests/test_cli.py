import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from npu_ooo.cli import build_parser, main
from npu_ooo.frontend import official_stablehlo_available, torch_xla_available


FRONTEND_AVAILABLE = bool(
    importlib.util.find_spec("torch")
    and torch_xla_available()
    and official_stablehlo_available()
)


class CliSurfaceTest(unittest.TestCase):
    def test_only_current_commands_are_registered(self) -> None:
        parser = build_parser()
        subparsers = next(
            action for action in parser._actions if hasattr(action, "choices") and action.choices
        )
        self.assertEqual(
            set(subparsers.choices),
            {"compile-model", "import-rtl-trace", "import-rtl-log"},
        )

    def test_removed_benchmark_command_is_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["attention"])


@unittest.skipUnless(FRONTEND_AVAILABLE, "requires PyTorch, Torch-XLA and official StableHLO")
class CompileModelCliTest(unittest.TestCase):
    def test_pytorch_module_exports_staged_artifacts_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                [
                    "compile-model",
                    "--torch-module",
                    "examples.torch_models:TwoMatmul",
                    "--input-shape",
                    "4,4",
                    "--input-shape",
                    "4,4",
                    "--input-shape",
                    "4,4",
                    "--tile-size",
                    "4",
                    "--policy",
                    "dynamic_ready_queue",
                    "--output-dir",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            expected_root = {
                "00_frontend",
                "01_graph_ir",
                "02_schedule_tile",
                "03_tisa",
                "04_backend",
                "05_runtime",
                "06_simulation",
                "07_trace",
                "README.md",
                "artifact_index.json",
                "manifest.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected_root)
            self.assertTrue((output / "00_frontend" / "generated.mlir").exists())
            self.assertTrue((output / "01_graph_ir" / "operator_graph.svg").exists())
            self.assertTrue((output / "02_schedule_tile" / "compile_statistics.json").exists())
            self.assertTrue((output / "03_tisa" / "tisa_program.json").exists())
            self.assertTrue((output / "07_trace" / "swimlane.svg").exists())

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["frontend_path"],
                "torch_export->torch_xla->official_stablehlo->canonical",
            )
            self.assertEqual(manifest["stablehlo_exporter"], "torch-xla")
            self.assertTrue(manifest["stablehlo_verified"])
            self.assertEqual(manifest["scheduler_target"], "tisa")
            self.assertGreater(manifest["tisa_instruction_count"], 0)


if __name__ == "__main__":
    unittest.main()
