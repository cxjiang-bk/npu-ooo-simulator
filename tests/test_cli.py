import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from npu_ooo.cli import main


class CliArtifactTest(unittest.TestCase):
    def test_two_mm_exports_compilation_graphs_and_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                [
                    "two-mm",
                    "--arch",
                    "minimal",
                    "--policy",
                    "dynamic_ready_queue",
                    "--dependency-window",
                    "4",
                    "--rob-entries",
                    "4",
                    "--address-scoreboard",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            expected = {
                "benchmark_case.json",
                "address_dependencies.json",
                "execution_graph.dot",
                "execution_graph.json",
                "machine.json",
                "manifest.json",
                "model_instance.json",
                "model_spec.json",
                "operator_graph.dot",
                "operator_graph.json",
                "operator_graph.svg",
                "perfetto.json",
                "schedule.json",
                "summary.json",
                "swimlane.svg",
                "tasks.csv",
                "tile_graph.dot",
                "tile_graph.json",
            }
            self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))
            operator_graph = json.loads((output / "operator_graph.json").read_text())
            execution_graph = json.loads((output / "execution_graph.json").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual([operator["op_id"] for operator in operator_graph["operators"]], ["gemm0", "gemm1"])
            self.assertEqual(len(execution_graph["tasks"]), 204)
            self.assertEqual(manifest["calibration_status"], "analytical")
            self.assertEqual(manifest["backend"], "analytical")
            self.assertEqual(manifest["simulator_config"]["dependency_window"], 4)
            self.assertEqual(manifest["simulator_config"]["rob_entries"], 4)
            self.assertTrue(manifest["simulator_config"]["address_scoreboard"])
            self.assertEqual(manifest["address_dependency_count"], len(json.loads((output / "address_dependencies.json").read_text())))
            self.assertIn("gemm0", (output / "operator_graph.svg").read_text())


if __name__ == "__main__":
    unittest.main()
