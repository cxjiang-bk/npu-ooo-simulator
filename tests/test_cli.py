import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from npu_ooo.arch import minimal_machine_config
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
                "swimlane.png",
                "tasks.csv",
                "tile_graph.dot",
                "tile_graph.json",
            }
            self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))
            self.assertTrue((output / "00_frontend" / "model_spec.json").exists())
            self.assertTrue((output / "01_graph_ir" / "operator_graph.json").exists())
            self.assertTrue((output / "02_schedule_tile" / "tile_graph.json").exists())
            self.assertTrue((output / "04_backend" / "execution_graph.json").exists())
            self.assertTrue((output / "06_simulation" / "summary.json").exists())
            self.assertTrue((output / "07_trace" / "swimlane.svg").exists())
            artifact_index = json.loads((output / "artifact_index.json").read_text())
            self.assertEqual(artifact_index["layout"], "staged")
            self.assertIn("01_graph_ir", {item["directory"] for item in artifact_index["stages"]})
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
            tasks_header = (output / "tasks.csv").read_text().splitlines()[0]
            self.assertIn("tile_id", tasks_header)
            self.assertIn("operator_id", tasks_header)
            summary = json.loads((output / "summary.json").read_text())
            self.assertIn("resource_utilization", summary["metrics"])
            self.assertIn("queue_peak_occupancy", summary["metrics"])

    def test_two_mm_sweep_exports_case_manifests_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                [
                    "sweep-two-mm",
                    "--architectures",
                    "minimal",
                    "--policies",
                    "static_pipeline,dynamic_ready_queue",
                    "--windows",
                    "1",
                    "--robs",
                    "1",
                    "--tile-sizes",
                    "16,32",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            sweep = json.loads((output / "sweep.json").read_text())
            self.assertEqual(len(sweep), 4)
            self.assertEqual({row["policy"] for row in sweep}, {"static_pipeline", "dynamic_ready_queue"})
            self.assertEqual({row["tile_size"] for row in sweep}, {16, 32})
            for row in sweep:
                case_dir = output / row["case_id"]
                self.assertTrue((case_dir / "manifest.json").exists())
                self.assertTrue((case_dir / "summary.json").exists())
                self.assertTrue((case_dir / "swimlane.svg").exists())
                self.assertTrue((case_dir / "00_frontend" / "model_spec.json").exists())
                self.assertTrue((case_dir / "01_graph_ir" / "operator_graph.json").exists())
                self.assertTrue((case_dir / "02_schedule_tile" / "tile_graph.json").exists())
                self.assertTrue((case_dir / "04_backend" / "execution_graph.json").exists())
                self.assertTrue((case_dir / "07_trace" / "perfetto.json").exists())
            self.assertTrue((output / "sweep.csv").exists())

    def test_elementwise_exports_aru_execution_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                [
                    "elementwise",
                    "--arch",
                    "minimal",
                    "--policy",
                    "dynamic_ready_queue",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            execution_graph = json.loads((output / "execution_graph.json").read_text())
            self.assertTrue(any(task["primitive"] == "elementwise" for task in execution_graph["tasks"]))
            self.assertTrue(any(task["resource"] == "ARU" for task in execution_graph["tasks"]))
            summary = json.loads((output / "summary.json").read_text())
            self.assertIn("ARU", summary["metrics"]["resource_utilization"])

    def test_reduce_exports_partial_accumulation_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                ["reduce", "--arch", "minimal", "--policy", "dynamic_ready_queue", "--output-dir", str(output)]
            )
            self.assertEqual(exit_code, 0)
            execution_graph = json.loads((output / "execution_graph.json").read_text())
            reduce_tasks = [task for task in execution_graph["tasks"] if task["primitive"] == "reduce"]
            self.assertTrue(reduce_tasks)
            self.assertTrue(any(task["resource"] == "ARU" for task in reduce_tasks))

    def test_softmax_exports_composite_primitive_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                ["softmax", "--arch", "minimal", "--policy", "dynamic_ready_queue", "--output-dir", str(output)]
            )
            self.assertEqual(exit_code, 0)
            execution_graph = json.loads((output / "execution_graph.json").read_text())
            primitives = {task["primitive"] for task in execution_graph["tasks"]}
            self.assertTrue({"reduce_max", "exp", "reduce_sum", "normalize"}.issubset(primitives))

    def test_rmsnorm_exports_sum_square_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                ["rmsnorm", "--arch", "minimal", "--policy", "dynamic_ready_queue", "--output-dir", str(output)]
            )
            self.assertEqual(exit_code, 0)
            execution_graph = json.loads((output / "execution_graph.json").read_text())
            primitives = {task["primitive"] for task in execution_graph["tasks"]}
            self.assertTrue({"square", "reduce_sum_square", "rmsnorm"}.issubset(primitives))

    def test_decoder_block_exports_mixed_operator_and_execution_graphs(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                [
                    "decoder-block",
                    "--arch",
                    "minimal",
                    "--policy",
                    "dynamic_ready_queue",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            operator_graph = json.loads((output / "operator_graph.json").read_text())
            self.assertEqual(
                [operator["op_type"] for operator in operator_graph["operators"]],
                ["rmsnorm", "matmul", "residual_add"],
            )
            execution_graph = json.loads((output / "execution_graph.json").read_text())
            primitives = {task["primitive"] for task in execution_graph["tasks"]}
            self.assertTrue({"rmsnorm", "matmul", "elementwise"}.issubset(primitives))
            self.assertGreater(
                execution_graph["attributes"]["cross_operator_dependency_count"],
                0,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["calibration_status"], "analytical")
            self.assertEqual((output / "swimlane.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_layernorm_exports_two_reduction_barriers(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                [
                    "layernorm",
                    "--arch",
                    "minimal",
                    "--policy",
                    "dynamic_ready_queue",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            execution_graph = json.loads((output / "execution_graph.json").read_text())
            primitives = {task["primitive"] for task in execution_graph["tasks"]}
            self.assertTrue({"reduce_sum", "layernorm_mean", "center", "reduce_sum_square", "layernorm"}.issubset(primitives))

    def test_attention_exports_qk_softmax_pv_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                [
                    "attention",
                    "--arch",
                    "minimal",
                    "--policy",
                    "dynamic_ready_queue",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            operator_graph = json.loads((output / "operator_graph.json").read_text())
            self.assertEqual(
                [operator["op_id"] for operator in operator_graph["operators"]],
                ["attention_scores", "attention_softmax", "attention_context"],
            )
            execution_graph = json.loads((output / "execution_graph.json").read_text())
            primitives = {task["primitive"] for task in execution_graph["tasks"]}
            self.assertTrue({"matmul", "reduce_max", "normalize"}.issubset(primitives))

    def test_transformer_block_exports_nine_operator_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                [
                    "transformer-block",
                    "--arch",
                    "minimal",
                    "--policy",
                    "dynamic_ready_queue",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            operator_graph = json.loads((output / "operator_graph.json").read_text())
            self.assertEqual(len(operator_graph["operators"]), 9)
            execution_graph = json.loads((output / "execution_graph.json").read_text())
            self.assertGreater(execution_graph["attributes"]["cross_operator_dependency_count"], 0)

    def test_model_block_exports_named_proxy_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                [
                    "model-block",
                    "--model-preset",
                    "llama2-7b",
                    "--tokens",
                    "16",
                    "--sequence",
                    "16",
                    "--head-dim",
                    "16",
                    "--intermediate",
                    "32",
                    "--policy",
                    "dynamic_ready_queue",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            model = json.loads((output / "model_spec.json").read_text())
            case = json.loads((output / "benchmark_case.json").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(model["model_id"], "llama2_7b")
            self.assertEqual(model["attributes"]["benchmark_status"], "proxy")
            self.assertEqual(model["attributes"]["proxy_shape"]["intermediate"], 32)
            self.assertEqual(case["phase"], "prefill")
            self.assertEqual(case["model_id"], "llama2_7b")
            self.assertGreater(manifest["total_cycles"], 0)

    def test_workload_sweep_keeps_graph_artifacts_per_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                [
                    "sweep-workloads",
                    "--workloads",
                    "elementwise,layernorm,decoder-block",
                    "--architectures",
                    "minimal",
                    "--policies",
                    "static_pipeline,dynamic_ready_queue",
                    "--windows",
                    "1",
                    "--robs",
                    "1",
                    "--tile-sizes",
                    "16",
                    "--dynamic-priorities",
                    "critical_path",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            sweep = json.loads((output / "sweep.json").read_text())
            self.assertEqual(len(sweep), 6)
            self.assertEqual({row["workload"] for row in sweep}, {"elementwise", "layernorm", "decoder-block"})
            for row in sweep:
                case_dir = output / (
                    f"{row['workload']}__{row['architecture']}__{row['policy']}"
                    f"__tile{row['tile_size']}__window{row['dependency_window']}__rob{row['rob_entries']}"
                    f"__priority{row['dynamic_priority']}"
                )
                self.assertTrue((case_dir / "operator_graph.json").exists())
                self.assertTrue((case_dir / "execution_graph.json").exists())
                self.assertTrue((case_dir / "swimlane.png").exists())

    def test_workload_sweep_accepts_model_preset_shape_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                [
                    "sweep-workloads",
                    "--workloads",
                    "gpt-j",
                    "--architectures",
                    "minimal",
                    "--policies",
                    "static_pipeline,dynamic_ready_queue",
                    "--windows",
                    "1",
                    "--robs",
                    "1",
                    "--tile-sizes",
                    "8",
                    "--dynamic-priorities",
                    "critical_path",
                    "--model-tokens",
                    "8",
                    "--model-sequence",
                    "8",
                    "--model-head-dim",
                    "8",
                    "--model-intermediate",
                    "16",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            sweep = json.loads((output / "sweep.json").read_text())
            self.assertEqual(len(sweep), 2)
            self.assertEqual({row["workload"] for row in sweep}, {"gpt-j"})
            model = json.loads(next(output.glob("*/model_spec.json")).read_text())
            self.assertEqual(model["attributes"]["proxy_shape"]["head_dim"], 8)

    def test_cli_accepts_canonical_machine_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "machine.json"
            output = Path(directory) / "run"
            config_path.write_text(
                json.dumps(minimal_machine_config().to_dict()),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "elementwise",
                        "--machine-config",
                        str(config_path),
                        "--policy",
                        "dynamic_ready_queue",
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(exit_code, 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["machine_hash"], minimal_machine_config().stable_hash())

    def test_cli_accepts_timing_table_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            timing_path = root / "timing.json"
            timing_path.write_text(
                json.dumps(
                    {
                        "name": "calibrated_probe_v0",
                        "entries": {
                            "elementwise": {
                                "duration_cycles": 1,
                                "initiation_interval_cycles": 1,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "run"
            exit_code = main(
                [
                    "elementwise",
                    "--timing-config",
                    str(timing_path),
                    "--policy",
                    "dynamic_ready_queue",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            manifest = json.loads((output / "manifest.json").read_text())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(manifest["backend"], "calibrated_probe_v0")
            self.assertEqual(summary["backend"], "calibrated_probe_v0")

    def test_workload_sweep_accepts_custom_architecture_label_with_machine_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            config_path = root / "machine.json"
            config_path.write_text(json.dumps(minimal_machine_config().to_dict()), encoding="utf-8")
            output = root / "sweep"
            exit_code = main(
                [
                    "sweep-workloads",
                    "--workloads",
                    "elementwise",
                    "--architectures",
                    "custom-profile",
                    "--machine-config",
                    str(config_path),
                    "--policies",
                    "static_pipeline",
                    "--windows",
                    "1",
                    "--robs",
                    "1",
                    "--tile-sizes",
                    "16",
                    "--dynamic-priorities",
                    "critical_path",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            sweep = json.loads((output / "sweep.json").read_text())
            self.assertEqual(sweep[0]["architecture"], "custom-profile")
            self.assertEqual(sweep[0]["total_cycles"], 6144.0)


if __name__ == "__main__":
    unittest.main()
