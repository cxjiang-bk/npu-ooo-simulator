import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.cli import build_parser, main
from npu_ooo.frontend import official_stablehlo_available, torch_xla_available
from npu_ooo.ir import (
    AccessType,
    BackendArtifact,
    BufferRegion,
    ExecutionGraph,
    ExecutionTask,
    OperatorGraph,
    OperatorSpec,
    TensorSpec,
    TISAInstruction,
    TISAOperand,
    TISAProgram,
    TileMem,
    UnitMap,
)
from npu_ooo.trace import ensure_output_layout, write_artifact_json


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
            {
                "compile",
                "compile-and-sim",
                "simulate",
                "paper-matrix",
                "import-rtl-trace",
                "import-rtl-log",
            },
        )

    def test_removed_legacy_command_is_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["compile-model"])

    def test_compile_and_sim_accepts_online_softmax_configuration(self) -> None:
        args = build_parser().parse_args(
            [
                "compile-and-sim",
                "--torch-module",
                "examples.torch_models:AttentionMicrograph",
                "--input-shape",
                "1,4,8",
                "--softmax-algorithm",
                "online",
            ]
        )
        self.assertEqual(args.softmax_algorithm, "online")

    def test_compile_and_sim_accepts_runtime_sequence_configuration(self) -> None:
        args = build_parser().parse_args(
            [
                "compile-and-sim",
                "--torch-module",
                "examples.torch_models:AttentionMicrograph",
                "--input-shape",
                "1,4,8",
                "--runtime-invocations",
                "3",
                "--runtime-inter-invocation-gap",
                "5",
            ]
        )
        self.assertEqual(args.runtime_invocations, 3)
        self.assertEqual(args.runtime_inter_invocation_gap, 5)

    def test_simulate_accepts_compile_package_and_runtime_manifest(self) -> None:
        args = build_parser().parse_args(
            [
                "simulate",
                "--compile-dir",
                "out/attention-compile",
                "--runtime-config",
                "runtime.json",
                "--policy",
                "dynamic_ready_queue",
            ]
        )
        self.assertEqual(args.compile_dir, Path("out/attention-compile"))
        self.assertEqual(args.runtime_config, Path("runtime.json"))
        self.assertEqual(args.policy, "dynamic_ready_queue")

    def test_compile_only_exposes_compiler_options(self) -> None:
        args = build_parser().parse_args(
            [
                "compile",
                "--torch-module",
                "examples.torch_models:AttentionMicrograph",
                "--input-shape",
                "1,4,8",
            ]
        )
        self.assertEqual(args.codegen_backend, "analytical")
        self.assertFalse(hasattr(args, "policy"))
        self.assertFalse(hasattr(args, "runtime_policy"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "compile",
                    "--torch-module",
                    "examples.torch_models:AttentionMicrograph",
                    "--input-shape",
                    "1,4,8",
                    "--policy",
                    "dynamic_ready_queue",
                ]
            )

    def test_paper_matrix_accepts_registry_and_variant_configuration(self) -> None:
        args = build_parser().parse_args(
            [
                "paper-matrix",
                "--benchmarks",
                "bert-base,gpt-j-6b-oneblk",
                "--variant",
                "micro",
                "--runtime-device-matrix",
            ]
        )
        self.assertEqual(args.benchmarks, "bert-base,gpt-j-6b-oneblk")
        self.assertEqual(args.variant, "micro")
        self.assertTrue(args.runtime_device_matrix)


class CompileAndSimCliTest(unittest.TestCase):
    @staticmethod
    def _write_minimal_compile_package(root: Path) -> None:
        ensure_output_layout(root)
        graph = OperatorGraph(
            graph_id="cli.test.graph",
            tensors=(
                TensorSpec("input", (4,), dtype="fp16"),
                TensorSpec("output", (4,), dtype="fp16"),
            ),
            operators=(
                OperatorSpec(
                    op_id="copy",
                    op_type="elementwise",
                    inputs=("input",),
                    outputs=("output",),
                ),
            ),
            edges=(),
        )
        input_mem = TileMem(
            base="input",
            scope="logical",
            tensor="input",
            offset_bytes=0,
            size_bytes=8,
            logical_starts=(0,),
            logical_shape=(4,),
        )
        output_mem = TileMem(
            base="output",
            scope="logical",
            tensor="output",
            offset_bytes=0,
            size_bytes=8,
            logical_starts=(0,),
            logical_shape=(4,),
        )
        instruction = TISAInstruction(
            tisa_id="copy.t0000",
            tile_id="copy.t0000",
            operator_id="copy",
            op_type="elementwise",
            operands=(
                TISAOperand("input", (4,), input_mem, AccessType.READ),
                TISAOperand("output", (4,), output_mem, AccessType.WRITE),
            ),
            unit_map=UnitMap("ARU"),
            payload_ref="payload:copy.t0000",
        )
        task = ExecutionTask(
            task_id="copy.task",
            tile_id="copy.t0000",
            operator_id="copy",
            primitive="elementwise",
            resource="ARU",
            reads=(
                BufferRegion("input", "DRAM", (4,), (0,), access=AccessType.READ, size_bytes=8),
            ),
            writes=(
                BufferRegion("output", "DRAM", (4,), (0,), access=AccessType.WRITE, size_bytes=8),
            ),
            duration_cycles=4,
        )
        backend = BackendArtifact(
            artifact_id="cli.test.backend",
            program=TISAProgram("cli.test.program", (instruction,)),
            execution_graph=ExecutionGraph("cli.test.execution", (task,)),
            payloads={"copy.t0000": ("copy.task",)},
        )
        write_artifact_json(graph, root / "01_gc" / "canonical_graph.json")
        write_artifact_json(backend, root / "04_backend" / "backend_artifact.json")
        write_artifact_json(minimal_machine_config(), root / "04_backend" / "machine.json")
        write_artifact_json(
            {
                "schema_version": 1,
                "artifact_kind": "compile_package",
                "artifact_id": backend.artifact_id,
                "compile_artifact_id": backend.artifact_id,
            },
            root / "manifest.json",
        )

    def test_simulate_runs_from_json_compile_package_without_frontend(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            compile_dir = Path(directory) / "compile"
            output = Path(directory) / "simulation"
            self._write_minimal_compile_package(compile_dir)

            exit_code = main(
                [
                    "simulate",
                    "--compile-dir",
                    str(compile_dir),
                    "--policy",
                    "dynamic_ready_queue",
                    "--output-dir",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "06_simulation" / "summary.json").exists())
            self.assertTrue((output / "07_trace" / "swimlane.svg").exists())
            self.assertTrue((output / "05_runtime" / "runtime_submission.json").exists())
            self.assertFalse((output / "00_frontend" / "generated.mlir").exists())
            self.assertFalse((output / "04_backend" / "backend_artifact.json").exists())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_kind"], "simulation_result")
            self.assertEqual(manifest["compile_artifact_id"], "cli.test.backend")
            summary = json.loads(
                (output / "06_simulation" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertGreater(summary["total_cycles"], 0)

    def test_simulate_applies_runtime_manifest_and_machine_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            compile_dir = root / "compile"
            runtime_config = root / "runtime.json"
            output = root / "simulation"
            self._write_minimal_compile_package(compile_dir)
            runtime_config.write_text(
                json.dumps(
                    {
                        "runtime_policy": "dynamic_ready_queue",
                        "runtime_launch_latency": 3,
                        "dynamic_layouts": {
                            "input": {
                                "shape": [4],
                                "strides_bytes": [2],
                                "layout": "runtime_contiguous",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "simulate",
                    "--compile-dir",
                    str(compile_dir),
                    "--runtime-config",
                    str(runtime_config),
                    "--arch",
                    "wide-mxu",
                    "--policy",
                    "dynamic_ready_queue",
                    "--output-dir",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["architecture"], "wide-mxu")
            self.assertEqual(manifest["runtime_policy"], "dynamic_ready_queue")
            self.assertEqual(manifest["dynamic_layout_binding_count"], 1)
            self.assertEqual(manifest["compile_artifact_id"], "cli.test.backend")
            runtime_submission = json.loads(
                (output / "05_runtime" / "runtime_submission.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(runtime_submission["dynamic_layouts"][0]["tensor"], "input")
            self.assertEqual(runtime_submission["launch_latency_cycles"], 3)

    @unittest.skipUnless(FRONTEND_AVAILABLE, "requires PyTorch, Torch-XLA and official StableHLO")
    def test_pytorch_module_exports_staged_artifacts_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                [
                    "compile-and-sim",
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
                "01_gc",
                "02_fc",
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
            self.assertTrue((output / "01_gc" / "operator_graph.svg").exists())
            self.assertTrue((output / "01_gc" / "gc_artifact.json").exists())
            self.assertTrue((output / "01_gc" / "compile_statistics.json").exists())
            self.assertTrue((output / "02_fc" / "tisa_dialect.json").exists())
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

    @unittest.skipUnless(FRONTEND_AVAILABLE, "requires PyTorch, Torch-XLA and official StableHLO")
    def test_paper_matrix_writes_case_summaries_without_flat_stage_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            exit_code = main(
                [
                    "paper-matrix",
                    "--benchmarks",
                    "bert-base",
                    "--tile-size",
                    "4",
                    "--output-dir",
                    str(output),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "README.md",
                    "matrix_manifest.json",
                    "matrix_index.json",
                    "sweep.csv",
                    "sweep.json",
                    "bert-base",
                },
            )
            case_dir = output / "bert-base" / "micro"
            self.assertTrue((case_dir / "summary.json").exists())
            self.assertTrue((case_dir / "manifest.json").exists())
            self.assertTrue((case_dir / "artifact_index.json").exists())
            self.assertTrue((case_dir / "00_frontend" / "generated.mlir").exists())
            self.assertTrue(
                (case_dir / "policy_matrix" / "runtime-static__device-static_pipeline" / "06_simulation" / "summary.json").exists()
            )
            self.assertTrue(
                (case_dir / "policy_matrix" / "runtime-static__device-static_pipeline" / "07_trace" / "swimlane.svg").exists()
            )
            self.assertFalse((output / "00_frontend").exists())
            records = json.loads((output / "sweep.json").read_text(encoding="utf-8"))
            self.assertEqual(len(records), 2)
            self.assertEqual({record["benchmark_id"] for record in records}, {"bert-base"})
            self.assertEqual({record["artifact_id"] for record in records}, {records[0]["artifact_id"]})


if __name__ == "__main__":
    unittest.main()
