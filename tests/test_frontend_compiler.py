import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path

from npu_ooo.arch import minimal_machine_config
from npu_ooo.benchmarks import (
    build_elementwise_case,
    build_elementwise_model,
    build_two_matmul_case,
    build_two_matmul_model,
)
from npu_ooo.compiler import compile_frontend_import, compile_model_instance
from npu_ooo.cli import main
from npu_ooo.frontend import FrontendImportError, JsonGraphAdapter, TorchExportAdapter, import_operator_graph


class FrontendCompilerTest(unittest.TestCase):
    def test_json_adapter_accepts_canonical_graph_and_preserves_provenance(self) -> None:
        model = build_elementwise_model(rows=16, cols=16)
        instance = model.instantiate(build_elementwise_case())
        imported = JsonGraphAdapter.from_payload(
            {"model_id": "json_add", "graph": instance.graph.to_dict()},
            variant="json-v1",
        )
        self.assertEqual(imported.frontend.value, "json")
        self.assertEqual(imported.model_id, "json_add")
        self.assertEqual(imported.graph.graph_id, instance.graph.graph_id)
        self.assertEqual(imported.validate(), ())

    def test_compiler_emits_eu_bound_tisa_groups_per_semantic_tile(self) -> None:
        model = build_two_matmul_model()
        instance = model.instantiate(build_two_matmul_case())
        compiled = compile_frontend_import(
            import_operator_graph(instance.graph, model_id=model.model_id),
            minimal_machine_config(),
            tile_size=32,
        )
        self.assertGreaterEqual(len(compiled.tisa_program.instructions), len(compiled.tile_graph.tiles))
        self.assertEqual(
            sum(len(value) for value in compiled.backend_artifact.payloads.values()),
            len(compiled.backend_artifact.execution_graph.tasks),
        )
        self.assertEqual(compiled.validate(), ())
        self.assertTrue(any(item.dependencies for item in compiled.tisa_program.instructions))
        self.assertEqual({item.op_type for item in compiled.tisa_program.instructions}, {"load", "matmul", "store"})
        self.assertEqual(
            {item.unit_map.unit for item in compiled.tisa_program.instructions},
            {"dma", "tensor"},
        )

    def test_compile_frontend_resolves_shape_symbols(self) -> None:
        model = build_elementwise_model(rows=32, cols=32)
        imported = JsonGraphAdapter.from_payload(
            {
                "model_id": "symbolic_add",
                "shape_environment": {"M": 8, "N": 12},
                "graph": model.templates[0].graph.to_dict(),
            }
        )
        compiled = compile_frontend_import(imported, minimal_machine_config(), tile_size=8)
        tensor = next(item for item in compiled.graph.tensors if item.name == "C")
        self.assertEqual(tensor.shape, (8, 12))

    def test_torch_adapter_reports_missing_dependency_at_frontend_boundary(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            with self.assertRaises(FrontendImportError) as context:
                TorchExportAdapter.export_module(object())
            self.assertIn("requires 'torch'", str(context.exception))

    def test_model_instance_uses_the_same_compiler_entry(self) -> None:
        model = build_elementwise_model(rows=16, cols=16)
        instance = model.instantiate(build_elementwise_case())
        compiled = compile_model_instance(instance, minimal_machine_config(), tile_size=8)
        self.assertEqual(compiled.frontend.model_id, instance.model_id)
        self.assertEqual(compiled.frontend.provenance["case_id"], instance.case_id)
        self.assertEqual(compiled.validate(), ())

    def test_compile_model_cli_exports_frontend_to_backend_artifacts(self) -> None:
        model = build_elementwise_model(rows=8, cols=8)
        instance = model.instantiate(build_elementwise_case())
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            graph_path = root / "graph.json"
            output = root / "out"
            graph_path.write_text(json.dumps(instance.graph.to_dict()), encoding="utf-8")
            exit_code = main(
                [
                    "compile-model",
                    "--graph-json",
                    str(graph_path),
                    "--arch",
                    "minimal",
                    "--policy",
                    "dynamic_ready_queue",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "frontend_import.json").exists())
            self.assertTrue((output / "tisa_program.json").exists())
            self.assertTrue((output / "backend_artifact.json").exists())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["compiler_pipeline"], "frontend->canonical->schedule->tile->tisa->analytical-backend")


if __name__ == "__main__":
    unittest.main()
