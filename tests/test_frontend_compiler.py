import unittest
import contextlib
import io
import importlib.util
import json
import tempfile
from unittest import mock
from pathlib import Path

from npu_ooo.arch import minimal_machine_config
from npu_ooo.benchmarks import (
    build_elementwise_case,
    build_elementwise_model,
    build_two_matmul_case,
    build_two_matmul_model,
)
from npu_ooo.compiler import (
    PassManager,
    compile_frontend_import,
    compile_model_instance,
    compile_stablehlo_text,
    compile_torch_module,
    compile_torch_module_through_stablehlo,
)
from npu_ooo.compiler.planner import default_schedule_planner
from npu_ooo.compiler.tisa_first import TISASemanticBuilder
from npu_ooo.cli import main
from npu_ooo.frontend import (
    FrontendImportError,
    JsonGraphAdapter,
    OfficialStableHLOAdapter,
    StableHLOAdapter,
    TorchExportAdapter,
    import_operator_graph,
    official_stablehlo_available,
    torch_xla_available,
)
from npu_ooo.ir import DataEdge, OperatorGraph, OperatorSpec, TensorSpec, build_tile_graph
from npu_ooo.simulator import simulate_execution_graph


class _FakeValue:
    def __init__(self, shape, dtype="torch.float16"):
        self.shape = tuple(shape)
        self.dtype = dtype


class _FakeNode:
    def __init__(self, name, op, *, target=None, args=(), kwargs=None, shape=(4, 8)):
        self.name = name
        self.op = op
        self.target = target or name
        self.args = args
        self.kwargs = kwargs or {}
        self.meta = {"val": _FakeValue(shape)}


class _FakeGraph:
    def __init__(self, nodes):
        self.nodes = tuple(nodes)


class _FakeGraphModule:
    def __init__(self, graph):
        self.graph = graph


class _FakeExportedProgram:
    def __init__(self, graph):
        self.graph_module = _FakeGraphModule(graph)


class FrontendCompilerTest(unittest.TestCase):
    @unittest.skipUnless(official_stablehlo_available(), "requires official StableHLO bindings")
    def test_official_stablehlo_parser_verifies_fixture(self) -> None:
        text = Path("examples/stablehlo/matmul.mlir").read_text(encoding="utf-8")
        module = OfficialStableHLOAdapter.parse_text(text, model_id="official-matmul")
        self.assertTrue(module.verified)
        self.assertEqual(module.producer, "external-stablehlo")
        self.assertIsNotNone(module.stablehlo_version)
        self.assertIn("stablehlo.dot_general", module.canonical_text)

        compiled = compile_stablehlo_text(
            text,
            minimal_machine_config(),
            model_id="official-matmul",
            tile_size=4,
        )
        self.assertEqual([operator.normalized_type for operator in compiled.graph.operators], ["matmul"])
        self.assertEqual(compiled.validate(), ())

    @unittest.skipUnless(official_stablehlo_available(), "requires official StableHLO bindings")
    def test_official_stablehlo_rejects_consumed_secondary_result(self) -> None:
        text = """
        module {
          func.func @main(%x: tensor<1x4x8xf32>) -> tensor<4xf32> {
            %one = stablehlo.constant dense<1.0> : tensor<4xf32>
            %zero = stablehlo.constant dense<0.0> : tensor<4xf32>
            %output, %mean, %variance = "stablehlo.batch_norm_training"(%x, %one, %zero)
              <{epsilon = 1.0E-5 : f32, feature_index = 1 : i64}> :
              (tensor<1x4x8xf32>, tensor<4xf32>, tensor<4xf32>) ->
              (tensor<1x4x8xf32>, tensor<4xf32>, tensor<4xf32>)
            return %mean : tensor<4xf32>
          }
        }
        """
        with self.assertRaisesRegex(
            FrontendImportError,
            "returning a secondary operation result",
        ):
            OfficialStableHLOAdapter.import_text(text)

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
        self.assertEqual(compiled.schedule.attributes["source"], "automatic-planner")
        self.assertEqual(compiled.validate(), ())
        self.assertTrue(any(item.dependencies for item in compiled.tisa_program.instructions))
        self.assertEqual({item.op_type for item in compiled.tisa_program.instructions}, {"load", "matmul", "store"})
        self.assertEqual(
            {item.unit_map.unit for item in compiled.tisa_program.instructions},
            {"dma", "tensor"},
        )

    def test_tisa_semantic_builder_is_independent_of_backend_tasks(self) -> None:
        model = build_two_matmul_model()
        instance = model.instantiate(build_two_matmul_case())
        compiled = compile_frontend_import(
            import_operator_graph(instance.graph, model_id=model.model_id),
            minimal_machine_config(),
            tile_size=32,
        )
        schedule = default_schedule_planner().plan(compiled.graph, tile_size=32)
        tile_graph = build_tile_graph(compiled.graph, schedule)
        rebuilt = TISASemanticBuilder().build(
            compiled.graph,
            schedule,
            tile_graph,
            minimal_machine_config(),
            program_id="independent.tisa",
        )
        self.assertEqual(
            [item.to_dict() for item in rebuilt.instructions],
            [item.to_dict() for item in compiled.tisa_program.instructions],
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

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch for the real export smoke")
    def test_real_torch_rmsnorm_compiles_to_tisa(self) -> None:
        import torch

        class RMSNorm(torch.nn.Module):
            def __init__(self, eps: float = 1e-5):
                super().__init__()
                self.eps = eps

            def forward(self, x):
                return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

        compiled = compile_torch_module(
            RMSNorm().eval(),
            (torch.randn(4, 8, dtype=torch.float16),),
            minimal_machine_config(),
            model_id="real-torch-rmsnorm",
            tile_size=4,
        )
        self.assertEqual([operator.normalized_type for operator in compiled.graph.operators], ["rmsnorm"])
        self.assertEqual(len(compiled.tile_graph.tiles), 2)
        self.assertGreater(len(compiled.tisa_program.instructions), 0)
        self.assertEqual(compiled.validate(), ())

        stablehlo = """
        module {
          func.func @main(%x: tensor<4x8xf16>) -> tensor<4x8xf16> {
            %eps = stablehlo.constant dense<1.000000e-05> : tensor<f32>
            %0 = stablehlo.multiply %x, %x : tensor<4x8xf16>
            %1 = stablehlo.reduce %0, %eps dimensions = [1] : (tensor<4x8xf16>, tensor<f32>) -> tensor<4x1xf16>
            %2 = stablehlo.add %1, %eps : tensor<4x1xf16>
            %3 = stablehlo.rsqrt %2 : tensor<4x1xf16>
            %4 = stablehlo.multiply %x, %3 : tensor<4x8xf16>
            return %4 : tensor<4x8xf16>
          }
        }
        """
        stable_compiled = compile_stablehlo_text(
            stablehlo,
            minimal_machine_config(),
            model_id="stable-rmsnorm-equivalent",
            tile_size=4,
            stablehlo_backend="textual",
        )
        torch_operator = compiled.graph.operators[0]
        stable_operator = stable_compiled.graph.operators[0]
        self.assertEqual(torch_operator.normalized_type, stable_operator.normalized_type)
        self.assertEqual(torch_operator.iteration_dims, stable_operator.iteration_dims)
        self.assertEqual(torch_operator.reduction_dims, stable_operator.reduction_dims)
        self.assertEqual(
            next(item.shape for item in compiled.graph.tensors if item.name == torch_operator.inputs[0]),
            next(item.shape for item in stable_compiled.graph.tensors if item.name == stable_operator.inputs[0]),
        )
        self.assertEqual(len(compiled.tile_graph.tiles), len(stable_compiled.tile_graph.tiles))
        self.assertEqual(
            len(compiled.tisa_program.instructions),
            len(stable_compiled.tisa_program.instructions),
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch for the real export smoke")
    def test_real_torch_linear_decomposes_and_compiles(self) -> None:
        import torch

        module = torch.nn.Linear(8, 12, bias=True).eval()
        compiled = compile_torch_module(
            module,
            (torch.randn(2, 4, 8),),
            minimal_machine_config(),
            model_id="real-torch-linear",
            tile_size=4,
        )
        self.assertEqual(
            [operator.normalized_type for operator in compiled.graph.operators],
            ["batched_matmul", "elementwise"],
        )
        self.assertTrue(compiled.graph.operators[0].attributes["rhs_transposed"])
        self.assertTrue(compiled.graph.operators[0].attributes["rhs_broadcast_batch"])
        parameters = {
            tensor.name
            for tensor in compiled.frontend.graph.tensors
            if tensor.attributes.get("source_kind") == "parameter"
        }
        self.assertEqual(len(parameters), 2)
        self.assertTrue(
            any(
                task.primitive == "load_transpose"
                and task.attributes.get("operand") == "rhs"
                for task in compiled.backend_artifact.execution_graph.tasks
            )
        )
        self.assertIn(
            "load_transpose",
            {instruction.op_type for instruction in compiled.tisa_program.instructions},
        )
        self.assertEqual(compiled.validate(), ())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch for the real export smoke")
    def test_real_torch_affine_layernorm_rank3_compiles(self) -> None:
        import torch

        compiled = compile_torch_module(
            torch.nn.LayerNorm(8, eps=3e-4).eval(),
            (torch.randn(2, 4, 8),),
            minimal_machine_config(),
            model_id="real-torch-layernorm",
            tile_size=4,
        )
        operator = compiled.graph.operators[0]
        self.assertEqual(operator.normalized_type, "layernorm")
        self.assertEqual(operator.iteration_dims, (("d0", 2), ("d1", 4)))
        self.assertEqual(operator.reduction_dims, (("d2", 8),))
        self.assertEqual(operator.attributes["epsilon"], 3e-4)
        self.assertTrue(operator.attributes["affine"])
        parameter_loads = {
            task.attributes.get("operand")
            for task in compiled.backend_artifact.execution_graph.tasks
            if task.attributes.get("affine") and task.primitive == "load"
        }
        self.assertEqual(parameter_loads, {"weight", "bias"})
        self.assertEqual(compiled.validate(), ())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch for the real export smoke")
    def test_real_torch_rmsnorm_rank3_fuses_and_compiles(self) -> None:
        import torch

        class RMSNorm(torch.nn.Module):
            def forward(self, x):
                return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)

        compiled = compile_torch_module(
            RMSNorm().eval(),
            (torch.randn(2, 4, 8, dtype=torch.float16),),
            minimal_machine_config(),
            model_id="real-torch-rmsnorm-rank3",
            tile_size=4,
        )
        operator = compiled.graph.operators[0]
        self.assertEqual(operator.normalized_type, "rmsnorm")
        self.assertEqual(operator.iteration_dims, (("d0", 2), ("d1", 4)))
        self.assertEqual(operator.reduction_dims, (("d2", 8),))
        self.assertEqual(compiled.validate(), ())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch for the StableHLO RMSNorm round-trip")
    def test_rmsnorm_round_trips_through_generated_stablehlo(self) -> None:
        import torch

        class RMSNorm(torch.nn.Module):
            def forward(self, x):
                return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)

        compiled = compile_torch_module_through_stablehlo(
            RMSNorm().eval(),
            (torch.randn(2, 4, 8, dtype=torch.float16),),
            minimal_machine_config(),
            model_id="stablehlo-rmsnorm-roundtrip",
            tile_size=4,
            stablehlo_backend="textual",
        )
        self.assertEqual([operator.normalized_type for operator in compiled.graph.operators], ["rmsnorm"])
        self.assertEqual(compiled.graph.operators[0].iteration_dims, (("d0", 2), ("d1", 4)))
        self.assertEqual(compiled.graph.operators[0].reduction_dims, (("d2", 8),))
        self.assertEqual(compiled.validate(), ())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch for the real export smoke")
    def test_real_torch_softmax_preserves_axis_and_physical_region_order(self) -> None:
        import torch

        class SoftmaxDimOne(torch.nn.Module):
            def forward(self, x):
                return torch.softmax(x, dim=1)

        compiled = compile_torch_module(
            SoftmaxDimOne().eval(),
            (torch.randn(2, 3, 4),),
            minimal_machine_config(),
            model_id="real-torch-softmax",
            tile_size=2,
        )
        operator = compiled.graph.operators[0]
        self.assertEqual(operator.iteration_dims, (("d0", 2), ("d2", 4)))
        self.assertEqual(operator.reduction_dims, (("d1", 3),))
        input_regions = [
            region
            for task in compiled.backend_artifact.execution_graph.tasks
            for region in task.reads
            if region.tensor == operator.inputs[0] and region.memory == "DRAM"
        ]
        self.assertIn((0, 2, 0), {region.starts for region in input_regions})
        self.assertEqual(compiled.validate(), ())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch for the real export smoke")
    def test_real_torch_attention_micrograph_compiles(self) -> None:
        import torch

        class Attention(torch.nn.Module):
            def forward(self, q, k, v):
                scores = torch.matmul(q, k.transpose(-2, -1))
                probabilities = torch.softmax(scores, dim=-1)
                return torch.matmul(probabilities, v)

        inputs = (
            torch.randn(2, 4, 8),
            torch.randn(2, 4, 8),
            torch.randn(2, 4, 8),
        )
        compiled = compile_torch_module(
            Attention().eval(),
            inputs,
            minimal_machine_config(),
            model_id="real-torch-attention",
            tile_size=2,
        )
        self.assertEqual(
            [operator.normalized_type for operator in compiled.graph.operators],
            ["batched_matmul", "softmax", "batched_matmul"],
        )
        self.assertTrue(compiled.graph.operators[0].attributes["rhs_transposed"])
        self.assertEqual(compiled.graph.operators[0].iteration_dims[0], ("B0", 2))
        self.assertEqual(
            [(edge.producer, edge.consumer) for edge in compiled.graph.edges],
            [("matmul", "softmax"), ("softmax", "matmul_1")],
        )
        self.assertTrue(
            any(
                task.primitive == "load_transpose"
                and task.attributes.get("operand") == "rhs"
                for task in compiled.backend_artifact.execution_graph.tasks
            )
        )
        self.assertIn(
            "load_transpose",
            {instruction.op_type for instruction in compiled.tisa_program.instructions},
        )
        self.assertEqual(compiled.validate(), ())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch for the real export smoke")
    def test_real_torch_attention_block_composes_supported_frontend_ops(self) -> None:
        import torch

        class AttentionBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(8)
                self.q_proj = torch.nn.Linear(8, 8)
                self.k_proj = torch.nn.Linear(8, 8)
                self.v_proj = torch.nn.Linear(8, 8)
                self.out_proj = torch.nn.Linear(8, 8)

            def forward(self, x):
                hidden = self.norm(x)
                q = self.q_proj(hidden)
                k = self.k_proj(hidden)
                v = self.v_proj(hidden)
                probabilities = torch.softmax(
                    torch.matmul(q, k.transpose(-2, -1)),
                    dim=-1,
                )
                context = torch.matmul(probabilities, v)
                return x + self.out_proj(context)

        compiled = compile_torch_module(
            AttentionBlock().eval(),
            (torch.randn(1, 4, 8),),
            minimal_machine_config(),
            model_id="real-torch-attention-block",
            tile_size=4,
        )
        operator_types = [operator.normalized_type for operator in compiled.graph.operators]
        self.assertEqual(operator_types[0], "layernorm")
        self.assertEqual(operator_types.count("batched_matmul"), 6)
        self.assertEqual(operator_types.count("softmax"), 1)
        self.assertEqual(operator_types.count("elementwise"), 5)
        self.assertEqual(len(compiled.graph.edges), 14)
        self.assertEqual(compiled.validate(), ())

    def test_torch_adapter_imports_fx_shape_and_provenance_without_torch_runtime(self) -> None:
        x = _FakeNode("x", "placeholder", target="x", shape=(4, 8))
        square = _FakeNode(
            "square",
            "call_function",
            target="aten.mul.Tensor",
            args=(x, x),
            shape=(4, 8),
        )
        reduce = _FakeNode(
            "reduce",
            "call_function",
            target="aten.mean.dim",
            args=(square,),
            kwargs={"dim": -1, "keepdim": True},
            shape=(4, 1),
        )
        epsilon = _FakeNode(
            "epsilon",
            "call_function",
            target="aten.add.Tensor",
            args=(reduce, 1e-5),
            shape=(4, 1),
        )
        norm = _FakeNode(
            "norm",
            "call_function",
            target="aten.rsqrt.default",
            args=(epsilon,),
            shape=(4, 1),
        )
        scaled = _FakeNode(
            "scaled",
            "call_function",
            target="aten.mul.Tensor",
            args=(x, norm),
            shape=(4, 8),
        )
        output = _FakeNode("output", "output", args=(scaled,), shape=(4, 8))
        imported = TorchExportAdapter.from_exported_program(
            _FakeExportedProgram(_FakeGraph([x, square, reduce, epsilon, norm, scaled, output])),
            model_id="realistic_rmsnorm",
            variant="torch-export-test",
        )
        self.assertEqual(imported.model_id, "realistic_rmsnorm")
        self.assertEqual(imported.graph.attributes["graph_inputs"], ["x"])
        self.assertEqual(imported.graph.attributes["graph_outputs"], ["scaled"])
        self.assertEqual(imported.graph.tensors[0].dtype, "float16")
        self.assertEqual(
            [operator.normalized_type for operator in imported.graph.operators],
            ["elementwise", "reduce", "elementwise", "elementwise", "elementwise"],
        )
        self.assertEqual(imported.graph.edges, (
            DataEdge("square", "reduce", "square"),
            DataEdge("reduce", "epsilon", "reduce"),
            DataEdge("epsilon", "norm", "epsilon"),
            DataEdge("norm", "scaled", "norm"),
        ))
        self.assertEqual(imported.validate(), ())

        fused = PassManager().run(imported.graph)
        self.assertEqual([operator.normalized_type for operator in fused.graph.operators], ["rmsnorm"])
        self.assertEqual(fused.graph.operators[0].attributes["epsilon"], 1e-5)
        self.assertEqual(fused.graph.operators[0].inputs, ("x",))
        self.assertEqual(fused.graph.operators[0].outputs, ("scaled",))
        self.assertEqual(fused.graph.validate(), ())
        compiled = compile_frontend_import(imported, minimal_machine_config(), tile_size=4)
        self.assertEqual(compiled.graph.operators[0].normalized_type, "rmsnorm")
        self.assertGreater(len(compiled.tisa_program.instructions), 0)
        self.assertEqual(compiled.validate(), ())

    def test_pass_manager_normalizes_aliases_and_infers_edges(self) -> None:
        graph = OperatorGraph(
            graph_id="pass_graph",
            tensors=(TensorSpec("x", (4, 8)), TensorSpec("y", (4, 8)), TensorSpec("z", (4, 8))),
            operators=(
                OperatorSpec("first", "aten.mm.default", ("x",), ("y",), iteration_dims=(("M", 4), ("N", 8)), reduction_dims=(("K", 8),)),
                OperatorSpec("second", "aten.add.Tensor", ("y", "x"), ("z",), iteration_dims=(("M", 4), ("N", 8))),
            ),
        )
        result = PassManager().run(graph)
        self.assertEqual([item.normalized_type for item in result.graph.operators], ["matmul", "elementwise"])
        self.assertEqual(result.graph.edges, (DataEdge("first", "second", "y"),))
        self.assertTrue(result.graph.attributes["canonicalized"])
        self.assertEqual(result.graph.validate(), ())

    def test_linear_decomposition_lowers_transposed_weight_and_broadcast_bias(self) -> None:
        graph = OperatorGraph(
            graph_id="linear_graph",
            tensors=(
                TensorSpec("x", (4, 8)),
                TensorSpec("weight", (12, 8)),
                TensorSpec("bias", (12,)),
                TensorSpec("y", (4, 12)),
            ),
            operators=(
                OperatorSpec(
                    "linear",
                    "matmul",
                    ("x", "weight", "bias"),
                    ("y",),
                    iteration_dims=(("M", 4), ("N", 12)),
                    reduction_dims=(("K", 8),),
                    attributes={
                        "frontend_target": "aten::linear",
                        "bias_input": "bias",
                    },
                ),
            ),
        )
        compiled = compile_frontend_import(
            import_operator_graph(graph, model_id="linear"),
            minimal_machine_config(),
            tile_size=4,
        )
        self.assertEqual(
            [operator.normalized_type for operator in compiled.graph.operators],
            ["matmul", "elementwise"],
        )
        self.assertTrue(compiled.graph.operators[0].attributes["rhs_transposed"])
        self.assertEqual(
            compiled.graph.edges,
            (DataEdge("linear.matmul", "linear.bias_add", "linear.matmul_output"),),
        )
        tasks = compiled.backend_artifact.execution_graph.tasks
        self.assertTrue(
            any(
                task.primitive == "load_transpose"
                and task.attributes.get("operand") == "rhs"
                for task in tasks
            )
        )
        self.assertFalse(
            any(
                task.primitive == "load_transpose"
                and task.attributes.get("operand") == "lhs"
                for task in tasks
            )
        )
        bias_reads = [
            region
            for task in tasks
            for region in task.reads
            if region.tensor == "bias"
        ]
        self.assertTrue(bias_reads)
        self.assertTrue(all(len(region.shape) == 1 for region in bias_reads))
        self.assertEqual(compiled.validate(), ())

    def test_stablehlo_text_import_and_rmsnorm_fusion(self) -> None:
        text = """
        module {
          func.func @main(%arg0: tensor<4x8xf16>) -> tensor<4x8xf16> {
            %cst = stablehlo.constant dense<1.000000e-05> : tensor<f32>
            %0 = stablehlo.multiply %arg0, %arg0 : tensor<4x8xf16>
            %1 = stablehlo.reduce %0, %cst dimensions = [1] : (tensor<4x8xf16>, tensor<f32>) -> tensor<4x1xf16>
            %2 = stablehlo.add %1, %cst : tensor<4x1xf16>
            %3 = stablehlo.rsqrt %2 : tensor<4x1xf16>
            %4 = stablehlo.multiply %arg0, %3 : tensor<4x8xf16>
            return %4 : tensor<4x8xf16>
          }
        }
        """
        imported = StableHLOAdapter.from_text(text, model_id="stable-rmsnorm")
        self.assertEqual(imported.frontend.value, "stablehlo")
        self.assertEqual(imported.graph.attributes["graph_inputs"], ["arg0"])
        self.assertEqual(imported.graph.attributes["graph_outputs"], ["4"])
        self.assertEqual([item.normalized_type for item in imported.graph.operators], [
            "elementwise", "reduce", "elementwise", "elementwise", "elementwise"
        ])
        self.assertEqual(imported.graph.operators[0].attributes["operand_arity"], 2)
        self.assertEqual(imported.graph.operators[0].inputs, ("arg0",))
        fused = PassManager().run(imported.graph)
        self.assertEqual([item.normalized_type for item in fused.graph.operators], ["rmsnorm"])
        compiled = compile_frontend_import(imported, minimal_machine_config(), tile_size=4)
        self.assertEqual(compiled.graph.operators[0].normalized_type, "rmsnorm")
        self.assertEqual(compiled.validate(), ())

    def test_stablehlo_pointwise_registry_compiles_with_operation_identity(self) -> None:
        text = """
        module {
          func.func @main(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
            %zero = stablehlo.constant dense<0.0> : tensor<f32>
            %sine = stablehlo.sine %x : tensor<2x4xf32>
            %relu = stablehlo.maximum %x, %zero : tensor<2x4xf32>
            %result = stablehlo.add %sine, %relu : tensor<2x4xf32>
            return %result : tensor<2x4xf32>
          }
        }
        """
        compiled = compile_stablehlo_text(
            text,
            minimal_machine_config(),
            model_id="pointwise-registry",
            tile_size=4,
            stablehlo_backend="textual",
        )
        self.assertEqual(
            [operator.normalized_type for operator in compiled.graph.operators],
            ["elementwise", "elementwise", "elementwise"],
        )
        sine = compiled.graph.operators[0]
        self.assertEqual(sine.attributes["stablehlo_op"], "stablehlo.sine")
        self.assertEqual(sine.attributes["semantic_family"], "elementwise")
        self.assertEqual(sine.attributes["semantic_op"], "sine")
        self.assertEqual(sine.attributes["backend_capability_key"], "pointwise.sine")
        relu = compiled.graph.operators[1]
        self.assertEqual(relu.attributes["operand_arity"], 2)
        self.assertEqual(relu.inputs, ("x",))

        compute = next(
            task
            for task in compiled.backend_artifact.execution_graph.tasks
            if task.operator_id == sine.op_id and task.primitive == "elementwise"
        )
        self.assertEqual(compute.attributes["semantic_op"], "sine")
        self.assertEqual(compute.attributes["frontend_target"], "stablehlo.sine")
        self.assertEqual(compute.attributes["timing_key"], "pointwise.sine")
        self.assertEqual(compute.attributes["operand_arity"], 1)
        relu_compute = next(
            task
            for task in compiled.backend_artifact.execution_graph.tasks
            if task.operator_id == relu.op_id and task.primitive == "elementwise"
        )
        self.assertEqual(relu_compute.attributes["operand_arity"], 2)
        self.assertEqual(relu_compute.attributes["input_count"], 1)
        sine_tiles = [tile for tile in compiled.tile_graph.tiles if tile.operator_id == sine.op_id]
        self.assertTrue(sine_tiles)
        self.assertTrue(all(tile.attributes["semantic_op"] == "sine" for tile in sine_tiles))
        self.assertTrue(
            all(
                tile.attributes["backend_capability_key"] == "pointwise.sine"
                for tile in sine_tiles
            )
        )
        sine_tisa = [
            instruction
            for instruction in compiled.tisa_program.instructions
            if instruction.operator_id == sine.op_id
        ]
        self.assertTrue(sine_tisa)
        self.assertTrue(
            all(instruction.attributes["semantic_op"] == "sine" for instruction in sine_tisa)
        )

        simulated = simulate_execution_graph(
            compiled.backend_artifact.execution_graph,
            minimal_machine_config(),
            policy="dynamic_ready_queue",
        )
        self.assertGreater(simulated.total_cycles, 0)
        self.assertTrue(simulated.perfetto_trace()["traceEvents"])
        self.assertEqual(compiled.validate(), ())

    def test_unregistered_stablehlo_op_fails_at_import_capability_boundary(self) -> None:
        text = """
        module {
          func.func @main(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
            %0 = stablehlo.custom_call %x : tensor<2x4xf32>
            return %0 : tensor<2x4xf32>
          }
        }
        """
        with self.assertRaisesRegex(
            FrontendImportError,
            "import capability boundary",
        ):
            StableHLOAdapter.from_text(text)

    def test_stablehlo_matmul_compiles_to_tisa(self) -> None:
        text = """
        module {
          func.func @main(%lhs: tensor<4x8xf16>, %rhs: tensor<8x12xf16>) -> tensor<4x12xf16> {
            %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0] : (tensor<4x8xf16>, tensor<8x12xf16>) -> tensor<4x12xf16>
            return %0 : tensor<4x12xf16>
          }
        }
        """
        imported = StableHLOAdapter.from_text(text, model_id="stable-matmul")
        self.assertEqual([item.normalized_type for item in imported.graph.operators], ["matmul"])
        operator = imported.graph.operators[0]
        self.assertEqual(operator.iteration_dims, (("M", 4), ("N", 12)))
        self.assertEqual(operator.reduction_dims, (("K", 8),))
        self.assertEqual(operator.attributes["dot_dimension_numbers"]["lhs_contracting_dimensions"], [1])
        compiled = compile_frontend_import(imported, minimal_machine_config(), tile_size=4)
        self.assertEqual({item.unit_map.unit for item in compiled.tisa_program.instructions}, {"dma", "tensor"})
        self.assertEqual(compiled.validate(), ())

    def test_stablehlo_reduce_uses_result_type_and_compiles(self) -> None:
        text = """
        module {
          func.func @main(%x: tensor<4x8xf16>) -> tensor<4x1xf16> {
            %init = stablehlo.constant dense<0.0> : tensor<f16>
            %0 = stablehlo.reduce %x, %init dimensions = [1] : (tensor<4x8xf16>, tensor<f16>) -> tensor<4x1xf16>
            return %0 : tensor<4x1xf16>
          }
        }
        """
        compiled = compile_stablehlo_text(
            text,
            minimal_machine_config(),
            model_id="stable-reduce",
            tile_size=4,
            stablehlo_backend="textual",
        )
        operator = compiled.graph.operators[0]
        self.assertEqual(operator.normalized_type, "reduce")
        self.assertEqual(operator.iteration_dims, (("d0", 4),))
        self.assertEqual(operator.reduction_dims, (("d1", 8),))
        self.assertEqual(next(item for item in compiled.graph.tensors if item.name == "0").shape, (4, 1))
        self.assertEqual(compiled.validate(), ())

    def test_stablehlo_dot_imports_batched_matmul(self) -> None:
        text = """
        module {
          func.func @main(%lhs: tensor<2x4x8xf16>, %rhs: tensor<2x8x12xf16>) -> tensor<2x4x12xf16> {
            %0 = stablehlo.dot_general %lhs, %rhs, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x4x8xf16>, tensor<2x8x12xf16>) -> tensor<2x4x12xf16>
            return %0 : tensor<2x4x12xf16>
          }
        }
        """
        imported = StableHLOAdapter.from_text(text)
        operator = imported.graph.operators[0]
        self.assertEqual(operator.normalized_type, "batched_matmul")
        self.assertEqual(operator.iteration_dims, (("B0", 2), ("M", 4), ("N", 12)))
        self.assertEqual(operator.reduction_dims, (("K", 8),))

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
            availability_path = root / "availability.json"
            availability_path.write_text(
                json.dumps({"tisa.add0.t0000.s00": 4}), encoding="utf-8"
            )
            exit_code = main(
                [
                    "compile-model",
                    "--graph-json",
                    str(graph_path),
                    "--arch",
                    "minimal",
                    "--policy",
                    "dynamic_ready_queue",
                    "--runtime-policy",
                    "dynamic_ready_queue",
                    "--runtime-chunk-size",
                    "1",
                    "--runtime-launch-latency",
                    "2",
                    "--runtime-synchronization-cycles",
                    "3",
                    "--runtime-buffer-policy",
                    "lifetime_reuse",
                    "--runtime-availability-config",
                    str(availability_path),
                    "--runtime-device-matrix",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "00_frontend" / "frontend_import.json").exists())
            self.assertTrue((output / "03_tisa" / "tisa_program.json").exists())
            self.assertTrue((output / "04_backend" / "backend_artifact.json").exists())
            self.assertTrue((output / "05_runtime" / "runtime_submission.json").exists())
            self.assertTrue((output / "06_simulation" / "tisa_instructions.csv").exists())
            self.assertTrue((output / "07_trace" / "perfetto.json").exists())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["compiler_pipeline"], "frontend->canonical->schedule->tile->tisa->analytical-backend")
            self.assertEqual(manifest["scheduler_target"], "tisa")
            self.assertEqual(
                manifest["tisa_decision_count"], manifest["tisa_instruction_count"]
            )
            self.assertEqual(manifest["payload_execution"], "run_to_completion")
            self.assertEqual(manifest["runtime_policy"], "dynamic_ready_queue")
            self.assertTrue(manifest["runtime_applied_to_device"])
            self.assertEqual(manifest["runtime_buffer_policy"], "lifetime_reuse")
            self.assertGreater(manifest["runtime_allocation_span_bytes"], 0)
            self.assertGreater(manifest["runtime_buffer_count"], 0)
            self.assertGreater(manifest["runtime_command_chunk_count"], 0)
            self.assertEqual(
                manifest["runtime_submit_busy_cycles"],
                manifest["runtime_command_chunk_count"] * 2,
            )
            self.assertEqual(manifest["runtime_request_wait_cycles"], 4)
            self.assertEqual(
                manifest["runtime_submit_cycles"],
                manifest["runtime_submit_busy_cycles"]
                + manifest["runtime_request_wait_cycles"],
            )
            self.assertEqual(manifest["runtime_descriptor_availability_count"], 1)
            self.assertEqual(manifest["runtime_synchronization_cycles"], 3)
            self.assertEqual(
                manifest["total_cycles_including_runtime"], manifest["total_cycles"]
            )
            self.assertGreaterEqual(
                manifest["device_finish_cycle"], manifest["device_start_cycle"]
            )
            summary = json.loads(
                (output / "06_simulation" / "summary.json").read_text()
            )
            self.assertEqual(
                len(summary["instruction_timings"]), manifest["tisa_instruction_count"]
            )
            swimlane = (output / "07_trace" / "swimlane.svg").read_text()
            self.assertIn(">TISA instruction</text>", swimlane)
            self.assertIn("TISA/", swimlane)
            self.assertIn(">Runtime submit</text>", swimlane)
            self.assertIn("Runtime/Submit[0]", swimlane)
            perfetto = json.loads(
                (output / "07_trace" / "perfetto.json").read_text()
            )
            runtime_events = [
                event for event in perfetto["traceEvents"] if event["pid"] == 0
            ]
            self.assertEqual(
                len(runtime_events), manifest["runtime_command_chunk_count"] * 2
            )
            compiled_artifact_id = json.loads(
                (output / "04_backend" / "backend_artifact.json").read_text()
            )["artifact_id"]
            matrix = json.loads(
                (output / "policy_matrix" / "sweep.json").read_text()
            )
            artifact_index = json.loads(
                (output / "artifact_index.json").read_text()
            )
            self.assertIn("policy_matrix", artifact_index["top_level_directories"])
            self.assertEqual(len(matrix), 4)
            self.assertEqual(
                {(row["runtime_policy"], row["device_policy"]) for row in matrix},
                {
                    ("static", "static_pipeline"),
                    ("static", "dynamic_ready_queue"),
                    ("dynamic_ready_queue", "static_pipeline"),
                    ("dynamic_ready_queue", "dynamic_ready_queue"),
                },
            )
            self.assertEqual({row["artifact_id"] for row in matrix}, {compiled_artifact_id})
            for row in matrix:
                matrix_case = output / "policy_matrix" / row["case_id"]
                self.assertTrue(
                    (matrix_case / "05_runtime" / "runtime_submission.json").exists()
                )
                self.assertTrue((matrix_case / "06_simulation" / "summary.json").exists())
                self.assertTrue((matrix_case / "07_trace" / "perfetto.json").exists())
                self.assertFalse(
                    (matrix_case / "03_tisa" / "compiled_artifact.json").exists()
                )

            primitive_output = root / "primitive-out"
            primitive_exit_code = main(
                [
                    "compile-model",
                    "--graph-json",
                    str(graph_path),
                    "--arch",
                    "minimal",
                    "--scheduler-target",
                    "primitive",
                    "--output-dir",
                    str(primitive_output),
                ]
            )
            self.assertEqual(primitive_exit_code, 0)
            primitive_manifest = json.loads(
                (primitive_output / "manifest.json").read_text()
            )
            self.assertEqual(primitive_manifest["scheduler_target"], "primitive")
            self.assertFalse(primitive_manifest["runtime_applied_to_device"])
            self.assertFalse(
                (primitive_output / "06_simulation" / "tisa_instructions.csv").exists()
            )

    def test_compile_model_cli_accepts_stablehlo_file(self) -> None:
        text = """
        module {
          func.func @main(%lhs: tensor<4x8xf16>, %rhs: tensor<8x12xf16>) -> tensor<4x12xf16> {
            %0 = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0] : (tensor<4x8xf16>, tensor<8x12xf16>) -> tensor<4x12xf16>
            return %0 : tensor<4x12xf16>
          }
        }
        """
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root / "matmul.mlir"
            output = root / "out"
            source.write_text(text, encoding="utf-8")
            exit_code = main([
                "compile-model",
                "--stablehlo-file",
                str(source),
                "--stablehlo-backend",
                "textual",
                "--arch",
                "minimal",
                "--output-dir",
                str(output),
            ])
            self.assertEqual(exit_code, 0)
            frontend = json.loads((output / "00_frontend" / "frontend_import.json").read_text())
            self.assertEqual(frontend["frontend"], "stablehlo")
            self.assertTrue((output / "03_tisa" / "tisa_program.json").exists())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch for the real export smoke")
    def test_compile_model_cli_accepts_torch_module_factory(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "out"
            exit_code = main([
                "compile-model",
                "--torch-module",
                "examples.torch_models:attention_block",
                "--input-shape",
                "1,4,8",
                "--tile-size",
                "4",
                "--arch",
                "minimal",
                "--output-dir",
                str(output),
            ])
            self.assertEqual(exit_code, 0)
            frontend = json.loads((output / "00_frontend" / "frontend_import.json").read_text())
            graph = json.loads((output / "01_graph_ir" / "canonical_graph.json").read_text())
            self.assertEqual(frontend["frontend"], "torch.export")
            self.assertIn("layernorm", {operator["op_type"] for operator in graph["operators"]})
            self.assertIn("batched_matmul", {operator["op_type"] for operator in graph["operators"]})
            self.assertTrue((output / "07_trace" / "swimlane.png").exists())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch for the StableHLO CLI smoke")
    def test_compile_model_cli_round_trips_torch_through_stablehlo(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "out"
            exit_code = main([
                "compile-model",
                "--torch-module",
                "examples.torch_models:attention_block",
                "--input-shape",
                "1,4,8",
                "--tile-size",
                "4",
                "--arch",
                "minimal",
                "--through-stablehlo",
                "--stablehlo-backend",
                "textual",
                "--output-dir",
                str(output),
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "00_frontend" / "generated.mlir").exists())
            self.assertTrue((output / "00_frontend" / "stablehlo_module.json").exists())
            self.assertTrue((output / "00_frontend" / "source_frontend_import.json").exists())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["frontend_path"], "torch_export->stablehlo_textual->stablehlo_import->canonical")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch for the StableHLO round-trip")
    def test_torch_export_round_trips_through_generated_stablehlo(self) -> None:
        import torch

        from examples.torch_models import attention_block

        inputs = (torch.randn(2, 4, 8),)
        direct = compile_torch_module(
            attention_block(), inputs, minimal_machine_config(), tile_size=4, model_id="attention"
        )
        round_trip = compile_torch_module_through_stablehlo(
            attention_block(),
            inputs,
            minimal_machine_config(),
            tile_size=4,
            model_id="attention",
            stablehlo_backend="textual",
        )
        self.assertIsNotNone(round_trip.stablehlo)
        self.assertIn("stablehlo.dot_general", round_trip.stablehlo.text)
        self.assertEqual(
            {operator.normalized_type for operator in direct.graph.operators},
            {operator.normalized_type for operator in round_trip.graph.operators},
        )
        self.assertEqual(len(direct.tile_graph.tiles), len(round_trip.tile_graph.tiles))
        self.assertEqual(len(direct.tisa_program.instructions), len(round_trip.tisa_program.instructions))
        self.assertEqual(round_trip.validate(), ())

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and official_stablehlo_available(),
        "requires PyTorch and official StableHLO bindings",
    )
    def test_torch_attention_round_trips_through_official_stablehlo(self) -> None:
        import torch

        from examples.torch_models import attention_block

        inputs = (torch.randn(1, 4, 8),)
        direct = compile_torch_module(
            attention_block(), inputs, minimal_machine_config(), tile_size=4, model_id="attention"
        )
        official = compile_torch_module_through_stablehlo(
            attention_block(), inputs, minimal_machine_config(), tile_size=4, model_id="attention"
        )
        self.assertTrue(official.stablehlo.verified)
        self.assertEqual(official.attributes["stablehlo_backend"], "official")
        self.assertFalse(official.attributes["stablehlo_fallback"])
        self.assertIn('"stablehlo.reduce"', official.stablehlo.text)
        self.assertIn("stablehlo.broadcast_in_dim", official.stablehlo.text)
        self.assertEqual(
            [operator.normalized_type for operator in direct.graph.operators],
            [operator.normalized_type for operator in official.graph.operators],
        )
        self.assertEqual(len(direct.tile_graph.tiles), len(official.tile_graph.tiles))
        self.assertEqual(len(direct.tisa_program.instructions), len(official.tisa_program.instructions))
        self.assertEqual(len(direct.backend_artifact.execution_graph.tasks), len(official.backend_artifact.execution_graph.tasks))
        self.assertEqual(official.validate(), ())

    @unittest.skipUnless(
        importlib.util.find_spec("torch")
        and official_stablehlo_available()
        and torch_xla_available(),
        "requires PyTorch, torch-xla and official StableHLO bindings",
    )
    def test_attention_micrograph_exports_through_torch_xla_stablehlo(self) -> None:
        import torch

        from examples.torch_models import attention_micrograph

        inputs = tuple(torch.randn(2, 4, 8) for _ in range(3))
        direct = compile_torch_module(
            attention_micrograph(), inputs, minimal_machine_config(), tile_size=2, model_id="xla-attention"
        )
        xla = compile_torch_module_through_stablehlo(
            attention_micrograph(),
            inputs,
            minimal_machine_config(),
            tile_size=2,
            model_id="xla-attention",
            stablehlo_exporter="torch-xla",
        )
        self.assertEqual(xla.attributes["stablehlo_exporter"], "torch-xla")
        self.assertEqual(xla.attributes["stablehlo_exporter_version"], "2.9.0")
        self.assertEqual(xla.stablehlo.producer, "torch-xla")
        self.assertGreater(xla.stablehlo.provenance["bytecode_size"], 0)
        self.assertEqual(
            [operator.normalized_type for operator in xla.graph.operators],
            ["batched_matmul", "softmax", "batched_matmul"],
        )
        self.assertEqual(len(xla.tile_graph.tiles), len(direct.tile_graph.tiles))
        self.assertEqual(len(xla.tisa_program.instructions), len(direct.tisa_program.instructions))
        self.assertEqual(
            len(xla.backend_artifact.execution_graph.tasks),
            len(direct.backend_artifact.execution_graph.tasks),
        )
        self.assertEqual(xla.validate(), ())

    @unittest.skipUnless(
        importlib.util.find_spec("torch")
        and official_stablehlo_available()
        and torch_xla_available(),
        "requires PyTorch, torch-xla and official StableHLO bindings",
    )
    def test_new_pointwise_module_exports_through_torch_xla_without_project_emitter(self) -> None:
        import torch

        class NewPointwiseOperator(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x) + torch.relu(x)

        with mock.patch(
            "npu_ooo.frontend.stablehlo_codegen.StableHLOGenerator.generate",
            side_effect=AssertionError("project StableHLO emitter must not run"),
        ):
            compiled = compile_torch_module_through_stablehlo(
                NewPointwiseOperator(),
                (torch.randn(2, 4),),
                minimal_machine_config(),
                tile_size=4,
                model_id="xla-new-pointwise",
                stablehlo_exporter="torch-xla",
            )

        self.assertEqual(compiled.stablehlo.producer, "torch-xla")
        self.assertIn("stablehlo.sine", compiled.stablehlo.text)
        self.assertIn("stablehlo.maximum", compiled.stablehlo.text)
        self.assertEqual(
            [operator.normalized_type for operator in compiled.graph.operators],
            ["elementwise", "elementwise", "elementwise"],
        )
        self.assertEqual(
            [operator.attributes["semantic_op"] for operator in compiled.graph.operators],
            ["sine", "maximum", "add"],
        )
        self.assertTrue(compiled.tile_graph.tiles)
        self.assertTrue(compiled.tisa_program.instructions)
        self.assertTrue(compiled.backend_artifact.execution_graph.tasks)
        simulated = simulate_execution_graph(
            compiled.backend_artifact.execution_graph,
            minimal_machine_config(),
            policy="dynamic_ready_queue",
        )
        self.assertGreater(simulated.total_cycles, 0)
        self.assertTrue(simulated.perfetto_trace()["traceEvents"])
        self.assertEqual(compiled.validate(), ())

    @unittest.skipUnless(
        importlib.util.find_spec("torch")
        and official_stablehlo_available()
        and torch_xla_available(),
        "requires PyTorch, torch-xla and official StableHLO bindings",
    )
    def test_attention_block_exports_through_torch_xla_stablehlo(self) -> None:
        import torch

        from examples.torch_models import attention_block

        inputs = (torch.randn(2, 4, 8),)
        direct = compile_torch_module(
            attention_block(),
            inputs,
            minimal_machine_config(),
            tile_size=4,
            model_id="xla-attention-block",
        )
        xla = compile_torch_module_through_stablehlo(
            attention_block(),
            inputs,
            minimal_machine_config(),
            tile_size=4,
            model_id="xla-attention-block",
            stablehlo_exporter="torch-xla",
        )
        self.assertEqual(
            sorted(operator.normalized_type for operator in xla.graph.operators),
            sorted(operator.normalized_type for operator in direct.graph.operators),
        )
        self.assertEqual(len(xla.tile_graph.tiles), len(direct.tile_graph.tiles))
        self.assertEqual(len(xla.tisa_program.instructions), len(direct.tisa_program.instructions))
        self.assertEqual(
            len(xla.backend_artifact.execution_graph.tasks),
            len(direct.backend_artifact.execution_graph.tasks),
        )
        layernorm = xla.graph.operators[0]
        self.assertEqual(layernorm.normalized_type, "layernorm")
        self.assertEqual(layernorm.attributes["fusion"], "torch_xla_batch_norm_layernorm")
        self.assertTrue(layernorm.attributes["flattened_norm_recovered"])
        self.assertEqual(
            sum(
                bool(operator.attributes.get("flattened_linear_recovered"))
                for operator in xla.graph.operators
                if operator.normalized_type == "batched_matmul"
            ),
            4,
        )
        self.assertEqual(xla.validate(), ())


if __name__ == "__main__":
    unittest.main()
