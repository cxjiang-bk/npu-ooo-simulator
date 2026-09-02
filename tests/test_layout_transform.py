import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.compiler import compile_operator_graph
from npu_ooo.frontend import FrontendImport, OfficialStableHLOModule
from npu_ooo.ir import (
    OperatorGraph,
    OperatorSpec,
    RuntimeLayoutBinding,
    TensorSpec,
    resolve_layout,
    allocate_buffer_bindings,
    create_runtime_submission,
)


def _frontend(graph: OperatorGraph):
    return FrontendImport(
        graph=graph,
        model_id=graph.graph_id,
        variant="layout-test",
        frontend="stablehlo",
    ), OfficialStableHLOModule(
        text="module {}",
        canonical_text="module {}",
        model_id=graph.graph_id,
    )


class LayoutResolutionTest(unittest.TestCase):
    def test_structured_stablehlo_byte_strides(self) -> None:
        info = resolve_layout(
            (2, 3),
            "f32",
            attributes={
                "layout_source": "stablehlo_encoding",
                "layout_encoding": "#strided<[4, 1], offset: 2>",
            },
        )
        self.assertEqual(info.strides_bytes, (16, 4))
        self.assertEqual(info.offset_bytes, 8)
        self.assertEqual(info.interval((1, 0), (1, 3)), (24, 12))
        self.assertEqual(info.allocation_size_bytes, 36)

    def test_minor_to_major_layout_is_resolved(self) -> None:
        info = resolve_layout(
            (2, 3),
            "f16",
            attributes={
                "layout_source": "stablehlo_encoding",
                "layout_encoding": "#layout<minor_to_major = [0, 1]>",
            },
        )
        self.assertEqual(info.strides_bytes, (2, 4))


class TransformStrideTest(unittest.TestCase):
    def test_transpose_maps_output_tile_to_source_region(self) -> None:
        graph = OperatorGraph(
            graph_id="transpose-stride-test",
            tensors=(
                TensorSpec("value", (2, 3), dtype="f32", attributes={"strides_bytes": [16, 4]}),
                TensorSpec("out", (3, 2), dtype="f32", attributes={"strides_bytes": [8, 4]}),
            ),
            operators=(
                OperatorSpec(
                    op_id="transpose",
                    op_type="transpose",
                    inputs=("value",),
                    outputs=("out",),
                    iteration_dims=(("d0", 3), ("d1", 2)),
                    attributes={
                        "transpose_dims": [1, 0],
                        "frontend_target": "stablehlo.transpose",
                    },
                ),
            ),
        )
        frontend, stablehlo = _frontend(graph)
        compiled = compile_operator_graph(
            graph,
            minimal_machine_config(),
            frontend=frontend,
            source_frontend=frontend,
            stablehlo=stablehlo,
            tile_size=2,
        )
        task = next(task for task in compiled.backend_artifact.execution_graph.tasks if task.primitive == "transpose")
        self.assertEqual(task.reads[0].starts, (0, 0))
        self.assertEqual(task.reads[0].shape, (2, 2))
        self.assertEqual(task.reads[0].strides_bytes, (16, 4))
        self.assertTrue(task.attributes["stride_aware"])

    def test_static_slice_preserves_non_unit_source_stride(self) -> None:
        graph = OperatorGraph(
            graph_id="slice-stride-test",
            tensors=(
                TensorSpec("value", (4, 4), dtype="f32", attributes={"strides_bytes": [32, 4]}),
                TensorSpec("out", (2, 2), dtype="f32"),
            ),
            operators=(
                OperatorSpec(
                    op_id="slice",
                    op_type="slice",
                    inputs=("value",),
                    outputs=("out",),
                    iteration_dims=(("d0", 2), ("d1", 2)),
                    attributes={
                        "slice_starts": [1, 0],
                        "slice_limits": [4, 4],
                        "slice_strides": [2, 2],
                        "frontend_target": "stablehlo.slice",
                    },
                ),
            ),
        )
        frontend, stablehlo = _frontend(graph)
        compiled = compile_operator_graph(
            graph,
            minimal_machine_config(),
            frontend=frontend,
            source_frontend=frontend,
            stablehlo=stablehlo,
            tile_size=2,
        )
        task = next(task for task in compiled.backend_artifact.execution_graph.tasks if task.primitive == "copy")
        self.assertEqual(task.reads[0].offset_bytes, 32)
        self.assertEqual(task.reads[0].size_bytes, 76)
        self.assertEqual(task.reads[0].strides_bytes, (64, 8))


class RuntimeLayoutBindingTest(unittest.TestCase):
    def test_invocation_layout_updates_physical_stride_contract(self) -> None:
        graph = OperatorGraph(
            graph_id="runtime-layout-test",
            tensors=(TensorSpec("x", (4, 3), dtype="f32"), TensorSpec("y", (4, 3), dtype="f32")),
            operators=(
                OperatorSpec(
                    op_id="negate",
                    op_type="elementwise",
                    inputs=("x",),
                    outputs=("y",),
                    iteration_dims=(("d0", 4), ("d1", 3)),
                    attributes={"semantic_op": "negate", "frontend_target": "stablehlo.negate"},
                ),
            ),
        )
        frontend, stablehlo = _frontend(graph)
        compiled = compile_operator_graph(
            graph,
            minimal_machine_config(),
            frontend=frontend,
            source_frontend=frontend,
            stablehlo=stablehlo,
            tile_size=2,
        )
        buffers = list(allocate_buffer_bindings(compiled.graph.tensors))
        buffers[0] = buffers[0].__class__(
            **{**buffers[0].__dict__, "size_bytes": 60}
        )
        submission = create_runtime_submission(
            compiled.backend_artifact,
            tuple(buffers),
            dynamic_layout_bindings=(
            RuntimeLayoutBinding("x", (4, 3), (16, 4), layout="runtime_strided"),
            ),
        )
        x = next(item for item in submission.buffers if item.tensor == "x")
        self.assertEqual(x.attributes["strides_bytes"], [16, 4])
        operand = next(
            item
            for item in submission.operands
            if item.tensor == "x" and item.offset_bytes == 32
        )
        self.assertEqual(operand.attributes["runtime_strides_bytes"], [16, 4])
        self.assertEqual(operand.attributes["address_source"], "dynamic_layout_binding")
        self.assertEqual(submission.validate(compiled.tisa_program), ())


if __name__ == "__main__":
    unittest.main()
