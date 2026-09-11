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
    allocate_memory_plan_bindings,
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
    def test_common_operator_tiling_preserves_dependencies_and_memory_plan(self) -> None:
        graph = OperatorGraph(
            graph_id="common-tiling-contract-test",
            tensors=(
                TensorSpec("lhs", (4, 8), dtype="f32"),
                TensorSpec("rhs", (4, 8), dtype="f32"),
                TensorSpec("sum", (4, 8), dtype="f32"),
                TensorSpec("reduced", (4,), dtype="f32"),
            ),
            operators=(
                OperatorSpec(
                    op_id="add",
                    op_type="elementwise",
                    inputs=("lhs", "rhs"),
                    outputs=("sum",),
                    iteration_dims=(("d0", 4), ("d1", 8)),
                    attributes={"semantic_op": "add", "frontend_target": "stablehlo.add"},
                ),
                OperatorSpec(
                    op_id="reduce",
                    op_type="reduce",
                    inputs=("sum",),
                    outputs=("reduced",),
                    iteration_dims=(("d0", 4),),
                    reduction_dims=(("d1", 8),),
                    attributes={"reducer": "add", "frontend_target": "stablehlo.reduce"},
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
        self.assertEqual(compiled.validate(), ())
        self.assertGreater(len(compiled.tile_graph.tiles), 8)
        self.assertTrue(any(dependency.tensor == "sum" for dependency in compiled.tile_graph.dependencies))
        self.assertTrue(compiled.backend_artifact.memory_plan.buffers)
        self.assertTrue(
            all(
                task.attributes.get("target_tisa_id")
                for task in compiled.backend_artifact.execution_graph.tasks
            )
        )

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

    def test_static_slice_maps_each_output_tile_to_strided_source_region(self) -> None:
        graph = OperatorGraph(
            graph_id="slice-multi-tile-test",
            tensors=(
                TensorSpec("value", (1, 1, 4, 16), dtype="f32"),
                TensorSpec("out", (1, 1, 4, 8), dtype="f32"),
            ),
            operators=(
                OperatorSpec(
                    op_id="slice",
                    op_type="slice",
                    inputs=("value",),
                    outputs=("out",),
                    iteration_dims=(("d0", 1), ("d1", 1), ("d2", 4), ("d3", 8)),
                    attributes={
                        "slice_starts": [0, 0, 0, 2],
                        "slice_limits": [1, 1, 4, 16],
                        "slice_strides": [1, 1, 1, 2],
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
            tile_size=4,
        )
        tasks = [
            task
            for task in compiled.backend_artifact.execution_graph.tasks
            if task.operator_id == "slice"
        ]
        self.assertEqual(len(tasks), 2)
        self.assertEqual([task.reads[0].starts for task in tasks], [(0, 0, 0, 2), (0, 0, 0, 10)])
        self.assertTrue(all(task.reads[0].shape == (1, 1, 4, 4) for task in tasks))
        self.assertTrue(all(task.reads[0].strides_bytes[-1] == 8 for task in tasks))

    def test_reshape_cross_boundary_regions_and_dependency_are_preserved(self) -> None:
        graph = OperatorGraph(
            graph_id="reshape-cross-boundary-test",
            tensors=(
                TensorSpec("value", (2, 3), dtype="f32"),
                TensorSpec("reshaped", (3, 2), dtype="f32"),
                TensorSpec("out", (3, 2), dtype="f32"),
            ),
            operators=(
                OperatorSpec(
                    op_id="reshape",
                    op_type="reshape",
                    inputs=("value",),
                    outputs=("reshaped",),
                    iteration_dims=(("d0", 3), ("d1", 2)),
                    attributes={"frontend_target": "stablehlo.reshape"},
                ),
                OperatorSpec(
                    op_id="consumer",
                    op_type="elementwise",
                    inputs=("reshaped",),
                    outputs=("out",),
                    iteration_dims=(("d0", 3), ("d1", 2)),
                    attributes={"semantic_op": "copy", "frontend_target": "stablehlo.copy"},
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
        reshape_task = next(
            task
            for task in compiled.backend_artifact.execution_graph.tasks
            if task.operator_id == "reshape" and task.tile_id.endswith("t0000")
        )
        self.assertEqual(len(reshape_task.reads), 3)
        self.assertEqual(len(reshape_task.writes), 3)
        self.assertEqual(
            [region.starts for region in reshape_task.reads],
            [(0, 0), (0, 2), (1, 0)],
        )
        edge = next(
            dependency
            for dependency in compiled.tile_graph.dependencies
            if dependency.tensor == "reshaped"
            and dependency.producer.startswith("reshape.")
        )
        self.assertEqual(len(edge.producer_regions), 3)
        self.assertEqual(edge.producer_regions[1][0], (1, 0))
        self.assertEqual(edge.producer_regions[2][0], (1, 1))
        self.assertEqual(edge.consumer_region, ((0, 0), (2, 2)))

    def test_concatenate_cross_input_boundary_regions_and_dependency_are_preserved(self) -> None:
        graph = OperatorGraph(
            graph_id="concatenate-cross-boundary-test",
            tensors=(
                TensorSpec("left", (2, 3), dtype="f32"),
                TensorSpec("right", (2, 3), dtype="f32"),
                TensorSpec("joined", (2, 6), dtype="f32"),
                TensorSpec("out", (2, 6), dtype="f32"),
            ),
            operators=(
                OperatorSpec(
                    op_id="concatenate",
                    op_type="concatenate",
                    inputs=("left", "right"),
                    outputs=("joined",),
                    iteration_dims=(("d0", 2), ("d1", 6)),
                    attributes={
                        "concatenate_dimension": 1,
                        "frontend_target": "stablehlo.concatenate",
                    },
                ),
                OperatorSpec(
                    op_id="consumer",
                    op_type="elementwise",
                    inputs=("joined",),
                    outputs=("out",),
                    iteration_dims=(("d0", 2), ("d1", 6)),
                    attributes={"semantic_op": "copy", "frontend_target": "stablehlo.copy"},
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
            tile_size=4,
        )
        concat_tasks = [
            task
            for task in compiled.backend_artifact.execution_graph.tasks
            if task.operator_id == "concatenate"
        ]
        self.assertEqual(len(concat_tasks), 3)
        first_tile = [task for task in concat_tasks if task.tile_id.endswith("t0000")]
        self.assertEqual(len(first_tile), 2)
        self.assertEqual(
            {(task.reads[0].tensor, task.reads[0].starts, task.writes[0].starts) for task in first_tile},
            {("left", (0, 0), (0, 0)), ("right", (0, 0), (0, 3))},
        )
        edge = next(
            dependency
            for dependency in compiled.tile_graph.dependencies
            if dependency.tensor == "joined"
            and dependency.producer.startswith("concatenate.")
        )
        self.assertEqual(len(edge.producer_regions), 2)
        self.assertEqual(edge.consumer_region, ((0, 0), (2, 4)))

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
        buffers = list(
            allocate_memory_plan_bindings(
                compiled.backend_artifact.memory_plan,
                minimal_machine_config(),
            )
        )
        root_index = next(
            index
            for index, buffer in enumerate(buffers)
            if buffer.tensor == "x" and buffer.attributes.get("external")
        )
        buffers[root_index] = buffers[root_index].__class__(
            **{**buffers[root_index].__dict__, "size_bytes": 60}
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
