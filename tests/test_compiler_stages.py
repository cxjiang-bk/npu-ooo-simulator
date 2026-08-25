import unittest
from dataclasses import replace

from npu_ooo.arch import minimal_machine_config
from npu_ooo.compiler import compile_operator_graph
from npu_ooo.frontend import FrontendImport, OfficialStableHLOModule
from npu_ooo.ir import OperatorGraph, OperatorSpec, TensorSpec


def _stable_frontend(graph: OperatorGraph) -> tuple[FrontendImport, OfficialStableHLOModule]:
    frontend = FrontendImport(
        graph=graph,
        model_id=graph.graph_id,
        variant="test",
        frontend="stablehlo",
    )
    stablehlo = OfficialStableHLOModule(
        text="module {}",
        canonical_text="module {}",
        model_id=graph.graph_id,
    )
    return frontend, stablehlo


class CompilerStageContractTest(unittest.TestCase):
    def test_gc_keeps_pass_snapshots_and_memory_plan_metadata(self) -> None:
        graph = OperatorGraph(
            graph_id="gc-metadata-test",
            tensors=(TensorSpec("x", (4, 8)), TensorSpec("y", (4, 8))),
            operators=(
                OperatorSpec(
                    op_id="softmax",
                    op_type="softmax",
                    inputs=("x",),
                    outputs=("y",),
                    iteration_dims=(("row", 4),),
                    reduction_dims=(("col", 8),),
                ),
            ),
        )
        frontend, stablehlo = _stable_frontend(graph)
        machine = minimal_machine_config()
        machine = replace(
            machine,
            memory_levels=tuple(
                replace(level, capacity_bytes=16) if level.name == "SRAM" else level
                for level in machine.memory_levels
            ),
        )
        compiled = compile_operator_graph(
            graph,
            machine,
            frontend=frontend,
            source_frontend=frontend,
            stablehlo=stablehlo,
            tile_size=4,
        )

        snapshots = compiled.gc_artifact.pass_dumps
        self.assertTrue(snapshots)
        self.assertEqual(
            [snapshot.pass_index for snapshot in snapshots],
            list(range(len(snapshots))),
        )
        self.assertEqual(snapshots[0].input_graph.to_dict(), graph.to_dict())
        self.assertEqual(snapshots[-1].output_graph.to_dict(), compiled.gc_artifact.graph.to_dict())
        self.assertEqual(compiled.gc_artifact.attributes["pass_count"], len(snapshots))

        operator_schedule = compiled.schedule.for_operator("softmax")
        ping_pong = operator_schedule.attributes["ping_pong"]
        self.assertTrue(ping_pong["enabled"])
        self.assertEqual(ping_pong["buffer_count"], 2)
        self.assertEqual(operator_schedule.attributes["residency_overflow_tensors"], ["x", "y"])

    def test_gc_fc_and_tisa_generator_have_distinct_contracts(self) -> None:
        graph = OperatorGraph(
            graph_id="stage-test",
            tensors=(
                TensorSpec("lhs", (4, 4)),
                TensorSpec("rhs", (4, 4)),
                TensorSpec("out", (4, 4)),
            ),
            operators=(
                OperatorSpec(
                    op_id="mm",
                    op_type="matmul",
                    inputs=("lhs", "rhs"),
                    outputs=("out",),
                    iteration_dims=(("m", 4), ("n", 4)),
                    reduction_dims=(("k", 4),),
                ),
            ),
        )
        frontend, stablehlo = _stable_frontend(graph)
        compiled = compile_operator_graph(
            graph,
            minimal_machine_config(),
            frontend=frontend,
            source_frontend=frontend,
            stablehlo=stablehlo,
            tile_size=2,
        )

        self.assertEqual(compiled.gc_artifact.to_dict()["paper_stage"], "GC")
        self.assertEqual(compiled.tisa_dialect.to_dict()["paper_stage"], "FC")
        self.assertEqual(compiled.tisa_program.attributes["paper_stage"], "TISA_GENERATOR")
        self.assertEqual(compiled.validate(), ())

    def test_composite_payload_is_not_scheduler_visible(self) -> None:
        graph = OperatorGraph(
            graph_id="softmax-stage-test",
            tensors=(TensorSpec("x", (4, 8)), TensorSpec("y", (4, 8))),
            operators=(
                OperatorSpec(
                    op_id="softmax",
                    op_type="softmax",
                    inputs=("x",),
                    outputs=("y",),
                    iteration_dims=(("row", 4),),
                    reduction_dims=(("col", 8),),
                ),
            ),
        )
        frontend, stablehlo = _stable_frontend(graph)
        compiled = compile_operator_graph(
            graph,
            minimal_machine_config(),
            frontend=frontend,
            source_frontend=frontend,
            stablehlo=stablehlo,
            tile_size=4,
        )

        visible_ops = {instruction.op_type for instruction in compiled.tisa_program.instructions}
        payload_ops = {
            task.primitive for task in compiled.backend_artifact.execution_graph.tasks
        }
        self.assertIn("softmax", visible_ops)
        self.assertNotIn("reduce_max", visible_ops)
        self.assertTrue({"reduce_max", "exp", "reduce_sum", "normalize"}.issubset(payload_ops))
        self.assertEqual(compiled.backend_artifact.validate(), ())


if __name__ == "__main__":
    unittest.main()
