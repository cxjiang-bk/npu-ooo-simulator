import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.benchmarks import build_decoder_block_case, build_decoder_block_model
from npu_ooo.ir import default_mixed_schedule
from npu_ooo.lowering import default_lowering_registry, lower_mixed_model
from npu_ooo.scheduler import SchedulerPolicy, schedule_execution_graph


class MixedLoweringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = build_decoder_block_model(tokens=64, hidden=96)
        self.instance = self.model.instantiate(build_decoder_block_case(tokens=64))
        self.machine = minimal_machine_config()
        self.schedule = default_mixed_schedule(self.instance.graph)
        self.lowered = lower_mixed_model(self.instance, self.machine, self.schedule)

    def test_registry_covers_current_semantic_lowerers(self) -> None:
        self.assertEqual(
            default_lowering_registry().supported_types,
            (
                "batched_matmul",
                "elementwise",
                "gemv",
                "layernorm",
                "matmul",
                "reduce",
                "residual_add",
                "rmsnorm",
                "softmax",
            ),
        )

    def test_mixed_graph_connects_root_memory_handoffs(self) -> None:
        graph = self.lowered.execution_graph
        self.assertEqual(graph.validate(), ())
        self.assertEqual(
            [schedule.stage_id for schedule in self.schedule.operator_schedules],
            [0, 1, 2],
        )
        self.assertEqual(
            [task.program_order for task in graph.tasks],
            list(range(len(graph.tasks))),
        )
        primitives = {task.primitive for task in graph.tasks}
        self.assertTrue({"rmsnorm", "matmul", "elementwise"}.issubset(primitives))

        projection_loads = [
            task
            for task in graph.tasks
            if task.operator_id == "projection0" and task.attributes.get("operand") == "lhs"
        ]
        self.assertTrue(projection_loads)
        self.assertTrue(
            all(
                any(
                    predecessor.startswith("rmsnorm0.") and predecessor.endswith(".store")
                    for predecessor in task.predecessors
                )
                for task in projection_loads
            )
        )
        residual_projection_loads = [
            task
            for task in graph.tasks
            if task.operator_id == "residual0" and task.attributes.get("operand") == 0
        ]
        self.assertTrue(residual_projection_loads)
        self.assertTrue(
            all(
                any(
                    predecessor.startswith("projection0.") and predecessor.endswith(".store")
                    for predecessor in task.predecessors
                )
                for task in residual_projection_loads
            )
        )
        self.assertGreater(self.lowered.statistics["cross_operator_dependency_count"], 0)

    def test_static_and_dynamic_schedule_the_same_mixed_graph(self) -> None:
        static = schedule_execution_graph(
            self.lowered.execution_graph,
            self.machine,
            SchedulerPolicy.STATIC_PIPELINE,
        )
        dynamic = schedule_execution_graph(
            self.lowered.execution_graph,
            self.machine,
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )
        self.assertEqual(static.graph_id, dynamic.graph_id)
        self.assertEqual(len(static.timings), len(self.lowered.execution_graph.tasks))
        self.assertEqual(len(dynamic.timings), len(self.lowered.execution_graph.tasks))
        self.assertEqual(
            dynamic.metrics["completed_tile_count"],
            len(self.lowered.tile_graph.tiles),
        )


if __name__ == "__main__":
    unittest.main()
