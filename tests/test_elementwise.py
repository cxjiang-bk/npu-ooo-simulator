import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.benchmarks import build_elementwise_case, build_elementwise_model
from npu_ooo.ir import default_elementwise_schedule
from npu_ooo.lowering import lower_elementwise
from npu_ooo.scheduler import SchedulerPolicy, schedule_execution_graph


class ElementwiseLoweringTest(unittest.TestCase):
    def test_residual_add_lowers_to_dma_aru_dma(self) -> None:
        model = build_elementwise_model(rows=64, cols=48)
        instance = model.instantiate(build_elementwise_case())
        schedule = default_elementwise_schedule(instance.graph)
        lowered = lower_elementwise(instance, minimal_machine_config(), schedule)

        self.assertEqual(lowered.statistics["tile_count"], 4)
        self.assertEqual(lowered.statistics["elements"], 64 * 48)
        self.assertEqual(lowered.statistics["task_count"], 16)
        self.assertEqual(lowered.execution_graph.validate(), ())
        primitives = [task.primitive for task in lowered.execution_graph.tasks]
        self.assertEqual(primitives[:4], ["load", "load", "elementwise", "store"])
        compute = next(task for task in lowered.execution_graph.tasks if task.primitive == "elementwise")
        self.assertEqual(compute.resource, "ARU")
        self.assertEqual(compute.attributes["input_count"], 2)

    def test_static_and_dynamic_share_elementwise_graph(self) -> None:
        model = build_elementwise_model(rows=64, cols=48)
        instance = model.instantiate(build_elementwise_case())
        lowered = lower_elementwise(instance, minimal_machine_config(), default_elementwise_schedule(instance.graph))
        static = schedule_execution_graph(
            lowered.execution_graph,
            minimal_machine_config(),
            SchedulerPolicy.STATIC_PIPELINE,
        )
        dynamic = schedule_execution_graph(
            lowered.execution_graph,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )
        self.assertEqual(static.graph_id, dynamic.graph_id)
        self.assertEqual(len(static.timings), len(dynamic.timings))
        self.assertEqual(static.metrics["completed_tile_count"], 4)
        self.assertIn("ARU", dynamic.metrics["resource_utilization"])
