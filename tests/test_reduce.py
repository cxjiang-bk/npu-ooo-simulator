import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.benchmarks import build_reduce_case, build_reduce_model
from npu_ooo.ir import default_reduce_schedule
from npu_ooo.lowering import lower_reduce
from npu_ooo.scheduler import SchedulerPolicy, schedule_execution_graph


class ReduceLoweringTest(unittest.TestCase):
    def test_row_reduce_keeps_partial_accumulation_chain(self) -> None:
        model = build_reduce_model(rows=64, cols=96)
        instance = model.instantiate(build_reduce_case())
        lowered = lower_reduce(instance, minimal_machine_config(), default_reduce_schedule(instance.graph))

        self.assertEqual(lowered.statistics["tile_count"], 6)
        self.assertEqual(lowered.statistics["input_elements"], 64 * 96)
        self.assertEqual(lowered.statistics["task_count"], 14)
        self.assertEqual(lowered.execution_graph.validate(), ())
        reductions = [task for task in lowered.execution_graph.tasks if task.primitive == "reduce"]
        self.assertEqual(len(reductions), 6)
        self.assertEqual(reductions[0].predecessors, ("reduce0.t0000.load",))
        self.assertIn("reduce0.t0000.reduce", reductions[1].predecessors)
        stores = [task for task in lowered.execution_graph.tasks if task.primitive == "store"]
        self.assertEqual(len(stores), 2)

    def test_reduce_scheduler_reports_aru_activity(self) -> None:
        model = build_reduce_model(rows=64, cols=96)
        instance = model.instantiate(build_reduce_case())
        lowered = lower_reduce(instance, minimal_machine_config(), default_reduce_schedule(instance.graph))
        result = schedule_execution_graph(
            lowered.execution_graph,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )
        self.assertEqual(result.metrics["completed_tile_count"], 6)
        self.assertGreater(result.metrics["resource_busy_cycles"]["ARU"], 0)

