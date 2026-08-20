import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.benchmarks import build_rmsnorm_case, build_rmsnorm_model
from npu_ooo.ir import default_rmsnorm_schedule
from npu_ooo.lowering import lower_rmsnorm
from npu_ooo.scheduler import SchedulerPolicy, schedule_execution_graph


class RMSNormLoweringTest(unittest.TestCase):
    def test_rmsnorm_keeps_sum_square_barrier(self) -> None:
        model = build_rmsnorm_model(rows=64, cols=96)
        instance = model.instantiate(build_rmsnorm_case())
        lowered = lower_rmsnorm(instance, minimal_machine_config(), default_rmsnorm_schedule(instance.graph))
        self.assertEqual(lowered.statistics["tile_count"], 6)
        self.assertEqual(lowered.statistics["task_count"], 30)
        self.assertEqual(lowered.statistics["input_elements"], 64 * 96)
        self.assertEqual(lowered.execution_graph.validate(), ())
        tasks = {task.task_id: task for task in lowered.execution_graph.tasks}
        self.assertIn("rmsnorm0.t0000.reduce_sum_square", tasks["rmsnorm0.t0001.reduce_sum_square"].predecessors)
        self.assertIn("rmsnorm0.t0002.reduce_sum_square", tasks["rmsnorm0.t0000.rmsnorm"].predecessors)

    def test_rmsnorm_runs_on_aru(self) -> None:
        model = build_rmsnorm_model(rows=64, cols=96)
        instance = model.instantiate(build_rmsnorm_case())
        lowered = lower_rmsnorm(instance, minimal_machine_config(), default_rmsnorm_schedule(instance.graph))
        result = schedule_execution_graph(
            lowered.execution_graph,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )
        self.assertEqual(result.metrics["completed_tile_count"], 6)
        self.assertGreater(result.metrics["resource_busy_cycles"]["ARU"], 0)
