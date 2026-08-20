import unittest

from npu_ooo.arch import minimal_machine_config
from npu_ooo.benchmarks import build_softmax_case, build_softmax_model
from npu_ooo.ir import default_softmax_schedule
from npu_ooo.lowering import lower_softmax
from npu_ooo.scheduler import SchedulerPolicy, SimulatorConfig, schedule_execution_graph


class SoftmaxLoweringTest(unittest.TestCase):
    def setUp(self) -> None:
        model = build_softmax_model(rows=64, cols=96)
        self.instance = model.instantiate(build_softmax_case())
        self.lowered = lower_softmax(
            self.instance,
            minimal_machine_config(),
            default_softmax_schedule(self.instance.graph),
        )

    def test_softmax_expands_composite_stage_dependencies(self) -> None:
        self.assertEqual(self.lowered.statistics["tile_count"], 6)
        self.assertEqual(self.lowered.statistics["task_count"], 36)
        self.assertEqual(self.lowered.statistics["composite_stage_count"], 4)
        tasks = {task.task_id: task for task in self.lowered.execution_graph.tasks}
        self.assertIn("softmax0.t0001.reduce_max", tasks["softmax0.t0002.reduce_max"].predecessors)
        self.assertIn("softmax0.t0002.reduce_max", tasks["softmax0.t0000.exp"].predecessors)
        self.assertIn("softmax0.t0001.reduce_sum", tasks["softmax0.t0002.reduce_sum"].predecessors)
        self.assertIn("softmax0.t0002.reduce_sum", tasks["softmax0.t0000.normalize"].predecessors)
        self.assertEqual(self.lowered.execution_graph.validate(), ())

    def test_dynamic_priority_is_an_experiment_dimension(self) -> None:
        machine = minimal_machine_config()
        critical = schedule_execution_graph(
            self.lowered.execution_graph,
            machine,
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
            simulator_config=SimulatorConfig(
                dependency_window=8,
                rob_entries=8,
                dynamic_priority="critical_path",
            ),
        )
        oldest = schedule_execution_graph(
            self.lowered.execution_graph,
            machine,
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
            simulator_config=SimulatorConfig(
                dependency_window=8,
                rob_entries=8,
                dynamic_priority="oldest_first",
            ),
        )
        self.assertLess(oldest.total_cycles, critical.total_cycles)
        self.assertEqual(oldest.metrics["simulator_config"]["dynamic_priority"], "oldest_first")

