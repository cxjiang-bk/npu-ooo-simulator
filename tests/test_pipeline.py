import unittest

from npu_ooo.arch import minimal_machine_config, wide_mxu_machine_config
from npu_ooo.benchmarks import build_two_matmul_case, build_two_matmul_model
from npu_ooo.ir import (
    EvaluationScope,
    OperatorSchedule,
    ScheduleSpec,
    default_two_matmul_schedule,
    enumerate_operator_tiles,
)
from npu_ooo.lowering import lower_two_matmul
from npu_ooo.scheduler import SchedulerPolicy, schedule_execution_graph


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        model = build_two_matmul_model()
        case = build_two_matmul_case(evaluation_scope=EvaluationScope.FULL_MODEL)
        self.instance = model.instantiate(case)
        self.schedule = default_two_matmul_schedule(self.instance.graph)

    def test_tile_expansion_retains_boundary_shapes(self) -> None:
        operator = self.instance.graph.operators[0]
        schedule = self.schedule.for_operator(operator.op_id)
        tiles = enumerate_operator_tiles(operator, schedule)
        self.assertEqual(len(tiles), 24)
        self.assertEqual(tiles[-1].bound_map["M"], (96, 128))
        self.assertEqual(tiles[-1].bound_map["L"], (64, 96))
        self.assertEqual(tiles[-1].bound_map["K"], (32, 64))

    def test_lowering_statistics_and_cross_operator_dependency(self) -> None:
        lowered = lower_two_matmul(self.instance, minimal_machine_config(), self.schedule)
        self.assertEqual(lowered.statistics["tile_count"], 60)
        self.assertEqual(lowered.statistics["macs"], 1_769_472)
        self.assertEqual(lowered.execution_graph.validate(), ())
        gemm1_loads = [
            task
            for task in lowered.execution_graph.tasks
            if task.operator_id == "gemm1" and task.attributes.get("operand") == "lhs"
        ]
        self.assertTrue(gemm1_loads)
        self.assertTrue(any(pred.startswith("gemm0.") and pred.endswith(".store") for pred in gemm1_loads[0].predecessors))

    def test_policies_share_graph_but_change_cycle_result(self) -> None:
        lowered = lower_two_matmul(self.instance, minimal_machine_config(), self.schedule)
        results = {
            policy: schedule_execution_graph(lowered.execution_graph, minimal_machine_config(), policy)
            for policy in SchedulerPolicy
        }
        self.assertEqual({result.graph_id for result in results.values()}, {lowered.execution_graph.graph_id})
        self.assertLess(results[SchedulerPolicy.STATIC_PIPELINE].total_cycles, results[SchedulerPolicy.SEQUENTIAL].total_cycles)
        self.assertLessEqual(results[SchedulerPolicy.DYNAMIC_READY_QUEUE].total_cycles, results[SchedulerPolicy.STATIC_PIPELINE].total_cycles)
        self.assertEqual(len(results[SchedulerPolicy.DYNAMIC_READY_QUEUE].timings), len(lowered.execution_graph.tasks))
        self.assertTrue(results[SchedulerPolicy.DYNAMIC_READY_QUEUE].perfetto_trace()["traceEvents"])

    def test_architecture_profile_changes_timing_without_lowering_change(self) -> None:
        minimal = lower_two_matmul(self.instance, minimal_machine_config(), self.schedule)
        wide = lower_two_matmul(self.instance, wide_mxu_machine_config(), self.schedule)
        minimal_shape = [
            (task.task_id, task.primitive, task.resource, task.predecessors)
            for task in minimal.execution_graph.tasks
        ]
        wide_shape = [
            (task.task_id, task.primitive, task.resource, task.predecessors)
            for task in wide.execution_graph.tasks
        ]
        self.assertEqual(minimal_shape, wide_shape)
        minimal_result = schedule_execution_graph(minimal.execution_graph, minimal_machine_config(), SchedulerPolicy.STATIC_PIPELINE)
        wide_result = schedule_execution_graph(wide.execution_graph, wide_mxu_machine_config(), SchedulerPolicy.STATIC_PIPELINE)
        self.assertLessEqual(wide_result.total_cycles, minimal_result.total_cycles)
        self.assertLess(
            wide_result.metrics["resource_busy_cycles"]["MXU"],
            minimal_result.metrics["resource_busy_cycles"]["MXU"],
        )

    def test_two_mm_schedule_tile_size_changes_tile_graph_only(self) -> None:
        coarse = lower_two_matmul(
            self.instance,
            minimal_machine_config(),
            ScheduleSpec(
                "two_mm_tile16",
                tuple(
                    OperatorSchedule(
                        operator_id=operator.op_id,
                        tile_sizes=tuple(
                            (name, min(16, int(extent)))
                            for name, extent in (*operator.iteration_dims, *operator.reduction_dims)
                        ),
                    )
                    for operator in self.instance.graph.operators
                ),
            ),
        )
        fine = lower_two_matmul(self.instance, minimal_machine_config(), default_two_matmul_schedule(self.instance.graph))
        self.assertGreater(coarse.statistics["tile_count"], fine.statistics["tile_count"])
        self.assertNotEqual(coarse.statistics["task_count"], fine.statistics["task_count"])


if __name__ == "__main__":
    unittest.main()
