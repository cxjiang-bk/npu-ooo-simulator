import unittest
from dataclasses import replace

from npu_ooo.arch import minimal_machine_config
from npu_ooo.ir import (
    AccessType,
    BackendArtifact,
    ExecutionGraph,
    ExecutionTask,
    TISADependency,
    TISAInstruction,
    TISAOperand,
    TISAProgram,
    TileMem,
    UnitMap,
)
from npu_ooo.scheduler import SchedulerPolicy, SimulatorConfig, schedule_tisa_program


def _operand(name: str, access: AccessType = AccessType.READ) -> TISAOperand:
    return TISAOperand(
        name=name,
        tile_shape=(4,),
        tile_mem=TileMem(
            base=name,
            scope="SRAM",
            tensor=name,
            offset_bytes=0,
            size_bytes=8,
        ),
        access_type=access,
    )


def _instruction(
    tisa_id: str,
    tile_id: str,
    unit: str,
    *,
    dependencies: tuple[str, ...] = (),
) -> TISAInstruction:
    return TISAInstruction(
        tisa_id=tisa_id,
        tile_id=tile_id,
        operator_id="micro",
        op_type="micro",
        operands=(_operand(f"buffer_{tisa_id}"),),
        unit_map=UnitMap(unit),
        dependencies=tuple(TISADependency(source) for source in dependencies),
        payload_ref=f"payload:{tisa_id}",
    )


def _artifact(
    instructions: tuple[TISAInstruction, ...],
    tasks: tuple[ExecutionTask, ...],
    payloads: dict[str, tuple[str, ...]],
) -> BackendArtifact:
    return BackendArtifact(
        artifact_id="micro.backend",
        program=TISAProgram("micro.tisa", instructions),
        execution_graph=ExecutionGraph("micro.execution", tasks),
        payloads=payloads,
    )


class TISADeviceSimulatorTest(unittest.TestCase):
    def _critical_path_artifact(self) -> BackendArtifact:
        instructions = (
            _instruction("short", "tile_short", "dma"),
            _instruction("critical", "tile_critical", "dma"),
            _instruction("consumer", "tile_consumer", "tensor", dependencies=("critical",)),
        )
        tasks = (
            ExecutionTask(
                "short.load",
                "tile_short",
                "micro",
                "load",
                "DMA",
                duration_cycles=10,
                program_order=0,
            ),
            ExecutionTask(
                "critical.load",
                "tile_critical",
                "micro",
                "load",
                "DMA",
                duration_cycles=10,
                program_order=1,
            ),
            ExecutionTask(
                "consumer.matmul",
                "tile_consumer",
                "micro",
                "matmul",
                "MXU",
                predecessors=("critical.load",),
                duration_cycles=100,
                program_order=2,
            ),
        )
        return _artifact(
            instructions,
            tasks,
            {
                "short": ("short.load",),
                "critical": ("critical.load",),
                "consumer": ("consumer.matmul",),
            },
        )

    def test_static_and_dynamic_schedule_the_same_tisa_artifact(self) -> None:
        artifact = self._critical_path_artifact()
        machine = minimal_machine_config()

        static = schedule_tisa_program(
            artifact, machine, SchedulerPolicy.STATIC_PIPELINE
        )
        dynamic = schedule_tisa_program(
            artifact, machine, SchedulerPolicy.DYNAMIC_READY_QUEUE
        )

        self.assertEqual(static.instruction_timing("short").issue, 0)
        self.assertEqual(static.instruction_timing("critical").issue, 10)
        self.assertEqual(dynamic.instruction_timing("critical").issue, 0)
        self.assertEqual(dynamic.instruction_timing("short").issue, 10)
        self.assertLess(dynamic.total_cycles, static.total_cycles)
        self.assertEqual(static.metrics["tisa_decision_count"], 3)
        self.assertEqual(dynamic.metrics["tisa_decision_count"], 3)
        self.assertEqual(static.metrics["payload_task_count"], 3)
        self.assertEqual(dynamic.metrics["payload_task_count"], 3)

    def test_tisa_dependency_requires_source_completion(self) -> None:
        artifact = self._critical_path_artifact()
        result = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )

        source = result.instruction_timing("critical")
        consumer = result.instruction_timing("consumer")
        self.assertGreaterEqual(consumer.issue, source.finish)
        self.assertGreaterEqual(result.timing("consumer.matmul").start, consumer.issue)

    def test_payload_runs_locally_after_one_tisa_issue(self) -> None:
        instruction = _instruction("softmax", "tile_softmax", "vector")
        tasks = (
            ExecutionTask(
                "softmax.exp",
                "tile_softmax",
                "micro",
                "exp",
                "ARU",
                duration_cycles=4,
                program_order=0,
            ),
            ExecutionTask(
                "softmax.normalize",
                "tile_softmax",
                "micro",
                "normalize",
                "ARU",
                predecessors=("softmax.exp",),
                duration_cycles=6,
                program_order=1,
            ),
        )
        artifact = _artifact(
            (instruction,),
            tasks,
            {"softmax": ("softmax.exp", "softmax.normalize")},
        )

        result = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
        )

        parent = result.instruction_timing("softmax")
        first = result.timing("softmax.exp")
        second = result.timing("softmax.normalize")
        self.assertEqual(result.metrics["tisa_decision_count"], 1)
        self.assertEqual(first.start, parent.issue)
        self.assertEqual(second.start, first.finish)
        self.assertEqual(parent.finish, second.finish)
        self.assertEqual(
            [
                event.details.get("parent_tisa_id")
                for event in result.events
                if event.event == "START"
            ],
            ["softmax", "softmax"],
        )

    def test_window_one_prevents_dynamic_bypass(self) -> None:
        artifact = self._critical_path_artifact()
        result = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
            simulator_config=SimulatorConfig(
                instruction_queue_depth=1,
                dependency_window=1,
                ready_queue_depth=1,
                rob_entries=1,
                max_inflight_tiles=1,
            ),
        )

        self.assertEqual(result.instruction_timing("short").issue, 0)
        self.assertEqual(result.instruction_timing("critical").issue, 10)
        self.assertEqual(result.metrics["rob_peak"], 1)

    def test_backend_artifact_rejects_duplicate_unowned_and_multi_resource_payloads(self) -> None:
        first = _instruction("first", "tile_first", "dma")
        second = _instruction("second", "tile_second", "dma")
        first_task = ExecutionTask(
            "first.load",
            "tile_first",
            "micro",
            "load",
            "DMA",
            duration_cycles=1,
        )
        second_task = ExecutionTask(
            "second.load",
            "tile_second",
            "micro",
            "load",
            "DMA",
            duration_cycles=1,
        )
        duplicate = _artifact(
            (first, replace(second, tile_id="tile_first")),
            (first_task,),
            {"first": ("first.load",), "second": ("first.load",)},
        )
        self.assertTrue(any("belongs to both" in issue for issue in duplicate.validate()))

        unowned = _artifact(
            (first,),
            (first_task, second_task),
            {"first": ("first.load",)},
        )
        self.assertTrue(any("is not owned" in issue for issue in unowned.validate()))

        mxu_task = replace(
            first_task,
            task_id="first.matmul",
            primitive="matmul",
            resource="MXU",
        )
        multi_resource = _artifact(
            (first,),
            (first_task, mxu_task),
            {"first": ("first.load", "first.matmul")},
        )
        self.assertTrue(
            any("spans multiple resources" in issue for issue in multi_resource.validate())
        )


if __name__ == "__main__":
    unittest.main()
