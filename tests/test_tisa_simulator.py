import unittest
from dataclasses import replace

from npu_ooo.arch import minimal_machine_config
from npu_ooo.ir import (
    AccessType,
    BackendArtifact,
    BufferBinding,
    ExecutionGraph,
    ExecutionTask,
    TISADependency,
    TISAInstruction,
    TISAOperand,
    TISAProgram,
    TileMem,
    UnitMap,
    create_runtime_submission,
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

    def test_runtime_submission_controls_reception_and_reports_overhead(self) -> None:
        artifact = self._critical_path_artifact()
        buffers = tuple(
            BufferBinding(
                tensor=f"buffer_{tisa_id}",
                base_address=0x1000 + index * 0x100,
                size_bytes=8,
                memory="SRAM",
                logical_scope="SRAM",
                alignment_bytes=0x100,
            )
            for index, tisa_id in enumerate(("short", "critical", "consumer"))
        )
        submission = create_runtime_submission(
            artifact,
            buffers,
            policy="static",
            chunk_size=1,
            launch_latency_cycles=3,
            synchronization_cycles=2,
        )
        result = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
            runtime_submission=submission,
        )

        self.assertEqual(result.metrics["runtime_policy"], "static")
        self.assertEqual(result.metrics["runtime_launch_count"], 3)
        self.assertEqual(result.metrics["runtime_submit_cycles"], 9)
        self.assertEqual(result.metrics["runtime_synchronization_cycles"], 2)
        self.assertEqual(len(result.runtime_timings), 3)
        self.assertGreaterEqual(result.instruction_timing("short").issue, 3)
        self.assertEqual(
            result.total_cycles,
            result.metrics["device_finish_cycle"] + 2,
        )
        self.assertEqual(
            [event.details["runtime_chunk_id"] for event in result.events if event.event == "TISA_RECEIVE"],
            [chunk.chunk_id for chunk in submission.commands],
        )

    def test_runtime_physical_ranges_drive_the_tisa_address_scoreboard(self) -> None:
        shared_operand = TISAOperand(
            name="shared",
            tile_shape=(4,),
            tile_mem=TileMem(base="shared", scope="SRAM", tensor="shared"),
            access_type=AccessType.WRITE,
        )
        instructions = (
            replace(
                _instruction("dma", "tile_dma", "dma"),
                operands=(shared_operand,),
            ),
            replace(
                _instruction("tensor", "tile_tensor", "tensor"),
                operands=(shared_operand,),
            ),
        )
        tasks = (
            ExecutionTask(
                "dma.load",
                "tile_dma",
                "micro",
                "load",
                "DMA",
                duration_cycles=10,
                program_order=0,
            ),
            ExecutionTask(
                "tensor.matmul",
                "tile_tensor",
                "micro",
                "matmul",
                "MXU",
                duration_cycles=10,
                program_order=1,
            ),
        )
        artifact = _artifact(
            instructions,
            tasks,
            {"dma": ("dma.load",), "tensor": ("tensor.matmul",)},
        )
        buffer = BufferBinding(
            tensor="shared",
            base_address=0x2000,
            size_bytes=32,
            memory="SRAM",
            logical_scope="SRAM",
            alignment_bytes=0x100,
        )
        disjoint = create_runtime_submission(
            artifact,
            (buffer,),
            operand_offsets={("dma", "shared"): 0, ("tensor", "shared"): 16},
            operand_sizes={("dma", "shared"): 8, ("tensor", "shared"): 8},
        )
        overlapping = create_runtime_submission(
            artifact,
            (buffer,),
            operand_offsets={("dma", "shared"): 0, ("tensor", "shared"): 0},
            operand_sizes={("dma", "shared"): 8, ("tensor", "shared"): 8},
        )
        config = SimulatorConfig(address_scoreboard=True)

        disjoint_result = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
            simulator_config=config,
            runtime_submission=disjoint,
        )
        overlapping_result = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
            simulator_config=config,
            runtime_submission=overlapping,
        )

        self.assertEqual(disjoint_result.metrics["address_scoreboard_scope"], "runtime_physical")
        self.assertEqual(disjoint_result.instruction_timing("dma").issue, 0)
        self.assertEqual(disjoint_result.instruction_timing("tensor").issue, 0)
        self.assertEqual(overlapping_result.instruction_timing("dma").issue, 0)
        self.assertEqual(overlapping_result.instruction_timing("tensor").issue, 10)
        self.assertGreater(overlapping_result.metrics["address_scoreboard_block_events"], 0)

    def test_dynamic_runtime_bypasses_an_unavailable_independent_descriptor(self) -> None:
        artifact = self._critical_path_artifact()
        buffers = tuple(
            BufferBinding(
                tensor=f"buffer_{tisa_id}",
                base_address=0x3000 + index * 0x100,
                size_bytes=8,
                memory="SRAM",
                logical_scope="SRAM",
                alignment_bytes=0x100,
            )
            for index, tisa_id in enumerate(("short", "critical", "consumer"))
        )
        availability = {"short": 10.0, "critical": 0.0, "consumer": 0.0}
        static_submission = create_runtime_submission(
            artifact,
            buffers,
            policy="static",
            chunk_size=1,
            launch_latency_cycles=1,
            descriptor_available_cycles=availability,
        )
        dynamic_submission = create_runtime_submission(
            artifact,
            buffers,
            policy="dynamic_ready_queue",
            chunk_size=1,
            launch_latency_cycles=1,
            descriptor_available_cycles=availability,
        )

        self.assertEqual(
            [item for chunk in static_submission.commands for item in chunk.tisa_ids],
            ["short", "critical", "consumer"],
        )
        self.assertEqual(
            [item for chunk in dynamic_submission.commands for item in chunk.tisa_ids],
            ["critical", "consumer", "short"],
        )
        static_result = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
            runtime_submission=static_submission,
        )
        dynamic_result = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            SchedulerPolicy.DYNAMIC_READY_QUEUE,
            runtime_submission=dynamic_submission,
        )

        self.assertEqual(static_result.metrics["runtime_submit_cycles"], 13)
        self.assertEqual(dynamic_result.metrics["runtime_submit_cycles"], 11)
        self.assertEqual(static_result.metrics["runtime_request_wait_cycles"], 10)
        self.assertEqual(dynamic_result.metrics["runtime_request_wait_cycles"], 8)
        self.assertLess(
            dynamic_result.instruction_timing("critical").issue,
            static_result.instruction_timing("critical").issue,
        )

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
