from dataclasses import replace
import unittest

from npu_ooo.arch import SchedulerPipelineConfig, minimal_machine_config
from npu_ooo.backend import CycleEventBackend
from npu_ooo.compiler import attach_static_control
from npu_ooo.execution import AnalyticalExecutionBackend
from npu_ooo.ir import (
    BackendArtifact,
    BufferBinding,
    ExecutionGraph,
    ExecutionTask,
    StaticControlProgram,
    StaticControlCommand,
    StaticEvent,
    TISADependency,
    TISAInstruction,
    TISAOperand,
    TISAProgram,
    TileMem,
    UnitMap,
    create_runtime_sequence,
    create_runtime_state_registry,
    create_runtime_submission,
)
from npu_ooo.scheduler import SimulatorConfig, schedule_tisa_program, schedule_tisa_sequence
from npu_ooo.scheduler.static import schedule_loaded_static_program
from npu_ooo.runtime import load_implicit_device_program
from npu_ooo.simulator import TimingTableModel


def _instruction(tisa_id, resource, dependencies=(), *, kind="RAW"):
    return TISAInstruction(
        tisa_id=tisa_id,
        tile_id=f"tile.{tisa_id}",
        operator_id=tisa_id,
        op_type="elementwise",
        operands=(
            TISAOperand(
                "operand",
                (4,),
                TileMem(
                    tisa_id,
                    "SRAM",
                    tensor=tisa_id,
                    offset_bytes=0,
                    size_bytes=8,
                ),
                "read",
            ),
        ),
        unit_map=UnitMap(resource),
        dependencies=tuple(
            TISADependency(
                source,
                kind,
                "full_region_ready" if kind != "BUFFER_REUSE" else "allocation_released",
                {"source": "test" if kind != "BUFFER_REUSE" else "target_memory_allocator"},
            )
            for source in dependencies
        ),
        payload_ref=f"payload:{tisa_id}",
        attributes={"stage_id": 0},
    )


def _artifact(*, buffer_reuse=False):
    instructions = (
        _instruction("source", "DMA"),
        _instruction("consumer_mxu", "MXU", ("source",)),
        _instruction("consumer_aru", "ARU", ("source",)),
        _instruction(
            "reuse",
            "DMA",
            ("consumer_mxu",),
            kind="BUFFER_REUSE" if buffer_reuse else "RAW",
        ),
        _instruction("independent", "ARU"),
    )
    durations = {
        "source": 5,
        "consumer_mxu": 3,
        "consumer_aru": 2,
        "reuse": 1,
        "independent": 1,
    }
    tasks = tuple(
        ExecutionTask(
            task_id=f"{item.tisa_id}.task",
            tile_id=item.tile_id,
            operator_id=item.operator_id,
            primitive="load" if item.unit_map.unit == "DMA" else "matmul"
            if item.unit_map.unit == "MXU"
            else "elementwise",
            resource=item.unit_map.unit,
            duration_cycles=durations[item.tisa_id],
            program_order=index,
        )
        for index, item in enumerate(instructions)
    )
    artifact = BackendArtifact(
        artifact_id="static.test",
        program=TISAProgram("static.test.program", instructions),
        execution_graph=ExecutionGraph("static.test.execution", tasks),
        payloads={item.tisa_id: (f"{item.tisa_id}.task",) for item in instructions},
    )
    return attach_static_control(artifact, minimal_machine_config())


class StaticControlCompilerTest(unittest.TestCase):
    def test_three_stage_multi_iteration_pipeline_overlaps_and_fences_double_buffer(self):
        rows = []
        tasks = []
        payloads = {}
        order = 0
        for iteration in range(3):
            slot = iteration % 2
            for stage, (name, resource, duration) in enumerate(
                (("gemm0", "MXU", 5), ("softmax", "ARU", 3), ("gemm1", "MXU", 4))
            ):
                tisa_id = f"{name}.{iteration}"
                dependencies = []
                if stage:
                    source = f"{'gemm0' if stage == 1 else 'softmax'}.{iteration}"
                    dependencies.append(TISADependency(source, "RAW", "full_region_ready"))
                if name == "gemm0" and iteration >= 2:
                    dependencies.append(
                        TISADependency(
                            f"gemm1.{iteration - 2}",
                            "BUFFER_REUSE",
                            "allocation_released",
                            {"source": "target_memory_allocator"},
                        )
                    )
                instruction = TISAInstruction(
                    tisa_id,
                    f"attention.t{iteration:04d}",
                    name,
                    name,
                    (
                        TISAOperand(
                            "slot",
                            (4,),
                            TileMem(
                                f"slot{slot}",
                                "SRAM",
                                tensor=f"slot{slot}",
                                buffer_id=f"slot{slot}",
                                allocation_id=f"slot{slot}",
                                offset_bytes=0,
                                size_bytes=8,
                            ),
                            "read_write",
                        ),
                    ),
                    UnitMap(resource),
                    tuple(dependencies),
                    {"iteration": iteration, "stage_id": stage},
                    f"payload:{tisa_id}",
                )
                rows.append(instruction)
                task_id = f"{tisa_id}.task"
                tasks.append(
                    ExecutionTask(
                        task_id,
                        instruction.tile_id,
                        name,
                        "matmul" if "gemm" in name else "softmax",
                        resource,
                        duration_cycles=duration,
                        program_order=order,
                    )
                )
                payloads[tisa_id] = (task_id,)
                order += 1
        artifact = attach_static_control(
            BackendArtifact(
                "three-stage",
                TISAProgram("three-stage.program", tuple(rows)),
                ExecutionGraph("three-stage.execution", tuple(tasks)),
                payloads,
            ),
            minimal_machine_config(),
        )
        control = artifact.static_control
        fence = next(
            command
            for stream in control.streams
            for command in stream.commands
            if command.kind == "fence" and command.tisa_id == "gemm0.2"
        )
        self.assertIn("SRAM:slot0", fence.buffer_slots)
        result = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            "static_streams",
            simulator_config=SimulatorConfig(
                pipeline=SchedulerPipelineConfig(issue_width=2)
            ),
            event_backend=CycleEventBackend(),
        )
        gemm_next = result.instruction_timing("gemm0.1")
        soft_current = result.instruction_timing("softmax.0")
        self.assertLess(gemm_next.issue, soft_current.finish)
        self.assertLess(soft_current.issue, gemm_next.finish)

    def test_control_contract_roundtrip_and_multiconsumer_event(self):
        artifact = _artifact(buffer_reuse=True)
        control = artifact.static_control
        self.assertIsNotNone(control)
        self.assertEqual(control.validate(set(item.tisa_id for item in artifact.program.instructions)), ())
        restored = StaticControlProgram.from_dict(control.to_dict())
        self.assertEqual(restored.to_dict(), control.to_dict())
        event = next(item for item in control.events if item.source_tisa_id == "source")
        self.assertEqual(set(event.consumers), {"consumer_mxu", "consumer_aru"})
        waits = [
            command
            for stream in control.streams
            for command in stream.commands
            if command.kind == "wait" and event.event_id in command.event_ids
        ]
        self.assertEqual(len(waits), 2)
        self.assertTrue(all(not event.attributes["consuming_wait"] for _item in waits))
        self.assertTrue(
            any(
                command.kind == "fence" and command.tisa_id == "reuse"
                for stream in control.streams
                for command in stream.commands
            )
        )
        self.assertEqual(
            StaticControlProgram.from_dict(control.to_dict()).control_hash,
            control.control_hash,
        )

    def test_missing_wait_fails_correctness_oracle(self):
        artifact = _artifact()
        control = artifact.static_control
        streams = []
        removed = False
        for stream in control.streams:
            commands = []
            for command in stream.commands:
                if not removed and command.kind == "wait":
                    removed = True
                    continue
                commands.append(replace(command, order=len(commands)))
            streams.append(replace(stream, commands=tuple(commands)))
        broken = replace(control, streams=tuple(streams))
        issues = replace(artifact, static_control=broken).validate()
        self.assertTrue(any("lacks wait/fence" in item for item in issues))


class StaticStreamExecutionTest(unittest.TestCase):
    def test_legacy_package_requires_recompile_for_static_streams(self):
        artifact = _artifact()
        legacy = replace(artifact, static_control=None)
        with self.assertRaisesRegex(ValueError, "recompile"):
            schedule_tisa_program(
                legacy,
                minimal_machine_config(),
                "static_streams",
                event_backend=CycleEventBackend(),
            )

    def test_unseen_runtime_alias_is_rejected_not_adaptively_rescheduled(self):
        artifact = _artifact()
        instructions = list(artifact.program.instructions)
        instructions[0] = replace(
            instructions[0],
            operands=(replace(instructions[0].operands[0], access_type="write"),),
        )
        artifact = replace(
            artifact,
            program=replace(artifact.program, instructions=tuple(instructions)),
            static_control=None,
        )
        artifact = attach_static_control(artifact, minimal_machine_config())
        buffers = tuple(
            BufferBinding(
                item.tisa_id,
                0x1000
                if item.tisa_id in {"source", "independent"}
                else 0x2000 + index * 0x100,
                8,
                "SRAM",
                "SRAM",
            )
            for index, item in enumerate(artifact.program.instructions)
        )
        submission = create_runtime_submission(artifact, buffers)
        with self.assertRaisesRegex(ValueError, "runtime binding introduces alias"):
            schedule_tisa_program(
                artifact,
                minimal_machine_config(),
                "static_streams",
                runtime_submission=submission,
                event_backend=CycleEventBackend(),
            )

    def test_set_uses_matching_partial_ready_feedback_not_submission(self):
        source = _instruction("source", "DMA")
        consumer = replace(
            _instruction("consumer", "MXU"),
            dependencies=(
                TISADependency(
                    "source",
                    "RAW",
                    "payload_ready:source.head",
                    {"source": "partial_test"},
                ),
            ),
        )
        artifact = BackendArtifact(
            "partial.static",
            TISAProgram("partial.static.program", (source, consumer)),
            ExecutionGraph(
                "partial.static.execution",
                (
                    ExecutionTask(
                        "source.head",
                        source.tile_id,
                        source.operator_id,
                        "load",
                        "DMA",
                        duration_cycles=3,
                        program_order=0,
                    ),
                    ExecutionTask(
                        "source.tail",
                        source.tile_id,
                        source.operator_id,
                        "load",
                        "DMA",
                        predecessors=("source.head",),
                        duration_cycles=7,
                        program_order=1,
                    ),
                    ExecutionTask(
                        "consumer.task",
                        consumer.tile_id,
                        consumer.operator_id,
                        "matmul",
                        "MXU",
                        duration_cycles=1,
                        program_order=2,
                    ),
                ),
            ),
            {"source": ("source.head", "source.tail"), "consumer": ("consumer.task",)},
        )
        artifact = attach_static_control(artifact, minimal_machine_config())
        result = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            "static_streams",
            event_backend=CycleEventBackend(),
        )
        self.assertGreater(result.instruction_timing("consumer").issue, 0)
        self.assertLess(
            result.instruction_timing("consumer").issue,
            result.instruction_timing("source").finish,
        )
        partial_set = next(
            event
            for event in result.events
            if event.event == "STATIC_SET"
            and any("payload_ready:source.head" in item for item in event.details["event_ids"])
        )
        self.assertGreaterEqual(partial_set.timestamp, 3)

    def test_event_generation_is_isolated_by_invocation(self):
        artifact = _artifact()
        event_sets = []
        for invocation_id in ("invocation.0", "invocation.1"):
            loaded = load_implicit_device_program(
                artifact, invocation_id=invocation_id
            )
            result = schedule_loaded_static_program(
                loaded,
                minimal_machine_config(),
                AnalyticalExecutionBackend(artifact, minimal_machine_config()),
            )
            event_sets.append(
                {
                    event_id
                    for event in result.events
                    if event.event == "STATIC_SET"
                    for event_id in event.details["event_ids"]
                }
            )
        self.assertTrue(event_sets[0])
        self.assertTrue(event_sets[0].isdisjoint(event_sets[1]))

    def test_static_stream_runtime_sequence_preserves_invocation_identity(self):
        artifact = _artifact()
        buffers = tuple(
            BufferBinding(
                item.tisa_id,
                0x1000 + index * 0x100,
                8,
                "SRAM",
                "SRAM",
            )
            for index, item in enumerate(artifact.program.instructions)
        )
        registry = create_runtime_state_registry(artifact, buffers)
        sequence = create_runtime_sequence(
            artifact,
            registry,
            invocation_count=2,
            inter_invocation_gap_cycles=3,
        )
        result = schedule_tisa_sequence(
            artifact,
            sequence,
            minimal_machine_config(),
            "static_streams",
            event_backend=CycleEventBackend(),
        )
        self.assertEqual(result.metrics["invocation_count"], 2)
        self.assertEqual(result.metrics["state_dependency_count"], 0)
        set_ids = {
            event_id
            for event in result.events
            if event.event == "STATIC_SET"
            for event_id in event.details["event_ids"]
        }
        self.assertTrue(
            any(sequence.invocations[0].submission_id in item for item in set_ids)
        )
        self.assertTrue(
            any(sequence.invocations[1].submission_id in item for item in set_ids)
        )

    def test_blocked_stream_does_not_stop_independent_eu_and_waits_real_done(self):
        artifact = _artifact()
        result = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            "static_streams",
            simulator_config=SimulatorConfig(
                pipeline=SchedulerPipelineConfig(issue_width=3)
            ),
            event_backend=CycleEventBackend(),
        )
        self.assertEqual(result.instruction_timing("source").issue, 0)
        self.assertEqual(result.instruction_timing("independent").issue, 0)
        self.assertGreaterEqual(
            result.instruction_timing("consumer_mxu").issue,
            result.instruction_timing("source").finish,
        )
        self.assertEqual(result.metrics["static_control_counts"]["wait"], 3)
        self.assertTrue(result.metrics["static_controls_consumed"])
        wait_events = [
            event for event in result.events if event.event == "STATIC_WAIT_SATISFIED"
        ]
        self.assertEqual(len(wait_events), 3)
        event_ids = [item.details["event_ids"][0] for item in wait_events[:2]]
        self.assertEqual(len(set(event_ids)), 1)

    def test_actual_latency_not_compile_estimate_releases_set(self):
        artifact = _artifact()
        timing = TimingTableModel.from_dict(
            {
                "entries": {
                    "source.task": {
                        "duration_cycles": 20,
                        "initiation_interval_cycles": 1,
                    }
                }
            }
        )
        result = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            "static_streams",
            timing_model=timing,
            event_backend=CycleEventBackend(),
        )
        self.assertGreaterEqual(result.instruction_timing("consumer_mxu").issue, 20)
        source_set = next(
            event
            for event in result.events
            if event.event == "STATIC_SET"
            and any("source" in item for item in event.details["event_ids"])
        )
        self.assertGreaterEqual(source_set.timestamp, 20)

    def test_zero_cost_control_is_explicit_and_dynamic_ignores_control(self):
        artifact = _artifact()
        zero = SimulatorConfig(
            pipeline=SchedulerPipelineConfig(
                control_latency=0,
                wait_latency=0,
                fence_latency=0,
            )
        )
        static = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            "static_streams",
            simulator_config=zero,
            event_backend=CycleEventBackend(),
        )
        self.assertEqual(static.metrics["static_control_busy_cycles"], 0)
        baseline_dynamic = schedule_tisa_program(
            artifact,
            minimal_machine_config(),
            "dynamic_ready_queue",
            event_backend=CycleEventBackend(),
        )
        extra_event = StaticEvent(
            "event.extra.static-order",
            "source",
            "static_extra_order",
            "stream",
            999,
            consumers=("independent",),
            source_kind="static_pipeline_order",
        )
        changed_streams = []
        for stream in artifact.static_control.streams:
            commands = list(stream.commands)
            source_positions = [
                index for index, item in enumerate(commands) if item.tisa_id == "source"
            ]
            if source_positions:
                position = max(source_positions) + 1
                commands.insert(
                    position,
                    StaticControlCommand(
                        "extra.set",
                        "set",
                        stream.stream_id,
                        position,
                        tisa_id="source",
                        event_ids=(extra_event.event_id,),
                        source_kind="static_pipeline_order",
                    ),
                )
            independent_position = next(
                (
                    index
                    for index, item in enumerate(commands)
                    if item.kind == "issue" and item.tisa_id == "independent"
                ),
                None,
            )
            if independent_position is not None:
                commands.insert(
                    independent_position,
                    StaticControlCommand(
                        "extra.wait",
                        "wait",
                        stream.stream_id,
                        independent_position,
                        tisa_id="independent",
                        event_ids=(extra_event.event_id,),
                        source_kind="static_pipeline_order",
                    ),
                )
            commands = [
                replace(
                    item,
                    command_id=f"{stream.stream_id}.changed{index:04d}",
                    order=index,
                )
                for index, item in enumerate(commands)
            ]
            changed_streams.append(replace(stream, commands=tuple(commands)))
        changed_control = replace(
            artifact.static_control,
            streams=tuple(changed_streams),
            events=(*artifact.static_control.events, extra_event),
            attributes={**artifact.static_control.attributes, "extra_static_only": True},
        )
        changed = replace(
            artifact,
            static_control=changed_control,
            attributes={
                **artifact.attributes,
                "static_control_hash": changed_control.control_hash,
            },
        )
        changed_dynamic = schedule_tisa_program(
            changed,
            minimal_machine_config(),
            "dynamic_ready_queue",
            event_backend=CycleEventBackend(),
        )
        self.assertEqual(baseline_dynamic.total_cycles, changed_dynamic.total_cycles)
        self.assertEqual(
            baseline_dynamic.instruction_timings,
            changed_dynamic.instruction_timings,
        )
        self.assertFalse(changed_dynamic.metrics["static_controls_consumed"])
        self.assertEqual(
            baseline_dynamic.metrics["shared_workload_hash"],
            changed_dynamic.metrics["shared_workload_hash"],
        )
        self.assertNotEqual(
            artifact.static_control.control_hash,
            changed_control.control_hash,
        )


if __name__ == "__main__":
    unittest.main()
