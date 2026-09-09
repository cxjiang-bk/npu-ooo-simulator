import unittest
from dataclasses import replace

from npu_ooo.arch import minimal_machine_config
from npu_ooo.execution import (
    AnalyticalExecutionBackend,
    IssueReceipt,
    IssueRequest,
)
from npu_ooo.ir import (
    BoundTISADescriptor,
    CompletionToken,
    LoadedDeviceProgram,
    RuntimeOperandBinding,
    TISAInstruction,
    TISAOperand,
    TileMem,
    UnitMap,
    allocate_memory_plan_bindings,
    create_runtime_submission,
)
from npu_ooo.runtime import load_device_program
from npu_ooo.runtime.loader import _add_runtime_alias_dependencies
from npu_ooo.scheduler.event import schedule_loaded_event_program

from test_target_memory_mapping import _compile


class _RejectingExecutionBackend:
    name = "rejecting"

    def __init__(self):
        self.accepted = []

    def registrations(self):
        return ()

    def can_accept(self, descriptor, cycle):
        return False, "queue_full"

    def issue(self, request):
        ready, reason = self.can_accept(request.descriptor, request.cycle)
        if ready:
            self.accepted.append(request.descriptor.descriptor_id)
        return IssueReceipt(
            accepted=False,
            descriptor_id=request.descriptor.descriptor_id,
            payload_handle=request.descriptor.payload_handle,
            rejection_reason=reason,
        )

    def next_accept_cycle(self, descriptor):
        return None

    def advance(self, cycle):
        return ()

    def next_feedback_cycle(self):
        return None

    def task_timings(self):
        return ()

    def trace_events(self):
        return ()


class _DelayedFeedbackBackend:
    name = "delayed_feedback"

    def __init__(self, delegate, delay):
        self.delegate = delegate
        self.delay = delay

    def registrations(self):
        return self.delegate.registrations()

    def estimate(self, handle):
        return self.delegate.estimate(handle)

    def can_accept(self, descriptor, cycle):
        return self.delegate.can_accept(descriptor, cycle)

    def next_accept_cycle(self, descriptor):
        return self.delegate.next_accept_cycle(descriptor)

    def issue(self, request):
        return self.delegate.issue(request)

    def advance(self, cycle):
        return tuple(
            replace(item, cycle=item.cycle + self.delay)
            for item in self.delegate.advance(cycle - self.delay)
        ) if cycle >= self.delay else ()

    def next_feedback_cycle(self):
        cycle = self.delegate.next_feedback_cycle()
        return cycle + self.delay if cycle is not None else None

    def task_timings(self):
        return self.delegate.task_timings()

    def trace_events(self):
        return self.delegate.trace_events()


class _InitiallyBusyBackend(_DelayedFeedbackBackend):
    def __init__(self, delegate, ready_cycle):
        super().__init__(delegate, 0)
        self.ready_cycle = ready_cycle

    def can_accept(self, descriptor, cycle):
        if cycle < self.ready_cycle:
            return False, "external_backpressure"
        return self.delegate.can_accept(descriptor, cycle)

    def next_accept_cycle(self, descriptor):
        delegate_cycle = self.delegate.next_accept_cycle(descriptor)
        return max(self.ready_cycle, delegate_cycle or 0)


class DeviceContractTest(unittest.TestCase):
    @staticmethod
    def _alias_descriptor(
        tisa_id: str,
        order: int,
        access: str,
        *,
        address: int = 0,
        size_bytes: int = 8,
        identity: str | None = None,
    ) -> BoundTISADescriptor:
        instruction = TISAInstruction(
            tisa_id=tisa_id,
            tile_id=f"tile.{tisa_id}",
            operator_id="alias",
            op_type="elementwise",
            operands=(
                TISAOperand(
                    "buffer",
                    (size_bytes,),
                    TileMem(
                        "buffer",
                        "SRAM",
                        tensor="buffer",
                        offset_bytes=0,
                        size_bytes=size_bytes,
                    ),
                    access,
                ),
            ),
            unit_map=UnitMap("DMA"),
            payload_ref=f"payload:{tisa_id}",
        )
        attributes = (
            {"allocation_identity": identity} if identity is not None else {}
        )
        operand = RuntimeOperandBinding(
            tisa_id=tisa_id,
            operand_name="buffer",
            tensor="buffer",
            logical_scope="SRAM",
            physical_scope="SRAM",
            address=address,
            size_bytes=size_bytes,
            access_type=access,
            offset_bytes=0,
            attributes=attributes,
        )
        return BoundTISADescriptor(
            descriptor_id=f"inv::{tisa_id}",
            invocation_id="inv",
            program_id="alias.program",
            artifact_id="alias.artifact",
            instruction=instruction,
            operands=(operand,),
            dependencies=(),
            completion_token=CompletionToken("inv", tisa_id),
            payload_handle=f"alias.artifact::{tisa_id}",
            program_order=order,
            submission_order=order,
        )

    @staticmethod
    def _alias_sources(descriptor: BoundTISADescriptor) -> set[tuple[str, str]]:
        return {
            (dependency.source.tisa_id, dependency.kind)
            for dependency in descriptor.dependencies
            if dependency.provenance.get("source") == "runtime_binding_alias"
        }

    def test_alias_writes_depend_only_on_last_writer_frontier(self):
        descriptors = tuple(
            self._alias_descriptor(f"w{index}", index, "write")
            for index in range(3)
        )
        loaded = _add_runtime_alias_dependencies(descriptors)
        self.assertEqual(self._alias_sources(loaded[0]), set())
        self.assertEqual(self._alias_sources(loaded[1]), {("w0", "WAW")})
        self.assertEqual(self._alias_sources(loaded[2]), {("w1", "WAW")})

    def test_alias_write_waits_for_reader_frontier_not_full_history(self):
        descriptors = (
            self._alias_descriptor("writer", 0, "write"),
            self._alias_descriptor("reader0", 1, "read"),
            self._alias_descriptor("reader1", 2, "read"),
            self._alias_descriptor("next_writer", 3, "write"),
        )
        loaded = _add_runtime_alias_dependencies(descriptors)
        self.assertEqual(self._alias_sources(loaded[1]), {("writer", "RAW")})
        self.assertEqual(self._alias_sources(loaded[2]), {("writer", "RAW")})
        self.assertEqual(
            self._alias_sources(loaded[3]),
            {("reader0", "WAR"), ("reader1", "WAR")},
        )

    def test_alias_frontier_tracks_partial_ranges_and_allocation_identity(self):
        partial = _add_runtime_alias_dependencies(
            (
                self._alias_descriptor("left", 0, "write", address=0, size_bytes=4),
                self._alias_descriptor("right", 1, "write", address=4, size_bytes=4),
                self._alias_descriptor("read_all", 2, "read", address=0, size_bytes=8),
            )
        )
        self.assertEqual(
            self._alias_sources(partial[2]),
            {("left", "RAW"), ("right", "RAW")},
        )
        identified = _add_runtime_alias_dependencies(
            (
                self._alias_descriptor("a", 0, "write", identity="alloc.a"),
                self._alias_descriptor("b", 1, "read", identity="alloc.b"),
            )
        )
        self.assertEqual(self._alias_sources(identified[1]), set())

    def _loaded(self, submission_id="invocation.0", launch_latency=2):
        machine = minimal_machine_config()
        compiled = _compile(machine)
        buffers = allocate_memory_plan_bindings(
            compiled.backend_artifact.memory_plan, machine
        )
        submission = create_runtime_submission(
            compiled.backend_artifact,
            buffers,
            submission_id=submission_id,
            chunk_size=1,
            launch_latency_cycles=launch_latency,
        )
        return compiled, load_device_program(compiled.backend_artifact, submission)

    def test_loaded_descriptors_bind_semantics_addresses_arrival_and_payload(self):
        compiled, loaded = self._loaded()
        self.assertEqual(loaded.validate(), ())
        self.assertEqual(len(loaded.descriptors), len(compiled.tisa_program.instructions))
        self.assertEqual(
            [item.arrival_cycle for item in loaded.envelopes],
            [2, 4, 6],
        )
        for descriptor in loaded.descriptors:
            self.assertEqual(
                descriptor.payload_handle,
                f"{compiled.backend_artifact.artifact_id}::{descriptor.instruction.tisa_id}",
            )
            self.assertEqual(
                {item.operand_name for item in descriptor.operands},
                {item.name for item in descriptor.instruction.operands},
            )
            self.assertTrue(all(item.physical_scope for item in descriptor.operands))

    def test_loaded_device_program_roundtrip_and_invocation_scoped_tokens(self):
        _compiled, first = self._loaded("decode.0", launch_latency=0)
        _compiled, second = self._loaded("decode.1", launch_latency=0)
        restored = LoadedDeviceProgram.from_dict(first.to_dict())
        self.assertEqual(restored.to_dict(), first.to_dict())
        self.assertEqual(
            [item.tisa_id for item in first.static_schedule.entries],
            [
                item.instruction.tisa_id
                for item in sorted(first.descriptors, key=lambda row: row.submission_order)
            ],
        )
        self.assertEqual(
            first.static_schedule.policy, "runtime_fixed_submission_overlap"
        )
        self.assertTrue(
            set(item.completion_token.token_id for item in first.descriptors).isdisjoint(
                item.completion_token.token_id for item in second.descriptors
            )
        )

    def test_rejected_execution_issue_does_not_create_false_acceptance(self):
        _compiled, loaded = self._loaded(launch_latency=0)
        backend = _RejectingExecutionBackend()
        descriptor = loaded.descriptors[0]
        ready, reason = backend.can_accept(descriptor, 0)
        self.assertFalse(ready)
        self.assertEqual(reason, "queue_full")
        receipt = backend.issue(IssueRequest(descriptor, 0))
        self.assertFalse(receipt.accepted)
        self.assertEqual(receipt.rejection_reason, "queue_full")
        self.assertEqual(backend.accepted, [])
        self.assertEqual(receipt.validate(), ())

    def test_execution_backend_owns_busy_state_and_emits_physical_done(self):
        compiled, loaded = self._loaded(launch_latency=0)
        backend = AnalyticalExecutionBackend(
            compiled.backend_artifact, minimal_machine_config()
        )
        descriptor = loaded.descriptors[0]
        receipt = backend.issue(IssueRequest(descriptor, 0))
        self.assertTrue(receipt.accepted)
        self.assertEqual(backend.advance(receipt.expected_done_cycle - 1), ())
        feedback = backend.advance(receipt.expected_done_cycle)
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0].kind, "execution_done")
        self.assertEqual(feedback[0].token, descriptor.completion_token)
        self.assertTrue(backend.task_timings())
        self.assertTrue(backend.trace_events())

    def test_execution_backpressure_prevents_false_scheduler_issue(self):
        compiled, loaded = self._loaded(launch_latency=0)
        execution = _InitiallyBusyBackend(
            AnalyticalExecutionBackend(
                compiled.backend_artifact, minimal_machine_config()
            ),
            ready_cycle=3,
        )
        result = schedule_loaded_event_program(
            loaded,
            minimal_machine_config(),
            "static_pipeline",
            execution,
        )
        self.assertEqual(result.instruction_timings[0].issue, 3)
        self.assertFalse(
            any(
                event.event == "TISA_ISSUE" and event.timestamp < 3
                for event in result.events
            )
        )

    def test_delayed_feedback_delays_dependency_wakeup(self):
        compiled, loaded = self._loaded(launch_latency=0)
        baseline = schedule_loaded_event_program(
            loaded,
            minimal_machine_config(),
            "static_pipeline",
            AnalyticalExecutionBackend(
                compiled.backend_artifact, minimal_machine_config()
            ),
        )
        delayed = schedule_loaded_event_program(
            loaded,
            minimal_machine_config(),
            "static_pipeline",
            _DelayedFeedbackBackend(
                AnalyticalExecutionBackend(
                    compiled.backend_artifact, minimal_machine_config()
                ),
                delay=5,
            ),
        )
        consumer = loaded.descriptors[1].instruction.tisa_id
        self.assertEqual(
            delayed.instruction_timing(consumer).issue,
            baseline.instruction_timing(consumer).issue + 5,
        )


if __name__ == "__main__":
    unittest.main()
