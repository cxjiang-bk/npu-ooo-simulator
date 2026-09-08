import unittest
from dataclasses import replace

from npu_ooo.arch import minimal_machine_config
from npu_ooo.execution import (
    AnalyticalExecutionBackend,
    IssueReceipt,
    IssueRequest,
)
from npu_ooo.ir import (
    LoadedDeviceProgram,
    allocate_memory_plan_bindings,
    create_runtime_submission,
)
from npu_ooo.runtime import load_device_program
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
