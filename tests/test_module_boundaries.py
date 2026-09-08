import inspect
import unittest

from npu_ooo.backend.analytical import AnalyticalEventBackend
from npu_ooo.backend.cycle import CycleEventBackend
from npu_ooo.scheduler.event import schedule_loaded_event_program
from npu_ooo.simulator.cycle import schedule_loaded_cycle_program


class ModuleBoundaryTest(unittest.TestCase):
    def test_scheduler_kernels_do_not_accept_or_read_backend_artifacts(self):
        event_source = inspect.getsource(schedule_loaded_event_program)
        cycle_source = inspect.getsource(schedule_loaded_cycle_program)
        for source in (event_source, cycle_source):
            self.assertNotIn("BackendArtifact", source)
            self.assertNotIn("execution_graph", source)
            self.assertNotIn("artifact.payloads", source)

    def test_cycle_scheduler_does_not_import_event_scheduler_privates(self):
        import npu_ooo.simulator.cycle as cycle_module

        source = inspect.getsource(cycle_module)
        self.assertNotIn("from .tisa import", source)
        self.assertIn("from npu_ooo.scheduler.semantics import", source)

    def test_production_backends_enter_through_device_simulator(self):
        for backend in (AnalyticalEventBackend, CycleEventBackend):
            source = inspect.getsource(backend.simulate)
            self.assertIn("DeviceSimulator", source)
            self.assertNotIn("simulate_tisa_artifact", source)


if __name__ == "__main__":
    unittest.main()
