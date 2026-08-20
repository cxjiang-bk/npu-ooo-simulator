import unittest

from npu_ooo.arch import (
    MachineConfig,
    MemoryLevelConfig,
    lpu_like_machine_config,
    minimal_machine_config,
    wide_mxu_machine_config,
)


class MachineConfigTest(unittest.TestCase):
    def test_profiles_validate_and_have_stable_hashes(self) -> None:
        for factory in (minimal_machine_config, wide_mxu_machine_config, lpu_like_machine_config):
            config = factory()
            self.assertEqual(config.validate(), ())
            self.assertEqual(config.stable_hash(), factory().stable_hash())
            self.assertTrue(config.stable_hash())

    def test_wide_mxu_changes_architecture_not_schema(self) -> None:
        base = minimal_machine_config()
        wide = wide_mxu_machine_config()
        self.assertEqual(base.memory_levels, wide.memory_levels)
        self.assertEqual(base.unit("MXU").count, 1)
        self.assertEqual(wide.unit("MXU").count, 2)
        self.assertNotEqual(base.stable_hash(), wide.stable_hash())

    def test_unknown_path_references_are_reported(self) -> None:
        base = minimal_machine_config()
        broken = MachineConfig(
            config_id="broken",
            memory_levels=base.memory_levels,
            execution_units=base.execution_units,
            transfer_paths=(
                base.transfer_paths[0].__class__(
                    "UNKNOWN", "SRAM", "DMA", bandwidth_bytes_per_cycle=1
                ),
            ),
            scheduler=base.scheduler,
        )
        issues = broken.validate()
        self.assertTrue(any("unknown source memory" in issue for issue in issues))

    def test_memory_cycle_is_reported(self) -> None:
        broken = MachineConfig(
            config_id="cycle",
            memory_levels=(
                MemoryLevelConfig("A", "B", 1024, 1, 1),
                MemoryLevelConfig("B", "A", 1024, 1, 1),
            ),
            execution_units=(),
            transfer_paths=(),
        )
        self.assertTrue(any("cycle" in issue for issue in broken.validate()))


if __name__ == "__main__":
    unittest.main()
