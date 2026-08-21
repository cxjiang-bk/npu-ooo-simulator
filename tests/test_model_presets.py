import unittest

from npu_ooo.benchmarks import available_model_presets, build_model_preset


class ModelPresetTest(unittest.TestCase):
    def test_presets_validate_and_keep_native_metadata(self) -> None:
        self.assertEqual(
            set(available_model_presets()),
            {"bert-base", "gpt-j", "llama2-7b", "deepseek-r1-16b"},
        )
        for name in available_model_presets():
            model, case = build_model_preset(name)
            self.assertEqual(model.validate(), ())
            self.assertEqual(case.validate(), ())
            self.assertEqual(model.attributes["benchmark_status"], "proxy")
            self.assertIn("native_config", model.attributes)
            self.assertEqual(case.model_id, model.model_id)
            self.assertEqual(case.attributes["preset"], name)

    def test_preset_shape_and_phase_overrides_are_serialized(self) -> None:
        model, case = build_model_preset(
            "llama2-7b",
            tokens=16,
            sequence=32,
            head_dim=16,
            intermediate=32,
            phase="decode",
        )
        self.assertEqual(model.shape_environment, {"M": 16, "S": 32, "D": 16, "H": 32})
        self.assertEqual(case.normalized_phase, "decode")
        self.assertEqual(case.sequence_length, 32)
        self.assertEqual(case.case_id, "llama2_7b_decode_one_block")
        self.assertEqual(model.attributes["proxy_shape"]["head_dim"], 16)


if __name__ == "__main__":
    unittest.main()
