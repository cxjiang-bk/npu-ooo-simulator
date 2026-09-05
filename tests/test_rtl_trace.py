import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from npu_ooo.backend import (
    import_rtl_completion_trace,
    load_rtl_completion_trace,
    parse_mxu_vcs_log,
)
from npu_ooo.cli import main


def _trace_payload() -> dict:
    return {
        "format": "npu_ooo.rtl_completion_trace.v1",
        "metadata": {"source": "unit-test", "calibration_status": "rtl-observed"},
        "records": [
            {
                "instruction_id": "a",
                "batch": 1,
                "m": 4,
                "n": 12,
                "k": 8,
                "descriptor_issue_cycle": 10,
                "compute_start_cycle": 14,
                "compute_done_cycle": 37,
                "psb_write_done_cycle": 41,
            },
            {
                "instruction_id": "b",
                "batch": 1,
                "m": 4,
                "n": 12,
                "k": 8,
                "descriptor_issue_cycle": 20,
                "compute_start_cycle": 24,
                "compute_done_cycle": 48,
                "psb_write_done_cycle": 52,
            },
        ],
    }


class RTLTraceImporterTest(unittest.TestCase):
    def test_mxu_vcs_console_log_preserves_descriptor_boundary(self) -> None:
        trace = parse_mxu_vcs_log(
            """
[100] Prepared instruction: M=16, N=8, K1=2, psum_en=0, bias_en=0
[110] Test 1 instruction accepted
[200] Prepared instruction: M=32, N=16, K1=1, psum_en=1, bias_en=0
[210] Test 2 instruction accepted
[220] Test 6 END instruction accepted
[310] Done Signal #1: instr_idx=0, task_done=0
[410] Done Signal #2: instr_idx=0, task_done=0
[510] Done Signal #3: instr_idx=999, task_done=1
"""
        )
        self.assertEqual(trace["metadata"]["compute_start_available"], False)
        self.assertEqual(trace["records"][0]["k"], 16)
        self.assertEqual(trace["records"][0]["descriptor_issue_cycle"], 110.0)
        self.assertEqual(trace["records"][0]["psb_write_done_cycle"], 310.0)
        self.assertEqual(trace["records"][1]["k"], 8)
        self.assertEqual(trace["metadata"]["ignored_end_acceptance_count"], 1)
        self.assertEqual(trace["metadata"]["ignored_end_done_count"], 1)

    def test_mxu_vcs_console_log_rejects_unmatched_acceptance(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 0 acceptance events"):
            parse_mxu_vcs_log(
                "[1] Prepared instruction: M=16, N=8, K1=1, psum_en=0, bias_en=0\n"
            )

    def test_json_import_aggregates_compute_interval_and_derives_ii(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(_trace_payload()), encoding="utf-8")
            profile = import_rtl_completion_trace(path, aggregation="median")
        self.assertEqual(profile["format"], "npu_ooo.systolic_mxu_profile.v1")
        self.assertEqual(profile["matmul_profiles"][0]["duration_cycles"], 23.5)
        self.assertEqual(profile["matmul_profiles"][0]["initiation_interval_cycles"], 10.0)
        self.assertEqual(profile["metadata"]["interval"], "compute_start_to_compute_done")
        self.assertEqual(profile["metadata"]["calibration_status"], "rtl-observed")

    def test_descriptor_interval_uses_psb_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(_trace_payload()), encoding="utf-8")
            profile = import_rtl_completion_trace(
                path,
                interval="descriptor_issue_to_done",
                aggregation="max",
            )
        self.assertEqual(profile["matmul_profiles"][0]["duration_cycles"], 32.0)
        self.assertEqual(profile["metadata"]["interval_end_fields"], ["psb_write_done_cycle"])

    def test_csv_input_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "instruction_id",
                        "batch",
                        "m",
                        "n",
                        "k",
                        "compute_start_cycle",
                        "compute_done_cycle",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "instruction_id": "csv-0",
                        "batch": 1,
                        "m": 4,
                        "n": 12,
                        "k": 8,
                        "compute_start_cycle": 2,
                        "compute_done_cycle": 25,
                    }
                )
            records, _metadata = load_rtl_completion_trace(path)
            self.assertEqual(records[0].shape_key, (1, 4, 12, 8))
            self.assertEqual(records[0].compute_done_cycle, 25.0)

    def test_missing_interval_marker_is_rejected(self) -> None:
        payload = _trace_payload()
        del payload["records"][0]["compute_done_cycle"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lacks compute_start_cycle or compute_done_cycle"):
                import_rtl_completion_trace(path)

    def test_cli_writes_provider_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            input_path = root / "trace.json"
            output_path = root / "profile.json"
            input_path.write_text(json.dumps(_trace_payload()), encoding="utf-8")
            exit_code = main(
                [
                    "import-rtl-trace",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--aggregation",
                    "p95",
                ]
            )
            self.assertEqual(exit_code, 0)
            profile = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(profile["metadata"]["aggregation"], "p95")
            self.assertEqual(profile["metadata"]["record_count"], 2)

if __name__ == "__main__":
    unittest.main()
