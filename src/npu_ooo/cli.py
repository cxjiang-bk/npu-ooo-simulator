from __future__ import annotations

import argparse
import json
from pathlib import Path

from npu_ooo.arch import lpu_like_machine_config, minimal_machine_config, wide_mxu_machine_config
from npu_ooo.benchmarks import build_two_matmul_case, build_two_matmul_model
from npu_ooo.ir import default_two_matmul_schedule
from npu_ooo.lowering import lower_two_matmul
from npu_ooo.scheduler import SchedulerPolicy, schedule_execution_graph
from npu_ooo.trace import (
    write_artifact_json,
    write_csv,
    write_execution_graph_dot,
    write_json,
    write_operator_graph_dot,
    write_operator_graph_svg,
    write_svg,
    write_tile_graph_dot,
)


def _machine(name: str):
    factories = {
        "minimal": minimal_machine_config,
        "wide-mxu": wide_mxu_machine_config,
        "lpu-like": lpu_like_machine_config,
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise ValueError(f"unknown architecture profile '{name}'") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run configurable NPU tile scheduling experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    two_mm = subparsers.add_parser("two-mm", help="compile and schedule the 2mm benchmark")
    two_mm.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    two_mm.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    two_mm.add_argument("--output-dir", type=Path, default=Path("out/two-mm"))
    return parser


def run_two_mm(args: argparse.Namespace) -> int:
    machine = _machine(args.arch)
    model = build_two_matmul_model()
    case = build_two_matmul_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_two_matmul_schedule(instance.graph)
    lowered = lower_two_matmul(instance, machine, schedule)
    result = schedule_execution_graph(lowered.execution_graph, machine, args.policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(machine, args.output_dir / "machine.json")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        {
            "schema_version": 1,
            "benchmark": case.case_id,
            "architecture": args.arch,
            "machine_hash": machine.stable_hash(),
            "schedule": schedule.schedule_id,
            "policy": result.policy,
            "calibration_status": result.metrics["calibration_status"],
            "total_cycles": result.total_cycles,
            "statistics": lowered.statistics,
        },
        args.output_dir / "manifest.json",
    )
    write_operator_graph_dot(instance.graph, args.output_dir / "operator_graph.dot")
    write_operator_graph_svg(instance.graph, args.output_dir / "operator_graph.svg")
    write_tile_graph_dot(lowered.tile_graph, args.output_dir / "tile_graph.dot")
    write_execution_graph_dot(lowered.execution_graph, args.output_dir / "execution_graph.dot")
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    print(
        json.dumps(
            {
                "architecture": args.arch,
                "policy": result.policy,
                "total_cycles": result.total_cycles,
                "statistics": lowered.statistics,
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "two-mm":
        return run_two_mm(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
