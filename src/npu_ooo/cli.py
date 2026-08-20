from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

from npu_ooo.arch import load_machine_config, lpu_like_machine_config, minimal_machine_config, wide_mxu_machine_config
from npu_ooo.benchmarks import (
    build_decoder_block_case,
    build_decoder_block_model,
    build_attention_case,
    build_attention_model,
    build_transformer_block_case,
    build_transformer_block_model,
    build_elementwise_case,
    build_elementwise_model,
    build_layernorm_case,
    build_layernorm_model,
    build_reduce_case,
    build_reduce_model,
    build_softmax_case,
    build_softmax_model,
    build_rmsnorm_case,
    build_rmsnorm_model,
    build_two_matmul_case,
    build_two_matmul_model,
)
from npu_ooo.ir import (
    default_elementwise_schedule,
    default_layernorm_schedule,
    default_mixed_schedule,
    default_reduce_schedule,
    default_softmax_schedule,
    default_rmsnorm_schedule,
    default_two_matmul_schedule,
)
from npu_ooo.lowering import (
    lower_elementwise,
    lower_layernorm,
    lower_mixed_model,
    lower_reduce,
    lower_rmsnorm,
    lower_softmax,
    lower_two_matmul,
)
from npu_ooo.scheduler import (
    SchedulerPolicy,
    SimulatorConfig,
    StaticPipelineConfig,
    schedule_execution_graph,
)
from npu_ooo.trace import (
    write_artifact_json,
    write_csv,
    write_execution_graph_dot,
    write_json,
    write_png,
    write_operator_graph_dot,
    write_operator_graph_svg,
    write_svg,
    write_tile_graph_dot,
)


def _machine(name: str, config_path: Path | None = None):
    if config_path is not None:
        return load_machine_config(config_path)
    factories = {
        "minimal": minimal_machine_config,
        "wide-mxu": wide_mxu_machine_config,
        "lpu-like": lpu_like_machine_config,
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise ValueError(f"unknown architecture profile '{name}'") from exc


def _parse_offsets(value: str | None) -> tuple[float, ...]:
    if value is None:
        return ()
    try:
        offsets = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--static-stage-offsets must be a comma-separated list of numbers") from exc
    if not offsets:
        raise ValueError("--static-stage-offsets must contain at least one offset")
    if any(offset < 0 for offset in offsets):
        raise ValueError("--static-stage-offsets values must be non-negative")
    return offsets


def _parse_list(value: str, *, name: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError(f"{name} must contain at least one value")
    return items


def _parse_positive_int_list(value: str, *, name: str) -> tuple[int, ...]:
    items = _parse_list(value, name=name)
    try:
        numbers = tuple(int(item) for item in items)
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of positive integers") from exc
    if any(number <= 0 for number in numbers):
        raise ValueError(f"{name} must contain only positive integers")
    return numbers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run configurable NPU tile scheduling experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    two_mm = subparsers.add_parser("two-mm", help="compile and schedule the 2mm benchmark")
    two_mm.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    two_mm.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    two_mm.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    two_mm.add_argument("--output-dir", type=Path, default=Path("out/two-mm"))
    two_mm.add_argument("--instruction-queue-depth", type=int)
    two_mm.add_argument("--rob-entries", type=int)
    two_mm.add_argument("--max-inflight-tiles", type=int)
    two_mm.add_argument("--dependency-window", type=int)
    two_mm.add_argument("--ready-queue-depth", type=int)
    two_mm.add_argument("--address-scoreboard", action="store_true")
    two_mm.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    two_mm.add_argument(
        "--static-stage-offsets",
        help="comma-separated static stage reservation offsets; enables explicit static pipeline reservations",
    )
    two_mm.add_argument("--static-stage-ii", type=float, default=1.0)
    elementwise = subparsers.add_parser("elementwise", help="compile and schedule a residual-add benchmark")
    elementwise.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    elementwise.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    elementwise.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    elementwise.add_argument("--output-dir", type=Path, default=Path("out/elementwise"))
    elementwise.add_argument("--instruction-queue-depth", type=int)
    elementwise.add_argument("--rob-entries", type=int)
    elementwise.add_argument("--max-inflight-tiles", type=int)
    elementwise.add_argument("--dependency-window", type=int)
    elementwise.add_argument("--ready-queue-depth", type=int)
    elementwise.add_argument("--address-scoreboard", action="store_true")
    elementwise.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    elementwise.add_argument("--static-stage-offsets")
    elementwise.add_argument("--static-stage-ii", type=float, default=1.0)
    reduce = subparsers.add_parser("reduce", help="compile and schedule a row-reduction benchmark")
    reduce.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    reduce.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    reduce.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    reduce.add_argument("--output-dir", type=Path, default=Path("out/reduce"))
    reduce.add_argument("--instruction-queue-depth", type=int)
    reduce.add_argument("--rob-entries", type=int)
    reduce.add_argument("--max-inflight-tiles", type=int)
    reduce.add_argument("--dependency-window", type=int)
    reduce.add_argument("--ready-queue-depth", type=int)
    reduce.add_argument("--address-scoreboard", action="store_true")
    reduce.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    reduce.add_argument("--static-stage-offsets")
    reduce.add_argument("--static-stage-ii", type=float, default=1.0)
    softmax = subparsers.add_parser("softmax", help="compile and schedule a row-softmax benchmark")
    softmax.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    softmax.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    softmax.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    softmax.add_argument("--output-dir", type=Path, default=Path("out/softmax"))
    softmax.add_argument("--instruction-queue-depth", type=int)
    softmax.add_argument("--rob-entries", type=int)
    softmax.add_argument("--max-inflight-tiles", type=int)
    softmax.add_argument("--dependency-window", type=int)
    softmax.add_argument("--ready-queue-depth", type=int)
    softmax.add_argument("--address-scoreboard", action="store_true")
    softmax.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    softmax.add_argument("--static-stage-offsets")
    softmax.add_argument("--static-stage-ii", type=float, default=1.0)
    rmsnorm = subparsers.add_parser("rmsnorm", help="compile and schedule an RMSNorm benchmark")
    rmsnorm.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    rmsnorm.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    rmsnorm.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    rmsnorm.add_argument("--output-dir", type=Path, default=Path("out/rmsnorm"))
    rmsnorm.add_argument("--instruction-queue-depth", type=int)
    rmsnorm.add_argument("--rob-entries", type=int)
    rmsnorm.add_argument("--max-inflight-tiles", type=int)
    rmsnorm.add_argument("--dependency-window", type=int)
    rmsnorm.add_argument("--ready-queue-depth", type=int)
    rmsnorm.add_argument("--address-scoreboard", action="store_true")
    rmsnorm.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    rmsnorm.add_argument("--static-stage-offsets")
    rmsnorm.add_argument("--static-stage-ii", type=float, default=1.0)
    decoder_block = subparsers.add_parser(
        "decoder-block",
        help="compile and schedule an RMSNorm -> Matmul -> ResidualAdd fragment",
    )
    decoder_block.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    decoder_block.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    decoder_block.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    decoder_block.add_argument("--output-dir", type=Path, default=Path("out/decoder-block"))
    decoder_block.add_argument("--instruction-queue-depth", type=int)
    decoder_block.add_argument("--rob-entries", type=int)
    decoder_block.add_argument("--max-inflight-tiles", type=int)
    decoder_block.add_argument("--dependency-window", type=int)
    decoder_block.add_argument("--ready-queue-depth", type=int)
    decoder_block.add_argument("--address-scoreboard", action="store_true")
    decoder_block.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    decoder_block.add_argument("--static-stage-offsets")
    decoder_block.add_argument("--static-stage-ii", type=float, default=1.0)
    attention = subparsers.add_parser(
        "attention",
        help="compile and schedule a single-head QK-softmax-PV attention fragment",
    )
    attention.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    attention.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    attention.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    attention.add_argument("--output-dir", type=Path, default=Path("out/attention"))
    attention.add_argument("--instruction-queue-depth", type=int)
    attention.add_argument("--rob-entries", type=int)
    attention.add_argument("--max-inflight-tiles", type=int)
    attention.add_argument("--dependency-window", type=int)
    attention.add_argument("--ready-queue-depth", type=int)
    attention.add_argument("--address-scoreboard", action="store_true")
    attention.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    attention.add_argument("--static-stage-offsets")
    attention.add_argument("--static-stage-ii", type=float, default=1.0)
    transformer_block = subparsers.add_parser(
        "transformer-block",
        help="compile and schedule a LayerNorm + attention + MLP decoder block skeleton",
    )
    transformer_block.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    transformer_block.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    transformer_block.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    transformer_block.add_argument("--output-dir", type=Path, default=Path("out/transformer-block"))
    transformer_block.add_argument("--instruction-queue-depth", type=int)
    transformer_block.add_argument("--rob-entries", type=int)
    transformer_block.add_argument("--max-inflight-tiles", type=int)
    transformer_block.add_argument("--dependency-window", type=int)
    transformer_block.add_argument("--ready-queue-depth", type=int)
    transformer_block.add_argument("--address-scoreboard", action="store_true")
    transformer_block.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    transformer_block.add_argument("--static-stage-offsets")
    transformer_block.add_argument("--static-stage-ii", type=float, default=1.0)
    layernorm = subparsers.add_parser("layernorm", help="compile and schedule a row-LayerNorm benchmark")
    layernorm.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    layernorm.add_argument("--machine-config", type=Path, help="load a canonical MachineConfig JSON")
    layernorm.add_argument("--policy", choices=tuple(policy.value for policy in SchedulerPolicy), default="static_pipeline")
    layernorm.add_argument("--output-dir", type=Path, default=Path("out/layernorm"))
    layernorm.add_argument("--instruction-queue-depth", type=int)
    layernorm.add_argument("--rob-entries", type=int)
    layernorm.add_argument("--max-inflight-tiles", type=int)
    layernorm.add_argument("--dependency-window", type=int)
    layernorm.add_argument("--ready-queue-depth", type=int)
    layernorm.add_argument("--address-scoreboard", action="store_true")
    layernorm.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    layernorm.add_argument("--static-stage-offsets")
    layernorm.add_argument("--static-stage-ii", type=float, default=1.0)
    sweep = subparsers.add_parser("sweep-two-mm", help="sweep 2mm architecture and scheduler parameters")
    sweep.add_argument("--architectures", default="minimal,wide-mxu")
    sweep.add_argument("--machine-config", type=Path, help="use one canonical MachineConfig JSON for every case")
    sweep.add_argument(
        "--policies",
        default=",".join(policy.value for policy in SchedulerPolicy),
    )
    sweep.add_argument("--windows", default="4,8")
    sweep.add_argument("--robs", default="4,8")
    sweep.add_argument("--tile-sizes", default="32")
    sweep.add_argument("--output-dir", type=Path, default=Path("out/sweep-two-mm"))
    sweep.add_argument("--address-scoreboard", action="store_true")
    sweep.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    sweep.add_argument("--static-stage-offsets")
    sweep.add_argument("--static-stage-ii", type=float, default=1.0)
    workload_sweep = subparsers.add_parser(
        "sweep-workloads",
        help="sweep multiple workloads, architectures and scheduler capacities",
    )
    workload_sweep.add_argument(
        "--workloads",
        default="two-mm,elementwise,reduce,softmax,rmsnorm,layernorm,decoder-block,attention,transformer-block",
    )
    workload_sweep.add_argument("--architectures", default="minimal,wide-mxu")
    workload_sweep.add_argument("--machine-config", type=Path, help="use one canonical MachineConfig JSON for every case")
    workload_sweep.add_argument(
        "--policies",
        default=",".join(policy.value for policy in SchedulerPolicy),
    )
    workload_sweep.add_argument("--windows", default="4,8")
    workload_sweep.add_argument("--robs", default="4,8")
    workload_sweep.add_argument("--tile-sizes", default="32")
    workload_sweep.add_argument("--output-dir", type=Path, default=Path("out/sweep-workloads"))
    workload_sweep.add_argument("--address-scoreboard", action="store_true")
    workload_sweep.add_argument("--dynamic-priority", choices=("critical_path", "oldest_first"), default="critical_path")
    workload_sweep.add_argument("--dynamic-priorities", default="critical_path,oldest_first")
    workload_sweep.add_argument("--static-stage-offsets")
    workload_sweep.add_argument("--static-stage-ii", type=float, default=1.0)
    return parser


def _simulator_config(
    *,
    dependency_window: int | None,
    rob_entries: int | None,
    address_scoreboard: bool,
    static_stage_offsets: tuple[float, ...] = (),
    static_stage_ii: float = 1.0,
    dynamic_priority: str = "critical_path",
) -> SimulatorConfig:
    return SimulatorConfig(
        dependency_window=dependency_window,
        rob_entries=rob_entries,
        address_scoreboard=address_scoreboard,
        dynamic_priority=dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(static_stage_offsets),
                stage_offsets=static_stage_offsets,
                initiation_interval_cycles=static_stage_ii,
            )
            if static_stage_offsets
            else None
        ),
    )


def run_two_mm(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_two_matmul_model()
    case = build_two_matmul_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_two_matmul_schedule(instance.graph)
    lowered = lower_two_matmul(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    effective_execution_graph = lowered.execution_graph
    result = schedule_execution_graph(
        effective_execution_graph,
        machine,
        args.policy,
        simulator_config=simulator_config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(effective_execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(
        result.metrics.get("address_hazards", []),
        args.output_dir / "address_dependencies.json",
    )
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
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
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
    write_png(result, args.output_dir / "swimlane.png")
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


def run_elementwise(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_elementwise_model()
    case = build_elementwise_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_elementwise_schedule(instance.graph)
    lowered = lower_elementwise(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(
        lowered.execution_graph,
        machine,
        args.policy,
        simulator_config=simulator_config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
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
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
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
    write_png(result, args.output_dir / "swimlane.png")
    print(json.dumps({"architecture": args.arch, "policy": result.policy, "total_cycles": result.total_cycles, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def run_reduce(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_reduce_model()
    case = build_reduce_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_reduce_schedule(instance.graph)
    lowered = lower_reduce(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(lowered.execution_graph, machine, args.policy, simulator_config=simulator_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
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
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
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
    write_png(result, args.output_dir / "swimlane.png")
    print(json.dumps({"architecture": args.arch, "policy": result.policy, "total_cycles": result.total_cycles, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def run_softmax(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_softmax_model()
    case = build_softmax_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_softmax_schedule(instance.graph)
    lowered = lower_softmax(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(lowered.execution_graph, machine, args.policy, simulator_config=simulator_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
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
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
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
    write_png(result, args.output_dir / "swimlane.png")
    print(json.dumps({"architecture": args.arch, "policy": result.policy, "total_cycles": result.total_cycles, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def run_rmsnorm(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_rmsnorm_model()
    case = build_rmsnorm_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_rmsnorm_schedule(instance.graph)
    lowered = lower_rmsnorm(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(lowered.execution_graph, machine, args.policy, simulator_config=simulator_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
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
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
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
    write_png(result, args.output_dir / "swimlane.png")
    print(json.dumps({"architecture": args.arch, "policy": result.policy, "total_cycles": result.total_cycles, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def run_decoder_block(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_decoder_block_model()
    case = build_decoder_block_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_mixed_schedule(instance.graph)
    lowered = lower_mixed_model(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(
        lowered.execution_graph,
        machine,
        args.policy,
        simulator_config=simulator_config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
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
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
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
    write_png(result, args.output_dir / "swimlane.png")
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


def run_attention(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_attention_model()
    case = build_attention_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_mixed_schedule(instance.graph)
    lowered = lower_mixed_model(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(lowered.execution_graph, machine, args.policy, simulator_config=simulator_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
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
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
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
    write_png(result, args.output_dir / "swimlane.png")
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


def run_transformer_block(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_transformer_block_model()
    case = build_transformer_block_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_mixed_schedule(instance.graph)
    lowered = lower_mixed_model(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(lowered.execution_graph, machine, args.policy, simulator_config=simulator_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
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
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
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
    write_png(result, args.output_dir / "swimlane.png")
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


def run_layernorm(args: argparse.Namespace) -> int:
    machine = _machine(args.arch, args.machine_config)
    model = build_layernorm_model()
    case = build_layernorm_case(
        architecture_profile=args.arch,
        scheduler_profile=args.policy,
    )
    instance = model.instantiate(case)
    schedule = default_layernorm_schedule(instance.graph)
    lowered = lower_layernorm(instance, machine, schedule)
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    simulator_config = SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        dynamic_priority=args.dynamic_priority,
        static_pipeline=(
            StaticPipelineConfig(
                stage_count=len(stage_offsets),
                stage_offsets=stage_offsets,
                initiation_interval_cycles=args.static_stage_ii,
            )
            if stage_offsets
            else None
        ),
    )
    result = schedule_execution_graph(lowered.execution_graph, machine, args.policy, simulator_config=simulator_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_json(model, args.output_dir / "model_spec.json")
    write_artifact_json(case, args.output_dir / "benchmark_case.json")
    write_artifact_json(instance, args.output_dir / "model_instance.json")
    write_artifact_json(instance.graph, args.output_dir / "operator_graph.json")
    write_artifact_json(schedule, args.output_dir / "schedule.json")
    write_artifact_json(lowered.tile_graph, args.output_dir / "tile_graph.json")
    write_artifact_json(lowered.execution_graph, args.output_dir / "execution_graph.json")
    write_artifact_json(result.metrics.get("address_hazards", []), args.output_dir / "address_dependencies.json")
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
            "backend": result.backend,
            "calibration_status": result.metrics["calibration_status"],
            "simulator_config": result.metrics["simulator_config"],
            "address_dependency_count": result.metrics.get("address_dependency_count", 0),
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
    write_png(result, args.output_dir / "swimlane.png")
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


def run_sweep_two_mm(args: argparse.Namespace) -> int:
    architectures = _parse_list(args.architectures, name="--architectures")
    policies = _parse_list(args.policies, name="--policies")
    supported_architectures = {"minimal", "wide-mxu", "lpu-like"}
    unknown_architectures = sorted(set(architectures) - supported_architectures)
    if unknown_architectures and args.machine_config is None:
        raise ValueError(f"unknown architecture profile(s): {', '.join(unknown_architectures)}")
    supported_policies = {policy.value for policy in SchedulerPolicy}
    unknown_policies = sorted(set(policies) - supported_policies)
    if unknown_policies:
        raise ValueError(f"unsupported scheduler policy(s): {', '.join(unknown_policies)}")
    windows = _parse_positive_int_list(args.windows, name="--windows")
    robs = _parse_positive_int_list(args.robs, name="--robs")
    tile_sizes = _parse_positive_int_list(args.tile_sizes, name="--tile-sizes")
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_two_matmul_model()
    records: list[dict[str, object]] = []
    results: dict[tuple[str, str, int, int, int], object] = {}
    lowered_by_arch: dict[tuple[str, int], tuple[object, object, object]] = {}
    for architecture, policy, dependency_window, rob_entries, tile_size in product(architectures, policies, windows, robs, tile_sizes):
        machine = _machine(architecture, args.machine_config)
        lowering_key = (architecture, tile_size)
        if lowering_key not in lowered_by_arch:
            case = build_two_matmul_case(
                architecture_profile=architecture,
                scheduler_profile=SchedulerPolicy.STATIC_PIPELINE.value,
            )
            instance = model.instantiate(case)
            schedule = default_two_matmul_schedule(instance.graph, tile_size=tile_size)
            lowered_by_arch[lowering_key] = (instance, schedule, lower_two_matmul(instance, machine, schedule))
        instance, schedule, lowered = lowered_by_arch[lowering_key]
        case = build_two_matmul_case(
            architecture_profile=architecture,
            scheduler_profile=policy,
        )
        static_config = stage_offsets if policy == SchedulerPolicy.STATIC_PIPELINE.value else ()
        simulator_config = _simulator_config(
            dependency_window=dependency_window,
            rob_entries=rob_entries,
            address_scoreboard=args.address_scoreboard,
            static_stage_offsets=static_config,
            static_stage_ii=args.static_stage_ii,
            dynamic_priority=args.dynamic_priority,
        )
        result = schedule_execution_graph(
            lowered.execution_graph,
            machine,
            policy,
            simulator_config=simulator_config,
        )
        key = (architecture, policy, dependency_window, rob_entries, tile_size)
        results[key] = result
        case_id = f"{architecture}__{policy}__tile{tile_size}__window{dependency_window}__rob{rob_entries}"
        case_dir = args.output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        write_json(result, case_dir / "summary.json")
        write_csv(result, case_dir / "tasks.csv")
        write_svg(result, case_dir / "swimlane.svg")
        write_png(result, case_dir / "swimlane.png")
        write_artifact_json(result.perfetto_trace(), case_dir / "perfetto.json")
        write_artifact_json(result.metrics.get("address_hazards", []), case_dir / "address_dependencies.json")
        write_artifact_json(
            {
                "schema_version": 1,
                "case_id": case_id,
                "benchmark": case.case_id,
                "architecture": architecture,
                "policy": policy,
                "machine_hash": machine.stable_hash(),
                "backend": result.backend,
                "calibration_status": result.metrics["calibration_status"],
                "simulator_config": result.metrics["simulator_config"],
                "total_cycles": result.total_cycles,
                "statistics": lowered.statistics,
            },
            case_dir / "manifest.json",
        )

    static_cycles = {
        (architecture, dependency_window, rob_entries, tile_size): results[(architecture, SchedulerPolicy.STATIC_PIPELINE.value, dependency_window, rob_entries, tile_size)].total_cycles
        for architecture, dependency_window, rob_entries, tile_size in product(architectures, windows, robs, tile_sizes)
        if (architecture, SchedulerPolicy.STATIC_PIPELINE.value, dependency_window, rob_entries, tile_size) in results
    }
    for (architecture, policy, dependency_window, rob_entries, tile_size), result in results.items():
        baseline = static_cycles.get((architecture, dependency_window, rob_entries, tile_size))
        metrics = result.metrics
        records.append(
            {
                "case_id": f"{architecture}__{policy}__tile{tile_size}__window{dependency_window}__rob{rob_entries}",
                "architecture": architecture,
                "policy": policy,
                "tile_size": tile_size,
                "dependency_window": dependency_window,
                "rob_entries": rob_entries,
                "total_cycles": result.total_cycles,
                "speedup_vs_static": (baseline / result.total_cycles if baseline is not None and result.total_cycles else None),
                "address_scoreboard": args.address_scoreboard,
                "dynamic_priority": args.dynamic_priority,
                "address_hazard_count": metrics.get("address_hazard_count", 0),
                "rob_peak": metrics.get("rob_peak", 0),
                "ready_set_peak": metrics.get("ready_set_peak", 0),
                "queue_wait_cycles": metrics.get("queue_wait_cycles", 0.0),
                "resource_utilization": json.dumps(metrics.get("resource_utilization", {}), sort_keys=True),
                "stall_by_reason": json.dumps(metrics.get("stall_by_reason", {}), sort_keys=True),
                "pipeline_drain_cycles": metrics.get("pipeline_drain_cycles", 0.0),
                "completed_tile_count": metrics.get("completed_tile_count", 0),
            }
        )
    records.sort(key=lambda record: (str(record["architecture"]), int(record["tile_size"]), str(record["policy"]), int(record["dependency_window"]), int(record["rob_entries"])))
    with (args.output_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_artifact_json(records, args.output_dir / "sweep.json")
    print(json.dumps({"case_count": len(records), "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def _workload_builders():
    return {
        "two-mm": (build_two_matmul_model, build_two_matmul_case),
        "elementwise": (build_elementwise_model, build_elementwise_case),
        "reduce": (build_reduce_model, build_reduce_case),
        "softmax": (build_softmax_model, build_softmax_case),
        "rmsnorm": (build_rmsnorm_model, build_rmsnorm_case),
        "layernorm": (build_layernorm_model, build_layernorm_case),
        "decoder-block": (build_decoder_block_model, build_decoder_block_case),
        "attention": (build_attention_model, build_attention_case),
        "transformer-block": (build_transformer_block_model, build_transformer_block_case),
    }


def run_sweep_workloads(args: argparse.Namespace) -> int:
    workloads = _parse_list(args.workloads, name="--workloads")
    architectures = _parse_list(args.architectures, name="--architectures")
    policies = _parse_list(args.policies, name="--policies")
    builders = _workload_builders()
    unknown_workloads = sorted(set(workloads) - set(builders))
    if unknown_workloads:
        raise ValueError(f"unknown workload(s): {', '.join(unknown_workloads)}")
    supported_architectures = {"minimal", "wide-mxu", "lpu-like"}
    unknown_architectures = sorted(set(architectures) - supported_architectures)
    if unknown_architectures and args.machine_config is None:
        raise ValueError(f"unknown architecture profile(s): {', '.join(unknown_architectures)}")
    supported_policies = {policy.value for policy in SchedulerPolicy}
    unknown_policies = sorted(set(policies) - supported_policies)
    if unknown_policies:
        raise ValueError(f"unsupported scheduler policy(s): {', '.join(unknown_policies)}")
    windows = _parse_positive_int_list(args.windows, name="--windows")
    robs = _parse_positive_int_list(args.robs, name="--robs")
    tile_sizes = _parse_positive_int_list(args.tile_sizes, name="--tile-sizes")
    dynamic_priorities = _parse_list(args.dynamic_priorities, name="--dynamic-priorities")
    supported_priorities = {"critical_path", "oldest_first"}
    unknown_priorities = sorted(set(dynamic_priorities) - supported_priorities)
    if unknown_priorities:
        raise ValueError(f"unsupported dynamic priority(s): {', '.join(unknown_priorities)}")
    stage_offsets = _parse_offsets(args.static_stage_offsets)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lowered_cache: dict[tuple[str, str, int], tuple[object, object, object, object]] = {}
    results: dict[tuple[str, str, str, int, int, int, str], object] = {}
    records: list[dict[str, object]] = []
    for workload, architecture, policy, dependency_window, rob_entries, tile_size, dynamic_priority in product(
        workloads, architectures, policies, windows, robs, tile_sizes, dynamic_priorities
    ):
        cache_key = (workload, architecture, tile_size)
        machine = _machine(architecture, args.machine_config)
        if cache_key not in lowered_cache:
            model_builder, _case_builder = builders[workload]
            model = model_builder()
            compile_case = builders[workload][1](
                architecture_profile=architecture,
                scheduler_profile=SchedulerPolicy.STATIC_PIPELINE.value,
            )
            instance = model.instantiate(compile_case)
            schedule = default_mixed_schedule(instance.graph, tile_size=tile_size)
            lowered = lower_mixed_model(instance, machine, schedule)
            lowered_cache[cache_key] = (model, instance, schedule, lowered)
        model, instance, schedule, lowered = lowered_cache[cache_key]
        case = builders[workload][1](
            architecture_profile=architecture,
            scheduler_profile=policy,
        )
        static_config = stage_offsets if policy == SchedulerPolicy.STATIC_PIPELINE.value else ()
        simulator_config = _simulator_config(
            dependency_window=dependency_window,
            rob_entries=rob_entries,
            address_scoreboard=args.address_scoreboard,
            static_stage_offsets=static_config,
            static_stage_ii=args.static_stage_ii,
            dynamic_priority=dynamic_priority,
        )
        result = schedule_execution_graph(
            lowered.execution_graph,
            machine,
            policy,
            simulator_config=simulator_config,
        )
        key = (workload, architecture, policy, dependency_window, rob_entries, tile_size, dynamic_priority)
        results[key] = result
        case_id = (
            f"{workload}__{architecture}__{policy}__tile{tile_size}"
            f"__window{dependency_window}__rob{rob_entries}__priority{dynamic_priority}"
        )
        case_dir = args.output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        write_artifact_json(model, case_dir / "model_spec.json")
        write_artifact_json(case, case_dir / "benchmark_case.json")
        write_artifact_json(instance, case_dir / "model_instance.json")
        write_artifact_json(instance.graph, case_dir / "operator_graph.json")
        write_artifact_json(schedule, case_dir / "schedule.json")
        write_artifact_json(lowered.tile_graph, case_dir / "tile_graph.json")
        write_artifact_json(lowered.execution_graph, case_dir / "execution_graph.json")
        write_artifact_json(machine, case_dir / "machine.json")
        write_json(result, case_dir / "summary.json")
        write_csv(result, case_dir / "tasks.csv")
        write_svg(result, case_dir / "swimlane.svg")
        write_png(result, case_dir / "swimlane.png")
        write_artifact_json(result.perfetto_trace(), case_dir / "perfetto.json")
        write_artifact_json(result.metrics.get("address_hazards", []), case_dir / "address_dependencies.json")
        write_artifact_json(
            {
                "schema_version": 1,
                "case_id": case_id,
                "benchmark": case.case_id,
                "workload": workload,
                "architecture": architecture,
                "policy": policy,
                "dynamic_priority": dynamic_priority,
                "machine_hash": machine.stable_hash(),
                "backend": result.backend,
                "calibration_status": result.metrics["calibration_status"],
                "simulator_config": result.metrics["simulator_config"],
                "total_cycles": result.total_cycles,
                "statistics": lowered.statistics,
            },
            case_dir / "manifest.json",
        )

    static_cycles = {
        (workload, architecture, dependency_window, rob_entries, tile_size, dynamic_priority): results[
            (
                workload,
                architecture,
                SchedulerPolicy.STATIC_PIPELINE.value,
                dependency_window,
                rob_entries,
                tile_size,
                dynamic_priority,
            )
        ].total_cycles
        for workload, architecture, dependency_window, rob_entries, tile_size, dynamic_priority in product(
            workloads, architectures, windows, robs, tile_sizes, dynamic_priorities
        )
        if (
            workload,
            architecture,
            SchedulerPolicy.STATIC_PIPELINE.value,
            dependency_window,
            rob_entries,
            tile_size,
            dynamic_priority,
        ) in results
    }
    for (workload, architecture, policy, dependency_window, rob_entries, tile_size, dynamic_priority), result in results.items():
        baseline = static_cycles.get((workload, architecture, dependency_window, rob_entries, tile_size, dynamic_priority))
        metrics = result.metrics
        records.append(
            {
                "workload": workload,
                "architecture": architecture,
                "policy": policy,
                "tile_size": tile_size,
                "dependency_window": dependency_window,
                "rob_entries": rob_entries,
                "total_cycles": result.total_cycles,
                "speedup_vs_static": (
                    baseline / result.total_cycles
                    if baseline is not None and result.total_cycles
                    else None
                ),
                "address_scoreboard": args.address_scoreboard,
                "dynamic_priority": dynamic_priority,
                "address_hazard_count": metrics.get("address_hazard_count", 0),
                "rob_peak": metrics.get("rob_peak", 0),
                "ready_set_peak": metrics.get("ready_set_peak", 0),
                "queue_wait_cycles": metrics.get("queue_wait_cycles", 0.0),
                "stall_by_reason": json.dumps(metrics.get("stall_by_reason", {}), sort_keys=True),
                "pipeline_drain_cycles": metrics.get("pipeline_drain_cycles", 0.0),
                "completed_tile_count": metrics.get("completed_tile_count", 0),
            }
        )
    records.sort(
        key=lambda record: (
            str(record["workload"]),
            str(record["architecture"]),
            int(record["tile_size"]),
            str(record["policy"]),
            int(record["dependency_window"]),
            int(record["rob_entries"]),
            str(record["dynamic_priority"]),
        )
    )
    with (args.output_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_artifact_json(records, args.output_dir / "sweep.json")
    print(json.dumps({"case_count": len(records), "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "two-mm":
        return run_two_mm(args)
    if args.command == "elementwise":
        return run_elementwise(args)
    if args.command == "reduce":
        return run_reduce(args)
    if args.command == "softmax":
        return run_softmax(args)
    if args.command == "rmsnorm":
        return run_rmsnorm(args)
    if args.command == "decoder-block":
        return run_decoder_block(args)
    if args.command == "attention":
        return run_attention(args)
    if args.command == "transformer-block":
        return run_transformer_block(args)
    if args.command == "layernorm":
        return run_layernorm(args)
    if args.command == "sweep-two-mm":
        return run_sweep_two_mm(args)
    if args.command == "sweep-workloads":
        return run_sweep_workloads(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
