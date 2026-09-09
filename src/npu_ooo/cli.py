from __future__ import annotations

"""Command-line entry points for the PyTorch-to-TISA research flow."""

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from npu_ooo.arch import (
    SchedulerPipelineConfig,
    load_machine_config,
    lpu_like_machine_config,
    minimal_machine_config,
    wide_mxu_machine_config,
)
from npu_ooo.backend import (
    AGGREGATIONS,
    INTERVALS,
    default_codegen_backend_registry,
    default_event_backend_registry,
    default_timing_provider_registry,
    import_rtl_completion_trace,
    load_mxu_vcs_log,
)
from npu_ooo.compiler import compile_torch_module
from npu_ooo.experiments import run_paper_benchmark_matrix, run_runtime_device_matrix
from npu_ooo.frontend import (
    LoadedWorkloadConfig,
    Workload,
    build_declarative_workload,
    load_workload_config,
)
from npu_ooo.ir import (
    BackendArtifact,
    DynamicIndexBinding,
    OperatorGraph,
    RuntimeLayoutBinding,
    RuntimeSequence,
    allocate_memory_plan_bindings,
    create_runtime_sequence,
    create_runtime_state_registry,
    create_runtime_submission,
)
from npu_ooo.scheduler import (
    SchedulerPolicy,
    SimulatorConfig,
    schedule_tisa_program,
    schedule_tisa_sequence,
)
from npu_ooo.runtime import load_device_program
from npu_ooo.trace import (
    ensure_output_layout,
    write_artifact_index,
    write_artifact_json,
    write_csv,
    write_execution_graph_dot,
    write_instruction_csv,
    write_json,
    write_operator_graph_dot,
    write_operator_graph_svg,
    write_png,
    write_svg,
    write_tile_graph_dot,
)


class _StoreSpecified(argparse.Action):
    """Store a value while recording that it came from the command line."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, values)
        specified = set(getattr(namespace, "_specified_options", ()))
        specified.add(self.dest)
        setattr(namespace, "_specified_options", specified)


class _AppendSpecified(argparse.Action):
    """Append a value and retain CLI-vs-config provenance."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        current = list(getattr(namespace, self.dest, None) or ())
        current.append(values)
        setattr(namespace, self.dest, current)
        specified = set(getattr(namespace, "_specified_options", ()))
        specified.add(self.dest)
        setattr(namespace, "_specified_options", specified)


class _StoreTrueSpecified(argparse.Action):
    """``store_true`` equivalent which records an explicit override."""

    def __init__(self, option_strings, dest, default=False, required=False, help=None) -> None:
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=True,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, True)
        specified = set(getattr(namespace, "_specified_options", ()))
        specified.add(self.dest)
        setattr(namespace, "_specified_options", specified)


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


def _timing_model(path: Path | None, provider: str | None = None):
    selected = provider or ("timing_table" if path is not None else "analytical")
    return default_timing_provider_registry().create(selected, path)


def _parse_positive_int_list(value: str, *, name: str) -> tuple[int, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError(f"{name} must contain at least one value")
    try:
        numbers = tuple(int(item) for item in items)
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of positive integers") from exc
    if any(number <= 0 for number in numbers):
        raise ValueError(f"{name} must contain only positive integers")
    return numbers


def _descriptor_availability(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"runtime availability config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid runtime availability JSON '{path}': {exc}") from exc
    if isinstance(payload, dict) and "descriptor_available_cycles" in payload:
        payload = payload["descriptor_available_cycles"]
    if not isinstance(payload, dict):
        raise ValueError("runtime availability config must be a TISA-id to cycle mapping")
    result: dict[str, float] = {}
    for tisa_id, cycle in payload.items():
        if not isinstance(tisa_id, str) or not tisa_id:
            raise ValueError("runtime availability TISA ids must be non-empty strings")
        if (
            isinstance(cycle, bool)
            or not isinstance(cycle, (int, float))
            or not math.isfinite(cycle)
            or cycle < 0
        ):
            raise ValueError(f"runtime availability for '{tisa_id}' must be non-negative")
        result[tisa_id] = float(cycle)
    return result


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{description} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {description} JSON '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _runtime_bindings(
    payload: dict[str, Any],
) -> tuple[
    tuple[DynamicIndexBinding, ...],
    tuple[RuntimeLayoutBinding, ...],
]:
    """Decode invocation-level dynamic bindings from a simulation manifest."""

    raw_indices = payload.get("dynamic_indices", ())
    if isinstance(raw_indices, dict):
        raw_indices = [
            {"expression_id": expression_id, "values": values}
            for expression_id, values in raw_indices.items()
        ]
    if not isinstance(raw_indices, (list, tuple)):
        raise ValueError("runtime_config.dynamic_indices must be a list or object")
    indices: list[DynamicIndexBinding] = []
    for item in raw_indices:
        if not isinstance(item, dict) or "expression_id" not in item or "values" not in item:
            raise ValueError("each dynamic index binding needs expression_id and values")
        try:
            binding = DynamicIndexBinding(
                str(item["expression_id"]),
                tuple(int(value) for value in item["values"]),
                attributes=item.get("attributes", {}),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid dynamic index binding") from exc
        issues = binding.validate()
        if issues:
            raise ValueError("invalid dynamic index binding: " + "; ".join(issues))
        indices.append(binding)

    raw_layouts = payload.get("dynamic_layouts", ())
    if isinstance(raw_layouts, dict):
        raw_layouts = [
            {"tensor": tensor, **spec} if isinstance(spec, dict) else {"tensor": tensor}
            for tensor, spec in raw_layouts.items()
        ]
    if not isinstance(raw_layouts, (list, tuple)):
        raise ValueError("runtime_config.dynamic_layouts must be a list or object")
    layouts: list[RuntimeLayoutBinding] = []
    for item in raw_layouts:
        if not isinstance(item, dict):
            raise ValueError("each dynamic layout binding must be an object")
        required = ("tensor", "shape", "strides_bytes")
        if any(key not in item for key in required):
            raise ValueError("each dynamic layout binding needs tensor, shape and strides_bytes")
        try:
            binding = RuntimeLayoutBinding(
                tensor=str(item["tensor"]),
                shape=tuple(int(value) for value in item["shape"]),
                strides_bytes=tuple(int(value) for value in item["strides_bytes"]),
                layout=str(item.get("layout", "runtime")),
                offset_bytes=int(item.get("offset_bytes", 0)),
                attributes=item.get("attributes", {}),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid dynamic layout binding") from exc
        issues = binding.validate()
        if issues:
            raise ValueError("invalid dynamic layout binding: " + "; ".join(issues))
        layouts.append(binding)
    return tuple(indices), tuple(layouts)


def _add_compile_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: str = "out/compile",
) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="schema-v1 JSON workload configuration",
    )
    parser.add_argument(
        "--torch-module",
        action=_StoreSpecified,
        metavar="MODULE:CLASS_OR_FACTORY",
        help="legacy shorthand for a zero-argument nn.Module class or factory",
    )
    parser.add_argument(
        "--input-shape",
        action=_AppendSpecified,
        metavar="D0,D1,...",
        help="legacy randn input shape; repeat once per positional module input",
    )
    parser.add_argument(
        "--input-dtype",
        action=_StoreSpecified,
        choices=("float16", "float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--model-id", action=_StoreSpecified)
    parser.add_argument("--tile-size", type=int, action=_StoreSpecified, default=32)
    parser.add_argument(
        "--tile-size-candidates",
        action=_StoreSpecified,
        help="comma-separated tile sizes ranked by the GC cost model",
    )
    parser.add_argument(
        "--softmax-algorithm",
        action=_StoreSpecified,
        choices=("materialized", "online"),
        default=None,
        help=(
            "Softmax payload strategy: materialized row-wise reductions or the "
            "analytical online state chain"
        ),
    )
    parser.add_argument(
        "--onchip-handoff",
        action=_StoreSpecified,
        choices=("root_memory", "attention_single_consumer"),
        default="root_memory",
        help="target-lowering policy for an eligible Matmul-to-Softmax edge",
    )
    parser.add_argument(
        "--arch",
        action=_StoreSpecified,
        choices=("minimal", "wide-mxu", "lpu-like"),
        default="minimal",
    )
    parser.add_argument("--machine-config", type=Path, action=_StoreSpecified)
    parser.add_argument(
        "--codegen-backend",
        action=_StoreSpecified,
        choices=default_codegen_backend_registry().names(),
        default="analytical",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(default_output_dir))


def _add_simulation_options(
    parser: argparse.ArgumentParser,
    *,
    include_architecture: bool,
    include_output_dir: bool,
    include_runtime_device_matrix: bool = False,
    manifest_overrides: bool = False,
) -> None:
    """Add options consumed after compilation by runtime/device simulation."""

    if include_architecture:
        parser.add_argument(
            "--arch",
            action=_StoreSpecified,
            choices=("minimal", "wide-mxu", "lpu-like"),
        )
        parser.add_argument("--machine-config", type=Path, action=_StoreSpecified)
    parser.add_argument(
        "--scheduler-config",
        type=Path,
        action=_StoreSpecified,
        help="cycle_event control pipeline JSON",
    )
    parser.add_argument("--timing-config", type=Path, action=_StoreSpecified)
    parser.add_argument(
        "--timing-provider",
        action=_StoreSpecified,
        choices=default_timing_provider_registry().names(),
        default=None,
    )
    parser.add_argument(
        "--event-backend",
        action=_StoreSpecified,
        choices=default_event_backend_registry().names(),
        default="analytical_event",
    )
    parser.add_argument(
        "--policy",
        action=_StoreSpecified,
        choices=tuple(policy.value for policy in SchedulerPolicy),
        default=SchedulerPolicy.STATIC_PIPELINE.value,
    )
    if include_output_dir:
        parser.add_argument("--output-dir", type=Path, default=Path("out/simulate"))
    parser.add_argument("--instruction-queue-depth", type=int, action=_StoreSpecified)
    parser.add_argument("--rob-entries", type=int, action=_StoreSpecified)
    parser.add_argument("--max-inflight-tiles", type=int, action=_StoreSpecified)
    parser.add_argument("--dependency-window", type=int, action=_StoreSpecified)
    parser.add_argument("--ready-queue-depth", type=int, action=_StoreSpecified)
    parser.add_argument("--address-scoreboard", action=_StoreTrueSpecified)
    parser.add_argument(
        "--memory-bank-scoreboard",
        action=_StoreTrueSpecified,
        help="model configured memory bank and read/write port conflicts",
    )
    parser.add_argument(
        "--dynamic-priority",
        action=_StoreSpecified,
        choices=("oldest_first", "compiler_hint", "oracle_critical_path", "critical_path"),
        default="oldest_first",
    )
    parser.add_argument(
        "--runtime-policy",
        action=_StoreSpecified,
        choices=("static", "dynamic_ready_queue"),
        default=None if manifest_overrides else "static",
    )
    parser.add_argument("--runtime-chunk-size", type=int, action=_StoreSpecified)
    parser.add_argument(
        "--runtime-base-address",
        type=lambda value: int(value, 0),
        action=_StoreSpecified,
        default=None if manifest_overrides else 0x10000000,
    )
    parser.add_argument(
        "--runtime-alignment",
        type=int,
        action=_StoreSpecified,
        default=None if manifest_overrides else 256,
    )
    parser.add_argument(
        "--runtime-buffer-policy",
        action=_StoreSpecified,
        choices=("linear", "lifetime_reuse"),
        default=None if manifest_overrides else "linear",
    )
    parser.add_argument("--runtime-availability-config", type=Path, action=_StoreSpecified)
    parser.add_argument(
        "--runtime-launch-latency",
        type=float,
        action=_StoreSpecified,
        default=None if manifest_overrides else 0.0,
    )
    parser.add_argument(
        "--runtime-synchronization-cycles",
        type=float,
        action=_StoreSpecified,
        default=None if manifest_overrides else 0.0,
    )
    parser.add_argument(
        "--runtime-invocations",
        type=int,
        action=_StoreSpecified,
        default=None if manifest_overrides else 1,
        help="number of repeated invocations sharing persistent runtime state",
    )
    parser.add_argument(
        "--runtime-inter-invocation-gap",
        type=float,
        action=_StoreSpecified,
        default=None if manifest_overrides else 0.0,
        help="cycles between state completion and the next invocation",
    )
    if include_runtime_device_matrix:
        parser.add_argument(
            "--runtime-device-matrix",
            action=_StoreTrueSpecified,
            help="run the four runtime/device static-dynamic policy combinations",
        )


def _add_compile_and_sim_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: str = "out/compile-and-sim",
) -> None:
    """Compose compile inputs with runtime/device options for the one-shot flow."""

    _add_compile_arguments(parser, default_output_dir=default_output_dir)
    _add_simulation_options(
        parser,
        include_architecture=False,
        include_output_dir=False,
        include_runtime_device_matrix=True,
    )


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add options that affect runtime submission or device simulation only."""

    parser.add_argument("--compile-dir", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path)
    _add_simulation_options(
        parser,
        include_architecture=True,
        include_output_dir=True,
        manifest_overrides=True,
    )


def _add_paper_matrix_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scheduler-config", type=Path, help="cycle_event control pipeline JSON")
    parser.add_argument(
        "--benchmarks",
        default="all",
        help="all or a comma-separated list of paper benchmark case ids",
    )
    parser.add_argument(
        "--variant",
        choices=("micro", "paper_shape"),
        default="micro",
        help="benchmark workload scale; micro is the default reproducible proxy",
    )
    parser.add_argument(
        "--layer-count",
        type=int,
        default=1,
        help="materialized block count; 1 preserves the one-block benchmark rows",
    )
    parser.add_argument(
        "--model-scope",
        choices=("one_block", "model_proxy"),
        default="one_block",
        help="one_block preserves the baseline; model_proxy adds embeddings and model shell",
    )
    parser.add_argument(
        "--deepseek-mode",
        choices=("dense", "moe_proxy"),
        default="dense",
        help="DeepSeek feed-forward path; moe_proxy consumes explicit sparse routing weights",
    )
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument(
        "--tile-size-candidates",
        help="comma-separated tile sizes ranked by the GC cost model",
    )
    parser.add_argument("--arch", choices=("minimal", "wide-mxu", "lpu-like"), default="minimal")
    parser.add_argument("--machine-config", type=Path)
    parser.add_argument("--timing-config", type=Path)
    parser.add_argument(
        "--timing-provider",
        choices=default_timing_provider_registry().names(),
        default=None,
    )
    parser.add_argument(
        "--event-backend",
        choices=default_event_backend_registry().names(),
        default="analytical_event",
    )
    parser.add_argument(
        "--codegen-backend",
        choices=default_codegen_backend_registry().names(),
        default="analytical",
    )
    parser.add_argument(
        "--softmax-algorithm",
        choices=("materialized", "online"),
        default=None,
        help="softmax strategy applied during GC/FC compilation",
    )
    parser.add_argument(
        "--onchip-handoff",
        choices=("root_memory", "attention_single_consumer"),
        default="root_memory",
        help="target-lowering policy for an eligible Matmul-to-Softmax edge",
    )
    parser.add_argument(
        "--runtime-device-matrix",
        action="store_true",
        help="also vary runtime submission policy, producing four combinations per case",
    )
    parser.add_argument(
        "--device-policies",
        default="static_pipeline,dynamic_ready_queue",
        help="comma-separated device scheduler policies",
    )
    parser.add_argument("--instruction-queue-depth", type=int)
    parser.add_argument("--rob-entries", type=int)
    parser.add_argument("--max-inflight-tiles", type=int)
    parser.add_argument("--dependency-window", type=int)
    parser.add_argument("--ready-queue-depth", type=int)
    parser.add_argument("--address-scoreboard", action="store_true")
    parser.add_argument("--memory-bank-scoreboard", action="store_true")
    parser.add_argument(
        "--dynamic-priority",
        choices=("oldest_first", "compiler_hint", "oracle_critical_path", "critical_path"),
        default="oldest_first",
    )
    parser.add_argument("--runtime-chunk-size", type=int)
    parser.add_argument("--runtime-launch-latency", type=float, default=0.0)
    parser.add_argument("--runtime-synchronization-cycles", type=float, default=0.0)
    parser.add_argument(
        "--request-count",
        type=int,
        default=1,
        help="sequential request replays that reuse the same compiled artifact",
    )
    parser.add_argument("--inter-request-gap", type=float, default=0.0)
    parser.add_argument("--runtime-availability-config", type=Path)
    parser.add_argument("--runtime-base-address", type=lambda value: int(value, 0), default=0x10000000)
    parser.add_argument("--runtime-alignment", type=int, default=256)
    parser.add_argument(
        "--runtime-buffer-policy",
        choices=("linear", "lifetime_reuse"),
        default="linear",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record an explicit error row and continue compiling other cases",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("out/paper-matrix"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile PyTorch modules to TISA and run scheduling experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compile_only = commands.add_parser(
        "compile",
        help="PyTorch -> torch.export -> Torch-XLA -> StableHLO -> TISA/backend compile package",
    )
    _add_compile_arguments(compile_only, default_output_dir="out/compile")

    compile_and_sim = commands.add_parser(
        "compile-and-sim",
        help="compile a PyTorch module and immediately run the simulator",
    )
    _add_compile_and_sim_arguments(
        compile_and_sim,
        default_output_dir="out/compile-and-sim",
    )

    simulate = commands.add_parser(
        "simulate",
        help="load a compile package, bind runtime values, and run the simulator",
    )
    _add_simulation_arguments(simulate)

    paper_matrix = commands.add_parser(
        "paper-matrix",
        help="compile the paper benchmark registry once per case and compare device policies",
    )
    _add_paper_matrix_arguments(paper_matrix)

    rtl_trace = commands.add_parser(
        "import-rtl-trace",
        help="convert an RTL completion trace into an MXU timing profile",
    )
    rtl_trace.add_argument("--input", type=Path, required=True)
    rtl_trace.add_argument("--output", type=Path, required=True)
    rtl_trace.add_argument("--interval", choices=INTERVALS, default=INTERVALS[0])
    rtl_trace.add_argument("--aggregation", choices=AGGREGATIONS, default="median")
    rtl_trace.add_argument("--unmatched-matmul", choices=("error", "analytical"), default="error")
    rtl_trace.add_argument("--name", default="systolic_mxu_profile")

    rtl_log = commands.add_parser(
        "import-rtl-log",
        help="parse the repository MXU VCS log into completion-trace JSON",
    )
    rtl_log.add_argument("--input", type=Path, required=True)
    rtl_log.add_argument("--output", type=Path, required=True)
    rtl_log.add_argument("--k-per-tile", type=int, default=8)
    return parser


def run_import_rtl_trace(args: argparse.Namespace) -> int:
    profile = import_rtl_completion_trace(
        args.input,
        interval=args.interval,
        aggregation=args.aggregation,
        unmatched_matmul=args.unmatched_matmul,
        name=args.name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "shape_count": len(profile["matmul_profiles"])},
            sort_keys=True,
        )
    )
    return 0


def run_import_rtl_log(args: argparse.Namespace) -> int:
    trace = load_mxu_vcs_log(args.input, k_per_tile=args.k_per_tile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "record_count": len(trace["records"])}, sort_keys=True))
    return 0


def _load_torch_module(specification: str, input_shapes: list[str], dtype_name: str):
    shapes = tuple(_parse_positive_int_list(value, name="--input-shape") for value in input_shapes)
    workload = build_declarative_workload(
        {
            "factory": specification,
            "kwargs": {},
            "dtype": dtype_name,
        },
        {
            "args": [
                {
                    "kind": "tensor",
                    "shape": list(shape),
                    "dtype": dtype_name,
                    "init": {"kind": "randn"},
                }
                for shape in shapes
            ]
        },
        seed=0,
        source="legacy_cli_shorthand",
    )
    factory_name = specification.split(":", 1)[-1].rsplit(".", 1)[-1]
    return workload.module, workload.args, factory_name


def _option_was_specified(args: argparse.Namespace, name: str) -> bool:
    return name in set(getattr(args, "_specified_options", ()))


def _config_option(
    args: argparse.Namespace,
    options: Mapping[str, Any],
    name: str,
    *,
    default: Any,
) -> Any:
    if _option_was_specified(args, name):
        return getattr(args, name)
    return options.get(name, default)


def _choice(value: Any, *, path: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{path}: must be one of: {', '.join(choices)}")
    return value


def _positive_integer(value: Any, *, path: str, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}: must be a positive integer")
    return value


def _non_negative_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{path}: must be a non-negative number")
    return float(value)


def _tile_size_candidates(value: Any, *, path: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_positive_int_list(value, name=path)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: must be a non-empty array of positive integers")
    return tuple(
        _positive_integer(item, path=f"{path}[{index}]")
        for index, item in enumerate(value)
    )


def _path_option(value: Any, *, path: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{path}: must be a non-empty path")
    return Path(value)


def _apply_runtime_config(
    args: argparse.Namespace,
    loaded: LoadedWorkloadConfig | None,
) -> dict[str, Any]:
    """Apply only existing compile-and-sim options from ``runtime``."""

    options = dict(loaded.runtime_options) if loaded is not None else {}
    names = (
        "scheduler_config",
        "timing_config",
        "timing_provider",
        "event_backend",
        "policy",
        "instruction_queue_depth",
        "rob_entries",
        "max_inflight_tiles",
        "dependency_window",
        "ready_queue_depth",
        "address_scoreboard",
        "memory_bank_scoreboard",
        "dynamic_priority",
        "runtime_policy",
        "runtime_chunk_size",
        "runtime_base_address",
        "runtime_alignment",
        "runtime_buffer_policy",
        "runtime_availability_config",
        "runtime_launch_latency",
        "runtime_synchronization_cycles",
        "runtime_invocations",
        "runtime_inter_invocation_gap",
        "runtime_device_matrix",
    )
    resolved: dict[str, Any] = {}
    for name in names:
        current = getattr(args, name)
        value = _config_option(args, options, name, default=current)
        if name in {
            "scheduler_config",
            "timing_config",
            "runtime_availability_config",
        }:
            value = _path_option(value, path=f"runtime.{name}")
        elif name in {
            "instruction_queue_depth",
            "rob_entries",
            "max_inflight_tiles",
            "dependency_window",
            "ready_queue_depth",
            "runtime_chunk_size",
        }:
            value = _positive_integer(value, path=f"runtime.{name}", optional=True)
        elif name in {"runtime_alignment", "runtime_invocations"}:
            value = _positive_integer(value, path=f"runtime.{name}")
        elif name == "runtime_base_address":
            if isinstance(value, str):
                try:
                    value = int(value, 0)
                except ValueError as exc:
                    raise ValueError("runtime.runtime_base_address: must be an integer") from exc
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("runtime.runtime_base_address: must be a non-negative integer")
        elif name in {
            "runtime_launch_latency",
            "runtime_synchronization_cycles",
            "runtime_inter_invocation_gap",
        }:
            value = _non_negative_number(value, path=f"runtime.{name}")
        elif name in {"address_scoreboard", "memory_bank_scoreboard", "runtime_device_matrix"}:
            if not isinstance(value, bool):
                raise ValueError(f"runtime.{name}: must be a boolean")
        elif name == "timing_provider" and value is not None:
            value = _choice(
                value,
                path="runtime.timing_provider",
                choices=default_timing_provider_registry().names(),
            )
        elif name == "event_backend":
            value = _choice(
                value,
                path="runtime.event_backend",
                choices=default_event_backend_registry().names(),
            )
        elif name == "policy":
            value = _choice(
                value,
                path="runtime.policy",
                choices=tuple(policy.value for policy in SchedulerPolicy),
            )
        elif name == "dynamic_priority":
            value = _choice(
                value,
                path="runtime.dynamic_priority",
                choices=("oldest_first", "compiler_hint", "oracle_critical_path", "critical_path"),
            )
        elif name == "runtime_policy":
            value = _choice(
                value,
                path="runtime.runtime_policy",
                choices=("static", "dynamic_ready_queue"),
            )
        elif name == "runtime_buffer_policy":
            value = _choice(
                value,
                path="runtime.runtime_buffer_policy",
                choices=("linear", "lifetime_reuse"),
            )
        setattr(args, name, value)
        resolved[name] = str(value) if isinstance(value, Path) else value
    return resolved


def _write_policy_matrix(
    root: Path,
    compiled,
    runtime_buffers,
    machine,
    args: argparse.Namespace,
    timing_model,
    simulator_config: SimulatorConfig,
    event_backend,
    descriptor_availability: dict[str, float],
) -> None:
    matrix_root = root / "policy_matrix"
    matrix_root.mkdir(parents=True, exist_ok=True)
    cases = run_runtime_device_matrix(
        compiled.backend_artifact,
        runtime_buffers,
        machine,
        chunk_size=args.runtime_chunk_size,
        launch_latency_cycles=args.runtime_launch_latency,
        synchronization_cycles=args.runtime_synchronization_cycles,
        descriptor_available_cycles=descriptor_availability,
        timing_model=timing_model,
        simulator_config=simulator_config,
        event_backend=event_backend,
    )
    baseline = next(
        case.result.total_cycles
        for case in cases
        if case.runtime_policy == "static"
        and case.device_policy == SchedulerPolicy.STATIC_PIPELINE.value
    )
    records: list[dict[str, object]] = []
    for case in cases:
        case_dir = matrix_root / case.case_id
        ensure_output_layout(case_dir)
        write_artifact_json(case.submission, case_dir / "runtime_submission.json")
        write_json(case.result, case_dir / "summary.json")
        write_csv(case.result, case_dir / "tasks.csv")
        write_instruction_csv(case.result, case_dir / "tisa_instructions.csv")
        write_svg(case.result, case_dir / "swimlane.svg")
        write_png(case.result, case_dir / "swimlane.png")
        write_artifact_json(case.result.perfetto_trace(), case_dir / "perfetto.json")
        record = {
            **case.to_dict(),
            "speedup_vs_static_runtime_static_device": (
                baseline / case.result.total_cycles if case.result.total_cycles else None
            ),
        }
        records.append(record)
        write_artifact_json(record, case_dir / "manifest.json")
    records.sort(key=lambda item: (str(item["runtime_policy"]), str(item["device_policy"])))
    with (matrix_root / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (matrix_root / "sweep.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _paper_case_ids(value: str) -> tuple[str, ...] | None:
    if value.strip().lower() == "all":
        return None
    case_ids = tuple(item.strip() for item in value.split(",") if item.strip())
    if not case_ids:
        raise ValueError("--benchmarks must be 'all' or a comma-separated list of case ids")
    return case_ids


def _paper_device_policies(value: str) -> tuple[str, ...]:
    policies = tuple(item.strip() for item in value.split(",") if item.strip())
    valid = {policy.value for policy in SchedulerPolicy}
    if not policies or any(policy not in valid for policy in policies):
        raise ValueError(
            "--device-policies must contain one or more of: " + ", ".join(sorted(valid))
        )
    if len(set(policies)) != len(policies):
        raise ValueError("--device-policies must not contain duplicates")
    return policies


def _write_compiled_case_artifacts(compiled, case_dir: Path, machine) -> None:
    """Write the shared staged compiler artifacts for one matrix case."""

    ensure_output_layout(case_dir)
    write_artifact_json(compiled.source_frontend, case_dir / "source_frontend_import.json")
    write_artifact_json(compiled.stablehlo, case_dir / "stablehlo_module.json")
    (case_dir / "00_frontend" / "generated.mlir").write_text(
        compiled.stablehlo.text, encoding="utf-8"
    )
    write_artifact_json(compiled.frontend, case_dir / "frontend_import.json")
    write_artifact_json(compiled.graph, case_dir / "canonical_graph.json")
    if compiled.gc_artifact is not None:
        write_artifact_json(compiled.gc_artifact, case_dir / "gc_artifact.json")
        pass_dump_dir = case_dir / "01_gc" / "pass_dumps"
        for snapshot in compiled.gc_artifact.pass_dumps:
            write_artifact_json(
                snapshot,
                pass_dump_dir / f"{snapshot.pass_index:02d}_{snapshot.pass_name}.json",
            )
    write_artifact_json(compiled.schedule, case_dir / "schedule.json")
    write_artifact_json(compiled.attributes["compile_statistics"], case_dir / "compile_statistics.json")
    write_artifact_json(compiled.tile_graph, case_dir / "tile_graph.json")
    if compiled.tisa_dialect is not None:
        write_artifact_json(compiled.tisa_dialect, case_dir / "tisa_dialect.json")
        write_artifact_json(compiled.tisa_dialect.attributes, case_dir / "fc_diagnostics.json")
    write_artifact_json(compiled.tisa_program, case_dir / "tisa_program.json")
    write_artifact_json(compiled, case_dir / "compiled_artifact.json")
    write_artifact_json(compiled.backend_artifact, case_dir / "backend_artifact.json")
    write_artifact_json(compiled.backend_artifact.execution_graph, case_dir / "execution_graph.json")
    write_artifact_json(machine, case_dir / "machine.json")
    write_operator_graph_dot(compiled.graph, case_dir / "operator_graph.dot")
    write_operator_graph_svg(compiled.graph, case_dir / "operator_graph.svg")
    write_tile_graph_dot(compiled.tile_graph, case_dir / "tile_graph.dot")
    write_execution_graph_dot(compiled.backend_artifact.execution_graph, case_dir / "execution_graph.dot")


def _write_paper_policy_artifacts(case_dir: Path, case) -> None:
    """Write one policy's runtime, simulation and trace artifacts."""

    policy_dir = case_dir / "policy_matrix" / case.case_id
    (policy_dir / "05_runtime").mkdir(parents=True, exist_ok=True)
    (policy_dir / "06_simulation").mkdir(parents=True, exist_ok=True)
    (policy_dir / "07_trace").mkdir(parents=True, exist_ok=True)
    runtime_name = (
        "runtime_sequence.json"
        if isinstance(case.submission, RuntimeSequence)
        else "runtime_submission.json"
    )
    write_artifact_json(case.submission, policy_dir / "05_runtime" / runtime_name)
    write_json(case.result, policy_dir / "06_simulation" / "summary.json")
    write_csv(case.result, policy_dir / "06_simulation" / "tasks.csv")
    write_instruction_csv(case.result, policy_dir / "06_simulation" / "tisa_instructions.csv")
    write_svg(case.result, policy_dir / "07_trace" / "swimlane.svg")
    write_png(case.result, policy_dir / "07_trace" / "swimlane.png")
    write_artifact_json(case.result.perfetto_trace(), policy_dir / "07_trace" / "perfetto.json")


def _paper_profile_name(run) -> str:
    if (
        run.model_scope == "one_block"
        and run.layer_count == 1
        and run.deepseek_mode in {"dense", "not_applicable"}
        and run.request_count == 1
        and run.inter_request_gap_cycles == 0
    ):
        return run.variant
    fields = [
        run.variant,
        f"scope-{run.model_scope}",
        f"layers-{run.layer_count}",
    ]
    if run.deepseek_mode != "not_applicable":
        fields.append(f"deepseek-{run.deepseek_mode}")
    fields.extend(
        (
            f"requests-{run.request_count}",
            f"gap-{run.inter_request_gap_cycles:g}",
        )
    )
    return "__".join(fields)


def _write_paper_matrix(root: Path, matrix, machine) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    index_entries: list[dict[str, Any]] = []
    for run in matrix.runs:
        profile = _paper_profile_name(run)
        case_dir = root / run.case_id / profile
        case_dir.mkdir(parents=True, exist_ok=True)
        if run.compiled is not None:
            _write_compiled_case_artifacts(run.compiled, case_dir, machine)
        for case in run.cases:
            _write_paper_policy_artifacts(case_dir, case)
        case_records = [dict(record) for record in run.to_records()]
        for record in case_records:
            record["case_output_dir"] = str(case_dir.relative_to(root))
            record["policy_output_dir"] = str(
                (case_dir / "policy_matrix" / str(record["policy_case_id"])).relative_to(root)
            ) if record.get("policy_case_id") else None
        records.extend(case_records)
        (case_dir / "summary.json").write_text(
            json.dumps(case_records, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (case_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_id": run.case_id,
                    "variant": run.variant,
                    "profile": profile,
                    "layer_count": run.layer_count,
                    "model_scope": run.model_scope,
                    "deepseek_mode": run.deepseek_mode,
                    "spec": dict(run.spec),
                    "status": "error" if run.error else "ok",
                    "error": run.error,
                    "artifact_id": run.artifact_id,
                    "program_id": run.program_id,
                    "tisa_instruction_count": run.tisa_instruction_count,
                    "tile_count": run.tile_count,
                    "primitive_task_count": run.primitive_task_count,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_artifact_index(case_dir)
        index_entries.append(
            {
                "benchmark_id": run.case_id,
                "variant": run.variant,
                "profile": profile,
                "layer_count": run.layer_count,
                "model_scope": run.model_scope,
                "deepseek_mode": run.deepseek_mode,
                "status": "error" if run.error else "ok",
                "case_output_dir": str(case_dir.relative_to(root)),
                "policy_output_dirs": [
                    str((case_dir / "policy_matrix" / case.case_id).relative_to(root))
                    for case in run.cases
                ],
            }
        )
    if records:
        fieldnames = tuple(sorted({key for record in records for key in record}))
        with (root / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                        for key, value in record.items()
                    }
                )
    (root / "sweep.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "matrix_index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": matrix.variant,
                "layer_count": matrix.layer_count,
                "model_scope": matrix.model_scope,
                "deepseek_mode": matrix.deepseek_mode,
                "request_count": matrix.request_count,
                "inter_request_gap_cycles": matrix.inter_request_gap_cycles,
                "case_count": len(index_entries),
                "cases": index_entries,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# 论文模型矩阵\n\n"
        "每个 benchmark 只编译一次，然后在同一份 TISA/backend artifact 和 buffer binding 上比较设备调度策略。"
        "默认 runtime 固定为 static，只改变 device policy；使用 `--runtime-device-matrix` 才会展开四组合。\n\n"
        "`sweep.csv/json` 是跨 case 汇总；默认实验写入 `<benchmark>/<variant>/`，扩展模型/请求实验使用带 scope、层数和请求数的 profile 目录。"
        "`matrix_index.json` 是本次运行实际 case 的权威索引；复用 output 目录时，旧 case 目录可能仍存在，"
        "应以该索引而不是目录枚举为准。当前 workload 是 scaled micro 或 representative paper_shape proxy，"
        "reference 字段来自论文，不能与 analytical cycle 混为绝对性能。\n",
        encoding="utf-8",
    )
    return records


def run_paper_matrix(args: argparse.Namespace) -> int:
    if args.variant == "paper_shape":
        print(
            "warning: paper_shape uses representative large inputs and may require substantial "
            "compile time and memory; it is not a full-model or absolute paper-performance run",
            file=sys.stderr,
        )
    machine = _machine(args.arch, args.machine_config)
    if args.softmax_algorithm is not None:
        machine = replace(
            machine,
            attributes={
                **dict(machine.attributes),
                "softmax_algorithm": args.softmax_algorithm,
            },
        )
    if args.onchip_handoff != "root_memory":
        machine = replace(
            machine,
            attributes={
                **dict(machine.attributes),
                "onchip_handoff_policy": args.onchip_handoff,
            },
        )
    tile_size_candidates = (
        _parse_positive_int_list(args.tile_size_candidates, name="--tile-size-candidates")
        if args.tile_size_candidates
        else None
    )
    simulator_config = _simulation_config(args)
    timing_model = _timing_model(args.timing_config, args.timing_provider)
    event_backend = default_event_backend_registry().create(args.event_backend)
    codegen_backend = default_codegen_backend_registry().create(args.codegen_backend)
    device_policies = _paper_device_policies(args.device_policies)
    descriptor_availability = _descriptor_availability(args.runtime_availability_config)
    runtime_policies = (
        ("static", "dynamic_ready_queue")
        if args.runtime_device_matrix
        else ("static",)
    )
    matrix = run_paper_benchmark_matrix(
        machine,
        case_ids=_paper_case_ids(args.benchmarks),
        variant=args.variant,
        layer_count=args.layer_count,
        model_scope=args.model_scope,
        deepseek_mode=args.deepseek_mode,
        tile_size=args.tile_size,
        tile_size_candidates=tile_size_candidates,
        runtime_policies=runtime_policies,
        device_policies=device_policies,
        runtime_chunk_size=args.runtime_chunk_size,
        runtime_launch_latency=args.runtime_launch_latency,
        runtime_synchronization_cycles=args.runtime_synchronization_cycles,
        descriptor_available_cycles=descriptor_availability,
        runtime_base_address=args.runtime_base_address,
        runtime_alignment=args.runtime_alignment,
        runtime_buffer_policy=args.runtime_buffer_policy,
        softmax_algorithm=args.softmax_algorithm,
        timing_model=timing_model,
        simulator_config=simulator_config,
        event_backend=event_backend,
        codegen_backend=codegen_backend,
        continue_on_error=args.continue_on_error,
        request_count=args.request_count,
        inter_request_gap_cycles=args.inter_request_gap,
    )
    records = _write_paper_matrix(args.output_dir, matrix, machine)
    manifest = {
        "schema_version": 1,
        "variant": args.variant,
        "layer_count": args.layer_count,
        "model_scope": args.model_scope,
        "deepseek_mode": args.deepseek_mode,
        "request_count": args.request_count,
        "inter_request_gap_cycles": args.inter_request_gap,
        "benchmarks": [run.case_id for run in matrix.runs],
        "architecture": args.arch,
        "machine_hash": machine.stable_hash(),
        "tile_size": args.tile_size,
        "tile_size_candidates": list(tile_size_candidates or (args.tile_size,)),
        "softmax_algorithm": machine.attributes.get("softmax_algorithm", "materialized"),
        "onchip_handoff_policy": machine.attributes.get(
            "onchip_handoff_policy", "root_memory"
        ),
        "codegen_backend": codegen_backend.name,
        "runtime_policies": list(runtime_policies),
        "device_policies": list(device_policies),
        "timing_provider": getattr(timing_model, "name", "analytical"),
        "event_backend": event_backend.name,
        "record_count": len(records),
        "failed_cases": [run.case_id for run in matrix.runs if run.error],
    }
    (args.output_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "case_count": len(matrix.runs),
                "record_count": len(records),
                "failed_cases": manifest["failed_cases"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if manifest["failed_cases"] else 0


def _compile_from_args(args: argparse.Namespace):
    loaded = load_workload_config(args.config) if args.config is not None else None
    legacy_fields = ("torch_module", "input_shape", "input_dtype")
    if loaded is not None and any(_option_was_specified(args, name) for name in legacy_fields):
        raise ValueError(
            "--config cannot be combined with --torch-module, --input-shape or "
            "--input-dtype; declare one workload input source"
        )
    if loaded is not None:
        workload = loaded.workload
    else:
        if not args.torch_module:
            raise ValueError("compile requires --config or --torch-module")
        if not args.input_shape:
            raise ValueError("legacy --torch-module shorthand requires at least one --input-shape")
        shapes = tuple(
            _parse_positive_int_list(value, name="--input-shape")
            for value in args.input_shape
        )
        workload = build_declarative_workload(
            {
                "factory": args.torch_module,
                "kwargs": {},
                "dtype": args.input_dtype,
            },
            {
                "args": [
                    {
                        "kind": "tensor",
                        "shape": list(shape),
                        "dtype": args.input_dtype,
                        "init": {"kind": "randn"},
                    }
                    for shape in shapes
                ]
            },
            seed=0,
            source="legacy_cli_shorthand",
        )

    options = dict(loaded.compile_options) if loaded is not None else {}
    arch = _choice(
        _config_option(args, options, "arch", default="minimal"),
        path="compile.arch",
        choices=("minimal", "wide-mxu", "lpu-like"),
    )
    machine_config = _path_option(
        _config_option(args, options, "machine_config", default=None),
        path="compile.machine_config",
    )
    if (
        _option_was_specified(args, "arch")
        and not _option_was_specified(args, "machine_config")
        and "machine_config" in options
    ):
        # An explicit built-in architecture replaces a machine file inherited
        # from JSON.  If both flags are explicit, existing --machine-config
        # precedence is preserved.
        machine_config = None
    tile_size = _positive_integer(
        _config_option(args, options, "tile_size", default=32),
        path="compile.tile_size",
    )
    tile_size_candidates = _tile_size_candidates(
        _config_option(args, options, "tile_size_candidates", default=None),
        path="compile.tile_size_candidates",
    )
    softmax_algorithm = _config_option(args, options, "softmax_algorithm", default=None)
    if softmax_algorithm is not None:
        softmax_algorithm = _choice(
            softmax_algorithm,
            path="compile.softmax_algorithm",
            choices=("materialized", "online"),
        )
    onchip_handoff = _choice(
        _config_option(args, options, "onchip_handoff", default="root_memory"),
        path="compile.onchip_handoff",
        choices=("root_memory", "attention_single_consumer"),
    )
    codegen_backend_name = _choice(
        _config_option(args, options, "codegen_backend", default="analytical"),
        path="compile.codegen_backend",
        choices=default_codegen_backend_registry().names(),
    )
    model_id = _config_option(
        args,
        options,
        "model_id",
        default=workload.experiment_id,
    )
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("compile.model_id: must be a non-empty string")

    args.arch = arch
    args.machine_config = machine_config
    args.tile_size = tile_size
    args.tile_size_candidates = tile_size_candidates
    args.softmax_algorithm = softmax_algorithm
    args.onchip_handoff = onchip_handoff
    args.codegen_backend = codegen_backend_name
    args.model_id = model_id

    machine = _machine(arch, machine_config)
    if softmax_algorithm is not None:
        machine = replace(
            machine,
            attributes={
                **dict(machine.attributes),
                "softmax_algorithm": softmax_algorithm,
            },
        )
    if onchip_handoff != "root_memory":
        machine = replace(
            machine,
            attributes={
                **dict(machine.attributes),
                "onchip_handoff_policy": onchip_handoff,
            },
        )
    codegen_backend = default_codegen_backend_registry().create(codegen_backend_name)
    runtime_resolved = (
        _apply_runtime_config(args, loaded)
        if args.command == "compile-and-sim"
        else {
            "consumed": False,
            "declared": dict(loaded.runtime_options) if loaded is not None else {},
        }
    )
    compiled = compile_torch_module(
        workload.module,
        workload.args,
        machine,
        kwargs=workload.kwargs,
        model_id=model_id,
        tile_size=tile_size,
        tile_size_candidates=tile_size_candidates,
        codegen_backend=codegen_backend,
    )
    resolved_config = {
        "schema_version": 1,
        "source_path": str(loaded.source_path) if loaded is not None else None,
        "source_sha256": loaded.source_sha256 if loaded is not None else None,
        "seed": workload.provenance.get("seed", 0),
        "experiment_id": workload.experiment_id,
        "construction": workload.provenance.get("construction"),
        "workload": workload.to_dict(),
        "compile": {
            "arch": arch,
            "machine_config": str(machine_config) if machine_config is not None else None,
            "tile_size": tile_size,
            "tile_size_candidates": list(tile_size_candidates or (tile_size,)),
            "softmax_algorithm": softmax_algorithm,
            "onchip_handoff": onchip_handoff,
            "codegen_backend": codegen_backend_name,
            "model_id": model_id,
        },
        "runtime": runtime_resolved,
        "precedence": "explicit_cli>json>defaults",
        "static_shapes": True,
    }
    factory_name = str(
        workload.provenance.get("model_factory")
        or workload.provenance.get("workload_factory")
        or type(workload.module).__name__
    )
    return compiled, machine, workload, resolved_config, factory_name


def _write_compile_artifacts(
    compiled,
    machine,
    output_dir: Path,
    workload: Workload | None = None,
    resolved_config: Mapping[str, Any] | None = None,
) -> None:
    """Persist the compiler-owned stages used by both CLI execution modes."""

    ensure_output_layout(output_dir)
    if workload is not None:
        write_artifact_json(workload, output_dir / "workload.json")
        write_artifact_json(
            workload.input_signature,
            output_dir / "input_signature.json",
        )
    if resolved_config is not None:
        write_artifact_json(resolved_config, output_dir / "resolved_workload_config.json")
    write_artifact_json(compiled.source_frontend, output_dir / "source_frontend_import.json")
    write_artifact_json(compiled.stablehlo, output_dir / "stablehlo_module.json")
    (output_dir / "00_frontend" / "generated.mlir").write_text(
        compiled.stablehlo.text,
        encoding="utf-8",
    )
    write_artifact_json(compiled.frontend, output_dir / "frontend_import.json")
    write_artifact_json(compiled.graph, output_dir / "canonical_graph.json")
    if compiled.gc_artifact is not None:
        write_artifact_json(compiled.gc_artifact, output_dir / "gc_artifact.json")
        pass_dump_dir = output_dir / "01_gc" / "pass_dumps"
        for snapshot in compiled.gc_artifact.pass_dumps:
            filename = f"{snapshot.pass_index:02d}_{snapshot.pass_name}.json"
            write_artifact_json(snapshot, pass_dump_dir / filename)
    write_artifact_json(compiled.schedule, output_dir / "schedule.json")
    write_artifact_json(
        compiled.attributes["compile_statistics"],
        output_dir / "compile_statistics.json",
    )
    write_artifact_json(compiled.tile_graph, output_dir / "tile_graph.json")
    if compiled.tisa_dialect is not None:
        write_artifact_json(compiled.tisa_dialect, output_dir / "tisa_dialect.json")
        write_artifact_json(
            compiled.tisa_dialect.attributes,
            output_dir / "fc_diagnostics.json",
        )
    if compiled.virtual_tisa_program is not None:
        write_artifact_json(
            compiled.virtual_tisa_program,
            output_dir / "virtual_tisa_program.json",
        )
    write_artifact_json(compiled.tisa_program, output_dir / "tisa_program.json")
    write_artifact_json(compiled, output_dir / "compiled_artifact.json")
    write_artifact_json(compiled.backend_artifact, output_dir / "backend_artifact.json")
    if compiled.backend_artifact.target_plan is None:
        raise ValueError("codegen backend did not produce a target plan")
    write_artifact_json(
        compiled.backend_artifact.target_plan,
        output_dir / "target_plan.json",
    )
    if compiled.backend_artifact.memory_plan is None:
        raise ValueError("codegen backend did not produce a target memory plan")
    write_artifact_json(
        compiled.backend_artifact.memory_plan,
        output_dir / "memory_plan.json",
    )
    write_artifact_json(compiled.backend_artifact.execution_graph, output_dir / "execution_graph.json")
    write_artifact_json(machine, output_dir / "machine.json")
    write_operator_graph_dot(compiled.graph, output_dir / "operator_graph.dot")
    write_operator_graph_svg(compiled.graph, output_dir / "operator_graph.svg")
    write_tile_graph_dot(compiled.tile_graph, output_dir / "tile_graph.dot")
    write_execution_graph_dot(compiled.backend_artifact.execution_graph, output_dir / "execution_graph.dot")


def _compile_manifest(compiled, machine) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "artifact_kind": "compile_package",
        "compiler_pipeline": compiled.attributes["compiler_pipeline"],
        "frontend_path": compiled.attributes["frontend_path"],
        "model_id": compiled.source_frontend.model_id,
        "workload_schema_version": 1,
        "workload_artifacts": {
            "workload": "00_frontend/workload.json",
            "input_signature": "00_frontend/input_signature.json",
            "resolved_config": "00_frontend/resolved_workload_config.json",
        },
        "stablehlo_exporter": "torch-xla",
        "stablehlo_exporter_version": compiled.attributes["stablehlo_exporter_version"],
        "stablehlo_verified": True,
        "stablehlo_version": compiled.attributes["stablehlo_version"],
        "architecture": machine.config_id,
        "machine_hash": machine.stable_hash(),
        "machine_topology_hash": machine.topology_hash(),
        "onchip_handoff_policy": machine.attributes.get(
            "onchip_handoff_policy", "root_memory"
        ),
        "memory_plan_schema": compiled.backend_artifact.memory_plan.schema_version,
        "target_plan_schema": compiled.backend_artifact.target_plan.schema_version,
        "codegen_backend": compiled.attributes["codegen_backend"],
        "tisa_program_id": compiled.tisa_program.program_id,
        "artifact_id": compiled.backend_artifact.artifact_id,
        "compile_artifacts": {
            "workload": "00_frontend/workload.json",
            "input_signature": "00_frontend/input_signature.json",
            "resolved_workload_config": "00_frontend/resolved_workload_config.json",
            "backend_artifact": "04_backend/backend_artifact.json",
            "canonical_graph": "01_gc/canonical_graph.json",
            "machine": "04_backend/machine.json",
            "tisa_program": "03_tisa/tisa_program.json",
            "virtual_tisa_program": "03_tisa/virtual_tisa_program.json",
            "target_plan": "04_backend/target_plan.json",
            "memory_plan": "04_backend/memory_plan.json",
        },
    }


def run_compile(args: argparse.Namespace) -> int:
    compiled, machine, workload, resolved_config, _factory_name = _compile_from_args(args)
    _write_compile_artifacts(
        compiled,
        machine,
        args.output_dir,
        workload,
        resolved_config,
    )
    manifest = _compile_manifest(compiled, machine)
    write_artifact_json(manifest, args.output_dir / "manifest.json")
    write_artifact_index(args.output_dir)
    print(json.dumps({
        "artifact_id": compiled.backend_artifact.artifact_id,
        "model_id": compiled.source_frontend.model_id,
        "tisa_instructions": len(compiled.tisa_program.instructions),
        "output_dir": str(args.output_dir),
    }, sort_keys=True))
    return 0


def run_compile_and_sim(args: argparse.Namespace) -> int:
    compiled, machine, workload, resolved_config, _factory_name = _compile_from_args(args)
    _write_compile_artifacts(
        compiled,
        machine,
        args.output_dir,
        workload,
        resolved_config,
    )

    if compiled.backend_artifact.memory_plan is None:
        raise ValueError("compiled artifact is missing target memory plan")
    runtime_buffers = allocate_memory_plan_bindings(
        compiled.backend_artifact.memory_plan,
        machine,
        base_address=args.runtime_base_address,
    )
    descriptor_availability = _descriptor_availability(args.runtime_availability_config)
    if args.runtime_invocations <= 0:
        raise ValueError("--runtime-invocations must be a positive integer")
    runtime_sequence = None
    if args.runtime_invocations > 1:
        if args.runtime_device_matrix:
            raise ValueError(
                "--runtime-device-matrix currently supports one invocation; "
                "run a RuntimeSequence separately for multi-step decode"
            )
        state_registry = create_runtime_state_registry(
            compiled.backend_artifact,
            runtime_buffers,
        )
        if not state_registry.state_ids():
            raise ValueError(
                "--runtime-invocations > 1 requires a compiled persistent state contract"
            )
        runtime_sequence = create_runtime_sequence(
            compiled.backend_artifact,
            state_registry,
            invocation_count=args.runtime_invocations,
            sequence_id=f"sequence.{compiled.tisa_program.program_id}",
            policy=args.runtime_policy,
            chunk_size=args.runtime_chunk_size,
            launch_latency_cycles=args.runtime_launch_latency,
            synchronization_cycles=args.runtime_synchronization_cycles,
            descriptor_available_cycles=descriptor_availability,
            inter_invocation_gap_cycles=args.runtime_inter_invocation_gap,
        )
        runtime_submission = runtime_sequence.invocations[0]
        write_artifact_json(runtime_sequence, args.output_dir / "runtime_sequence.json")
    else:
        runtime_submission = create_runtime_submission(
            compiled.backend_artifact,
            runtime_buffers,
            submission_id=f"submission.{compiled.tisa_program.program_id}",
            policy=args.runtime_policy,
            chunk_size=args.runtime_chunk_size,
            launch_latency_cycles=args.runtime_launch_latency,
            synchronization_cycles=args.runtime_synchronization_cycles,
            descriptor_available_cycles=descriptor_availability,
        )
    write_artifact_json(runtime_submission, args.output_dir / "runtime_submission.json")
    write_artifact_json(
        load_device_program(compiled.backend_artifact, runtime_submission),
        args.output_dir / "bound_device_program.json",
    )

    simulator_config = _simulation_config(args)
    timing_model = _timing_model(args.timing_config, args.timing_provider)
    event_backend = default_event_backend_registry().create(args.event_backend)
    if runtime_sequence is not None:
        result = schedule_tisa_sequence(
            compiled.backend_artifact,
            runtime_sequence,
            machine,
            args.policy,
            timing_model=timing_model,
            simulator_config=simulator_config,
            event_backend=event_backend,
        )
    else:
        result = schedule_tisa_program(
            compiled.backend_artifact,
            machine,
            args.policy,
            timing_model=timing_model,
            simulator_config=simulator_config,
            runtime_submission=runtime_submission,
            event_backend=event_backend,
        )
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_instruction_csv(result, args.output_dir / "tisa_instructions.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        result.metrics.get("address_hazards", []),
        args.output_dir / "address_dependencies.json",
    )

    allocation_span = (
        max(buffer.end_address for buffer in runtime_buffers)
        - min(buffer.base_address for buffer in runtime_buffers)
        if runtime_buffers
        else 0
    )
    manifest = {
        "schema_version": 1,
        "artifact_kind": "simulation_result",
        "compiler_pipeline": compiled.attributes["compiler_pipeline"],
        "frontend_path": compiled.attributes["frontend_path"],
        "model_id": compiled.source_frontend.model_id,
        "workload_schema_version": 1,
        "workload_artifacts": {
            "workload": "00_frontend/workload.json",
            "input_signature": "00_frontend/input_signature.json",
            "resolved_config": "00_frontend/resolved_workload_config.json",
        },
        "stablehlo_exporter": "torch-xla",
        "stablehlo_exporter_version": compiled.attributes["stablehlo_exporter_version"],
        "stablehlo_verified": True,
        "stablehlo_version": compiled.attributes["stablehlo_version"],
        "architecture": machine.config_id,
        "softmax_algorithm": machine.attributes.get("softmax_algorithm", "materialized"),
        "onchip_handoff_policy": machine.attributes.get(
            "onchip_handoff_policy", "root_memory"
        ),
        "machine_hash": machine.stable_hash(),
        "codegen_backend": compiled.attributes["codegen_backend"],
        "timing_provider": getattr(timing_model, "name", "analytical"),
        "event_backend": event_backend.name,
        "runtime_policy": runtime_submission.policy,
        "runtime_buffer_policy": "compiled_memory_plan",
        "requested_legacy_runtime_buffer_policy": args.runtime_buffer_policy,
        "runtime_command_chunk_count": len(runtime_submission.commands),
        "runtime_buffer_count": len(runtime_submission.buffers),
        "runtime_invocation_count": (
            len(runtime_sequence.invocations) if runtime_sequence is not None else 1
        ),
        "runtime_state_contract": (
            runtime_sequence.attributes.get("state_contract")
            if runtime_sequence is not None
            else runtime_submission.attributes.get("state_contract")
        ),
        "runtime_state_ids": (
            list(runtime_sequence.state_registry.state_ids())
            if runtime_sequence is not None
            else [
                item["state_id"]
                for item in runtime_submission.attributes.get("state_buffers", ())
                if isinstance(item, dict) and "state_id" in item
            ]
        ),
        "runtime_state_dependency_count": (
            len(runtime_sequence.dependencies) if runtime_sequence is not None else 0
        ),
        "runtime_allocation_span_bytes": allocation_span,
        "policy": result.policy,
        "scheduler_target": "tisa",
        "tisa_instruction_count": len(compiled.tisa_program.instructions),
        "primitive_task_count": len(compiled.backend_artifact.execution_graph.tasks),
        "total_cycles": result.total_cycles,
        "total_cycles_including_runtime": result.metrics.get(
            "total_cycles_including_runtime", result.total_cycles
        ),
        "calibration_status": result.metrics["calibration_status"],
        "simulator_config": simulator_config.to_dict(),
    }
    write_artifact_json(manifest, args.output_dir / "manifest.json")

    if args.runtime_device_matrix:
        _write_policy_matrix(
            args.output_dir,
            compiled,
            runtime_buffers,
            machine,
            args,
            timing_model,
            simulator_config,
            event_backend,
            descriptor_availability,
        )
    write_artifact_index(args.output_dir)
    print(
        json.dumps(
            {
                "model_id": compiled.source_frontend.model_id,
                "tisa_instructions": len(compiled.tisa_program.instructions),
                "total_cycles": result.total_cycles,
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def _load_compile_package(root: Path):
    """Load the portable compiler-owned files without importing PyTorch."""

    root = root.expanduser().resolve()
    backend_path = root / "04_backend" / "backend_artifact.json"
    graph_path = root / "01_gc" / "canonical_graph.json"
    machine_path = root / "04_backend" / "machine.json"
    for path in (backend_path, graph_path, machine_path):
        if not path.is_file():
            raise ValueError(f"compile package is missing required artifact: {path}")
    backend = BackendArtifact.from_dict(_read_json_object(backend_path, description="backend artifact"))
    if backend.memory_plan is None:
        raise ValueError(
            "legacy compile package lacks target memory plan schema v2; recompile it "
            "before independent simulation"
        )
    if backend.target_plan is None:
        raise ValueError(
            "compile package lacks target lowering plan schema v1; recompile it "
            "before independent simulation"
        )
    graph = OperatorGraph.from_dict(_read_json_object(graph_path, description="canonical graph"))
    from npu_ooo.arch import MachineConfig

    machine = MachineConfig.from_dict(_read_json_object(machine_path, description="machine config"))
    manifest_path = root / "manifest.json"
    manifest = (
        _read_json_object(manifest_path, description="compile manifest")
        if manifest_path.is_file()
        else {}
    )
    if manifest.get("artifact_kind") not in {None, "compile_package", "simulation_result"}:
        raise ValueError(f"'{manifest_path}' is not a compatible compile/simulation manifest")
    manifest_artifact_id = manifest.get("artifact_id", manifest.get("compile_artifact_id"))
    if manifest_artifact_id not in {None, backend.artifact_id}:
        raise ValueError("compile manifest artifact_id does not match backend artifact")
    return root, backend, graph, machine, manifest


def _simulation_config(args: argparse.Namespace) -> SimulatorConfig:
    pipeline = None
    if args.scheduler_config is not None:
        if args.event_backend != "cycle_event":
            raise ValueError("--scheduler-config requires --event-backend cycle_event")
        pipeline = SchedulerPipelineConfig.from_dict(
            _read_json_object(args.scheduler_config, description="scheduler pipeline")
        )
    return SimulatorConfig(
        instruction_queue_depth=args.instruction_queue_depth,
        rob_entries=args.rob_entries,
        max_inflight_tiles=args.max_inflight_tiles,
        dependency_window=args.dependency_window,
        ready_queue_depth=args.ready_queue_depth,
        address_scoreboard=args.address_scoreboard,
        memory_bank_scoreboard=args.memory_bank_scoreboard,
        dynamic_priority=args.dynamic_priority,
        pipeline=pipeline,
    )


def run_simulate(args: argparse.Namespace) -> int:
    compile_root, artifact, graph, compiled_machine, compile_manifest = _load_compile_package(
        args.compile_dir
    )
    runtime_payload = (
        _read_json_object(args.runtime_config, description="runtime config")
        if args.runtime_config is not None
        else {}
    )
    machine = (
        _machine(args.arch, args.machine_config)
        if args.arch is not None or args.machine_config is not None
        else compiled_machine
    )
    if machine.topology_hash() != artifact.memory_plan.machine_topology_hash:
        raise ValueError(
            "simulate MachineConfig changes storage topology, transfer engines, or operand "
            "placement; target lowering must be rerun"
        )
    runtime_policy = args.runtime_policy or str(runtime_payload.get("runtime_policy", "static"))
    chunk_size = args.runtime_chunk_size
    if chunk_size is None and runtime_payload.get("runtime_chunk_size") is not None:
        chunk_size = int(runtime_payload["runtime_chunk_size"])
    base_address = args.runtime_base_address
    if base_address is None:
        base_address = int(runtime_payload.get("runtime_base_address", 0x10000000))
    _requested_alignment = args.runtime_alignment
    if _requested_alignment is None:
        _requested_alignment = int(runtime_payload.get("runtime_alignment", 256))
    _requested_buffer_policy = args.runtime_buffer_policy or str(
        runtime_payload.get("runtime_buffer_policy", "linear")
    )
    launch_latency = args.runtime_launch_latency
    if launch_latency is None:
        launch_latency = float(runtime_payload.get("runtime_launch_latency", 0.0))
    synchronization = args.runtime_synchronization_cycles
    if synchronization is None:
        synchronization = float(runtime_payload.get("runtime_synchronization_cycles", 0.0))
    gap = args.runtime_inter_invocation_gap
    if gap is None:
        gap = float(runtime_payload.get("runtime_inter_invocation_gap", 0.0))
    invocation_count = args.runtime_invocations
    if invocation_count is None:
        invocation_count = int(runtime_payload.get("runtime_invocations", 1))
    if invocation_count <= 0:
        raise ValueError("runtime invocation count must be positive")
    availability_path = args.runtime_availability_config
    if availability_path is None and runtime_payload.get("runtime_availability_config"):
        availability_path = (
            args.runtime_config.parent
            / str(runtime_payload["runtime_availability_config"])
        ).resolve()
    if availability_path is not None:
        descriptor_availability = _descriptor_availability(availability_path)
    else:
        inline_availability = runtime_payload.get("descriptor_available_cycles", {})
        if not isinstance(inline_availability, dict):
            raise ValueError("runtime_config.descriptor_available_cycles must be an object")
        descriptor_availability = {}
        for tisa_id, cycle in inline_availability.items():
            if not isinstance(tisa_id, str) or not tisa_id:
                raise ValueError("descriptor availability TISA ids must be non-empty strings")
            if (
                isinstance(cycle, bool)
                or not isinstance(cycle, (int, float))
                or not math.isfinite(cycle)
                or cycle < 0
            ):
                raise ValueError(f"descriptor availability for '{tisa_id}' must be non-negative")
            descriptor_availability[tisa_id] = float(cycle)
    buffers = allocate_memory_plan_bindings(
        artifact.memory_plan,
        machine,
        base_address=base_address,
    )

    invocation_payloads = runtime_payload.get("invocations")
    if invocation_payloads is not None:
        if not isinstance(invocation_payloads, list) or len(invocation_payloads) != invocation_count:
            raise ValueError("runtime_config.invocations length must equal runtime invocation count")
    else:
        invocation_payloads = [runtime_payload] * invocation_count
    invocation_indices: list[tuple[DynamicIndexBinding, ...]] = []
    invocation_layouts: list[tuple[RuntimeLayoutBinding, ...]] = []
    for item in invocation_payloads:
        if not isinstance(item, dict):
            raise ValueError("each runtime invocation must be an object")
        indices, layouts = _runtime_bindings(item)
        invocation_indices.append(indices)
        invocation_layouts.append(layouts)

    runtime_sequence = None
    if invocation_count > 1:
        state_registry = create_runtime_state_registry(artifact, buffers)
        if not state_registry.state_ids():
            raise ValueError("multiple runtime invocations require a persistent state contract")
        runtime_sequence = create_runtime_sequence(
            artifact,
            state_registry,
            invocation_count=invocation_count,
            sequence_id=f"sequence.{artifact.program.program_id}",
            policy=runtime_policy,
            chunk_size=chunk_size,
            launch_latency_cycles=launch_latency,
            synchronization_cycles=synchronization,
            descriptor_available_cycles=descriptor_availability,
            inter_invocation_gap_cycles=gap,
            invocation_dynamic_indices=invocation_indices,
            invocation_dynamic_layouts=invocation_layouts,
        )
        runtime_submission = runtime_sequence.invocations[0]
    else:
        runtime_submission = create_runtime_submission(
            artifact,
            buffers,
            submission_id=f"submission.{artifact.program.program_id}",
            policy=runtime_policy,
            chunk_size=chunk_size,
            launch_latency_cycles=launch_latency,
            synchronization_cycles=synchronization,
            descriptor_available_cycles=descriptor_availability,
            dynamic_index_bindings=invocation_indices[0],
            dynamic_layout_bindings=invocation_layouts[0],
        )

    timing_model = _timing_model(args.timing_config, args.timing_provider)
    event_backend = default_event_backend_registry().create(args.event_backend)
    simulator_config = _simulation_config(args)
    if runtime_sequence is not None:
        result = schedule_tisa_sequence(
            artifact,
            runtime_sequence,
            machine,
            args.policy,
            timing_model=timing_model,
            simulator_config=simulator_config,
            event_backend=event_backend,
        )
    else:
        result = schedule_tisa_program(
            artifact,
            machine,
            args.policy,
            timing_model=timing_model,
            simulator_config=simulator_config,
            runtime_submission=runtime_submission,
            event_backend=event_backend,
        )

    ensure_output_layout(args.output_dir)
    write_artifact_json(runtime_submission, args.output_dir / "runtime_submission.json")
    write_artifact_json(
        load_device_program(artifact, runtime_submission),
        args.output_dir / "bound_device_program.json",
    )
    if runtime_sequence is not None:
        write_artifact_json(runtime_sequence, args.output_dir / "runtime_sequence.json")
    write_json(result, args.output_dir / "summary.json")
    write_csv(result, args.output_dir / "tasks.csv")
    write_instruction_csv(result, args.output_dir / "tisa_instructions.csv")
    write_svg(result, args.output_dir / "swimlane.svg")
    write_png(result, args.output_dir / "swimlane.png")
    write_artifact_json(result.perfetto_trace(), args.output_dir / "perfetto.json")
    write_artifact_json(
        result.metrics.get("address_hazards", []),
        args.output_dir / "address_dependencies.json",
    )
    manifest = {
        "schema_version": 1,
        "artifact_kind": "simulation_result",
        "compile_package": str(compile_root),
        "compile_artifact_id": artifact.artifact_id,
        "compile_program_id": artifact.program.program_id,
        "compile_manifest": compile_manifest,
        "architecture": machine.config_id,
        "machine_hash": machine.stable_hash(),
        "timing_provider": getattr(timing_model, "name", "analytical"),
        "event_backend": event_backend.name,
        "policy": result.policy,
        "runtime_policy": runtime_submission.policy,
        "runtime_invocation_count": invocation_count,
        "dynamic_index_binding_count": sum(len(item) for item in invocation_indices),
        "dynamic_layout_binding_count": sum(len(item) for item in invocation_layouts),
        "total_cycles": result.total_cycles,
        "total_cycles_including_runtime": result.metrics.get(
            "total_cycles_including_runtime", result.total_cycles
        ),
        "simulator_config": simulator_config.to_dict(),
    }
    write_artifact_json(manifest, args.output_dir / "manifest.json")
    write_artifact_index(args.output_dir)
    print(json.dumps({
        "compile_artifact_id": artifact.artifact_id,
        "total_cycles": result.total_cycles,
        "output_dir": str(args.output_dir),
    }, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runners = {
        "compile": run_compile,
        "compile-and-sim": run_compile_and_sim,
        "simulate": run_simulate,
        "paper-matrix": run_paper_matrix,
        "import-rtl-trace": run_import_rtl_trace,
        "import-rtl-log": run_import_rtl_log,
    }
    return runners[args.command](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
