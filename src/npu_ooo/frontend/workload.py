"""Generic, benchmark-independent PyTorch workload configuration.

The compiler still accepts ``module, args, kwargs`` as its framework boundary.
This module only makes construction of those values explicit, reproducible and
auditable; it does not add dynamic shapes or bypass torch.export/Torch-XLA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


_ROOT_FIELDS = {
    "schema_version",
    "seed",
    "experiment_id",
    "model",
    "inputs",
    "workload",
    "compile",
    "runtime",
}
_MODEL_FIELDS = {"factory", "kwargs", "dtype"}
_WORKLOAD_FIELDS = {"factory", "kwargs"}
_INPUT_FIELDS = {"args", "kwargs"}
_TENSOR_FIELDS = {"kind", "name", "shape", "dtype", "init"}
_CONTAINER_FIELDS = {"kind", "items"}
_SCALAR_FIELDS = {"kind", "value"}
_COMPILE_FIELDS = {
    "arch",
    "machine_config",
    "tile_size",
    "tile_size_candidates",
    "softmax_algorithm",
    "onchip_handoff",
    "codegen_backend",
    "model_id",
}
_RUNTIME_FIELDS = {
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
}
_PATH_OPTIONS = {
    "machine_config",
    "scheduler_config",
    "timing_config",
    "runtime_availability_config",
}
_DTYPES = {
    "float32",
    "float16",
    "bfloat16",
    "int64",
    "int32",
    "int16",
    "int8",
    "uint8",
    "bool",
}
_FLOAT_DTYPES = {"float32", "float16", "bfloat16"}
_INTEGER_DTYPES = {"int64", "int32", "int16", "int8", "uint8"}


def _error(path: str, message: str) -> ValueError:
    return ValueError(f"{path}: {message}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _error(path, "must be an object")
    if any(not isinstance(key, str) or not key for key in value):
        raise _error(path, "keys must be non-empty strings")
    return dict(value)


def _only_fields(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise _error(path, "unknown field(s): " + ", ".join(unknown))


def _import_callable(specification: Any, path: str) -> Any:
    if not isinstance(specification, str) or ":" not in specification:
        raise _error(path, "must use MODULE:CLASS_OR_FACTORY syntax")
    module_name, attribute_path = specification.split(":", 1)
    if not module_name or not attribute_path:
        raise _error(path, "must use MODULE:CLASS_OR_FACTORY syntax")
    try:
        python_module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise _error(path, f"cannot import module '{module_name}': {exc}") from exc
    value: Any = python_module
    try:
        for attribute in attribute_path.split("."):
            value = getattr(value, attribute)
    except AttributeError as exc:
        raise _error(path, f"'{specification}' does not exist") from exc
    if not callable(value):
        raise _error(path, f"'{specification}' is not callable")
    return value


def _torch_dtype(torch: Any, name: Any, path: str, *, floating_only: bool = False) -> Any:
    if not isinstance(name, str) or name not in _DTYPES:
        raise _error(path, "must be one of: " + ", ".join(sorted(_DTYPES)))
    if floating_only and name not in _FLOAT_DTYPES:
        raise _error(path, "model dtype must be float32, float16 or bfloat16")
    return getattr(torch, name)


def _shape(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise _error(path, "must be an array of compile-time integer extents")
    result: list[int] = []
    for index, extent in enumerate(value):
        if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
            raise _error(f"{path}[{index}]", "must be a positive integer")
        result.append(extent)
    return tuple(result)


def _initializer(value: Any, path: str) -> dict[str, Any]:
    if isinstance(value, str):
        result = {"kind": value}
    else:
        result = _object(value, path)
    kind = result.get("kind")
    if not isinstance(kind, str):
        raise _error(f"{path}.kind", "must be a string")
    fields = {
        "zeros": {"kind"},
        "ones": {"kind"},
        "randn": {"kind"},
        "randint": {"kind", "low", "high"},
        "explicit": {"kind", "values"},
        "arange": {"kind", "start", "step"},
        "file": {"kind", "path"},
    }
    if kind not in fields:
        raise _error(
            f"{path}.kind",
            "must be one of: " + ", ".join(fields),
        )
    _only_fields(result, fields[kind], path)
    return result


def _numel(shape: Sequence[int]) -> int:
    return math.prod(shape) if shape else 1


def _tensor_from_spec(
    spec: Mapping[str, Any],
    *,
    path: str,
    config_dir: Path,
    initializers: dict[str, Any],
) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise _error(path, "PyTorch is required to construct tensor inputs") from exc

    _only_fields(spec, _TENSOR_FIELDS, path)
    if spec.get("kind") != "tensor":
        raise _error(f"{path}.kind", "must be 'tensor'")
    name = spec.get("name")
    if name is not None and (not isinstance(name, str) or not name):
        raise _error(f"{path}.name", "must be a non-empty string")
    if "shape" not in spec:
        raise _error(f"{path}.shape", "is required (use [] for a scalar tensor)")
    shape = _shape(spec["shape"], f"{path}.shape")
    if "dtype" not in spec:
        raise _error(f"{path}.dtype", "is required")
    dtype_name = spec["dtype"]
    dtype = _torch_dtype(torch, dtype_name, f"{path}.dtype")
    if "init" not in spec:
        raise _error(f"{path}.init", "is required")
    init = _initializer(spec["init"], f"{path}.init")
    kind = init["kind"]
    count = _numel(shape)
    audit_init = dict(init)

    if kind == "zeros":
        tensor = torch.zeros(shape, dtype=dtype)
    elif kind == "ones":
        tensor = torch.ones(shape, dtype=dtype)
    elif kind == "randn":
        if dtype_name not in _FLOAT_DTYPES:
            raise _error(f"{path}.init.kind", "randn requires a floating tensor dtype")
        tensor = torch.randn(shape, dtype=dtype)
    elif kind == "randint":
        if dtype_name not in _INTEGER_DTYPES and dtype_name != "bool":
            raise _error(f"{path}.init.kind", "randint requires an integer or bool tensor dtype")
        low = init.get("low")
        high = init.get("high")
        if isinstance(low, bool) or not isinstance(low, int):
            raise _error(f"{path}.init.low", "must be an integer")
        if isinstance(high, bool) or not isinstance(high, int):
            raise _error(f"{path}.init.high", "must be an integer")
        if low >= high:
            raise _error(f"{path}.init", "requires low < high")
        generated_dtype = torch.int64 if dtype_name == "bool" else dtype
        try:
            tensor = torch.randint(low, high, shape, dtype=generated_dtype)
        except RuntimeError as exc:
            raise _error(f"{path}.init", f"invalid randint range for {dtype_name}: {exc}") from exc
        if dtype_name == "bool":
            tensor = tensor.to(dtype=torch.bool)
    elif kind in {"explicit", "file"}:
        values: Any
        if kind == "explicit":
            if "values" not in init:
                raise _error(f"{path}.init.values", "is required")
            values = init["values"]
        else:
            raw_file = init.get("path")
            if not isinstance(raw_file, str) or not raw_file:
                raise _error(f"{path}.init.path", "must be a non-empty string")
            source = (config_dir / raw_file).resolve()
            audit_init["resolved_path"] = str(source)
            try:
                source_bytes = source.read_bytes()
                values = json.loads(source_bytes.decode("utf-8"))
            except FileNotFoundError as exc:
                raise _error(f"{path}.init.path", f"file does not exist: {source}") from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise _error(f"{path}.init.path", f"invalid JSON '{source}': {exc}") from exc
            audit_init["sha256"] = hashlib.sha256(source_bytes).hexdigest()
            if isinstance(values, dict) and set(values) == {"values"}:
                values = values["values"]
        try:
            tensor = torch.tensor(values, dtype=dtype)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise _error(f"{path}.init", f"cannot construct tensor from explicit data: {exc}") from exc
        if tensor.numel() != count:
            raise _error(
                f"{path}.init",
                f"explicit data has {tensor.numel()} values but shape requires {count}",
            )
        tensor = tensor.reshape(shape)
    else:
        start = init.get("start", 0)
        step = init.get("step", 1)
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise _error(f"{path}.init.start", "must be a number")
        if isinstance(step, bool) or not isinstance(step, (int, float)) or step == 0:
            raise _error(f"{path}.init.step", "must be a non-zero number")
        generation_dtype = dtype if dtype_name in _FLOAT_DTYPES else torch.int64
        tensor = (
            torch.arange(count, dtype=generation_dtype) * step + start
        ).to(dtype=dtype).reshape(shape)

    initializers[path] = {
        **audit_init,
        "name": name,
    }
    return tensor


def _input_value(
    value: Any,
    *,
    path: str,
    config_dir: Path,
    initializers: dict[str, Any],
) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [
            _input_value(item, path=f"{path}[{index}]", config_dir=config_dir, initializers=initializers)
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        raise _error(path, "unsupported input node")
    kind = value.get("kind")
    if kind == "tensor":
        return _tensor_from_spec(
            value,
            path=path,
            config_dir=config_dir,
            initializers=initializers,
        )
    if kind in {"tuple", "list"}:
        _only_fields(value, _CONTAINER_FIELDS, path)
        items = value.get("items")
        if not isinstance(items, list):
            raise _error(f"{path}.items", "must be an array")
        converted = [
            _input_value(
                item,
                path=f"{path}.items[{index}]",
                config_dir=config_dir,
                initializers=initializers,
            )
            for index, item in enumerate(items)
        ]
        return tuple(converted) if kind == "tuple" else converted
    if kind == "dict":
        _only_fields(value, _CONTAINER_FIELDS, path)
        items = _object(value.get("items"), f"{path}.items")
        return {
            key: _input_value(
                item,
                path=f"{path}.items.{key}",
                config_dir=config_dir,
                initializers=initializers,
            )
            for key, item in items.items()
        }
    if kind == "scalar":
        _only_fields(value, _SCALAR_FIELDS, path)
        scalar = value.get("value")
        if scalar is not None and not isinstance(scalar, (bool, int, float, str)):
            raise _error(f"{path}.value", "must be a JSON scalar or null")
        return scalar
    if kind is None:
        raise _error(path, "object input nodes require an explicit kind")
    raise _error(f"{path}.kind", "unsupported input kind")


def _describe_value(value: Any, path: str, initializers: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        result = {
            "kind": "tensor",
            "path": path,
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype).removeprefix("torch."),
        }
        if path in initializers:
            result["init"] = dict(initializers[path])
        return result
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "path": path,
            "items": [
                _describe_value(item, f"{path}[{index}]", initializers)
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "path": path,
            "items": [
                _describe_value(item, f"{path}[{index}]", initializers)
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "path": path,
            "items": {
                key: _describe_value(item, f"{path}.{key}", initializers)
                for key, item in value.items()
            },
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return {
            "kind": "python_scalar",
            "path": path,
            "python_type": "NoneType" if value is None else type(value).__name__,
            "value": value,
        }
    raise _error(path, f"unsupported runtime input value of type {type(value).__name__}")


def _input_signature(
    module: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    initializers: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        signature = inspect.signature(module.forward)
        bound = signature.bind(*args, **kwargs)
    except TypeError as exc:
        raise _error("inputs", f"do not match forward{inspect.signature(module.forward)}: {exc}") from exc
    positional_names: list[str] = []
    vararg_name: str | None = None
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            positional_names.append(parameter.name)
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            vararg_name = parameter.name
    described_args = []
    for index, value in enumerate(args):
        argument_name = (
            positional_names[index]
            if index < len(positional_names)
            else f"{vararg_name}[{index - len(positional_names)}]"
        )
        item = _describe_value(value, f"inputs.args[{index}]", initializers)
        item["forward_argument"] = argument_name
        described_args.append(item)
    described_kwargs = {}
    for name, value in kwargs.items():
        item = _describe_value(value, f"inputs.kwargs.{name}", initializers)
        item["forward_argument"] = name
        described_kwargs[name] = item
    return {
        "forward_signature": str(signature),
        "bound_arguments": list(bound.arguments),
        "args": described_args,
        "kwargs": described_kwargs,
    }


@dataclass(frozen=True)
class Workload:
    """A concrete module and static example-input tree ready for compilation."""

    module: Any
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    experiment_id: str = "torch_model"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    input_signature: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ValueError("Workload requires PyTorch") from exc
        if not isinstance(self.module, torch.nn.Module):
            raise ValueError("workload.module must be torch.nn.Module")
        if not isinstance(self.args, tuple):
            raise ValueError("workload.args must be a tuple")
        if not isinstance(self.kwargs, Mapping):
            raise ValueError("workload.kwargs must be a mapping")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("workload.provenance must be a mapping")
        if not isinstance(self.input_signature, Mapping):
            raise ValueError("workload.input_signature must be a mapping")
        if not isinstance(self.experiment_id, str) or not self.experiment_id:
            raise ValueError("workload.experiment_id must be a non-empty string")
        normalized_kwargs = dict(self.kwargs)
        derived_signature = _input_signature(
            self.module,
            self.args,
            normalized_kwargs,
            {},
        )
        signature = dict(self.input_signature) if self.input_signature else derived_signature
        object.__setattr__(self, "kwargs", normalized_kwargs)
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "input_signature", signature)

    @property
    def inputs(self) -> tuple[Any, ...]:
        """Compatibility alias for positional-only workload callers."""

        return self.args

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "module": {
                "class": f"{type(self.module).__module__}:{type(self.module).__qualname__}",
                "training": bool(self.module.training),
            },
            "input_signature": dict(self.input_signature),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class LoadedWorkloadConfig:
    """Validated JSON plus the constructed generic workload."""

    workload: Workload
    compile_options: Mapping[str, Any]
    runtime_options: Mapping[str, Any]
    source_path: Path
    source_sha256: str
    source_config: Mapping[str, Any]


def build_declarative_workload(
    model: Mapping[str, Any],
    inputs: Mapping[str, Any],
    *,
    seed: int = 0,
    experiment_id: str | None = None,
    config_dir: Path | None = None,
    source: str = "declarative",
) -> Workload:
    """Construct a module and static input tree from validated JSON-like data."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ValueError("workload construction requires PyTorch") from exc
    model = _object(model, "model")
    inputs = _object(inputs, "inputs")
    _only_fields(model, _MODEL_FIELDS, "model")
    _only_fields(inputs, _INPUT_FIELDS, "inputs")
    if "factory" not in model:
        raise _error("model.factory", "is required")
    factory_spec = model["factory"]
    constructor = _import_callable(factory_spec, "model.factory")
    constructor_kwargs = _object(model.get("kwargs", {}), "model.kwargs")
    dtype_name = model.get("dtype", "float32")
    dtype = _torch_dtype(torch, dtype_name, "model.dtype", floating_only=True)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise _error("seed", "must be a non-negative integer")
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        module = constructor(**constructor_kwargs)
    except Exception as exc:
        raise _error("model.kwargs", f"factory '{factory_spec}' failed: {exc}") from exc
    if not isinstance(module, torch.nn.Module):
        raise _error("model.factory", "must return torch.nn.Module")
    module = module.eval().to(dtype=dtype)

    raw_args = inputs.get("args", [])
    if not isinstance(raw_args, list):
        raise _error("inputs.args", "must be an array")
    raw_kwargs = _object(inputs.get("kwargs", {}), "inputs.kwargs")
    directory = (config_dir or Path.cwd()).resolve()
    initializers: dict[str, Any] = {}
    args = tuple(
        _input_value(
            value,
            path=f"inputs.args[{index}]",
            config_dir=directory,
            initializers=initializers,
        )
        for index, value in enumerate(raw_args)
    )
    kwargs = {
        name: _input_value(
            value,
            path=f"inputs.kwargs.{name}",
            config_dir=directory,
            initializers=initializers,
        )
        for name, value in raw_kwargs.items()
    }
    signature = _input_signature(module, args, kwargs, initializers)
    selected_id = experiment_id or str(factory_spec).split(":", 1)[-1].rsplit(".", 1)[-1]
    return Workload(
        module=module,
        args=args,
        kwargs=kwargs,
        experiment_id=selected_id,
        provenance={
            "construction": source,
            "seed": seed,
            "model_factory": factory_spec,
            "model_kwargs": constructor_kwargs,
            "model_dtype": dtype_name,
            "config_directory": str(directory),
        },
        input_signature=signature,
    )


def make_workload(
    module: Any,
    args: Sequence[Any],
    *,
    kwargs: Mapping[str, Any] | None = None,
    experiment_id: str = "torch_model",
    provenance: Mapping[str, Any] | None = None,
) -> Workload:
    """Wrap Python-built inputs in the same contract used by JSON workloads."""

    return Workload(
        module=module,
        args=tuple(args),
        kwargs=dict(kwargs or {}),
        experiment_id=experiment_id,
        provenance=dict(provenance or {}),
    )


def _option_section(
    value: Any,
    *,
    path: str,
    allowed: set[str],
    config_dir: Path,
) -> dict[str, Any]:
    result = _object(value, path)
    _only_fields(result, allowed, path)
    for name in set(result) & _PATH_OPTIONS:
        raw_path = result[name]
        if not isinstance(raw_path, str) or not raw_path:
            raise _error(f"{path}.{name}", "must be a non-empty path string")
        result[name] = str((config_dir / raw_path).resolve())
    return result


def load_workload_config(path: str | Path) -> LoadedWorkloadConfig:
    """Read, validate and construct one schema-v1 static workload."""

    source_path = Path(path).expanduser().resolve()
    try:
        source_bytes = source_path.read_bytes()
        payload = json.loads(source_bytes.decode("utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"workload config does not exist: {source_path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid workload config JSON '{source_path}': {exc}") from exc
    root = _object(payload, "config")
    _only_fields(root, _ROOT_FIELDS, "config")
    if root.get("schema_version") != 1:
        raise _error("schema_version", "must be 1")
    seed = root.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise _error("seed", "must be a non-negative integer")
    experiment_id = root.get("experiment_id")
    if experiment_id is not None and (not isinstance(experiment_id, str) or not experiment_id):
        raise _error("experiment_id", "must be a non-empty string")
    declarative = "model" in root or "inputs" in root
    factory_mode = "workload" in root
    if declarative == factory_mode:
        raise _error(
            "config",
            "declare exactly one of model+inputs or workload",
        )
    config_dir = source_path.parent
    if factory_mode:
        workload_spec = _object(root["workload"], "workload")
        _only_fields(workload_spec, _WORKLOAD_FIELDS, "workload")
        if "factory" not in workload_spec:
            raise _error("workload.factory", "is required")
        factory = _import_callable(workload_spec["factory"], "workload.factory")
        factory_kwargs = _object(workload_spec.get("kwargs", {}), "workload.kwargs")
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ValueError("workload construction requires PyTorch") from exc
        torch.manual_seed(seed)
        random.seed(seed)
        try:
            workload = factory(**factory_kwargs)
        except Exception as exc:
            raise _error("workload.kwargs", f"factory '{workload_spec['factory']}' failed: {exc}") from exc
        if not isinstance(workload, Workload):
            raise _error("workload.factory", "must return npu_ooo.frontend.Workload")
        provenance = {
            **dict(workload.provenance),
            "construction": "python_workload_factory",
            "seed": seed,
            "workload_factory": workload_spec["factory"],
            "workload_kwargs": factory_kwargs,
            "config_directory": str(config_dir),
        }
        workload = Workload(
            workload.module,
            workload.args,
            workload.kwargs,
            experiment_id=experiment_id or workload.experiment_id,
            provenance=provenance,
            input_signature=workload.input_signature,
        )
    else:
        if "model" not in root:
            raise _error("model", "is required")
        if "inputs" not in root:
            raise _error("inputs", "is required")
        workload = build_declarative_workload(
            root["model"],
            root["inputs"],
            seed=seed,
            experiment_id=experiment_id,
            config_dir=config_dir,
            source="json_declarative",
        )
    compile_options = _option_section(
        root.get("compile", {}),
        path="compile",
        allowed=_COMPILE_FIELDS,
        config_dir=config_dir,
    )
    runtime_options = _option_section(
        root.get("runtime", {}),
        path="runtime",
        allowed=_RUNTIME_FIELDS,
        config_dir=config_dir,
    )
    return LoadedWorkloadConfig(
        workload=workload,
        compile_options=compile_options,
        runtime_options=runtime_options,
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_config=root,
    )


__all__ = [
    "LoadedWorkloadConfig",
    "Workload",
    "build_declarative_workload",
    "load_workload_config",
    "make_workload",
]
