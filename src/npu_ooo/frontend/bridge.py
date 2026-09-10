from __future__ import annotations

"""PyTorch export capture and source-graph provenance."""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from npu_ooo.ir import (
    DataEdge,
    ModelFamily,
    OperatorGraph,
    OperatorSpec,
    SemanticOpType,
    TensorSpec,
)


class FrontendImportError(ValueError):
    """A user-facing error at the framework-to-canonical-IR boundary."""


class FrontendKind(str, Enum):
    TORCH_EXPORT = "torch.export"
    STABLEHLO = "stablehlo"


def normalize_shape_environment(
    shape_environment: Mapping[str, int] | None,
) -> dict[str, int]:
    """Validate and normalize symbolic dimension bindings at the frontend boundary."""

    if shape_environment is None:
        return {}
    if not isinstance(shape_environment, Mapping):
        raise FrontendImportError("shape_environment must be a mapping of symbol to positive integer")
    normalized: dict[str, int] = {}
    for raw_name, raw_value in shape_environment.items():
        name = str(raw_name)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise FrontendImportError(
                f"shape_environment symbol '{name}' must be a valid identifier"
            )
        if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
            raise FrontendImportError(
                f"shape_environment value for '{name}' must be a positive integer"
            )
        normalized[name] = raw_value
    return normalized


@dataclass(frozen=True)
class FrontendImport:
    """A graph plus model metadata produced by a framework bridge."""

    graph: OperatorGraph
    model_id: str
    variant: str
    shape_environment: Mapping[str, int] = field(default_factory=dict)
    frontend: FrontendKind | str = FrontendKind.TORCH_EXPORT
    provenance: Mapping[str, Any] = field(default_factory=dict)
    family: ModelFamily | str = ModelFamily.SYNTHETIC

    def validate(self) -> tuple[str, ...]:
        issues = list(self.graph.validate())
        if not self.model_id:
            issues.append("frontend model_id must not be empty")
        if not self.variant:
            issues.append("frontend model variant must not be empty")
        try:
            normalize_shape_environment(self.shape_environment)
        except FrontendImportError as exc:
            issues.append(str(exc))
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        frontend = self.frontend.value if isinstance(self.frontend, FrontendKind) else str(self.frontend)
        family = self.family.value if isinstance(self.family, ModelFamily) else str(self.family)
        return {
            "frontend": frontend,
            "model_id": self.model_id,
            "variant": self.variant,
            "family": family,
            "shape_environment": dict(self.shape_environment),
            "provenance": dict(self.provenance),
            "graph": self.graph.to_dict(),
        }


class TorchExportAdapter:
    """Bridge a ``torch.export`` ExportedProgram to canonical OperatorGraph.

    This adapter deliberately uses duck typing for ExportedProgram/FX nodes,
    which keeps the core package importable without torch.  The initial
    supported semantic set is enough for 2mm, MLP and attention micrographs;
    unsupported nodes remain explicit in the graph and fail later at the
    lowering registry with a precise operator type.
    """

    kind = FrontendKind.TORCH_EXPORT

    @classmethod
    def capture_module(
        cls,
        module: Any,
        args: Sequence[Any] = (),
        *,
        kwargs: Mapping[str, Any] | None = None,
        dynamic_shapes: Mapping[str, Any] | None = None,
    ) -> Any:
        """Capture a module once so another exporter can consume the program."""

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise FrontendImportError(
                "TorchExportAdapter requires 'torch'; install a pinned PyTorch version "
                "before using the torch.export frontend"
            ) from exc
        try:
            export_kwargs: dict[str, Any] = {"kwargs": dict(kwargs or {})}
            if dynamic_shapes is not None:
                export_kwargs["dynamic_shapes"] = dynamic_shapes
            return torch.export.export(module, tuple(args), **export_kwargs)
        except Exception as exc:  # torch exposes several version-specific exception types
            raise FrontendImportError(f"torch.export failed: {exc}") from exc

    @classmethod
    def from_exported_program(
        cls,
        exported_program: Any,
        *,
        model_id: str = "torch_export_model",
        variant: str = "torch-export-v0",
        shape_environment: Mapping[str, int] | None = None,
    ) -> FrontendImport:
        graph_module = getattr(exported_program, "graph_module", None)
        if graph_module is None or not hasattr(graph_module, "graph"):
            raise FrontendImportError("expected a torch.export ExportedProgram with graph_module.graph")
        input_sources: dict[str, dict[str, str]] = {}
        graph_signature = getattr(exported_program, "graph_signature", None)
        for spec in getattr(graph_signature, "input_specs", ()):
            argument = getattr(spec, "arg", None)
            name = getattr(argument, "name", None)
            if not isinstance(name, str) or not name:
                continue
            kind = getattr(spec, "kind", None)
            kind_name = str(getattr(kind, "name", kind)).lower()
            if "parameter" in kind_name:
                source_kind = "parameter"
            elif "buffer" in kind_name:
                source_kind = "buffer"
            elif "constant" in kind_name:
                source_kind = "constant"
            else:
                source_kind = "input"
            input_sources[name] = {
                "source_kind": source_kind,
                "source_target": str(getattr(spec, "target", "") or ""),
            }
        normalized_environment = normalize_shape_environment(shape_environment)
        graph = _operator_graph_from_fx_graph(
            graph_module.graph,
            normalized_environment,
            input_sources=input_sources,
            graph_id=(
                "torch_export_graph"
                if model_id == "torch_export_model"
                else f"{model_id}.graph"
            ),
        )
        result = FrontendImport(
            graph=graph,
            model_id=model_id,
            variant=variant,
            shape_environment=normalized_environment,
            frontend=cls.kind,
            provenance={
                "source": "torch.export",
                "graph_module": type(graph_module).__name__,
                "exported_program": type(exported_program).__name__,
            },
            family=ModelFamily.SYNTHETIC,
        )
        issues = result.validate()
        if issues:
            raise FrontendImportError("invalid torch.export import: " + "; ".join(issues))
        return result


def _flatten_nodes(value: Any) -> list[Any]:
    if hasattr(value, "op") and hasattr(value, "name"):
        return [value]
    if isinstance(value, (tuple, list)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten_nodes(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_flatten_nodes(item))
        return result
    return []


def _target_name(target: Any) -> str:
    for attribute in ("_schema", "_overloadpacket"):
        value = getattr(target, attribute, None)
        name = getattr(value, "name", None)
        if isinstance(name, str) and name:
            return name
    name_method = getattr(target, "name", None)
    if callable(name_method):
        try:
            return str(name_method())
        except TypeError:
            pass
    return str(target)


def _semantic_target(target: Any) -> str:
    name = _target_name(target).lower()
    if any(
        token in name
        for token in (
            "aten.mm",
            "aten::mm",
            "aten.matmul",
            "aten::matmul",
            "aten.linear",
            "aten::linear",
        )
    ):
        return SemanticOpType.MATMUL.value
    if any(token in name for token in ("aten.bmm", "aten::bmm")):
        return SemanticOpType.BATCHED_MATMUL.value
    if any(token in name for token in ("rms_norm", "rmsnorm", "aten.rms_norm")):
        return SemanticOpType.RMSNORM.value
    if any(token in name for token in ("native_layer_norm", "layer_norm", "layernorm")):
        return SemanticOpType.LAYERNORM.value
    if any(token in name for token in ("softmax", "_safe_softmax")):
        return SemanticOpType.SOFTMAX.value
    if any(
        token in name
        for token in (
            "aten.sum",
            "aten::sum",
            "aten.mean",
            "aten::mean",
            "aten.amax",
            "aten::amax",
            "aten.max",
        )
    ):
        return SemanticOpType.REDUCE.value
    if any(
        token in name
        for token in (
            "aten.add",
            "aten::add",
            "aten.mul",
            "aten::mul",
            "aten.sub",
            "aten::sub",
            "aten.div",
            "aten::div",
            "aten.rsqrt",
            "aten::rsqrt",
            "aten.sqrt",
            "aten::sqrt",
            "aten.pow",
            "aten::pow",
        )
    ):
        return SemanticOpType.ELEMENTWISE.value
    if any(token in name for token in ("reshape", "view", "flatten")):
        return SemanticOpType.RESHAPE.value
    if any(token in name for token in ("transpose", "permute", "t.default")):
        return SemanticOpType.TRANSPOSE.value
    return name.replace("::", ".")


def _constant_metadata(value: Any) -> Any:
    """Return a JSON-safe representation of a non-node FX argument.

    Exported graphs contain Python scalars, tuples and lists in kwargs.  The
    frontend must retain these attributes for decomposition passes (epsilon,
    axes, transpose flags, etc.) without importing torch in the canonical IR.
    Tensor-like values are represented by a stable type marker instead of
    serializing storage or device-specific objects.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_constant_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _constant_metadata(item) for key, item in value.items()}
    if hasattr(value, "name") and hasattr(value, "op"):
        return None
    return {"type": type(value).__name__, "repr": str(value)}


def _shape_from_meta(node: Any) -> tuple[int | str, ...] | None:
    meta = getattr(node, "meta", {}) or {}
    value = meta.get("val", meta.get("tensor_meta"))
    shape = getattr(value, "shape", None)
    if shape is None and isinstance(value, (tuple, list)) and value:
        shape = getattr(value[0], "shape", None)
    if shape is None:
        return None
    result: list[int | str] = []
    for dimension in tuple(shape):
        if isinstance(dimension, bool):
            return None
        try:
            integer = int(dimension)
        except (TypeError, ValueError):
            text = str(dimension)
            if not text:
                return None
            result.append(text)
        else:
            if integer <= 0:
                return None
            result.append(integer)
    return tuple(result)


def _dtype_from_meta(node: Any) -> str:
    value = (getattr(node, "meta", {}) or {}).get("val")
    dtype = getattr(value, "dtype", None)
    text = str(dtype) if dtype is not None else "fp16"
    return text.removeprefix("torch.")


def _dim_argument(node: Any, input_rank: int) -> tuple[int, ...]:
    kwargs = getattr(node, "kwargs", {}) or {}
    dim = kwargs.get("dim")
    args = getattr(node, "args", ())
    if dim is None and len(args) > 1 and _semantic_target(getattr(node, "target", None)) in {
        SemanticOpType.REDUCE.value,
        SemanticOpType.SOFTMAX.value,
    }:
        dim = args[1]
    if dim is None:
        return tuple(range(input_rank))
    if isinstance(dim, int):
        dimensions = (dim,)
    elif isinstance(dim, (tuple, list)):
        dimensions = tuple(int(item) for item in dim)
    else:
        raise FrontendImportError(f"node '{node.name}' has a non-constant reduction dimension")
    return tuple(item if item >= 0 else input_rank + item for item in dimensions)


def _operator_graph_from_fx_graph(
    fx_graph: Any,
    shape_environment: Mapping[str, int],
    *,
    input_sources: Mapping[str, Mapping[str, str]] | None = None,
    graph_id: str = "torch_export_graph",
) -> OperatorGraph:
    nodes = list(fx_graph.nodes)
    tensors: dict[str, TensorSpec] = {}
    operators: list[OperatorSpec] = []
    produced_by: dict[str, str] = {}
    graph_inputs: list[str] = []
    graph_outputs: list[str] = []

    for node in nodes:
        name = str(node.name)
        if node.op in {"placeholder", "get_attr"}:
            shape = _shape_from_meta(node)
            if shape is None:
                raise FrontendImportError(
                    f"node '{name}' has no tensor shape metadata; run torch.export with shape propagation"
                )
            # Inference modules may expose bookkeeping scalars (for example
            # BatchNorm's num_batches_tracked) as unused export placeholders.
            # They are not part of the dataflow and cannot be represented as a
            # dense tensor tile, so omit only this provably dead zero-rank case.
            if not shape and not getattr(node, "users", {}):
                continue
            source = dict((input_sources or {}).get(name, {}))
            source_kind = source.get(
                "source_kind",
                "input" if node.op == "placeholder" else "parameter",
            )
            tensors[name] = TensorSpec(
                name,
                shape,
                _dtype_from_meta(node),
                attributes={
                    "source_node": name,
                    "source_kind": source_kind,
                    "source_target": source.get("source_target", ""),
                    "target": str(getattr(node, "target", name)),
                },
            )
            if node.op == "placeholder":
                graph_inputs.append(name)

    for node in nodes:
        if node.op == "output":
            graph_outputs.extend(str(item.name) for item in _flatten_nodes(getattr(node, "args", ())))
            continue
        if node.op not in {"call_function", "call_method", "call_module"}:
            continue
        name = str(node.name)
        input_nodes = _flatten_nodes(getattr(node, "args", ())) + _flatten_nodes(getattr(node, "kwargs", {}))
        input_names = tuple(dict.fromkeys(str(item.name) for item in input_nodes))
        if not input_names:
            raise FrontendImportError(f"node '{name}' has no tensor inputs")
        input_shapes = [tensors[item].shape for item in input_names if item in tensors]
        if not input_shapes:
            raise FrontendImportError(f"node '{name}' references values without shape metadata")
        output_shape = _shape_from_meta(node) or input_shapes[0]
        output_dtype = _dtype_from_meta(node)
        target = getattr(node, "target", None)
        target_name = _target_name(target)
        tensors[name] = TensorSpec(
            name,
            output_shape,
            output_dtype,
            attributes={
                "source_node": name,
                "source_kind": "activation",
                "frontend_target": target_name,
            },
        )
        op_type = _semantic_target(target)
        is_linear_target = "linear" in target_name.lower()
        if (
            op_type == SemanticOpType.MATMUL.value
            and not is_linear_target
            and len(input_shapes) >= 2
            and (len(input_shapes[0]) > 2 or len(input_shapes[1]) > 2)
        ):
            op_type = SemanticOpType.BATCHED_MATMUL.value
        rank = len(output_shape)
        iteration_dims: tuple[tuple[str, Any], ...]
        reduction_dims: tuple[tuple[str, Any], ...]
        if op_type in {SemanticOpType.MATMUL.value, SemanticOpType.BATCHED_MATMUL.value}:
            if len(input_shapes[0]) < 2 or len(input_shapes[1]) < 2:
                raise FrontendImportError(f"matmul node '{name}' requires rank >= 2 operands")
            if op_type == SemanticOpType.BATCHED_MATMUL.value:
                rhs_broadcast_batch = (
                    len(input_shapes[1]) == 2 and len(input_shapes[0]) > 2
                )
                if not rhs_broadcast_batch and input_shapes[0][:-2] != input_shapes[1][:-2]:
                    raise FrontendImportError(
                        f"batched matmul node '{name}' uses broadcast batch dimensions; "
                        "batch broadcasting is not implemented"
                    )
                iteration_dims = tuple(
                    (f"B{axis}", extent)
                    for axis, extent in enumerate(output_shape[:-2])
                ) + (("M", output_shape[-2]), ("N", output_shape[-1]))
            elif is_linear_target and len(output_shape) > 2:
                iteration_dims = tuple(
                    (f"B{axis}", extent)
                    for axis, extent in enumerate(output_shape[:-2])
                ) + (("M", output_shape[-2]), ("N", output_shape[-1]))
            else:
                iteration_dims = (("M", output_shape[-2]), ("N", output_shape[-1]))
            reduction_dims = (("K", input_shapes[0][-1]),)
        elif op_type == SemanticOpType.REDUCE.value:
            dims = _dim_argument(node, len(input_shapes[0]))
            reduction_dims = tuple((f"d{axis}", input_shapes[0][axis]) for axis in dims)
            iteration_dims = tuple(
                (f"d{axis}", input_shapes[0][axis])
                for axis in range(len(input_shapes[0]))
                if axis not in dims
            )
        elif op_type in {SemanticOpType.RMSNORM.value, SemanticOpType.LAYERNORM.value, SemanticOpType.SOFTMAX.value}:
            if not input_shapes[0]:
                raise FrontendImportError(f"normalized node '{name}' requires a ranked tensor input")
            if op_type == SemanticOpType.SOFTMAX.value:
                axes = _dim_argument(node, len(input_shapes[0]))
            elif op_type == SemanticOpType.LAYERNORM.value:
                node_args = getattr(node, "args", ()) or ()
                node_kwargs = getattr(node, "kwargs", {}) or {}
                normalized_shape = node_kwargs.get(
                    "normalized_shape",
                    node_args[1] if len(node_args) > 1 else (input_shapes[0][-1],),
                )
                normalized_rank = (
                    len(normalized_shape)
                    if isinstance(normalized_shape, (tuple, list))
                    else 1
                )
                axes = tuple(range(len(input_shapes[0]) - normalized_rank, len(input_shapes[0])))
            else:
                axis = getattr(node, "kwargs", {}).get("dim")
                if axis is None:
                    axis = getattr(node, "kwargs", {}).get("axis")
                if axis is None:
                    axis = len(input_shapes[0]) - 1
                axes = (
                    tuple(int(item) for item in axis)
                    if isinstance(axis, (tuple, list))
                    else (int(axis),)
                )
            axes = tuple(item if item >= 0 else len(input_shapes[0]) + item for item in axes)
            reduction_dims = tuple((f"d{axis}", input_shapes[0][axis]) for axis in axes)
            iteration_dims = tuple(
                (f"d{axis}", input_shapes[0][axis])
                for axis in range(len(input_shapes[0]))
                if axis not in axes
            )
        else:
            iteration_dims = tuple((f"d{axis}", value) for axis, value in enumerate(output_shape))
            reduction_dims = ()
        semantic_attributes: dict[str, Any] = {}
        node_args = getattr(node, "args", ()) or ()
        node_kwargs = getattr(node, "kwargs", {}) or {}
        if op_type == SemanticOpType.LAYERNORM.value:
            normalized_shape = node_kwargs.get(
                "normalized_shape",
                node_args[1] if len(node_args) > 1 else [input_shapes[0][-1]],
            )
            epsilon = node_kwargs.get(
                "eps",
                node_args[4] if len(node_args) > 4 else 1e-5,
            )
            semantic_attributes.update(
                {
                    "normalized_shape": _constant_metadata(normalized_shape),
                    "epsilon": float(epsilon),
                    "affine": len(input_names) == 3,
                }
            )
        elif op_type == SemanticOpType.BATCHED_MATMUL.value:
            semantic_attributes["rhs_broadcast_batch"] = (
                len(input_shapes[1]) == 2 and len(input_shapes[0]) > 2
            )
        elif op_type == SemanticOpType.SOFTMAX.value:
            semantic_attributes["axes"] = list(_dim_argument(node, len(input_shapes[0])))
        elif op_type == SemanticOpType.TRANSPOSE.value:
            normalized_target = target_name.lower().replace("::", ".")
            if "permute" in normalized_target:
                permutation = node_args[1] if len(node_args) > 1 else None
                if not isinstance(permutation, (tuple, list)) or not all(
                    isinstance(item, int) for item in permutation
                ):
                    raise FrontendImportError(
                        f"permute node '{name}' requires a constant permutation"
                    )
                normalized = tuple(
                    item if item >= 0 else len(input_shapes[0]) + item
                    for item in permutation
                )
                if sorted(normalized) != list(range(len(input_shapes[0]))):
                    raise FrontendImportError(
                        f"permute node '{name}' has invalid permutation {normalized}"
                    )
                semantic_attributes["transpose_dims"] = list(normalized)
            else:
                if len(node_args) < 3 or not all(
                    isinstance(item, int) for item in node_args[1:3]
                ):
                    raise FrontendImportError(
                        f"transpose node '{name}' requires constant dimensions"
                    )
                transpose_dims = tuple(
                    item if item >= 0 else len(input_shapes[0]) + item
                    for item in node_args[1:3]
                )
                semantic_attributes["transpose_dims"] = list(transpose_dims)
        operators.append(
            OperatorSpec(
                op_id=name,
                op_type=op_type,
                inputs=input_names,
                outputs=(name,),
                iteration_dims=iteration_dims,
                reduction_dims=reduction_dims,
                attributes={
                    "frontend_target": target_name,
                    "frontend_node_op": str(getattr(node, "op", "")),
                    "input_occurrences": [str(item.name) for item in input_nodes],
                    "constant_args": {
                        "args": _constant_metadata(getattr(node, "args", ())),
                        "kwargs": _constant_metadata(getattr(node, "kwargs", {})),
                    },
                    "bias_input": (
                        input_names[2]
                        if op_type == SemanticOpType.MATMUL.value
                        and "linear" in target_name.lower()
                        and len(input_names) >= 3
                        else None
                    ),
                    **semantic_attributes,
                },
                provenance={
                    "frontend": FrontendKind.TORCH_EXPORT.value,
                    "source_node": name,
                    "source_target": target_name,
                },
            )
        )
        produced_by[name] = name

    edges: list[DataEdge] = []
    edge_keys: set[tuple[str, str, str]] = set()
    for operator in operators:
        for input_name in operator.inputs:
            producer = produced_by.get(input_name)
            if producer is not None and producer != operator.op_id:
                edge = DataEdge(producer, operator.op_id, input_name)
                if (edge.producer, edge.consumer, edge.tensor) not in edge_keys:
                    edges.append(edge)
                    edge_keys.add((edge.producer, edge.consumer, edge.tensor))
    if not graph_outputs:
        consumed = {tensor for operator in operators for tensor in operator.inputs}
        graph_outputs = [
            operator.outputs[0]
            for operator in operators
            if operator.outputs and operator.outputs[0] not in consumed
        ]
    graph = OperatorGraph(
        graph_id=graph_id,
        tensors=tuple(tensors.values()),
        operators=tuple(operators),
        edges=tuple(edges),
        attributes={
            "frontend": FrontendKind.TORCH_EXPORT.value,
            "shape_environment": dict(shape_environment),
            "graph_inputs": graph_inputs,
            "graph_outputs": graph_outputs,
            "node_count": len(nodes),
        },
    )
    issues = graph.validate()
    if issues:
        raise FrontendImportError("torch.export graph normalization failed: " + "; ".join(issues))
    return graph
