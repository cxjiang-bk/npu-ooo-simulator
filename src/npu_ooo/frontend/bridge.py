from __future__ import annotations

"""Framework bridge implementations.

The compiler consumes ``FrontendImport.graph`` and does not depend on FX,
StableHLO, or an individual framework after this boundary.  StableHLO input
will use the same result type in a later adapter; keeping the boundary here
prevents framework-specific node names from leaking into scheduling code.
"""

from dataclasses import dataclass, field
from enum import Enum
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
    CANONICAL = "canonical"
    JSON = "json"
    TORCH_EXPORT = "torch.export"
    STABLEHLO = "stablehlo"


@dataclass(frozen=True)
class FrontendImport:
    """A graph plus model metadata produced by a framework bridge."""

    graph: OperatorGraph
    model_id: str
    variant: str
    shape_environment: Mapping[str, int] = field(default_factory=dict)
    frontend: FrontendKind | str = FrontendKind.CANONICAL
    provenance: Mapping[str, Any] = field(default_factory=dict)
    family: ModelFamily | str = ModelFamily.SYNTHETIC

    def validate(self) -> tuple[str, ...]:
        issues = list(self.graph.validate())
        if not self.model_id:
            issues.append("frontend model_id must not be empty")
        if not self.variant:
            issues.append("frontend model variant must not be empty")
        if not isinstance(self.shape_environment, Mapping):
            issues.append("frontend shape_environment must be a mapping")
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


def _operator_from_dict(payload: Mapping[str, Any]) -> OperatorSpec:
    return OperatorSpec(
        op_id=str(payload["op_id"]),
        op_type=str(payload["op_type"]),
        inputs=tuple(str(item) for item in payload.get("inputs", ())),
        outputs=tuple(str(item) for item in payload.get("outputs", ())),
        iteration_dims=tuple(
            (str(item[0]), item[1]) for item in payload.get("iteration_dims", ())
        ),
        reduction_dims=tuple(
            (str(item[0]), item[1]) for item in payload.get("reduction_dims", ())
        ),
        attributes=dict(payload.get("attributes", {})),
        provenance=dict(payload.get("provenance", {})),
    )


def operator_graph_from_dict(payload: Mapping[str, Any]) -> OperatorGraph:
    """Parse the stable canonical graph JSON format emitted by this project."""

    try:
        graph = OperatorGraph(
            graph_id=str(payload["graph_id"]),
            tensors=tuple(
                TensorSpec(
                    name=str(item["name"]),
                    shape=tuple(item.get("shape", ())),
                    dtype=str(item.get("dtype", "fp16")),
                    layout=str(item.get("layout", "dense")),
                    attributes=dict(item.get("attributes", {})),
                )
                for item in payload.get("tensors", ())
            ),
            operators=tuple(_operator_from_dict(item) for item in payload.get("operators", ())),
            edges=tuple(
                DataEdge(
                    producer=str(item["producer"]),
                    consumer=str(item["consumer"]),
                    tensor=str(item["tensor"]),
                )
                for item in payload.get("edges", ())
            ),
            attributes=dict(payload.get("attributes", {})),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FrontendImportError(f"invalid canonical operator graph JSON: {exc}") from exc
    issues = graph.validate()
    if issues:
        raise FrontendImportError("invalid canonical operator graph: " + "; ".join(issues))
    return graph


class JsonGraphAdapter:
    """Import an already normalized graph without requiring framework packages."""

    kind = FrontendKind.JSON

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        model_id: str | None = None,
        variant: str = "imported-v0",
        shape_environment: Mapping[str, int] | None = None,
        family: ModelFamily | str = ModelFamily.SYNTHETIC,
    ) -> FrontendImport:
        graph_payload = payload.get("graph", payload)
        graph = operator_graph_from_dict(graph_payload)
        result = FrontendImport(
            graph=graph,
            model_id=model_id or str(payload.get("model_id", graph.graph_id)),
            variant=str(payload.get("variant", variant)),
            shape_environment=dict(shape_environment or payload.get("shape_environment", {})),
            frontend=cls.kind,
            provenance={"source": "canonical-json"},
            family=family,
        )
        issues = result.validate()
        if issues:
            raise FrontendImportError("invalid frontend import: " + "; ".join(issues))
        return result


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
    def export_module(
        cls,
        module: Any,
        args: Sequence[Any] = (),
        *,
        kwargs: Mapping[str, Any] | None = None,
        dynamic_shapes: Mapping[str, Any] | None = None,
    ) -> FrontendImport:
        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise FrontendImportError(
                "TorchExportAdapter requires 'torch'; install a pinned PyTorch version "
                "before using the torch.export frontend"
            ) from exc
        try:
            exported = torch.export.export(
                module,
                tuple(args),
                kwargs=dict(kwargs or {}),
                dynamic_shapes=dynamic_shapes,
            )
        except Exception as exc:  # torch exposes several version-specific exception types
            raise FrontendImportError(f"torch.export failed: {exc}") from exc
        return cls.from_exported_program(exported)

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
        graph = _operator_graph_from_fx_graph(graph_module.graph, shape_environment or {})
        result = FrontendImport(
            graph=graph,
            model_id=model_id,
            variant=variant,
            shape_environment=dict(shape_environment or {}),
            frontend=cls.kind,
            provenance={"source": "torch.export", "graph_module": type(graph_module).__name__},
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
    if any(token in name for token in ("aten.mm", "aten::mm", "aten.matmul", "aten::matmul")):
        return SemanticOpType.MATMUL.value
    if any(token in name for token in ("aten.bmm", "aten::bmm")):
        return SemanticOpType.BATCHED_MATMUL.value
    if any(token in name for token in ("softmax", "_safe_softmax")):
        return SemanticOpType.SOFTMAX.value
    if any(token in name for token in ("aten.sum", "aten::sum", "aten.amax", "aten::amax", "aten.max")):
        return SemanticOpType.REDUCE.value
    if any(token in name for token in ("aten.add", "aten::add", "aten.mul", "aten::mul", "aten.sub", "aten::sub", "aten.div", "aten::div")):
        return SemanticOpType.ELEMENTWISE.value
    if any(token in name for token in ("reshape", "view", "flatten")):
        return SemanticOpType.RESHAPE.value
    if any(token in name for token in ("transpose", "permute", "t.default")):
        return SemanticOpType.TRANSPOSE.value
    return name.replace("::", ".")


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
    if dim is None and len(args) > 1 and _semantic_target(getattr(node, "target", None)) == SemanticOpType.REDUCE.value:
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


def _operator_graph_from_fx_graph(fx_graph: Any, shape_environment: Mapping[str, int]) -> OperatorGraph:
    nodes = list(fx_graph.nodes)
    tensors: dict[str, TensorSpec] = {}
    operators: list[OperatorSpec] = []
    produced_by: dict[str, str] = {}

    for node in nodes:
        name = str(node.name)
        if node.op in {"placeholder", "get_attr"}:
            shape = _shape_from_meta(node)
            if shape is None:
                raise FrontendImportError(
                    f"node '{name}' has no tensor shape metadata; run torch.export with shape propagation"
                )
            tensors[name] = TensorSpec(name, shape, _dtype_from_meta(node), attributes={"source_node": name})

    for node in nodes:
        if node.op != "call_function":
            continue
        name = str(node.name)
        input_nodes = _flatten_nodes(getattr(node, "args", ())) + _flatten_nodes(getattr(node, "kwargs", {}))
        input_names = tuple(str(item.name) for item in input_nodes)
        if not input_names:
            raise FrontendImportError(f"node '{name}' has no tensor inputs")
        input_shapes = [tensors[item].shape for item in input_names if item in tensors]
        if not input_shapes:
            raise FrontendImportError(f"node '{name}' references values without shape metadata")
        output_shape = _shape_from_meta(node) or input_shapes[0]
        output_dtype = _dtype_from_meta(node)
        tensors[name] = TensorSpec(name, output_shape, output_dtype, attributes={"source_node": name})
        op_type = _semantic_target(getattr(node, "target", None))
        rank = len(output_shape)
        iteration_dims: tuple[tuple[str, Any], ...]
        reduction_dims: tuple[tuple[str, Any], ...]
        if op_type in {SemanticOpType.MATMUL.value, SemanticOpType.BATCHED_MATMUL.value}:
            if len(input_shapes[0]) < 2 or len(input_shapes[1]) < 2:
                raise FrontendImportError(f"matmul node '{name}' requires rank >= 2 operands")
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
        else:
            iteration_dims = tuple((f"d{axis}", value) for axis, value in enumerate(output_shape))
            reduction_dims = ()
        operators.append(
            OperatorSpec(
                op_id=name,
                op_type=op_type,
                inputs=input_names,
                outputs=(name,),
                iteration_dims=iteration_dims,
                reduction_dims=reduction_dims,
                attributes={"frontend_target": _target_name(getattr(node, "target", None))},
                provenance={"frontend": FrontendKind.TORCH_EXPORT.value, "source_node": name},
            )
        )
        produced_by[name] = name

    edges: list[DataEdge] = []
    for operator in operators:
        for input_name in operator.inputs:
            producer = produced_by.get(input_name)
            if producer is not None and producer != operator.op_id:
                edges.append(DataEdge(producer, operator.op_id, input_name))
    graph = OperatorGraph(
        graph_id="torch_export_graph",
        tensors=tuple(tensors.values()),
        operators=tuple(operators),
        edges=tuple(edges),
        attributes={"frontend": FrontendKind.TORCH_EXPORT.value, "shape_environment": dict(shape_environment)},
    )
    issues = graph.validate()
    if issues:
        raise FrontendImportError("torch.export graph normalization failed: " + "; ".join(issues))
    return graph


def import_operator_graph(graph: OperatorGraph, *, model_id: str | None = None, variant: str = "canonical-v0") -> FrontendImport:
    """Wrap an existing canonical graph so all frontends share one result type."""

    issues = graph.validate()
    if issues:
        raise FrontendImportError("invalid canonical graph: " + "; ".join(issues))
    return FrontendImport(
        graph=graph,
        model_id=model_id or graph.graph_id,
        variant=variant,
        frontend=FrontendKind.CANONICAL,
        provenance={"source": "canonical-operator-graph"},
    )
