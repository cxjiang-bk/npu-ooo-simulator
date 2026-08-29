from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


ShapeValue = int | str


class SemanticOpType(str, Enum):
    """Model-level operation names retained through tile lowering."""

    MATMUL = "matmul"
    BATCHED_MATMUL = "batched_matmul"
    GEMV = "gemv"
    CONV2D = "conv2d"
    BATCH_NORM = "batch_norm"
    ELEMENTWISE = "elementwise"
    REDUCE = "reduce"
    SOFTMAX = "softmax"
    ATTENTION = "attention"
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    SWIGLU = "swiglu"
    EMBEDDING = "embedding"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    POOL = "pool"
    RESIDUAL_ADD = "residual_add"
    MOE_DISPATCH = "moe_dispatch"
    SLICE = "slice"
    CONCATENATE = "concatenate"
    KV_CACHE_UPDATE = "kv_cache_update"


def _validate_shape_value(value: ShapeValue, *, context: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a positive integer or symbol")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{context} must be positive")
        return
    if isinstance(value, str) and value and value.replace("_", "a").isalnum():
        return
    raise ValueError(f"{context} must be a positive integer or symbol")


def _resolve(value: ShapeValue, environment: Mapping[str, int], *, context: str) -> int:
    if isinstance(value, int):
        return value
    if value not in environment:
        raise ValueError(f"{context} references unknown shape symbol '{value}'")
    resolved = environment[value]
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved <= 0:
        raise ValueError(f"shape symbol '{value}' must resolve to a positive integer")
    return resolved


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[ShapeValue, ...]
    dtype: str = "fp16"
    layout: str = "dense"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.name:
            issues.append("tensor name must not be empty")
        # StableHLO permits zero-rank tensors.  They are used for scalar
        # constants/operands and must remain rank-0 through the compiler
        # boundary instead of being rewritten as a length-one vector.
        for index, value in enumerate(self.shape):
            try:
                _validate_shape_value(value, context=f"tensor '{self.name}' dim {index}")
            except ValueError as exc:
                issues.append(str(exc))
        if not self.dtype:
            issues.append(f"tensor '{self.name}' dtype must not be empty")
        return tuple(issues)

    def resolve(self, environment: Mapping[str, int]) -> "TensorSpec":
        issues = self.validate()
        if issues:
            raise ValueError("; ".join(issues))
        return TensorSpec(
            name=self.name,
            shape=tuple(
                _resolve(value, environment, context=f"tensor '{self.name}'")
                for value in self.shape
            ),
            dtype=self.dtype,
            layout=self.layout,
            attributes=dict(self.attributes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class DataEdge:
    producer: str
    consumer: str
    tensor: str

    def to_dict(self) -> dict[str, str]:
        return {
            "producer": self.producer,
            "consumer": self.consumer,
            "tensor": self.tensor,
        }


@dataclass(frozen=True)
class OperatorSpec:
    op_id: str
    op_type: str | SemanticOpType
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    iteration_dims: tuple[tuple[str, ShapeValue], ...] = ()
    reduction_dims: tuple[tuple[str, ShapeValue], ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def normalized_type(self) -> str:
        return self.op_type.value if isinstance(self.op_type, SemanticOpType) else str(self.op_type)

    def validate(self, tensors: Mapping[str, TensorSpec]) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.op_id:
            issues.append("operator id must not be empty")
        if not self.normalized_type:
            issues.append(f"operator '{self.op_id}' type must not be empty")
        if not self.inputs:
            issues.append(f"operator '{self.op_id}' must have an input or explicit source")
        if not self.outputs:
            issues.append(f"operator '{self.op_id}' must have an output")
        for name in (*self.inputs, *self.outputs):
            if name not in tensors:
                issues.append(f"operator '{self.op_id}' references unknown tensor '{name}'")
        for label, dimensions in (
            ("iteration", self.iteration_dims),
            ("reduction", self.reduction_dims),
        ):
            seen: set[str] = set()
            for dim, value in dimensions:
                if not dim:
                    issues.append(f"operator '{self.op_id}' has an empty {label} dimension")
                if dim in seen:
                    issues.append(f"operator '{self.op_id}' repeats {label} dimension '{dim}'")
                seen.add(dim)
                try:
                    _validate_shape_value(value, context=f"operator '{self.op_id}' dim '{dim}'")
                except ValueError as exc:
                    issues.append(str(exc))
        overlap = {dim for dim, _ in self.iteration_dims} & {
            dim for dim, _ in self.reduction_dims
        }
        for dim in sorted(overlap):
            issues.append(f"operator '{self.op_id}' uses dimension '{dim}' as both iteration and reduction")
        return tuple(issues)

    def resolve(self, environment: Mapping[str, int]) -> "OperatorSpec":
        return OperatorSpec(
            op_id=self.op_id,
            op_type=self.normalized_type,
            inputs=self.inputs,
            outputs=self.outputs,
            iteration_dims=tuple(
                (name, _resolve(value, environment, context=f"operator '{self.op_id}'"))
                for name, value in self.iteration_dims
            ),
            reduction_dims=tuple(
                (name, _resolve(value, environment, context=f"operator '{self.op_id}'"))
                for name, value in self.reduction_dims
            ),
            attributes=dict(self.attributes),
            provenance=dict(self.provenance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "op_type": self.normalized_type,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "iteration_dims": [[name, value] for name, value in self.iteration_dims],
            "reduction_dims": [[name, value] for name, value in self.reduction_dims],
            "attributes": dict(self.attributes),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class OperatorGraph:
    graph_id: str
    tensors: tuple[TensorSpec, ...]
    operators: tuple[OperatorSpec, ...]
    edges: tuple[DataEdge, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        tensors_by_name = {tensor.name: tensor for tensor in self.tensors}
        if len(tensors_by_name) != len(self.tensors):
            issues.append("tensor names must be unique")
        for tensor in self.tensors:
            issues.extend(tensor.validate())

        operators_by_id = {operator.op_id: operator for operator in self.operators}
        if len(operators_by_id) != len(self.operators):
            issues.append("operator ids must be unique")
        for operator in self.operators:
            issues.extend(operator.validate(tensors_by_name))

        for edge in self.edges:
            if edge.producer not in operators_by_id:
                issues.append(f"edge references unknown producer '{edge.producer}'")
            if edge.consumer not in operators_by_id:
                issues.append(f"edge references unknown consumer '{edge.consumer}'")
            if edge.tensor not in tensors_by_name:
                issues.append(f"edge references unknown tensor '{edge.tensor}'")
                continue
            producer = operators_by_id.get(edge.producer)
            consumer = operators_by_id.get(edge.consumer)
            if producer is not None and edge.tensor not in producer.outputs:
                issues.append(f"tensor '{edge.tensor}' is not an output of '{edge.producer}'")
            if consumer is not None and edge.tensor not in consumer.inputs:
                issues.append(f"tensor '{edge.tensor}' is not an input of '{edge.consumer}'")

        try:
            self.topological_order()
        except ValueError as exc:
            issues.append(str(exc))
        return tuple(issues)

    def topological_order(self) -> tuple[str, ...]:
        operator_ids = [operator.op_id for operator in self.operators]
        outgoing = {operator_id: set() for operator_id in operator_ids}
        indegree = {operator_id: 0 for operator_id in operator_ids}
        for edge in self.edges:
            if edge.producer not in outgoing or edge.consumer not in indegree:
                raise ValueError("cannot topologically order graph with unknown edge endpoint")
            if edge.consumer not in outgoing[edge.producer]:
                outgoing[edge.producer].add(edge.consumer)
                indegree[edge.consumer] += 1
        order_index = {operator_id: index for index, operator_id in enumerate(operator_ids)}
        ready = sorted(
            (operator_id for operator_id, degree in indegree.items() if degree == 0),
            key=order_index.__getitem__,
        )
        result: list[str] = []
        while ready:
            current = ready.pop(0)
            result.append(current)
            for successor in sorted(outgoing[current], key=order_index.__getitem__):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort(key=order_index.__getitem__)
        if len(result) != len(operator_ids):
            raise ValueError(f"operator graph '{self.graph_id}' contains a cycle")
        return tuple(result)

    def resolve(self, environment: Mapping[str, int]) -> "OperatorGraph":
        issues = self.validate()
        if issues:
            raise ValueError("; ".join(issues))
        resolved = OperatorGraph(
            graph_id=self.graph_id,
            tensors=tuple(tensor.resolve(environment) for tensor in self.tensors),
            operators=tuple(operator.resolve(environment) for operator in self.operators),
            edges=self.edges,
            attributes=dict(self.attributes),
        )
        resolved_issues = resolved.validate()
        if resolved_issues:
            raise ValueError("; ".join(resolved_issues))
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "operators": [operator.to_dict() for operator in self.operators],
            "edges": [edge.to_dict() for edge in self.edges],
            "attributes": dict(self.attributes),
        }
