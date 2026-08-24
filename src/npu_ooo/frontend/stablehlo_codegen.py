from __future__ import annotations

"""Generate a dependency-light StableHLO textual module from frontend IR.

This is deliberately a small graph-level emitter, not a replacement for the
XLA/StableHLO compiler.  It emits standard StableHLO-shaped primitives for the
operator set supported by this repository and preserves the source graph in
the module provenance.  The emitted text is then consumed by
``StableHLOAdapter`` just like an external StableHLO module.
"""

from dataclasses import dataclass, field
import re
from typing import Any

from npu_ooo.ir import OperatorGraph, OperatorSpec, TensorSpec

from .bridge import FrontendImport, FrontendImportError, FrontendKind


@dataclass(frozen=True)
class StableHLOModule:
    """Inspectable StableHLO assembly plus source-to-module provenance."""

    text: str
    model_id: str
    variant: str = "stablehlo-generated-v0"
    source_frontend: str = FrontendKind.TORCH_EXPORT.value
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.text.strip():
            issues.append("generated StableHLO module must not be empty")
        if "module" not in self.text or "func.func" not in self.text:
            issues.append("generated StableHLO module must contain module and func.func")
        if not self.model_id:
            issues.append("generated StableHLO model_id must not be empty")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model_id": self.model_id,
            "variant": self.variant,
            "source_frontend": self.source_frontend,
            "provenance": dict(self.provenance),
        }


def _value(name: str) -> str:
    return f"%{name}"


def _dtype(dtype: str) -> str:
    normalized = str(dtype).removeprefix("torch.").lower()
    return {
        "float16": "f16",
        "half": "f16",
        "float32": "f32",
        "float64": "f64",
        "bfloat16": "bf16",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "bool": "i1",
    }.get(normalized, normalized)


def _tensor_type(tensor: TensorSpec, *, shape: tuple[int | str, ...] | None = None) -> str:
    dimensions = tuple(tensor.shape if shape is None else shape)
    if dimensions:
        return "tensor<" + "x".join((*[str(item) for item in dimensions], _dtype(tensor.dtype))) + ">"
    return f"tensor<{_dtype(tensor.dtype)}>"


def _shape_with_reduction(shape: tuple[int | str, ...], axes: tuple[int, ...]) -> tuple[int | str, ...]:
    return tuple(1 if axis in axes else extent for axis, extent in enumerate(shape))


def _axis_from_name(name: str, rank: int) -> int:
    match = re.fullmatch(r"d(\d+)", str(name))
    if match is None:
        raise FrontendImportError(f"cannot map StableHLO reduction dimension '{name}' to a tensor axis")
    axis = int(match.group(1))
    if axis < 0 or axis >= rank:
        raise FrontendImportError(f"reduction axis {axis} is outside rank {rank}")
    return axis


def _scalar_constants(operator: OperatorSpec) -> list[Any]:
    payload = operator.attributes.get("constant_args", {})
    args = payload.get("args", ()) if isinstance(payload, dict) else ()
    values = args if isinstance(args, (tuple, list)) else (args,)
    result: list[Any] = []
    for value in values:
        if isinstance(value, bool) or isinstance(value, (int, float)):
            result.append(value)
    return result


def _target(operator: OperatorSpec) -> str:
    return str(operator.attributes.get("frontend_target", "")).lower().replace("::", ".")


def _elementwise_target(operator: OperatorSpec) -> str:
    target = _target(operator)
    mapping = (
        ("mul", "multiply"),
        ("add", "add"),
        ("sub", "subtract"),
        ("div", "divide"),
        ("rsqrt", "rsqrt"),
        ("sqrt", "sqrt"),
        ("pow", "power"),
        ("exp", "exponential"),
        ("maximum", "maximum"),
        ("minimum", "minimum"),
    )
    for token, stable_name in mapping:
        if token in target:
            return stable_name
    if operator.normalized_type == "residual_add":
        return "add"
    raise FrontendImportError(
        f"cannot lower elementwise operator '{operator.op_id}' target '{target}' to StableHLO"
    )


class StableHLOGenerator:
    """Emit StableHLO text from a ``FrontendImport`` graph."""

    def __init__(self, *, variant: str = "stablehlo-generated-v0") -> None:
        self.variant = variant

    def generate(self, imported: FrontendImport) -> StableHLOModule:
        issues = imported.validate()
        if issues:
            raise FrontendImportError("cannot generate StableHLO from invalid import: " + "; ".join(issues))
        graph = imported.graph
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        graph_inputs = tuple(str(item) for item in graph.attributes.get("graph_inputs", ()))
        graph_outputs = tuple(str(item) for item in graph.attributes.get("graph_outputs", ()))
        if not graph_inputs:
            graph_inputs = tuple(tensor.name for tensor in graph.tensors if tensor.attributes.get("source_kind") in {"input", "parameter", "buffer"})
        if not graph_outputs:
            graph_outputs = self._infer_outputs(graph)
        if not graph_inputs or not graph_outputs:
            raise FrontendImportError("StableHLO generation requires graph inputs and outputs")
        if len(graph_outputs) != 1:
            raise FrontendImportError(
                "StableHLO generation currently supports one graph output; "
                "tuple/multi-result lowering is not implemented"
            )

        lines = ["module {", "  func.func @main("]
        arguments = []
        for name in graph_inputs:
            try:
                arguments.append(f"    {_value(name)}: {_tensor_type(tensors[name])}")
            except KeyError as exc:
                raise FrontendImportError(f"StableHLO graph input '{name}' has no TensorSpec") from exc
        lines.append(",\n".join(arguments))
        return_type = ", ".join(_tensor_type(tensors[name]) for name in graph_outputs)
        lines.append(f") -> {return_type} {{")
        emitted: list[str] = []
        for operator_id in graph.topological_order():
            operator = next(item for item in graph.operators if item.op_id == operator_id)
            emitted.extend(self._emit_operator(operator, tensors))
        lines.extend(f"    {line}" for line in emitted)
        return_values = ", ".join(_value(name) for name in graph_outputs)
        lines.append(f"    return {return_values} : {return_type}")
        lines.extend(("  }", "}"))
        module = StableHLOModule(
            text="\n".join(lines) + "\n",
            model_id=imported.model_id,
            variant=self.variant,
            source_frontend=(
                imported.frontend.value
                if hasattr(imported.frontend, "value")
                else str(imported.frontend)
            ),
            provenance={
                "source": "frontend-import",
                "source_graph_id": graph.graph_id,
                "source_frontend": (
                    imported.frontend.value
                    if hasattr(imported.frontend, "value")
                    else str(imported.frontend)
                ),
                "generator": "stablehlo-graph-v0",
                "operator_count": len(graph.operators),
            },
        )
        issues = module.validate()
        if issues:
            raise FrontendImportError("generated StableHLO is invalid: " + "; ".join(issues))
        return module

    @staticmethod
    def _infer_outputs(graph: OperatorGraph) -> tuple[str, ...]:
        consumed = {tensor for operator in graph.operators for tensor in operator.inputs}
        return tuple(
            operator.outputs[0]
            for operator in graph.operators
            if operator.outputs and operator.outputs[0] not in consumed
        )

    def _emit_constant(self, name: str, value: Any, dtype: str) -> str:
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, float):
            rendered = repr(value)
        else:
            rendered = str(value)
        return f"{_value(name)} = stablehlo.constant dense<{rendered}> : tensor<{_dtype(dtype)}>"

    def _emit_operator(self, operator: OperatorSpec, tensors: dict[str, TensorSpec]) -> list[str]:
        if len(operator.outputs) != 1:
            raise FrontendImportError(f"StableHLO generation requires one output for '{operator.op_id}'")
        output_name = operator.outputs[0]
        output = tensors[output_name]
        if operator.normalized_type in {"matmul", "batched_matmul", "gemv"}:
            return self._emit_matmul(operator, tensors, output)
        if operator.normalized_type == "reduce":
            return self._emit_reduce(operator, tensors, output)
        if operator.normalized_type == "softmax":
            return self._emit_softmax(operator, tensors, output)
        if operator.normalized_type == "rmsnorm":
            return self._emit_rmsnorm(operator, tensors, output)
        if operator.normalized_type == "layernorm":
            return self._emit_layernorm(operator, tensors, output)
        if operator.normalized_type in {"elementwise", "residual_add"}:
            return self._emit_elementwise(operator, tensors, output)
        if operator.normalized_type == "transpose":
            return self._emit_transpose(operator, tensors, output)
        raise FrontendImportError(
            f"StableHLO generation does not support semantic operator '{operator.normalized_type}'"
        )

    def _emit_matmul(self, operator: OperatorSpec, tensors: dict[str, TensorSpec], output: TensorSpec) -> list[str]:
        if len(operator.inputs) < 2:
            raise FrontendImportError(f"Matmul '{operator.op_id}' requires two inputs")
        lhs = tensors[operator.inputs[0]]
        rhs = tensors[operator.inputs[1]]
        lhs_rank = len(lhs.shape)
        rhs_rank = len(rhs.shape)
        if lhs_rank < 2 or rhs_rank < 2:
            raise FrontendImportError(f"Matmul '{operator.op_id}' requires rank >= 2 inputs")
        lhs_contract = lhs_rank - 1
        if operator.attributes.get("rhs_transposed") or "linear" in _target(operator):
            rhs_contract = rhs_rank - 1
        else:
            rhs_contract = rhs_rank - 2
        lhs_batch_rank = max(0, lhs_rank - 2)
        rhs_batch_rank = max(0, rhs_rank - 2)
        batching = min(lhs_batch_rank, rhs_batch_rank)
        if batching and lhs.shape[:batching] != rhs.shape[:batching]:
            raise FrontendImportError(f"Matmul '{operator.op_id}' has incompatible batch dimensions")
        lhs_batch = tuple(range(batching))
        rhs_batch = tuple(range(batching))
        dot_name = output.name
        extra: list[str] = []
        if len(operator.inputs) == 3:
            dot_name = f"{operator.op_id}.matmul_output"
        dot_type = _tensor_type(output)
        lhs_type = _tensor_type(lhs)
        rhs_type = _tensor_type(rhs)
        extra.append(
            f"{_value(dot_name)} = stablehlo.dot_general {_value(lhs.name)}, {_value(rhs.name)}, "
            f"batching_dims = [{', '.join(str(item) for item in lhs_batch)}] x "
            f"[{', '.join(str(item) for item in rhs_batch)}], "
            f"contracting_dims = [{lhs_contract}] x [{rhs_contract}] : "
            f"({lhs_type}, {rhs_type}) -> {dot_type}"
        )
        if len(operator.inputs) == 3:
            bias = tensors[operator.inputs[2]]
            extra.append(
                f"{_value(output.name)} = stablehlo.add {_value(dot_name)}, {_value(bias.name)} : {dot_type}"
            )
        return extra

    def _emit_reduce(self, operator: OperatorSpec, tensors: dict[str, TensorSpec], output: TensorSpec) -> list[str]:
        source = tensors[operator.inputs[0]]
        axes = tuple(_axis_from_name(name, len(source.shape)) for name, _ in operator.reduction_dims)
        init_name = f"{operator.op_id}.init"
        lines = [self._emit_constant(init_name, 0.0, source.dtype)]
        axes_text = ", ".join(str(axis) for axis in axes)
        lines.append(
            f"{_value(output.name)} = stablehlo.reduce {_value(source.name)}, {_value(init_name)} "
            f"dimensions = [{axes_text}] reducer = add : "
            f"({_tensor_type(source)}, tensor<{_dtype(source.dtype)}>) -> {_tensor_type(output)}"
        )
        return lines

    def _emit_elementwise(self, operator: OperatorSpec, tensors: dict[str, TensorSpec], output: TensorSpec) -> list[str]:
        target = _elementwise_target(operator)
        raw_occurrences = operator.attributes.get("input_occurrences", operator.inputs)
        operands = [_value(tensors[str(name)].name) for name in raw_occurrences]
        scalar_values = _scalar_constants(operator)
        lines: list[str] = []
        for index, value in enumerate(scalar_values):
            name = f"{operator.op_id}.const{index}"
            lines.append(self._emit_constant(name, value, output.dtype))
            operands.append(_value(name))
        if not operands:
            raise FrontendImportError(f"elementwise '{operator.op_id}' has no operands")
        lines.append(
            f"{_value(output.name)} = stablehlo.{target} {', '.join(operands)} : {_tensor_type(output)}"
        )
        return lines

    def _emit_transpose(self, operator: OperatorSpec, tensors: dict[str, TensorSpec], output: TensorSpec) -> list[str]:
        source = tensors[operator.inputs[0]]
        permutation = operator.attributes.get("transpose_dims")
        if not isinstance(permutation, (tuple, list)):
            permutation = list(reversed(range(len(source.shape))))
        # torch.export records ``transpose`` as the two swapped axes, while
        # StableHLO expects the complete permutation.  Preserve a complete
        # permutation when one was supplied by a StableHLO-origin graph.
        if len(permutation) == 2 and len(source.shape) != 2:
            first, second = (int(item) for item in permutation)
            permutation = list(range(len(source.shape)))
            permutation[first], permutation[second] = permutation[second], permutation[first]
        return [
            f"{_value(output.name)} = stablehlo.transpose {_value(source.name)}, dimensions = "
            f"[{', '.join(str(item) for item in permutation)}] : {_tensor_type(output)}"
        ]

    def _emit_softmax(self, operator: OperatorSpec, tensors: dict[str, TensorSpec], output: TensorSpec) -> list[str]:
        source = tensors[operator.inputs[0]]
        axes = tuple(int(item) for item in operator.attributes.get("axes", (len(source.shape) - 1,)))
        reduced_shape = _shape_with_reduction(source.shape, axes)
        reduced = TensorSpec(f"{operator.op_id}.reduced", reduced_shape, source.dtype)
        lines: list[str] = []
        neg_inf = f"{operator.op_id}.neg_inf"
        zero = f"{operator.op_id}.zero"
        lines.append(self._emit_constant(neg_inf, -3.4028235e38, source.dtype))
        lines.append(self._emit_constant(zero, 0.0, source.dtype))
        axis_text = ", ".join(str(axis) for axis in axes)
        max_name = f"{operator.op_id}.max"
        shifted_name = f"{operator.op_id}.shifted"
        exp_name = f"{operator.op_id}.exp"
        sum_name = f"{operator.op_id}.sum"
        lines.append(
            f"{_value(max_name)} = stablehlo.reduce {_value(source.name)}, {_value(neg_inf)} "
            f"dimensions = [{axis_text}] reducer = maximum : "
            f"({_tensor_type(source)}, tensor<{_dtype(source.dtype)}>) -> {_tensor_type(reduced)}"
        )
        lines.append(
            f"{_value(shifted_name)} = stablehlo.subtract {_value(source.name)}, {_value(max_name)} : {_tensor_type(source)}"
        )
        lines.append(
            f"{_value(exp_name)} = stablehlo.exponential {_value(shifted_name)} : {_tensor_type(source)}"
        )
        lines.append(
            f"{_value(sum_name)} = stablehlo.reduce {_value(exp_name)}, {_value(zero)} "
            f"dimensions = [{axis_text}] reducer = add : "
            f"({_tensor_type(source)}, tensor<{_dtype(source.dtype)}>) -> {_tensor_type(reduced)}"
        )
        lines.append(
            f"{_value(output.name)} = stablehlo.divide {_value(exp_name)}, {_value(sum_name)} : {_tensor_type(output)}"
        )
        return lines

    def _emit_rmsnorm(self, operator: OperatorSpec, tensors: dict[str, TensorSpec], output: TensorSpec) -> list[str]:
        source = tensors[operator.inputs[0]]
        axis = _axis_from_name(operator.reduction_dims[0][0], len(source.shape))
        reduced_shape = _shape_with_reduction(source.shape, (axis,))
        reduced = TensorSpec(f"{operator.op_id}.reduced", reduced_shape, source.dtype)
        square = f"{operator.op_id}.square"
        sum_name = f"{operator.op_id}.sum"
        epsilon = f"{operator.op_id}.epsilon"
        shifted = f"{operator.op_id}.epsilon_sum"
        inverse = f"{operator.op_id}.inverse"
        lines = [
            f"{_value(square)} = stablehlo.multiply {_value(source.name)}, {_value(source.name)} : {_tensor_type(source)}",
            self._emit_constant(f"{operator.op_id}.zero", 0.0, source.dtype),
            f"{_value(sum_name)} = stablehlo.reduce {_value(square)}, {_value(operator.op_id + '.zero')} dimensions = [{axis}] reducer = add : ({_tensor_type(source)}, tensor<{_dtype(source.dtype)}>) -> {_tensor_type(reduced)}",
            self._emit_constant(epsilon, operator.attributes.get("epsilon", 1e-5), source.dtype),
            f"{_value(shifted)} = stablehlo.add {_value(sum_name)}, {_value(epsilon)} : {_tensor_type(reduced)}",
            f"{_value(inverse)} = stablehlo.rsqrt {_value(shifted)} : {_tensor_type(reduced)}",
            f"{_value(output.name)} = stablehlo.multiply {_value(source.name)}, {_value(inverse)} : {_tensor_type(output)}",
        ]
        return lines

    def _emit_layernorm(self, operator: OperatorSpec, tensors: dict[str, TensorSpec], output: TensorSpec) -> list[str]:
        if len(operator.reduction_dims) != 1:
            raise FrontendImportError("generated LayerNorm currently supports one normalized axis")
        source = tensors[operator.inputs[0]]
        axis = _axis_from_name(operator.reduction_dims[0][0], len(source.shape))
        reduced_shape = _shape_with_reduction(source.shape, (axis,))
        reduced = TensorSpec(f"{operator.op_id}.reduced", reduced_shape, source.dtype)
        prefix = operator.op_id
        names = {
            "zero": f"{prefix}.zero",
            "sum": f"{prefix}.sum",
            "mean": f"{prefix}.mean",
            "center": f"{prefix}.center",
            "square": f"{prefix}.square",
            "variance_sum": f"{prefix}.variance_sum",
            "variance": f"{prefix}.variance",
            "epsilon": f"{prefix}.epsilon",
            "variance_eps": f"{prefix}.variance_eps",
            "inverse": f"{prefix}.inverse",
            "normalized": f"{prefix}.normalized",
        }
        denominator = source.shape[axis]
        lines = [self._emit_constant(names["zero"], 0.0, source.dtype)]
        axis_text = str(axis)
        lines.extend(
            [
                f"{_value(names['sum'])} = stablehlo.reduce {_value(source.name)}, {_value(names['zero'])} dimensions = [{axis_text}] reducer = add : ({_tensor_type(source)}, tensor<{_dtype(source.dtype)}>) -> {_tensor_type(reduced)}",
                self._emit_constant(f"{prefix}.denominator", denominator, source.dtype),
                f"{_value(names['mean'])} = stablehlo.divide {_value(names['sum'])}, {_value(prefix + '.denominator')} : {_tensor_type(reduced)}",
                f"{_value(names['center'])} = stablehlo.subtract {_value(source.name)}, {_value(names['mean'])} : {_tensor_type(source)}",
                f"{_value(names['square'])} = stablehlo.multiply {_value(names['center'])}, {_value(names['center'])} : {_tensor_type(source)}",
                f"{_value(names['variance_sum'])} = stablehlo.reduce {_value(names['square'])}, {_value(names['zero'])} dimensions = [{axis_text}] reducer = add : ({_tensor_type(source)}, tensor<{_dtype(source.dtype)}>) -> {_tensor_type(reduced)}",
                f"{_value(names['variance'])} = stablehlo.divide {_value(names['variance_sum'])}, {_value(prefix + '.denominator')} : {_tensor_type(reduced)}",
                self._emit_constant(names["epsilon"], operator.attributes.get("epsilon", 1e-5), source.dtype),
                f"{_value(names['variance_eps'])} = stablehlo.add {_value(names['variance'])}, {_value(names['epsilon'])} : {_tensor_type(reduced)}",
                f"{_value(names['inverse'])} = stablehlo.rsqrt {_value(names['variance_eps'])} : {_tensor_type(reduced)}",
            ]
        )
        if len(operator.inputs) == 3:
            lines.append(
                f"{_value(names['normalized'])} = stablehlo.multiply {_value(names['center'])}, {_value(names['inverse'])} : {_tensor_type(source)}"
            )
            final_name = names["normalized"]
            weight = tensors[operator.inputs[1]]
            bias = tensors[operator.inputs[2]]
            scaled = f"{prefix}.scaled"
            lines.extend(
                [
                    f"{_value(scaled)} = stablehlo.multiply {_value(final_name)}, {_value(weight.name)} : {_tensor_type(source)}",
                    f"{_value(output.name)} = stablehlo.add {_value(scaled)}, {_value(bias.name)} : {_tensor_type(output)}",
                ]
            )
        else:
            lines.append(
                f"{_value(output.name)} = stablehlo.multiply {_value(names['center'])}, {_value(names['inverse'])} : {_tensor_type(output)}"
            )
        return lines


def generate_stablehlo(imported: FrontendImport, *, variant: str = "stablehlo-generated-v0") -> StableHLOModule:
    return StableHLOGenerator(variant=variant).generate(imported)


__all__ = ["StableHLOGenerator", "StableHLOModule", "generate_stablehlo"]
