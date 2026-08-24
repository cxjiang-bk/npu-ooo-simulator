from __future__ import annotations

"""Official StableHLO/MLIR binding integration.

The dependency-light emitter in :mod:`stablehlo_codegen` is useful for
regression tests, but it intentionally does not implement the MLIR grammar.
This module is the opt-in official path: it emits a small valid StableHLO
module, parses it with the OpenXLA StableHLO Python wheel, runs the MLIR
verifier, and projects the verified module back into ``FrontendImport``.

The exporter boundary is deliberately separate from the dialect bindings.
This package verifies and imports StableHLO; a PyTorch/XLA or torch-mlir
bridge can be connected later without changing the verifier contract.
"""

from dataclasses import dataclass, field
import importlib.metadata
import re
from typing import Any, Mapping

from npu_ooo.ir import TensorSpec

from .bridge import FrontendImport, FrontendImportError, FrontendKind
from .stablehlo import StableHLOAdapter
from .stablehlo_codegen import (
    StableHLOGenerator,
    _axis_from_name,
    _dtype,
    _tensor_type,
    _value,
)


def _bindings() -> tuple[Any, Any, Any]:
    """Load the official MLIR bindings lazily with an actionable error."""

    try:
        from mlir.ir import Context, Module
        import mlir.dialects.stablehlo as stablehlo_dialect
    except ModuleNotFoundError as exc:
        raise FrontendImportError(
            "official StableHLO bindings are unavailable; install the OpenXLA "
            "StableHLO wheel (see docs/install-stablehlo.md)"
        ) from exc
    return Context, Module, stablehlo_dialect


def official_stablehlo_available() -> bool:
    try:
        _bindings()
    except FrontendImportError:
        return False
    return True


def official_stablehlo_version() -> str | None:
    for distribution in ("stablehlo", "mlir-python-bindings"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


@dataclass(frozen=True)
class OfficialStableHLOModule:
    """A verified module produced/parsed through official MLIR bindings."""

    text: str
    canonical_text: str
    model_id: str
    variant: str = "stablehlo-official-v1"
    stablehlo_version: str | None = None
    verified: bool = True
    producer: str = "project-stablehlo-legalizer"
    verifier: str = "official-stablehlo-mlir"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.text.strip():
            issues.append("official StableHLO text must not be empty")
        if not self.canonical_text.strip():
            issues.append("official StableHLO canonical text must not be empty")
        if not self.model_id:
            issues.append("official StableHLO model_id must not be empty")
        if not self.verified:
            issues.append("official StableHLO module must be verified")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "canonical_text": self.canonical_text,
            "model_id": self.model_id,
            "variant": self.variant,
            "stablehlo_version": self.stablehlo_version,
            "verified": self.verified,
            "producer": self.producer,
            "verifier": self.verifier,
            "provenance": dict(self.provenance),
        }


class OfficialStableHLOGenerator(StableHLOGenerator):
    """Emit the supported subset using official StableHLO syntax/semantics."""

    def __init__(self, *, variant: str = "stablehlo-official-v1") -> None:
        super().__init__(variant=variant)
        self._lines: list[str] = []
        self._tensors: dict[str, TensorSpec] = {}
        self._broadcast_counter = 0

    def generate(self, imported: FrontendImport) -> OfficialStableHLOModule:
        issues = imported.validate()
        if issues:
            raise FrontendImportError("cannot generate official StableHLO: " + "; ".join(issues))
        graph = imported.graph
        self._tensors = {tensor.name: tensor for tensor in graph.tensors}
        self._lines = []
        self._broadcast_counter = 0
        graph_inputs = tuple(str(item) for item in graph.attributes.get("graph_inputs", ()))
        graph_outputs = tuple(str(item) for item in graph.attributes.get("graph_outputs", ()))
        if not graph_inputs:
            graph_inputs = tuple(
                tensor.name
                for tensor in graph.tensors
                if tensor.attributes.get("source_kind") in {"input", "parameter", "buffer"}
            )
        if not graph_outputs:
            graph_outputs = self._infer_outputs(graph)
        if len(graph_outputs) != 1:
            raise FrontendImportError("official StableHLO generator currently supports one graph output")
        try:
            arguments = [f"    {_value(name)}: {_tensor_type(self._tensors[name])}" for name in graph_inputs]
            return_type = _tensor_type(self._tensors[graph_outputs[0]])
        except KeyError as exc:
            raise FrontendImportError(f"official StableHLO graph references unknown tensor: {exc}") from exc

        self._lines.extend(("module {", "  func.func @main(", ",\n".join(arguments), f") -> {return_type} {{"))
        for operator_id in graph.topological_order():
            operator = next(item for item in graph.operators if item.op_id == operator_id)
            for line in self._emit_official_operator(operator):
                self._lines.extend(f"    {part}" for part in line.splitlines())
        self._lines.append(f"    return {_value(graph_outputs[0])} : {return_type}")
        self._lines.extend(("  }", "}"))
        text = "\n".join(self._lines) + "\n"
        module, canonical, _ = _parse_verified(text)
        del module
        result = OfficialStableHLOModule(
            text=text,
            canonical_text=canonical,
            model_id=imported.model_id,
            variant=self.variant,
            stablehlo_version=official_stablehlo_version(),
            provenance={
                "source": "frontend-import",
                "source_graph_id": graph.graph_id,
                "source_frontend": imported.frontend.value
                if hasattr(imported.frontend, "value")
                else str(imported.frontend),
                "generator": "official-stablehlo-graph-v1",
                "operator_count": len(graph.operators),
            },
        )
        result_issues = result.validate()
        if result_issues:
            raise FrontendImportError("generated official StableHLO is invalid: " + "; ".join(result_issues))
        return result

    def _tensor(self, name: str) -> TensorSpec:
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise FrontendImportError(f"unknown tensor '{name}' in official StableHLO generator") from exc

    def _broadcast(
        self,
        name: str,
        target: TensorSpec,
        *,
        dimensions: tuple[int, ...] | None = None,
        sink: list[str] | None = None,
    ) -> str:
        source = self._tensor(name)
        if source.shape == target.shape:
            return _value(name)
        if len(source.shape) > len(target.shape):
            raise FrontendImportError(f"cannot broadcast {source.shape} to {target.shape}")
        if dimensions is None:
            dimensions = tuple(range(len(target.shape) - len(source.shape), len(target.shape)))
        self._broadcast_counter += 1
        result_name = f"broadcast_{self._broadcast_counter}"
        line = (
            f"{_value(result_name)} = stablehlo.broadcast_in_dim {_value(name)}, "
            f"dims = [{', '.join(str(item) for item in dimensions)}] : "
            f"({_tensor_type(source)}) -> {_tensor_type(target)}"
        )
        (sink if sink is not None else self._lines).append(line)
        self._tensors[result_name] = TensorSpec(result_name, target.shape, target.dtype)
        return _value(result_name)

    def _constant(self, name: str, value: Any, dtype: str) -> str:
        is_float = _dtype(dtype) in {"f16", "bf16", "f32", "f64"}
        if value is True:
            rendered = "true"
        elif value is False:
            rendered = "false"
        elif is_float and isinstance(value, int):
            rendered = f"{value}.0"
        elif isinstance(value, float):
            rendered = repr(value)
            if "e" in rendered.lower() and "." not in rendered.split("e", 1)[0].split("E", 1)[0]:
                marker = "e" if "e" in rendered else "E"
                mantissa, exponent = rendered.split(marker, 1)
                rendered = f"{mantissa}.0{marker}{exponent}"
        else:
            rendered = str(value)
        self._tensors[name] = TensorSpec(name, (), dtype)
        return f"{_value(name)} = stablehlo.constant dense<{rendered}> : tensor<{_dtype(dtype)}>"

    def _reduce(self, name: str, source: TensorSpec, axes: tuple[int, ...], reducer: str) -> tuple[str, str]:
        natural_shape = tuple(extent for axis, extent in enumerate(source.shape) if axis not in axes)
        result_name = name
        result_type = _tensor_type(TensorSpec(result_name, natural_shape, source.dtype))
        init_name = f"{name}.init"
        init_line = self._constant(init_name, -3.4028235e38 if reducer == "maximum" else 0.0, source.dtype)
        scalar_type = f"tensor<{_dtype(source.dtype)}>"
        reducer_op = "maximum" if reducer == "maximum" else "add"
        dimensions = ", ".join(str(axis) for axis in axes)
        line = (
            f"{_value(result_name)} = \"stablehlo.reduce\"({_value(source.name)}, {_value(init_name)}) ({{\n"
            f"^bb0(%arg0: {scalar_type}, %arg1: {scalar_type}):\n"
            f"  %result = stablehlo.{reducer_op} %arg0, %arg1 : {scalar_type}\n"
            f"  \"stablehlo.return\"(%result) : ({scalar_type}) -> ()\n"
            f"}}) {{ dimensions = array<i64: {dimensions}> }} : "
            f"({_tensor_type(source)}, {scalar_type}) -> {result_type}"
        )
        return init_line, line

    def _emit_official_operator(self, operator: OperatorSpec) -> list[str]:
        if len(operator.outputs) != 1:
            raise FrontendImportError(f"official StableHLO requires one output for '{operator.op_id}'")
        output = self._tensor(operator.outputs[0])
        if operator.normalized_type in {"matmul", "batched_matmul", "gemv"}:
            lhs, rhs = self._tensor(operator.inputs[0]), self._tensor(operator.inputs[1])
            lhs_rank, rhs_rank = len(lhs.shape), len(rhs.shape)
            lhs_contract = lhs_rank - 1
            target = str(operator.attributes.get("frontend_target", "")).lower()
            rhs_contract = (
                rhs_rank - 1
                if operator.attributes.get("rhs_transposed") or "linear" in target
                else rhs_rank - 2
            )
            batching = min(max(0, lhs_rank - 2), max(0, rhs_rank - 2))
            lhs_batch, rhs_batch = tuple(range(batching)), tuple(range(batching))
            dot_name = output.name if len(operator.inputs) != 3 else f"{operator.op_id}.matmul_output"
            self._tensors[dot_name] = TensorSpec(dot_name, output.shape, output.dtype)
            lines = [
                f"{_value(dot_name)} = stablehlo.dot_general {_value(lhs.name)}, {_value(rhs.name)}, "
                f"batching_dims = [{', '.join(map(str, lhs_batch))}] x [{', '.join(map(str, rhs_batch))}], "
                f"contracting_dims = [{lhs_contract}] x [{rhs_contract}] : "
                f"({_tensor_type(lhs)}, {_tensor_type(rhs)}) -> {_tensor_type(output)}"
            ]
            if len(operator.inputs) == 3:
                bias = self._tensor(operator.inputs[2])
                bias_value = self._broadcast(bias.name, output, sink=lines)
                lines.append(f"{_value(output.name)} = stablehlo.add {_value(dot_name)}, {bias_value} : {_tensor_type(output)}")
            return lines
        if operator.normalized_type == "reduce":
            source = self._tensor(operator.inputs[0])
            axes = tuple(_axis_from_name(axis, len(source.shape)) for axis, _ in operator.reduction_dims)
            natural_shape = tuple(extent for axis, extent in enumerate(source.shape) if axis not in axes)
            natural = TensorSpec(f"{output.name}.reduced", natural_shape, source.dtype)
            target = str(operator.attributes.get("frontend_target", "")).lower()
            reducer = str(operator.attributes.get("reducer", "add"))
            reduce_name = natural.name if output.shape != natural_shape or "mean" in target else output.name
            init_line, reduce_line = self._reduce(reduce_name, source, axes, reducer)
            self._tensors[reduce_name] = TensorSpec(reduce_name, natural_shape, source.dtype)
            lines = [init_line, reduce_line]
            value_name = reduce_name
            if "mean" in target:
                extent = 1
                for axis in axes:
                    if not isinstance(source.shape[axis], int):
                        raise FrontendImportError("official StableHLO mean requires static reduction extents")
                    extent *= source.shape[axis]
                denominator = f"{output.name}.denominator"
                lines.append(self._constant(denominator, extent, source.dtype))
                denominator_value = self._broadcast(
                    denominator,
                    self._tensors[reduce_name],
                    dimensions=(),
                    sink=lines,
                )
                value_name = f"{output.name}.mean"
                lines.append(
                    f"{_value(value_name)} = stablehlo.divide {_value(reduce_name)}, "
                    f"{denominator_value} : {_tensor_type(self._tensors[reduce_name])}"
                )
                self._tensors[value_name] = TensorSpec(value_name, natural_shape, source.dtype)
            if output.shape != natural_shape:
                dimensions = tuple(index for index in range(len(source.shape)) if index not in axes)
                lines.append(
                    f"{_value(output.name)} = stablehlo.broadcast_in_dim {_value(value_name)}, "
                    f"dims = [{', '.join(map(str, dimensions))}] : "
                    f"({_tensor_type(self._tensors[value_name])}) -> {_tensor_type(output)}"
                )
            elif value_name != output.name:
                one = f"{output.name}.one"
                lines.append(self._constant(one, 1.0, source.dtype))
                one_value = self._broadcast(one, output, dimensions=(), sink=lines)
                lines.append(
                    f"{_value(output.name)} = stablehlo.multiply {_value(value_name)}, "
                    f"{one_value} : {_tensor_type(output)}"
                )
            return lines
        if operator.normalized_type == "softmax":
            source = self._tensor(operator.inputs[0])
            axes = tuple(int(item) for item in operator.attributes.get("axes", (len(source.shape) - 1,)))
            reduced_shape = tuple(extent for axis, extent in enumerate(source.shape) if axis not in axes)
            reduced = TensorSpec(f"{operator.op_id}.reduced", reduced_shape, source.dtype)
            neg = f"{operator.op_id}.neg_inf"
            zero = f"{operator.op_id}.zero"
            lines = [self._constant(neg, -3.4028235e38, source.dtype), self._constant(zero, 0.0, source.dtype)]
            _, maximum = self._reduce(f"{operator.op_id}.max", source, axes, "maximum")
            self._tensors[f"{operator.op_id}.max"] = reduced
            lines.append(maximum.replace(_value(f"{operator.op_id}.max.init"), _value(neg)))
            max_value = self._broadcast(f"{operator.op_id}.max", source, dimensions=tuple(i for i in range(len(source.shape)) if i not in axes), sink=lines)
            shifted = f"{operator.op_id}.shifted"
            exp = f"{operator.op_id}.exp"
            total = f"{operator.op_id}.sum"
            lines.extend([
                f"{_value(shifted)} = stablehlo.subtract {_value(source.name)}, {max_value} : {_tensor_type(source)}",
                f"{_value(exp)} = stablehlo.exponential {_value(shifted)} : {_tensor_type(source)}",
            ])
            self._tensors[exp] = TensorSpec(exp, source.shape, source.dtype)
            _, total_line = self._reduce(total, self._tensors[exp], axes, "add")
            lines.append(total_line.replace(_value(f"{total}.init"), _value(zero)))
            self._tensors[total] = reduced
            total_value = self._broadcast(total, source, dimensions=tuple(i for i in range(len(source.shape)) if i not in axes), sink=lines)
            lines.append(f"{_value(output.name)} = stablehlo.divide {_value(exp)}, {total_value} : {_tensor_type(output)}")
            return lines
        if operator.normalized_type in {"rmsnorm", "layernorm"}:
            return self._emit_norm(operator, output)
        if operator.normalized_type in {"elementwise", "residual_add"}:
            target = str(operator.attributes.get("frontend_target", "add")).lower()
            names = (("mul", "multiply"), ("add", "add"), ("sub", "subtract"), ("div", "divide"), ("rsqrt", "rsqrt"), ("exp", "exponential"), ("pow", "power"))
            stable_name = next((value for token, value in names if token in target), "add")
            lines: list[str] = []
            raw_occurrences = operator.attributes.get("input_occurrences", operator.inputs)
            occurrences = tuple(str(item) for item in raw_occurrences)
            operands = [self._broadcast(name, output, sink=lines) for name in occurrences]
            constant_args = operator.attributes.get("constant_args", {})
            scalar_args = constant_args.get("args", ()) if isinstance(constant_args, Mapping) else ()
            for index, value in enumerate(scalar_args):
                if not isinstance(value, (bool, int, float)):
                    continue
                constant_name = f"{operator.op_id}.const{index}"
                lines.append(self._constant(constant_name, value, output.dtype))
                operands.append(self._broadcast(constant_name, output, dimensions=(), sink=lines))
            if not operands:
                raise FrontendImportError(f"elementwise '{operator.op_id}' has no operands")
            lines.append(f"{_value(output.name)} = stablehlo.{stable_name} {', '.join(operands)} : {_tensor_type(output)}")
            return lines
        if operator.normalized_type == "transpose":
            source = self._tensor(operator.inputs[0])
            permutation = operator.attributes.get("transpose_dims", tuple(reversed(range(len(source.shape)))))
            if len(permutation) == 2 and len(source.shape) != 2:
                permutation = list(range(len(source.shape)))
                permutation[operator.attributes["transpose_dims"][0]], permutation[operator.attributes["transpose_dims"][1]] = permutation[operator.attributes["transpose_dims"][1]], permutation[operator.attributes["transpose_dims"][0]]
            return [f"{_value(output.name)} = \"stablehlo.transpose\"({_value(source.name)}) {{permutation = array<i64: {', '.join(map(str, permutation))}>}} : ({_tensor_type(source)}) -> {_tensor_type(output)}"]
        raise FrontendImportError(f"official StableHLO generation does not support '{operator.normalized_type}'")

    def _emit_norm(self, operator: OperatorSpec, output: TensorSpec) -> list[str]:
        source = self._tensor(operator.inputs[0])
        axis = _axis_from_name(operator.reduction_dims[0][0], len(source.shape))
        reduced_shape = tuple(extent for index, extent in enumerate(source.shape) if index != axis)
        reduced = TensorSpec(f"{operator.op_id}.reduced", reduced_shape, source.dtype)
        lines = [f"    {self._constant(operator.op_id + '.zero', 0.0, source.dtype)}"]
        zero = operator.op_id + ".zero"
        square = operator.op_id + ".square"
        if operator.normalized_type == "rmsnorm":
            lines.append(f"{_value(square)} = stablehlo.multiply {_value(source.name)}, {_value(source.name)} : {_tensor_type(source)}")
            self._tensors[square] = TensorSpec(square, source.shape, source.dtype)
            reduction_source = self._tensors[square]
        else:
            reduction_source = source
        _, sum_line = self._reduce(operator.op_id + ".sum", reduction_source, (axis,), "add")
        lines.append(sum_line.replace(_value(operator.op_id + ".sum.init"), _value(zero)))
        self._tensors[operator.op_id + ".sum"] = reduced
        denominator = source.shape[axis]
        lines.append(self._constant(operator.op_id + ".denominator", denominator, source.dtype))
        denominator_value = self._broadcast(operator.op_id + ".denominator", reduced, dimensions=(), sink=lines)
        lines.append(f"{_value(operator.op_id + '.mean')} = stablehlo.divide {_value(operator.op_id + '.sum')}, {denominator_value} : {_tensor_type(reduced)}")
        self._tensors[operator.op_id + ".mean"] = reduced
        if operator.normalized_type == "layernorm":
            mean_b = self._broadcast(operator.op_id + ".mean", source, dimensions=tuple(i for i in range(len(source.shape)) if i != axis), sink=lines)
            center = operator.op_id + ".center"
            lines.append(f"{_value(center)} = stablehlo.subtract {_value(source.name)}, {mean_b} : {_tensor_type(source)}")
            self._tensors[center] = TensorSpec(center, source.shape, source.dtype)
            square_center = operator.op_id + ".square"
            lines.append(f"{_value(square_center)} = stablehlo.multiply {_value(center)}, {_value(center)} : {_tensor_type(source)}")
            self._tensors[square_center] = TensorSpec(square_center, source.shape, source.dtype)
            _, variance_sum = self._reduce(operator.op_id + ".variance_sum", self._tensors[square_center], (axis,), "add")
            lines.append(variance_sum.replace(_value(operator.op_id + ".variance_sum.init"), _value(zero)))
            self._tensors[operator.op_id + ".variance_sum"] = reduced
            lines.append(self._constant(operator.op_id + ".epsilon", operator.attributes.get("epsilon", 1e-5), source.dtype))
            variance_sum_value = _value(operator.op_id + ".variance_sum")
            denominator_value = self._broadcast(operator.op_id + ".denominator", reduced, dimensions=(), sink=lines)
            lines.append(f"{_value(operator.op_id + '.variance')} = stablehlo.divide {variance_sum_value}, {denominator_value} : {_tensor_type(reduced)}")
            self._tensors[operator.op_id + ".variance"] = reduced
            epsilon_value = self._broadcast(operator.op_id + ".epsilon", reduced, dimensions=(), sink=lines)
            lines.append(f"{_value(operator.op_id + '.variance_eps')} = stablehlo.add {_value(operator.op_id + '.variance')}, {epsilon_value} : {_tensor_type(reduced)}")
            variance_input = operator.op_id + ".variance_eps"
        else:
            lines.append(self._constant(operator.op_id + ".epsilon", operator.attributes.get("epsilon", 1e-5), source.dtype))
            epsilon_value = self._broadcast(operator.op_id + ".epsilon", reduced, dimensions=(), sink=lines)
            lines.append(f"{_value(operator.op_id + '.variance_eps')} = stablehlo.add {_value(operator.op_id + '.mean')}, {epsilon_value} : {_tensor_type(reduced)}")
            variance_input = operator.op_id + ".variance_eps"
        lines.append(f"{_value(operator.op_id + '.inverse')} = stablehlo.rsqrt {_value(variance_input)} : {_tensor_type(reduced)}")
        self._tensors[operator.op_id + ".inverse"] = reduced
        inverse_b = self._broadcast(operator.op_id + ".inverse", source, dimensions=tuple(i for i in range(len(source.shape)) if i != axis), sink=lines)
        has_affine = operator.normalized_type == "layernorm" and len(operator.inputs) == 3
        normalized = operator.op_id + ".normalized" if has_affine else output.name
        normalized_source = center if operator.normalized_type == "layernorm" else source.name
        lines.append(f"{_value(normalized)} = stablehlo.multiply {_value(normalized_source)}, {inverse_b} : {_tensor_type(source)}")
        self._tensors[normalized] = TensorSpec(normalized, source.shape, source.dtype)
        if has_affine:
            weight_b = self._broadcast(operator.inputs[1], source, sink=lines)
            bias_b = self._broadcast(operator.inputs[2], output, sink=lines)
            scaled = operator.op_id + ".scaled"
            lines.append(f"{_value(scaled)} = stablehlo.multiply {_value(normalized)}, {weight_b} : {_tensor_type(source)}")
            self._tensors[scaled] = TensorSpec(scaled, source.shape, source.dtype)
            lines.append(f"{_value(output.name)} = stablehlo.add {_value(scaled)}, {bias_b} : {_tensor_type(output)}")
        return [line[4:] if line.startswith("    ") else line for line in lines]


def _parse_verified(text: str) -> tuple[Any, str, Any]:
    Context, Module, dialect = _bindings()
    with Context() as context:
        dialect.register_dialect(context)
        try:
            module = Module.parse(text)
            module.operation.verify()
        except Exception as exc:
            raise FrontendImportError(f"official StableHLO parse/verify failed: {exc}") from exc
        return module, str(module), context


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"-?\d+", text))


def _project_module(module: Any) -> str:
    """Project a verified MLIR module to the adapter's readable subset."""

    functions = [op for op in module.operation.regions[0].blocks[0] if op.operation.name == "func.func"]
    if not functions:
        raise FrontendImportError("official StableHLO module has no func.func entry point")
    function = functions[0]
    block = function.regions[0].blocks[0]
    value_names: dict[Any, str] = {arg: f"arg{index}" for index, arg in enumerate(block.arguments)}
    lines = ["module {", "  func.func @main("]
    lines.append(",\n".join(f"    %{value_names[arg]}: {arg.type}" for arg in block.arguments))
    return_type = str(function.attributes["function_type"]).split("->", 1)[-1].strip().strip(")")
    lines.append(f") -> {return_type} {{")
    counter = 0
    omitted_results: set[Any] = set()
    for operation in block:
        name = operation.name
        if name == "func.return":
            if any(value in omitted_results for value in operation.operands):
                raise FrontendImportError(
                    "official StableHLO projection does not support returning a secondary operation result"
                )
            returns = [value_names[value] for value in operation.operands]
            return_types = ", ".join(str(value.type) for value in operation.operands)
            lines.append(f"    return {', '.join('%' + item for item in returns)} : {return_types}")
            continue
        if not name.startswith("stablehlo."):
            continue
        if not operation.results:
            continue
        result_names: list[str] = []
        for result_index, operation_result in enumerate(operation.results):
            result_name = f"v{counter}"
            counter += 1
            value_names[operation_result] = result_name
            result_names.append(result_name)
            if result_index:
                omitted_results.add(operation_result)
        result = operation.results[0]
        result_name = result_names[0]
        unsupported_operands = [value for value in operation.operands if value in omitted_results]
        if unsupported_operands:
            raise FrontendImportError(
                f"official StableHLO projection does not support consuming a secondary result of '{name}'"
            )
        operands = [value_names[value] for value in operation.operands]
        operand_types = [str(value.type) for value in operation.operands]
        result_type = str(result.type)
        if name == "stablehlo.broadcast_in_dim":
            value_names[result] = value_names[operation.operands[0]]
            continue
        if name == "stablehlo.constant":
            value = str(operation.attributes["value"])
            dense = value.split(":", 1)[0].strip()
            lines.append(f"    %{result_name} = stablehlo.constant {dense} : {result_type}")
        elif name == "stablehlo.reduce":
            dims_text = str(operation.attributes["dimensions"])
            dims_match = re.search(r"array<i64:\s*([^>]*)>", dims_text)
            dims = _ints(dims_match.group(1) if dims_match else dims_text)
            reducer = "add"
            if operation.regions and operation.regions[0].blocks:
                region_ops = list(operation.regions[0].blocks[0])
                for region_op in region_ops:
                    if region_op.name.startswith("stablehlo.") and region_op.name not in {"stablehlo.return"}:
                        reducer = region_op.name.removeprefix("stablehlo.")
                        break
            lines.append(
                f"    %{result_name} = stablehlo.reduce %{operands[0]}, %{operands[1]} dimensions = [{', '.join(map(str, dims))}] reducer = {reducer} : "
                f"({', '.join(operand_types)}) -> {result_type}"
            )
        elif name == "stablehlo.dot_general":
            attr = str(operation.attributes["dot_dimension_numbers"])
            lhs_contract = re.search(r"lhs_contracting_dimensions\s*=\s*\[([^]]*)\]", attr)
            rhs_contract = re.search(r"rhs_contracting_dimensions\s*=\s*\[([^]]*)\]", attr)
            lhs_batch = re.search(r"lhs_batching_dimensions\s*=\s*\[([^]]*)\]", attr)
            rhs_batch = re.search(r"rhs_batching_dimensions\s*=\s*\[([^]]*)\]", attr)
            def vals(match: re.Match[str] | None) -> str:
                return match.group(1).strip() if match else ""
            lines.append(
                f"    %{result_name} = stablehlo.dot_general %{operands[0]}, %{operands[1]}, "
                f"batching_dims = [{vals(lhs_batch)}] x [{vals(rhs_batch)}], contracting_dims = [{vals(lhs_contract)}] x [{vals(rhs_contract)}] : "
                f"({', '.join(operand_types)}) -> {result_type}"
            )
        elif name == "stablehlo.transpose":
            permutation_text = str(operation.attributes["permutation"])
            permutation_match = re.search(r"array<i64:\s*([^>]*)>", permutation_text)
            permutation = _ints(permutation_match.group(1) if permutation_match else permutation_text)
            lines.append(f"    %{result_name} = stablehlo.transpose %{operands[0]}, dimensions = [{', '.join(map(str, permutation))}] : {result_type}")
        elif name == "stablehlo.batch_norm_training":
            feature_index = str(operation.attributes["feature_index"]).split(":", 1)[0].strip()
            epsilon = str(operation.attributes["epsilon"]).split(":", 1)[0].strip()
            lines.append(
                f"    %{result_name} = stablehlo.batch_norm_training "
                f"{', '.join('%' + item for item in operands)} "
                f"feature_index = {feature_index} epsilon = {epsilon} : "
                f"({', '.join(operand_types)}) -> {result_type}"
            )
        else:
            target = name.removeprefix("stablehlo.")
            lines.append(f"    %{result_name} = stablehlo.{target} {', '.join('%' + item for item in operands)} : {result_type}")
    lines.extend(("  }", "}"))
    return "\n".join(lines) + "\n"


class OfficialStableHLOAdapter:
    """Parse, verify and import StableHLO through official MLIR bindings."""

    kind = FrontendKind.STABLEHLO

    @classmethod
    def parse_text(cls, text: str, *, model_id: str = "stablehlo_model", variant: str = "stablehlo-official-v1") -> OfficialStableHLOModule:
        if not isinstance(text, str) or not text.strip():
            raise FrontendImportError("official StableHLO text must be a non-empty string")
        _, canonical, _ = _parse_verified(text)
        return OfficialStableHLOModule(
            text=text,
            canonical_text=canonical,
            model_id=model_id,
            variant=variant,
            stablehlo_version=official_stablehlo_version(),
            producer="external-stablehlo",
            provenance={"source": "official-stablehlo-text", "verifier": "mlir.ir.Operation.verify"},
        )

    @classmethod
    def import_text(cls, text: str, *, model_id: str = "stablehlo_model", variant: str = "stablehlo-official-v1", shape_environment: Mapping[str, int] | None = None) -> FrontendImport:
        module_obj, canonical, _ = _parse_verified(text)
        projected = _project_module(module_obj)
        imported = StableHLOAdapter.from_text(projected, model_id=model_id, variant=variant, shape_environment=shape_environment)
        return FrontendImport(
            graph=imported.graph,
            model_id=imported.model_id,
            variant=imported.variant,
            shape_environment=imported.shape_environment,
            frontend=imported.frontend,
            provenance={
                **dict(imported.provenance),
                "source": "official-stablehlo-bindings",
                "canonical_assembly": canonical,
                "verifier": "mlir.ir.Operation.verify",
                "stablehlo_version": official_stablehlo_version(),
            },
            family=imported.family,
        )

    @classmethod
    def from_file(cls, path: str, **kwargs: Any) -> OfficialStableHLOModule:
        from pathlib import Path

        try:
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise FrontendImportError(f"StableHLO file does not exist: {path}") from exc
        return cls.parse_text(text, **kwargs)


__all__ = [
    "OfficialStableHLOAdapter",
    "OfficialStableHLOGenerator",
    "OfficialStableHLOModule",
    "official_stablehlo_available",
    "official_stablehlo_version",
]
