from __future__ import annotations

"""Static specialization for the dynamic shape subset emitted by Torch-XLA.

This pass deliberately operates on parsed StableHLO operation lines rather
than replacing ``?`` with a string substitution.  It evaluates the shape
tensor dataflow used by Torch-XLA (dimension-size, reshape, concatenate and
maximum), rewrites dynamic broadcasts to static broadcasts, removes dead
shape-only operations, and leaves final verification to official StableHLO.
"""

from dataclasses import dataclass
import re
from typing import Any, Mapping

from npu_ooo.ir import OperatorGraph

from .bridge import FrontendImportError, normalize_shape_environment


_VALUE = re.compile(r"%[A-Za-z_0-9][\w.$-]*")
_TENSOR = re.compile(r"tensor<([^>]+)>")
_OPERATION = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[A-Za-z_0-9][\w.$-]*)\s*=\s*"
    r"(?P<target>stablehlo\.[A-Za-z_][\w.]*)\s+(?P<body>.*?)\s*:\s*"
    r"(?P<signature>.+?)\s*$"
)
_DYNAMIC_OP = re.compile(r"stablehlo\.(?:get_dimension_size|dynamic_[A-Za-z_][\w.]*)")


@dataclass(frozen=True)
class ShapeSpecializationResult:
    text: str
    dynamic_operations: tuple[str, ...]
    removed_shape_operations: tuple[str, ...]
    shape_environment: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": "torch-xla-dynamic-shape-specialization-v1",
            "text": self.text,
            "dynamic_operations": list(self.dynamic_operations),
            "removed_shape_operations": list(self.removed_shape_operations),
            "shape_environment": dict(self.shape_environment),
        }


def _tensor_parts(type_text: str) -> tuple[tuple[int, ...] | None, str]:
    match = _TENSOR.search(type_text)
    if match is None:
        raise FrontendImportError(f"shape specialization expected tensor type, got '{type_text}'")
    payload = match.group(1).split(",", 1)[0].strip()
    parts = [part.strip() for part in payload.split("x") if part.strip()]
    if not parts:
        raise FrontendImportError(f"shape specialization found an empty tensor type '{type_text}'")
    dtype = parts[-1]
    dimensions: list[int] = []
    for dimension in parts[:-1]:
        if dimension in {"?", "*"}:
            return None, dtype
        try:
            dimensions.append(int(dimension))
        except ValueError as exc:
            raise FrontendImportError(
                f"shape specialization cannot parse tensor dimension '{dimension}'"
            ) from exc
    return tuple(dimensions), dtype


def _format_tensor(shape: tuple[int, ...], dtype: str) -> str:
    return f"tensor<{'x'.join(str(value) for value in (*shape, dtype))}>"


def _signature_types(signature: str) -> tuple[list[str], str]:
    tensor_matches = list(_TENSOR.finditer(signature))
    if not tensor_matches:
        raise FrontendImportError(f"shape specialization expected tensor signature, got '{signature}'")
    result_match = re.search(r"->\s*(tensor<[^>]+>)\s*$", signature)
    if result_match is not None:
        result_type = result_match.group(1)
        operand_text = signature[: result_match.start()].strip().strip("()")
        operand_types = [match.group(0) for match in _TENSOR.finditer(operand_text)]
        return operand_types, result_type
    return [], tensor_matches[-1].group(0)


def _replace_tensor_types(line: str, shapes: list[tuple[int, ...] | None]) -> str:
    cursor = 0
    index = 0
    pieces: list[str] = []
    for match in _TENSOR.finditer(line):
        pieces.append(line[cursor : match.start()])
        shape, dtype = _tensor_parts(match.group(0))
        replacement = match.group(0)
        if shape is None and index < len(shapes) and shapes[index] is not None:
            replacement = _format_tensor(shapes[index], dtype)
        pieces.append(replacement)
        cursor = match.end()
        index += 1
    pieces.append(line[cursor:])
    return "".join(pieces)


def _constant_values(body: str, shape: tuple[int, ...] | None) -> list[int] | None:
    dense = re.search(r"dense<([^>]+)>", body)
    if dense is None or shape is None:
        return None
    raw = dense.group(1).strip().strip("[]")
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            return None
    if len(values) == 1 and shape:
        values *= max(1, _product(shape))
    return values if len(values) == _product(shape) else None


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def _dimension_attribute(body: str) -> int:
    match = re.search(r"\bdim\s*=\s*(-?\d+)", body)
    if match is None:
        raise FrontendImportError("shape specialization get_dimension_size is missing dim")
    return int(match.group(1))


def _broadcast_dimensions(body: str) -> tuple[int, ...]:
    match = re.search(r"\bdims\s*=\s*\[([^\]]*)\]", body)
    if match is None:
        raise FrontendImportError("shape specialization dynamic broadcast is missing dims")
    values = tuple(int(item.strip()) for item in match.group(1).split(",") if item.strip())
    return values


def _slice_sizes(body: str) -> tuple[int, ...]:
    """Read the static output extents of a dynamic slice."""

    # StableHLO's custom assembly spells this attribute ``sizes``.  Keep the
    # older ``slice_sizes`` spelling for dependency-light fixtures.
    match = re.search(r"\b(?:sizes|slice_sizes)\s*=\s*\[([^\]]*)\]", body)
    if match is None:
        raise FrontendImportError("shape specialization dynamic slice is missing slice_sizes")
    values: list[int] = []
    for raw in match.group(1).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise FrontendImportError(
                "shape specialization dynamic slice sizes must be constant integers"
            ) from exc
        if value <= 0:
            raise FrontendImportError(
                "shape specialization dynamic slice sizes must be positive"
            )
        values.append(value)
    return tuple(values)


def _graph_argument_shapes(
    graph: OperatorGraph,
    shape_environment: Mapping[str, int],
) -> dict[str, tuple[int, ...]]:
    tensors = {tensor.name: tensor for tensor in graph.tensors}
    inputs = graph.attributes.get("graph_inputs", ())
    result: dict[str, tuple[int, ...]] = {}
    for name in inputs:
        tensor = tensors.get(str(name))
        if tensor is None:
            continue
        resolved: list[int] = []
        for value in tensor.shape:
            if isinstance(value, int):
                resolved.append(value)
            elif isinstance(value, str) and value in shape_environment:
                resolved.append(shape_environment[value])
            else:
                break
        else:
            result[str(name)] = tuple(resolved)
    return result


def _specialize_function_signature(
    line: str,
    argument_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
) -> str:
    shapes = [*argument_shapes, *output_shapes]
    return _replace_tensor_types(line, list(shapes))


def _specialize_operation(
    line: str,
    value_shapes: dict[str, tuple[int, ...]],
    shape_values: dict[str, list[int]],
    shape_only: set[str],
) -> tuple[str, str | None]:
    match = _OPERATION.match(line)
    if match is None:
        dynamic = _DYNAMIC_OP.search(line)
        if dynamic:
            raise FrontendImportError(
                f"shape specialization cannot parse dynamic StableHLO operation line: {line.strip()}"
            )
        return line, None
    result = match.group("result")
    target = match.group("target")
    body = match.group("body")
    signature = match.group("signature")
    operand_names = _VALUE.findall(body)
    operand_shapes = [value_shapes.get(name.removeprefix("%")) for name in operand_names]
    operand_shape_values = [shape_values.get(name.removeprefix("%")) for name in operand_names]
    operand_types, result_type = _signature_types(signature)
    result_shape, result_dtype = _tensor_parts(result_type)
    normalized_result = result.removeprefix("%")

    if target == "stablehlo.constant":
        if result_shape is not None:
            value_shapes[normalized_result] = result_shape
        if result_shape is not None and result_dtype in {"i32", "ui32", "i64", "ui64"}:
            values = _constant_values(body, result_shape)
            if values is not None:
                shape_values[normalized_result] = values
                shape_only.add(normalized_result)
        return line, None

    if target == "stablehlo.get_dimension_size":
        if not operand_names:
            raise FrontendImportError("shape specialization get_dimension_size has no operand")
        source_shape = value_shapes.get(operand_names[0].removeprefix("%"))
        dimension = _dimension_attribute(body)
        if source_shape is None or dimension < 0 or dimension >= len(source_shape):
            raise FrontendImportError(
                f"shape specialization cannot resolve {result}: source shape/dim is unknown"
            )
        shape_values[normalized_result] = [source_shape[dimension]]
        shape_only.add(normalized_result)
        return line, normalized_result

    if target == "stablehlo.reshape" and operand_names and operand_shape_values[0] is not None:
        shape_values[normalized_result] = list(operand_shape_values[0])
        shape_only.add(normalized_result)
        return line, normalized_result

    if target == "stablehlo.concatenate" and all(value is not None for value in operand_shape_values):
        dimension_match = re.search(r"\bdim\s*=\s*(-?\d+)", body)
        dimension = int(dimension_match.group(1)) if dimension_match else 0
        if dimension != 0:
            raise FrontendImportError("shape specialization only supports shape-vector concatenate dim=0")
        shape_values[normalized_result] = [item for value in operand_shape_values for item in value or ()]
        shape_only.add(normalized_result)
        return line, normalized_result

    if target == "stablehlo.maximum" and all(value is not None for value in operand_shape_values):
        values = [value for value in operand_shape_values if value is not None]
        if not values or any(len(value) != len(values[0]) for value in values):
            raise FrontendImportError(f"shape specialization cannot evaluate maximum '{result}'")
        shape_values[normalized_result] = [max(items) for items in zip(*values)]
        shape_only.add(normalized_result)
        return line, normalized_result

    if target == "stablehlo.dynamic_broadcast_in_dim":
        if len(operand_names) < 2:
            raise FrontendImportError(f"shape specialization dynamic broadcast '{result}' needs data and shape operands")
        source_name = operand_names[0].removeprefix("%")
        target_values = shape_values.get(operand_names[1].removeprefix("%"))
        source_shape = value_shapes.get(source_name)
        if source_shape is None or target_values is None:
            raise FrontendImportError(f"shape specialization cannot resolve dynamic broadcast '{result}'")
        output_shape = tuple(int(value) for value in target_values)
        dimensions = _broadcast_dimensions(body)
        if len(dimensions) != len(source_shape) or any(
            axis < 0 or axis >= len(output_shape) for axis in dimensions
        ):
            raise FrontendImportError(f"shape specialization dynamic broadcast '{result}' has invalid dims")
        for source_axis, output_axis in enumerate(dimensions):
            if source_shape[source_axis] not in {1, output_shape[output_axis]}:
                raise FrontendImportError(f"shape specialization dynamic broadcast '{result}' has incompatible extent")
        value_shapes[normalized_result] = output_shape
        source_type = _format_tensor(source_shape, _tensor_parts(operand_types[0])[1])
        output_type = _format_tensor(output_shape, result_dtype)
        rewritten = (
            f"{match.group('indent')}{result} = stablehlo.broadcast_in_dim "
            f"{operand_names[0]}, dims = [{', '.join(map(str, dimensions))}] : "
            f"({source_type}) -> {output_type}"
        )
        return rewritten, normalized_result

    if target == "stablehlo.dynamic_reshape":
        if len(operand_names) < 2:
            raise FrontendImportError(
                f"shape specialization dynamic reshape '{result}' needs data and shape operands"
            )
        source_name = operand_names[0].removeprefix("%")
        source_shape = value_shapes.get(source_name)
        target_values = shape_values.get(operand_names[1].removeprefix("%"))
        if source_shape is None or target_values is None:
            raise FrontendImportError(
                f"shape specialization cannot resolve dynamic reshape '{result}'"
            )
        output_shape = tuple(int(value) for value in target_values)
        if not output_shape or any(value <= 0 for value in output_shape):
            raise FrontendImportError(
                f"shape specialization dynamic reshape '{result}' requires positive output dimensions"
            )
        if _product(source_shape) != _product(output_shape):
            raise FrontendImportError(
                f"shape specialization dynamic reshape '{result}' changes element count "
                f"from {_product(source_shape)} to {_product(output_shape)}"
            )
        value_shapes[normalized_result] = output_shape
        source_type = _format_tensor(source_shape, result_dtype)
        output_type = _format_tensor(output_shape, result_dtype)
        rewritten = (
            f"{match.group('indent')}{result} = stablehlo.reshape "
            f"{operand_names[0]} : ({source_type}) -> {output_type}"
        )
        return rewritten, normalized_result

    if target == "stablehlo.dynamic_slice":
        if not operand_names:
            raise FrontendImportError(f"shape specialization dynamic slice '{result}' has no data operand")
        source_name = operand_names[0].removeprefix("%")
        source_shape = value_shapes.get(source_name)
        raw_start_values = [shape_values.get(name.removeprefix("%")) for name in operand_names[1:]]
        if source_shape is None:
            raise FrontendImportError(
                f"shape specialization cannot resolve dynamic slice '{result}' source shape"
            )
        sizes = _slice_sizes(body)
        if len(sizes) != len(source_shape):
            raise FrontendImportError(
                f"shape specialization dynamic slice '{result}' rank does not match operand"
            )
        if len(raw_start_values) == len(source_shape) and all(
            values is not None and len(values) == 1 for values in raw_start_values
        ):
            start_values = [int(values[0]) for values in raw_start_values if values is not None]
        else:
            # Preserve the operation and its symbolic index operands.  The
            # canonical importer records DynamicIndexExpr and runtime binds
            # the values after descriptor submission.
            value_shapes[normalized_result] = tuple(int(value) for value in sizes)
            return line, None
        if len(start_values) != len(source_shape):
            raise FrontendImportError(
                f"shape specialization dynamic slice '{result}' requires one start index per axis"
            )
        starts: list[int] = []
        limits: list[int] = []
        for axis, (extent, size, start_value) in enumerate(zip(source_shape, sizes, start_values)):
            if not isinstance(extent, int):
                raise FrontendImportError(
                    f"shape specialization dynamic slice '{result}' requires resolved operand shape"
                )
            if size > extent:
                raise FrontendImportError(
                    f"shape specialization dynamic slice '{result}' size {size} exceeds axis {axis} extent {extent}"
                )
            # StableHLO clamps dynamic starts to the largest legal start.
            start = max(0, min(int(start_value), extent - size))
            starts.append(start)
            limits.append(start + size)
        source_type = _format_tensor(source_shape, _tensor_parts(operand_types[0])[1])
        output_type = _format_tensor(sizes, result_dtype)
        value_shapes[normalized_result] = sizes
        ranges = ", ".join(f"{start}:{limit}" for start, limit in zip(starts, limits))
        rewritten = (
            f"{match.group('indent')}{result} = stablehlo.slice "
            f"{operand_names[0]} [{ranges}] : ({source_type}) -> {output_type}"
        )
        return rewritten, normalized_result

    if target == "stablehlo.dynamic_update_slice":
        if len(operand_names) < 3 or operand_shapes[0] is None:
            raise FrontendImportError(
                f"shape specialization dynamic update slice '{result}' needs resolved operand shape"
            )
        if result_shape is None:
            result_shape = tuple(int(value) for value in operand_shapes[0])
        if tuple(result_shape) != tuple(operand_shapes[0]):
            raise FrontendImportError(
                f"shape specialization dynamic update slice '{result}' result shape must match operand"
            )
        value_shapes[normalized_result] = tuple(result_shape)
        return line, None

    if target.startswith("stablehlo.dynamic_"):
        raise FrontendImportError(
            f"shape specialization does not support dynamic operation '{target}'"
        )

    if operand_shapes and all(shape is not None for shape in operand_shapes):
        first_shape = operand_shapes[0]
        if result_shape is None and first_shape is not None:
            value_shapes[normalized_result] = first_shape
            rewritten = _replace_tensor_types(line, [*operand_shapes, first_shape])
            return rewritten, None
    if result_shape is not None:
        value_shapes[normalized_result] = result_shape
    return _replace_tensor_types(line, [*operand_shapes, result_shape]), None


def specialize_stablehlo(
    text: str,
    source_graph: OperatorGraph,
    *,
    shape_environment: Mapping[str, int] | None = None,
) -> ShapeSpecializationResult:
    """Specialize the supported Torch-XLA dynamic shape dataflow."""

    if not text.strip():
        raise FrontendImportError("shape specialization requires non-empty StableHLO text")
    normalized_environment = normalize_shape_environment(shape_environment)
    argument_shapes = _graph_argument_shapes(source_graph, normalized_environment)
    graph_inputs = tuple(str(item) for item in source_graph.attributes.get("graph_inputs", ()))
    graph_outputs = tuple(str(item) for item in source_graph.attributes.get("graph_outputs", ()))
    tensors = {tensor.name: tensor for tensor in source_graph.tensors}
    output_shapes = tuple(
        tuple(
            int(value) if isinstance(value, int) else normalized_environment[str(value)]
            for value in tensors[name].shape
        )
        for name in graph_outputs
        if name in tensors
        and all(
            isinstance(value, int)
            or (isinstance(value, str) and value in normalized_environment)
            for value in tensors[name].shape
        )
    )
    value_shapes = dict(argument_shapes)
    shape_values: dict[str, list[int]] = {}
    shape_only: set[str] = set()
    dynamic_operations: set[str] = set()
    rewritten_lines: list[str] = []
    removed_shape_operations: list[str] = []
    lines = text.splitlines()
    for line in lines:
        dynamic_operations.update(_DYNAMIC_OP.findall(line))
        if "func.func" in line:
            argument_names = [
                name.removeprefix("%")
                for name in _VALUE.findall(line.split(")", 1)[0])
            ]
            for stable_name, source_name in zip(argument_names, graph_inputs):
                shape = argument_shapes.get(source_name)
                if shape is not None:
                    value_shapes[stable_name] = shape
            rewritten_lines.append(
                _specialize_function_signature(
                    line,
                    tuple(argument_shapes.get(name, ()) for name in graph_inputs),
                    output_shapes,
                )
            )
            continue
        if re.match(r"\s*(?:func\.)?return\b", line):
            returned_shapes = [
                value_shapes.get(name.removeprefix("%"))
                for name in _VALUE.findall(line.split(":", 1)[0])
            ]
            rewritten_lines.append(_replace_tensor_types(line, returned_shapes))
            continue
        rewritten, removed = _specialize_operation(line, value_shapes, shape_values, shape_only)
        rewritten_lines.append(rewritten)

    # Remove shape-only operations after dynamic broadcasts no longer consume
    # their shape operands.  Iterate because a concatenate may only be used by
    # a now-dead maximum, for example.
    kept = list(rewritten_lines)
    changed = True
    while changed:
        changed = False
        used = {
            name.removeprefix("%")
            for line in kept
            for name in _VALUE.findall(line)
        }
        next_lines: list[str] = []
        for line in kept:
            match = _OPERATION.match(line)
            if match is not None:
                result = match.group("result").removeprefix("%")
                if result in shape_only and result not in used - {result}:
                    removed_shape_operations.append(result)
                    changed = True
                    continue
            next_lines.append(line)
        kept = next_lines

    specialized_text = "\n".join(kept).rstrip() + "\n"
    return ShapeSpecializationResult(
        text=specialized_text,
        dynamic_operations=tuple(sorted(dynamic_operations)),
        removed_shape_operations=tuple(sorted(set(removed_shape_operations))),
        shape_environment=normalized_environment,
    )


__all__ = ["ShapeSpecializationResult", "specialize_stablehlo"]
