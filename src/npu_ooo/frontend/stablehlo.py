from __future__ import annotations

"""Internal semantic parser for verified StableHLO projections.

The official adapter owns external MLIR parsing and verification.  This module
only turns the verified, project-supported operation projection into the
Canonical OperatorGraph. Unsupported constructs fail at this boundary instead
of being silently dropped.
"""

import re
from dataclasses import replace
from typing import Any, Mapping

from npu_ooo.ir import (
    DataEdge,
    DynamicIndexExpr,
    ModelFamily,
    OperatorGraph,
    OperatorSpec,
    SemanticOpType,
    TensorSpec,
)

from .bridge import (
    FrontendImport,
    FrontendImportError,
    FrontendKind,
    normalize_shape_environment,
)
from .stablehlo_semantics import (
    normalize_stablehlo_op_name,
    registered_stablehlo_ops,
    stablehlo_capability,
)


_TENSOR_TYPE = re.compile(r"tensor<([^>]+)>")
_VALUE_NAME = re.compile(r"%[A-Za-z_0-9][\w.$-]*")
_FUNCTION = re.compile(r"func\.func\s+@(?P<name>[\w$.-]+)\s*\((?P<args>[^)]*)\)")
_ARGUMENT = re.compile(r"(?P<name>%[A-Za-z_][\w.$-]*)\s*:\s*(?P<type>tensor<[^>]+>)")
_RESULT = re.compile(
    r"^\s*(?P<result>%[A-Za-z_0-9][\w.$-]*)\s*=\s*"
    r"(?P<target>(?:stablehlo|mhlo)\.[A-Za-z_][\w.]*)\s*"
    r"(?P<body>.*?)\s*:\s*(?P<types>tensor<[^>]+>|\([^\n]*\)\s*->\s*tensor<[^>]+>)\s*$"
)
_RETURN = re.compile(r"^\s*(?:func\.)?return\s+(?P<values>[^:]+)")


def _tensor_type(text: str) -> tuple[tuple[int | str, ...], str]:
    match = _TENSOR_TYPE.search(text)
    if match is None:
        raise FrontendImportError(f"StableHLO value is missing a tensor type: {text.strip()}")
    # Tensor encodings follow the element type after a comma, for example
    # ``tensor<2x3xf32, #layout>``.  They are metadata, not part of dtype.
    shape_payload = match.group(1).split(",", 1)[0].strip()
    parts = [part.strip() for part in shape_payload.split("x") if part.strip()]
    if not parts:
        raise FrontendImportError(f"StableHLO tensor type has no shape: {text.strip()}")
    dtype = parts[-1]
    dimensions: list[int | str] = []
    for index, part in enumerate(parts[:-1]):
        if part in {"?", "*"}:
            dimensions.append(f"D{index}")
            continue
        try:
            value = int(part)
        except ValueError:
            if not part or not part.replace("_", "a").isalnum():
                raise FrontendImportError(f"unsupported StableHLO dimension '{part}'")
            dimensions.append(part)
        else:
            if value <= 0:
                raise FrontendImportError(f"StableHLO dimensions must be positive, got {value}")
            dimensions.append(value)
    return tuple(dimensions), dtype


def _tensor_layout_encoding(text: str) -> str | None:
    """Return a StableHLO tensor encoding without interpreting affine maps."""

    match = _TENSOR_TYPE.search(text)
    if match is None or "," not in match.group(1):
        return None
    encoding = match.group(1).split(",", 1)[1].strip()
    return encoding or None


def _tensor_attributes(type_text: str, *, source_kind: str, source_node: str, frontend_target: str) -> dict[str, Any]:
    encoding = _tensor_layout_encoding(type_text)
    attributes: dict[str, Any] = {
        "source_kind": source_kind,
        "source_node": source_node,
        "frontend_target": frontend_target,
        "layout_source": "stablehlo_encoding" if encoding else "default_dense",
    }
    if encoding:
        attributes["layout_encoding"] = encoding
    return attributes


def _result_tensor_type(type_signature: str) -> tuple[tuple[int | str, ...], str]:
    """Read the result type instead of the first operand type in a signature."""

    result_text = type_signature.rsplit("->", 1)[-1]
    return _tensor_type(result_text)


def _dimensions(text: str, rank: int) -> tuple[int, ...]:
    match = re.search(r"dimensions\s*=\s*(?:dense<)?\[([^\]]*)\]", text)
    if match is None:
        return (rank - 1,)
    values: list[int] = []
    for raw in match.group(1).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise FrontendImportError(f"StableHLO reduction dimension is not constant: {raw}") from exc
        if value < 0 or value >= rank:
            raise FrontendImportError(f"StableHLO reduction dimension {value} is outside rank {rank}")
        values.append(value)
    return tuple(values or (rank - 1,))


def _dimension_list(text: str, name: str) -> tuple[int, ...] | None:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*\[([^\]]*)\]", text)
    if match is None:
        return None
    values: list[int] = []
    for raw in match.group(1).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            values.append(int(raw))
        except ValueError as exc:
            raise FrontendImportError(
                f"StableHLO dot dimension '{name}' is not constant: {raw}"
            ) from exc
    return tuple(values)


def _reducer(text: str) -> str | None:
    """Read the dependency-light reducer marker emitted by our generator."""

    match = re.search(r"\breducer\s*=\s*([A-Za-z_][\w.]*)", text)
    return match.group(1).lower() if match else None


def _named_integer(text: str, name: str) -> int | None:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(-?\d+)", text)
    return int(match.group(1)) if match else None


def _named_integer_list(text: str, name: str) -> tuple[int, ...] | None:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*\[([^\]]*)\]", text)
    if match is None:
        return None
    values: list[int] = []
    for raw in match.group(1).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            values.append(int(raw))
        except ValueError as exc:
            raise FrontendImportError(
                f"StableHLO {name} must contain constant integers"
            ) from exc
    return tuple(values)


def _named_float(text: str, name: str) -> float | None:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)",
        text,
    )
    return float(match.group(1)) if match else None


def _transpose_dimensions(text: str, rank: int) -> tuple[int, ...] | None:
    values = _dimension_list(text, "dimensions")
    if values is None:
        return None
    if len(values) != rank or sorted(values) != list(range(rank)):
        raise FrontendImportError(
            f"StableHLO transpose dimensions must be a permutation of rank {rank}"
        )
    return values


def _dot_dimension_numbers(
    text: str,
    lhs_shape: tuple[int | str, ...],
    rhs_shape: tuple[int | str, ...],
) -> dict[str, tuple[int, ...]]:
    """Parse the common textual forms of StableHLO dot dimension numbers."""

    compact_contracting = re.search(
        r"\bcontracting_dims\s*=\s*\[([^\]]*)\]\s*x\s*\[([^\]]*)\]",
        text,
    )
    compact_batching = re.search(
        r"\bbatching_dims\s*=\s*\[([^\]]*)\]\s*x\s*\[([^\]]*)\]",
        text,
    )

    def compact_values(match: re.Match[str] | None, group: int) -> tuple[int, ...] | None:
        if match is None:
            return None
        raw_values = match.group(group).strip()
        if not raw_values:
            return ()
        try:
            return tuple(int(item.strip()) for item in raw_values.split(",") if item.strip())
        except ValueError as exc:
            raise FrontendImportError("StableHLO dot dimensions must be constant integers") from exc

    lhs_contracting = compact_values(compact_contracting, 1)
    rhs_contracting = compact_values(compact_contracting, 2)
    lhs_batching = compact_values(compact_batching, 1)
    rhs_batching = compact_values(compact_batching, 2)

    if lhs_contracting is None:
        lhs_contracting = _dimension_list(text, "lhs_contracting_dimensions")
    if rhs_contracting is None:
        rhs_contracting = _dimension_list(text, "rhs_contracting_dimensions")
    if lhs_batching is None:
        lhs_batching = _dimension_list(text, "lhs_batching_dimensions")
    if rhs_batching is None:
        rhs_batching = _dimension_list(text, "rhs_batching_dimensions")

    # The dependency-light fixture form may omit the attribute.  Use the
    # conventional rank-2 matmul contract, but record it explicitly below.
    if lhs_contracting is None and rhs_contracting is None:
        lhs_contracting = (len(lhs_shape) - 1,)
        rhs_contracting = (len(rhs_shape) - 2,)
    if lhs_contracting is None or rhs_contracting is None:
        raise FrontendImportError("StableHLO dot must define both lhs and rhs contracting dimensions")
    lhs_batching = lhs_batching or ()
    rhs_batching = rhs_batching or ()
    if len(lhs_contracting) != len(rhs_contracting):
        raise FrontendImportError("StableHLO dot lhs/rhs contracting dimension counts differ")
    if len(lhs_batching) != len(rhs_batching):
        raise FrontendImportError("StableHLO dot lhs/rhs batching dimension counts differ")

    pairs = (
        ("lhs_contracting_dimensions", lhs_contracting, lhs_shape),
        ("rhs_contracting_dimensions", rhs_contracting, rhs_shape),
        ("lhs_batching_dimensions", lhs_batching, lhs_shape),
        ("rhs_batching_dimensions", rhs_batching, rhs_shape),
    )
    for label, dimensions, shape in pairs:
        for dimension in dimensions:
            if dimension < 0 or dimension >= len(shape):
                raise FrontendImportError(
                    f"StableHLO dot {label} contains {dimension}, outside rank {len(shape)}"
                )
    for lhs_dimension, rhs_dimension in zip(lhs_contracting, rhs_contracting):
        if lhs_shape[lhs_dimension] != rhs_shape[rhs_dimension]:
            raise FrontendImportError(
                "StableHLO dot contracting extents differ: "
                f"lhs dim {lhs_dimension}={lhs_shape[lhs_dimension]}, "
                f"rhs dim {rhs_dimension}={rhs_shape[rhs_dimension]}"
            )
    for lhs_dimension, rhs_dimension in zip(lhs_batching, rhs_batching):
        if lhs_shape[lhs_dimension] != rhs_shape[rhs_dimension]:
            raise FrontendImportError(
                "StableHLO dot batching extents differ: "
                f"lhs dim {lhs_dimension}={lhs_shape[lhs_dimension]}, "
                f"rhs dim {rhs_dimension}={rhs_shape[rhs_dimension]}"
            )
    return {
        "lhs_contracting_dimensions": lhs_contracting,
        "rhs_contracting_dimensions": rhs_contracting,
        "lhs_batching_dimensions": lhs_batching,
        "rhs_batching_dimensions": rhs_batching,
    }


def _constant_value(text: str) -> float | int | bool | None:
    dense = re.search(r"dense<([^>]+)>", text)
    raw = dense.group(1).strip() if dense else text.strip()
    raw = raw.strip("[]")
    if "," in raw:
        raw = raw.split(",", 1)[0].strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw.replace("e", "E"))
        except ValueError:
            return None


def _constant_args(body: str, constants: Mapping[str, Any]) -> list[Any]:
    values: list[Any] = []
    for name in _VALUE_NAME.findall(body):
        if name in constants:
            values.append(constants[name])
    scalar_text = re.sub(r"%[A-Za-z_0-9][\w.$-]*", " ", body)
    for raw in re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", scalar_text):
        values.append(_constant_value(raw))
    return values


def _graph_from_text(text: str, *, graph_id: str) -> OperatorGraph:
    function = _FUNCTION.search(text)
    if function is None:
        raise FrontendImportError("StableHLO text must contain a func.func @name entry point")
    graph_inputs: list[str] = []
    tensors: dict[str, TensorSpec] = {}
    for argument in _ARGUMENT.finditer(function.group("args")):
        type_text = argument.group("type")
        shape, dtype = _tensor_type(type_text)
        name = argument.group("name").removeprefix("%")
        tensors[name] = TensorSpec(
            name=name,
            shape=shape,
            dtype=dtype,
            attributes={
                **_tensor_attributes(
                    type_text,
                    source_kind="input",
                    source_node=name,
                    frontend_target="",
                ),
                "frontend": "stablehlo",
            },
        )
        graph_inputs.append(name)

    constants: dict[str, Any] = {}
    operators: list[OperatorSpec] = []
    produced_by: dict[str, str] = {}
    graph_outputs: list[str] = []
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        return_match = _RETURN.match(stripped)
        if return_match:
            graph_outputs.extend(_VALUE_NAME.findall(return_match.group("values")))
            continue
        match = _RESULT.match(line)
        if match is None:
            if "stablehlo." in line and "=" in line:
                raise FrontendImportError(f"unsupported StableHLO operation syntax: {stripped}")
            continue
        result_name = match.group("result")
        target = match.group("target")
        body = match.group("body")
        type_signature = match.group("types")
        result_type_text = type_signature.rsplit("->", 1)[-1].strip()
        result_shape, result_dtype = _result_tensor_type(type_signature)
        if target.endswith(".constant"):
            constants[result_name] = _constant_value(body)
            normalized_name = result_name.removeprefix("%")
            # Keep scalar constants as rank-0 tensors.  The empty tuple is a
            # valid StableHLO shape; checking for ``is not None`` avoids
            # silently dropping ``tensor<f32>`` operands from later ops.
            if result_shape is not None:
                tensors[normalized_name] = TensorSpec(
                    name=normalized_name,
                    shape=result_shape,
                    dtype=result_dtype,
                    attributes={
                        **_tensor_attributes(
                            result_type_text,
                            source_kind="constant",
                            source_node=normalized_name,
                            frontend_target=target,
                        ),
                        "constant_value": constants[result_name],
                    },
                )
            continue
        all_values = _VALUE_NAME.findall(body)
        operand_arity = len(all_values)
        normalized_target = normalize_stablehlo_op_name(target)
        capability = stablehlo_capability(normalized_target)
        if capability is None:
            raise FrontendImportError(
                f"missing StableHLO capability for '{target}' "
                f"(normalized as '{normalized_target}') at the import boundary; "
                "register a StableHLOOpCapability entry before compiling it. "
                "Known operations: "
                + ", ".join(registered_stablehlo_ops())
            )
        if not capability.supports_arity(operand_arity):
            expected = (
                str(capability.min_operands)
                if capability.min_operands == capability.max_operands
                else f"{capability.min_operands}..{capability.max_operands}"
            )
            raise FrontendImportError(
                f"StableHLO operation '{target}' has {operand_arity} operands; "
                f"the import capability expects {expected}"
            )
        input_occurrences = tuple(
            name.removeprefix("%")
            for name in all_values
            if name.removeprefix("%") in tensors or name.removeprefix("%") in produced_by
        )
        input_names = tuple(dict.fromkeys(input_occurrences))
        if not input_names:
            raise FrontendImportError(
                f"StableHLO operation '{target}' has no tensor operands in supported form"
            )
        op_type = capability.semantic_family
        input_shapes = [tensors[name.removeprefix("%")].shape for name in input_names]
        operation_attributes: dict[str, Any] = {}
        if normalized_target == "stablehlo.dynamic_slice":
            source_name = input_names[0].removeprefix("%")
            source_shape = input_shapes[0]
            sizes = _named_integer_list(body, "sizes") or _named_integer_list(body, "slice_sizes")
            if sizes is None or len(sizes) != len(source_shape):
                raise FrontendImportError(
                    f"StableHLO dynamic_slice '{result_name}' requires one positive size per source axis"
                )
            if any(size <= 0 for size in sizes):
                raise FrontendImportError(
                    f"StableHLO dynamic_slice '{result_name}' sizes must be positive"
                )
            index_names = input_names[1:]
            if len(index_names) != len(source_shape):
                raise FrontendImportError(
                    f"StableHLO dynamic_slice '{result_name}' requires one index operand per source axis"
                )
            operation_attributes["dynamic_index"] = DynamicIndexExpr(
                expression_id=f"{result_name.removeprefix('%')}.index",
                source_tensor=source_name,
                index_operands=tuple(name.removeprefix("%") for name in index_names),
                index_rank=len(source_shape),
                clamp_bounds=tuple(
                    (0, max(0, int(extent) - size)) if isinstance(extent, int) else (0, None)
                    for extent, size in zip(source_shape, sizes)
                ),
                attributes={"operation": "dynamic_slice", "sizes": list(sizes)},
            ).to_dict()
            operation_attributes.update(
                {
                    "slice_sizes": list(sizes),
                    "dynamic_index_operands": [name.removeprefix("%") for name in index_names],
                }
            )
            # Index operands belong to DynamicIndexExpr metadata.  The
            # canonical dataflow input remains the sliced tensor.
            input_names = (input_names[0],)
            iteration_dims = tuple((f"d{axis}", value) for axis, value in enumerate(result_shape))
            reduction_dims = ()
        elif normalized_target == "stablehlo.dynamic_update_slice":
            if len(input_names) < 3:
                raise FrontendImportError(
                    f"StableHLO dynamic_update_slice '{result_name}' requires operand, update and index vector"
                )
            source_name = input_names[0].removeprefix("%")
            update_name = input_names[1].removeprefix("%")
            source_shape = input_shapes[0]
            update_shape = input_shapes[1]
            if tuple(result_shape) != tuple(source_shape):
                raise FrontendImportError(
                    f"StableHLO dynamic_update_slice '{result_name}' result shape must match operand"
                )
            if len(update_shape) != len(source_shape):
                raise FrontendImportError(
                    f"StableHLO dynamic_update_slice '{result_name}' update rank must match operand rank"
                )
            operation_attributes["dynamic_index"] = DynamicIndexExpr(
                expression_id=f"{result_name.removeprefix('%')}.index",
                source_tensor=source_name,
                index_operands=tuple(name.removeprefix("%") for name in input_names[2:]),
                index_rank=len(source_shape),
                clamp_bounds=tuple(
                    (0, max(0, int(extent) - int(update_extent)))
                    if isinstance(extent, int) and isinstance(update_extent, int)
                    else (0, None)
                    for extent, update_extent in zip(source_shape, update_shape)
                ),
                attributes={
                    "operation": "dynamic_update_slice",
                    "update_tensor": update_name,
                    "update_shape": list(update_shape),
                },
            ).to_dict()
            operation_attributes.update(
                {
                    "state_update": True,
                    "stateful": True,
                    "state_id": source_name,
                    "state_buffer": source_name,
                    "update_tensor": update_name,
                    "dynamic_index_operands": [name.removeprefix("%") for name in input_names[2:]],
                }
            )
            source_tensor = tensors[source_name]
            tensors[source_name] = replace(
                source_tensor,
                attributes={
                    **dict(source_tensor.attributes),
                    "persistent": True,
                    "state_id": source_name,
                    "state_buffer": source_name,
                },
            )
            input_names = (input_names[0], input_names[1])
            iteration_dims = tuple((f"d{axis}", value) for axis, value in enumerate(result_shape))
            reduction_dims = ()
        elif normalized_target == "stablehlo.convert":
            source_shape = input_shapes[0]
            if source_shape != result_shape:
                raise FrontendImportError(
                    f"StableHLO convert operation '{result_name}' changes shape from "
                    f"{source_shape} to {result_shape}; shape-changing convert is invalid"
                )
            operation_attributes.update(
                {
                    "conversion": True,
                    "source_dtype": tensors[input_names[0]].dtype,
                    "target_dtype": result_dtype,
                    "conversion_kind": "dtype_cast",
                }
            )
            iteration_dims = tuple((f"d{axis}", value) for axis, value in enumerate(result_shape))
            reduction_dims = ()
        elif op_type == SemanticOpType.BATCH_NORM.value:
            if len(input_shapes) != 5 or len(result_shape) != 4 or any(len(shape) != 1 for shape in input_shapes[1:]):
                raise FrontendImportError(
                    f"StableHLO batch_norm_inference '{result_name}' requires rank-4 input and four rank-1 statistics"
                )
            if input_shapes[1] != input_shapes[2] or input_shapes[1] != input_shapes[3] or input_shapes[1] != input_shapes[4]:
                raise FrontendImportError(
                    f"StableHLO batch_norm_inference '{result_name}' statistics must have matching channel shapes"
                )
            feature_index = _named_integer(body, "feature_index")
            epsilon = _named_float(body, "epsilon")
            if feature_index is None or feature_index < 0 or feature_index >= len(result_shape):
                raise FrontendImportError(
                    f"StableHLO batch_norm_inference '{result_name}' has an invalid feature_index"
                )
            if input_shapes[1][0] != result_shape[feature_index]:
                raise FrontendImportError(
                    f"StableHLO batch_norm_inference '{result_name}' statistics do not match feature extent"
                )
            operation_attributes.update(
                {
                    "feature_index": feature_index,
                    "epsilon": epsilon if epsilon is not None else 1e-5,
                    "training": False,
                }
            )
            iteration_dims = tuple((f"d{axis}", value) for axis, value in enumerate(result_shape))
            reduction_dims = ()
        elif op_type == SemanticOpType.POOL.value:
            if len(input_shapes) < 1 or len(input_shapes[0]) != 4 or len(result_shape) != 4:
                raise FrontendImportError(
                    f"StableHLO reduce_window '{result_name}' currently requires rank-4 NCHW tensors"
                )
            # The init value is a scalar constant and is not a data operand in
            # the canonical pool contract.
            input_names = (input_names[0],)
            input_shape = input_shapes[0]
            window_dimensions = _named_integer_list(body, "window_dimensions")
            window_strides = _named_integer_list(body, "window_strides")
            padding = _named_integer_list(body, "padding")
            base_dilations = _named_integer_list(body, "base_dilations") or (1, 1, 1, 1)
            window_dilations = _named_integer_list(body, "window_dilations") or (1, 1, 1, 1)
            reducer = re.search(r"\breducer\s*=\s*([A-Za-z_][\w.]*)", body)
            reducer_name = reducer.group(1).lower() if reducer else "add"
            if window_dimensions is None or len(window_dimensions) != 4:
                raise FrontendImportError(f"StableHLO reduce_window '{result_name}' requires four window_dimensions")
            if window_strides is None or len(window_strides) != 4:
                raise FrontendImportError(f"StableHLO reduce_window '{result_name}' requires four window_strides")
            if padding is None or len(padding) != 8:
                raise FrontendImportError(f"StableHLO reduce_window '{result_name}' requires eight padding values")
            if base_dilations != (1, 1, 1, 1) or window_dilations != (1, 1, 1, 1):
                raise FrontendImportError(f"StableHLO reduce_window '{result_name}' currently requires unit dilations")
            if window_dimensions[:2] != (1, 1) or window_strides[:2] != (1, 1) or any(padding[index] for index in (0, 1, 2, 3)):
                raise FrontendImportError(f"StableHLO reduce_window '{result_name}' currently requires N/C-preserving windows")
            if reducer_name not in {"maximum", "add"}:
                raise FrontendImportError(
                    f"StableHLO reduce_window '{result_name}' reducer '{reducer_name}' is unsupported"
                )
            expected_spatial = tuple(
                (input_shape[axis + 2] + padding[2 * (axis + 2)] + padding[2 * (axis + 2) + 1] - window_dimensions[axis + 2])
                // window_strides[axis + 2] + 1
                for axis in range(2)
            )
            if tuple(result_shape) != (input_shape[0], input_shape[1], *expected_spatial):
                raise FrontendImportError(
                    f"StableHLO reduce_window '{result_name}' result shape {result_shape} does not match "
                    f"window/padding (expected {(input_shape[0], input_shape[1], *expected_spatial)})"
                )
            operation_attributes.update(
                {
                    "window_dimensions": list(window_dimensions),
                    "window_strides": list(window_strides),
                    "padding": list(padding),
                    "base_dilations": list(base_dilations),
                    "window_dilations": list(window_dilations),
                    "pool_reducer": reducer_name,
                }
            )
            iteration_dims = tuple((f"d{axis}", value) for axis, value in enumerate(result_shape))
            reduction_dims = ()
        elif op_type == SemanticOpType.CONV2D.value:
            if len(input_shapes) != 2 or len(input_shapes[0]) != 4 or len(input_shapes[1]) != 4 or len(result_shape) != 4:
                raise FrontendImportError(
                    f"StableHLO convolution '{result_name}' currently requires rank-4 NCHW/OIHW tensors"
                )
            lhs_shape, rhs_shape = input_shapes
            dimension_numbers = re.search(
                r"dimension_numbers\s*=\s*([^\s]+(?:\s*[^\s]+)*?)\s+(?:window_strides|strides)\s*=",
                body,
            )
            dimensions_text = dimension_numbers.group(1) if dimension_numbers else ""
            if dimensions_text and not (
                "[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]" in dimensions_text
            ):
                raise FrontendImportError(
                    f"StableHLO convolution '{result_name}' uses unsupported dimension_numbers "
                    f"'{dimensions_text}'"
                )
            strides = _named_integer_list(body, "window_strides")
            if strides is None:
                strides = _named_integer_list(body, "strides")
            padding = _named_integer_list(body, "padding")
            lhs_dilation = _named_integer_list(body, "lhs_dilation")
            rhs_dilation = _named_integer_list(body, "rhs_dilation")
            feature_groups = _named_integer(body, "feature_group_count") or 1
            batch_groups = _named_integer(body, "batch_group_count") or 1
            if strides is None or len(strides) != 2:
                raise FrontendImportError(
                    f"StableHLO convolution '{result_name}' requires two window_strides"
                )
            if padding is None or len(padding) != 4:
                raise FrontendImportError(
                    f"StableHLO convolution '{result_name}' requires [h_low,h_high,w_low,w_high] padding"
                )
            lhs_dilation = lhs_dilation or (1, 1)
            rhs_dilation = rhs_dilation or (1, 1)
            if lhs_dilation != (1, 1) or rhs_dilation != (1, 1):
                raise FrontendImportError(
                    f"StableHLO convolution '{result_name}' currently requires unit dilation"
                )
            if any(value <= 0 for value in (*strides, feature_groups, batch_groups)):
                raise FrontendImportError(
                    f"StableHLO convolution '{result_name}' has invalid stride/group attributes"
                )
            if feature_groups != 1 or batch_groups != 1:
                raise FrontendImportError(
                    f"StableHLO convolution '{result_name}' currently requires feature_group_count=batch_group_count=1"
                )
            if lhs_shape[1] != rhs_shape[1] or result_shape[0] != lhs_shape[0] or result_shape[1] != rhs_shape[0]:
                raise FrontendImportError(
                    f"StableHLO convolution '{result_name}' has incompatible channel/batch shapes"
                )
            expected_spatial = tuple(
                (lhs_shape[axis + 2] + padding[2 * axis] + padding[2 * axis + 1] - rhs_shape[axis + 2]) // strides[axis] + 1
                for axis in range(2)
            )
            if tuple(result_shape[2:]) != expected_spatial:
                raise FrontendImportError(
                    f"StableHLO convolution '{result_name}' result shape {result_shape} does not match "
                    f"window/padding (expected {(lhs_shape[0], rhs_shape[0], *expected_spatial)})"
                )
            iteration_dims = (
                ("N", result_shape[0]),
                ("O", result_shape[1]),
                ("OH", result_shape[2]),
                ("OW", result_shape[3]),
            )
            reduction_dims = (("K", rhs_shape[1] * rhs_shape[2] * rhs_shape[3]),)
            operation_attributes.update(
                {
                    "convolution_dimension_numbers": "nchw_oihw_nchw",
                    "window_strides": list(strides),
                    "padding": list(padding),
                    "lhs_dilation": list(lhs_dilation),
                    "rhs_dilation": list(rhs_dilation),
                    "feature_group_count": feature_groups,
                    "batch_group_count": batch_groups,
                    "kernel_shape": list(rhs_shape[2:]),
                    "input_channels": lhs_shape[1],
                    "output_channels": rhs_shape[0],
                }
            )
        elif op_type == SemanticOpType.MATMUL.value:
            if len(input_shapes) < 2 or len(result_shape) < 2:
                raise FrontendImportError(f"StableHLO dot operation '{result_name}' requires rank >= 2 operands")
            dot_dimensions = _dot_dimension_numbers(body, input_shapes[0], input_shapes[1])
            lhs = input_shapes[0]
            rhs = input_shapes[1]
            lhs_batch = dot_dimensions["lhs_batching_dimensions"]
            rhs_batch = dot_dimensions["rhs_batching_dimensions"]
            lhs_contract = dot_dimensions["lhs_contracting_dimensions"]
            rhs_contract = dot_dimensions["rhs_contracting_dimensions"]
            if len(lhs_contract) != 1:
                raise FrontendImportError(
                    f"StableHLO dot operation '{result_name}' requires exactly one contracting dimension"
                )
            if len(lhs_batch) != len(rhs_batch):
                raise FrontendImportError("StableHLO dot batching dimension counts differ")
            lhs_free = tuple(index for index in range(len(lhs)) if index not in (*lhs_batch, *lhs_contract))
            rhs_free = tuple(index for index in range(len(rhs)) if index not in (*rhs_batch, *rhs_contract))
            if not lhs_free or len(rhs_free) != 1:
                raise FrontendImportError(
                    f"StableHLO dot operation '{result_name}' must have one free RHS matrix dimension"
                )
            # XLA permits a rank-N activation multiplied by a rank-2 weight
            # without spelling the leading activation dimensions as batching
            # dimensions.  The canonical lowering treats those leading free
            # LHS dimensions as implicit broadcast batches.
            implicit_lhs_batch = ()
            lhs_matrix_free = lhs_free
            if not lhs_batch and not rhs_batch and len(lhs_free) > 1:
                implicit_lhs_batch = lhs_free[:-1]
                lhs_matrix_free = lhs_free[-1:]
            if len(lhs_matrix_free) != 1:
                raise FrontendImportError(
                    f"StableHLO dot operation '{result_name}' has unsupported free LHS dimensions"
                )
            batch_lhs_dims = (*lhs_batch, *implicit_lhs_batch)
            expected_shape = tuple(lhs[index] for index in batch_lhs_dims) + tuple(lhs[index] for index in lhs_matrix_free) + tuple(rhs[index] for index in rhs_free)
            if tuple(result_shape) != expected_shape:
                raise FrontendImportError(
                    f"StableHLO dot operation '{result_name}' result shape {result_shape} does not match "
                    f"dot dimensions (expected {expected_shape})"
                )
            iteration_dims = tuple(
                [(f"B{index}", lhs[dimension]) for index, dimension in enumerate(batch_lhs_dims)]
                + [("M", lhs[lhs_matrix_free[0]]), ("N", rhs[rhs_free[0]])]
            )
            reduction_dims = (("K", lhs[lhs_contract[0]]),)
            rhs_contract_axis = rhs_contract[0]
            if rhs_contract_axis not in {len(rhs) - 1, len(rhs) - 2}:
                raise FrontendImportError(
                    f"StableHLO dot operation '{result_name}' contracts RHS axis {rhs_contract_axis}; "
                    "the canonical lowering requires the last or penultimate axis"
                )
            operation_attributes["dot_dimension_numbers"] = {
                name: list(value) for name, value in dot_dimensions.items()
            }
            operation_attributes["rhs_transposed"] = rhs_contract_axis == len(rhs) - 1
            operation_attributes["rhs_broadcast_batch"] = bool(batch_lhs_dims and not rhs_batch)
            if batch_lhs_dims or rhs_batch or len(lhs) > 2 or len(rhs) > 2:
                op_type = SemanticOpType.BATCHED_MATMUL.value
        elif op_type == SemanticOpType.REDUCE.value:
            # StableHLO reduce has one data operand followed by a scalar
            # reducer-init operand.  The init value is semantic metadata, not
            # a second tensor input to the canonical reduce operator.
            input_names = (input_names[0],)
            axes = _dimensions(body, len(input_shapes[0]))
            reduction_dims = tuple((f"d{axis}", input_shapes[0][axis]) for axis in axes)
            iteration_dims = tuple(
                (f"d{axis}", input_shapes[0][axis])
                for axis in range(len(input_shapes[0]))
                if axis not in axes
            )
            operation_attributes["reducer"] = _reducer(body) or "add"
        elif op_type in {SemanticOpType.RMSNORM.value, SemanticOpType.LAYERNORM.value, SemanticOpType.SOFTMAX.value}:
            axes = _dimensions(body, len(input_shapes[0]))
            reduction_dims = tuple((f"d{axis}", input_shapes[0][axis]) for axis in axes)
            iteration_dims = tuple(
                (f"d{axis}", input_shapes[0][axis])
                for axis in range(len(input_shapes[0]))
                if axis not in axes
            )
        elif op_type == SemanticOpType.RESHAPE.value:
            iteration_dims = tuple((f"d{axis}", value) for axis, value in enumerate(result_shape))
            reduction_dims = ()
            if normalized_target == "stablehlo.broadcast_in_dim":
                dimensions = _dimension_list(body, "dims")
                source_shape = input_shapes[0]
                if dimensions is None:
                    raise FrontendImportError(
                        f"StableHLO broadcast_in_dim '{result_name}' is missing broadcast dimensions"
                    )
                if len(dimensions) != len(source_shape):
                    raise FrontendImportError(
                        f"StableHLO broadcast_in_dim '{result_name}' maps {len(source_shape)} "
                        f"operand dimensions using {len(dimensions)} entries"
                    )
                if len(set(dimensions)) != len(dimensions) or any(
                    axis < 0 or axis >= len(result_shape) for axis in dimensions
                ):
                    raise FrontendImportError(
                        f"StableHLO broadcast_in_dim '{result_name}' has invalid dimensions "
                        f"{dimensions} for result rank {len(result_shape)}"
                    )
                for source_axis, result_axis in enumerate(dimensions):
                    source_extent = source_shape[source_axis]
                    result_extent = result_shape[result_axis]
                    if source_extent != 1 and source_extent != result_extent:
                        raise FrontendImportError(
                            f"StableHLO broadcast_in_dim '{result_name}' cannot broadcast "
                            f"operand dimension {source_axis}={source_extent} to "
                            f"result dimension {result_axis}={result_extent}"
                        )
                operation_attributes.update(
                    {
                        "broadcast": True,
                        "broadcast_dimensions": list(dimensions),
                    }
                )
        elif op_type == "slice":
            starts = _named_integer_list(body, "starts")
            limits = _named_integer_list(body, "limits")
            strides = _named_integer_list(body, "strides")
            if starts is None or limits is None or strides is None:
                raise FrontendImportError(
                    f"StableHLO slice operation '{result_name}' requires starts, limits and strides"
                )
            if not (
                len(starts) == len(limits) == len(strides) == len(input_shapes[0])
            ):
                raise FrontendImportError(
                    f"StableHLO slice operation '{result_name}' index rank does not match operand rank"
                )
            operand_shape = input_shapes[0]
            if any(
                start < 0
                or limit < start
                or limit > extent
                or stride <= 0
                for start, limit, stride, extent in zip(
                    starts, limits, strides, operand_shape
                )
                if isinstance(extent, int)
            ):
                raise FrontendImportError(
                    f"StableHLO slice operation '{result_name}' has invalid bounds"
                )
            if any(
                not isinstance(extent, int)
                for extent in operand_shape
            ):
                raise FrontendImportError(
                    f"StableHLO slice operation '{result_name}' requires a resolved operand shape"
                )
            expected_shape = tuple(
                (limit - start + stride - 1) // stride
                for start, limit, stride in zip(starts, limits, strides)
            )
            if tuple(result_shape) != expected_shape:
                raise FrontendImportError(
                    f"StableHLO slice operation '{result_name}' result shape {result_shape} "
                    f"does not match bounds {expected_shape}"
                )
            operation_attributes.update(
                {
                    "slice_starts": list(starts),
                    "slice_limits": list(limits),
                    "slice_strides": list(strides),
                }
            )
            iteration_dims = tuple((f"d{axis}", value) for axis, value in enumerate(result_shape))
            reduction_dims = ()
        elif op_type == "concatenate":
            dimension = _named_integer(body, "dim")
            if dimension is None:
                dimension = _named_integer(body, "dimension")
            if dimension is None or dimension < 0 or dimension >= len(result_shape):
                raise FrontendImportError(
                    f"StableHLO concatenate operation '{result_name}' has an invalid dimension"
                )
            if any(
                len(shape) != len(result_shape)
                or any(
                    axis != dimension and extent != result_shape[axis]
                    for axis, extent in enumerate(shape)
                )
                for shape in input_shapes
            ) or sum(shape[dimension] for shape in input_shapes) != result_shape[dimension]:
                raise FrontendImportError(
                    f"StableHLO concatenate operation '{result_name}' has incompatible shapes"
                )
            operation_attributes["concatenate_dimension"] = dimension
            iteration_dims = tuple((f"d{axis}", value) for axis, value in enumerate(result_shape))
            reduction_dims = ()
        elif op_type == SemanticOpType.TRANSPOSE.value:
            iteration_dims = tuple((f"d{axis}", value) for axis, value in enumerate(result_shape))
            reduction_dims = ()
            operation_attributes["transpose_dims"] = _transpose_dimensions(body, len(input_shapes[0]))
        elif "batch_norm_training" in target:
            iteration_dims = tuple((f"d{axis}", value) for axis, value in enumerate(result_shape))
            reduction_dims = ()
            feature_index = _named_integer(body, "feature_index")
            epsilon = _named_float(body, "epsilon")
            if feature_index is None or feature_index < 0 or feature_index >= len(result_shape):
                raise FrontendImportError(
                    f"StableHLO batch_norm_training '{result_name}' has an invalid feature_index"
                )
            operation_attributes["feature_index"] = feature_index
            operation_attributes["epsilon"] = epsilon if epsilon is not None else 1e-5
        else:
            iteration_dims = tuple((f"d{axis}", value) for axis, value in enumerate(result_shape))
            reduction_dims = ()
        constants_for_op = _constant_args(body, constants)
        operators.append(
            OperatorSpec(
                op_id=result_name.removeprefix("%"),
                op_type=op_type,
                inputs=tuple(name.removeprefix("%") for name in input_names),
                outputs=(result_name.removeprefix("%"),),
                iteration_dims=iteration_dims,
                reduction_dims=reduction_dims,
                attributes={
                    "frontend_target": target,
                    "frontend_node_op": "stablehlo",
                    "stablehlo_op": capability.op_name,
                    "semantic_family": capability.semantic_family,
                    "semantic_op": capability.op_name.removeprefix("stablehlo."),
                    "operand_arity": operand_arity,
                    "requires_recovery": capability.requires_recovery,
                    "backend_capability_key": capability.backend_capability_key,
                    "input_occurrences": list(input_occurrences),
                    "constant_args": {"args": constants_for_op, "kwargs": {}},
                    **operation_attributes,
                },
                provenance={
                    "frontend": FrontendKind.STABLEHLO.value,
                    "source_node": result_name.removeprefix("%"),
                    "source_target": target,
                },
            )
        )
        result_tensor_name = result_name.removeprefix("%")
        tensors[result_tensor_name] = TensorSpec(
            name=result_tensor_name,
            shape=result_shape,
            dtype=result_dtype,
            attributes={
                **_tensor_attributes(
                    result_type_text,
                    source_kind="activation",
                    source_node=result_tensor_name,
                    frontend_target=target,
                ),
                **(
                    {
                        "alias_of": str(operation_attributes["state_buffer"]),
                        "persistent": True,
                        "state_id": str(operation_attributes["state_id"]),
                        "state_buffer": str(operation_attributes["state_buffer"]),
                    }
                    if operation_attributes.get("state_update")
                    else {}
                ),
            },
        )
        produced_by[result_tensor_name] = result_tensor_name

    if not graph_outputs:
        consumed = {tensor for operator in operators for tensor in operator.inputs}
        graph_outputs = [
            operator.outputs[0]
            for operator in operators
            if operator.outputs and operator.outputs[0] not in consumed
        ]
    producer_by_tensor = {
        output: operator.op_id
        for operator in operators
        for output in operator.outputs
    }
    edges: list[DataEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for operator in operators:
        for tensor in operator.inputs:
            producer = producer_by_tensor.get(tensor)
            if producer is None or producer == operator.op_id:
                continue
            key = (producer, operator.op_id, tensor)
            if key not in seen:
                edges.append(DataEdge(*key))
                seen.add(key)
    graph = OperatorGraph(
        graph_id=graph_id,
        tensors=tuple(tensors.values()),
        operators=tuple(operators),
        edges=tuple(edges),
        attributes={
            "frontend": FrontendKind.STABLEHLO.value,
            "entry_point": function.group("name"),
            "graph_inputs": [name.removeprefix("%") for name in graph_inputs],
            "graph_outputs": [name.removeprefix("%") for name in graph_outputs],
            "stablehlo_parser": "textual-subset-v0",
        },
    )
    issues = graph.validate()
    if issues:
        raise FrontendImportError("StableHLO graph normalization failed: " + "; ".join(issues))
    return graph


class StableHLOAdapter:
    """Import the official adapter's verified StableHLO projection."""

    kind = FrontendKind.STABLEHLO

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        model_id: str = "stablehlo_model",
        variant: str = "stablehlo-v0",
        shape_environment: Mapping[str, int] | None = None,
    ) -> FrontendImport:
        if not isinstance(text, str) or not text.strip():
            raise FrontendImportError("StableHLO text must be a non-empty string")
        graph = _graph_from_text(text, graph_id=f"{model_id}.graph")
        imported = FrontendImport(
            graph=graph,
            model_id=model_id,
            variant=variant,
            shape_environment=normalize_shape_environment(shape_environment),
            frontend=cls.kind,
            provenance={"source": "stablehlo-text", "parser": "textual-subset-v0"},
            family=ModelFamily.SYNTHETIC,
        )
        issues = imported.validate()
        if issues:
            raise FrontendImportError("invalid StableHLO import: " + "; ".join(issues))
        return imported

__all__ = ["StableHLOAdapter"]
