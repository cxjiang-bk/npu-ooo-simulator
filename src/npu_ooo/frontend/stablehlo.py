from __future__ import annotations

"""Internal semantic parser for verified StableHLO projections.

The official adapter owns external MLIR parsing and verification.  This module
only turns the verified, project-supported operation projection into the
Canonical OperatorGraph. Unsupported constructs fail at this boundary instead
of being silently dropped.
"""

import re
from typing import Any, Mapping

from npu_ooo.ir import (
    DataEdge,
    ModelFamily,
    OperatorGraph,
    OperatorSpec,
    SemanticOpType,
    TensorSpec,
)

from .bridge import FrontendImport, FrontendImportError, FrontendKind
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
    parts = [part.strip() for part in match.group(1).split("x") if part.strip()]
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
        shape, dtype = _tensor_type(argument.group("type"))
        name = argument.group("name").removeprefix("%")
        tensors[name] = TensorSpec(
            name=name,
            shape=shape,
            dtype=dtype,
            attributes={"source_kind": "input", "source_node": name, "frontend": "stablehlo"},
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
        result_shape, result_dtype = _result_tensor_type(match.group("types"))
        if target.endswith(".constant"):
            constants[result_name] = _constant_value(body)
            normalized_name = result_name.removeprefix("%")
            if result_shape:
                tensors[normalized_name] = TensorSpec(
                    name=normalized_name,
                    shape=result_shape,
                    dtype=result_dtype,
                    attributes={
                        "source_kind": "constant",
                        "source_node": normalized_name,
                        "frontend_target": target,
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
        if normalized_target == "stablehlo.convert":
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
        tensors[result_name.removeprefix("%")] = TensorSpec(
            name=result_name.removeprefix("%"),
            shape=result_shape,
            dtype=result_dtype,
            attributes={
                "source_kind": "activation",
                "source_node": result_name.removeprefix("%"),
                "frontend_target": target,
            },
        )
        produced_by[result_name.removeprefix("%")] = result_name.removeprefix("%")

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
            shape_environment=dict(shape_environment or {}),
            frontend=cls.kind,
            provenance={"source": "stablehlo-text", "parser": "textual-subset-v0"},
            family=ModelFamily.SYNTHETIC,
        )
        issues = imported.validate()
        if issues:
            raise FrontendImportError("invalid StableHLO import: " + "; ".join(issues))
        return imported

__all__ = ["StableHLOAdapter"]
