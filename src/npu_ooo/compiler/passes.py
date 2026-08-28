from __future__ import annotations

"""Small, inspectable graph passes used before scheduling and lowering.

Each transformation is intentionally separate so every frontend can dump and
validate the IR between transformations.  The initial composite pass handles
the common explicit RMSNorm formula emitted by ``torch.export``; unsupported
patterns remain visible instead of being silently rewritten.
"""

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Protocol

from npu_ooo.ir import DataEdge, OperatorGraph, OperatorSpec, TensorSpec

if TYPE_CHECKING:
    from .fusion_patterns import SemanticFusionPatternRegistry


@dataclass(frozen=True)
class PassDiagnostic:
    level: str
    pass_name: str
    message: str


@dataclass(frozen=True)
class PassSnapshot:
    """Input/output graphs for one GC pass invocation."""

    pass_index: int
    pass_name: str
    input_graph: OperatorGraph
    output_graph: OperatorGraph
    diagnostics: tuple[PassDiagnostic, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "paper_stage": "GC",
            "pass_index": self.pass_index,
            "pass_name": self.pass_name,
            "input_graph": self.input_graph.to_dict(),
            "output_graph": self.output_graph.to_dict(),
            "diagnostics": [
                {
                    "level": item.level,
                    "pass": item.pass_name,
                    "message": item.message,
                }
                for item in self.diagnostics
            ],
        }


@dataclass(frozen=True)
class PassResult:
    graph: OperatorGraph
    diagnostics: tuple[PassDiagnostic, ...] = ()
    snapshots: tuple[PassSnapshot, ...] = ()


class GraphPass(Protocol):
    name: str

    def run(self, graph: OperatorGraph) -> PassResult:
        ...


_TYPE_ALIASES = {
    "aten.mm.default": "matmul",
    "aten.matmul.default": "matmul",
    "aten.linear.default": "matmul",
    "aten.bmm.default": "batched_matmul",
    "aten.softmax.int": "softmax",
    "aten._softmax.default": "softmax",
    "aten.sum.dim_intlist": "reduce",
    "aten.amax.default": "reduce",
    "aten.add.tensor": "elementwise",
    "aten.mul.tensor": "elementwise",
    "aten.sub.tensor": "elementwise",
    "aten.div.tensor": "elementwise",
    "aten.rsqrt.default": "elementwise",
    "aten.sqrt.default": "elementwise",
    "aten.pow.Tensor_Scalar": "elementwise",
    "aten.reshape.default": "reshape",
    "aten.view.default": "reshape",
    "aten.transpose.int": "transpose",
    "aten.permute.default": "transpose",
}


def _canonical_type(operator: OperatorSpec) -> str:
    normalized = operator.normalized_type.strip().lower().replace("::", ".")
    return _TYPE_ALIASES.get(normalized, normalized)


class CanonicalizeGraphPass:
    """Normalize aliases and infer missing producer-consumer edges."""

    name = "canonicalize"

    def run(self, graph: OperatorGraph) -> PassResult:
        issues = graph.validate()
        if issues:
            raise ValueError("canonicalize input graph is invalid: " + "; ".join(issues))

        operators = tuple(
            OperatorSpec(
                op_id=operator.op_id,
                op_type=_canonical_type(operator),
                inputs=operator.inputs,
                outputs=operator.outputs,
                iteration_dims=operator.iteration_dims,
                reduction_dims=operator.reduction_dims,
                attributes=dict(operator.attributes),
                provenance=dict(operator.provenance),
            )
            for operator in graph.operators
        )
        producer_by_tensor: dict[str, str] = {}
        for operator in operators:
            for tensor in operator.outputs:
                producer_by_tensor[tensor] = operator.op_id

        edges: list[DataEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for edge in graph.edges:
            key = (edge.producer, edge.consumer, edge.tensor)
            if key not in seen_edges:
                edges.append(edge)
                seen_edges.add(key)
        for operator in operators:
            for tensor in operator.inputs:
                producer = producer_by_tensor.get(tensor)
                if producer is None or producer == operator.op_id:
                    continue
                key = (producer, operator.op_id, tensor)
                if key not in seen_edges:
                    edges.append(DataEdge(*key))
                    seen_edges.add(key)

        attributes = {
            **dict(graph.attributes),
            "canonicalized": True,
            "canonical_passes": [*dict(graph.attributes).get("canonical_passes", ()), self.name],
        }
        result = OperatorGraph(
            graph_id=graph.graph_id,
            tensors=graph.tensors,
            operators=operators,
            edges=tuple(edges),
            attributes=attributes,
        )
        result_issues = result.validate()
        if result_issues:
            raise ValueError("canonicalize produced an invalid graph: " + "; ".join(result_issues))
        changed = result.to_dict() != graph.to_dict()
        diagnostic = PassDiagnostic(
            "info",
            self.name,
            "normalized operator aliases and inferred missing data edges"
            if changed
            else "graph already canonical",
        )
        return PassResult(result, (diagnostic,))


def _infer_edges(operators: tuple[OperatorSpec, ...]) -> tuple[DataEdge, ...]:
    producer_by_tensor = {
        tensor: operator.op_id
        for operator in operators
        for tensor in operator.outputs
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
    return tuple(edges)


class LinearDecompositionPass:
    """Lower ``aten.linear`` to Matmul with transposed RHS and optional bias Add."""

    name = "decompose_linear"

    def run(self, graph: OperatorGraph) -> PassResult:
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        rewritten: list[OperatorSpec] = []
        added_tensors: list[TensorSpec] = []
        decomposition_groups: list[dict[str, object]] = []
        decomposition_count = 0
        for operator in graph.operators:
            target = _frontend_target(operator)
            if operator.normalized_type not in {"matmul", "batched_matmul"} or "linear" not in target:
                rewritten.append(operator)
                continue
            if len(operator.inputs) not in {2, 3} or len(operator.outputs) != 1:
                raise ValueError(
                    f"linear operator '{operator.op_id}' requires input, weight, optional bias, and one output"
                )
            input_tensor = tensors[operator.inputs[0]]
            weight_tensor = tensors[operator.inputs[1]]
            output_tensor = tensors[operator.outputs[0]]
            if len(input_tensor.shape) < 2 or len(weight_tensor.shape) != 2 or len(output_tensor.shape) != len(input_tensor.shape):
                raise NotImplementedError(
                    f"linear operator '{operator.op_id}' requires ranked input/output and rank-2 weight"
                )
            if (
                input_tensor.shape[-1] != weight_tensor.shape[1]
                or output_tensor.shape != (*input_tensor.shape[:-1], weight_tensor.shape[0])
            ):
                raise ValueError(f"linear operator '{operator.op_id}' has inconsistent input/weight/output shapes")

            has_bias = len(operator.inputs) == 3
            matmul_output = operator.outputs[0]
            if has_bias:
                bias_tensor = tensors[operator.inputs[2]]
                if bias_tensor.shape != (output_tensor.shape[-1],):
                    raise ValueError(
                        f"linear operator '{operator.op_id}' bias shape {bias_tensor.shape} "
                        f"does not match output width {output_tensor.shape[-1]}"
                    )
                matmul_output = f"{operator.op_id}.matmul_output"
                added_tensors.append(
                    TensorSpec(
                        name=matmul_output,
                        shape=output_tensor.shape,
                        dtype=output_tensor.dtype,
                        layout=output_tensor.layout,
                        attributes={
                            "source_kind": "compiler_temporary",
                            "decomposition": self.name,
                            "source_operator": operator.op_id,
                        },
                    )
                )
            matmul = OperatorSpec(
                op_id=f"{operator.op_id}.matmul" if has_bias else operator.op_id,
                op_type=(
                    "batched_matmul"
                    if len(input_tensor.shape) > 2
                    else "matmul"
                ),
                inputs=operator.inputs[:2],
                outputs=(matmul_output,),
                iteration_dims=operator.iteration_dims,
                reduction_dims=operator.reduction_dims,
                attributes={
                    **dict(operator.attributes),
                    "rhs_transposed": True,
                    "rhs_broadcast_batch": len(input_tensor.shape) > 2,
                    "decomposition": self.name,
                    "source_linear": operator.op_id,
                    "bias_input": None,
                },
                provenance={
                    **dict(operator.provenance),
                    "compiler_pass": self.name,
                    "source_operator": operator.op_id,
                },
            )
            rewritten.append(matmul)
            generated = [matmul.op_id]
            if has_bias:
                add = OperatorSpec(
                    op_id=f"{operator.op_id}.bias_add",
                    op_type="elementwise",
                    inputs=(matmul_output, operator.inputs[2]),
                    outputs=operator.outputs,
                    iteration_dims=operator.iteration_dims,
                    attributes={
                        "frontend_target": "compiler.broadcast_add",
                        "broadcast": "numpy",
                        "decomposition": self.name,
                        "source_linear": operator.op_id,
                    },
                    provenance={
                        **dict(operator.provenance),
                        "compiler_pass": self.name,
                        "source_operator": operator.op_id,
                    },
                )
                rewritten.append(add)
                generated.append(add.op_id)
            decomposition_groups.append(
                {"kind": "linear", "source_operator": operator.op_id, "generated_ops": generated}
            )
            decomposition_count += 1

        if not decomposition_count:
            return PassResult(
                graph,
                (PassDiagnostic("info", self.name, "no aten.linear operators found"),),
            )
        operators = tuple(rewritten)
        result = OperatorGraph(
            graph_id=graph.graph_id,
            tensors=(*graph.tensors, *added_tensors),
            operators=operators,
            edges=_infer_edges(operators),
            attributes={
                **dict(graph.attributes),
                "canonical_passes": [*dict(graph.attributes).get("canonical_passes", ()), self.name],
                "decomposition_groups": [
                    *dict(graph.attributes).get("decomposition_groups", ()),
                    *decomposition_groups,
                ],
            },
        )
        issues = result.validate()
        if issues:
            raise ValueError("decompose_linear produced an invalid graph: " + "; ".join(issues))
        return PassResult(
            result,
            (
                PassDiagnostic(
                    "info",
                    self.name,
                    f"decomposed {decomposition_count} aten.linear operator(s)",
                ),
            ),
        )


def _frontend_target(operator: OperatorSpec) -> str:
    return str(operator.attributes.get("frontend_target", "")).lower().replace("::", ".")


def _input_occurrences(operator: OperatorSpec) -> tuple[str, ...]:
    raw = operator.attributes.get("input_occurrences")
    if isinstance(raw, (tuple, list)):
        return tuple(str(item) for item in raw)
    return operator.inputs


class FoldTransposeIntoMatmulPass:
    """Fold a single-use RHS last-two-dimension transpose into Matmul metadata."""

    name = "fold_transpose_into_matmul"

    def run(self, graph: OperatorGraph) -> PassResult:
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        operators = list(graph.operators)
        consumers: dict[str, list[OperatorSpec]] = {}
        for operator in operators:
            for tensor in operator.inputs:
                consumers.setdefault(tensor, []).append(operator)

        replacements: dict[str, OperatorSpec] = {}
        removed_operator_ids: set[str] = set()
        removed_tensors: set[str] = set()
        fold_groups: list[dict[str, str]] = []
        for transpose in operators:
            if transpose.normalized_type != "transpose":
                continue
            if len(transpose.inputs) != 1 or len(transpose.outputs) != 1:
                continue
            source_tensor = tensors.get(transpose.inputs[0])
            if source_tensor is None:
                continue
            dimensions = tuple(transpose.attributes.get("transpose_dims", ()))
            rank = len(source_tensor.shape)
            last_two = (rank - 2, rank - 1)
            full_last_two_swap = tuple([*range(max(0, rank - 2)), rank - 1, rank - 2])
            if dimensions != last_two and dimensions != full_last_two_swap:
                continue
            output_tensor = transpose.outputs[0]
            users = consumers.get(output_tensor, ())
            if len(users) != 1:
                continue
            matmul = users[0]
            if matmul.normalized_type not in {"matmul", "batched_matmul"}:
                continue
            if (
                len(matmul.inputs) < 2
                or matmul.inputs[1] != output_tensor
                or matmul.inputs.count(output_tensor) != 1
            ):
                continue
            rewritten_inputs = tuple(
                transpose.inputs[0] if tensor == output_tensor else tensor
                for tensor in matmul.inputs
            )
            rewritten_occurrences = tuple(
                transpose.inputs[0] if tensor == output_tensor else tensor
                for tensor in _input_occurrences(matmul)
            )
            replacements[matmul.op_id] = OperatorSpec(
                op_id=matmul.op_id,
                op_type=matmul.normalized_type,
                inputs=rewritten_inputs,
                outputs=matmul.outputs,
                iteration_dims=matmul.iteration_dims,
                reduction_dims=matmul.reduction_dims,
                attributes={
                    **dict(matmul.attributes),
                    "rhs_transposed": True,
                    "input_occurrences": list(rewritten_occurrences),
                    "folded_transpose": transpose.op_id,
                },
                provenance={
                    **dict(matmul.provenance),
                    "compiler_pass": self.name,
                    "folded_transpose": transpose.op_id,
                },
            )
            removed_operator_ids.add(transpose.op_id)
            removed_tensors.add(output_tensor)
            fold_groups.append(
                {
                    "kind": "rhs_transpose",
                    "transpose": transpose.op_id,
                    "matmul": matmul.op_id,
                }
            )

        if not replacements:
            return PassResult(
                graph,
                (PassDiagnostic("info", self.name, "no foldable Matmul RHS transpose found"),),
            )
        rewritten_operators = tuple(
            replacements.get(operator.op_id, operator)
            for operator in operators
            if operator.op_id not in removed_operator_ids
        )
        result = OperatorGraph(
            graph_id=graph.graph_id,
            tensors=tuple(
                tensor for tensor in graph.tensors if tensor.name not in removed_tensors
            ),
            operators=rewritten_operators,
            edges=_infer_edges(rewritten_operators),
            attributes={
                **dict(graph.attributes),
                "canonical_passes": [*dict(graph.attributes).get("canonical_passes", ()), self.name],
                "fold_groups": [*dict(graph.attributes).get("fold_groups", ()), *fold_groups],
            },
        )
        issues = result.validate()
        if issues:
            raise ValueError("fold_transpose_into_matmul produced an invalid graph: " + "; ".join(issues))
        return PassResult(
            result,
            (
                PassDiagnostic(
                    "info",
                    self.name,
                    f"folded {len(replacements)} RHS transpose operation(s)",
                ),
            ),
        )


def _is_elementwise_mul(operator: OperatorSpec) -> bool:
    target = _frontend_target(operator)
    return operator.normalized_type == "elementwise" and any(
        token in target for token in ("aten.mul", "stablehlo.multiply", "stablehlo.mul")
    )


def _is_elementwise_add(operator: OperatorSpec) -> bool:
    target = _frontend_target(operator)
    return operator.normalized_type == "elementwise" and any(
        token in target for token in ("aten.add", "stablehlo.add")
    )


def _single_output(operator: OperatorSpec) -> str:
    if len(operator.outputs) != 1:
        raise ValueError(f"composite fusion requires one output for '{operator.op_id}'")
    return operator.outputs[0]


def _constant_scalar(operator: OperatorSpec) -> float | int | None:
    constants = operator.attributes.get("constant_args", {})
    args = constants.get("args", ()) if isinstance(constants, dict) else ()
    values = args if isinstance(args, (tuple, list)) else (args,)
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return value
    return None


def _constant_tensor(tensor: str, tensors: dict[str, TensorSpec]) -> bool:
    spec = tensors.get(tensor)
    return bool(spec is not None and spec.attributes.get("source_kind") == "constant")


def _unwrap_reshape(
    tensor: str,
    producer: dict[str, OperatorSpec],
) -> tuple[str, tuple[OperatorSpec, ...]]:
    """Remove a chain of single-input reshapes while retaining its operators."""

    current = tensor
    reshapes: list[OperatorSpec] = []
    while True:
        operation = producer.get(current)
        if operation is None or operation.normalized_type != "reshape" or len(operation.inputs) != 1:
            return current, tuple(reshapes)
        reshapes.append(operation)
        current = operation.inputs[0]


def _is_rmsnorm_square(
    operation: OperatorSpec,
    tensors: dict[str, TensorSpec],
) -> str | None:
    """Return the squared input for multiply/power square expressions."""

    if operation.normalized_type != "elementwise":
        return None
    nonconstant = [
        tensor for tensor in operation.inputs if not _constant_tensor(tensor, tensors)
    ]
    target = _frontend_target(operation)
    if any(token in target for token in ("stablehlo.multiply", "stablehlo.mul", "aten.mul")):
        return nonconstant[0] if len(nonconstant) == 2 and nonconstant[0] == nonconstant[1] else None
    if any(token in target for token in ("stablehlo.power", "aten.pow")):
        exponent = _constant_scalar(operation)
        return nonconstant[0] if len(nonconstant) == 1 and exponent == 2 else None
    return None


def _is_elementwise(operator: OperatorSpec, *names: str) -> bool:
    if operator.normalized_type != "elementwise":
        return False
    target = _frontend_target(operator)
    return any(name in target for name in names)


def _is_passthrough_convert(operator: OperatorSpec) -> bool:
    """Return whether an elementwise StableHLO convert is shape-preserving."""

    return (
        _is_elementwise(operator, "stablehlo.convert")
        and len(operator.inputs) == 1
        and len(operator.outputs) == 1
    )


def _unwrap_convert_chain(
    tensor: str,
    producer: dict[str, OperatorSpec],
) -> tuple[str, tuple[OperatorSpec, ...]]:
    """Remove a single-input convert chain while retaining its operations."""

    current = tensor
    converts: list[OperatorSpec] = []
    while True:
        operation = producer.get(current)
        if operation is None or not _is_passthrough_convert(operation):
            return current, tuple(converts)
        converts.append(operation)
        current = operation.inputs[0]


def _producer_map(operators: list[OperatorSpec]) -> dict[str, OperatorSpec]:
    return {tensor: operator for operator in operators for tensor in operator.outputs}


def _consumer_map(operators: list[OperatorSpec]) -> dict[str, list[OperatorSpec]]:
    consumers: dict[str, list[OperatorSpec]] = {}
    for operator in operators:
        for tensor in operator.inputs:
            consumers.setdefault(tensor, []).append(operator)
    return consumers


class RecoverStableHLOLayerNormPass:
    """Recover the constrained batch-norm form torch-xla uses for LayerNorm."""

    name = "recover_stablehlo_layernorm"
    repeat_until_stable = True

    def run(self, graph: OperatorGraph) -> PassResult:
        operators = list(graph.operators)
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        producer = _producer_map(operators)
        consumers = _consumer_map(operators)
        for batch_norm in operators:
            if _frontend_target(batch_norm) != "stablehlo.batch_norm_training":
                continue
            if len(batch_norm.inputs) != 3 or len(batch_norm.outputs) != 1:
                continue
            norm_source = batch_norm.inputs[0]
            norm_source_spec = tensors.get(norm_source)
            if norm_source_spec is None or len(norm_source_spec.shape) < 2:
                continue
            feature_index = batch_norm.attributes.get("feature_index")
            if not isinstance(feature_index, int) or feature_index == len(norm_source_spec.shape) - 1:
                continue

            # batch_norm_training reduces every non-feature axis.  It is
            # equivalent to last-axis LayerNorm when all other reduction axes
            # are unit extents.  torch-xla may first reshape [*outer, hidden]
            # to [1, prod(outer), hidden], making each feature an independent
            # LayerNorm row.
            norm_reduction_axis = len(norm_source_spec.shape) - 1
            other_reduced_axes = (
                axis
                for axis in range(len(norm_source_spec.shape))
                if axis not in {feature_index, norm_reduction_axis}
            )
            if any(norm_source_spec.shape[axis] != 1 for axis in other_reduced_axes):
                continue
            constants = batch_norm.attributes.get("constant_args", {})
            args = constants.get("args", ()) if isinstance(constants, dict) else ()
            if len(args) < 2 or args[0] != 1.0 or args[1] != 0.0:
                continue

            normalized_users = consumers.get(batch_norm.outputs[0], ())
            if len(normalized_users) != 1:
                continue
            post_reshape = (
                normalized_users[0]
                if normalized_users[0].normalized_type == "reshape"
                else None
            )
            if post_reshape is not None:
                post_users = consumers.get(post_reshape.outputs[0], ())
                if len(post_users) != 1:
                    continue
                scaled = post_users[0]
                normalized_tensor = post_reshape.outputs[0]
            else:
                scaled = normalized_users[0]
                normalized_tensor = batch_norm.outputs[0]
            if not _is_elementwise(scaled, "stablehlo.multiply") or len(scaled.inputs) != 2:
                continue
            weight = next((item for item in scaled.inputs if item != normalized_tensor), None)
            if weight is None:
                continue
            scaled_users = consumers.get(scaled.outputs[0], ())
            if len(scaled_users) != 1:
                continue
            shifted = scaled_users[0]
            if not _is_elementwise(shifted, "stablehlo.add") or len(shifted.inputs) != 2:
                continue
            bias = next((item for item in shifted.inputs if item != scaled.outputs[0]), None)
            output = _single_output(shifted)
            output_spec = tensors.get(output)
            weight_spec = tensors.get(weight) if weight else None
            bias_spec = tensors.get(bias) if bias else None
            pre_reshape = producer.get(norm_source)
            if pre_reshape is not None and pre_reshape.normalized_type != "reshape":
                pre_reshape = None
            source = pre_reshape.inputs[0] if pre_reshape is not None else norm_source
            source_spec = tensors.get(source)
            if source_spec is None or not all(
                isinstance(extent, int) for extent in source_spec.shape
            ):
                continue
            reduction_axis = len(source_spec.shape) - 1
            reduction_extent = source_spec.shape[reduction_axis]
            expected_rows = math.prod(source_spec.shape[:-1])
            flattened_norm = pre_reshape is not None or post_reshape is not None
            if flattened_norm:
                if (
                    pre_reshape is None
                    or post_reshape is None
                    or consumers.get(pre_reshape.outputs[0], ()) != [batch_norm]
                    or tuple(norm_source_spec.shape)
                    != (1, expected_rows, reduction_extent)
                    or feature_index != 1
                    or tensors[post_reshape.outputs[0]].shape != source_spec.shape
                ):
                    continue
            if (
                output_spec is None
                or output_spec.shape != source_spec.shape
                or weight_spec is None
                or bias_spec is None
                or weight_spec.shape != (reduction_extent,)
                or bias_spec.shape != (reduction_extent,)
            ):
                continue

            chain = (
                *((pre_reshape,) if pre_reshape is not None else ()),
                batch_norm,
                *((post_reshape,) if post_reshape is not None else ()),
                scaled,
                shifted,
            )
            matched_ids = {item.op_id for item in chain}
            fused = OperatorSpec(
                op_id=output,
                op_type="layernorm",
                inputs=(source, weight, bias),
                outputs=(output,),
                iteration_dims=tuple(
                    (f"d{axis}", extent)
                    for axis, extent in enumerate(source_spec.shape)
                    if axis != reduction_axis
                ),
                reduction_dims=((f"d{reduction_axis}", reduction_extent),),
                attributes={
                    "frontend_target": "stablehlo.layer_norm",
                    "axis": f"d{reduction_axis}",
                    "epsilon": batch_norm.attributes.get("epsilon", 1e-5),
                    "affine": True,
                    "fusion": "torch_xla_batch_norm_layernorm",
                    "feature_index": feature_index,
                    "flattened_norm_recovered": flattened_norm,
                    "equivalence_requires_unit_axes": [
                        axis
                        for axis in range(len(norm_source_spec.shape))
                        if axis not in {feature_index, norm_reduction_axis}
                    ],
                    "source_ops": [item.op_id for item in chain],
                },
                provenance={
                    "compiler_pass": self.name,
                    "source_graph": graph.graph_id,
                    "source_ops": [item.op_id for item in chain],
                },
            )
            result = _rebuild_graph(
                graph,
                operators,
                matched_ids,
                fused,
                operators.index(batch_norm),
                fusion_kind=self.name,
            )
            return PassResult(
                result,
                (
                    PassDiagnostic(
                        "info",
                        self.name,
                        f"recovered torch-xla LayerNorm as '{output}' under row-wise equivalence",
                    ),
                ),
            )
        return PassResult(
            graph,
            (PassDiagnostic("info", self.name, "no equivalent torch-xla LayerNorm pattern found"),),
        )


class RecoverStableHLOFlattenedLinearPass:
    """Fold torch-xla's flatten/dot/bias/unflatten Linear representation."""

    name = "recover_stablehlo_flattened_linear"

    def run(self, graph: OperatorGraph) -> PassResult:
        operators = list(graph.operators)
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        producer = _producer_map(operators)
        consumers = _consumer_map(operators)
        replacements: dict[str, OperatorSpec] = {}
        tensor_replacements: dict[str, TensorSpec] = {}
        removed_operator_ids: set[str] = set()
        removed_tensors: set[str] = set()
        candidate_input_reshape_ids: set[str] = set()
        recovered: list[dict[str, object]] = []

        for unflatten in operators:
            if unflatten.normalized_type != "reshape" or len(unflatten.inputs) != 1:
                continue
            add = producer.get(unflatten.inputs[0])
            if add is None or not _is_elementwise(add, "stablehlo.add") or len(add.inputs) != 2:
                continue
            dot = next(
                (
                    producer.get(item)
                    for item in add.inputs
                    if producer.get(item) is not None
                    and producer[item].normalized_type in {"matmul", "batched_matmul"}
                ),
                None,
            )
            if dot is None or len(dot.inputs) != 2 or len(dot.outputs) != 1:
                continue
            bias = next((item for item in add.inputs if item != dot.outputs[0]), None)
            flatten = producer.get(dot.inputs[0])
            if (
                bias is None
                or flatten is None
                or flatten.normalized_type != "reshape"
                or len(flatten.inputs) != 1
                or consumers.get(dot.outputs[0], ()) != [add]
                or consumers.get(add.outputs[0], ()) != [unflatten]
            ):
                continue
            source = flatten.inputs[0]
            source_spec = tensors.get(source)
            flat_input_spec = tensors.get(flatten.outputs[0])
            dot_output_spec = tensors.get(dot.outputs[0])
            flat_output_spec = tensors.get(add.outputs[0])
            final_output = _single_output(unflatten)
            final_spec = tensors.get(final_output)
            rhs_spec = tensors.get(dot.inputs[1])
            bias_spec = tensors.get(bias)
            if any(
                item is None
                for item in (
                    source_spec,
                    flat_input_spec,
                    dot_output_spec,
                    flat_output_spec,
                    final_spec,
                    rhs_spec,
                    bias_spec,
                )
            ):
                continue
            if (
                len(source_spec.shape) < 3
                or len(flat_input_spec.shape) != 2
                or len(rhs_spec.shape) != 2
                or not all(isinstance(value, int) for value in source_spec.shape)
            ):
                continue
            leading = tuple(source_spec.shape[:-2])
            m_extent, k_extent = source_spec.shape[-2:]
            n_extent = final_spec.shape[-1]
            flat_rows = math.prod((*leading, m_extent))
            expected_input = (flat_rows, k_extent)
            expected_flat_output = (flat_rows, n_extent)
            expected_output = (*leading, m_extent, n_extent)
            if (
                flat_input_spec.shape != expected_input
                or dot_output_spec.shape != expected_flat_output
                or flat_output_spec.shape != expected_flat_output
                or final_spec.shape != expected_output
                or bias_spec.shape != (n_extent,)
                or dot.reduction_dims != (("K", k_extent),)
            ):
                continue

            iteration_dims = tuple(
                [(f"B{index}", extent) for index, extent in enumerate(leading)]
                + [("M", m_extent), ("N", n_extent)]
            )
            replacements[dot.op_id] = OperatorSpec(
                op_id=dot.op_id,
                op_type="batched_matmul",
                inputs=(source, dot.inputs[1]),
                outputs=dot.outputs,
                iteration_dims=iteration_dims,
                reduction_dims=(("K", k_extent),),
                attributes={
                    **dict(dot.attributes),
                    "input_occurrences": [source, dot.inputs[1]],
                    "rhs_broadcast_batch": True,
                    "flattened_linear_recovered": True,
                    "folded_input_reshape": flatten.op_id,
                    "folded_output_reshape": unflatten.op_id,
                },
                provenance={
                    **dict(dot.provenance),
                    "compiler_pass": self.name,
                    "folded_input_reshape": flatten.op_id,
                    "folded_output_reshape": unflatten.op_id,
                },
            )
            replacements[add.op_id] = OperatorSpec(
                op_id=add.op_id,
                op_type="elementwise",
                inputs=(dot.outputs[0], bias),
                outputs=(final_output,),
                iteration_dims=iteration_dims,
                reduction_dims=(),
                attributes={
                    **dict(add.attributes),
                    "input_occurrences": [dot.outputs[0], bias],
                    "flattened_linear_recovered": True,
                    "folded_output_reshape": unflatten.op_id,
                },
                provenance={
                    **dict(add.provenance),
                    "compiler_pass": self.name,
                    "folded_output_reshape": unflatten.op_id,
                },
            )
            tensor_replacements[dot.outputs[0]] = TensorSpec(
                name=dot.outputs[0],
                shape=expected_output,
                dtype=dot_output_spec.dtype,
                layout=dot_output_spec.layout,
                attributes={
                    **dict(dot_output_spec.attributes),
                    "view_recovered_from": dot_output_spec.shape,
                },
            )
            removed_operator_ids.add(unflatten.op_id)
            removed_tensors.add(add.outputs[0])
            candidate_input_reshape_ids.add(flatten.op_id)
            recovered.append(
                {
                    "dot": dot.op_id,
                    "bias_add": add.op_id,
                    "input_reshape": flatten.op_id,
                    "output_reshape": unflatten.op_id,
                }
            )

        if not replacements:
            return PassResult(
                graph,
                (PassDiagnostic("info", self.name, "no flattened StableHLO Linear pattern found"),),
            )

        rewritten_operators = [
            replacements.get(operator.op_id, operator)
            for operator in operators
            if operator.op_id not in removed_operator_ids
        ]
        rewritten_consumers = _consumer_map(rewritten_operators)
        for flatten in operators:
            if flatten.op_id not in candidate_input_reshape_ids or len(flatten.outputs) != 1:
                continue
            flat_output = flatten.outputs[0]
            if flat_output not in rewritten_consumers and flat_output not in graph.attributes.get("graph_outputs", ()):
                removed_operator_ids.add(flatten.op_id)
                removed_tensors.add(flat_output)
        rewritten_operators = [
            replacements.get(operator.op_id, operator)
            for operator in operators
            if operator.op_id not in removed_operator_ids
        ]
        rewritten_tuple = tuple(rewritten_operators)
        result = OperatorGraph(
            graph_id=graph.graph_id,
            tensors=tuple(
                tensor_replacements.get(tensor.name, tensor)
                for tensor in graph.tensors
                if tensor.name not in removed_tensors
            ),
            operators=rewritten_tuple,
            edges=_infer_edges(rewritten_tuple),
            attributes={
                **dict(graph.attributes),
                "canonical_passes": [*dict(graph.attributes).get("canonical_passes", ()), self.name],
                "view_recovery_groups": [
                    *dict(graph.attributes).get("view_recovery_groups", ()),
                    *recovered,
                ],
            },
        )
        issues = result.validate()
        if issues:
            raise ValueError(f"{self.name} produced an invalid graph: " + "; ".join(issues))
        return PassResult(
            result,
            (
                PassDiagnostic(
                    "info",
                    self.name,
                    f"recovered {len(recovered)} flattened StableHLO Linear operation(s)",
                ),
            ),
        )


def _rebuild_graph(
    graph: OperatorGraph,
    operators: list[OperatorSpec],
    matched_ids: set[str],
    fused: OperatorSpec,
    replacement_index: int,
    *,
    fusion_kind: str,
) -> OperatorGraph:
    rewritten: list[OperatorSpec] = []
    inserted = False
    for index, operator in enumerate(operators):
        if index == replacement_index:
            rewritten.append(fused)
            inserted = True
        if operator.op_id not in matched_ids:
            rewritten.append(operator)
    if not inserted:
        rewritten.append(fused)
    removed_tensors = {
        output
        for operator in operators
        if operator.op_id in matched_ids and operator.op_id != fused.op_id
        for output in operator.outputs
        if output not in fused.outputs
    }
    result = OperatorGraph(
        graph_id=graph.graph_id,
        tensors=tuple(tensor for tensor in graph.tensors if tensor.name not in removed_tensors),
        operators=tuple(rewritten),
        edges=_infer_edges(tuple(rewritten)),
        attributes={
            **dict(graph.attributes),
            "canonical_passes": [*dict(graph.attributes).get("canonical_passes", ()), fusion_kind],
        },
    )
    issues = result.validate()
    if issues:
        raise ValueError(f"{fusion_kind} produced an invalid graph: " + "; ".join(issues))
    return result


class RMSNormFusionPass:
    """Recognize ``x*x -> sum -> add(eps) -> rsqrt -> x*scale``."""

    name = "fuse_rmsnorm"
    repeat_until_stable = True

    def _try_general_pattern(
        self,
        graph: OperatorGraph,
        operators: list[OperatorSpec],
        tensors: dict[str, TensorSpec],
        producer: dict[str, OperatorSpec],
        consumers: dict[str, list[OperatorSpec]],
    ) -> PassResult | None:
        """Fuse Torch-XLA's power/reduction RMSNorm form when proven safe."""

        for final in operators:
            if not _is_elementwise_mul(final) or len(_input_occurrences(final)) != 2:
                continue
            final_inputs = _input_occurrences(final)
            for normalized_tensor in final_inputs:
                normalized = producer.get(normalized_tensor)
                if normalized is None or not _is_elementwise_mul(normalized):
                    continue
                normalized_inputs = _input_occurrences(normalized)
                if len(normalized_inputs) != 2:
                    continue
                source_tensor: str | None = None
                inverse_tensor: str | None = None
                inverse_op: OperatorSpec | None = None
                for candidate in normalized_inputs:
                    base, _reshapes = _unwrap_reshape(candidate, producer)
                    operation = producer.get(base)
                    if operation is not None and _is_elementwise(operation, "stablehlo.rsqrt"):
                        inverse_tensor, inverse_op = candidate, operation
                    else:
                        source_tensor = candidate
                if source_tensor is None or inverse_tensor is None or inverse_op is None:
                    continue
                weight_tensor = next(
                    (tensor for tensor in final_inputs if tensor != normalized_tensor), None
                )
                if weight_tensor is None:
                    continue

                inverse_base, inverse_reshapes = _unwrap_reshape(inverse_tensor, producer)
                rsqrt = producer.get(inverse_base)
                if rsqrt is None or not _is_elementwise(rsqrt, "stablehlo.rsqrt"):
                    continue
                rsqrt_inputs = _input_occurrences(rsqrt)
                if len(rsqrt_inputs) != 1:
                    continue
                add = producer.get(rsqrt_inputs[0])
                if add is None or not _is_elementwise_add(add):
                    continue
                variance_input = next(
                    (tensor for tensor in _input_occurrences(add)
                     if not _constant_tensor(tensor, tensors)),
                    None,
                )
                if variance_input is None:
                    continue
                variance_base, variance_reshapes = _unwrap_reshape(variance_input, producer)
                variance_operation = producer.get(variance_base)
                scale_operation: OperatorSpec | None = None
                reduce_tensor: str | None = None
                if variance_operation is not None and variance_operation.normalized_type == "reduce":
                    reduce_tensor = variance_base
                elif variance_operation is not None and _is_elementwise_mul(variance_operation):
                    scale_operation = variance_operation
                    reduce_tensor = next(
                        (tensor for tensor in _input_occurrences(variance_operation)
                         if not _constant_tensor(tensor, tensors)),
                        None,
                    )
                if reduce_tensor is None:
                    continue
                reduce = producer.get(reduce_tensor)
                if reduce is None or reduce.normalized_type != "reduce" or len(reduce.inputs) != 1:
                    continue
                square = producer.get(reduce.inputs[0])
                if square is None:
                    continue
                squared_source = _is_rmsnorm_square(square, tensors)
                if squared_source is None:
                    continue
                source_base, source_reshapes = _unwrap_reshape(source_tensor, producer)
                if squared_source != source_base:
                    continue
                source_spec = tensors.get(source_base)
                output_tensor = _single_output(final)
                output_spec = tensors.get(output_tensor)
                if source_spec is None or output_spec is None or source_spec.shape != output_spec.shape:
                    continue
                if not reduce.iteration_dims or len(reduce.reduction_dims) != 1:
                    continue
                if consumers.get(normalized_tensor, ()) != [final]:
                    continue
                if consumers.get(square.outputs[0], ()) != [reduce]:
                    continue
                if source_reshapes:
                    # A source-side view transform needs an explicit layout
                    # proof before it can be removed by semantic fusion.
                    continue
                bridge_operations = (
                    *source_reshapes,
                    square,
                    reduce,
                    *(tuple([scale_operation]) if scale_operation is not None else ()),
                    *variance_reshapes,
                    add,
                    rsqrt,
                    *inverse_reshapes,
                    normalized,
                    final,
                )
                matched_ids = {operation.op_id for operation in bridge_operations}
                if len(matched_ids) != len(bridge_operations):
                    continue
                source_ops = [operation.op_id for operation in bridge_operations]
                fused = OperatorSpec(
                    op_id=f"{final.op_id}.rmsnorm",
                    op_type="rmsnorm",
                    inputs=(source_base, weight_tensor),
                    outputs=(output_tensor,),
                    iteration_dims=reduce.iteration_dims,
                    reduction_dims=reduce.reduction_dims,
                    attributes={
                        "axis": reduce.reduction_dims[0][0],
                        "epsilon": (
                            _constant_scalar(add)
                            if _constant_scalar(add) is not None
                            else 1e-5
                        ),
                        "affine": True,
                        "fusion": "rmsnorm",
                        "source_ops": source_ops,
                    },
                    provenance={
                        "compiler_pass": self.name,
                        "source_graph": graph.graph_id,
                        "source_ops": source_ops,
                    },
                )
                result = _rebuild_graph(
                    graph,
                    operators,
                    matched_ids,
                    fused,
                    min(operators.index(operation) for operation in bridge_operations),
                    fusion_kind=self.name,
                )
                return PassResult(
                    result,
                    (
                        PassDiagnostic(
                            "info",
                            self.name,
                            f"fused {len(source_ops)} frontend nodes into '{fused.op_id}'",
                        ),
                    ),
                )
        return None

    def run(self, graph: OperatorGraph) -> PassResult:
        operators = list(graph.operators)
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        producer_by_tensor = {
            tensor: operator
            for operator in operators
            for tensor in operator.outputs
        }
        consumers: dict[str, list[OperatorSpec]] = {}
        for operator in operators:
            for tensor in operator.inputs:
                consumers.setdefault(tensor, []).append(operator)

        general = self._try_general_pattern(
            graph,
            operators,
            tensors,
            producer_by_tensor,
            consumers,
        )
        if general is not None:
            return general

        for final in operators:
            if not _is_elementwise_mul(final):
                continue
            final_occurrences = _input_occurrences(final)
            if len(final_occurrences) != 2:
                continue
            for rsqrt_tensor in final.inputs:
                rsqrt = producer_by_tensor.get(rsqrt_tensor)
                if rsqrt is None or not any(
                    token in _frontend_target(rsqrt)
                    for token in ("aten.rsqrt", "stablehlo.rsqrt")
                ):
                    continue
                rsqrt_inputs = _input_occurrences(rsqrt)
                if len(rsqrt_inputs) != 1:
                    continue
                add = producer_by_tensor.get(rsqrt_inputs[0])
                if add is None or not _is_elementwise_add(add) or len(add.inputs) != 1:
                    continue
                normalization = producer_by_tensor.get(add.inputs[0])
                mean = (
                    normalization
                    if normalization is not None
                    and _is_elementwise(normalization, "stablehlo.divide")
                    and len(normalization.inputs) == 1
                    else None
                )
                reduce = producer_by_tensor.get(mean.inputs[0]) if mean is not None else normalization
                if reduce is None or reduce.normalized_type != "reduce" or len(reduce.inputs) != 1:
                    continue
                square = producer_by_tensor.get(reduce.inputs[0])
                if square is None or not _is_elementwise_mul(square):
                    continue
                square_occurrences = _input_occurrences(square)
                if len(square_occurrences) != 2 or square_occurrences[0] != square_occurrences[1]:
                    continue
                input_tensor = square_occurrences[0]
                if input_tensor not in tensors or input_tensor not in final_occurrences:
                    continue
                if len(final_occurrences) != 2 or rsqrt_tensor not in final_occurrences:
                    continue
                output_tensor = _single_output(final)
                input_spec = tensors[input_tensor]
                output_spec = tensors.get(output_tensor)
                if output_spec is None or output_spec.shape != input_spec.shape:
                    continue
                if not reduce.iteration_dims or len(reduce.reduction_dims) != 1:
                    continue
                internal = (square, reduce, *((mean,) if mean is not None else ()), add, rsqrt)
                reduce_consumer = mean if mean is not None else add
                if any(
                    consumers.get(_single_output(operator), ())
                    != [next_op]
                    for operator, next_op in (
                        (square, reduce),
                        (reduce, reduce_consumer),
                        *((((mean, add),) if mean is not None else ())),
                        (add, rsqrt),
                        (rsqrt, final),
                    )
                ):
                    continue
                epsilon = _constant_scalar(add)
                matched_ids = {operator.op_id for operator in (*internal, final)}
                source_ops = [operator.op_id for operator in (*internal, final)]
                fused = OperatorSpec(
                    op_id=f"{final.op_id}.rmsnorm",
                    op_type="rmsnorm",
                    inputs=(input_tensor,),
                    outputs=(output_tensor,),
                    iteration_dims=reduce.iteration_dims,
                    reduction_dims=reduce.reduction_dims,
                    attributes={
                        "axis": reduce.reduction_dims[0][0],
                        "epsilon": epsilon if epsilon is not None else 1e-5,
                        "fusion": "rmsnorm",
                        "source_ops": source_ops,
                    },
                    provenance={
                        "compiler_pass": self.name,
                        "source_graph": graph.graph_id,
                        "source_ops": source_ops,
                    },
                )
                replacement_index = min(operators.index(operator) for operator in internal)
                rewritten: list[OperatorSpec] = []
                for index, operator in enumerate(operators):
                    if index == replacement_index:
                        rewritten.append(fused)
                    if operator.op_id not in matched_ids:
                        rewritten.append(operator)
                kept_tensors = tuple(
                    tensor
                    for tensor in graph.tensors
                    if tensor.name not in {
                        _single_output(operator) for operator in internal
                    }
                )
                edges: list[DataEdge] = []
                seen_edges: set[tuple[str, str, str]] = set()
                for operator in rewritten:
                    for tensor in operator.inputs:
                        producer = next(
                            (candidate.op_id for candidate in rewritten if tensor in candidate.outputs),
                            None,
                        )
                        if producer is None or producer == operator.op_id:
                            continue
                        key = (producer, operator.op_id, tensor)
                        if key not in seen_edges:
                            edges.append(DataEdge(*key))
                            seen_edges.add(key)
                attributes = {
                    **dict(graph.attributes),
                    "canonical_passes": [*dict(graph.attributes).get("canonical_passes", ()), self.name],
                    "fusion_groups": [
                        *dict(graph.attributes).get("fusion_groups", ()),
                        {"kind": "rmsnorm", "operator": fused.op_id, "source_ops": source_ops},
                    ],
                }
                result = OperatorGraph(
                    graph_id=graph.graph_id,
                    tensors=kept_tensors,
                    operators=tuple(rewritten),
                    edges=tuple(edges),
                    attributes=attributes,
                )
                issues = result.validate()
                if issues:
                    raise ValueError("fuse_rmsnorm produced an invalid graph: " + "; ".join(issues))
                return PassResult(
                    result,
                    (
                        PassDiagnostic(
                            "info",
                            self.name,
                            f"fused {len(source_ops)} frontend nodes into '{fused.op_id}'",
                        ),
                    ),
                )

        return PassResult(
            graph,
            (PassDiagnostic("info", self.name, "no supported RMSNorm composite pattern found"),),
        )


class SoftmaxFusionPass:
    """Recover stable softmax from max/subtract/exp/sum/divide primitives."""

    name = "fuse_softmax"

    def run(self, graph: OperatorGraph) -> PassResult:
        operators = list(graph.operators)
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        producer = _producer_map(operators)
        consumers = _consumer_map(operators)
        for final in operators:
            if not _is_elementwise(final, "stablehlo.divide") or len(final.inputs) != 2:
                continue
            exp = producer.get(final.inputs[0])
            total = producer.get(final.inputs[1])
            if exp is None or total is None or not _is_elementwise(exp, "stablehlo.exponential"):
                continue
            if total.normalized_type != "reduce" or total.attributes.get("reducer") != "add":
                continue
            if not total.inputs or total.inputs[0] != exp.outputs[0]:
                continue
            shifted = producer.get(exp.inputs[0]) if exp.inputs else None
            if shifted is None or not _is_elementwise(shifted, "stablehlo.subtract") or len(shifted.inputs) != 2:
                continue
            source = shifted.inputs[0]
            maximum = producer.get(shifted.inputs[1])
            if maximum is None or maximum.normalized_type != "reduce":
                continue
            if maximum.attributes.get("reducer") != "maximum" or maximum.inputs != (source,):
                continue
            if maximum.reduction_dims != total.reduction_dims:
                continue
            output = _single_output(final)
            source_spec = tensors.get(source)
            output_spec = tensors.get(output)
            if source_spec is None or output_spec is None or source_spec.shape != output_spec.shape:
                continue
            chain = (maximum, shifted, exp, total, final)
            chain_ids = {item.op_id for item in chain}
            if any(
                consumers.get(_single_output(item), ()) != [next_item]
                for item, next_item in ((maximum, shifted), (shifted, exp), (total, final))
            ):
                continue
            exp_consumers = consumers.get(_single_output(exp), ())
            if len(exp_consumers) != 2 or total not in exp_consumers or final not in exp_consumers:
                continue
            axes = [int(name[1:]) for name, _ in maximum.reduction_dims if name.startswith("d") and name[1:].isdigit()]
            fused = OperatorSpec(
                op_id=output,
                op_type="softmax",
                inputs=(source,),
                outputs=(output,),
                iteration_dims=maximum.iteration_dims,
                reduction_dims=maximum.reduction_dims,
                attributes={
                    "frontend_target": "stablehlo.softmax",
                    "axes": axes,
                    "fusion": "softmax",
                    "source_ops": [item.op_id for item in chain],
                },
                provenance={
                    "compiler_pass": self.name,
                    "source_graph": graph.graph_id,
                    "source_ops": [item.op_id for item in chain],
                },
            )
            result = _rebuild_graph(
                graph,
                operators,
                chain_ids,
                fused,
                min(operators.index(item) for item in chain),
                fusion_kind=self.name,
            )
            return PassResult(
                result,
                (PassDiagnostic("info", self.name, f"fused {len(chain)} primitives into '{output}'"),),
            )
        return PassResult(graph, (PassDiagnostic("info", self.name, "no supported Softmax composite pattern found"),))


class LayerNormFusionPass:
    """Recover LayerNorm from the primitive mean/variance normalization chain."""

    name = "fuse_layernorm"

    def run(self, graph: OperatorGraph) -> PassResult:
        operators = list(graph.operators)
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        producer = _producer_map(operators)
        consumers = _consumer_map(operators)
        for final in operators:
            if not _is_elementwise(final, "stablehlo.add", "stablehlo.multiply"):
                continue
            affine = _is_elementwise(final, "stablehlo.add") and len(final.inputs) == 2
            normalized_tensor: str | None
            affine_ops: list[OperatorSpec] = []
            affine_inputs: tuple[str, ...]
            if affine:
                scaled = producer.get(final.inputs[0])
                if scaled is None or not _is_elementwise(scaled, "stablehlo.multiply") or len(scaled.inputs) != 2:
                    continue
                normalized_tensor = scaled.inputs[0]
                affine_inputs = (scaled.inputs[1], final.inputs[1])
                affine_ops = [scaled, final]
            else:
                if not _is_elementwise(final, "stablehlo.multiply") or len(final.inputs) != 2:
                    continue
                # A normalized value feeding affine scale/bias is an internal
                # node; only fuse it as the final result when it has no users.
                if consumers.get(_single_output(final), ()):
                    continue
                normalized_tensor = None
                affine_inputs = ()
            normalized = producer.get(normalized_tensor) if normalized_tensor else final
            if normalized is None or not _is_elementwise(normalized, "stablehlo.multiply") or len(normalized.inputs) != 2:
                continue
            center = producer.get(normalized.inputs[0])
            inverse = producer.get(normalized.inputs[1])
            if center is None or inverse is None or not _is_elementwise(center, "stablehlo.subtract"):
                continue
            if not _is_elementwise(inverse, "stablehlo.rsqrt"):
                continue
            source, mean_tensor = center.inputs
            mean = producer.get(mean_tensor)
            if mean is None or not _is_elementwise(mean, "stablehlo.divide") or len(mean.inputs) != 1:
                continue
            sum_op = producer.get(mean.inputs[0])
            if sum_op is None or sum_op.normalized_type != "reduce" or sum_op.inputs != (source,):
                continue
            variance_eps = producer.get(inverse.inputs[0])
            if variance_eps is None or not _is_elementwise(variance_eps, "stablehlo.add") or len(variance_eps.inputs) != 1:
                continue
            variance = producer.get(variance_eps.inputs[0])
            variance_sum = producer.get(variance.inputs[0]) if variance and variance.inputs else None
            square = producer.get(variance_sum.inputs[0]) if variance_sum and variance_sum.inputs else None
            if (
                variance is None
                or not _is_elementwise(variance, "stablehlo.divide")
                or variance_sum is None
                or variance_sum.normalized_type != "reduce"
                or square is None
                or not _is_elementwise(square, "stablehlo.multiply")
                or _input_occurrences(square) != (center.outputs[0], center.outputs[0])
                or sum_op.attributes.get("reducer") != "add"
                or variance_sum.attributes.get("reducer") != "add"
            ):
                continue
            if sum_op.reduction_dims != variance_sum.reduction_dims:
                continue
            square_users = consumers.get(square.outputs[0], ())
            if square_users != [variance_sum]:
                continue
            output = _single_output(final)
            source_spec = tensors.get(source)
            output_spec = tensors.get(output)
            if source_spec is None or output_spec is None or source_spec.shape != output_spec.shape:
                continue
            chain = [sum_op, mean, center, square, variance_sum, variance, variance_eps, inverse, normalized, *affine_ops]
            chain_ids = {item.op_id for item in chain}
            strict_chain = (
                (sum_op, mean),
                (mean, center),
                (variance_sum, variance),
                (variance, variance_eps),
                (variance_eps, inverse),
                (inverse, normalized),
            )
            if any(consumers.get(_single_output(item), ()) != [next_item] for item, next_item in strict_chain):
                continue
            center_consumers = consumers.get(_single_output(center), ())
            if len(center_consumers) != 2 or square not in center_consumers or normalized not in center_consumers:
                continue
            epsilon = _constant_scalar(variance_eps)
            axis = sum_op.reduction_dims[0][0] if sum_op.reduction_dims else "d0"
            fused = OperatorSpec(
                op_id=output,
                op_type="layernorm",
                inputs=(source, *affine_inputs),
                outputs=(output,),
                iteration_dims=sum_op.iteration_dims,
                reduction_dims=sum_op.reduction_dims,
                attributes={
                    "frontend_target": "stablehlo.layer_norm",
                    "axis": axis,
                    "epsilon": epsilon if epsilon is not None else 1e-5,
                    "affine": bool(affine_inputs),
                    "fusion": "layernorm",
                    "source_ops": [item.op_id for item in chain],
                },
                provenance={
                    "compiler_pass": self.name,
                    "source_graph": graph.graph_id,
                    "source_ops": [item.op_id for item in chain],
                },
            )
            result = _rebuild_graph(
                graph,
                operators,
                chain_ids,
                fused,
                min(operators.index(item) for item in chain),
                fusion_kind=self.name,
            )
            return PassResult(
                result,
                (PassDiagnostic("info", self.name, f"fused {len(chain)} primitives into '{output}'"),),
            )
        return PassResult(graph, (PassDiagnostic("info", self.name, "no supported LayerNorm composite pattern found"),))


def _tensor_elements(tensor: TensorSpec | None) -> int | None:
    if tensor is None or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in tensor.shape
    ):
        return None
    return math.prod(tensor.shape)


class AttentionRegionPass:
    """Recover an observable QK/Softmax/PV attention region.

    This pass annotates existing operators instead of replacing them.  The
    region is therefore useful to GC/FC analysis while every member remains a
    scheduler-visible TISA operation.
    """

    name = "recover_attention_region"

    @staticmethod
    def _upstream_qk_path(
        tensor: str,
        producer: dict[str, OperatorSpec],
    ) -> tuple[OperatorSpec, tuple[OperatorSpec, ...]] | None:
        def visit(
            current: str,
            active: frozenset[str],
        ) -> list[tuple[OperatorSpec, tuple[OperatorSpec, ...]]]:
            operation = producer.get(current)
            if operation is None or operation.op_id in active:
                return []
            if operation.normalized_type in {"matmul", "batched_matmul"}:
                return [(operation, ())]
            if operation.normalized_type not in {"reshape", "transpose", "elementwise"}:
                return []
            candidates: list[tuple[OperatorSpec, tuple[OperatorSpec, ...]]] = []
            for input_tensor in operation.inputs:
                for qk, path in visit(input_tensor, active | {operation.op_id}):
                    candidates.append((qk, (*path, operation)))
            return candidates

        candidates = visit(tensor, frozenset())
        by_qk: dict[str, list[tuple[OperatorSpec, ...]]] = {}
        qk_by_id: dict[str, OperatorSpec] = {}
        for qk, path in candidates:
            qk_by_id[qk.op_id] = qk
            by_qk.setdefault(qk.op_id, []).append(path)
        if len(by_qk) != 1:
            return None
        qk_id = next(iter(by_qk))
        path = min(by_qk[qk_id], key=lambda items: (len(items), tuple(op.op_id for op in items)))
        return qk_by_id[qk_id], path

    @staticmethod
    def _downstream_pv_path(
        tensor: str,
        consumers: dict[str, list[OperatorSpec]],
    ) -> tuple[OperatorSpec, tuple[OperatorSpec, ...], str] | None:
        candidates: list[tuple[OperatorSpec, tuple[OperatorSpec, ...], str]] = []

        def visit(
            current: str,
            path: tuple[OperatorSpec, ...],
            active: frozenset[str],
        ) -> None:
            for operation in consumers.get(current, ()):
                if operation.op_id in active:
                    continue
                if operation.normalized_type in {"matmul", "batched_matmul"}:
                    candidates.append((operation, path, current))
                    continue
                if (
                    operation.normalized_type in {"reshape", "transpose"}
                    and len(operation.outputs) == 1
                ):
                    visit(
                        operation.outputs[0],
                        (*path, operation),
                        active | {operation.op_id},
                    )

        visit(tensor, (), frozenset())
        unique = {
            (pv.op_id, tuple(operation.op_id for operation in path), input_tensor):
            (pv, path, input_tensor)
            for pv, path, input_tensor in candidates
        }
        return next(iter(unique.values())) if len(unique) == 1 else None

    def run(self, graph: OperatorGraph) -> PassResult:
        operators = list(graph.operators)
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        producer = _producer_map(operators)
        consumers = _consumer_map(operators)
        regions: list[dict[str, object]] = []
        roles_by_operator: dict[str, tuple[str, str]] = {}
        claimed: set[str] = set()

        for softmax in operators:
            if softmax.normalized_type != "softmax" or len(softmax.inputs) != 1:
                continue
            upstream = self._upstream_qk_path(softmax.inputs[0], producer)
            downstream = self._downstream_pv_path(_single_output(softmax), consumers)
            if upstream is None or downstream is None:
                continue
            qk, score_path = upstream
            pv, probability_path, probability_tensor = downstream
            members = (qk, *score_path, softmax, *probability_path, pv)
            member_ids = [operation.op_id for operation in members]
            if len(set(member_ids)) != len(member_ids) or claimed.intersection(member_ids):
                continue
            qk_elements = _tensor_elements(tensors.get(_single_output(qk)))
            score_elements = _tensor_elements(tensors.get(softmax.inputs[0]))
            probability_elements = _tensor_elements(tensors.get(_single_output(softmax)))
            pv_input_elements = _tensor_elements(tensors.get(probability_tensor))
            if (
                qk_elements is None
                or qk_elements != score_elements
                or probability_elements is None
                or probability_elements != pv_input_elements
                or tensors[_single_output(softmax)].shape != tensors[softmax.inputs[0]].shape
                or probability_tensor not in pv.inputs
                or len(softmax.reduction_dims) != 1
            ):
                continue

            region_id = f"attention_region_{len(regions):04d}"
            claimed.update(member_ids)
            roles_by_operator[qk.op_id] = (region_id, "qk_matmul")
            roles_by_operator[softmax.op_id] = (region_id, "softmax")
            roles_by_operator[pv.op_id] = (region_id, "pv_matmul")
            for operation in score_path:
                roles_by_operator[operation.op_id] = (region_id, "score_transform")
            for operation in probability_path:
                roles_by_operator[operation.op_id] = (region_id, "probability_transform")

            member_set = set(member_ids)
            external_inputs = tuple(
                dict.fromkeys(
                    tensor
                    for operation in members
                    for tensor in operation.inputs
                    if producer.get(tensor) is None
                    or producer[tensor].op_id not in member_set
                )
            )
            regions.append(
                {
                    "region_id": region_id,
                    "semantic_family": "attention",
                    "members": member_ids,
                    "roles": {
                        "qk_matmul": qk.op_id,
                        "score_transforms": [operation.op_id for operation in score_path],
                        "softmax": softmax.op_id,
                        "probability_transforms": [
                            operation.op_id for operation in probability_path
                        ],
                        "pv_matmul": pv.op_id,
                    },
                    "inputs": list(external_inputs),
                    "outputs": list(pv.outputs),
                    "scheduler_visibility": "member_tisa_instructions",
                    "opaque": False,
                }
            )

        if not regions:
            return PassResult(
                graph,
                (PassDiagnostic("info", self.name, "no supported Attention region found"),),
            )

        annotated = tuple(
            replace(
                operation,
                attributes={
                    **dict(operation.attributes),
                    "semantic_region_id": roles_by_operator[operation.op_id][0],
                    "semantic_region_family": "attention",
                    "semantic_region_role": roles_by_operator[operation.op_id][1],
                    "semantic_region_opaque": False,
                },
            )
            if operation.op_id in roles_by_operator
            else operation
            for operation in graph.operators
        )
        result = OperatorGraph(
            graph_id=graph.graph_id,
            tensors=graph.tensors,
            operators=annotated,
            edges=graph.edges,
            attributes={
                **dict(graph.attributes),
                "canonical_passes": [
                    *dict(graph.attributes).get("canonical_passes", ()),
                    self.name,
                ],
                "semantic_regions": [
                    *dict(graph.attributes).get("semantic_regions", ()),
                    *regions,
                ],
            },
        )
        issues = result.validate()
        if issues:
            raise ValueError("recover_attention_region produced an invalid graph: " + "; ".join(issues))
        return PassResult(
            result,
            (
                PassDiagnostic(
                    "info",
                    self.name,
                    f"recovered {len(regions)} observable Attention region(s)",
                ),
            ),
        )


class SwiGLUFusionPass:
    """Recover ``silu(gate) * up`` as a vector semantic operator."""

    name = "fuse_swiglu"
    repeat_until_stable = True

    def run(self, graph: OperatorGraph) -> PassResult:
        operators = list(graph.operators)
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        producer = _producer_map(operators)
        consumers = _consumer_map(operators)
        for final in operators:
            if not _is_elementwise_mul(final):
                continue
            final_inputs = _input_occurrences(final)
            if len(final_inputs) != 2:
                continue
            for silu_tensor in final_inputs:
                silu_base, silu_converts = _unwrap_convert_chain(silu_tensor, producer)
                silu = producer.get(silu_base)
                if silu is None or not _is_elementwise_mul(silu):
                    continue
                silu_inputs = _input_occurrences(silu)
                if len(silu_inputs) != 2:
                    continue
                logistic = next(
                    (
                        producer.get(tensor)
                        for tensor in silu_inputs
                        if producer.get(tensor) is not None
                        and _is_elementwise(producer[tensor], "stablehlo.logistic")
                    ),
                    None,
                )
                if logistic is None or len(logistic.inputs) != 1:
                    continue
                gate_tensor = logistic.inputs[0]
                if sorted(silu_inputs) != sorted((gate_tensor, _single_output(logistic))):
                    continue
                up_tensor = final_inputs[1] if final_inputs[0] == silu_tensor else final_inputs[0]
                output_tensor = _single_output(final)
                gate_spec = tensors.get(gate_tensor)
                up_spec = tensors.get(up_tensor)
                output_spec = tensors.get(output_tensor)
                if (
                    gate_spec is None
                    or up_spec is None
                    or output_spec is None
                    or gate_spec.shape != up_spec.shape
                    or gate_spec.shape != output_spec.shape
                    or gate_spec.dtype != up_spec.dtype
                    or gate_spec.dtype != output_spec.dtype
                    or consumers.get(_single_output(logistic), ()) != [silu]
                ):
                    continue
                conversion_chain = tuple(reversed(silu_converts))
                value_chain = (silu, *conversion_chain, final)
                if any(
                    consumers.get(_single_output(current), ()) != [successor]
                    for current, successor in zip(value_chain, value_chain[1:])
                ):
                    continue
                conversion_steps = [
                    {
                        "source_dtype": operation.attributes.get("source_dtype"),
                        "target_dtype": operation.attributes.get("target_dtype"),
                    }
                    for operation in conversion_chain
                ]
                if any(
                    not step["source_dtype"] or not step["target_dtype"]
                    for step in conversion_steps
                ):
                    continue
                matched_operations = (logistic, silu, *conversion_chain, final)
                source_ops = [operation.op_id for operation in matched_operations]
                fused = OperatorSpec(
                    op_id=f"{final.op_id}.swiglu",
                    op_type="swiglu",
                    inputs=(gate_tensor, up_tensor),
                    outputs=(output_tensor,),
                    iteration_dims=final.iteration_dims,
                    attributes={
                        "frontend_target": "semantic.swiglu",
                        "semantic_family": "swiglu",
                        "semantic_op": "swiglu",
                        "operand_arity": 2,
                        "backend_capability_key": "swiglu",
                        "activation": "silu",
                        "conversion_steps": conversion_steps,
                        "fusion": "swiglu",
                        "source_ops": source_ops,
                    },
                    provenance={
                        "compiler_pass": self.name,
                        "source_graph": graph.graph_id,
                        "source_ops": source_ops,
                    },
                )
                result = _rebuild_graph(
                    graph,
                    operators,
                    {operation.op_id for operation in matched_operations},
                    fused,
                    min(operators.index(operation) for operation in matched_operations),
                    fusion_kind=self.name,
                )
                return PassResult(
                    result,
                    (
                        PassDiagnostic(
                            "info",
                            self.name,
                            f"fused SwiGLU primitives into '{fused.op_id}'",
                        ),
                    ),
                )

        return PassResult(
            graph,
            (PassDiagnostic("info", self.name, "no supported SwiGLU pattern found"),),
        )


class PassManager:
    """Run ordered graph passes with stable per-pass diagnostics."""

    def __init__(self, passes: tuple[GraphPass, ...] | None = None) -> None:
        self.passes = _default_pipeline() if passes is None else passes

    def run(self, graph: OperatorGraph) -> PassResult:
        current = graph
        diagnostics: list[PassDiagnostic] = []
        snapshots: list[PassSnapshot] = []
        pass_index = 0
        for graph_pass in self.passes:
            repeat_limit = len(current.operators) + 1
            for _iteration in range(repeat_limit):
                input_graph = current
                result = graph_pass.run(current)
                current = result.graph
                diagnostics.extend(result.diagnostics)
                snapshots.append(
                    PassSnapshot(
                        pass_index=pass_index,
                        pass_name=graph_pass.name,
                        input_graph=input_graph,
                        output_graph=current,
                        diagnostics=result.diagnostics,
                    )
                )
                pass_index += 1
                if not getattr(graph_pass, "repeat_until_stable", False):
                    break
                if current.to_dict() == input_graph.to_dict():
                    break
        return PassResult(current, tuple(diagnostics), tuple(snapshots))


def _default_pipeline(
    registry: SemanticFusionPatternRegistry | None = None,
) -> tuple[GraphPass, ...]:
    """Compose structural GC passes with registered semantic patterns.

    The import is intentionally lazy: ``fusion_patterns`` imports the pass
    classes, while this module owns ``PassManager``.  Delaying the registry
    import keeps both modules independently usable without a circular import.
    """

    if registry is None:
        from .fusion_patterns import default_semantic_fusion_registry

        registry = default_semantic_fusion_registry()

    structural = (
        (10, 0, CanonicalizeGraphPass()),
        (15, 1, LinearDecompositionPass()),
        (30, 2, RecoverStableHLOFlattenedLinearPass()),
        (40, 3, FoldTransposeIntoMatmulPass()),
    )
    issues = registry.validate()
    if issues:
        raise ValueError("invalid semantic fusion registry: " + "; ".join(issues))
    semantic = tuple(
        (pattern.priority, index + len(structural), pattern.graph_pass)
        for index, pattern in enumerate(registry.patterns())
    )
    return tuple(
        graph_pass
        for _priority, _order, graph_pass in sorted(
            (*structural, *semantic), key=lambda item: (item[0], item[1])
        )
    )


def default_pass_manager(
    *, fusion_registry: SemanticFusionPatternRegistry | None = None
) -> PassManager:
    """Build the default GC pass manager, optionally with a custom registry."""

    return PassManager(_default_pipeline(fusion_registry))


__all__ = [
    "AttentionRegionPass",
    "CanonicalizeGraphPass",
    "GraphPass",
    "PassDiagnostic",
    "PassManager",
    "PassSnapshot",
    "PassResult",
    "RecoverStableHLOFlattenedLinearPass",
    "RecoverStableHLOLayerNormPass",
    "LayerNormFusionPass",
    "RMSNormFusionPass",
    "SoftmaxFusionPass",
    "SwiGLUFusionPass",
    "default_pass_manager",
]
