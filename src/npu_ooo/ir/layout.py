"""Tensor layout resolution shared by compiler, lowering and runtime.

StableHLO tensor encodings are dialect-defined metadata.  The compiler may
only turn an encoding into byte strides when the encoding carries enough
information to prove the mapping.  Unknown encodings remain logical and are
handled conservatively by callers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

from .dtype import dtype_bytes


@dataclass(frozen=True)
class LayoutInfo:
    """Resolved layout contract for one concrete tensor shape."""

    shape: tuple[int, ...]
    dtype: str
    layout: str
    source: str
    encoding: str | None
    strides_bytes: tuple[int, ...] | None
    stride_unit: str | None = None
    offset_bytes: int = 0
    reason: str | None = None

    @property
    def concrete(self) -> bool:
        return self.strides_bytes is not None

    @property
    def contiguous(self) -> bool:
        """Whether logical axes follow dense row-major element order."""

        return self.strides_bytes == _dense_strides(self.shape, self.element_size_bytes)

    @property
    def element_size_bytes(self) -> int:
        return dtype_bytes(self.dtype, default=2)

    @property
    def allocation_size_bytes(self) -> int:
        """Return the byte span needed for the complete logical tensor."""

        if self.strides_bytes is None:
            return math.prod(self.shape) * self.element_size_bytes
        if not self.shape:
            return self.offset_bytes + self.element_size_bytes
        return self.offset_bytes + self.element_size_bytes + sum(
            (extent - 1) * stride
            for extent, stride in zip(self.shape, self.strides_bytes)
        )

    def interval(
        self,
        starts: Sequence[int],
        shape: Sequence[int],
        *,
        strides_bytes: Sequence[int] | None = None,
    ) -> tuple[int, int] | None:
        """Resolve a logical region to ``(offset, span)`` in bytes.

        ``strides_bytes`` is used for transformed accesses such as a strided
        slice.  It must have the same rank as the tensor and is expressed in
        the source tensor's physical coordinates.
        """

        if self.strides_bytes is None:
            return None
        starts_tuple = tuple(int(value) for value in starts)
        shape_tuple = tuple(int(value) for value in shape)
        if len(starts_tuple) != len(self.shape) or len(shape_tuple) != len(self.shape):
            return None
        if any(
            start < 0 or extent <= 0 or start + extent > limit
            for start, extent, limit in zip(starts_tuple, shape_tuple, self.shape)
        ):
            return None
        strides = tuple(self.strides_bytes if strides_bytes is None else strides_bytes)
        if len(strides) != len(self.shape) or any(stride < 0 for stride in strides):
            return None
        # The logical start is always expressed in the source tensor's
        # coordinates.  A transformed access may use different strides only
        # for the span calculation; its origin still uses the base layout.
        offset = self.offset_bytes + sum(
            start * base_stride
            for start, base_stride in zip(starts_tuple, self.strides_bytes)
        )
        span = self.element_size_bytes + sum(
            (extent - 1) * stride
            for extent, stride in zip(shape_tuple, strides)
        )
        return offset, span


_INT_LIST = r"\[\s*([^\]]*?)\s*\]"


def _integer_list(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, (tuple, list)):
        try:
            values = tuple(int(item) for item in value)
        except (TypeError, ValueError):
            return None
        if any(isinstance(item, bool) or item < 0 for item in values):
            return None
        return values
    if not isinstance(value, str):
        return None
    match = re.search(_INT_LIST, value)
    if match is None:
        return None
    raw = [item.strip() for item in match.group(1).split(",") if item.strip()]
    try:
        values = tuple(int(item) for item in raw)
    except ValueError:
        return None
    return values if all(item >= 0 for item in values) else None


def _dense_strides(shape: tuple[int, ...], element_size: int) -> tuple[int, ...]:
    result: list[int] = []
    stride = element_size
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def _ordered_strides(
    shape: tuple[int, ...],
    element_size: int,
    minor_to_major: tuple[int, ...],
) -> tuple[int, ...] | None:
    if len(minor_to_major) != len(shape) or sorted(minor_to_major) != list(range(len(shape))):
        return None
    strides = [0] * len(shape)
    stride = element_size
    for axis in minor_to_major:
        strides[axis] = stride
        stride *= shape[axis]
    return tuple(strides)


def _encoding_fields(encoding: str) -> dict[str, Any]:
    """Extract the small, dialect-independent structured subset.

    This intentionally accepts common MLIR spellings such as
    ``minor_to_major = [1, 0]`` and ``strides_bytes = [16, 4]``.  Affine maps
    and opaque attribute names are left unresolved.
    """

    fields: dict[str, Any] = {}
    for name in ("strides_bytes", "byte_strides", "strides", "minor_to_major", "permutation", "order"):
        match = re.search(rf"\b{name}\s*=\s*{_INT_LIST}", encoding)
        if match:
            fields[name] = _integer_list(match.group(0))
    strided = re.search(r"\bstrided\s*<\s*\[([^\]]*)\]", encoding)
    if strided:
        fields["strides"] = _integer_list("[" + strided.group(1) + "]")
    offset = re.search(r"\boffset\s*[:=]\s*(-?\d+)", encoding)
    if offset:
        fields["offset"] = int(offset.group(1))
    unit = re.search(r"\b(?:stride_)?unit\s*=\s*([A-Za-z_]+)", encoding)
    if unit:
        fields["stride_unit"] = unit.group(1).lower()
    if re.search(r"\b(?:byte_strides|strides_bytes)\b", encoding):
        fields["stride_unit"] = "bytes"
    return fields


def resolve_layout(
    shape: Sequence[int],
    dtype: str,
    *,
    layout: str = "dense",
    attributes: Mapping[str, Any] | None = None,
) -> LayoutInfo:
    """Resolve a tensor's layout into concrete byte strides when possible.

    Precedence is explicit ``strides_bytes`` metadata, structured StableHLO
    encoding, named layout, then dense row-major.  A non-structured encoding
    is intentionally reported as opaque instead of guessed.
    """

    concrete_shape = tuple(int(value) for value in shape)
    if any(value <= 0 for value in concrete_shape):
        raise ValueError("layout shape must contain positive integers")
    attrs = attributes if isinstance(attributes, Mapping) else {}
    element_size = dtype_bytes(dtype, default=2)
    encoding = attrs.get("layout_encoding")
    encoding_text = str(encoding) if encoding not in {None, ""} else None
    source = str(attrs.get("layout_source", "default_dense"))
    explicit_offset = attrs.get("layout_offset_bytes", 0)
    if isinstance(explicit_offset, bool) or not isinstance(explicit_offset, int) or explicit_offset < 0:
        raise ValueError("layout_offset_bytes must be a non-negative integer")

    explicit = _integer_list(attrs.get("strides_bytes"))
    if attrs.get("strides_bytes") is not None:
        if explicit is None or len(explicit) != len(concrete_shape):
            raise ValueError("strides_bytes rank and values must match tensor shape")
        return LayoutInfo(
            concrete_shape,
            dtype,
            str(layout),
            "explicit_strides",
            encoding_text,
            explicit,
            "bytes",
            explicit_offset,
        )

    if encoding_text:
        fields = _encoding_fields(encoding_text)
        encoded_bytes = fields.get("strides_bytes") or fields.get("byte_strides")
        if encoded_bytes is not None:
            if len(encoded_bytes) != len(concrete_shape):
                raise ValueError("StableHLO byte-stride encoding rank does not match tensor shape")
            return LayoutInfo(
                concrete_shape,
                dtype,
                f"stablehlo:{encoding_text}",
                "stablehlo_encoding",
                encoding_text,
                encoded_bytes,
                "bytes",
                int(fields.get("offset", 0)),
            )
        encoded_strides = fields.get("strides")
        if encoded_strides is not None:
            if len(encoded_strides) != len(concrete_shape):
                raise ValueError("StableHLO stride encoding rank does not match tensor shape")
            unit = fields.get("stride_unit", "elements")
            if unit in {"byte", "bytes"}:
                strides = encoded_strides
            elif unit in {"element", "elements"}:
                strides = tuple(value * element_size for value in encoded_strides)
            else:
                raise ValueError(f"unsupported StableHLO stride unit '{unit}'")
            return LayoutInfo(
                concrete_shape,
                dtype,
                f"stablehlo:{encoding_text}",
                "stablehlo_encoding",
                encoding_text,
                strides,
                str(unit),
                int(fields.get("offset", 0)) * (1 if str(unit) in {"byte", "bytes"} else element_size),
            )
        order = fields.get("minor_to_major") or fields.get("permutation") or fields.get("order")
        if order is not None:
            strides = _ordered_strides(concrete_shape, element_size, order)
            if strides is not None:
                return LayoutInfo(
                    concrete_shape,
                    dtype,
                    f"stablehlo:{encoding_text}",
                    "stablehlo_encoding",
                    encoding_text,
                    strides,
                    "bytes",
                    int(fields.get("offset", 0)),
                )
        # Bare tags such as ``#row_major`` are dialect-owned until their
        # definition is available.  Keep their logical geometry conservative.
        return LayoutInfo(
            concrete_shape,
            dtype,
            f"stablehlo:{encoding_text}",
            "opaque_encoding",
            encoding_text,
            None,
            None,
            0,
            reason="encoding does not expose a verifiable stride mapping",
        )

    named = str(layout or "dense").lower()
    if named in {"dense", "row_major", "c_contiguous"}:
        strides = _dense_strides(concrete_shape, element_size)
    elif named in {"column_major", "fortran", "f_contiguous"}:
        strides = _ordered_strides(
            concrete_shape, element_size, tuple(range(len(concrete_shape)))
        )
    else:
        strides = None
    if strides is None:
        return LayoutInfo(
            concrete_shape,
            dtype,
            str(layout),
            source,
            encoding_text,
            None,
            None,
            0,
            reason=f"layout '{layout}' has no registered stride mapping",
        )
    return LayoutInfo(concrete_shape, dtype, str(layout), source, encoding_text, strides, "bytes", 0)


def tensor_layout(tensor: Any) -> LayoutInfo:
    """Resolve layout metadata from a TensorSpec-like object."""

    return resolve_layout(
        tuple(tensor.shape),
        str(tensor.dtype),
        layout=str(getattr(tensor, "layout", "dense")),
        attributes=getattr(tensor, "attributes", {}),
    )


__all__ = ["LayoutInfo", "resolve_layout", "tensor_layout"]
