from __future__ import annotations
from typing import Protocol, runtime_checkable
from dataclasses import dataclass

import sympy as sp

from ..field import Field
from ..types import PsType
from ..sympyextensions import TypedSymbol, DynamicType


@dataclass(frozen=True)
class FieldBufferSpec:
    """Specification for a field's memory buffers."""

    dtype: PsType | DynamicType
    """Data type of buffer entries"""

    base_ptr_name: str
    """Name of the buffer's base pointer"""

    shape: tuple[TypedSymbol | int, ...]
    """Shape of the buffer's interior region"""

    strides: tuple[TypedSymbol | int, ...]
    """The buffer's linearization strides"""

    @property
    def rank(self) -> int:
        """The buffer's rank; the dimensionality of its index space."""
        return len(self.shape)


@dataclass(frozen=True)
class IterationLimits:
    """Kernel iteration limits prescribed by a field."""

    bounds: tuple[int | TypedSymbol, ...]
    """Iteration bounds for each dimension. Lower bound is implicitly ``0``."""

    loop_order: tuple[int, ...]
    """Ideal loop order for iterating this field, with coordinates ordered slowest-to-fastest."""

    @property
    def rank(self) -> int:
        """Rank (dimensionality) of the iteration space."""
        return len(self.bounds)

    @staticmethod
    def from_legacy_field(field: Field) -> IterationLimits:
        return IterationLimits(field.spatial_shape, field.layout)


@runtime_checkable
class IField(Protocol):
    """Interface for algebraic fields to the pystencils code generator.

    Algebraic fields in pystencils are functions

    .. math::
        f: I \\to \\mathcal{D}

    from an *index space* :math:`I \\subset \\mathbb{Z}^{k}` to some domain :math:`\\mathcal{D}`.
    The dimensionality :math:`k` is the field's *spatial rank* (or just *rank*).
    In code generation, fields are backed by :math:`m`-dimensional memory buffers (where :math:`m \\ge k`).
    The `IField` protocol defines the interface any algebraic field type must define
    to communicate its memory properties to the code generator.

    Each algebraic field type must provide the following information to the code generator:

    - its `name <IField.name>`;
    - its memory data type (`dtype <IField.dtype>`);
    - its *buffer specification* defining the required memory properties of the field's underlying buffers;
    - its *iteration limits* defining the valid index space for a kernel's spatial iteration.
    """

    @property
    def name(self) -> str:
        """Name of the field"""
        ...

    @property
    def dtype(self) -> PsType | DynamicType:
        """Data type of the field's elements"""
        ...

    def get_buffer_spec(self) -> FieldBufferSpec:
        """Return the buffer specification defining the field's memory properties"""
        ...

    def get_iteration_limits(self) -> IterationLimits:
        """Return the iteration limits for kernels operating on this field"""
        ...


@runtime_checkable
class IFieldAccess(Protocol):
    """Interface for expressions that should be translated to buffer accesses for an algebraic field.

    This protocol should be implemented by any expression types that constitute *field accesses*,
    to communicate the associated field and buffer indices to the code generator.
    """

    @property
    def field(self) -> IField:
        """Field associated with this field access"""
        ...

    def get_buffer_indices(self) -> tuple[sp.Basic, ...]:
        """Indices into the field's memory buffer for this field access"""
        ...
