from __future__ import annotations
from typing import Protocol, runtime_checkable, TypeVar
from types import ModuleType

from dataclasses import dataclass

import sympy as sp
import numpy as np

from .patch import PatchGrid

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
        :label: eq:algebraic-field-def

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

    **Universal Buffer Model and Indexing Rules**

    At runtime, pystencils fields are backed by contiguous, linearized memory buffers.
    Between the algebraic front-end, the code generator, and any runtime system,
    the indexing semantics of these buffers are governed by the following universal buffer model
    (see also :numref:`Fig. %s <fig:IFieldBufferModel>`).

    An m-dimensional buffer of element type ``T`` is a contiguous memory region
    of size

    .. code::

        B = allocShape[0] * ... * allocShape[m-1] * sizeof(T)

    bytes.

    The buffer's memory is linearized according to the linearization strides
    ``(strides[0], ..., strides[m-  1])``.

    The buffer is split into an *inner region* and a *padding region*.
    The inner region has the format ``shape[0], ..., shape[m-  1]``,
    and is embedded at an offset ``(offset[0], ..., offset[m-  1])``.

    The buffer's *base pointer* identifies the first entry of the inner region.
    The memory address of buffer element ``(i[0], ..., i[m-1])`` is calculated as in
    :numref:`Lst. %s <listing:index-linearization>`:

    .. code-block::
        :caption: Linearization of buffer indices
        :name: listing:index-linearization

        addr = base_ptr + strides[0] * i[0] + strides[1] * i[1] + ... + strides[m-1] * i[m-1]

    This has a few consequences:

    - The first entry of the inner region always has buffer index ``(0, ..., 0)``
    - The inner region is covered exactly by the indices
      ``(0 : shape[0] - 1, 0 : shape[1] - 1, ..., 0 : shape[m-  1] - 1)``
    - The padding region is accessed by negative indices, and indices with ``i[k] >= shape[k]``

    .. figure:: /_static/img/IFieldBufferModel.svg
        :name: fig:IFieldBufferModel

        Universal buffer model for pystencils fields.
        Properties modelled by `FieldBufferSpec` are shown in bold;
        hidden properties are printed thin.

    The universal buffer model is realized by the `IField` protocol via ``get_buffer_spec``.
    The `FieldBufferSpec` instance returned by `get_buffer_spec`
    defines the buffer's data type ``T``; (the name of) its base pointer;
    as well as its inner region shape and its strides as constants or symbols.
    These properties are printed bold in :numref:`Fig. %s <fig:IFieldBufferModel>`.
    Base pointer and strides are used by the backend for index calculations
    (:numref:`Lst. %s <listing:index-linearization>`),
    while the shape may be used to define the iteration space.

    **Field Access to Buffer Indices**

    The rules for how algebraic field accesses are translated into buffer indices,
    and how the buffer is split into inner and padding regions for a given field,
    depend on the field class.
    `TensorField`, for instance, allows the presence of *ghost layers* which it maps into the padding region.

    **NumPy Arrays as Buffers**

    NumPy and its sister libraries (CuPy, DPNP) are the prime runtime environments for pystencils,
    and their ``ndarray`` classes the foremost implementation of field buffers.
    However, ``ndarray`` does not implement the universal buffer protocol, as it does not differentiate
    between the inner and padding regions.

    When creating an ``ndarray`` for a given field, that array must have shape
    ``(allocShape[0], ..., allocShape[m-1])``;
    but when passing it to a kernel, its base pointer must be extracted from the correct offset.
    These two aspects of array handling are realized by fields implementing the following protocols:

    - `CreateNdArray.create_ndarray`: Gets the underlying array module and the required spatial shape;
      must create and return an ``ndarray`` of the given module to hold the data for the field.
      The spatial shape is the shape of the *inner region* of the field's index space :math:`I`
      (see :eq:`eq:algebraic-field-def`).
      For instance, ``TensorField.create_ndarray(xp, spatial_shape)`` will create an array of
      shape ``(spatial_shape[0] + 2 * ghost_layers, ...) + tensor_shape`` to accomodate its ghost layers,
      while also setting up the array's strides to match its memory layout.
      ``create_ndarray`` is primarily used by the `PatchData` class to allocate its arrays.
    - `ViewNdArray.view_ndarray`: Gets an ``ndarray`` instance backing the field,
      and must return a *view* into the inner region of that array.
      For instance, ``TensorField.view_ndarray(arr)`` will return ``arr[ghost_layers:-ghost_layers, ..., 0, ..., 0]``,
      cutting of its ghost layers in the spatial buffer dimensions.
    """

    @property
    def name(self) -> str:
        """Name of the field"""
        ...

    @property
    def dtype(self) -> PsType | DynamicType:
        """Data type of the field's elements"""
        ...

    @property
    def grid(self) -> PatchGrid | None:
        """The patch grid that defines this field's index space"""

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


TArray = TypeVar("TArray")


@runtime_checkable
class CreateNdArray(Protocol):
    """Create an `ndarray` instance for a field. Mix in with `IField`."""

    def create_ndarray(
        self,
        array_module: ModuleType,
        spatial_shape: tuple[int, ...],
        *,
        dtype: np.dtype | None = None,
        **kwargs
    ):
        """Create an ``array_module.ndarray`` backing this field, with the given ``inner_shape``.

        If this field is defined on a `PatchGrid`, `spatial_shape` must reflect
        the number of nodes on that grid (i.e. number of cells, number of vertices, etc...)

        Args:
            array_module: Reference to the array module (NumPy, CuPy, DPNP)
            spatial_shape: Shape of the field's spatial index space
            dtype: Data type of the field entries; if ``None``, infer from the field type
            kwargs: Keyword arguments forwarded to the array module's array creation
                    routine (ususally ``.zeros()``).
        """


@runtime_checkable
class ViewNdArray(Protocol):
    """Extract the interior view onto an `ndarray` for a field. Mix in with `IField`."""

    def view_ndarray(self, arr: TArray) -> TArray:
        """Return a *view* into the inner region of the given ``ndarray`` backing this field"""
        ...
