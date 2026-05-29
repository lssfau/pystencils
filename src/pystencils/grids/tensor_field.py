from __future__ import annotations

from typing import SupportsIndex, cast
from enum import Enum, auto

import sympy as sp

from sympy.printing import StrPrinter
from sympy.printing.latex import LatexPrinter

from .protocols import (
    IField,
    IFieldAccess,
    FieldBufferSpec,
    IterationLimits,
)

from ..types import UserTypeSpec, create_numeric_type, PsNumericType
from ..sympyextensions import TypedSymbol, DynamicType
from ..sympyextensions.atom_proxy import AtomProxy
from ..defaults import DEFAULTS


class MemoryLayout(Enum):
    """Common field memory layouts available in pystencils."""

    LEFTMOST = auto()
    """Leftmost-first (Fortran-style) layout

    The leftmost buffer coordinate has stride one, and strides increase monotonically left-to-right.
    """

    RIGHTMOST = auto()
    """Rightmost-first (C/NumPy-style) layout

    The rightmost buffer coordinate has stride one, and strides increase monotonically right-to-left.
    """

    ZYXF = auto()
    """Naturally ordered Array-of-Structures layout.

    Spatial dimensions are ordered Z-Y-X (x fastest, z the slowest coordinate).
    Values for each spatial index are stored in packed fashion;
    for multidimensional per-point data, memory order inside the pack is determined by the field type.
    """

    FZYX = LEFTMOST
    """Naturally ordered Structure-of-Arrays layout.

    Spatial dimensions are ordered Z-Y-X (x fastest, z the slowest coordinate).
    Values for each spatial index are linearized separately, such that the same structure
    entry on adjacent nodes are also adjacent in memory (i.e. ``x``-stride is always ``1``.)
    """

    FORTRAN = LEFTMOST
    """See `MemoryLayout.LEFTMOST`"""

    SOA = LEFTMOST
    """See `MemoryLayout.FZYX`"""

    C = RIGHTMOST
    """See `MemoryLayout.RIGHTMOST`"""

    NUMPY = RIGHTMOST
    """See `MemoryLayout.LEFTMOST`"""

    AOS = ZYXF
    """See `MemoryLayout.ZYXF`"""

    def linearization_order(
        self, rank: int, spatial_rank: int | None = None
    ) -> tuple[int, ...]:
        """Order in which buffer dimensions are linearized in memory. Ordered fastest-to-slowest."""
        match self:
            case MemoryLayout.LEFTMOST:
                return tuple(range(rank))
            case MemoryLayout.RIGHTMOST:
                return tuple(range(rank)[::-1])
            case MemoryLayout.ZYXF:
                if spatial_rank is None:
                    raise ValueError(
                        "Cannot infer linearization order for layout ZYXF without spatial rank"
                    )
                return tuple(range(spatial_rank, rank)[::-1]) + tuple(
                    range(spatial_rank)
                )
            case _:
                assert False, "unreachable code"


class TensorField(IField):
    """Tensor field mapping each point of a :math:`d`-dimensional index space to a rank :math:`n` tensor.

    A tensor field is a function

    .. math::
        f: I \\to T^{k_1 \\times \\cdots \\times k_n}

    from an index space :math:`I \\subset \\mathbb{Z}^{d}`
    to a tensor space over :math:`T`.
    :math:`T` may be :math:`\\mathbb{R}` or a specific numeric `data type <pystencils.types>`.

    .. note::
        Degenerate tensor shapes (i.e. :math:`n_i = 1` for any :math:`i`)
        are not supported.

    **Accessing Values** Field entries can be accessed using the ``[]`` and ``()`` operators.
    Spatial offsets must be given in ``[]``; they are interpreted *relative* to the current node.
    Tensor indices are passed to ``()``.

    *Examples:*

    - Access vector entry 1 at the current node:

    .. code-block:: Python

        f(1)

    - Access scalar entry at the eastern neighbor node:

    .. code-block:: Python

        f[1, 0]()

    - Access tensor entry ``(0, 0)`` at the north-west neighbor node:

    .. code-block:: Python

        f[-1, 1](0, 0)

    Args:
        name: Name of the tensor field
        spatial_rank: Dimensionality of the index space :math:`I`
        tensor_shape: Shape of the field's tensors
        dtype: Data type of the field's tensor entries
        layout: Memory layout of the field's memory buffers at runtime
    """

    def __init__(
        self,
        name: str,
        spatial_rank: int,
        tensor_shape: tuple[int, ...] = (),
        *,
        dtype: UserTypeSpec | DynamicType = DynamicType.NUMERIC_TYPE,
        layout: str | MemoryLayout = MemoryLayout.NUMPY,
    ) -> None:
        if spatial_rank < 1:
            raise ValueError(f"Invalid spatial rank: {spatial_rank}")

        if any(ts == 1 for ts in tensor_shape):
            raise ValueError(
                f"Invalid degenerate tensor shape: {tensor_shape}. "
                "Trivial tensor dimensions (with extent = 1) are not permitted."
            )

        self._name = name
        self._dtype = (
            dtype if isinstance(dtype, DynamicType) else create_numeric_type(dtype)
        )
        self._spatial_rank = spatial_rank
        self._tensor_shape = tensor_shape
        self._layout = (
            layout if isinstance(layout, MemoryLayout) else MemoryLayout[layout.upper()]
        )

        self._buffer_rank = spatial_rank + len(tensor_shape)
        self._buffer_spec = self._create_buffer_spec()
        self._iteration_limits = self._create_iteration_limits()

    @property
    def name(self) -> str:
        """The field's name"""
        return self._name

    @property
    def dtype(self) -> PsNumericType | DynamicType:
        """Data type of tensor entries"""
        return self._dtype

    @property
    def layout(self) -> MemoryLayout:
        """Memory layout of runtime buffers"""
        return self._layout

    @property
    def spatial_rank(self) -> int:
        """Dimensionality of the spatial index space"""
        return self._spatial_rank

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        """Shape of the field's tensors"""
        return self._tensor_shape

    @property
    def tensor_rank(self) -> int:
        """Rank of the field's tensors"""
        return len(self._tensor_shape)

    #   IField

    def get_buffer_spec(self) -> FieldBufferSpec:
        return self._buffer_spec

    def get_iteration_limits(self) -> IterationLimits:
        return self._iteration_limits

    #   Access

    def __call__(self, *indices: SupportsIndex | sp.Basic) -> TensorFieldAccess:
        """Access an entry of the tensor at the current iteration index."""
        return TensorFieldAccess(self, (0,) * self._spatial_rank, tuple(indices))

    class _GetitemHelper:
        def __init__(
            self,
            field: TensorField,
            spatial_offsets: tuple[SupportsIndex | sp.Basic, ...],
        ):
            self._field = field
            self._spatial_offsets = spatial_offsets

        def __call__(self, *indices: SupportsIndex | sp.Basic) -> TensorFieldAccess:
            return TensorFieldAccess(self._field, self._spatial_offsets, tuple(indices))

    def __getitem__(
        self, key: SupportsIndex | sp.Basic | tuple[SupportsIndex | sp.Basic, ...]
    ) -> TensorField._GetitemHelper:
        """Access tensor entries at an offset relative to the current iteration index."""
        if not isinstance(key, tuple):
            key = (key,)
        return self._GetitemHelper(self, key)

    #   Printing

    def __str__(self) -> str:
        return f"{self._name}{self.tensor_shape}: {self._dtype}[{self._spatial_rank}D]"

    def __repr__(self) -> str:
        return (
            f"TensorField({repr(self._name)}, {repr(self._spatial_rank)}, {repr(self._tensor_shape)}, "
            f"dtype={repr(self._dtype)}, layout={repr(self._layout)})"
        )

    def _latexname(self) -> str:
        #   Underscores mess up latex printing, so replace them
        return r"\operatorname{" + self._name.replace("_", "-") + "}"

    def _repr_latex_(self) -> str:
        typename = (
            self.dtype._latexname()
            if isinstance(self.dtype, DynamicType)
            else rf"\mathrm{{{self.dtype}}}"
        )

        if self.tensor_shape:
            ishape = r"\times".join(str(s) for s in self.tensor_shape)
            image = rf"{typename}^{{{ishape}}}"
        else:
            image = typename

        return rf"${self._latexname()} : \mathbb{{Z}}^{self._spatial_rank} \to {image}$"

    #   PRIVATE

    def _args(self) -> tuple:
        return (
            self._name,
            self._dtype,
            self._spatial_rank,
            self._tensor_shape,
            self._layout,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorField):
            return False

        return self._args() == other._args()

    def __hash__(self) -> int:
        return hash((type(self),) + self._args())

    def _create_buffer_spec(self) -> FieldBufferSpec:
        ptr_name = DEFAULTS.field_pointer_name(self._name)

        shape: tuple[TypedSymbol | int, ...] = (
            tuple(
                TypedSymbol(
                    DEFAULTS.field_shape_name(self._name, coord), DynamicType.INDEX_TYPE
                )
                for coord in range(self._spatial_rank)
            )
            + self._tensor_shape
        )

        strides: tuple[TypedSymbol | int, ...] = tuple(
            (
                TypedSymbol(
                    DEFAULTS.field_stride_name(self._name, coord),
                    DynamicType.INDEX_TYPE,
                )
            )
            for coord in range(self._buffer_rank)
        )

        return FieldBufferSpec(self._dtype, ptr_name, shape, strides)

    def _create_iteration_limits(self) -> IterationLimits:
        lin_order = self._layout.linearization_order(
            self._spatial_rank, self._spatial_rank
        )

        return IterationLimits(
            self._buffer_spec.shape[: self._spatial_rank], lin_order[::-1]
        )


class TensorFieldAccess(sp.Expr, IFieldAccess):
    """Access into a tensor field.

    Expression representing an access to a single tensor entry of a tensor field.

    Create instances of this class using the ``[]`` and ``()`` operators of `TensorField`.
    """

    def __new__(
        cls,
        tfield: TensorField | AtomProxy[TensorField],
        spatial_offsets: tuple[SupportsIndex | sp.Basic, ...] | sp.Tuple,
        tensor_indices: tuple[SupportsIndex | sp.Basic, ...] | sp.Tuple,
    ) -> TensorFieldAccess:
        if not isinstance(tfield, AtomProxy):
            tfield = AtomProxy(tfield)
        if not isinstance(spatial_offsets, sp.Tuple):
            spatial_offsets = sp.sympify(spatial_offsets)
        if not isinstance(tensor_indices, sp.Tuple):
            tensor_indices = sp.sympify(tensor_indices)

        if len(spatial_offsets) != tfield.get().spatial_rank:
            raise IndexError(
                f"Invalid number of spatial offsets: {len(spatial_offsets)}. Expected {tfield.get().spatial_rank}"
            )

        if len(tensor_indices) != tfield.get().tensor_rank:
            raise IndexError(
                f"Invalid number of tensor indices: {len(tensor_indices)}. Expected {tfield.get().tensor_rank}"
            )

        for c, (i, imax) in enumerate(zip(tensor_indices, tfield.get().tensor_shape)):
            try:
                if int(i) < 0 or int(i) >= imax:  # type: ignore
                    raise IndexError(f"Tensor index in entry {c} out of range: {i}")
            except TypeError:
                pass

        obj: TensorFieldAccess = super().__new__(
            cls, tfield, spatial_offsets, tensor_indices
        )

        return obj

    #   Args

    @property
    def _tfield_proxy(self) -> AtomProxy[TensorField]:
        return cast(AtomProxy[TensorField], self._args[0])

    @property
    def _spatial_offsets(self) -> sp.Tuple:
        return cast(sp.Tuple, self._args[1])

    @property
    def _tensor_indices(self) -> sp.Tuple:
        return cast(sp.Tuple, self._args[2])

    #   Properties

    @property
    def field(self) -> TensorField:
        """Field accessed by this expression"""
        return self._tfield_proxy.get()

    @property
    def offsets(self) -> sp.Tuple:
        """Offsets relative to the current iteration point of this access"""
        return self._spatial_offsets

    @property
    def indices(self) -> sp.Tuple:
        """Tensor indices of this access"""
        return self._tensor_indices

    #   IFieldAccess

    def get_buffer_indices(self) -> tuple[sp.Basic, ...]:
        spatial_indices: tuple[sp.Basic, ...] = tuple(
            ctr + ox  # type: ignore
            for ctr, ox in zip(DEFAULTS.spatial_counters, self._spatial_offsets)
        )
        tensor_indices: tuple[sp.Basic, ...] = tuple(self._tensor_indices)
        return spatial_indices + tensor_indices

    #   Printing

    def _sympystr(self, printer: StrPrinter) -> str:
        offsets = ", ".join((printer._print(ox) for ox in self.offsets))
        index = ", ".join((printer._print(i) for i in self.indices))
        return f"{self.field.name}[{offsets}]({index})"

    def _latex(self, printer: LatexPrinter):
        offsets = printer._print(self.offsets)
        index = printer._print(self.indices)
        lattice_name = self.field._latexname()
        return rf"{lattice_name}_{{{offsets}}}^{{{index}}}"
