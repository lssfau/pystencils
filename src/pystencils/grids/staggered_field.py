from __future__ import annotations

from enum import Enum
from typing import Any, cast, TypeVar, Generic

import sympy as sp
from .protocols import FieldBufferSpec, IterationLimits
from ..types import UserTypeSpec, create_numeric_type, PsNumericType
from ..defaults import DEFAULTS
from ..sympyextensions.typed_sympy import DynamicType, TypedSymbol
from ..sympyextensions.atom_proxy import AtomProxy

from sympy.printing import StrPrinter
from sympy.printing.latex import LatexPrinter

from ..stencil import direction_string_to_offset


class Staggering(Enum):
    """Lightweight description of a staggered stencil by its positive edges.

    Each member's value is the tuple of *positive* staggered entries -- the cell
    centre ``(0, ..., 0)`` followed by the stored edges. The negative counterparts
    (and hence the full ``DdQq`` stencil) are derived on demand.
    """

    D2Q5 = ((0, 0), (0, 1), (1, 0))
    D2Q9 = ((0, 0), (0, 1), (1, 0), (1, 1), (1, -1))
    D3Q7 = ((0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1))
    D3Q15 = ((0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1),
             (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1))
    D3Q19 = ((0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1),
             (1, 1, 0), (1, -1, 0), (0, 1, 1), (1, 0, 1), (0, 1, -1), (1, 0, -1))
    D3Q27 = ((0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1),
             (1, 1, 0), (1, -1, 0), (0, 1, 1), (1, 0, 1), (0, 1, -1), (1, 0, -1),
             (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1))

    def __init__(self, *positive_entries: tuple[int, ...]):
        self._entries: tuple[tuple[int, ...], ...] = positive_entries

    @property
    def entries(self) -> tuple[tuple[int, ...], ...]:
        """The positive (stored) staggered entries, starting with the centre."""
        return self._entries

    @property
    def staggered_entries(self) -> tuple[tuple[int, ...], ...]:
        """Alias of `entries`; the stored edges of this staggering."""
        return self._entries

    @property
    def D(self) -> int:
        """Spatial dimensionality."""
        return len(self._entries[0])

    @property
    def Q(self) -> int:
        """Number of full-stencil directions."""
        return 2 * len(self._entries) - 1

    def staggered_index(self, direction) -> int:
        """Index of a stored edge; a direction's inverse maps to the same index."""
        d = tuple(direction)
        if d in self._entries:
            return self._entries.index(d)
        return self._entries.index(tuple(-c for c in d))

    def staggered_direction(self, index: int) -> tuple[int, ...]:
        """The stored edge with the given index."""
        return self._entries[index]

    @property
    def stencil_entries(self) -> tuple[tuple[int, ...], ...]:
        """Full ``DdQq`` stencil: the centre, then each stored edge and its inverse."""
        full: list[tuple[int, ...]] = [self._entries[0]]
        for e in self._entries[1:]:
            full.append(e)
            full.append(tuple(-c for c in e))
        return tuple(full)

    def __getitem__(self, index: int) -> tuple[int, ...]:
        """The full-stencil direction with the given index."""
        return self.stencil_entries[index]


class StaggeredFieldAccess(sp.Expr):
    """A direct access to the value stored on one edge of a staggered grid cell.

    An access is fully described by the grid, the spatial offset of the cell, the
    *stored* edge index (one of the positive ``staggered_entries``), and -- for
    vector grids -- a component index. There is deliberately no sign or
    "reflected" mode: reading the value associated with a non-stored direction is
    expressed as an ordinary ``-1 * access`` via :attr:`StaggeredGrid.face`.
    """

    def __new__(
        cls,
        staggered_grid: StaggeredField | AtomProxy[StaggeredField],
        spatial_offsets: tuple | sp.Tuple,
        edge_index: int | sp.Expr,
        vector_index: int | None = None,
    ) -> StaggeredFieldAccess:
        if not isinstance(staggered_grid, AtomProxy):
            staggered_grid = AtomProxy(staggered_grid)
        if not isinstance(spatial_offsets, sp.Tuple):
            spatial_offsets = sp.sympify(spatial_offsets)
        if not isinstance(edge_index, sp.Basic):
            edge_index = sp.sympify(edge_index)
        if vector_index is None:
            obj: StaggeredFieldAccess = super().__new__(
                cls, staggered_grid, spatial_offsets, edge_index
            )
        else:
            obj = super().__new__(
                cls, staggered_grid, spatial_offsets, edge_index, sp.Integer(vector_index)
            )
        return obj

    def _staggered_grid_proxy(self) -> AtomProxy[StaggeredField]:
        return cast(AtomProxy[StaggeredField], self._args[0])

    def _hashable_content(self) -> tuple:
        return (self.offsets, self.index, self.staggered_grid,) + super()._hashable_content()

    #   Properties

    @property
    def staggered_grid(self) -> StaggeredField:
        return self._staggered_grid_proxy().get()

    @property
    def offsets(self) -> sp.Tuple:
        return cast(sp.Tuple, self._args[1])

    @property
    def index(self) -> sp.Expr:
        """Index of the stored edge this access refers to."""
        return cast(sp.Expr, self._args[2])

    @property
    def vector_index(self) -> sp.Integer | None:
        """Component index for vector staggered grids; None for scalar grids."""
        return cast(sp.Integer, self._args[3]) if len(self._args) > 3 else None

    @property
    def direction(self):
        # `index` refers to a stored edge, so the physical direction is the
        # corresponding staggered-stencil direction.
        return self.staggered_grid.stencil.staggered_direction(int(self.index))

    def get_buffer_spec(self) -> FieldBufferSpec:
        return self.staggered_grid.get_buffer_spec()

    #   Vector component access
    def __call__(self, k: int) -> StaggeredFieldAccess:
        """Select the ``k``-th vector component of this access.

        Only valid for vector staggered grids (created with a non-empty
        ``index_shape``)::

            u[1, -1, 0, "e"](1)   # component 1 of the east-edge value
        """
        if not self.staggered_grid.index_shape:
            raise IndexError(
                "Component access is only valid for vector staggered grids "
                "(those created with a non-empty index_shape)."
            )
        if self.vector_index is not None:
            raise IndexError("A vector component has already been selected.")
        args = list(self._args[:3]) + [sp.Integer(int(k))]
        return type(self)(*args)

    def _sympystr(self, printer: StrPrinter) -> str:
        offsets = ", ".join((printer._print(ox) for ox in self.offsets))
        index = printer._print(self.index)
        base = f"{self.staggered_grid.name}[{offsets}]({index})"
        if self.vector_index is not None:
            return f"{base}({printer._print(self.vector_index)})"
        return base

    def _latex(self, printer: LatexPrinter):
        offsets = printer._print(self.offsets)
        index = printer._print(self.index)
        lattice_name = self.staggered_grid._latexname()
        base = rf"{lattice_name}_{{{offsets}}}^{{{index}}}"
        if self.vector_index is not None:
            return rf"{base}_{{{printer._print(self.vector_index)}}}"
        return base

    @property
    def field(self) -> StaggeredField:
        return self.staggered_grid

    def get_buffer_indices(self):
        spatial = tuple(
            ctr + ox for ctr, ox in zip(DEFAULTS.spatial_counters, self.offsets)
        )
        if self.vector_index is not None:
            return spatial + (self.index, self.vector_index)
        return spatial + (self.index,)


View_T = TypeVar("View_T", bound=StaggeredFieldAccess)


class StaggeredField:

    class _AccessProxy(Generic[View_T]):
        """Shared subscript parsing and direction resolution for staggered accesses."""

        def __init__(self, staggered_grid: StaggeredField, view_class: type[View_T]):
            self._staggered_grid = staggered_grid
            self._view_class = view_class

        def _parse_key(self, key) -> tuple[tuple, Any]:
            """Split a subscript into (spatial offsets, selector)."""
            D = self._staggered_grid.stencil.D
            if not isinstance(key, tuple):
                key = (key,)
            key = tuple(key)

            #   A trailing string is always a direction selector.
            if isinstance(key[-1], str):
                offsets = key[:-1]
                if len(offsets) == 0:
                    offsets = (0,) * D
                elif len(offsets) != D:
                    raise IndexError(
                        f"Expected {D} spatial offsets before direction "
                        f"{key[-1]!r}, got {len(offsets)}."
                    )
                return offsets, key[-1]

            #   All-integer keys.
            if len(key) == D + 1:
                return key[:D], key[D]
            elif len(key) == D:
                #   Offsets only -> default to the central entry.
                return key, 0
            elif len(key) == 1:
                #   A single selector (direction or index) at the home cell.
                return (0,) * D, key[0]
            else:
                raise IndexError(
                    f"Invalid staggered access key {key!r} for a {D}D grid: expected a "
                    f"direction, {D} offsets, or {D} offsets followed by a direction."
                )

        def _direction_offset(self, selector) -> tuple:
            """Resolve a selector to a full-stencil direction offset tuple."""
            stencil = self._staggered_grid.stencil
            if isinstance(selector, str):
                d = tuple(direction_string_to_offset(selector.upper()))
                if stencil.D == 2:
                    d = d[:2]
                return d
            elif isinstance(selector, (tuple, sp.Tuple)):
                return tuple(int(x) for x in selector)
            else:
                return stencil[int(selector)]

    class _EdgeProxy(_AccessProxy[View_T]):
        """``u[...]``: direct access to the value stored on a grid edge.

        Only the edges that physically exist (the positive ``staggered_entries``)
        may be addressed::

            u["e"]              # east-edge value at the home cell
            u[1, -1, 0, "e"]    # ...at neighbour (1, -1, 0)
            u["e"](1)           # ...component 1 (vector grids)
            u[i]                # the i-th stored edge

        To read the value associated with a non-stored direction (e.g. ``"w"``),
        use :attr:`StaggeredGrid.face`.
        """

        def __getitem__(self, key) -> View_T:
            offsets, selector = self._parse_key(key)
            sg = self._staggered_grid
            stencil = sg.stencil

            if isinstance(selector, (str, tuple, sp.Tuple)):
                ci = self._direction_offset(selector)
                if ci not in stencil.staggered_entries:
                    raise ValueError(
                        f"Direction {selector!r} ({ci}) is not a stored edge of grid "
                        f"{sg.name!r}; stored edges are {stencil.staggered_entries}. "
                        f"Use {sg.name}.face[...] to read a non-stored direction."
                    )
                idx = stencil.staggered_index(ci)
            else:
                idx = int(selector)
                if not 0 <= idx <= stencil.Q // 2:
                    raise IndexError(
                        f"Stored-edge index {idx} out of range for grid {sg.name!r} "
                        f"(expected 0..{stencil.Q // 2})."
                    )
            return self._view_class(sg, tuple(offsets), idx)

    class _FaceProxy(_AccessProxy[View_T]):
        """``u.face[...]``: the value on a cell face, addressed by a full-stencil direction.

        For a stored face this is just the stored value; for the opposite
        direction it is the negated value on the corresponding face of the
        neighbouring cell (the same face, seen with reversed orientation)::

            u.face["e"]   # ->  u[0, 0, 0, "e"]
            u.face["w"]   # -> -u[-1, 0, 0, "e"]
        """

        def __getitem__(self, key):
            offsets, selector = self._parse_key(key)
            sg = self._staggered_grid
            stencil = sg.stencil
            offsets = tuple(offsets)

            ci = self._direction_offset(selector)
            idx = stencil.staggered_index(ci)
            if ci in stencil.staggered_entries:
                return self._view_class(sg, offsets, idx)
            #   Reflected direction: negate the stored value on the neighbour's edge.
            shifted = tuple(o + c for o, c in zip(offsets, ci))
            return sp.Mul(sp.Integer(-1), self._view_class(sg, shifted, idx))

    def __init__(
        self,
        name: str,
        stencil: Staggering,
        dtype: UserTypeSpec | DynamicType = DynamicType.NUMERIC_TYPE,
        index_shape: tuple[int, ...] = (),
    ):
        if not isinstance(dtype, DynamicType):
            dtype = create_numeric_type(dtype)

        self._name: str = name
        self._dtype: PsNumericType | DynamicType = dtype
        self._stencil = stencil
        self._index_shape: tuple[int, ...] = index_shape

        spatial_rank = self._stencil.D
        staggered_entries = self._stencil.Q // 2 + 1
        total_rank = spatial_rank + 1 + len(index_shape)

        self._buffer_spec = FieldBufferSpec(
            self._dtype,
            DEFAULTS.field_pointer_name(self._name),
            tuple(
                TypedSymbol(f"_size_{self._name}_{c}", DynamicType.INDEX_TYPE)
                for c in range(spatial_rank)
            )
            + (staggered_entries,)
            + index_shape,
            tuple(
                TypedSymbol(f"_stride_{self._name}_{c}", DynamicType.INDEX_TYPE)
                for c in range(total_rank)
            ),
        )

    #   Properties

    @property
    def name(self) -> str:
        return self._name

    @property
    def stencil(self) -> Staggering:
        return self._stencil

    @property
    def dtype(self) -> PsNumericType | DynamicType:
        return self._dtype

    @property
    def index_shape(self) -> tuple[int, ...]:
        return self._index_shape

    @property
    def grid(self):
        """Patch grid defining this field's index space.

        A standalone staggered grid is not embedded in a patch grid, so this is
        always ``None``. The property exists to satisfy the `IField` protocol.
        """
        return None

    def get_buffer_spec(self) -> FieldBufferSpec:
        return self._buffer_spec

    def get_iteration_limits(self) -> IterationLimits:
        spatial_rank = self._stencil.D
        return IterationLimits(
            self._buffer_spec.shape[:spatial_rank],
            tuple(range(spatial_rank)[::-1]),
        )

    #   Accesses

    def __getitem__(self, key) -> StaggeredFieldAccess:
        """Access the value stored on a grid edge; see `_EdgeProxy`."""
        return StaggeredField._EdgeProxy(self, StaggeredFieldAccess)[key]

    @property
    def face(self) -> _FaceProxy[StaggeredFieldAccess]:
        """Face accessor for full-stencil directions; see `_FaceProxy`."""
        return StaggeredField._FaceProxy(self, StaggeredFieldAccess)

    #   Infrastructure

    def _hashable_content(self) -> tuple:
        return (self._name, self._dtype, self._stencil, self._index_shape)

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False

        return self._hashable_content() == cast(StaggeredField, other)._hashable_content()

    def __hash__(self) -> int:
        return hash((type(self),) + self._hashable_content())

    #   Printing

    def __str__(self) -> str:
        return f"{self._name}[{self._stencil.name}] : {self._dtype}"

    def __repr__(self) -> str:
        args = f"{repr(self._name)}, {repr(self._stencil)}, {repr(self._dtype)}"
        if self._index_shape:
            args += f", index_shape={repr(self._index_shape)}"
        return f"{type(self).__name__}({args})"

    def _latexname(self) -> str:
        #   Underscores mess up latex printing, so replace them
        return r"\operatorname{" + self._name.replace("_", "-") + "}"

    def _repr_latex_(self) -> str:
        latex = (
            rf"{self._latexname()} : \mathrm{{{self._stencil.name}}}"
            rf" \left[\texttt{{{self._dtype}}}\right]"
        )
        return "$" + latex + "$"
