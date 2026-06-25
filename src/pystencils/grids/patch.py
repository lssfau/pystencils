from __future__ import annotations

from typing import cast, SupportsIndex, Iterable, Any, overload
from enum import Enum, auto

import sympy as sp
from dataclasses import dataclass

from ..sympyextensions import tcast, TypedSymbol, DynamicType
from ..defaults import DEFAULTS


class VariablePlacement(Enum):
    """Variable placements on a patch's grid."""

    VERTICES = auto()
    """Signifies variables located on the grid vertices"""

    CELLS = auto()
    """Signifies variables located on the grid cells"""


@dataclass(frozen=True)
class PatchGrid:
    """Proxy representing a grid superimposed on a patch."""

    patch: Patch
    """Parent patch of the grid"""

    placement: VariablePlacement
    """Variable placement of the grid, i.e. vertices or cells"""


class Patch:
    r"""Cuboid volume of space, discretized by a cartesian grid.

    Patches are algebraic objects representing (patches of) simulation domains in pystencils.
    A patch :math:`\Omega` is an :math:`m`-dimensional multi-interval
    :math:`[\ell_1, u_1] \times \cdots \times [\ell_m, u_m]`
    given by its lower and upper corners :math:`(\ell_1, \dots, \ell_m)` and :math:`(u_1, \dots, u_m)`.

    Each patch is host to a numerical grid of :math:`N_1 \times \cdots \times N_m` *vertices* (points),
    with spacings :math:`h_k = \frac{u_k - \ell_k}{N_k - 1}` in each dimension.
    The grid vertices are

    .. math::

        \mathrm{vertices} (\Omega) = \left\{
            (i_k \cdot h_k)_{k = 1, \dots, m}
            \; \vert \;
            (i_k = 0, \dots, N_k - 1)_{k = 1, \dots, m}
        \right\}.

    The cuboid spaces enclosed by adjacent vertices are the grid's *cells*.
    Each cell encloses a volume :math:`h_1 \times \cdots \times h_m` according to the grid spacing;
    cell :math:`\mathsf{c}_{\boldsymbol{i}}` therefore claims the multi-interval

    .. math::

        \mathsf{c}_{\boldsymbol{i}} = \prod_{k=1}^m [i_k \cdot h_k, (i_k + 1) \cdot h_k].

    There are one fewer cells than vertices in each dimension; the grid's cells are therefore

    .. math::

        \mathrm{cells} (\Omega) = \left\{
            \mathsf{c}_{(i_1, \dots, i_m)}
            \; \vert \;
            (i_k = 0, \dots, N_k - 2)_{k = 1, \dots, m}
        \right\}.

    The `Patch` class models these patches algebraically, and allows for the definition of *fields*
    on its vertex and cell grids. All its attributes are symbolic expressions.

    Args:
        name: Name of the patch
        x_min: Lower corner :math:`(\ell_1, \dots, \ell_m)` of the patch.
               Can be omitted; in this case, the lower corner is set to the origin.
        x_max: Upper corner :math:`(u_1, \dots, u_k)` of the patch.
        num_vertices: Number of vertices in each dimension. If given, ``num_cells`` is inferred; do not specify both.
        num_cells: Number of cells in each dimension. If given, ``num_vertices`` is inferred; do not specify both.
    """

    @overload
    def __init__(
        self,
        name: str,
        x_max: tuple[Any, ...],
        /,
        *,
        num_vertices: Iterable[sp.Basic] | None = None,
        num_cells: Iterable[sp.Basic] | None = None,
    ): ...

    @overload
    def __init__(
        self,
        name: str,
        x_min: tuple[Any, ...],
        x_max: tuple[Any, ...],
        /,
        *,
        num_vertices: Iterable[sp.Basic] | None = None,
        num_cells: Iterable[sp.Basic] | None = None,
    ): ...

    def __init__(
        self,
        name: str,
        *corners: tuple[Any, ...],
        num_vertices: Iterable[sp.Basic] | None = None,
        num_cells: Iterable[sp.Basic] | None = None,
    ):
        match corners:
            case [upper]:
                x_max = upper
                dim = len(x_max)
                x_min = (0,) * dim
            case [lower, upper]:
                x_min = lower
                x_max = upper
                if len(x_min) != len(x_max):
                    raise ValueError("x_min and x_max must have the same length")
                dim = len(x_max)
            case _:
                raise ValueError("Invalid number of positional arguments.")

        self._name = name
        self._dim = dim
        self._x_min: tuple[sp.Basic, ...] = tuple(sp.sympify(x) for x in x_min)
        self._x_max: tuple[sp.Basic, ...] = tuple(sp.sympify(x) for x in x_max)
        self._extents: tuple[sp.Basic, ...] = tuple(
            x1 - x0 for x0, x1 in zip(self._x_min, self._x_max)
        )

        if num_vertices is not None and num_cells is not None:
            raise ValueError("At most one of num_vertices and num_cells may be set.")

        if num_vertices is not None:
            num_vertices = tuple(num_vertices)
            num_cells = tuple(nx - 1 for nx in num_vertices)
        elif num_cells is not None:
            num_cells = tuple(num_cells)
            num_vertices = tuple(nx + 1 for nx in num_cells)
        else:
            num_vertices = tuple(
                TypedSymbol(f"N_{i}", DynamicType.INDEX_TYPE) for i in range(dim)
            )
            num_cells = tuple(nx - 1 for nx in num_vertices)

        self._num_vertices: tuple[sp.Basic, ...] = tuple(num_vertices)
        self._num_cells: tuple[sp.Basic, ...] = tuple(num_cells)

        self._spacing: tuple[sp.Basic, ...] = tuple(
            xs / tcast.auto(n - 1) for xs, n in zip(self._extents, self._num_vertices)
        )

    @property
    def name(self) -> str:
        """Name of this patch"""
        return self._name

    @property
    def dimensionality(self) -> int:
        """Dimensionality of this patch"""
        return self._dim

    @property
    def x_min(self) -> sp.ImmutableMatrix:
        """Lower corner as a SymPy matrix"""
        return sp.ImmutableMatrix(self._x_min)

    @property
    def x_max(self) -> sp.ImmutableMatrix:
        """Upper corner as a SymPy matrix"""
        return sp.ImmutableMatrix(self._x_max)

    @property
    def extents(self) -> sp.ImmutableMatrix:
        """Total extents of the patch as a SymPy matrix"""
        return sp.ImmutableMatrix(self._extents)

    @property
    def spacing(self) -> sp.ImmutableMatrix:
        """Vertex spacing as a SymPy matrix"""
        return sp.ImmutableMatrix(self._spacing)

    @property
    def num_vertices(self) -> sp.ImmutableMatrix:
        """Number of vertices as a SymPy matrix"""
        return sp.ImmutableMatrix(self._num_vertices)

    @property
    def num_cells(self) -> sp.ImmutableMatrix:
        """Number of cells as a SymPy matrix"""
        return sp.ImmutableMatrix(self._num_cells)

    @property
    def cells(self) -> PatchGrid:
        """Proxy indicating the patch's cell grid; use to associate algebraic fields with the patch's cells."""
        return PatchGrid(self, VariablePlacement.CELLS)

    @property
    def vertices(self) -> PatchGrid:
        """Proxy indicating the patch's vertex grid; use to associate algebraic fields with the patch's vertices."""
        return PatchGrid(self, VariablePlacement.VERTICES)

    def vertex(self, *idcs: SupportsIndex | sp.Basic) -> sp.ImmutableMatrix:
        """Coordinates of a vertex identified by the given relative indices.

        Like for field accesses, indices are interpreted relative to the current iteration point.

        **Examples**

        Coordinates of the current vertex:

        .. code::

            patch.vertex()

        Coordinates of the north-east neighbor vertex on a 2D patch:

        .. code::

            patch.vertex(-1, 1)

        """
        return sp.ImmutableMatrix(
            [
                low + h * i
                for low, h, i in zip(
                    self._x_min, self._spacing, self._resolve_indices(idcs)
                )
            ]
        )

    def cell_center(self, *idcs: SupportsIndex | sp.Basic) -> sp.ImmutableMatrix:
        """Coordinates of the center of a cell identified by the given relative indices.

        Like for field accesses, indices are interpreted relative to the current iteration point.

        **Examples**

        Coordinates of the current cell's center:

        .. code::

            patch.cell_center()

        Coordinates of the center of the north-east neighbor cell on a 2D patch:

        .. code::

            patch.cell_center(-1, 1)

        """
        return sp.ImmutableMatrix(
            [
                low + (i + sp.Rational(1, 2)) * h
                for low, h, i in zip(
                    self._x_min, self._spacing, self._resolve_indices(idcs)
                )
            ]
        )

    def _resolve_indices(
        self, idcs: tuple[SupportsIndex | sp.Basic, ...]
    ) -> tuple[sp.Basic, ...]:
        if not idcs:
            idcs = (sp.Number(0),) * self._dim

        if len(idcs) != self._dim:
            raise ValueError(
                f"Invalid size of index tuple: {len(idcs)}. Expected {self._dim}"
            )

        counters = DEFAULTS.spatial_counters[: self._dim]

        return tuple(tcast.auto(i + c) for i, c in zip(idcs, counters))

    def atoms(self, *types) -> set[sp.Basic]:
        return (
            self.x_min.atoms(*types)
            | self.x_max.atoms(*types)
            | self.num_vertices.atoms(*types)
            | self.num_cells.atoms(*types)
        )

    def _args(self) -> tuple:
        return (
            self._name,
            self._x_min,
            self._x_max,
            self._num_vertices,
            self._num_cells,
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False

        return self._args() == cast(Patch, other)._args()

    def __hash__(self) -> int:
        return hash((type(self),) + self._args())

    def _repr_latex_(self) -> str:
        extents = " \\times ".join(
            rf"[{sp.latex(low)}, {sp.latex(high)}]"
            for low, high in zip(self.x_min, self.x_max)
        )
        num_vertices = ", ".join(sp.latex(nv) for nv in self.num_vertices)
        return rf"${self._name} \left( {extents}, n_v = ({num_vertices}) \right)$"
