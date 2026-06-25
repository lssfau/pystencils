from __future__ import annotations

from typing import Mapping, Any, Iterable, TYPE_CHECKING, overload
from types import ModuleType

import sympy as sp
import numpy as np

from ..codegen import Target
from ..types import UserTypeSpec, create_type, PsType, PsIntegerType

from .patch import Patch, VariablePlacement
from .protocols import IField, CreateNdArray
from ..sympyextensions import DynamicType, TypedSymbol, ReductionOp

if TYPE_CHECKING:
    from .pyvista import PatchDataPyVistaBridge


class PatchData:
    """Manage simulation data and field arrays for a given patch.

    `PatchData` is the runtime incarnation for pystencils `patches <Patch>`;
    instances of `PatchData` manage simulation data associated with a single `Patch`.
    A `Patch` serves as a blueprint for `PatchData` objects.
    When creating a `PatchData`, numerical values must be provided for all symbols occuring
    in the patch's definition; the symbolic expressions for the patch corners, numbers of vertices and cells,
    are thus evaluated to concrete numbers.

    Multiple `PatchData` instances can be created for the same algebraic `Patch`; each can
    have different values for the patch's defining symbols.

    `Operators <pystencils.flow.operator.Operator>` can be invoked directly on `PatchData` objects
    to run them with the symbol values and data arrays stored by a given `PatchData`.

    **Data Management**

    `PatchData` allocates and manages values and data arrays for SymPy symbols and fields associated with a patch.

    To set values for symbols, pass them in a dictionary to the `PatchData` constructor.
    You can set single symbols to single values, and also associate tuples of values with tuples of symbols.
    Value *must* be provided this way for all symbols occuring in the given patch.

    *Example:*

    .. code::

        Nv = ps.symbols("N_:3", ps.index_t)  # symbols for number of vertices
        P = Patch("P", (1, 1, 1), num_vertices=Nv)

        c = sp.Symbol("c")  # a single symbol

        Pdata = PatchData(P, {Nv: (32, 16, 8), c: 0.01})

    `PatchData` uses NumPy to convert and store symbol values with the correct data type, depending on their
    associated symbol (using `default_dtype <PatchData.default_dtype>` for untyped symbols,
    and the symbols' type for typed symbols).

    To allocate and manage data arrays for algebraic fields, pass them to the ``fields`` keyword argument:

    .. code::

        f = TensorField("f", P.cells, ())
        g = TensorField("g", P.cells, (3,))

        Pdata = PatchData(P, {...}, fields=[f, g])

    `PatchData` will then create ``ndarray`` instances for these fields.
    The array module (NumPy, CuPy, DPND) is inferred from on the ``target`` parameter;
    arrays are created through the `CreateNdArray` protocol (see `IField`).

    **Accessing Data**

    Symbol values and arrays can be accessed using the ``[]`` operator, e.g. ``Pdata[f]`` for the
    ``ndarray`` backing the field ``f``.
    Arrays and values can also be set through ``[]``.

    Args:
        patch: The blueprint patch for this data container
        vars: Dictionary mapping symbols to their values
        fields: Fields associated with the patch for which data arrays should be allocated
        target: Hardware target with which this patch data is used.
                Should match the `target <CreateKernelConfig.target>` of any operators run on this patch.
        default_dtype: The default numeric data type for symbol values and data arrays; used for untyped symbols
                and whenever `DynamicType.NUMERIC_TYPE` is encountered
        index_dtype: The index data type used for arrays;
                substituted whenever `DynamicType.INDEX_TYPE` is encountered.
    """

    def __init__(
        self,
        patch: Patch,
        vars: Mapping[sp.Symbol | tuple[sp.Symbol, ...], Any] | None = None,
        *,
        fields: Iterable[IField] = (),
        target: Target = Target.GenericCPU,
        default_dtype: UserTypeSpec = np.float64,
        index_dtype: UserTypeSpec = np.int64,
    ):
        #   Process variables dict
        if vars is None:
            vars = dict()
        else:
            vars = dict(vars)

        var_tuples = [(k, v) for k, v in vars.items() if isinstance(k, tuple)]
        for k, v in var_tuples:
            del vars[k]
            for ki, vi in zip(k, v):
                vars[ki] = vi

        required_symbols = patch.atoms(sp.Symbol)
        for symb in required_symbols:
            if symb not in vars:
                raise KeyError(
                    f"No value was provided for symbol {symb} occuring in patch definition"
                )

        #   Set target and array module

        if target.is_gpu():
            import cupy as cp

            self._xp = cp
            self._xp_ndarray = cp.ndarray
        elif target == Target.SYCL:
            import dpctl.tensor as dp

            self._xp = dp
            self._xp_ndarray = dp.usm_ndarray
        else:
            self._xp = np
            self._xp_ndarray = np.ndarray

        self._target = target

        #   Default data type

        default_dtype = create_type(default_dtype)
        if default_dtype.numpy_dtype is None:
            raise ValueError("Don't know any NumPy type for given default dtype")
        self._default_dtype: np.dtype = default_dtype.numpy_dtype

        #   Index data type

        index_dtype = create_type(index_dtype)

        if not isinstance(index_dtype, PsIntegerType):
            raise ValueError(
                f"Invalid index type: {index_dtype}. Must be an integer type."
            )

        if index_dtype.numpy_dtype is None:
            raise ValueError("Don't know any NumPy type for given index dtype")

        self._index_dtype: np.dtype[np.integer] = index_dtype.numpy_dtype

        self._data: dict[sp.Symbol | IField, Any] = dict()
        for k, v in vars.items():
            if not isinstance(k, sp.Symbol):
                raise ValueError(f"Invalid data key {k}. Must be a SymPy symbol.")
            self.set_data(k, v)

        #   Set patch geometry

        self._patch = patch

        self._x_min = np.array(
            [self._default_dtype.type(xm) for xm in self._patch.x_min.subs(self._data)]
        )
        self._x_max = np.array(
            [self._default_dtype.type(xm) for xm in self._patch.x_max.subs(self._data)]
        )
        self._extents = np.array(
            [
                self._default_dtype.type(ex)
                for ex in self._patch.extents.subs(self._data)
            ]
        )
        self._spacing = np.array(
            [
                self._default_dtype.type(sx)
                for sx in self._patch.spacing.subs(self._data)
            ]
        )
        self._num_vertices = tuple(
            int(nx) for nx in self._patch.num_vertices.subs(self._data)
        )
        self._num_cells = tuple(
            int(nx) for nx in self._patch.num_cells.subs(self._data)
        )

        #   Add field arrays

        for f in fields:
            self.add_field(f)

    @property
    def patch(self) -> Patch:
        """Blueprint patch of this data container"""
        return self._patch

    @property
    def dimensionality(self) -> int:
        """Dimensionality of this container's patch"""
        return self._patch.dimensionality

    @property
    def x_min(self) -> np.typing.NDArray[np.float64]:
        """Lower corner of this patch incarnation"""
        return self._x_min.copy()

    @property
    def x_max(self) -> np.typing.NDArray[np.float64]:
        """Upper corner of this patch incarnation"""
        return self._x_max.copy()

    @property
    def extents(self) -> np.typing.NDArray[np.float64]:
        """Extents of this patch incarnation"""
        return self._extents.copy()

    @property
    def spacing(self) -> np.typing.NDArray[np.float64]:
        """Grid spacing of this patch incarnation"""
        return self._spacing.copy()

    @property
    def num_vertices(self) -> tuple[int, ...]:
        """Number of vertices in this patch incarnation"""
        return self._num_vertices

    @property
    def num_cells(self) -> tuple[int, ...]:
        """Number of cells in this patch incarnation"""
        return self._num_cells

    @property
    def target(self) -> Target:
        """Primary hardware target for this data container"""
        return self._target

    @property
    def array_module(self) -> ModuleType:
        """Array module used for field arrays"""
        return self._xp

    @property
    def default_dtype(self) -> np.dtype:
        """Default data type for field arrays and symbol values.

        This data type will be used for values of untyped symbols,
        in arrays for fields typed as `DynamicType.NUMERIC_TYPE`,
        as well as the patch's geometry attributes.
        """
        return self._default_dtype

    def add_field(self, f: IField, *, copy_of: IField | None = None):
        """Add a field to this data container, and allocate an ``ndarray`` using the default ``array_module``.

        Args:
            f: Field modelling `CreateNdArray`
            copy_of: If given, initialize the array as a copy of another field's data array.
                     The two must have the same shape, and be created on the same array module.
        """
        if not isinstance(f, CreateNdArray):
            raise ValueError(
                f"Cannot create data array for field {f}: Field doesn't implement CreateNdArray protocol"
            )

        if f in self._data:
            raise ValueError(f"Duplicate field: {f}")

        if f.grid is None or f.grid.patch != self._patch:
            raise ValueError(
                f"Given field {f} was not associated with given patch {self._patch}"
            )

        nptype: np.dtype = self._resolve_type(f.dtype)

        match f.grid.placement:
            case VariablePlacement.VERTICES:
                spatial_shape = self._num_vertices
            case VariablePlacement.CELLS:
                spatial_shape = self._num_cells

        self._data[f] = f.create_ndarray(self._xp, spatial_shape, dtype=nptype)

        if copy_of is not None:
            self._data[f][:] = self._data[copy_of]

    @overload
    def add_reduction(self, symb: TypedSymbol, op: str | ReductionOp, /):
        """Add an output buffer for a reduction symbol.

        Args:
            symb: The reduction symbol
            op: The reduction operation; if given, the symbol's value is initialized
                with the neutral element of the operation
        """

    @overload
    def add_reduction(
        self,
        symb: TypedSymbol,
        initial_value: int | float | np.generic,
        /,  # noqa: W504
    ):
        """Add an output buffer for a reduction symbol.

        Args:
            symb: The reduction symbol
            initial_value: Initial value for the symbol; must be given if `op` is not provided
        """

    def add_reduction(
        self,
        symb: TypedSymbol,
        op_or_value: str | ReductionOp | int | float | np.generic,
    ):
        if symb in self._data:
            raise ValueError(f"Duplicate data key: {symb}")

        nptype: np.dtype = self._resolve_type(symb.dtype)

        if isinstance(op_or_value, str):
            op_or_value = ReductionOp(op_or_value)

        if not isinstance(op_or_value, ReductionOp):
            value = op_or_value
        else:
            value = nptype.type(op_or_value.neutral_element)

        self._data[symb] = self._xp.array([value], dtype=nptype)

    def set_data(self, key: IField | sp.Symbol, value: Any):
        """Store a value for a symbol, converting it to the correct data type.

        If ``symb`` is an untyped symbol, the ``value`` will be converted to
        the `default_dtype <PatchData.default_dtype>`.
        """
        match key:
            case TypedSymbol():
                dtype = self._resolve_type(key.dtype)
                self._data[key] = dtype.type(value)
            case sp.Symbol():
                self._data[key] = self._default_dtype.type(value)
            case _:
                self._data[key] = value

    def swap(self, k0, k1):
        """Swap the data objects of two keys."""
        self._data[k0], self._data[k1] = self._data[k1], self._data[k0]

    def __getitem__(self, key: IField | sp.Symbol | tuple[sp.Symbol, ...]) -> Any:
        if isinstance(key, tuple):
            return tuple(self._data[x] for x in key)
        return self._data[key]

    def __setitem__(self, key: IField | sp.Symbol, val: Any):
        self.set_data(key, val)

    def asnumpy(self, key: Any) -> np.ndarray:
        """Return a copy of the data array for the given key as a NumPy array.

        If ``key`` is backed by a NumPy, CuPy or DPNP ``ndarray``, copies the data
        into an new NumPy array and returns that copy.

        Raises:
            KeyError
                If key is either not a valid data key, or does not store an ``ndarray``.
        """
        val = self._data[key]
        if isinstance(val, np.ndarray):
            return val.copy()
        elif isinstance(val, self._xp_ndarray):
            return self._xp.asnumpy(val)
        else:
            raise KeyError(f"{key} did not identify an ndarray")

    @property
    def data(self) -> dict[IField | sp.Symbol, Any]:
        return self._data

    @property
    def args(self) -> dict[str, Any]:
        return {k.name: v for k, v in self._data.items()}

    def _resolve_type(self, dtype: PsType | DynamicType) -> np.dtype:
        match dtype:
            case DynamicType.NUMERIC_TYPE:
                return self._default_dtype
            case DynamicType.INDEX_TYPE:
                return self._index_dtype
            case pstype if pstype.numpy_dtype is not None:
                return pstype.numpy_dtype
            case _:
                raise ValueError(
                    f"Can't determine runtime type for pystencils type {dtype}"
                )

    #   PyVista Bridge

    def viz(self) -> PatchDataPyVistaBridge:
        try:
            import pyvista  # noqa: F401
        except ImportError:  # pragma: no cover
            raise RuntimeError("Cannot create visualizer: PyVista is not installed")

        from .pyvista import PatchDataPyVistaBridge

        return PatchDataPyVistaBridge(self)
