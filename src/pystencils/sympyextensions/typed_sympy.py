from __future__ import annotations
from typing import cast

import sympy as sp
from enum import Enum, auto

from ..types import PsType, PsNumericType, create_type, UserTypeSpec

from sympy.logic.boolalg import Boolean

from warnings import warn

from .atom_proxy import AtomProxy


def is_loop_counter_symbol(symbol):
    from ..defaults import DEFAULTS

    try:
        return DEFAULTS.spatial_counters.index(symbol)
    except ValueError:
        return None


class DynamicType(Enum):
    """Dynamic data type that will be resolved during kernel creation"""

    NUMERIC_TYPE = auto()
    """Use the default numeric type set for the kernel"""

    INDEX_TYPE = auto()
    """Use the default index type set for the kernel.

    This is guaranteed to be an interger type.
    """

    def __str__(self) -> str:
        match self:
            case DynamicType.NUMERIC_TYPE:
                return "ps::numeric_t"
            case DynamicType.INDEX_TYPE:
                return "ps::index_t"

    def _latexname(self) -> str:
        match self:
            case DynamicType.NUMERIC_TYPE:
                return r"\mathbb{R}"
            case DynamicType.INDEX_TYPE:
                return r"\mathbb{Z}"


numeric_t = DynamicType.NUMERIC_TYPE
index_t = DynamicType.INDEX_TYPE


class TypeAtom(AtomProxy[PsType | DynamicType]):
    """Wrapper around a type to disguise it as a SymPy atom."""


def assumptions_from_dtype(dtype: PsType | DynamicType):
    """Derives SymPy assumptions from :class:`PsAbstractType`

    Args:
        dtype (PsAbstractType): a pystencils data type
    Returns:
        A dict of SymPy assumptions
    """
    assumptions = dict()

    match dtype:
        case DynamicType.INDEX_TYPE:
            assumptions.update({"integer": True, "real": True})
        case DynamicType.NUMERIC_TYPE:
            assumptions.update({"real": True})
        case PsNumericType():
            if dtype.is_int():
                assumptions.update({"integer": True})
            if dtype.is_uint():
                assumptions.update({"negative": False})
            if dtype.is_int() or dtype.is_float():
                assumptions.update({"real": True})

    return assumptions


class TypedSymbol(sp.Symbol):
    """A symbol annotated with a data type.

    Args:
        name (`str`): Name of the symbol
        dtype (`UserTypeSpec` | `DynamicType`): The symbol's data type
    """

    _dtype: PsType | DynamicType

    def __new__(cls, *args, **kwds):
        obj = TypedSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, name: str, dtype: UserTypeSpec | DynamicType, **kwargs):
        if not isinstance(dtype, DynamicType):
            dtype = create_type(dtype)

        assumptions = assumptions_from_dtype(dtype)
        assumptions.update(kwargs)

        obj = super(TypedSymbol, cls).__xnew__(cls, name, **assumptions)
        obj._dtype = dtype

        return obj

    __xnew__ = staticmethod(__new_stage2__)  # type: ignore
    __xnew_cached_ = staticmethod(sp.core.cacheit(__new_stage2__))

    @property
    def dtype(self) -> PsType | DynamicType:
        #   mypy: ignore
        return self._dtype

    def _hashable_content(self):
        #   mypy: ignore
        return super()._hashable_content(), hash(self._dtype)

    def __getnewargs__(self):
        return self.name, self.dtype

    def __getnewargs_ex__(self):
        return (self.name, self.dtype), self.assumptions0

    @property
    def canonical(self):
        return self

    @property
    def reversed(self):
        return self

    @property
    def headers(self) -> set[str]:
        return self.dtype.required_headers if isinstance(self.dtype, PsType) else set()


def typed_symbols(names: str, dtype: UserTypeSpec | DynamicType, **args):
    """Wrapper around `sp.symbols <sympy.core.symbol.symbols>` that creates
    `TypedSymbol` instances with the given data type"""
    return sp.symbols(names, cls=TypedSymbol, dtype=dtype, **args)


def symbols(names: str, dtype: UserTypeSpec | DynamicType | None = None, **args):
    """Wrapper around `sp.symbols <sympy.core.symbol.symbols>` that creates
    `TypedSymbol` instances if a data type is given"""
    if dtype is not None:
        return typed_symbols(names, dtype, **args)
    else:
        return sp.symbols(names, **args)


class TypeCast(sp.Function):
    """Explicitly cast an expression to a data type."""

    @staticmethod
    def as_numeric(expr):
        return TypeCast(expr, DynamicType.NUMERIC_TYPE)

    @staticmethod
    def as_index(expr):
        return TypeCast(expr, DynamicType.INDEX_TYPE)

    @staticmethod
    def auto(expr):
        return AutoCast(expr)

    @property
    def expr(self) -> sp.Basic:
        return self.args[0]

    @property
    def dtype(self) -> PsType | DynamicType:
        return cast(TypeAtom, self._args[1]).get()

    def __new__(cls, expr: sp.Basic, dtype: UserTypeSpec | DynamicType | TypeAtom):
        tatom: TypeAtom
        match dtype:
            case TypeAtom():
                tatom = dtype
            case DynamicType():
                tatom = TypeAtom(dtype)
            case _:
                tatom = TypeAtom(create_type(dtype))

        return super().__new__(cls, expr, tatom)

    @classmethod
    def eval(cls, expr: sp.Basic, tatom: TypeAtom) -> TypeCast | None:
        dtype = tatom.get()
        if cls is not BoolCast and isinstance(dtype, PsNumericType) and dtype.is_bool():
            return BoolCast(expr, tatom)

        return None

    def _eval_is_integer(self):
        if self.dtype == DynamicType.INDEX_TYPE:
            return True
        if isinstance(self.dtype, PsNumericType) and self.dtype.is_int():
            return True

    def _eval_is_real(self):
        if isinstance(self.dtype, DynamicType):
            return True
        if isinstance(self.dtype, PsNumericType) and (
            self.dtype.is_float() or self.dtype.is_int()
        ):
            return True

    def _eval_is_nonnegative(self):
        if isinstance(self.dtype, PsNumericType) and self.dtype.is_uint():
            return True

    def _sympystr(self, printer):
        expr = printer._print(self.expr)
        dtype = str(self.dtype)
        return f"cast< {dtype} >({expr})"


class BoolCast(TypeCast, Boolean):
    pass


class AutoCast(sp.Function):
    """Automatic cast; convert expression type to match the surrounding context."""

    @classmethod
    def eval(cls, expr: sp.Basic) -> sp.Basic | None:
        if isinstance(expr, sp.Number):
            return expr

        return None

    def _sympystr(self, printer):
        expr = printer._print(self.args[0])
        return f"autocast({expr})"


tcast = TypeCast


class CastFunc(TypeCast):
    def __new__(cls, *args, **kwargs):
        warn(
            "CastFunc is deprecated and will be removed in pystencils 2.1. "
            "Use `pystencils.tcast` instead.",
            FutureWarning,
        )
        return TypeCast.__new__(cls, *args, **kwargs)
