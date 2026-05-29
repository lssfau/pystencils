from typing import Generic, TypeVar, Hashable
from functools import total_ordering

import sympy as sp

TWrapped = TypeVar("TWrapped", bound=Hashable)


@total_ordering
class _CompareByStr(Generic[TWrapped]):
    def __init__(self, obj: TWrapped):
        self._obj = obj

    def __eq__(self, value: object) -> bool:
        return isinstance(value, _CompareByStr) and value._obj == self._obj

    def __hash__(self) -> int:
        return hash(self._obj)

    def __lt__(self, other):
        if not isinstance(other, _CompareByStr):
            return NotImplemented

        return str(self._obj) < str(other._obj)


class AtomProxy(Generic[TWrapped], sp.Atom):
    """Disguise an arbitrary hashable type as a SymPy atom."""

    _wrapped: TWrapped

    def __new__(cls, wrapped: TWrapped):
        obj = super().__new__(cls)
        obj._wrapped = wrapped
        return obj

    def _sympystr(self, *args, **kwargs):
        return str(self._wrapped)

    def get(self) -> TWrapped:
        return self._wrapped

    def _hashable_content(self):
        #   _CompareByStr necessary to make sp.Basic.compare work
        return (_CompareByStr(self._wrapped),)

    def __getnewargs__(self):  # type: ignore
        return (self._wrapped,)
