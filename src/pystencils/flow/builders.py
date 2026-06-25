from __future__ import annotations
from typing import Callable, overload, Iterable, cast, Sequence

import sympy as sp

from .flowgraph import (
    Flowgraph,
    FlowgraphNode,
    Bottom,
    EquationsBlock,
    Cases,
    FlowgraphAssignment,
    Let,
    Export,
    Store,
    SymbolicMemoryLoc,
    Reduce,
    Subgraph,
)
from ..types import UserTypeSpec
from ..sympyextensions.typed_sympy import TypedSymbol
from ..sympyextensions.reduction import ReductionOp


class let_export_adaptor:
    def __init__(
        self,
        builder: EquationsBlockBuilder,
        assignment_class: type[FlowgraphAssignment],
    ):
        self._builder = builder
        self._asm_class = assignment_class

    @overload
    def __setitem__(
        self,
        lhs: str | sp.Symbol | slice,
        rhs: sp.Basic,
    ): ...

    @overload
    def __setitem__(
        self,
        lhs: tuple[str | sp.Symbol | slice, ...],
        rhs: Iterable[sp.Basic],
    ): ...

    def __setitem__(
        self,
        lhs: str | sp.Symbol | slice | tuple[str | sp.Symbol | slice, ...],
        rhs: sp.Basic | Iterable[sp.Basic],
    ):
        if isinstance(lhs, tuple):
            rhs = tuple(cast(Iterable[sp.Basic], rhs))
            if len(rhs) != len(rhs):
                raise ValueError(
                    "Wrong number of elements on multi-assignment right-hand side"
                )
            for s, e in zip(lhs, rhs, strict=True):
                self._builder._add_assignment(
                    self._asm_class(self._builder._parse_symbol(s), e)
                )
        else:
            self._builder._add_assignment(
                self._asm_class(self._builder._parse_symbol(lhs), sp.sympify(rhs))
            )


class store_adaptor:
    def __init__(self, builder: EquationsBlockBuilder):
        self._builder = builder

    @overload
    def __setitem__(
        self,
        mem_loc: SymbolicMemoryLoc,
        rhs: sp.Basic,
    ): ...

    @overload
    def __setitem__(
        self,
        mem_loc: tuple[SymbolicMemoryLoc, ...],
        rhs: Iterable[sp.Basic],
    ): ...

    def __setitem__(
        self,
        mem_loc: SymbolicMemoryLoc | tuple[SymbolicMemoryLoc, ...],
        rhs: sp.Basic | Iterable[sp.Basic],
    ):
        if isinstance(mem_loc, tuple):
            rhs = tuple(cast(Iterable[sp.Basic], rhs))

            if len(mem_loc) != len(rhs):
                raise ValueError(
                    "Wrong number of elements on assignment right-hand side"
                )

            for loc, expr in zip(mem_loc, rhs, strict=True):
                self._builder._add_assignment(Store(loc, expr))
        else:
            self._builder._add_assignment(Store(mem_loc, sp.sympify(rhs)))


class reduce_adaptor:
    def __init__(self, builder: EquationsBlockBuilder):
        self._builder = builder

    def __setitem__(
        self, reduction_spec: tuple[TypedSymbol | slice, str], rhs: sp.Basic
    ):
        symb_spec, op = reduction_spec
        symb = self._builder._parse_symbol(symb_spec)
        if not isinstance(symb, TypedSymbol):
            raise ValueError("Can only reduce onto typed symbols")
        self._builder._add_assignment(Reduce(symb, rhs, ReductionOp(op)))


class EquationsBlockBuilder(let_export_adaptor):
    """Builder for equation blocks. Used by the `block <pystencils.flow.block>` decorator."""

    def __init__(self, predecessors: set[FlowgraphNode]) -> None:
        self._assignments: list[FlowgraphAssignment] = []
        self._predecessors = predecessors

        super().__init__(self, Let)

        self._let = let_export_adaptor(self, Let)
        self._export = let_export_adaptor(self, Export)
        self._store = store_adaptor(self)
        self._reduce = reduce_adaptor(self)

    def _parse_symbol(self, symb_descr: str | sp.Symbol | slice):
        dtype: UserTypeSpec | None = None
        symb: str | sp.Symbol

        if isinstance(symb_descr, slice):
            symb, dtype = symb_descr.start, symb_descr.stop
        else:
            symb = symb_descr

        if isinstance(symb, str):
            if dtype is not None:
                symb = TypedSymbol(symb, dtype)
            else:
                symb = sp.Symbol(symb)

        return symb

    def _add_assignment(self, asm: FlowgraphAssignment):
        if any(a.lhs == asm.lhs for a in self._assignments):
            raise ValueError(f"Duplicate assignment left-hand side: {asm.lhs}")

        self._assignments.append(asm)

    def connect(self, node: FlowgraphNode):
        """Add another node as a predecessor to this block."""
        self._predecessors.add(node)

    @property
    def let(self):
        """Create a private subexpression.

        **Syntax:**

        .. code:: Python

            _eq.let[symbol] = value
            _eq.let["symbol_name": "symbol_type"] = value
        """
        return self._let

    @property
    def export(self):
        """Create an exported subexpression.

        **Syntax:**

        .. code:: Python

            _eq.store[symbol] = value
            _eq.store["symbol_name": "symbol_type"] = value
        """
        return self._export

    @property
    def store(self):
        """Store a value to a memory location.

        **Syntax:**

        .. code:: Python

            _eq.store[memory_location] = value

        Valid memory locations are modelled by the type `SymbolicMemoryLoc`.
        """
        return self._store

    @property
    def reduce(self):
        """Reduce a value onto a memory location using a given commutative operator.

        **Syntax:**

        .. code:: Python

            _eq.reduce[typed_symbol, "op"] = value
            _eq.reduce["symbol_name": "symbol_type", "op"] = value

        Valid reduction operators are ``+``, ``-``, ``*``, ``min`` and ``max``.
        The reduction target must be a `typed symbol <TypedSymbol>`.
        """
        return self._reduce


@overload
def block(
    *,
    preds: Iterable[FlowgraphNode] | None = None,
    name: str | None = None,
) -> Callable[[Callable[[EquationsBlockBuilder], None]], EquationsBlock]: ...


@overload
def block(func: Callable[[EquationsBlockBuilder], None], /) -> EquationsBlock: ...


def block(
    func: Callable[[EquationsBlockBuilder], None] | None = None,
    /,
    preds: Iterable[FlowgraphNode] | None = None,
    name: str | None = None,
):
    """Define a flowgraph block using ``let`` syntax.

    `block` is a decorator used to define flowgraph blocks
    using Python function syntax.

    Args:
        - preds: Predecessor nodes to this block
        - name: Name of this block. If none is given, the function's name will be used
    """
    predecessors: set[FlowgraphNode] = set() if preds is None else set(preds)

    def decorate(func: Callable[[EquationsBlockBuilder], None]) -> EquationsBlock:
        builder = EquationsBlockBuilder(predecessors)
        func(builder)
        return EquationsBlock(
            builder._assignments,
            builder._predecessors,
            name=func.__name__ if name is None else name,
        )

    if func is not None:
        return decorate(func)
    else:
        return decorate


class CasesBuilder:
    def __init__(self) -> None:
        self._branches: list[tuple[sp.Basic, Flowgraph]] = []

    @overload
    def case(
        self, cond: sp.Basic, *, preds: Iterable[FlowgraphNode] | None = None
    ) -> Callable[[Callable[[EquationsBlockBuilder], None]], None]: ...

    @overload
    def case(self, cond: sp.Basic, subgraph: Flowgraph) -> None: ...

    def case(
        self,
        cond: sp.Basic,
        subgraph: Flowgraph | None = None,
        preds: Iterable[FlowgraphNode] | None = None,
    ):
        if subgraph is not None:
            if not isinstance(subgraph, Flowgraph):
                raise ValueError(
                    "Subgraph passed to `case` was not a valid flowgraph."
                    " Hint: Tie nodes and their predecessors into a flowgraph using `tie()`."
                )
            self._branches.append((sp.sympify(cond), subgraph))
            return None

        def decorator(func: Callable[[EquationsBlockBuilder], None]):
            mdef = block(preds=preds)(func)
            self._branches.append((sp.sympify(cond), tie(mdef, name=func.__name__)))

        return decorator


@overload
def cases(
    *,
    preds: Iterable[FlowgraphNode] | None = None,
    name: str | None = None,
) -> Callable[[Callable[[CasesBuilder], None]], Cases]: ...


@overload
def cases(func: Callable[[CasesBuilder], None], /) -> Cases: ...


def cases(
    func: Callable[[CasesBuilder], None] | None = None,
    /,
    preds: Iterable[FlowgraphNode] | None = None,
    name: str | None = None,
):
    """Define a flowgraph cases block.

    `cases` is a decorator used to define flowgraph case blocks
    using Python function syntax.
    """
    predecessors: tuple[FlowgraphNode, ...] = () if preds is None else tuple(preds)

    def decorate(func: Callable[[CasesBuilder], None]) -> Cases:
        builder = CasesBuilder()
        func(builder)
        return Cases(
            builder._branches,
            predecessors,
            name=func.__name__ if name is None else name,
        )

    if func is not None:
        return decorate(func)
    else:
        return decorate


def guarded_block(
    cond: sp.Basic,
    *,
    preds: Iterable[FlowgraphNode] | None = None,
    name: str | None = None,
) -> Callable[[Callable[[EquationsBlockBuilder], None]], Cases]:
    """Define a guarded block that is protected by a condition."""
    outer_predecessors = set() if preds is None else set(preds)

    def decorator(func: Callable[[EquationsBlockBuilder], None]) -> Cases:
        gblock = block(func)

        if gblock.exports:
            raise ValueError(
                f"A guarded block cannnot export symbols. Exports encountered: {gblock.exports}"
            )

        # move all preds added via `connect()` to the outside
        all_predecessors = outer_predecessors | gblock.predecessors
        gblock = gblock.replace_predecessors([])

        return Cases(
            [(cond, tie(gblock, name=func.__name__))], all_predecessors, name=name
        )

    return decorator


def tie(*nodes: FlowgraphNode, name: str | None = None) -> Flowgraph:
    """Tie multiple nodes together into a flowgraph"""
    match nodes:
        case [Bottom()]:
            return Flowgraph(cast(Bottom, nodes[0]), name=name)
        case _:
            if any(isinstance(n, Bottom) for n in nodes):
                raise ValueError(
                    "Cannot tie a Bottom node into a subgraph with other nodes."
                )
            return Flowgraph(Bottom(nodes), name=name)


@overload
def subgraph(
    graph: Flowgraph, /, *, preds: Sequence[FlowgraphNode] = ()
) -> Subgraph: ...


@overload
def subgraph(
    *nodes: FlowgraphNode, preds: Sequence[FlowgraphNode] = (), name: str | None = None
) -> Subgraph: ...


def subgraph(
    *args: Flowgraph | FlowgraphNode,
    preds: Sequence[FlowgraphNode] = (),
    name: str | None = None,
) -> Subgraph:
    match args:
        case [Flowgraph()]:
            return Subgraph(cast(Flowgraph, args[0]), preds, name=args[0].name)
        case _:
            return Subgraph(
                tie(*cast(tuple[FlowgraphNode, ...], args), name=name), preds, name=name
            )
