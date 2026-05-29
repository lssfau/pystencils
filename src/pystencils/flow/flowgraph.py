from __future__ import annotations
from typing import cast, TypeVar, Generic, Sequence, Generator, Iterable, Mapping
from abc import ABC, abstractmethod
from functools import reduce
from itertools import chain
from functools import cache

import sympy as sp

from ..field import Field
from ..grids.protocols import IField, IFieldAccess
from ..sympyextensions.typed_sympy import TypedSymbol
from ..sympyextensions.pointers import mem_acc
from ..sympyextensions.reduction import ReductionOp

LHS_T = TypeVar("LHS_T", bound=sp.Basic)
SubstitutionMapping = Mapping[sp.Basic, sp.Basic]


class Flowgraph:
    """A closed flowgraph in canonical form.

    Upon creation, flowgraphs will be canonicalized and checked for consistency.

    Args:
        bottom: The `Bottom` node identifying the graph.
    """

    def __init__(self, bottom: Bottom, *, name: str | None = None) -> None:
        from .canonicalize_flowgraph import CanonicalizeFlowgraph

        canon = CanonicalizeFlowgraph()
        canon_result = canon(bottom)

        self._bottom = canon_result.bottom
        self._free_symbols = canon_result.free_symbols
        self._exports = canon_result.exports
        self._effects = canon_result.effects

        self._fields_read = canon_result.fields_read
        self._fields_written = canon_result.fields_written

        if name is not None:
            self._name = name
        else:
            h = abs(hash(self._bottom))
            self._name = f"flowgraph_{hex(h)[2:8]}"

    @property
    def bottom(self) -> Bottom:
        """Bottom node of this flowgraph"""
        return self._bottom

    @property
    def name(self) -> str:
        return self._name

    @property
    def free_symbols(self) -> frozenset[sp.Symbol]:
        """Parameters to this flowgraph"""
        return self._free_symbols

    @property
    def exports(self) -> frozenset[sp.Symbol]:
        """Symbols exported from this flowgraph"""
        return self._exports

    @property
    def effects(self) -> frozenset[Effect]:
        """Side effects of this flowgraph"""
        return self._effects

    @property
    def fields(self) -> frozenset[Field | IField]:
        return self._fields_read | self._fields_written

    @property
    def fields_read(self) -> frozenset[Field | IField]:
        return self._fields_read

    @property
    def fields_written(self) -> frozenset[Field | IField]:
        return self._fields_written

    def atoms(self, *types) -> set[sp.Basic]:
        return set().union(*(n.atoms(*types) for n in self.walk()))

    def walk(self) -> Generator[FlowgraphNode, None, None]:
        return self._bottom.walk()

    def subs(self, substitutions: SubstitutionMapping, **kwargs) -> Flowgraph:
        """Create a new flowgraph from this one by performing the given substitutions.

        This method acts as the SymPy ``subs`` method, with the difference that symbol substitutions
        of the kind ``symbol -> expr`` are only applied if ``symbol`` is a free symbol of the flowgraph.

        See the `subs <sympy.core.basic.Basic.subs>` method of ``sp.Basic`` for details.
        """
        substitutions = dict(substitutions)
        symbs = set(
            k
            for k in substitutions.keys()
            if isinstance(k, sp.Symbol) and not isinstance(k, Field.Access)
        )
        for symb in symbs:
            if symb not in self.free_symbols:
                del substitutions[symb]

        @cache
        def _recursive_subs(node: FlowgraphNode) -> FlowgraphNode:
            return node.subs(substitutions, **kwargs).replace_predecessors(
                [_recursive_subs(p) for p in node.predecessors]
            )

        bot = cast(Bottom, _recursive_subs(self._bottom))
        return Flowgraph(bot, name=self._name)

    def list_topological(self) -> tuple[FlowgraphNode, ...]:
        edges: list[tuple[int, int]] = []
        nodes: list[FlowgraphNode] = list(self.walk())
        node_indices: dict[FlowgraphNode, int] = {
            node: i for i, node in enumerate(nodes)
        }

        for node_index, node in enumerate(nodes):
            for pred in node.predecessors:
                edges.append((node_indices[pred], node_index))

        permutation = sp.topological_sort((range(len(nodes)), edges))
        return tuple(nodes[i] for i in permutation)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Flowgraph):
            return False

        return self._bottom == other._bottom and self._name == other._name

    def __hash__(self) -> int:
        return hash((type(self), self._bottom, self._name))

    def __str__(self) -> str:
        from .printing import PlainTextPrinter

        printer = PlainTextPrinter()
        return printer.print(self)

    def _repr_markdown_(self) -> str:
        return f"```\n{str(self)}\n```"


class FlowgraphAssignment(ABC, Generic[LHS_T]):
    """Base class for assignments."""

    __match_args__ = ("lhs", "rhs")

    def __init__(self, lhs: LHS_T, rhs: sp.Basic):
        self._check_lhs(lhs)
        self._lhs = lhs
        self._rhs = sp.sympify(rhs)

    @classmethod
    @abstractmethod
    def _check_lhs(cls, lhs: sp.Basic): ...

    @property
    def lhs(self) -> LHS_T:
        return self._lhs

    @property
    def rhs(self) -> sp.Basic:
        return self._rhs

    def atoms(self, *types) -> set[sp.Basic]:
        return self._lhs.atoms(*types) | self._rhs.atoms(*types)

    def subs(self, substitutions: SubstitutionMapping, **kwargs):
        """Perform substitutions inside this assignment.

        See the `subs <sympy.core.basic.Basic.subs>` method of ``sp.Basic`` for details.
        """
        return type(self)(
            cast(LHS_T, self._lhs.subs(substitutions, **kwargs)),
            self._rhs.subs(substitutions, **kwargs),
        )

    def _hashable_contents(self) -> tuple:
        return (self._lhs, self._rhs)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        return (
            self._hashable_contents()
            == cast(FlowgraphAssignment, other)._hashable_contents()
        )

    def __hash__(self) -> int:
        return hash((type(self),) + self._hashable_contents())

    def __str__(self) -> str:
        from .printing import PlainTextPrinter

        return PlainTextPrinter().print(self)

    def __repr__(self) -> str:
        asm_type: str = type(self).__name__
        args = ", ".join(repr(arg) for arg in self._hashable_contents())
        return f"{asm_type}({args})"

    def _repr_markdown_(self) -> str:
        return f"```\n{str(self)}\n```"


class Let(FlowgraphAssignment[sp.Symbol | TypedSymbol]):
    """`Let` assignments define subexpressions to be used within the same `EquationsBlock`."""

    @classmethod
    def _check_lhs(cls, lhs: sp.Basic):
        if isinstance(lhs, Field.Access) or not isinstance(
            lhs, sp.Symbol | TypedSymbol
        ):
            raise TypeError(
                f"Invalid type of left-hand side to `let`-assignment: {type(lhs)}"
            )


class Export(FlowgraphAssignment[sp.Symbol | TypedSymbol]):
    """`Export` assignments define subexpressions that can be
    used within the same `EquationsBlock` and its direct successors in the flowgraph."""

    @classmethod
    def _check_lhs(cls, lhs: sp.Basic):
        if isinstance(lhs, Field.Access) or not isinstance(
            lhs, sp.Symbol | TypedSymbol
        ):
            raise TypeError(
                f"Invalid type of left-hand side to `export`-assignment: {type(lhs)}"
            )


SymbolicMemoryLoc = Field.Access | mem_acc | IFieldAccess
"""Expression types that are valid memory locations and can be used on the LHS of `Store`."""


class Store(FlowgraphAssignment[SymbolicMemoryLoc]):  # type: ignore
    """`Store` assignments represent writing a value to an abstract memory location.

    Stores act as exports to the bottom node (:math:`\\bot`),
    and may therefore only occur in instances of `EquationsBlock` that have :math:`\\bot` as a successor.
    """

    @classmethod
    def _check_lhs(cls, lhs: sp.Basic):
        if not isinstance(lhs, SymbolicMemoryLoc):
            raise TypeError(
                f"Invalid type of left-hand side to `store`-assignment: {type(lhs)}"
            )


class Reduce(FlowgraphAssignment[TypedSymbol]):
    """`Reduce` assignments perform reductions onto a target typed symbol with a given operation.

    Like `Store`, `Reduce` assignments may only occur in blocks that connect to :math:`\\bot`.
    """

    __match_args__ = FlowgraphAssignment.__match_args__ + ("op",)

    def __init__(self, lhs: TypedSymbol, rhs: sp.Basic, op: ReductionOp):
        super().__init__(lhs, rhs)
        self._op = op

    @classmethod
    def _check_lhs(cls, lhs: sp.Basic):
        if not isinstance(lhs, TypedSymbol) or isinstance(lhs, Field.Access):
            raise TypeError(
                f"Invalid type of left-hand side to `reduce`-assignment: {type(lhs)}"
            )

    def subs(self, substitutions: SubstitutionMapping, **kwargs):
        return Reduce(
            cast(TypedSymbol, self._lhs.subs(substitutions, **kwargs)),
            self._rhs.subs(substitutions, **kwargs),
            self._op,
        )

    def _hashable_contents(self) -> tuple:
        return super()._hashable_contents() + (self._op,)

    @property
    def op(self) -> ReductionOp:
        return self._op


Effect = Store | Reduce


class FlowgraphNode(ABC):
    """Base class for data-flow graph nodes in pystencils.flow.

    Data flow graphs are immutable.

    Each node in the data flow graph exposes
     - zero or more predecessors
     - zero or more free symbols
       (that must be provided by exports from direct predecessors)
     - zero or more exported symbols (that can be consumed by successors)

    A node's successors are not tracked explicitly, as that would break the immutability.

    Nodes shall be hashable and equality-comparable.
    """

    _hash: int | None

    @staticmethod
    def _check_predecessors(predecessors: frozenset[FlowgraphNode]):
        if any(isinstance(pred, Bottom) for pred in predecessors):
            raise ValueError("Cannot add a bottom node as a predecessor.")

    def __init__(
        self, predecessors: Iterable[FlowgraphNode], name: str | None = None
    ) -> None:
        predecessors = frozenset(predecessors)
        self._check_predecessors(predecessors)
        self._predecessors = predecessors

        self._name = name

        self._hash = None

    @property
    def name(self) -> str:
        if self._name is None:
            h = abs(hash(self))
            return f"{type(self).__name__.lower()}_{hex(h)[2:8]}"

        return self._name

    @property
    def predecessors(self) -> frozenset[FlowgraphNode]:
        """This node's predecessors"""
        return self._predecessors

    @property
    def free_symbols(self) -> frozenset[sp.Symbol]:
        """Free symbols of this node.

        A symbol is free if it is used in this node's equations but not defined in this node.
        Free symbols must be imported from predecessors.
        """
        return frozenset()

    @property
    def exports(self) -> frozenset[sp.Symbol]:
        """Symbols exported by this node"""
        return frozenset()

    @property
    def effects(self) -> frozenset[Effect]:
        """Effectful assignments in this node"""
        return frozenset()

    @abstractmethod
    def subs(self, substitutions: SubstitutionMapping, **kwargs) -> FlowgraphNode: ...

    @abstractmethod
    def replace_predecessors(
        self, predecessors: Iterable[FlowgraphNode]
    ) -> FlowgraphNode:
        """Return a copy of this node with a new list of predecessors"""

    def walk(self) -> Generator[FlowgraphNode, None, None]:
        """Iterate the subgraph above this node in depth-first fashion."""
        seen: set[FlowgraphNode] = set()
        yield from self._walk_impl(seen)

    def _walk_impl(
        self, seen: set[FlowgraphNode]
    ) -> Generator[FlowgraphNode, None, None]:
        if self not in seen:
            seen.add(self)
            yield self

            for pred in sorted(self.predecessors, key=lambda n: n.name):
                yield from pred._walk_impl(seen)

    def atoms(self, *types) -> set[sp.Basic]:
        return set()

    @abstractmethod
    def _hashable_contents(self) -> tuple: ...

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(
                (type(self), self._name, self._predecessors) + self._hashable_contents()
            )
        return self._hash

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return False

        other = cast(FlowgraphNode, other)

        return (
            self._name == other._name
            and (self._predecessors == other._predecessors)
            and (self._hashable_contents() == other._hashable_contents())
        )

    def __str__(self) -> str:
        from .printing import PlainTextPrinter

        return PlainTextPrinter().print(self)

    def _repr_markdown_(self) -> str:
        return f"```\n{str(self)}\n```"


class Top(FlowgraphNode):
    """A flowgraph's top node.

    All instances of this class behave identically.
    """

    def __init__(self) -> None:
        super().__init__(frozenset(), name="⊤")

    def subs(self, substitutions: SubstitutionMapping, **kwargs) -> Top:
        return self

    def replace_predecessors(self, predecessors: Iterable[FlowgraphNode]) -> Top:
        if predecessors:
            raise ValueError("Cannot add predecessors to the top node")
        return self

    def _hashable_contents(self) -> tuple:
        return ()


class Bottom(FlowgraphNode):
    """A flowgraph's bottom node."""

    def __init__(self, predecessors: Iterable[FlowgraphNode]) -> None:
        super().__init__(predecessors, name="⊥")

    def replace_predecessors(self, predecessors: Iterable[FlowgraphNode]) -> Bottom:
        return Bottom(predecessors)

    def subs(self, substitutions: SubstitutionMapping, **kwargs) -> Bottom:
        return self

    def _hashable_contents(self) -> tuple:
        return ()


class EquationsBlock(FlowgraphNode):
    """A block of symbol assignments in static single-assignment form."""

    __match_args__ = ("assignments",)

    @staticmethod
    def _topological_order(
        asms: Sequence[FlowgraphAssignment],
    ) -> tuple[FlowgraphAssignment, ...]:
        edges = []
        symbol_defs: dict[sp.Symbol, int] = {
            asm.lhs: i
            for i, asm in enumerate(asms)
            if (
                isinstance(asm.lhs, sp.Symbol) and not isinstance(asm.lhs, Field.Access)
            )
        }

        for use_index, asm in enumerate(asms):
            for symb in asm.rhs.atoms(sp.Symbol):
                def_index = symbol_defs.get(symb)
                if def_index is not None:
                    edges.append((def_index, use_index))

        permutation = sp.topological_sort((range(len(asms)), edges))
        return tuple(asms[i] for i in permutation)

    def __init__(
        self,
        assignments: Sequence[FlowgraphAssignment],
        predecessors: Iterable[FlowgraphNode],
        name: str | None = None,
    ) -> None:
        super().__init__(predecessors, name=name)

        left_hand_sides: set[sp.Basic] = set()
        for asm in assignments:
            if asm.lhs in left_hand_sides:
                raise ValueError(
                    f"Violation of SSA form: Encountered duplicate left-hand side {asm.lhs}."
                )
            left_hand_sides.add(asm.lhs)

        self._assignments: tuple[FlowgraphAssignment, ...] = (
            EquationsBlock._topological_order(assignments)
        )

        self._symbols_exported: frozenset[sp.Symbol] = frozenset(
            a.lhs for a in self._assignments if isinstance(a, Export)
        )

        self._symbols_defined: frozenset[sp.Symbol] = (
            frozenset(a.lhs for a in self._assignments if isinstance(a, Let))
            | self._symbols_exported
        )

        self._symbols_used: frozenset[sp.Symbol] = frozenset(
            sym
            for sym in chain.from_iterable(
                a.rhs.atoms(sp.Symbol) for a in self._assignments
            )
            if not isinstance(sym, Field.Access)  # Field.Access inherits from sp.Symbol
        )

        self._free_symbols = self._symbols_used - self._symbols_defined

    @property
    def assignments(self) -> tuple[FlowgraphAssignment, ...]:
        return self._assignments

    @property
    def free_symbols(self) -> frozenset[sp.Symbol]:
        return self._free_symbols

    @property
    def exports(self) -> frozenset[sp.Symbol]:
        return self._symbols_exported

    @property
    def effects(self) -> frozenset[Effect]:
        return frozenset(asm for asm in self._assignments if isinstance(asm, Effect))

    def atoms(self, *types) -> set[sp.Basic]:
        return set().union(*(asm.atoms(*types) for asm in self._assignments))

    def subs(self, substitutions: SubstitutionMapping, **kwargs) -> EquationsBlock:
        """Apply substitutions to equations in this block.

        .. note::
            Substitutions of expressions for symbols (``symbol -> expr``) are only applied
            if ``symbol`` is a free symbol of this block.
        """
        substitutions = dict(substitutions)
        symbs = set(
            k
            for k in substitutions.keys()
            if isinstance(k, sp.Symbol) and not isinstance(k, Field.Access)
        )
        for symb in symbs:
            if symb not in self.free_symbols:
                del substitutions[symb]

        asms = [asm.subs(substitutions, **kwargs) for asm in self._assignments]
        return EquationsBlock(asms, predecessors=self._predecessors, name=self._name)

    def replace_predecessors(
        self, predecessors: Iterable[FlowgraphNode]
    ) -> EquationsBlock:
        return EquationsBlock(self._assignments, predecessors, name=self._name)

    def _hashable_contents(self) -> tuple:
        return (self._assignments,)


class Subgraph(FlowgraphNode):
    """Another flowgraph embedded as a subgraph."""

    __match_args__ = ("graph",)

    def __init__(
        self,
        subgraph: Flowgraph,
        predecessors: Iterable[FlowgraphNode],
        *,
        name: str | None = None,
    ):
        super().__init__(predecessors, name=name)

        self._graph = subgraph

    @property
    def graph(self) -> Flowgraph:
        return self._graph

    def replace_predecessors(
        self, predecessors: Iterable[FlowgraphNode]
    ) -> FlowgraphNode:
        return Subgraph(self._graph, predecessors, name=self._name)

    @property
    def free_symbols(self) -> frozenset[sp.Symbol]:
        return self._graph.free_symbols

    @property
    def exports(self) -> frozenset[sp.Symbol]:
        return self._graph.exports

    @property
    def effects(self) -> frozenset[Effect]:
        return self._graph.effects

    def atoms(self, *types) -> set[sp.Basic]:
        return self._graph.atoms(*types)

    def subs(self, substitutions: SubstitutionMapping, **kwargs) -> Subgraph:
        """Apply substitutions in this subgraph and return a new, transformed subgraph.

        .. note::
            Substitutions of expressions for symbols (``symbol -> expr``) are only applied
            if ``symbol`` is a free symbol of this subgraph.
        """
        return Subgraph(
            self._graph.subs(substitutions, **kwargs),
            self._predecessors,
            name=self._name,
        )

    def _hashable_contents(self) -> tuple:
        return (self._graph,)


class Cases(FlowgraphNode):
    """Case distinction."""

    __match_args__ = "branches"

    def _are_conditions_complete(
        self, conditions: Iterable[sp.Basic]
    ) -> tuple[bool, sp.Basic]:
        # Determine completeness of conditions
        covered_expr = reduce(lambda f, g: f | g, conditions)
        missing_cond = sp.simplify(sp.Not(covered_expr))
        is_complete = sp.sympify(missing_cond) is sp.false

        return bool(is_complete), missing_cond

    def __init__(
        self,
        branches: Iterable[tuple[sp.Basic, Flowgraph]],
        predecessors: Iterable[FlowgraphNode],
        name: str | None = None,
    ) -> None:
        super().__init__(predecessors, name=name)

        if not branches:
            raise ValueError("Case distinction branches cannot be empty")

        branches = list(branches)

        #   Determine exports
        exports = branches[0][1].exports

        for branch in branches[1:]:
            if branch[1].exports != exports:
                raise ValueError(
                    "All branches in a case distinction must have the same set of exports."
                )

        free_symbols = frozenset().union(
            *(
                set(s for s in b[0].atoms(sp.Symbol) if not isinstance(s, Field.Access))
                | b[1].free_symbols
                for b in branches
            )
        )

        if free_symbols & exports:
            raise ValueError(
                f"Symbol name conflicts: Exported symbols {free_symbols & exports} may not occur in conditions."
            )

        #   Determine completeness of cases

        is_complete, missing_cond = self._are_conditions_complete(
            b[0] for b in branches
        )
        if exports and not is_complete:
            raise ValueError(
                "Case distinction with exports must be complete."
                f"The following case is not covered: {missing_cond}"
            )

        if is_complete:
            #   If case distinction is complete, simplify last condition to `True`
            branches[-1] = (sp.true, branches[-1][1])

        self._branches = tuple(branches)
        self._is_complete = is_complete

        self._exports: frozenset[sp.Symbol] = frozenset(exports)

        self._free_symbols: frozenset[sp.Symbol] = free_symbols

        self._effects: frozenset[Effect] = frozenset().union(
            *(b[1].effects for b in branches)
        )

    @property
    def branches(self) -> tuple[tuple[sp.Basic, Flowgraph], ...]:
        return self._branches

    @property
    def conditions(self) -> tuple[sp.Basic, ...]:
        return tuple(b[0] for b in self._branches)

    @property
    def subgraphs(self) -> tuple[Flowgraph, ...]:
        return tuple(b[1] for b in self._branches)

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    @property
    def free_symbols(self) -> frozenset[sp.Symbol]:
        return self._free_symbols

    @property
    def exports(self) -> frozenset[sp.Symbol]:
        return self._exports

    @property
    def effects(self) -> frozenset[Effect]:
        return self._effects

    def atoms(self, *types) -> set[sp.Basic]:
        return set().union(
            *(
                cond.atoms(*types) | subgr.atoms(*types)
                for cond, subgr in self._branches
            )
        )

    def subs(self, substitutions: SubstitutionMapping, **kwargs) -> Cases:
        """Apply substitutions inside this case distinction.

        .. note::
            Substitutions of expressions for symbols (``symbol -> expr``) are only applied
            to a branch subgraph if ``symbol`` is a free symbol of that subgraph.
        """
        branches = [
            (c.subs(substitutions, **kwargs), b.subs(substitutions, **kwargs))
            for c, b in self._branches
        ]
        return Cases(branches, predecessors=self._predecessors, name=self._name)

    def replace_predecessors(
        self, predecessors: Iterable[FlowgraphNode]
    ) -> FlowgraphNode:
        return Cases(self._branches, predecessors, name=self._name)

    def _hashable_contents(self) -> tuple:
        return (self._branches,)
