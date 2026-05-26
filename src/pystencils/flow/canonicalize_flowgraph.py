from __future__ import annotations

from collections import defaultdict
from itertools import chain
from typing import TypeVar, cast
from dataclasses import dataclass, field

import sympy as sp

from ..field import Field
from .flowgraph import (
    FlowgraphNode,
    Bottom,
    Top,
    EquationsBlock,
    FlowgraphAssignment,
    Cases,
    Effect,
    Subgraph,
)

Node_T = TypeVar("Node_T", bound=FlowgraphNode)


class CanonicalizationError(Exception):
    """Indicates that a flowgraph violates a canonicality condition."""


@dataclass
class CanonicalizationResult:
    """Result of the flowgraph canonicalizer"""

    bottom: Bottom
    free_symbols: frozenset[sp.Symbol]
    exports: frozenset[sp.Symbol]
    effects: frozenset[Effect]
    fields_written: frozenset[Field]
    fields_read: frozenset[Field]


@dataclass
class CanonicalizeContext:
    successors: defaultdict[FlowgraphNode, set[FlowgraphNode]] = field(
        default_factory=lambda: defaultdict(set)
    )

    free_symbols: set[sp.Symbol] = field(default_factory=set)


class CanonicalizeFlowgraph:
    """Check a flowgraph for consistency and transform it into canonical form.

    A flowgraph must meet the following requirements to be eligible for code generation:

    - It has an explicit bottom node :math:`\\bot` and an explicit top node :math:`\\top`
    - Each node apart from :math:`\\top` has at least one predecessor
      (i.e. :math:`\\top` is the only root)
    - Each node has at least one successor (i.e. :math:`\\bot` is the only sink)
    - On each node, each free symbol must be exported by exactly one predecessor
      (where :math:`\\top` implicitly exports all symbols that are globally free)
    - Each memory location and reduction target is written to no more than once

    Instances of this class will check these conditions; rectify some of them if possible,
    and raise an error otherwise.
    In particular, `CanonicalizeFlowgraph` will add explicit bottom and top nodes
    if these are missing, and introduce edges from :math:`\\top` to any node with
    dangling free symbols, and to any node that has no predecessors.

    The canonicalizer returns a `CanonicalizationResult`, containing the canonicalized
    graph together with its sets of free symbols, exports, and side effects.
    """

    def __call__(self, graph: FlowgraphNode) -> CanonicalizationResult:
        if not isinstance(graph, Bottom):
            bottom = Bottom([graph])
        else:
            bottom = graph

        cc = CanonicalizeContext()
        bottom = self._check_and_collect_edges(bottom, cc, {Top(): Top()})
        self._check_side_effects(bottom, cc)

        exports: frozenset[sp.Symbol] = frozenset().union(
            *(node.exports for node in bottom.predecessors)
        )

        effects: frozenset[Effect] = frozenset().union(
            *(node.effects for node in bottom.predecessors)
        )

        fields_read = frozenset().union(
            *(self._collect_field_reads(n) for n in bottom.walk())
        )

        fields_written = frozenset(
            ef.lhs.field for ef in effects if isinstance(ef.lhs, Field.Access)
        )

        return CanonicalizationResult(
            bottom,
            frozenset(cc.free_symbols),
            exports,
            effects,
            fields_written,
            fields_read,
        )

    def _check_and_collect_edges(
        self,
        node: Node_T,
        cc: CanonicalizeContext,
        memory: dict[FlowgraphNode, FlowgraphNode],
    ) -> Node_T:
        """Performs the first pass of backward analyses and transformations:

        - Check that all free symbols of a node are imported from exactly one predecessor,
          adding :math:`\\top` to the node's predecessors if necessary
        - Populates the canonicalization context object with successors for each node
        """
        if node in memory:
            return cast(Node_T, memory[node])

        exported_symbols = list(chain(*(p.exports for p in node.predecessors)))
        unique_exported_symbols: set[sp.Symbol] = set()

        for symb in exported_symbols:
            if symb in unique_exported_symbols:
                raise CanonicalizationError(
                    f"Flowgraph is ill-formed: Predecessor providing free symbol {symb} is ambiguous\n"
                    f"    At: {node}"
                )
            unique_exported_symbols.add(symb)

        imports_from_top = node.free_symbols - unique_exported_symbols
        cc.free_symbols |= imports_from_top

        preds = [
            self._check_and_collect_edges(p, cc, memory) for p in node.predecessors
        ]

        #   Add edge to top where necessary
        if (not preds) or (imports_from_top and Top() not in preds):
            preds.append(Top())

        node = cast(Node_T, node.replace_predecessors(preds))
        memory[node] = node

        for pred in node.predecessors:
            cc.successors[pred].add(node)

        return node

    def _check_side_effects(self, bottom: Bottom, cc: CanonicalizeContext):
        seen_effect_targets: set[sp.Basic] = set()
        for node in bottom.walk():
            for effect_target in set(ef.lhs for ef in node.effects):
                if effect_target in seen_effect_targets:
                    raise CanonicalizationError(
                        "Ambiguous side effect: "
                        f"Effect target {effect_target} is assigned in two different locations"
                    )
                seen_effect_targets.add(effect_target)

    def _collect_field_reads(
        self, node: FlowgraphNode | FlowgraphAssignment
    ) -> frozenset[Field]:
        match node:
            case EquationsBlock(assignments):
                return frozenset().union(
                    *(self._collect_field_reads(asm) for asm in assignments)
                )
            case FlowgraphAssignment(lhs, rhs):
                accesses: set[Field.Access] = lhs.atoms(Field.Access) | rhs.atoms(
                    Field.Access
                )
                if isinstance(lhs, Field.Access):
                    accesses.remove(lhs)
                return frozenset(a.field for a in accesses)
            case Cases():
                conds_accesses: frozenset[Field.Access] = frozenset().union(
                    *(cond.atoms(Field.Access) for cond in node.conditions)
                )

                conds_fields: frozenset[Field] = frozenset(
                    a.field for a in conds_accesses
                )
                subgrs_fields: frozenset[Field] = frozenset().union(
                    *(
                        chain.from_iterable(
                            self._collect_field_reads(node) for node in subgr.walk()
                        )
                        for subgr in node.subgraphs
                    )
                )
                return conds_fields | subgrs_fields
            case Subgraph(g):
                return frozenset().union(
                    *(self._collect_field_reads(n) for n in g.walk())
                )
            case Top() | Bottom():
                return frozenset()
            case _:
                assert False, "unexpected node"
