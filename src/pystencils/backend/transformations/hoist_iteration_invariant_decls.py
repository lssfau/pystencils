from typing import cast, Sequence

from ..kernelcreation import KernelCreationContext
from ..ast import PsAstNode
from ..ast.structural import (
    PsBlock,
    PsLoop,
    PsConditional,
    PsDeclaration,
    PsAssignment,
    PsStructuralNode,
    PsPragma,
)
from ..ast.expressions import (
    PsExpression,
    PsSymbolExpr,
    PsConstantExpr,
    PsLiteralExpr,
    PsCall,
    PsBufferAcc,
    PsSubscript,
    PsLookup,
    PsUnOp,
    PsBinOp,
    PsArrayInitList,
)
from ..ast.axes import PsAxesCube, PsAxisRange, PsIterationAxis
from ..ast.util import determine_memory_object

from ...types import PsDereferencableType
from ..memory import PsSymbol
from ..functions import PsMathFunction

__all__ = ["HoistIterationInvariantDeclarations"]


class HoistContext:
    def __init__(self) -> None:
        self.hoisted_nodes: list[PsDeclaration] = []
        self.assigned_symbols: set[PsSymbol] = set()
        self.mutated_symbols: set[PsSymbol] = set()
        self.invariant_symbols: set[PsSymbol] = set()

    def _is_invariant(self, expr: PsExpression) -> bool:
        def args_invariant(expr):
            return all(
                self._is_invariant(cast(PsExpression, arg)) for arg in expr.children
            )

        match expr:
            case PsSymbolExpr(symbol):
                return (symbol not in self.assigned_symbols) or (
                    symbol in self.invariant_symbols
                )

            case PsConstantExpr() | PsLiteralExpr():
                return True

            case PsCall(func):
                return isinstance(func, PsMathFunction) and args_invariant(expr)

            case PsSubscript() | PsLookup():
                return determine_memory_object(expr)[1] and args_invariant(expr)

            case PsBufferAcc(ptr, _):
                #   Regular pointer derefs are never invariant, since we cannot reason about aliasing
                ptr_type = cast(PsDereferencableType, ptr.get_dtype())
                return ptr_type.base_type.const and args_invariant(expr)

            case PsUnOp() | PsBinOp() | PsArrayInitList():
                return args_invariant(expr)

            case _:
                return False


class HoistIterationInvariantDeclarations:
    """Hoist loop-invariant declarations out of the loop nest.

    This transformation moves loop-invariant symbol declarations outside of the loop
    nest to prevent their repeated execution within the loops.
    If this transformation results in the complete elimination of a loop body, the respective loop
    is removed.

    `HoistIterationInvariantDeclarations` assumes that symbols are canonical;
    in particular, each symbol may have at most one declaration.
    To ensure this, a `CanonicalizeSymbols` pass should be run before `HoistIterationInvariantDeclarations`.

    `HoistIterationInvariantDeclarations` assumes that all `PsMathFunction` s are pure (have no side effects),
    but makes no such assumption about instances of `CFunction`.
    """

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx

    def __call__(self, node: PsAstNode) -> PsAstNode:
        return self.visit(node)

    def visit(self, node: PsAstNode) -> PsAstNode:
        """Search the outermost loop and start the hoisting cascade there."""
        match node:
            case PsLoop() | PsAxesCube() | PsIterationAxis():
                temp_block = PsBlock([node])
                temp_block = cast(PsBlock, self.visit(temp_block))
                if temp_block.statements == [node]:
                    return node
                else:
                    return temp_block

            case PsBlock(statements):
                statements_new: list[PsStructuralNode] = []
                for stmt in statements:
                    if isinstance(stmt, PsLoop | PsAxesCube | PsIterationAxis):
                        loop = stmt
                        hc = self._hoist(loop)
                        self._insert_hoisted_nodes(
                            statements_new, hc.hoisted_nodes, loop
                        )
                    else:
                        self.visit(stmt)
                        statements_new.append(stmt)

                node.statements = statements_new
                return node

            case PsConditional(_, then, els):
                self.visit(then)
                if els is not None:
                    self.visit(els)
                return node

            case _:
                #   if the node is none of the above, end the search
                return node

        #   end match

    def _hoist(self, iteration_node: PsLoop | PsAxesCube | PsIterationAxis) -> HoistContext:
        """Hoist invariant declarations out of the given loop."""
        hc = HoistContext()

        match iteration_node:
            case PsLoop(ctr) | PsIterationAxis(PsAxisRange(ctr)):
                hc.assigned_symbols.add(ctr.symbol)
                hc.mutated_symbols.add(ctr.symbol)
            case PsAxesCube(xranges):
                for xrange in xranges:
                    hc.assigned_symbols.add(xrange.counter.symbol)
                    hc.mutated_symbols.add(xrange.counter.symbol)

        self._prepare_hoist(iteration_node.body, hc)
        self._hoist_from_block(iteration_node.body, hc)
        return hc

    def _prepare_hoist(self, node: PsAstNode, hc: HoistContext):
        """Collect all symbols assigned within a loop body,
        and recursively apply loop-invariant code motion to any nested loops."""
        match node:
            case PsExpression():
                return

            case PsDeclaration(PsSymbolExpr(lhs_symb), _):
                hc.assigned_symbols.add(lhs_symb)

            case PsAssignment(PsSymbolExpr(lhs_symb), _):
                hc.assigned_symbols.add(lhs_symb)
                hc.mutated_symbols.add(lhs_symb)

            case PsAssignment(_, _):
                return

            case PsBlock(statements):
                statements_new: list[PsStructuralNode] = []
                for stmt in statements:
                    if isinstance(stmt, PsLoop | PsAxesCube | PsIterationAxis):
                        loop = stmt
                        nested_hc = self._hoist(loop)
                        hc.assigned_symbols |= nested_hc.assigned_symbols
                        hc.mutated_symbols |= nested_hc.mutated_symbols
                        self._insert_hoisted_nodes(
                            statements_new, nested_hc.hoisted_nodes, loop
                        )
                    else:
                        self._prepare_hoist(stmt, hc)
                        statements_new.append(stmt)
                node.statements = statements_new

            case _:
                for c in node.children:
                    self._prepare_hoist(c, hc)

    def _hoist_from_block(self, block: PsBlock, hc: HoistContext):
        """Hoist invariant declarations from the given block, and any directly nested blocks.

        This method processes only statements of the given block, and any blocks directly nested inside it.
        It does not descend into control structures like conditionals and nested loops.
        """
        statements_new: list[PsStructuralNode] = []

        for node in block.statements:
            if isinstance(node, PsDeclaration):
                lhs_symb = node.declared_symbol
                if lhs_symb not in hc.mutated_symbols and hc._is_invariant(node.rhs):
                    hc.hoisted_nodes.append(node)
                    hc.invariant_symbols.add(node.declared_symbol)
                else:
                    statements_new.append(node)
            else:
                if isinstance(node, PsBlock):
                    self._hoist_from_block(node, hc)
                statements_new.append(node)

        block.statements = statements_new

    def _insert_hoisted_nodes(
        self,
        statements: list[PsStructuralNode],
        hoisted: Sequence[PsStructuralNode],
        iteration_node: PsLoop | PsAxesCube | PsIterationAxis,
    ):
        """Insert hoisted nodes before the loop in the given statement list.

        This function will take care to not disconnect any pragmas from loops."""
        pragmas: list[PsPragma] = []

        if isinstance(iteration_node, PsLoop):
            while statements and isinstance(statements[-1], PsPragma):
                pragmas.append(cast(PsPragma, statements.pop()))

        statements += hoisted

        if iteration_node.body.statements:
            statements += pragmas[::-1]
            statements.append(iteration_node)
