from typing import overload

from ..memory import PsSymbol
from ..ast import PsAstNode
from ..ast.structural import PsStructuralNode, PsBlock
from ..ast.expressions import PsExpression, PsSymbolExpr


@overload
def substitute_symbols(node: PsBlock, subs: dict[PsSymbol, PsExpression]) -> PsBlock:
    pass


@overload
def substitute_symbols(
    node: PsExpression, subs: dict[PsSymbol, PsExpression]
) -> PsExpression:
    pass


@overload
def substitute_symbols(
    node: PsStructuralNode, subs: dict[PsSymbol, PsExpression]
) -> PsStructuralNode:
    pass


@overload
def substitute_symbols(
    node: PsAstNode, subs: dict[PsSymbol, PsExpression]
) -> PsAstNode:
    pass


def substitute_symbols(
    node: PsAstNode, subs: dict[PsSymbol, PsExpression]
) -> PsAstNode:
    """Substitute expressions for symbols throughout a subtree."""
    match node:
        case PsSymbolExpr(symb) if symb in subs:
            return subs[symb].clone()
        case _:
            node.children = [substitute_symbols(c, subs) for c in node.children]
            return node


def collapse_blocks(node: PsStructuralNode) -> PsStructuralNode:
    """Collapse trivially nested blocks to improve readability.

    Blocks that just have another block as their single child are collapsed.
    """

    match node:
        case PsBlock([PsBlock()]):
            return collapse_blocks(node.statements[0])
        case _:
            node.children = [
                (collapse_blocks(c) if isinstance(c, PsStructuralNode) else c)
                for c in node.children
            ]
            return node
