from typing import TypeVar, cast

from ..kernelcreation import KernelCreationContext
from ..ast import PsAstNode
from ..ast.structural import PsDeclaresSymbolTrait
from ..ast.iteration import dfs_preorder
from .canonicalize_symbols import CanonicalizeSymbols, CanonContext

__all__ = ["CanonicalClone"]


Node_T = TypeVar("Node_T", bound=PsAstNode)


class CanonicalClone:
    """Clone a subtree, and rename all symbols declared inside it to retain canonicality."""

    def __init__(self, ctx: KernelCreationContext) -> None:
        self._ctx = ctx
        self._canonicalize = CanonicalizeSymbols(ctx, constify=False)

    def __call__(self, node: Node_T) -> Node_T:
        ast_clone = node.clone()
        declared_symbols = set(
            decl.declared_symbol
            for decl in dfs_preorder(node)
            if isinstance(decl, PsDeclaresSymbolTrait)
        )
        cc = CanonContext(self._ctx)
        cc.encountered_symbols = declared_symbols
        self._canonicalize.visit(ast_clone, cc)
        return cast(Node_T, ast_clone)
