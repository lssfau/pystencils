from __future__ import annotations
from typing import Iterable

from ..kernelcreation import KernelCreationContext, Typifier
from ..kernelcreation.context import ReductionInfo

from ..ast.structural import PsBlock, PsDeclaration, PsAssignment, PsStructuralNode
from ..ast.expressions import PsExpression, PsMemAcc
from ..constants import PsConstant
from ..functions import PsReductionWriteBack


class ReductionsToMemory:
    """Introduce IR nodes for performing reductions to memory.

    This transformer takes a `block <PsBlock>` and adds to it the declarations
    and write-back IR functions for the given list of reductions.
    Modulo variable declarations are prepended, and write-back logic is appended
    to the end of the block.
    """

    def __init__(self, ctx: KernelCreationContext, reductions: Iterable[ReductionInfo]):
        self._ctx = ctx
        self._reductions = tuple(reductions)
        self._typify = Typifier(ctx)

    def __call__(self, block: PsBlock) -> PsBlock:
        nodes_before: list[PsStructuralNode] = []
        nodes_after: list[PsStructuralNode] = []

        for rinfo in self._reductions:
            mv_decl, write_back = self._handle_reduction(rinfo)
            nodes_before.append(mv_decl)
            nodes_after.append(write_back)

        block.statements = nodes_before + block.statements + nodes_after
        return block

    def _handle_reduction(
        self, reduction_info: ReductionInfo
    ) -> tuple[PsDeclaration, PsAssignment]:
        local_symbol_expr = PsExpression.make(reduction_info.local_symbol)
        local_symbol_decl = self._typify(
            PsDeclaration(local_symbol_expr, reduction_info.init_val.clone())
        )

        ptr_symbol_expr = PsExpression.make(reduction_info.writeback_ptr_symbol)
        ptr_access = PsMemAcc(
            ptr_symbol_expr, PsExpression.make(PsConstant(0, self._ctx.index_dtype))
        )
        write_back_call = PsReductionWriteBack(reduction_info.op)(
            ptr_symbol_expr.clone(), local_symbol_expr.clone()
        )
        write_back_asm = self._typify(PsAssignment(ptr_access, write_back_call))

        return local_symbol_decl, write_back_asm
