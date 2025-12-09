from __future__ import annotations
from typing import overload, cast
from dataclasses import dataclass
from itertools import chain

from ..kernelcreation import KernelCreationContext

from ..memory import PsSymbol

from ..ast.analysis import collect_undefined_symbols
from ..ast.structural import (
    PsStructuralNode,
    PsLoop,
    PsBlock,
    PsDeclaration,
    PsPragma,
    PsAssignment,
)
from ..ast.expressions import PsSymbolExpr
from ..ast.vector import PsVecBroadcast, PsVecHorizontal
from ..ast.axes import (
    PsAxisRange,
    PsLoopAxis,
    PsParallelLoopAxis,
    PsSimdAxis,
    PsIterationAxis,
)
from ..ast.iteration import dfs_preorder

from .ast_vectorizer import AstVectorizer, VectorizationAxis, VectorizationContext
from .eliminate_constants import TypifyAndFold


@dataclass
class ModuloVariablePack:
    reduction_id: str
    symbols: tuple[PsSymbol, ...]
    declarations: tuple[PsStructuralNode, ...]
    reduction: tuple[PsStructuralNode, ...]


@dataclass
class MaterializationContext:
    modulo_variables: dict[str, list[ModuloVariablePack]]
    """A stack of modulo variable packs for each reduction target"""


class MaterializeAxes:
    def __init__(self, ctx: KernelCreationContext) -> None:
        self._ctx = ctx
        self._type_fold = TypifyAndFold(ctx)

    @overload
    def __call__(self, ast: PsBlock) -> PsBlock: ...

    @overload
    def __call__(self, ast: PsIterationAxis) -> PsStructuralNode: ...

    @overload
    def __call__(self, ast: PsStructuralNode) -> PsStructuralNode: ...

    def __call__(self, ast: PsStructuralNode) -> PsStructuralNode:
        #   Initialize modulo variable stacks
        mc = MaterializationContext(
            {
                name: [ModuloVariablePack(name, (rinfo.local_symbol,), (), ())]
                for name, rinfo in self._ctx.reduction_data.items()
            }
        )

        result = self.visit(ast, mc)

        nodes_before: list[PsStructuralNode] = []
        nodes_after: list[PsStructuralNode] = []
        for mvstack in mc.modulo_variables.values():
            if mvstack:
                for mvpack in mvstack[::-1]:
                    nodes_before += list(mvpack.declarations)
                    nodes_after = list(mvpack.reduction) + nodes_after

        if nodes_before or nodes_after:
            if not isinstance(result, PsBlock):
                result = PsBlock([result])
            result.statements = nodes_before + result.statements + nodes_after

        return result

    def visit(
        self, node: PsStructuralNode, mc: MaterializationContext
    ) -> PsStructuralNode:
        match node:
            case PsLoopAxis(PsAxisRange(ctr, start, stop, step), body):
                return PsLoop(
                    ctr, start, stop, step, cast(PsBlock, self.visit(body, mc))
                )

            case PsParallelLoopAxis(PsAxisRange(ctr, start, stop, step), body):
                parallel_construct = "omp parallel"

                if node.num_threads is not None:
                    parallel_construct += f" num_threads({node.num_threads})"

                new_body = cast(PsBlock, self.visit(body, mc))

                nodes_before: list[PsStructuralNode] = []
                nodes_after: list[PsStructuralNode] = []

                for reduction_id, mvstack in mc.modulo_variables.items():
                    #   Perform all but the final reduction per MV stack
                    while len(mvstack) > 1:
                        mvpack = mvstack.pop()
                        nodes_before += list(mvpack.declarations)
                        nodes_after = list(mvpack.reduction) + nodes_after

                    rinfo = self._ctx.reduction_data[reduction_id]
                    parallel_construct += (
                        f" reduction({rinfo.op.value}: {rinfo.local_symbol.name})"
                    )

                for_construct = "omp for"

                if node.schedule is not None:
                    for_construct += f" schedule({node.schedule})"

                if node.collapse is not None:
                    for_construct += f" collapse({node.collapse})"

                return PsBlock(
                    [
                        PsPragma(parallel_construct),
                        PsBlock(
                            nodes_before
                            + [
                                PsPragma(for_construct),
                                PsLoop(
                                    ctr,
                                    start,
                                    stop,
                                    step,
                                    new_body,
                                ),
                            ]
                            + nodes_after
                        ),
                    ]
                )

            case PsSimdAxis(lanes, PsAxisRange(ctr, start, _, step), body):
                counter_symbol = ctr.symbol
                scalar_ctr_decl = PsDeclaration(ctr, start)

                va = VectorizationAxis(counter_symbol, step)
                vc = VectorizationContext(self._ctx, lanes, va)
                vectorize = AstVectorizer(self._ctx)

                #   Set up and push vectorized modulo variables

                outer_mvs = self._find_modulo_variables(body, mc)
                for mv_symb, reduction_id in outer_mvs.items():
                    rinfo = self._ctx.reduction_data[reduction_id]
                    vector_mv_symb = vc.vectorize_symbol(mv_symb)
                    vector_mv_init = self._type_fold(
                        PsDeclaration(
                            PsSymbolExpr(vector_mv_symb),
                            PsVecBroadcast(
                                lanes,
                                rinfo.init_val.clone(),
                            ),
                        )
                    )
                    mv_horizontal_reduce = PsAssignment(
                        PsSymbolExpr(mv_symb),
                        PsVecHorizontal(
                            PsSymbolExpr(mv_symb),
                            PsSymbolExpr(vector_mv_symb),
                            rinfo.op,
                        ),
                    )
                    mc.modulo_variables[reduction_id].append(
                        ModuloVariablePack(
                            reduction_id,
                            (vector_mv_symb,),
                            (vector_mv_init,),
                            (mv_horizontal_reduce,),
                        )
                    )

                simd_body = vectorize(body, vc)

                prepend_decls = [scalar_ctr_decl]

                counter_decl = vectorize.get_counter_declaration(vc)
                if vc.vectorized_symbols[counter_symbol] in collect_undefined_symbols(
                    simd_body
                ):
                    prepend_decls.append(counter_decl)

                simd_body.statements = prepend_decls + simd_body.statements

                return simd_body

            case PsIterationAxis():
                raise NotImplementedError(
                    f"Don't know how to materialize axis of type {type(node)}"
                )

            case other:
                other.children = [
                    (self.visit(c, mc) if isinstance(c, PsStructuralNode) else c)
                    for c in other.children
                ]
                return other

    def _find_modulo_variables(
        self, ast: PsStructuralNode, mc: MaterializationContext
    ) -> dict[PsSymbol, str]:
        assigned_symbols: set[PsSymbol] = set(
            asm.lhs.symbol
            for asm in dfs_preorder(ast)
            if isinstance(asm, PsAssignment) and isinstance(asm.lhs, PsSymbolExpr)
        )

        result: dict[PsSymbol, str] = dict()

        for mvpack in chain.from_iterable(mc.modulo_variables.values()):
            for mvsymb in mvpack.symbols:
                if mvsymb in assigned_symbols:
                    result[mvsymb] = mvpack.reduction_id

        return result
