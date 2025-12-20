from __future__ import annotations
from typing import overload, cast, Iterable
from dataclasses import dataclass
from itertools import chain

from ..kernelcreation import Typifier
from ..kernelcreation.context import KernelCreationContext, ReductionInfo

from ..memory import PsSymbol
from ..functions import GpuGridScope, PsGpuIndexingFunction

from ..ast.analysis import collect_undefined_symbols
from ..ast.structural import (
    PsStructuralNode,
    PsLoop,
    PsBlock,
    PsDeclaration,
    PsPragma,
    PsAssignment,
    PsConditional,
)
from ..ast.expressions import PsExpression, PsSymbolExpr, PsLt, PsAnd
from ..ast.vector import PsVecBroadcast, PsVecHorizontal
from ..ast.axes import (
    PsAxisRange,
    PsLoopAxis,
    PsParallelLoopAxis,
    PsSimdAxis,
    PsIterationAxis,
    PsGpuIndexingAxis,
    PsGpuBlockAxis,
    PsGpuThreadAxis,
    PsGpuBlockXThreadAxis,
)
from ..ast.iteration import dfs_preorder

from .ast_vectorizer import AstVectorizer, VectorizationAxis, VectorizationContext
from .eliminate_constants import TypifyAndFold
from .rewrite import collapse_blocks


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
    """Materialize iteration axes.

    This transformer converts all iteration axis in a given AST to their lower-level
    implementation.
    It introduces loops for loop axes,
    OpenMP constructs for parallel loops,
    applies vectorization in SIMD axes
    and constructs GPU index translations for GPU axes.

    The axis materializer furthermore introduces declarations and agglomeration of modulo
    variables for reductions occuring in the kernel.
    """

    def __init__(self, ctx: KernelCreationContext) -> None:
        self._ctx = ctx
        self._type_fold = TypifyAndFold(ctx)
        self._typify = Typifier(ctx)

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
                name: [self._create_root_mvpack(name, rinfo)]
                for name, rinfo in self._ctx.reduction_data.items()
            }
        )

        if not isinstance(ast, PsBlock):
            ast = PsBlock([ast])

        #   Transform remaining AST
        body = cast(PsBlock, self.visit(ast, mc))

        #   Introduce and collapse reduction modulo variables
        body = self._collapse_modulo_variables(body, mc.modulo_variables.keys(), mc)

        body = cast(PsBlock, collapse_blocks(body))

        return body

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

                    last_mvpack = mvstack[0]
                    assert len(last_mvpack.symbols) == 1

                    parallel_construct += (
                        f" reduction({rinfo.op.value}: {last_mvpack.symbols[0].name})"
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

            case PsGpuIndexingAxis():
                #   Combine as many immediately nested indexing axes as possible into one guard
                nested_axes, body = self._collect_nested_gpu_axes(node)
                body = cast(PsBlock, self.visit(body, mc))
                result = self._handle_nested_gpu_axes(nested_axes, body)
                return result

            case PsIterationAxis():
                raise NotImplementedError(
                    f"Don't know how to materialize axis of type {type(node)}"
                )

            case _:
                node.children = [
                    (self.visit(c, mc) if isinstance(c, PsStructuralNode) else c)
                    for c in node.children
                ]
                return node

    def _create_root_mvpack(
        self, name: str, reduction_info: ReductionInfo
    ) -> ModuloVariablePack:
        local_symbol = reduction_info.local_symbol

        return ModuloVariablePack(name, (local_symbol,), (), ())

    def _collapse_modulo_variables(
        self,
        block: PsBlock,
        var_names: Iterable[str],
        mc: MaterializationContext,
    ) -> PsBlock:
        nodes_before: list[PsStructuralNode] = []
        nodes_after: list[PsStructuralNode] = []
        for mvstack in [mc.modulo_variables[name] for name in var_names]:
            if mvstack:
                for mvpack in mvstack:
                    nodes_before += list(mvpack.declarations)
                    nodes_after = list(mvpack.reduction) + nodes_after
                mvstack.clear()

        block.statements = nodes_before + block.statements + nodes_after

        return block

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

    def _collect_nested_gpu_axes(
        self, first: PsGpuIndexingAxis
    ) -> tuple[list[PsGpuIndexingAxis], PsBlock]:
        nested_axes: list[PsGpuIndexingAxis] = []
        candidate: PsGpuIndexingAxis | None = first

        while candidate is not None:
            nested_axes.append(candidate)

            next_candidate: PsGpuIndexingAxis | None = None
            if len(candidate.body.statements) == 1:
                if isinstance(candidate.body.statements[0], PsGpuIndexingAxis):
                    next_candidate = candidate.body.statements[0]

            candidate = next_candidate

        body = nested_axes[-1].body
        return nested_axes, body

    def _gpu_axis_ctr_and_guard(
        self, axis: PsGpuIndexingAxis
    ) -> tuple[PsDeclaration, PsExpression]:
        gpu_idx: PsExpression
        match axis:
            case PsGpuBlockAxis(gpu_dim):
                gpu_idx = PsGpuIndexingFunction(GpuGridScope.blockIdx, gpu_dim)()

            case PsGpuThreadAxis(gpu_dim):
                gpu_idx = PsGpuIndexingFunction(GpuGridScope.threadIdx, gpu_dim)()

            case PsGpuBlockXThreadAxis(gpu_dim):
                blockIdx = PsGpuIndexingFunction(GpuGridScope.blockIdx, gpu_dim)
                blockDim = PsGpuIndexingFunction(GpuGridScope.blockDim, gpu_dim)
                threadIdx = PsGpuIndexingFunction(GpuGridScope.threadIdx, gpu_dim)
                gpu_idx = blockIdx() * blockDim() + threadIdx()

        xrange = axis.range

        ctr_decl = self._type_fold(
            PsDeclaration(xrange.counter, xrange.start + xrange.step * gpu_idx)
        )
        ctr_guard = PsLt(xrange.counter.clone(), xrange.stop)

        return ctr_decl, ctr_guard

    def _handle_nested_gpu_axes(self, axes: list[PsGpuIndexingAxis], body: PsBlock):
        ctr_decls: list[PsDeclaration] = []
        guard_expr: PsExpression | None = None

        for ax in axes:
            ctr_decl, ctr_guard = self._gpu_axis_ctr_and_guard(ax)
            ctr_decls.append(ctr_decl)

            if guard_expr:
                guard_expr = PsAnd(guard_expr, ctr_guard)
            else:
                guard_expr = ctr_guard

        assert guard_expr is not None
        guard_expr = self._type_fold(guard_expr)

        return PsBlock(ctr_decls + [PsConditional(guard_expr, body)])
