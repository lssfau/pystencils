from enum import Enum, auto
from typing import Callable, overload, Sequence

from ..kernelcreation.context import ReductionInfo

from ..kernelcreation import KernelCreationContext
from ..constants import PsConstant
from ..ast import PsAstNode
from ..ast.structural import (
    PsLoop,
    PsBlock,
    PsDeclaration,
    PsAssignment,
    PsStructuralNode,
)
from ..ast.expressions import PsExpression, PsTernary, PsGt, PsSymbolExpr
from ..ast.vector import PsVecBroadcast, PsVecHorizontal
from ..ast.analysis import collect_undefined_symbols

from .ast_vectorizer import VectorizationAxis, VectorizationContext, AstVectorizer
from .rewrite import substitute_symbols


class LoopVectorizer:
    """Vectorize loops.

    The loop vectorizer provides methods to vectorize single loops inside an AST
    using a given number of vector lanes.
    During vectorization, the loop body is transformed using the `AstVectorizer`,
    The loop's limits are adapted according to the number of vector lanes,
    and a block treating trailing iterations is optionally added.

    Args:
        ctx: The current kernel creation context
        lanes: The number of vector lanes to use
        trailing_iters: Mode for the treatment of trailing iterations
    """

    class TrailingItersTreatment(Enum):
        """How to treat trailing iterations during loop vectorization."""

        SCALAR_LOOP = auto()
        """Cover trailing iterations using a scalar remainder loop."""

        MASKED_BLOCK = auto()
        """Cover trailing iterations using a masked block."""

        NONE = auto()
        """Assume that the loop iteration count is a multiple of the number of lanes
        and do not cover any trailing iterations"""

    def __init__(
        self,
        ctx: KernelCreationContext,
        lanes: int,
        reductions: Sequence[ReductionInfo] = (),
        trailing_iters: TrailingItersTreatment = TrailingItersTreatment.SCALAR_LOOP,
    ):
        self._ctx = ctx
        self._lanes = lanes
        self._reductions = reductions
        self._trailing_iters = trailing_iters

        from ..kernelcreation import Typifier
        from .eliminate_constants import TypifyAndFold

        self._typify = Typifier(ctx)
        self._vectorize_ast = AstVectorizer(ctx)
        self._type_fold = TypifyAndFold(ctx)

    @overload
    def vectorize_select_loops(
        self, node: PsBlock, predicate: Callable[[PsLoop], bool]
    ) -> PsBlock: ...

    @overload
    def vectorize_select_loops(
        self, node: PsLoop, predicate: Callable[[PsLoop], bool]
    ) -> PsLoop | PsBlock: ...

    @overload
    def vectorize_select_loops(
        self, node: PsAstNode, predicate: Callable[[PsLoop], bool]
    ) -> PsAstNode: ...

    def vectorize_select_loops(
        self, node: PsAstNode, predicate: Callable[[PsLoop], bool]
    ) -> PsAstNode:
        """Select and vectorize loops from a syntax tree according to a predicate.

        Finds each loop inside a subtree and evaluates ``predicate`` on them.
        If ``predicate(loop)`` evaluates to `True`, the loop is vectorized.

        Loops nested inside a vectorized loop will not be processed.

        Args:
            node: Root of the subtree to process
            predicate: Callback telling the vectorizer which loops to vectorize
        """
        match node:
            case PsLoop() if predicate(node):
                return self.vectorize_loop(node)
            case PsExpression():
                return node
            case _:
                node.children = [
                    self.vectorize_select_loops(c, predicate) for c in node.children
                ]
                return node

    def __call__(self, loop: PsLoop) -> PsLoop | PsBlock:
        return self.vectorize_loop(loop)

    def vectorize_loop(self, loop: PsLoop) -> PsLoop | PsBlock:
        """Vectorize the given loop."""
        scalar_ctr_expr = loop.counter
        scalar_ctr = scalar_ctr_expr.symbol

        #   Prepare axis
        axis = VectorizationAxis(scalar_ctr, step=loop.step)

        #   Prepare vectorization context
        vc = VectorizationContext(self._ctx, self._lanes, axis)

        #   Prepare vector counter
        vector_counter_decl = self._vectorize_ast.get_counter_declaration(vc)

        #   Prepare reductions found in loop body
        simd_init_local_reduction_vars: list[PsStructuralNode] = []
        simd_writeback_local_reduction_vars: list[PsStructuralNode] = []
        for stmt in loop.body.statements:
            if isinstance(stmt, PsAssignment) and isinstance(stmt.lhs, PsSymbolExpr):
                for reduction_info in self._reductions:
                    if stmt.lhs.symbol == reduction_info.local_symbol:
                        # Vectorize symbol for local copy
                        local_symbol = stmt.lhs.symbol
                        vector_symb = vc.vectorize_symbol(local_symbol)

                        # Declare and init vector
                        simd_init_local_reduction_vars += [
                            self._type_fold(
                                PsDeclaration(
                                    PsSymbolExpr(vector_symb),
                                    PsVecBroadcast(
                                        self._lanes,
                                        reduction_info.init_val.clone(),
                                    ),
                                )
                            )
                        ]

                        # Write back vectorization result
                        simd_writeback_local_reduction_vars += [
                            PsAssignment(
                                PsSymbolExpr(local_symbol),
                                PsVecHorizontal(
                                    PsSymbolExpr(local_symbol),
                                    PsSymbolExpr(vector_symb),
                                    reduction_info.op,
                                ),
                            )
                        ]

                        break

        #   Generate vectorized loop body
        simd_body = self._vectorize_ast(loop.body, vc)

        if vc.vectorized_symbols[scalar_ctr] in collect_undefined_symbols(simd_body):
            simd_body.statements.insert(0, vector_counter_decl)

        #   Build new loop limits
        simd_start = loop.start.clone()

        simd_step = self._ctx.get_new_symbol(
            f"__{scalar_ctr.name}_simd_step", scalar_ctr.get_dtype()
        )
        simd_step_decl = self._type_fold(
            PsDeclaration(
                PsExpression.make(simd_step),
                loop.step.clone() * PsExpression.make(PsConstant(self._lanes)),
            )
        )

        #   Each iteration must satisfy `ctr + step * (lanes - 1) < stop`
        simd_stop = self._ctx.get_new_symbol(
            f"__{scalar_ctr.name}_simd_stop", scalar_ctr.get_dtype()
        )
        simd_stop_decl = self._type_fold(
            PsDeclaration(
                PsExpression.make(simd_stop),
                loop.stop.clone()
                - (
                    PsExpression.make(PsConstant(self._lanes))
                    - PsExpression.make(PsConstant(1))
                )
                * loop.step.clone(),
            )
        )

        simd_loop = PsLoop(
            PsExpression.make(scalar_ctr),
            simd_start,
            PsExpression.make(simd_stop),
            PsExpression.make(simd_step),
            simd_body,
        )

        #   Treat trailing iterations
        match self._trailing_iters:
            case LoopVectorizer.TrailingItersTreatment.SCALAR_LOOP:
                trailing_start = self._ctx.get_new_symbol(
                    f"__{scalar_ctr.name}_trailing_start", scalar_ctr.get_dtype()
                )

                trailing_start_decl = self._type_fold(
                    PsDeclaration(
                        PsExpression.make(trailing_start),
                        PsTernary(
                            #   If at least one vectorized iteration took place...
                            PsGt(
                                PsExpression.make(simd_stop),
                                simd_start.clone(),
                            ),
                            #   start from the smallest non-valid multiple of simd_step, offset from simd_start
                            (
                                (
                                    PsExpression.make(simd_stop)
                                    - simd_start.clone()
                                    - PsExpression.make(PsConstant(1))
                                )
                                / PsExpression.make(simd_step)
                                + PsExpression.make(PsConstant(1))
                            )
                            * PsExpression.make(simd_step)
                            + simd_start.clone(),
                            #   otherwise start at zero
                            simd_start.clone(),
                        ),
                    )
                )

                trailing_ctr = self._ctx.duplicate_symbol(scalar_ctr)
                trailing_loop_body = substitute_symbols(
                    loop.body.clone(), {scalar_ctr: PsExpression.make(trailing_ctr)}
                )
                trailing_loop = PsLoop(
                    PsExpression.make(trailing_ctr),
                    PsExpression.make(trailing_start),
                    loop.stop.clone(),
                    loop.step.clone(),
                    trailing_loop_body,
                )

                return PsBlock(
                    simd_init_local_reduction_vars
                    + [simd_stop_decl, simd_step_decl, simd_loop]
                    + simd_writeback_local_reduction_vars
                    + [
                        trailing_start_decl,
                        trailing_loop,
                    ]
                )

            case LoopVectorizer.TrailingItersTreatment.MASKED_BLOCK:
                raise NotImplementedError()

            case LoopVectorizer.TrailingItersTreatment.NONE:
                return PsBlock(
                    simd_init_local_reduction_vars
                    + [
                        simd_stop_decl,
                        simd_step_decl,
                        simd_loop,
                    ]
                    + simd_writeback_local_reduction_vars
                )
