from __future__ import annotations

import math

from .generic_gpu import GenericGpu
from ..ast.expressions import (
    PsExpression,
    PsLiteralExpr,
    PsCall,
    PsAnd,
    PsConstantExpr,
    PsSymbolExpr,
    PsEq,
    PsNot
)
from ..ast.structural import (
    PsConditional,
    PsStatement,
    PsAssignment,
    PsBlock,
    PsStructuralNode,
)
from ..constants import PsConstant
from ..exceptions import MaterializationError
from ..functions import CFunction
from ..literals import PsLiteral
from ..reduction_op_mapping import reduction_op_to_expr
from ...sympyextensions import ReductionOp
from ...types import PsIeeeFloatType, PsCustomType, PsPointerType, PsScalarType
from ...types.quick import SInt, UInt


class CudaPlatform(GenericGpu):
    """Platform for the CUDA GPU target."""

    @property
    def required_headers(self) -> set[str]:
        return super().required_headers | {'"pystencils_runtime/cuda.cuh"'}

    def resolve_reduction(
        self,
        ptr_expr: PsExpression,
        symbol_expr: PsExpression,
        reduction_op: ReductionOp,
    ) -> PsStructuralNode:
        stype = symbol_expr.dtype
        ptrtype = ptr_expr.dtype

        assert isinstance(ptr_expr, PsSymbolExpr) and isinstance(ptrtype, PsPointerType)
        assert isinstance(symbol_expr, PsSymbolExpr) and isinstance(stype, PsScalarType)

        if not isinstance(stype, PsIeeeFloatType) or stype.width not in (32, 64):
            raise MaterializationError(
                "Cannot materialize reduction on GPU: "
                "Atomics are only available for float32/64 datatypes"
            )

        # workaround for subtractions -> use additions for reducing intermediate results
        # similar to OpenMP reductions: local copies (negative sign) are added at the end
        match reduction_op:
            case ReductionOp.Sub:
                actual_reduction_op = ReductionOp.Add
            case _:
                actual_reduction_op = reduction_op

        # get neutral element for reduction op
        init_val_expr = self._typify(self._ctx.reduction_data[ptr_expr.symbol.name].init_val.clone())
        neutral_elem_expr: PsExpression
        if isinstance(init_val_expr, PsCall):
            neutral_elem_expr = self.select_function(init_val_expr)
        else:
            neutral_elem_expr = init_val_expr

        # check if thread is valid for performing reduction
        ispace = self._ctx.get_iteration_space()
        is_valid_thread = PsNot(PsEq(symbol_expr, neutral_elem_expr))

        cond: PsExpression
        shuffles: tuple[PsAssignment, ...]
        if self._warp_size and self._assume_warp_aligned_block_size:
            # perform local warp reductions
            def gen_shuffle_instr(offset: int):
                full_mask = PsLiteralExpr(PsLiteral("0xffffffff", UInt(32)))
                return PsCall(
                    CFunction("__shfl_xor_sync", [UInt(32), stype, SInt(32)], stype),
                    [
                        full_mask,
                        symbol_expr,
                        PsConstantExpr(PsConstant(offset, SInt(32))),
                    ],
                )

            # set up shuffle instructions for warp-level reduction
            num_shuffles = math.frexp(self._warp_size)[1]
            shuffles = tuple(
                PsAssignment(
                    symbol_expr,
                    reduction_op_to_expr(
                        actual_reduction_op,
                        symbol_expr,
                        gen_shuffle_instr(pow(2, i - 1)),
                    ),
                )
                for i in reversed(range(1, num_shuffles))
            )

            # find first thread in warp
            first_thread_in_warp = self._first_thread_in_warp(ispace)

            # set condition to only execute atomic operation on first valid thread in warp
            cond = (
                PsAnd(is_valid_thread, first_thread_in_warp)
                if is_valid_thread
                else first_thread_in_warp
            )
        else:
            # no optimization: only execute atomic add on valid thread
            shuffles = ()
            cond = is_valid_thread

        # use atomic operation
        match actual_reduction_op:
            case ReductionOp.Min | ReductionOp.Max | ReductionOp.Mul:
                impl_namespace = "pystencils::runtime::gpu::"
            case _:
                impl_namespace = ""

        func = CFunction(
            f"{impl_namespace}atomic{actual_reduction_op.name}", [ptrtype, stype], PsCustomType("void")
        )
        func_args = (ptr_expr, symbol_expr)

        # assemble warp reduction
        return PsBlock(
            list(shuffles) + [
                PsConditional(
                    cond, PsBlock([PsStatement(PsCall(func, func_args))])
                )
            ]
        )
