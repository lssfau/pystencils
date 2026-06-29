from __future__ import annotations

import math
import operator
from functools import reduce
import numpy as np

from ..ast.structural import (
    PsStructuralNode,
    PsBlock,
    PsConditional,
    PsStatement,
    PsDeclaration,
    PsAssignment,
)
from ..memory import PsSymbol
from ..reduction_op_mapping import reduction_op_to_expr
from ...codegen.gpu_indexing import dim3, GpuIndexing
from ...sympyextensions import ReductionOp
from ...types import PsIntegerType, PsCustomType, PsPointerType, PsScalarType
from ...types.quick import UInt
from ..exceptions import MaterializationError
from .platform import Platform

from ..kernelcreation import (
    Typifier,
)

from ..constants import PsConstant
from ..kernelcreation.context import KernelCreationContext
from ..ast.expressions import (
    PsExpression,
    PsLiteralExpr,
    PsCast,
    PsCall,
    PsAdd,
    PsRem,
    PsEq,
    PsSymbolExpr,
    PsAnd,
    PsNe,
)
from ...types import PsSignedIntegerType, PsIeeeFloatType
from ..literals import PsLiteral

from ..functions import (
    MathFunctions,
    CFunction,
    PsMathFunction,
    PsConstantFunction,
    ConstantFunctions,
    PsRngEngineFunction,
    PsGpuIntrinsicFunction,
    GpuFpIntrinsics,
    PsGpuIndexingFunction,
    GpuGridScope,
)

int32 = PsSignedIntegerType(width=32, const=False)

BLOCK_IDX = [
    PsLiteralExpr(PsLiteral(f"blockIdx.{coord}", int32)) for coord in ("x", "y", "z")
]
THREAD_IDX = [
    PsLiteralExpr(PsLiteral(f"threadIdx.{coord}", int32)) for coord in ("x", "y", "z")
]
BLOCK_DIM = [
    PsLiteralExpr(PsLiteral(f"blockDim.{coord}", int32)) for coord in ("x", "y", "z")
]
GRID_DIM = [
    PsLiteralExpr(PsLiteral(f"gridDim.{coord}", int32)) for coord in ("x", "y", "z")
]

INDEXING_SCOPES: dict[GpuGridScope, list[PsLiteralExpr]] = {
    GpuGridScope.blockIdx: BLOCK_IDX,
    GpuGridScope.threadIdx: THREAD_IDX,
    GpuGridScope.blockDim: BLOCK_DIM,
    GpuGridScope.gridDim: GRID_DIM,
}


class GenericGpu(Platform):
    """Common base platform for CUDA- and HIP-type GPU targets.

    Args:
        ctx: The kernel creation context
        assume_warp_aligned_block_size: ``True`` if the platform can assume that total GPU block
            sizes at runtime will always be a multiple of the warp size
        warp_size: Size of a GPU warp
    """

    @property
    def required_headers(self) -> set[str]:
        headers = {'"pystencils_runtime/generic_gpu.hpp"'}

        if self._use_cub_reductions:
            headers |= {'"pystencils_runtime/bits/gpu_reductions.h"'}

        return headers

    def __init__(
        self,
        ctx: KernelCreationContext,
        indexing_rank: int,
        default_block_size: dim3 | None,
        *,
        assume_warp_aligned_block_size: bool = False,
        warp_size: int | None = None,
        use_cub_reductions: bool = False,
    ) -> None:
        super().__init__(ctx)

        self._indexing_rank = indexing_rank

        self._default_block_size = default_block_size

        self._assume_warp_aligned_block_size = assume_warp_aligned_block_size
        self._warp_size = warp_size

        self._use_cub_reductions = use_cub_reductions

        self._typify = Typifier(ctx)

    @property
    def default_block_size(self) -> dim3 | None:
        return self._default_block_size

    @staticmethod
    def gen_warp_reduce(
        symbol_expr: PsSymbolExpr, op: ReductionOp, warp_size: int, mask_size: int = 8
    ) -> tuple[PsExpression, list[PsStructuralNode]]:
        """Set up shuffle instructions for warp-level reduction"""

        # perform local warp reduction for given offset
        def gen_shuffle_instr(offset: int):
            stype = symbol_expr.dtype
            assert isinstance(stype, PsScalarType)

            return PsCall(
                CFunction("__shfl_xor_sync", [UInt(32), stype, UInt(32)], stype),
                [
                    PsExpression.make(PsLiteral(f"0x{'f' * mask_size}", UInt(32))),
                    symbol_expr,
                    PsExpression.make(PsConstant(offset, UInt(32))),
                ],
            )

        target_lane_id = PsExpression.make(PsConstant(0, UInt(32)))

        # get all shuffle statements for a given warp size
        num_shuffles = math.frexp(warp_size)[1]
        return (
            target_lane_id,
            list(
                PsAssignment(
                    symbol_expr,
                    reduction_op_to_expr(
                        op,
                        symbol_expr,
                        gen_shuffle_instr(pow(2, i - 1)),
                    ),
                )
                for i in reversed(range(1, num_shuffles))
            ),
        )

    @staticmethod
    def _local_thread_index_per_dim(num_dims: int) -> tuple[PsExpression, ...]:
        """Returns thread indices multiplied with block dimension strides per dimension."""

        return tuple(
            idx * reduce(operator.mul, BLOCK_DIM[:i]) if i > 0 else idx
            for i, idx in enumerate(THREAD_IDX[:num_dims])
        )

    @staticmethod
    def _local_thread_id(num_dims: int) -> PsExpression:
        """Returns sum of all local thread indices."""

        tids_per_dim = GenericGpu._local_thread_index_per_dim(num_dims)
        tid: PsExpression = tids_per_dim[0]
        for t in tids_per_dim[1:]:
            tid = PsAdd(tid, t)

        return tid

    def select_function(self, call: PsCall) -> PsExpression:
        call_func = call.function

        dtype = call.get_dtype()
        expr: PsExpression | None = None

        if isinstance(call_func, PsGpuIndexingFunction):
            return self._typify(
                PsCast(
                    call.get_dtype(),
                    INDEXING_SCOPES[call_func.scope][call_func.dimension],
                )
            )

        elif isinstance(
            call_func, PsMathFunction | PsConstantFunction | PsGpuIntrinsicFunction
        ):
            func = call_func.func
            arg_types = (dtype,) * call.function.arg_count

            if isinstance(dtype, PsIeeeFloatType):
                match func:
                    case (
                        MathFunctions.Exp
                        | MathFunctions.Log
                        | MathFunctions.Sin
                        | MathFunctions.Cos
                        | MathFunctions.Sqrt
                        | MathFunctions.Ceil
                        | MathFunctions.Floor
                    ) if dtype.width in (16, 32, 64):
                        prefix = "h" if dtype.width == 16 else ""
                        suffix = "f" if dtype.width == 32 else ""
                        name = f"{prefix}{func.function_name}{suffix}"
                        call.function = CFunction(name, arg_types, dtype)
                        expr = call

                    case (
                        MathFunctions.Pow
                        | MathFunctions.Tan
                        | MathFunctions.Sinh
                        | MathFunctions.Cosh
                        | MathFunctions.Tanh
                        | MathFunctions.ASin
                        | MathFunctions.ACos
                        | MathFunctions.ATan
                        | MathFunctions.ATan2
                    ) if dtype.width in (32, 64):
                        #   These are unavailable for fp16
                        suffix = "f" if dtype.width == 32 else ""
                        name = f"{func.function_name}{suffix}"
                        call.function = CFunction(name, arg_types, dtype)
                        expr = call

                    case (
                        MathFunctions.Min | MathFunctions.Max | MathFunctions.Abs
                    ) if dtype.width in (32, 64):
                        suffix = "f" if dtype.width == 32 else ""
                        name = f"f{func.function_name}{suffix}"
                        call.function = CFunction(name, arg_types, dtype)
                        expr = call

                    case MathFunctions.Abs if dtype.width == 16:
                        call.function = CFunction(" __habs", arg_types, dtype)
                        expr = call

                    case ConstantFunctions.Pi:
                        assert dtype.numpy_dtype is not None
                        expr = PsExpression.make(
                            PsConstant(dtype.numpy_dtype.type(np.pi), dtype)
                        )

                    case ConstantFunctions.E:
                        assert dtype.numpy_dtype is not None
                        expr = PsExpression.make(
                            PsConstant(dtype.numpy_dtype.type(np.e), dtype)
                        )

                    case ConstantFunctions.PosInfinity:
                        expr = PsExpression.make(
                            PsLiteral(f"PS_FP{dtype.width}_INFINITY", dtype)
                        )

                    case ConstantFunctions.NegInfinity:
                        expr = PsExpression.make(
                            PsLiteral(f"PS_FP{dtype.width}_NEG_INFINITY", dtype)
                        )

                    case GpuFpIntrinsics() if dtype.width == 32:
                        name = f"__f{func.function_name}"
                        call.function = CFunction(name, arg_types, dtype)
                        expr = call

                    case _:
                        raise MaterializationError(
                            f"Cannot materialize call to function {func}"
                        )

            if isinstance(dtype, PsIntegerType):
                expr = self._select_integer_function(call)

        elif isinstance(call.function, PsRngEngineFunction):
            spec = call.function.rng_spec
            ctr_type = spec.int_arg_type
            atypes = (spec.int_arg_type,) * call.function.arg_count

            rng_func = CFunction(
                f"pystencils::runtime::random::{spec.rng_name}< {ctr_type.c_string()} >",
                atypes,
                spec.short_array_type,
            )

            expr = rng_func(*call.args)

        if expr is not None:
            if expr.dtype is None:
                typify = Typifier(self._ctx)
                typify(expr)

            return expr

        raise MaterializationError(
            f"No implementation available for function {call_func} on data type {dtype}"
        )

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
        actual_reduction_op: ReductionOp
        match reduction_op:
            case ReductionOp.Sub:
                actual_reduction_op = ReductionOp.Add
            case _:
                actual_reduction_op = reduction_op

        # determine atomic function from reduction operation and setup call
        impl_namespace: str
        match actual_reduction_op:
            case ReductionOp.Min | ReductionOp.Max | ReductionOp.Mul:
                impl_namespace = "pystencils::runtime::gpu::"
            case _:
                impl_namespace = ""

        func = CFunction(
            f"{impl_namespace}atomic{actual_reduction_op.name}",
            [ptrtype, stype],
            PsCustomType("void"),
        )
        func_args = (ptr_expr, symbol_expr)

        # get neutral element for reduction op
        init_val_expr = self._typify(
            self._ctx.reduction_data[ptr_expr.symbol.name].init_val.clone()
        )
        neutral_elem_expr: PsExpression
        if isinstance(init_val_expr, PsCall):
            neutral_elem_expr = self.select_function(init_val_expr)
        else:
            neutral_elem_expr = init_val_expr

        # check if thread is valid for performing reduction
        ispace = self._ctx.get_iteration_space()
        effective_rank = GpuIndexing.get_effective_rank(
            self._indexing_rank, ispace.rank
        )

        def is_valid_thread(accum: PsExpression):
            return PsNe(accum, neutral_elem_expr)

        cond: PsExpression
        reduction_stmts: list[PsStructuralNode]
        if self._use_cub_reductions:
            if self._default_block_size is None:
                raise MaterializationError(
                    "CUB reductions require the GPU default block size to be set."
                )

            cub_operator = f"pystencils::runtime::gpu::{actual_reduction_op.name}"
            template_params = f"{stype.c_string()}, {cub_operator}"
            for bs in self._default_block_size:
                template_params += f", {bs}"

            zero_expr = PsExpression.make(PsConstant(0, self._ctx.index_dtype))
            first_thread_in_block = reduce(
                PsAnd, (PsEq(idx, zero_expr) for idx in THREAD_IDX[:effective_rank])  # type: ignore
            )

            block_accum = PsSymbolExpr(PsSymbol("block_accum", stype))

            return PsBlock(
                [
                    # perform block reduce
                    PsDeclaration(
                        block_accum,
                        PsCall(
                            CFunction(
                                f"pystencils::runtime::gpu::cub::block_reduce < {template_params} >",
                                [stype],
                                stype,
                            ),
                            [symbol_expr],
                        ),
                    ),
                    # last step: atomic operation on first thread
                    PsConditional(
                        PsAnd(is_valid_thread(block_accum), first_thread_in_block),
                        PsBlock([PsStatement(PsCall(func, (ptr_expr, block_accum)))]),
                    ),
                ]
            )
        elif self._warp_size and self._assume_warp_aligned_block_size:
            warp_size_expr = PsExpression.make(
                PsConstant(self._warp_size, self._ctx.index_dtype)
            )

            # perform warp-level reduction on local symbol
            target_lane_id, warp_reduce_stmts = self.gen_warp_reduce(
                symbol_expr,
                actual_reduction_op,
                self._warp_size,
            )

            # declare symbols for thread and warp lane ids
            thread_id = PsSymbolExpr(PsSymbol("thread_id", UInt(32)))
            lane_id = PsSymbolExpr(PsSymbol("lane_id", UInt(32)))

            symbol_decls: list[PsStructuralNode] = [
                PsDeclaration(thread_id, self._local_thread_id(effective_rank)),
                PsDeclaration(lane_id, PsRem(thread_id, warp_size_expr)),
            ]

            reduction_stmts = symbol_decls + warp_reduce_stmts

            # set condition to only execute atomic operation on first valid thread in warp
            cond = PsAnd(is_valid_thread(symbol_expr), PsEq(lane_id, target_lane_id))
        else:
            # no optimization: only execute atomic add on valid thread
            reduction_stmts = []
            cond = is_valid_thread(symbol_expr)

        # assemble final reduction
        return PsBlock(
            reduction_stmts
            + [PsConditional(cond, PsBlock([PsStatement(PsCall(func, func_args))]))]
        )
