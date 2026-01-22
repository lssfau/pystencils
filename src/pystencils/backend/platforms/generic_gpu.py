from __future__ import annotations

import operator
from functools import reduce
import numpy as np

from ...types import PsIntegerType
from ...types.quick import SInt
from ..exceptions import MaterializationError
from .platform import Platform

from ..kernelcreation import (
    Typifier,
    IterationSpace,
)

from ..constants import PsConstant
from ..kernelcreation.context import KernelCreationContext
from ..ast.expressions import (
    PsExpression,
    PsLiteralExpr,
    PsCast,
    PsCall,
    PsConstantExpr,
    PsAdd,
    PsRem,
    PsEq,
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
    RngSpec,
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
        return {'"pystencils_runtime/generic_gpu.hpp"'}

    def __init__(
        self,
        ctx: KernelCreationContext,
        *,
        assume_warp_aligned_block_size: bool = False,
        warp_size: int | None = None,
    ) -> None:
        super().__init__(ctx)

        self._assume_warp_aligned_block_size = assume_warp_aligned_block_size
        self._warp_size = warp_size

        self._typify = Typifier(ctx)

    @staticmethod
    def _block_local_thread_index_per_dim(
        ispace: IterationSpace,
    ) -> tuple[PsExpression, ...]:
        """Returns thread indices multiplied with block dimension strides per dimension."""

        return tuple(
            idx * reduce(operator.mul, BLOCK_DIM[:i]) if i > 0 else idx
            for i, idx in enumerate(THREAD_IDX[: ispace.rank])
        )

    def _first_thread_in_warp(self, ispace: IterationSpace) -> PsExpression:
        """Returns expression that determines whether a thread is the first within a warp."""

        tids_per_dim = GenericGpu._block_local_thread_index_per_dim(ispace)
        tid: PsExpression = tids_per_dim[0]
        for t in tids_per_dim[1:]:
            tid = PsAdd(tid, t)

        return PsEq(
            PsRem(tid, PsConstantExpr(PsConstant(self._warp_size, SInt(32)))),
            PsConstantExpr(PsConstant(0, SInt(32))),
        )

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
            atypes = (spec.int_arg_type,) * call.function.arg_count

            match spec:
                case RngSpec.PhiloxFp32:
                    rng_func = CFunction(
                        "pystencils::runtime::random::philox_fp32x4", atypes, spec.dtype
                    )
                case RngSpec.PhiloxFp64:
                    rng_func = CFunction(
                        "pystencils::runtime::random::philox_fp64x2", atypes, spec.dtype
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
