from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from functools import reduce
import numpy as np

from ...types import (
    constify,
    deconstify,
    PsIntegerType,
)
from ...types.quick import SInt
from ..exceptions import MaterializationError
from .platform import Platform

from ..memory import PsSymbol
from ..kernelcreation import (
    Typifier,
    IterationSpace,
    FullIterationSpace,
    SparseIterationSpace,
    AstFactory,
)

from ..constants import PsConstant
from ..kernelcreation.context import KernelCreationContext
from ..ast.structural import (
    PsBlock,
    PsConditional,
    PsDeclaration,
)
from ..ast.expressions import (
    PsExpression,
    PsLiteralExpr,
    PsCast,
    PsCall,
    PsLookup,
    PsBufferAcc,
    PsConstantExpr,
    PsAdd,
    PsRem,
    PsEq,
)
from ..ast.expressions import PsLt, PsAnd
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


class ThreadMapping(ABC):

    @abstractmethod
    def __call__(self, ispace: IterationSpace) -> dict[PsSymbol, PsExpression]:
        """Map the current thread index onto a point in the given iteration space.

        Implementations of this method must return a declaration for each dimension counter
        of the given iteration space.
        """


class Linear3DMapping(ThreadMapping):
    """3D globally linearized mapping, where each thread is assigned a work item according to
    its location in the global launch grid."""

    def __call__(self, ispace: IterationSpace) -> dict[PsSymbol, PsExpression]:
        match ispace:
            case FullIterationSpace():
                return self._dense_mapping(ispace)
            case SparseIterationSpace():
                return self._sparse_mapping(ispace)
            case _:
                assert False, "unexpected iteration space"

    def _dense_mapping(
        self, ispace: FullIterationSpace
    ) -> dict[PsSymbol, PsExpression]:
        if ispace.rank > 3:
            raise MaterializationError(
                f"Cannot handle {ispace.rank}-dimensional iteration space "
                "using the Linear3D GPU thread index mapping."
            )

        dimensions = ispace.dimensions_in_loop_order()
        idx_map: dict[PsSymbol, PsExpression] = dict()

        for coord, dim in enumerate(dimensions[::-1]):
            tid = self._linear_thread_idx(coord)
            idx_map[dim.counter] = dim.start + dim.step * PsCast(
                deconstify(dim.counter.get_dtype()), tid
            )

        return idx_map

    def _sparse_mapping(
        self, ispace: SparseIterationSpace
    ) -> dict[PsSymbol, PsExpression]:
        sparse_ctr = PsExpression.make(ispace.sparse_counter)
        thread_idx = self._linear_thread_idx(0)
        idx_map: dict[PsSymbol, PsExpression] = {
            ispace.sparse_counter: PsCast(
                deconstify(sparse_ctr.get_dtype()), thread_idx
            )
        }
        return idx_map

    def _linear_thread_idx(self, coord: int):
        block_size = BLOCK_DIM[coord]
        block_idx = BLOCK_IDX[coord]
        thread_idx = THREAD_IDX[coord]
        return block_idx * block_size + thread_idx


class Blockwise4DMapping(ThreadMapping):
    """Blockwise index mapping for up to 4D iteration spaces, where the outer three dimensions
    are mapped to block indices."""

    _indices_fastest_first = [  # slowest to fastest
        THREAD_IDX[0],
        BLOCK_IDX[0],
        BLOCK_IDX[1],
        BLOCK_IDX[2],
    ]

    def __call__(self, ispace: IterationSpace) -> dict[PsSymbol, PsExpression]:
        match ispace:
            case FullIterationSpace():
                return self._dense_mapping(ispace)
            case SparseIterationSpace():
                return self._sparse_mapping(ispace)
            case _:
                assert False, "unexpected iteration space"

    def _dense_mapping(
        self, ispace: FullIterationSpace
    ) -> dict[PsSymbol, PsExpression]:
        if ispace.rank > 4:
            raise MaterializationError(
                f"Cannot handle {ispace.rank}-dimensional iteration space "
                "using the Blockwise4D GPU thread index mapping."
            )

        dimensions = ispace.dimensions_in_loop_order()
        idx_map: dict[PsSymbol, PsExpression] = dict()

        for dim, tid in zip(dimensions[::-1], self._indices_fastest_first):
            idx_map[dim.counter] = dim.start + dim.step * PsCast(
                deconstify(dim.counter.get_dtype()), tid
            )

        return idx_map

    def _sparse_mapping(
        self, ispace: SparseIterationSpace
    ) -> dict[PsSymbol, PsExpression]:
        sparse_ctr = PsExpression.make(ispace.sparse_counter)
        thread_idx = self._indices_fastest_first[0]
        idx_map: dict[PsSymbol, PsExpression] = {
            ispace.sparse_counter: PsCast(
                deconstify(sparse_ctr.get_dtype()), thread_idx
            )
        }
        return idx_map


class GenericGpu(Platform):
    """Common base platform for CUDA- and HIP-type GPU targets.

    Args:
        ctx: The kernel creation context
        omit_range_check: If `True`, generated index translation code will not check if the point identified
            by block and thread indices is actually contained in the iteration space
        thread_mapping: Callback object which defines the mapping of thread indices onto iteration space points
    """

    @property
    def required_headers(self) -> set[str]:
        return {'"pystencils_runtime/generic_gpu.hpp"'}

    def __init__(
        self,
        ctx: KernelCreationContext,
        assume_warp_aligned_block_size: bool,
        warp_size: int | None,
        thread_mapping: ThreadMapping | None = None,
    ) -> None:
        super().__init__(ctx)

        self._assume_warp_aligned_block_size = assume_warp_aligned_block_size
        self._warp_size = warp_size

        self._thread_mapping = (
            thread_mapping if thread_mapping is not None else Linear3DMapping()
        )

        self._typify = Typifier(ctx)

    def materialize_iteration_space(
        self, body: PsBlock, ispace: IterationSpace
    ) -> PsBlock:
        if isinstance(ispace, FullIterationSpace):
            return self._prepend_dense_translation(body, ispace)
        elif isinstance(ispace, SparseIterationSpace):
            return self._prepend_sparse_translation(body, ispace)
        else:
            raise MaterializationError(f"Unknown type of iteration space: {ispace}")

    @staticmethod
    def _get_condition_for_translation(ispace: IterationSpace) -> PsExpression:
        """Return the condition that determines whether the current thread
        lies inside the iteration space."""

        if isinstance(ispace, FullIterationSpace):
            conds = []

            dimensions = ispace.dimensions_in_loop_order()

            for dim in dimensions:
                ctr_expr = PsExpression.make(dim.counter)
                conds.append(PsLt(ctr_expr, dim.stop))

            condition: PsExpression = conds[0]
            for cond in conds[1:]:
                condition = PsAnd(condition, cond)

            return condition
        elif isinstance(ispace, SparseIterationSpace):
            sparse_ctr_expr = PsExpression.make(ispace.sparse_counter)
            stop = PsExpression.make(ispace.index_list.shape[0])

            return PsLt(sparse_ctr_expr.clone(), stop)
        else:
            raise MaterializationError(f"Unknown type of iteration space: {ispace}")

    @staticmethod
    def _block_local_thread_index_per_dim(ispace: IterationSpace) -> tuple[PsExpression, ...]:
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

    def select_function(
        self, call: PsCall
    ) -> PsExpression:
        call_func = call.function

        dtype = call.get_dtype()
        arg_types = (dtype,) * call.function.arg_count
        expr: PsExpression | None = None

        if isinstance(call_func, PsMathFunction | PsConstantFunction):
            func = call_func.func
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
                        expr = PsExpression.make(PsLiteral(f"PS_FP{dtype.width}_INFINITY", dtype))

                    case ConstantFunctions.NegInfinity:
                        expr = PsExpression.make(PsLiteral(f"PS_FP{dtype.width}_NEG_INFINITY", dtype))

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
            f"No implementation available for function {func} on data type {dtype}"
        )

    #   Internals

    def _prepend_dense_translation(
        self, body: PsBlock, ispace: FullIterationSpace
    ) -> PsBlock:
        ctr_mapping = self._thread_mapping(ispace)

        indexing_decls = []

        dimensions = ispace.dimensions_in_loop_order()

        for dim in dimensions:
            # counter declarations must be ordered slowest-to-fastest
            # such that inner dimensions can depend on outer ones

            dim.counter.dtype = constify(dim.counter.get_dtype())

            ctr_expr = PsExpression.make(dim.counter)
            indexing_decls.append(
                self._typify(PsDeclaration(ctr_expr, ctr_mapping[dim.counter]))
            )

        cond = self._get_condition_for_translation(ispace)
        return PsBlock(indexing_decls + [PsConditional(cond, body)])

    def _prepend_sparse_translation(
        self, body: PsBlock, ispace: SparseIterationSpace
    ) -> PsBlock:
        factory = AstFactory(self._ctx)
        ispace.sparse_counter.dtype = constify(ispace.sparse_counter.get_dtype())

        sparse_ctr_expr = PsExpression.make(ispace.sparse_counter)
        ctr_mapping = self._thread_mapping(ispace)

        sparse_idx_decl = self._typify(
            PsDeclaration(sparse_ctr_expr, ctr_mapping[ispace.sparse_counter])
        )

        mappings = [
            PsDeclaration(
                PsExpression.make(ctr),
                PsLookup(
                    PsBufferAcc(
                        ispace.index_list.base_pointer,
                        (sparse_ctr_expr.clone(), factory.parse_index(0)),
                    ),
                    coord.name,
                ),
            )
            for ctr, coord in zip(ispace.spatial_indices, ispace.coordinate_members)
        ]
        body.statements = mappings + body.statements

        cond = self._get_condition_for_translation(ispace)
        return PsBlock([sparse_idx_decl, PsConditional(cond, body)])
