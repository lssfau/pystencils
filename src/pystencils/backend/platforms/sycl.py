from __future__ import annotations

import numpy as np

from ...codegen.properties import SYCLItem, SYCLNDItem
from ...types import (
    PsCustomType,
    PsIeeeFloatType,
    PsIntegerType,
    PsUnsignedIntegerType,
    constify,
)
from ..ast.expressions import (
    PsAnd,
    PsBufferAcc,
    PsCall,
    PsCast,
    PsExpression,
    PsLookup,
    PsLt,
    PsSubscript,
    PsSymbolExpr,
)
from ..ast.structural import PsBlock, PsConditional, PsDeclaration
from ..constants import PsConstant
from ..exceptions import MaterializationError
from ..extensions.cpp import CppMethodCall
from ..functions import (
    CFunction,
    ConstantFunctions,
    MathFunctions,
    PsConstantFunction,
    PsIrFunction,
    PsMathFunction,
    PsRngEngineFunction,
)
from ..kernelcreation import AstFactory, KernelCreationContext, Typifier
from ..kernelcreation.iteration_space import (
    FullIterationSpace,
    IterationSpace,
    SparseIterationSpace,
)
from .platform import Platform


class SyclPlatform(Platform):

    def __init__(
        self,
        ctx: KernelCreationContext,
        automatic_block_size: bool = False,
    ):
        super().__init__(ctx)

        self._automatic_block_size = automatic_block_size

    @property
    def required_headers(self) -> set[str]:
        return {"<sycl/sycl.hpp>", '"pystencils_runtime/generic_cpu.hpp"'}

    def materialize_iteration_space(
        self, body: PsBlock, ispace: IterationSpace
    ) -> PsBlock:
        if isinstance(ispace, FullIterationSpace):
            return self._prepend_dense_translation(body, ispace)
        elif isinstance(ispace, SparseIterationSpace):
            return self._prepend_sparse_translation(body, ispace)
        else:
            raise MaterializationError(f"Unknown type of iteration space: {ispace}")

    def select_function(self, call: PsCall) -> PsExpression:
        assert isinstance(call.function, PsIrFunction)

        dtype = call.get_dtype()

        expr: PsExpression | None = None
        if isinstance(call.function, PsMathFunction | PsConstantFunction):
            func = call.function.func
            arg_types = (dtype,) * call.function.arg_count

            if isinstance(dtype, PsIeeeFloatType) and dtype.width in (16, 32, 64):
                match func:
                    case (
                        MathFunctions.Exp
                        | MathFunctions.Log
                        | MathFunctions.Sin
                        | MathFunctions.Cos
                        | MathFunctions.Tan
                        | MathFunctions.Sinh
                        | MathFunctions.Cosh
                        | MathFunctions.Tanh
                        | MathFunctions.ASin
                        | MathFunctions.ACos
                        | MathFunctions.ATan
                        | MathFunctions.ATan2
                        | MathFunctions.Pow
                        | MathFunctions.Sqrt
                        | MathFunctions.Floor
                        | MathFunctions.Ceil
                    ):
                        call.function = CFunction(
                            f"sycl::{func.function_name}", arg_types, dtype
                        )
                        expr = call

                    case MathFunctions.Abs | MathFunctions.Min | MathFunctions.Max:
                        call.function = CFunction(
                            f"sycl::f{func.function_name}", arg_types, dtype
                        )
                        expr = call

                match func:
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

                    case ConstantFunctions.PosInfinity | ConstantFunctions.NegInfinity:
                        call.function = CFunction(
                            f"std::numeric_limits< {dtype.c_string()} >::infinity",
                            [],
                            dtype,
                        )
                        if func == ConstantFunctions.NegInfinity:
                            expr = -call
                        else:
                            expr = call

            elif isinstance(dtype, PsIntegerType):
                expr = self._select_integer_function(call)

        elif isinstance(call.function, PsRngEngineFunction):
            spec = call.function.rng_spec
            atypes = (spec.int_arg_type,) * call.function.arg_count
            ctr_type = spec.int_arg_type

            rng_func = CFunction(
                f"pystencils::runtime::random::{spec.rng_name}< {ctr_type.c_string()} >",
                atypes,
                spec.short_array_type,
            )

            expr = rng_func(*call.args)

        if expr is not None:
            if expr.dtype is None:
                typify = Typifier(self._ctx)
                expr = typify(expr)
            return expr
        else:
            raise MaterializationError(
                f"No implementation available for function {func} on data type {dtype}"
            )

    def _prepend_dense_translation(
        self, body: PsBlock, ispace: FullIterationSpace
    ) -> PsBlock:
        rank = ispace.rank
        id_type = self._id_type(rank)
        id_symbol = PsExpression.make(self._ctx.get_symbol("id", id_type))
        id_decl = self._id_declaration(rank, id_symbol)

        dimensions = ispace.dimensions_in_loop_order()
        indexing_decls = [id_decl]
        conds = []

        #   Other than in CUDA, SYCL ids are linearized in C order
        #   The leftmost entry of an ID varies slowest, and the rightmost entry varies fastest
        #   See https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:multi-dim-linearization

        for i, dim in enumerate(dimensions):
            #   Slowest to fastest
            coord = PsExpression.make(PsConstant(i, self._ctx.index_dtype))
            work_item_idx = PsSubscript(id_symbol, (coord,))

            dim.counter.dtype = constify(dim.counter.get_dtype())
            # work_item_idx.dtype = dim.counter.get_dtype()
            work_item_idx.dtype = PsUnsignedIntegerType(64)

            ctr = PsExpression.make(dim.counter)
            indexing_decls.append(
                PsDeclaration(
                    ctr,
                    dim.start + PsCast(self._ctx.index_dtype, work_item_idx) * dim.step,
                )
            )
            conds.append(PsLt(ctr, dim.stop))

        if conds:
            condition: PsExpression = conds[0]
            for cond in conds[1:]:
                condition = PsAnd(condition, cond)
            ast = PsBlock(indexing_decls + [PsConditional(condition, body)])
        else:
            body.statements = indexing_decls + body.statements
            ast = body

        return ast

    def _prepend_sparse_translation(
        self, body: PsBlock, ispace: SparseIterationSpace
    ) -> PsBlock:
        factory = AstFactory(self._ctx)

        id_type = PsCustomType("sycl::id< 1 >", const=True)
        id_symbol = PsExpression.make(self._ctx.get_symbol("id", id_type))

        zero = PsExpression.make(PsConstant(0, self._ctx.index_dtype))
        subscript = PsSubscript(id_symbol, (zero,))

        ispace.sparse_counter.dtype = constify(ispace.sparse_counter.get_dtype())
        subscript.dtype = ispace.sparse_counter.get_dtype()

        sparse_ctr = PsExpression.make(ispace.sparse_counter)
        sparse_idx_decl = PsDeclaration(sparse_ctr, subscript)

        mappings = [
            PsDeclaration(
                PsExpression.make(ctr),
                PsLookup(
                    PsBufferAcc(
                        ispace.index_list.base_pointer,
                        (sparse_ctr, factory.parse_index(0)),
                    ),
                    coord.name,
                ),
            )
            for ctr, coord in zip(ispace.spatial_indices, ispace.coordinate_members)
        ]
        body.statements = mappings + body.statements

        stop = PsExpression.make(ispace.index_list.shape[0])
        condition = PsLt(sparse_ctr, stop)
        return PsBlock([sparse_idx_decl, PsConditional(condition, body)])

    def _item_type(self, rank: int):
        if not self._automatic_block_size:
            return PsCustomType(f"sycl::nd_item< {rank} >", const=True)
        else:
            return PsCustomType(f"sycl::item< {rank} >", const=True)

    def _id_type(self, rank: int):
        return PsCustomType(f"sycl::id< {rank} >", const=True)

    def _id_declaration(self, rank: int, id: PsSymbolExpr) -> PsDeclaration:
        item_type = self._item_type(rank)

        item_symbol = self._ctx.get_symbol("sycl_item", item_type)
        if self._automatic_block_size:
            item_symbol.add_property(SYCLItem(rank))
        else:
            item_symbol.add_property(SYCLNDItem(rank))
        item = PsExpression.make(item_symbol)

        if not self._automatic_block_size:
            rhs = CppMethodCall(item, "get_global_id", self._id_type(rank))
        else:
            rhs = CppMethodCall(item, "get_id", self._id_type(rank))

        return PsDeclaration(id, rhs)
