from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

from pystencils.backend.ast.expressions import PsCall

from ..functions import (
    CFunction,
    PsMathFunction,
    MathFunctions,
    PsConstantFunction,
    ConstantFunctions,
)
from ...types import PsIntegerType, PsIeeeFloatType

from .platform import Platform
from ..exceptions import MaterializationError

from ..kernelcreation import AstFactory
from ..kernelcreation.iteration_space import (
    IterationSpace,
    FullIterationSpace,
    SparseIterationSpace,
)

from ..constants import PsConstant
from ..ast.structural import PsDeclaration, PsLoop, PsBlock
from ..ast.expressions import (
    PsSymbolExpr,
    PsExpression,
    PsBufferAcc,
    PsLookup,
)
from ..ast.vector import PsVecMemAcc
from ..kernelcreation import Typifier
from ...types import PsVectorType, PsCustomType


class GenericCpu(Platform):
    """Generic CPU platform.

    The `GenericCPU` platform models the following execution environment:

     - Generic multicore CPU architecture
     - Iteration space represented by a loop nest, kernels are executed as a whole
     - C standard library math functions available (``#include <math.h>`` or ``#include <cmath>``)
    """

    @property
    def required_headers(self) -> set[str]:
        return {"<cmath>", "<limits>"}

    def materialize_iteration_space(
        self, body: PsBlock, ispace: IterationSpace
    ) -> PsBlock:
        if isinstance(ispace, FullIterationSpace):
            return self._create_domain_loops(body, ispace)
        elif isinstance(ispace, SparseIterationSpace):
            return self._create_sparse_loop(body, ispace)
        else:
            raise MaterializationError(f"Unknown type of iteration space: {ispace}")

    def select_function(self, call: PsCall) -> PsExpression:
        assert isinstance(call.function, (PsMathFunction | PsConstantFunction))

        func = call.function.func
        dtype = call.get_dtype()
        arg_types = (dtype,) * call.function.arg_count

        expr: PsExpression | None = None

        if isinstance(dtype, PsIeeeFloatType):
            if dtype.width in (32, 64):
                match func:
                    case (
                        MathFunctions.Exp
                        | MathFunctions.Log
                        | MathFunctions.Sin
                        | MathFunctions.Cos
                        | MathFunctions.Tan
                        | MathFunctions.Sinh
                        | MathFunctions.Cosh
                        | MathFunctions.ASin
                        | MathFunctions.ACos
                        | MathFunctions.ATan
                        | MathFunctions.ATan2
                        | MathFunctions.Pow
                        | MathFunctions.Sqrt
                        | MathFunctions.Floor
                        | MathFunctions.Ceil
                    ):
                        call.function = CFunction(func.function_name, arg_types, dtype)
                        expr = call
                    case MathFunctions.Abs | MathFunctions.Min | MathFunctions.Max:
                        call.function = CFunction(
                            "f" + func.function_name, arg_types, dtype
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

        if expr is not None:
            if expr.dtype is None:
                typify = Typifier(self._ctx)
                expr = typify(expr)
            return expr
        else:
            raise MaterializationError(
                f"No implementation available for function {func} on data type {dtype}"
            )

    #   Internals

    def _create_domain_loops(
        self, body: PsBlock, ispace: FullIterationSpace
    ) -> PsBlock:
        factory = AstFactory(self._ctx)

        #   Determine loop order by permuting dimensions
        archetype_field = ispace.archetype_field
        if archetype_field is not None:
            loop_order = archetype_field.layout
        else:
            loop_order = None

        loops = factory.loops_from_ispace(ispace, body, loop_order)
        return PsBlock([loops])

    def _create_sparse_loop(self, body: PsBlock, ispace: SparseIterationSpace):
        factory = AstFactory(self._ctx)

        mappings = [
            PsDeclaration(
                PsSymbolExpr(ctr),
                PsLookup(
                    PsBufferAcc(
                        ispace.index_list.base_pointer,
                        (
                            PsExpression.make(ispace.sparse_counter),
                            factory.parse_index(0),
                        ),
                    ),
                    coord.name,
                ),
            )
            for ctr, coord in zip(ispace.spatial_indices, ispace.coordinate_members)
        ]

        body = PsBlock(mappings + body.statements)

        loop = PsLoop(
            PsSymbolExpr(ispace.sparse_counter),
            PsExpression.make(PsConstant(0, self._ctx.index_dtype)),
            PsExpression.make(ispace.index_list.shape[0]),
            PsExpression.make(PsConstant(1, self._ctx.index_dtype)),
            body,
        )

        return PsBlock([loop])


class GenericVectorCpu(GenericCpu, ABC):
    """Base class for CPU platforms with vectorization support through intrinsics."""

    @abstractmethod
    def type_intrinsic(self, vector_type: PsVectorType) -> PsCustomType:
        """Return the intrinsic vector type for the given generic vector type,
        or raise a `MaterializationError` if type is not supported."""

    @abstractmethod
    def constant_intrinsic(self, c: PsConstant) -> PsExpression:
        """Return an expression that initializes a constant vector,
        or raise a `MaterializationError` if not supported."""

    @abstractmethod
    def op_intrinsic(
        self, expr: PsExpression, operands: Sequence[PsExpression]
    ) -> PsExpression:
        """Return an expression intrinsically invoking the given operation
        or raise a `MaterializationError` if not supported."""

    @abstractmethod
    def math_func_intrinsic(
        self, expr: PsCall, operands: Sequence[PsExpression]
    ) -> PsExpression:
        """Return an expression intrinsically invoking the given mathematical
        function or raise a `MaterializationError` if not supported."""

    @abstractmethod
    def vector_load(self, acc: PsVecMemAcc) -> PsExpression:
        """Return an expression intrinsically performing a vector load,
        or raise a `MaterializationError` if not supported."""

    @abstractmethod
    def vector_store(self, acc: PsVecMemAcc, arg: PsExpression) -> PsExpression:
        """Return an expression intrinsically performing a vector store,
        or raise a `MaterializationError` if not supported."""
