from __future__ import annotations
from typing import cast, Sequence
from abc import ABC, abstractmethod

from ..kernelcreation import KernelCreationContext
from ..constants import PsConstant
from ..memory import PsSymbol
from ..ast.structural import PsAstNode, PsDeclaration, PsAssignment, PsStatement
from ..ast.expressions import PsExpression, PsCall, PsCast, PsLiteral
from ...types import PsCustomType, PsVectorType, constify, deconstify
from ..ast.expressions import PsSymbolExpr, PsConstantExpr, PsUnOp, PsBinOp
from ..ast.vector import PsVecMemAcc, PsVecHorizontal
from ..exceptions import MaterializationError
from ..functions import CFunction, PsMathFunction


__all__ = ["SelectIntrinsics"]


class SelectionContext:
    def __init__(self, visitor: SelectIntrinsics):
        self._visitor = visitor
        self._ctx = visitor._ctx
        self._intrin_symbols: dict[PsSymbol, PsSymbol] = dict()
        self._lane_mask: PsSymbol | None = None

    def get_intrin_symbol(self, symb: PsSymbol) -> PsSymbol:
        if symb not in self._intrin_symbols:
            assert isinstance(symb.dtype, PsVectorType)
            intrin_type = self._visitor.type_intrinsic(deconstify(symb.dtype), self)

            if symb.dtype.const:
                intrin_type = constify(intrin_type)

            replacement = self._ctx.duplicate_symbol(symb, intrin_type)
            self._intrin_symbols[symb] = replacement

        return self._intrin_symbols[symb]


class SelectIntrinsics(ABC):
    """Lower IR vector types to intrinsic vector types, and IR vector operations to intrinsic vector operations.

    Implementations of this transformation will replace all vectorial IR elements
    by conforming implementations using
    compiler intrinsics for the given execution platform.

    A subclass implementing this visitor's abstract methods must be set up
    for each vector CPU platform.

    Args:
        ctx: The current kernel creation context
        use_builtin_convertvector: If `True`, type conversions between SIMD
            vectors use the compiler builtin ``__builtin_convertvector``
            instead of instrinsics. It is supported by Clang >= 3.7, GCC >= 9.1,
            and ICX. Not supported by ICC or MSVC. Activate if you need type
            conversions not natively supported by your CPU, e.g. conversion from
            64bit integer to double on an x86 AVX machine. Defaults to `False`.

    Raises:
        MaterializationError: If a vector type or operation cannot be represented by intrinsics
            on the given platform
    """

    #   Selection methods to be implemented by subclasses

    @abstractmethod
    def type_intrinsic(self, vector_type: PsVectorType, sc: SelectionContext) -> PsCustomType:
        """Return the intrinsic vector type for the given generic vector type,
        or raise a `MaterializationError` if type is not supported."""

    @abstractmethod
    def constant_intrinsic(self, c: PsConstant, sc: SelectionContext) -> PsExpression:
        """Return an expression that initializes a constant vector,
        or raise a `MaterializationError` if not supported."""

    @abstractmethod
    def op_intrinsic(
        self, expr: PsUnOp | PsBinOp, operands: Sequence[PsExpression], sc: SelectionContext
    ) -> PsExpression:
        """Return an expression intrinsically invoking the given operation
        or raise a `MaterializationError` if not supported."""

    @abstractmethod
    def math_func_intrinsic(
        self, expr: PsCall, operands: Sequence[PsExpression], sc: SelectionContext
    ) -> PsExpression:
        """Return an expression intrinsically invoking the given mathematical
        function or raise a `MaterializationError` if not supported."""

    @abstractmethod
    def vector_load(self, acc: PsVecMemAcc, sc: SelectionContext) -> PsExpression:
        """Return an expression intrinsically performing a vector load,
        or raise a `MaterializationError` if not supported."""

    @abstractmethod
    def vector_store(self, acc: PsVecMemAcc, arg: PsExpression, sc: SelectionContext) -> PsExpression:
        """Return an expression intrinsically performing a vector store,
        or raise a `MaterializationError` if not supported."""

    #   Visitor

    def __init__(
        self,
        ctx: KernelCreationContext,
        use_builtin_convertvector: bool = False,
    ):
        self._ctx = ctx
        self._use_builtin_convertvector = use_builtin_convertvector

    def __call__(self, node: PsAstNode) -> PsAstNode:
        return self.visit(node, SelectionContext(self))

    def visit(self, node: PsAstNode, sc: SelectionContext) -> PsAstNode:
        match node:
            case PsExpression() if isinstance(node.dtype, PsVectorType):
                return self.visit_expr(node, sc)

            case PsDeclaration(lhs, rhs) if isinstance(lhs.dtype, PsVectorType):
                lhs_new = cast(PsSymbolExpr, self.visit_expr(lhs, sc))
                rhs_new = self.visit_expr(rhs, sc)
                return PsDeclaration(lhs_new, rhs_new)

            case PsAssignment(lhs, rhs) if isinstance(lhs, PsVecMemAcc):
                new_rhs = self.visit_expr(rhs, sc)
                return PsStatement(self.vector_store(lhs, new_rhs, sc))

            case PsAssignment(lhs, rhs) if isinstance(rhs, PsVecHorizontal):
                new_rhs = self.visit_expr(rhs, sc)
                return PsAssignment(lhs, new_rhs)

            case _:
                node.children = [self.visit(c, sc) for c in node.children]

        return node

    def visit_expr(self, expr: PsExpression, sc: SelectionContext) -> PsExpression:
        if not isinstance(expr.dtype, PsVectorType):
            # special case: result type of horizontal reduction is scalar
            if isinstance(expr, PsVecHorizontal):
                scalar_op = expr.scalar_operand
                vector_op_to_scalar = self.visit_expr(expr.vector_operand, sc)
                return self.op_intrinsic(
                    expr, [scalar_op, vector_op_to_scalar], sc
                )
            else:
                return expr

        match expr:
            case PsSymbolExpr(symb):
                return PsSymbolExpr(sc.get_intrin_symbol(symb))

            case PsConstantExpr(c):
                return self.constant_intrinsic(c, sc)

            case PsCast(target_type, operand) if self._use_builtin_convertvector:
                assert isinstance(target_type, PsVectorType)
                op = self.visit_expr(operand, sc)

                rtype = PsCustomType(
                    f"{target_type.scalar_type.c_string()} __attribute__((__vector_size__({target_type.itemsize})))"
                )
                target_type_literal = PsExpression.make(PsLiteral(rtype.name, rtype))

                func = CFunction(
                    "__builtin_convertvector", (op.get_dtype(), rtype), target_type
                )
                intrinsic = func(op, target_type_literal)
                intrinsic.dtype = func.return_type
                return intrinsic

            case PsUnOp(operand):
                op = self.visit_expr(operand, sc)
                return self.op_intrinsic(expr, [op], sc)

            case PsBinOp(operand1, operand2):
                op1 = self.visit_expr(operand1, sc)
                op2 = self.visit_expr(operand2, sc)

                return self.op_intrinsic(expr, [op1, op2], sc)

            case PsVecMemAcc():
                return self.vector_load(expr, sc)

            case PsCall(function, args) if isinstance(function, PsMathFunction):
                arguments = [self.visit_expr(a, sc) for a in args]
                return self.math_func_intrinsic(expr, arguments, sc)

            case _:
                raise MaterializationError(
                    f"Unable to select intrinsic implementation for {expr}"
                )
