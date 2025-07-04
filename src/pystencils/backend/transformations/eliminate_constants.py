from typing import cast, Iterable, overload
from collections import defaultdict

import numpy as np

from ..kernelcreation import KernelCreationContext, Typifier

from ..ast import PsAstNode
from ..ast.structural import PsBlock, PsDeclaration, PsStructuralNode
from ..ast.expressions import (
    PsExpression,
    PsConstantExpr,
    PsSymbolExpr,
    PsLiteralExpr,
    PsBinOp,
    PsAdd,
    PsSub,
    PsMul,
    PsDiv,
    PsIntDiv,
    PsRem,
    PsAnd,
    PsOr,
    PsRel,
    PsNeg,
    PsNot,
    PsCall,
    PsEq,
    PsGe,
    PsLe,
    PsLt,
    PsGt,
    PsNe,
    PsTernary,
    PsCast,
)
from ..ast.vector import PsVecBroadcast
from ..ast.util import AstEqWrapper
from ..exceptions import PsInternalCompilerError

from ..constants import PsConstant
from ..memory import PsSymbol
from ..functions import PsMathFunction
from ...types import PsNumericType, PsBoolType, PsScalarType, PsVectorType, constify


__all__ = ["EliminateConstants"]


class ECContext:
    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx
        self._extracted_constants: dict[AstEqWrapper, PsSymbol] = dict()

        from ..emission import IRAstPrinter

        self._printer = IRAstPrinter(indent_width=0, annotate_constants=False)

    @property
    def extractions(self) -> Iterable[tuple[PsSymbol, PsExpression]]:
        return [
            (symb, cast(PsExpression, w.n))
            for (w, symb) in self._extracted_constants.items()
        ]

    def _get_symb_name(self, expr: PsExpression):
        code = self._printer(expr)
        code = code.lower()
        #   remove spaces
        code = "".join(code.split())

        def valid_char(c):
            return (ord("0") <= ord(c) <= ord("9")) or (ord("a") <= ord(c) <= ord("z"))

        charmap = {"+": "p", "-": "s", "*": "m", "/": "o"}
        charmap = defaultdict(lambda: "_", charmap)  # type: ignore

        code = "".join((c if valid_char(c) else charmap[c]) for c in code)
        return f"__c_{code}"

    def extract_expression(self, expr: PsExpression) -> PsSymbolExpr:
        dtype = expr.get_dtype()
        expr_wrapped = AstEqWrapper(expr)

        if expr_wrapped not in self._extracted_constants:
            symb_name = self._get_symb_name(expr)
            symb = self._ctx.get_new_symbol(symb_name, constify(dtype))

            self._extracted_constants[expr_wrapped] = symb
        else:
            symb = self._extracted_constants[expr_wrapped]

        return PsSymbolExpr(symb)


class EliminateConstants:
    """Eliminate constant expressions in various ways.

    - Constant folding: Nontrivial constant integer (and optionally floating point) expressions
      are evaluated and replaced by their result
    - Idempotence elimination: Idempotent operations (e.g. addition of zero, multiplication with one)
      are replaced by their result
    - Dominance elimination: Multiplication by zero is replaced by zero
    - Constant extraction: Optionally, nontrivial constant expressions are extracted and listed at the beginning of
      the outermost block.
    """

    def __init__(
        self,
        ctx: KernelCreationContext,
        extract_constant_exprs: bool = False,
        fold_integers: bool = True,
        fold_relations: bool = True,
        fold_floats: bool = False,
    ):
        self._ctx = ctx
        self._typify = Typifier(ctx)

        self._fold_integers = fold_integers
        self._fold_relations = fold_relations
        self._fold_floats = fold_floats
        self._extract_constant_exprs = extract_constant_exprs

    @overload
    def __call__(self, node: PsExpression) -> PsExpression:
        pass

    @overload
    def __call__(self, node: PsBlock) -> PsBlock:
        pass

    @overload
    def __call__(self, node: PsAstNode) -> PsAstNode:
        pass

    def __call__(self, node: PsAstNode) -> PsAstNode:
        ecc = ECContext(self._ctx)

        node = self.visit(node, ecc)

        if ecc.extractions:
            if not isinstance(node, PsStructuralNode):
                raise PsInternalCompilerError(
                    f"Cannot extract constant expressions from outermost node {node}"
                )

            prepend_decls = [
                PsDeclaration(PsExpression.make(symb), expr)
                for symb, expr in ecc.extractions
            ]

            if not isinstance(node, PsBlock):
                node = PsBlock(prepend_decls + [node])
            else:
                node.children = prepend_decls + list(node.children)

        return node

    def visit(self, node: PsAstNode, ecc: ECContext) -> PsAstNode:
        match node:
            case PsExpression():
                transformed_expr, _ = self.visit_expr(node, ecc)
                return transformed_expr
            case _:
                node.children = [self.visit(c, ecc) for c in node.children]
                return node

    def visit_expr(
        self, expr: PsExpression, ecc: ECContext
    ) -> tuple[PsExpression, bool]:
        """Transformation of expressions.

        Returns:
            (transformed_expr, is_const): The tranformed expression, and a flag indicating whether it is constant
        """
        #   Return constants and literals as they are
        if isinstance(expr, (PsConstantExpr, PsLiteralExpr)):
            return expr, True

        #   Shortcut symbols
        if isinstance(expr, PsSymbolExpr):
            return expr, False

        subtree_results = [
            self.visit_expr(cast(PsExpression, c), ecc) for c in expr.children
        ]
        expr.children = [r[0] for r in subtree_results]
        subtree_constness = [r[1] for r in subtree_results]

        #   Eliminate idempotence, dominance. constant (broad)casts, and trivial relations
        match expr:
            #   Additive idempotence: Addition and subtraction of zero
            case PsAdd(PsConstantExpr(c), other_op) if np.all(c.value == 0):
                return other_op, all(subtree_constness)

            case PsAdd(other_op, PsConstantExpr(c)) if np.all(c.value == 0):
                return other_op, all(subtree_constness)

            case PsSub(other_op, PsConstantExpr(c)) if np.all(c.value == 0):
                return other_op, all(subtree_constness)

            #   Additive idempotence: Subtraction from zero
            case PsSub(PsConstantExpr(c), other_op) if np.all(c.value == 0):
                other_transformed, is_const = self.visit_expr(
                    self._typify(-other_op), ecc
                )
                return other_transformed, is_const

            #   Multiplicative idempotence: Multiplication with and division by one
            case PsMul(PsConstantExpr(c), other_op) if np.all(c.value == 1):
                return other_op, all(subtree_constness)

            case PsMul(other_op, PsConstantExpr(c)) if np.all(c.value == 1):
                return other_op, all(subtree_constness)

            case PsDiv(other_op, PsConstantExpr(c)) | PsIntDiv(
                other_op, PsConstantExpr(c)
            ) if np.all(c.value == 1):
                return other_op, all(subtree_constness)

            #   Trivial remainder at division by one
            case PsRem(other_op, PsConstantExpr(c)) if np.all(c.value == 1):
                zero = self._typify(PsConstantExpr(PsConstant(0, c.get_dtype())))
                return zero, True

            #   Multiplicative dominance: 0 * x = 0
            case PsMul(PsConstantExpr(c), other_op) if np.all(c.value == 0):
                return PsConstantExpr(c), True

            case PsMul(other_op, PsConstantExpr(c)) if np.all(c.value == 0):
                return PsConstantExpr(c), True

            #   Logical idempotence
            case PsAnd(PsConstantExpr(c), other_op) if np.all(c.value):
                return other_op, all(subtree_constness)

            case PsAnd(other_op, PsConstantExpr(c)) if np.all(c.value):
                return other_op, all(subtree_constness)

            case PsOr(PsConstantExpr(c), other_op) if not np.any(c.value):
                return other_op, all(subtree_constness)

            case PsOr(other_op, PsConstantExpr(c)) if not np.any(c.value):
                return other_op, all(subtree_constness)

            #   Logical dominance
            case PsAnd(PsConstantExpr(c), other_op) if not np.any(c.value):
                return PsConstantExpr(c), True

            case PsAnd(other_op, PsConstantExpr(c)) if not np.any(c.value):
                return PsConstantExpr(c), True

            case PsOr(PsConstantExpr(c), other_op) if np.all(c.value):
                return PsConstantExpr(c), True

            case PsOr(other_op, PsConstantExpr(c)) if np.all(c.value):
                return PsConstantExpr(c), True

            #   Trivial (broad)casts
            case PsCast(target_type, PsConstantExpr(c)):
                assert isinstance(target_type, PsNumericType)
                return PsConstantExpr(c.reinterpret_as(target_type)), True

            case PsCast(target_type, op) if target_type == op.get_dtype():
                return op, all(subtree_constness)

            case PsVecBroadcast(lanes, PsConstantExpr(c)):
                scalar_type = c.get_dtype()
                assert isinstance(scalar_type, PsScalarType)
                vec_type = PsVectorType(scalar_type, lanes)
                return PsConstantExpr(PsConstant(c.value, vec_type)), True

            #   Trivial comparisons
            case (
                PsEq(op1, op2) | PsGe(op1, op2) | PsLe(op1, op2)
            ) if op1.structurally_equal(op2):
                arg_dtype = op1.get_dtype()
                bool_type = (
                    PsVectorType(PsBoolType(), arg_dtype.vector_entries)
                    if isinstance(arg_dtype, PsVectorType)
                    else PsBoolType()
                )
                true = self._typify(PsConstantExpr(PsConstant(True, bool_type)))
                return true, True

            case (
                PsNe(op1, op2) | PsGt(op1, op2) | PsLt(op1, op2)
            ) if op1.structurally_equal(op2):
                arg_dtype = op1.get_dtype()
                bool_type = (
                    PsVectorType(PsBoolType(), arg_dtype.vector_entries)
                    if isinstance(arg_dtype, PsVectorType)
                    else PsBoolType()
                )
                false = self._typify(PsConstantExpr(PsConstant(False, bool_type)))
                return false, True

            #   Trivial ternaries
            case PsTernary(PsConstantExpr(c), then, els):
                if c.value:
                    return then, subtree_constness[1]
                else:
                    return els, subtree_constness[2]

        # end match: no idempotence or dominance encountered

        #   Detect constant expressions
        if all(subtree_constness):
            dtype = expr.get_dtype()

            is_rel = isinstance(expr, PsRel)

            if isinstance(dtype, PsNumericType):
                is_int = dtype.is_int()
                is_float = dtype.is_float()
                is_bool = dtype.is_bool()
            else:
                is_int = is_float = is_bool = False

            do_fold = (
                is_bool
                or (self._fold_integers and is_int)
                or (self._fold_floats and is_float)
                or (self._fold_relations and is_rel)
            )

            folded: PsConstant | None

            match expr:
                case PsNeg(operand) | PsNot(operand):
                    if isinstance(operand, PsConstantExpr):
                        val = operand.constant.value
                        py_operator = expr.python_operator

                        if do_fold and py_operator is not None:
                            assert isinstance(dtype, PsNumericType)
                            folded = PsConstant(py_operator(val), dtype)
                            return self._typify(PsConstantExpr(folded)), True

                    return expr, True

                case PsBinOp(op1, op2):
                    if isinstance(op1, PsConstantExpr) and isinstance(
                        op2, PsConstantExpr
                    ):
                        v1 = op1.constant.value
                        v2 = op2.constant.value

                        if do_fold:
                            assert isinstance(dtype, PsNumericType)
                            py_operator = expr.python_operator

                            folded = None
                            if py_operator is not None:
                                folded = PsConstant(
                                    py_operator(v1, v2),
                                    dtype,
                                )
                            elif isinstance(expr, PsDiv):
                                if is_int:
                                    from ...utils import c_intdiv

                                    folded = PsConstant(c_intdiv(v1, v2), dtype)
                                elif (
                                    isinstance(dtype, PsNumericType)
                                    and dtype.is_float()
                                ):
                                    folded = PsConstant(v1 / v2, dtype)

                            if folded is not None:
                                return self._typify(PsConstantExpr(folded)), True

                    return expr, True

                case PsCall(PsMathFunction(), _):
                    #   TODO: Some math functions (min/max) might be safely folded
                    return expr, True
        # end if: this expression is not constant

        #   If required, extract constant subexpressions
        if self._extract_constant_exprs:
            for i, (child, is_const) in enumerate(subtree_results):
                if is_const and not isinstance(child, (PsConstantExpr, PsLiteralExpr)):
                    replacement = ecc.extract_expression(child)
                    expr.set_child(i, replacement)

        #   Any other expressions are not considered constant even if their arguments are
        return expr, False
