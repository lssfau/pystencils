from typing import Any
import pytest
import numpy as np
import sympy as sp

from pystencils import TypedSymbol, Assignment
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    Typifier,
    AstFactory,
)
from pystencils.backend.ast.structural import PsBlock, PsDeclaration
from pystencils.backend.ast.expressions import PsExpression, PsConstantExpr
from pystencils.backend.memory import PsSymbol
from pystencils.backend.constants import PsConstant
from pystencils.backend.transformations import EliminateConstants

from pystencils.backend.ast.expressions import (
    PsAnd,
    PsOr,
    PsNot,
    PsEq,
    PsGt,
    PsTernary,
    PsRem,
    PsIntDiv,
    PsCast
)

from pystencils.types.quick import Int, Fp, Bool
from pystencils.types import PsVectorType, create_numeric_type, constify, create_type


class Exprs:
    def __init__(self, mode: str):
        self.mode = mode

        if mode == "scalar":
            self._itype = Int(32)
            self._ftype = Fp(32)
            self._btype = Bool()
        else:
            self._itype = PsVectorType(Int(32), 4)
            self._ftype = PsVectorType(Fp(32), 4)
            self._btype = PsVectorType(Bool(), 4)

        self.x, self.y, self.z = [
            PsExpression.make(PsSymbol(name, self._ftype)) for name in "xyz"
        ]
        self.p, self.q, self.r = [
            PsExpression.make(PsSymbol(name, self._itype)) for name in "pqr"
        ]
        self.a, self.b, self.c = [
            PsExpression.make(PsSymbol(name, self._btype)) for name in "abc"
        ]

        self.true = PsExpression.make(PsConstant(True, self._btype))
        self.false = PsExpression.make(PsConstant(False, self._btype))

    def __call__(self, val) -> PsExpression:
        match val:
            case int():
                return PsExpression.make(PsConstant(val, self._itype))
            case float():
                return PsExpression.make(PsConstant(val, self._ftype))
            case np.ndarray():
                return PsExpression.make(
                    PsConstant(
                        val, PsVectorType(create_numeric_type(val.dtype), len(val))
                    )
                )
            case _:
                raise ValueError()


@pytest.fixture(scope="module", params=["scalar", "vector"])
def exprs(request):
    return Exprs(request.param)


def test_idempotence(exprs):
    e = exprs
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify(e(42.0) * (e(1.0) + e(0.0)) - e(0.0))
    result = elim(expr)
    assert isinstance(result, PsConstantExpr) and result.structurally_equal(e(42.0))

    expr = typify((e.x + e(0.0)) * e(3.5) + (e(1.0) * e.y + e(0.0)) * e(42.0))
    result = elim(expr)
    assert result.structurally_equal(e.x * e(3.5) + e.y * e(42.0))

    expr = typify((e(3.5) * e(1.0)) + (e(42.0) * e(1.0)))
    result = elim(expr)
    #   do not fold floats by default
    assert expr.structurally_equal(e(3.5) + e(42.0))

    expr = typify(e(1.0) * e.x + e(0.0) + (e(0.0) + e(0.0) + e(1.0) + e(0.0)) * e.y)
    result = elim(expr)
    assert result.structurally_equal(e.x + e.y)

    expr = typify(e(0.0) - e(3.2))
    result = elim(expr)
    assert result.structurally_equal(-e(3.2))


def test_int_folding(exprs):
    e = exprs
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify((e(1) * e.p + e(1) * -e(3)) + e(1) * e(12))
    result = elim(expr)
    assert result.structurally_equal((e.p + e(-3)) + e(12))

    expr = typify((e(1) + e(1) + e(1) + e(0) + e(0) + e(1)) * (e(1) + e(1) + e(1)))
    result = elim(expr)
    assert result.structurally_equal(e(12))


def test_zero_dominance(exprs):
    e = exprs
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify((e(0.0) * e.x) + (e.y * e(0.0)) + e(1.0))
    result = elim(expr)
    assert result.structurally_equal(e(1.0))

    expr = typify((e(3) + e(12) * (e.p + e.q) + e.p / (e(3) * e.q)) * e(0))
    result = elim(expr)
    assert result.structurally_equal(e(0))


def test_divisions(exprs):
    e = exprs
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify(e(3.5) / e(1.0))
    result = elim(expr)
    assert result.structurally_equal(e(3.5))

    expr = typify(e(3) / e(1))
    result = elim(expr)
    assert result.structurally_equal(e(3))

    expr = typify(PsRem(e(3), e(1)))
    result = elim(expr)
    assert result.structurally_equal(e(0))

    expr = typify(PsIntDiv(e(12), e(3)))
    result = elim(expr)
    assert result.structurally_equal(e(4))

    expr = typify(e(12) / e(3))
    result = elim(expr)
    assert result.structurally_equal(e(4))

    expr = typify(PsIntDiv(e(4), e(3)))
    result = elim(expr)
    assert result.structurally_equal(e(1))

    expr = typify(PsIntDiv(-e(4), e(3)))
    result = elim(expr)
    assert result.structurally_equal(e(-1))

    expr = typify(PsIntDiv(e(4), -e(3)))
    result = elim(expr)
    assert result.structurally_equal(e(-1))

    expr = typify(PsIntDiv(-e(4), -e(3)))
    result = elim(expr)
    assert result.structurally_equal(e(1))

    expr = typify(PsRem(e(4), e(3)))
    result = elim(expr)
    assert result.structurally_equal(e(1))

    expr = typify(PsRem(-e(4), e(3)))
    result = elim(expr)
    assert result.structurally_equal(e(-1))

    expr = typify(PsRem(e(4), -e(3)))
    result = elim(expr)
    assert result.structurally_equal(e(1))

    expr = typify(PsRem(-e(4), -e(3)))
    result = elim(expr)
    assert result.structurally_equal(e(-1))


def test_fold_floats(exprs):
    e = exprs
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx, fold_floats=True)

    expr = typify(e(8.0) / e(2.0))
    result = elim(expr)
    assert result.structurally_equal(e(4.0))

    expr = typify(e(3.0) * e(12.0) / e(6.0))
    result = elim(expr)
    assert result.structurally_equal(e(6.0))


def test_boolean_folding(exprs):
    e = exprs
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify(PsNot(PsAnd(e.false, PsOr(e.true, e.a))))
    result = elim(expr)
    assert result.structurally_equal(e.true)

    expr = typify(PsOr(PsAnd(e.a, e.b), PsNot(e.false)))
    result = elim(expr)
    assert result.structurally_equal(e.true)

    expr = typify(PsAnd(e.c, PsAnd(e.true, PsAnd(e.a, PsOr(e.false, e.b)))))
    result = elim(expr)
    assert result.structurally_equal(PsAnd(e.c, PsAnd(e.a, e.b)))

    expr = typify(PsAnd(e.false, PsAnd(e.c, e.a)))
    result = elim(expr)
    assert result.structurally_equal(e.false)

    expr = typify(PsAnd(PsOr(e.a, e.false), e.false))
    result = elim(expr)
    assert result.structurally_equal(e.false)


def test_relations_folding(exprs):
    e = exprs
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify(PsGt(e.p * e(0), -e(1)))
    result = elim(expr)
    assert result.structurally_equal(e.true)

    expr = typify(PsEq(e(1) + e(1) + e(1), e(3)))
    result = elim(expr)
    assert result.structurally_equal(e.true)

    expr = typify(PsEq(-e(1), -e(3)))
    result = elim(expr)
    assert result.structurally_equal(e.false)

    expr = typify(PsEq(e.x + e.y, e(1.0) * (e.x + e.y)))
    result = elim(expr)
    assert result.structurally_equal(e.true)

    expr = typify(PsGt(e.x + e.y, e(1.0) * (e.x + e.y)))
    result = elim(expr)
    assert result.structurally_equal(e.false)


def test_ternary_folding():
    e = Exprs("scalar")

    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify(PsTernary(e.true, e.x, e.y))
    result = elim(expr)
    assert result.structurally_equal(e.x)

    expr = typify(PsTernary(e.false, e.x, e.y))
    result = elim(expr)
    assert result.structurally_equal(e.y)

    expr = typify(
        PsTernary(PsGt(e(1), e(0)), PsTernary(PsEq(e(1), e(12)), e.x, e.y), e.z)
    )
    result = elim(expr)
    assert result.structurally_equal(e.y)

    expr = typify(PsTernary(PsGt(e.x, e.y), e.x + e(0.0), e.y * e(1.0)))
    result = elim(expr)
    assert result.structurally_equal(PsTernary(PsGt(e.x, e.y), e.x, e.y))


def test_fold_vectors():
    e = Exprs("vector")

    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx, fold_floats=True)

    expr = typify(
        e(np.array([1, 3, 2, -4]))
        - e(np.array([5, -1, -2, 6])) * e(np.array([1, -1, 1, -1]))
    )
    result = elim(expr)
    assert result.structurally_equal(e(np.array([-4, 2, 4, 2])))

    expr = typify(
        e(np.array([3.0, 1.0, 2.0, 4.0])) * e(np.array([1.0, -1.0, 1.0, -1.0]))
        + e(np.array([2.0, 3.0, 1.0, 4.0]))
    )
    result = elim(expr)
    assert result.structurally_equal(e(np.array([5.0, 2.0, 3.0, 0.0])))

    expr = typify(
        PsOr(
            PsNot(e(np.array([False, False, True, True]))),
            e(np.array([False, True, False, True])),
        )
    )
    result = elim(expr)
    assert result.structurally_equal(e(np.array([True, True, False, True])))


def test_fold_casts(exprs):
    e = exprs
    
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx, fold_floats=True)

    target_type = create_type("float16")
    if e.mode == "vector":
        target_type = PsVectorType(target_type, 4)

    expr = typify(PsCast(target_type, e(41.2)))
    result = elim(expr)

    assert isinstance(result, PsConstantExpr)
    np.testing.assert_equal(result.constant.value, e(41.2).constant.value.astype("float16"))


def test_extract_constant_subexprs():
    ctx = KernelCreationContext(default_dtype=create_numeric_type("float64"))
    factory = AstFactory(ctx)
    elim = EliminateConstants(ctx, extract_constant_exprs=True)

    x, y, z = sp.symbols("x, y, z")
    q, w = TypedSymbol("q", "float32"), TypedSymbol("w", "float32")

    block = PsBlock(
        [
            factory.parse_sympy(Assignment(x, sp.Rational(3, 2))),
            factory.parse_sympy(Assignment(y, x + sp.Rational(7, 4))),
            factory.parse_sympy(Assignment(z, y - sp.Rational(12, 5))),
            factory.parse_sympy(Assignment(q, w + sp.Rational(7, 4))),
            factory.parse_sympy(Assignment(z, y - sp.Rational(12, 5) + z * sp.sin(41))),
        ]
    )

    result = elim(block)

    assert len(result.statements) == 9

    c_symb = ctx.find_symbol("__c_3_0o2_0")
    assert c_symb is None

    c_symb = ctx.find_symbol("__c_7_0o4_0")
    assert c_symb is not None
    assert c_symb.dtype == constify(ctx.default_dtype)

    c_symb = ctx.find_symbol("__c_s12_0o5_0")
    assert c_symb is not None
    assert c_symb.dtype == constify(ctx.default_dtype)

    #   Make sure symbol was duplicated
    c_symb = ctx.find_symbol("__c_7_0o4_0__0")
    assert c_symb is not None
    assert c_symb.dtype == constify(create_numeric_type("float32"))

    c_symb = ctx.find_symbol("__c_sin_41_0_")
    assert c_symb is not None
    assert c_symb.dtype == constify(ctx.default_dtype)


def test_extract_vector_constants():
    ctx = KernelCreationContext(default_dtype=create_numeric_type("float64"))
    factory = AstFactory(ctx)
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx, extract_constant_exprs=True)

    vtype = PsVectorType(ctx.default_dtype, 8)
    x, y, z = TypedSymbol("x", vtype), TypedSymbol("y", vtype), TypedSymbol("z", vtype)

    num = typify.typify_expression(
        PsExpression.make(
            PsConstant(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
        ),
        vtype,
    )[0]

    denom = typify.typify_expression(PsExpression.make(PsConstant(3.0)), vtype)[0]

    vconstant = num / denom

    block = PsBlock(
        [
            factory.parse_sympy(Assignment(x, y - sp.Rational(3, 2))),
            PsDeclaration(
                factory.parse_sympy(z),
                typify(factory.parse_sympy(y) + num / denom),
            ),
        ]
    )

    result = elim(block)

    assert len(result.statements) == 4
    assert isinstance(result.statements[1], PsDeclaration)
    assert result.statements[1].rhs.structurally_equal(vconstant)
