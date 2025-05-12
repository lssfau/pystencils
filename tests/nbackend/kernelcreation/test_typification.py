import pytest
import sympy as sp
import numpy as np

from typing import cast

from pystencils import Assignment, TypedSymbol, Field, FieldType, AddAugmentedAssignment
from pystencils.sympyextensions import ReductionOp
from pystencils.sympyextensions.pointers import mem_acc

from pystencils.backend.ast.structural import (
    PsDeclaration,
    PsAssignment,
    PsExpression,
    PsConditional,
    PsBlock,
)
from pystencils.backend.ast.expressions import (
    PsArrayInitList,
    PsCast,
    PsConstantExpr,
    PsSymbolExpr,
    PsSubscript,
    PsBinOp,
    PsAnd,
    PsOr,
    PsNot,
    PsEq,
    PsNe,
    PsGe,
    PsLe,
    PsGt,
    PsLt,
    PsCall,
    PsTernary,
    PsMemAcc
)
from pystencils.backend.ast.vector import PsVecBroadcast, PsVecHorizontal
from pystencils.backend.ast import dfs_preorder
from pystencils.backend.ast.expressions import PsAdd
from pystencils.backend.constants import PsConstant
from pystencils.backend.functions import CFunction, PsConstantFunction, ConstantFunctions, PsReductionWriteBack
from pystencils.types import constify, create_type, create_numeric_type, PsVectorType, PsTypeError, PsPointerType
from pystencils.types.quick import Fp, Int, Bool, Arr, Ptr
from pystencils.backend.kernelcreation.context import KernelCreationContext
from pystencils.backend.kernelcreation.freeze import FreezeExpressions
from pystencils.backend.kernelcreation.typification import Typifier, TypificationError

from pystencils.sympyextensions.integer_functions import (
    bit_shift_left,
    bit_shift_right,
    bitwise_and,
    bitwise_xor,
    bitwise_or,
)


def test_typify_simple():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x, y, z = sp.symbols("x, y, z")
    asm = Assignment(z, 2 * x + y)

    fasm = freeze(asm)
    fasm = typify(fasm)

    assert isinstance(fasm, PsDeclaration)

    def check(expr):
        assert expr.dtype == ctx.default_dtype
        match expr:
            case PsConstantExpr(cs):
                assert cs.value == 2
                assert cs.dtype == constify(ctx.default_dtype)
            case PsSymbolExpr(symb):
                assert symb.name in "xyz"
                assert symb.dtype == ctx.default_dtype
            case PsBinOp(op1, op2):
                check(op1)
                check(op2)
            case _:
                pytest.fail(f"Unexpected expression: {expr}")

    check(fasm.lhs)
    check(fasm.rhs)


def test_typify_constants():
    ctx = KernelCreationContext(default_dtype=Fp(32))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    for constant in [sp.sympify(0), sp.sympify(1), sp.Rational(1, 2), sp.pi, sp.E, sp.oo, - sp.oo]:
        #   Constant on its own
        expr, _ = typify.typify_expression(freeze(constant), ctx.default_dtype)

        for node in dfs_preorder(expr):
            assert isinstance(node, PsExpression)
            assert node.dtype == ctx.default_dtype
            match node:
                case PsConstantExpr(c):
                    assert c.dtype == constify(ctx.default_dtype)
                case PsCall(func) if isinstance(func, PsConstantFunction):
                    assert func.dtype == constify(ctx.default_dtype)


def test_constants_contextual_typing():
    ctx = KernelCreationContext(default_dtype=Fp(32))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    fp16 = Fp(16)
    x = TypedSymbol("x", fp16)

    for constant in [sp.sympify(0), sp.sympify(1), sp.Rational(1, 2), sp.pi, sp.E, sp.oo, - sp.oo]:
        expr = freeze(constant) + freeze(x)  # Freeze separately such that SymPy does not simplify
        expr = typify(expr)

        assert isinstance(expr, PsAdd)

        for node in dfs_preorder(expr):
            assert isinstance(node, PsExpression)
            assert node.dtype == fp16
            match node:
                case PsConstantExpr(c):
                    assert c.dtype == constify(fp16)
                case PsCall(func) if isinstance(func, PsConstantFunction):
                    assert func.dtype == constify(fp16)


def test_no_integer_infinities_and_transcendentals():
    ctx = KernelCreationContext(default_dtype=Fp(32))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    for sp_expr in [sp.oo, - sp.oo, sp.pi, sp.E]:
        expr = freeze(sp_expr)
        with pytest.raises(PsTypeError):
            typify.typify_expression(expr, Int(32))


def test_typify_reduction_writeback():
    dtype = Fp(32)

    def _typify(arg1, arg2):
        ctx = KernelCreationContext(default_dtype=dtype)
        typify = Typifier(ctx)
        freeze = FreezeExpressions(ctx)

        ptr_expr = freeze(arg1)
        symbol_expr = freeze(arg2)

        writeback = PsCall(PsReductionWriteBack(ReductionOp.Add), [ptr_expr, symbol_expr])
        return typify(writeback)

    # check types for successful usage of PsReductionWriteBack
    successful_writeback = _typify(TypedSymbol("ptr", PsPointerType(dtype)), sp.Symbol("w"))

    assert successful_writeback.dtype == dtype

    ptr_arg, symbol_arg = successful_writeback.args
    assert ptr_arg.dtype == PsPointerType(dtype)
    assert symbol_arg.dtype == dtype

    # failing case: no pointer passed as first arg
    with pytest.raises(TypificationError):
        _ = _typify(sp.Symbol("a"), sp.Symbol("b"))

    # failing case: no scalar passed as second arg
    with pytest.raises(TypificationError):
        _ = _typify(TypedSymbol("c", PsPointerType(dtype)), TypedSymbol("d", PsVectorType(dtype, 4)))


def test_lhs_constness():
    default_type = Fp(32)
    ctx = KernelCreationContext(default_dtype=default_type)
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    f = Field.create_generic(
        "f", 1, index_shape=(1,), dtype=default_type, field_type=FieldType.CUSTOM
    )
    f_const = Field.create_generic(
        "f_const",
        1,
        index_shape=(1,),
        dtype=constify(default_type),
        field_type=FieldType.CUSTOM,
    )

    x, y, z = sp.symbols("x, y, z")

    #   Can assign to non-const LHS
    asm = typify(freeze(Assignment(f.absolute_access([0], [0]), x + y)))
    assert not asm.lhs.get_dtype().const

    #   Cannot assign to const left-hand side
    with pytest.raises(TypificationError):
        _ = typify(freeze(Assignment(f_const.absolute_access([0], [0]), x + y)))

    np_struct = np.dtype([("size", np.uint32), ("data", np.float32)], align=True)
    struct_type = constify(create_type(np_struct))
    struct_field = Field.create_generic(
        "struct_field", 1, dtype=struct_type, field_type=FieldType.CUSTOM
    )

    with pytest.raises(TypificationError):
        _ = typify(freeze(Assignment(struct_field.absolute_access([0], "data"), x)))

    #   Const LHS is only OK in declarations

    q = ctx.get_symbol("q", Fp(32, const=True))
    ast = PsDeclaration(PsExpression.make(q), PsExpression.make(q))
    ast = typify(ast)
    assert ast.lhs.dtype == Fp(32)

    ast = PsAssignment(PsExpression.make(q), PsExpression.make(q))
    with pytest.raises(TypificationError):
        typify(ast)


def test_typify_structs():
    ctx = KernelCreationContext(default_dtype=Fp(32))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    np_struct = np.dtype([("size", np.uint32), ("data", np.float32)], align=True)
    f = Field.create_generic("f", 1, dtype=np_struct, field_type=FieldType.CUSTOM)
    x = sp.Symbol("x")

    #   Good
    asm = Assignment(x, f.absolute_access((0,), "data"))
    fasm = freeze(asm)
    fasm = typify(fasm)

    asm = Assignment(f.absolute_access((0,), "data"), x)
    fasm = freeze(asm)
    fasm = typify(fasm)

    #   Bad
    asm = Assignment(x, f.absolute_access((0,), "size"))
    fasm = freeze(asm)
    with pytest.raises(TypificationError):
        fasm = typify(fasm)


def test_default_typing():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x, y, z = sp.symbols("x, y, z")
    expr = freeze(2 * x + 3 * y + z - 4)
    expr = typify(expr)

    def check(expr):
        assert expr.dtype == ctx.default_dtype
        match expr:
            case PsConstantExpr(cs):
                assert cs.value in (2, 3, -4)
                assert cs.dtype == constify(ctx.default_dtype)
            case PsSymbolExpr(symb):
                assert symb.name in "xyz"
                assert symb.dtype == ctx.default_dtype
            case PsBinOp(op1, op2):
                check(op1)
                check(op2)
            case _:
                pytest.fail(f"Unexpected expression: {expr}")

    check(expr)


def test_inline_arrays_1d():
    ctx = KernelCreationContext(default_dtype=Fp(32))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x = sp.Symbol("x")
    y = TypedSymbol("y", Fp(16))
    idx = TypedSymbol("idx", Int(32))

    arr: PsArrayInitList = cast(PsArrayInitList, freeze(sp.Tuple(1, 2, 3, 4)))
    decl = PsDeclaration(freeze(x), freeze(y) + PsSubscript(arr, (freeze(idx),)))
    #   The array elements should learn their type from the context, which gets it from `y`

    decl = typify(decl)
    assert decl.lhs.dtype == Fp(16)
    assert decl.rhs.dtype == Fp(16)

    assert arr.dtype == Arr(Fp(16), (4,))
    for item in arr.items:
        assert item.dtype == Fp(16)


def test_inline_arrays_3d():
    ctx = KernelCreationContext(default_dtype=Fp(32))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x = sp.Symbol("x")
    y = TypedSymbol("y", Fp(16))
    idx = [TypedSymbol(f"idx_{i}", Int(32)) for i in range(3)]

    arr: PsArrayInitList = freeze(
        sp.Tuple(((1, 2), (3, 4), (5, 6)), ((5, 6), (7, 8), (9, 10)))
    )
    decl = PsDeclaration(
        freeze(x),
        freeze(y) + PsSubscript(arr, (freeze(idx[0]), freeze(idx[1]), freeze(idx[2]))),
    )
    #   The array elements should learn their type from the context, which gets it from `y`

    decl = typify(decl)
    assert decl.lhs.dtype == Fp(16)
    assert decl.rhs.dtype == Fp(16)

    assert arr.dtype == Arr(Fp(16), (2, 3, 2))
    assert arr.shape == (2, 3, 2)
    for item in arr.items:
        assert item.dtype == Fp(16)


def test_array_subscript():
    ctx = KernelCreationContext(default_dtype=Fp(16))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    arr = sp.IndexedBase(TypedSymbol("arr", Arr(Fp(32), (16,))))
    expr = freeze(arr[3])
    expr = typify(expr)

    assert expr.dtype == Fp(32)

    arr = sp.IndexedBase(TypedSymbol("arr2", Arr(Fp(32), (7, 31))))
    expr = freeze(arr[3, 5])
    expr = typify(expr)

    assert expr.dtype == Fp(32)


def test_invalid_subscript():
    ctx = KernelCreationContext(default_dtype=Fp(16))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    non_arr = sp.IndexedBase(TypedSymbol("non_arr", Int(64)))
    expr = freeze(non_arr[3])

    with pytest.raises(TypificationError):
        expr = typify(expr)

    wrong_shape_arr = sp.IndexedBase(
        TypedSymbol("wrong_shape_arr", Arr(Fp(32), (7, 31, 5)))
    )
    expr = freeze(wrong_shape_arr[3, 5])

    with pytest.raises(TypificationError):
        expr = typify(expr)

    #   raw pointers are not arrays, cannot enter subscript
    ptr = sp.IndexedBase(
        TypedSymbol("ptr", Ptr(Int(16)))
    )
    expr = freeze(ptr[37])
    
    with pytest.raises(TypificationError):
        expr = typify(expr)

    
def test_mem_acc():
    ctx = KernelCreationContext(default_dtype=Fp(16))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    ptr = TypedSymbol("ptr", Ptr(Int(64)))
    idx = TypedSymbol("idx", Int(32))
    
    expr = freeze(mem_acc(ptr, idx))
    expr = typify(expr)

    assert isinstance(expr, PsMemAcc)
    assert expr.dtype == Int(64)
    assert expr.offset.dtype == Int(32)


def test_invalid_mem_acc():
    ctx = KernelCreationContext(default_dtype=Fp(16))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    non_ptr = TypedSymbol("non_ptr", Int(64))
    idx = TypedSymbol("idx", Int(32))
    
    expr = freeze(mem_acc(non_ptr, idx))
    
    with pytest.raises(TypificationError):
        _ = typify(expr)
    
    arr = TypedSymbol("arr", Arr(Int(64), (31,)))
    idx = TypedSymbol("idx", Int(32))
    
    expr = freeze(mem_acc(arr, idx))
    
    with pytest.raises(TypificationError):
        _ = typify(expr)


def test_lhs_inference():
    ctx = KernelCreationContext(default_dtype=create_numeric_type(np.float64))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x, y, z = sp.symbols("x, y, z")
    q = TypedSymbol("q", np.float32)
    w = TypedSymbol("w", np.float16)

    #   Type of the LHS is propagated to untyped RHS symbols

    asm = Assignment(x, 3 - q)
    fasm = typify(freeze(asm))

    assert ctx.get_symbol("x").dtype == Fp(32)
    assert fasm.lhs.dtype == Fp(32)

    asm = Assignment(y, 3 - w)
    fasm = typify(freeze(asm))

    assert ctx.get_symbol("y").dtype == Fp(16)
    assert fasm.lhs.dtype == Fp(16)

    fasm = PsAssignment(PsExpression.make(ctx.get_symbol("z")), freeze(3 - w))
    fasm = typify(fasm)

    assert ctx.get_symbol("z").dtype == Fp(16)
    assert fasm.lhs.dtype == Fp(16)

    fasm = PsDeclaration(
        PsExpression.make(ctx.get_symbol("r")), PsLe(freeze(q), freeze(2 * q))
    )
    fasm = typify(fasm)

    assert ctx.get_symbol("r").dtype == Bool()
    assert fasm.lhs.dtype == Bool()
    assert fasm.rhs.dtype == Bool()


def test_array_declarations():
    ctx = KernelCreationContext(default_dtype=Fp(32))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x, y, z = sp.symbols("x, y, z")
    
    #   Array type fallback to default
    arr1 = sp.Symbol("arr1")
    decl = freeze(Assignment(arr1, sp.Tuple(1, 2, 3, 4)))
    decl = typify(decl)

    assert ctx.get_symbol("arr1").dtype == Arr(Fp(32), (4,))
    assert decl.lhs.dtype == decl.rhs.dtype == Arr(Fp(32), (4,))

    #   Array type determined by default-typed symbol
    arr2 = sp.Symbol("arr2")
    decl = freeze(Assignment(arr2, sp.Tuple((x, y, -7), (3, -2, 51))))
    decl = typify(decl)

    assert ctx.get_symbol("arr2").dtype == Arr(Fp(32), (2, 3))
    assert decl.lhs.dtype == decl.rhs.dtype == Arr(Fp(32), (2, 3))

    #   Array type determined by pre-typed symbol
    q = TypedSymbol("q", Fp(16))
    arr3 = sp.Symbol("arr3")
    decl = freeze(Assignment(arr3, sp.Tuple((q, 2), (-q, 0.123))))
    decl = typify(decl)

    assert ctx.get_symbol("arr3").dtype == Arr(Fp(16), (2, 2))
    assert decl.lhs.dtype == decl.rhs.dtype == Arr(Fp(16), (2, 2))

    #   Array type determined by LHS symbol
    arr4 = TypedSymbol("arr4", Arr(Int(16), 4))
    decl = freeze(Assignment(arr4, sp.Tuple(11, 1, 4, 2)))
    decl = typify(decl)

    assert decl.lhs.dtype == decl.rhs.dtype == Arr(Int(16), 4)


def test_erronous_typing():
    ctx = KernelCreationContext(default_dtype=create_numeric_type(np.float64))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x, y, z = sp.symbols("x, y, z")
    q = TypedSymbol("q", np.float32)
    w = TypedSymbol("w", np.float16)

    expr = freeze(2 * x + 3 * y + q - 4)

    with pytest.raises(TypificationError):
        typify(expr)

    #   Conflict between LHS and RHS symbols
    asm = Assignment(q, 3 - w)
    fasm = freeze(asm)
    with pytest.raises(TypificationError):
        typify(fasm)

    #   Do not propagate types back from LHS symbols to RHS symbols
    asm = Assignment(q, 3 - x)
    fasm = freeze(asm)
    with pytest.raises(TypificationError):
        typify(fasm)

    asm = AddAugmentedAssignment(z, 3 - q)
    fasm = freeze(asm)
    with pytest.raises(TypificationError):
        typify(fasm)


def test_invalid_indices():
    ctx = KernelCreationContext(default_dtype=create_numeric_type(np.float64))
    typify = Typifier(ctx)

    arr = PsExpression.make(ctx.get_symbol("arr", Arr(Fp(64), (61,))))
    x, y, z = [PsExpression.make(ctx.get_symbol(x)) for x in "xyz"]

    #   Using default-typed symbols as array indices is illegal when the default type is a float

    fasm = PsAssignment(PsSubscript(arr, (x + y,)), z)

    with pytest.raises(TypificationError):
        typify(fasm)

    fasm = PsAssignment(z, PsSubscript(arr, (x + y,)))

    with pytest.raises(TypificationError):
        typify(fasm)


def test_typify_integer_binops():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    ctx.get_symbol("x", ctx.index_dtype)
    ctx.get_symbol("y", ctx.index_dtype)

    x, y = sp.symbols("x, y")
    expr = bit_shift_left(
        bit_shift_right(bitwise_and(2, 2), bitwise_or(x, y)), bitwise_xor(2, 2)
    )
    expr = freeze(expr)
    expr = typify(expr)

    def check(expr):
        match expr:
            case PsConstantExpr(cs):
                assert cs.value == 2
                assert cs.dtype == constify(ctx.index_dtype)
            case PsSymbolExpr(symb):
                assert symb.name in "xyz"
                assert symb.dtype == ctx.index_dtype
            case PsBinOp(op1, op2):
                check(op1)
                check(op2)
            case _:
                pytest.fail(f"Unexpected expression: {expr}")

    check(expr)


def test_typify_integer_binops_floating_arg():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x = sp.Symbol("x")
    expr = bit_shift_left(x, 2)
    expr = freeze(expr)

    with pytest.raises(TypificationError):
        expr = typify(expr)


def test_typify_integer_binops_in_floating_context():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    ctx.get_symbol("i", ctx.index_dtype)

    x, i = sp.symbols("x, i")
    expr = x + bit_shift_left(i, 2)
    expr = freeze(expr)

    with pytest.raises(TypificationError):
        expr = typify(expr)


def test_typify_constant_clones():
    ctx = KernelCreationContext(default_dtype=Fp(32))
    typify = Typifier(ctx)

    c = PsConstantExpr(PsConstant(3.0))
    x = PsSymbolExpr(ctx.get_symbol("x"))
    expr = c + x
    expr_clone = expr.clone()

    expr = typify(expr)

    assert expr_clone.operand1.dtype is None
    assert cast(PsConstantExpr, expr_clone.operand1).constant.dtype is None


def test_typify_bools_and_relations():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)

    true = PsConstantExpr(PsConstant(True, Bool()))
    p, q = [PsExpression.make(ctx.get_symbol(name, Bool())) for name in "pq"]
    x, y = [PsExpression.make(ctx.get_symbol(name, Fp(32))) for name in "xy"]

    expr = PsAnd(PsEq(x, y), PsAnd(true, PsNot(PsOr(p, q))))
    expr = typify(expr)

    assert expr.dtype == Bool() 


def test_bool_in_numerical_context():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)

    true = PsConstantExpr(PsConstant(True, Bool()))
    p, q = [PsExpression.make(ctx.get_symbol(name, Bool())) for name in "pq"]

    expr = true + (p - q)
    with pytest.raises(TypificationError):
        typify(expr)


@pytest.mark.parametrize("rel", [PsEq, PsNe, PsLt, PsGt, PsLe, PsGe])
def test_typify_conditionals(rel):
    ctx = KernelCreationContext()
    typify = Typifier(ctx)

    x, y = [PsExpression.make(ctx.get_symbol(name, Fp(32))) for name in "xy"]

    cond = PsConditional(rel(x, y), PsBlock([]))
    cond = typify(cond)
    assert cond.condition.dtype == Bool()


def test_invalid_conditions():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)

    x, y = [PsExpression.make(ctx.get_symbol(name, Fp(32))) for name in "xy"]
    p, q = [PsExpression.make(ctx.get_symbol(name, Bool())) for name in "pq"]

    cond = PsConditional(x + y, PsBlock([]))
    with pytest.raises(TypificationError):
        typify(cond)


def test_typify_ternary():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)

    x, y = [PsExpression.make(ctx.get_symbol(name, Fp(32))) for name in "xy"]
    a, b = [PsExpression.make(ctx.get_symbol(name, Int(32))) for name in "ab"]
    p, q = [PsExpression.make(ctx.get_symbol(name, Bool())) for name in "pq"]

    expr = PsTernary(p, x, y)
    expr = typify(expr)
    assert expr.dtype == Fp(32)

    expr = PsTernary(PsAnd(p, q), a, b + a)
    expr = typify(expr)
    assert expr.dtype == Int(32)

    expr = PsTernary(PsAnd(p, q), a, x)
    with pytest.raises(TypificationError):
        typify(expr)

    expr = PsTernary(y, a, b)
    with pytest.raises(TypificationError):
        typify(expr)


def test_cfunction():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    x, y = [PsExpression.make(ctx.get_symbol(name, Fp(32))) for name in "xy"]
    p, q = [PsExpression.make(ctx.get_symbol(name, Int(32))) for name in "pq"]

    def _threeway(x: np.float32, y: np.float32) -> np.int32:
        assert False

    threeway = CFunction.parse(_threeway)

    result = typify(PsCall(threeway, [x, y]))

    assert result.get_dtype() == Int(32)
    assert result.args[0].get_dtype() == Fp(32)
    assert result.args[1].get_dtype() == Fp(32)

    with pytest.raises(TypificationError):
        _ = typify(PsCall(threeway, (x, p)))


def test_typify_typecast():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)

    x, y = [PsExpression.make(ctx.get_symbol(name, Fp(32))) for name in "xy"]
    p, q = [PsExpression.make(ctx.get_symbol(name, Int(32))) for name in "pq"]

    #   Explicit target type
    expr = typify(PsCast(Int(64), x))
    assert expr.dtype == expr.target_type == Int(64)

    #   Infer target type from context
    cast_expr = PsCast(None, p)
    expr = typify(y + cast_expr)
    assert expr.dtype == Fp(32)
    assert cast_expr.dtype == cast_expr.target_type == Fp(32)

    #   Invalid target type
    expr = p + PsCast(Fp(64), q)
    with pytest.raises(TypificationError):
        typify(expr)


def test_typify_integer_vectors():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)

    a, b, c = [PsExpression.make(ctx.get_symbol(name, PsVectorType(Int(32), 4))) for name in "abc"]
    d, e = [PsExpression.make(ctx.get_symbol(name, Int(32))) for name in "de"]

    result = typify(a + (b / c) - a * c)
    assert result.get_dtype() == PsVectorType(Int(32), 4)

    result = typify(PsVecBroadcast(4, d - e) - PsVecBroadcast(4, e / d))
    assert result.get_dtype() == PsVectorType(Int(32), 4)


def test_typify_bool_vectors():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)

    x, y = [PsExpression.make(ctx.get_symbol(name, PsVectorType(Fp(32), 4))) for name in "xy"]
    p, q = [PsExpression.make(ctx.get_symbol(name, PsVectorType(Bool(), 4))) for name in "pq"]

    result = typify(PsAnd(PsOr(p, q), p))
    assert result.get_dtype() == PsVectorType(Bool(), 4)

    result = typify(PsAnd(PsLt(x, y), PsGe(y, x)))
    assert result.get_dtype() == PsVectorType(Bool(), 4)


def test_propagate_constant_type_in_broadcast():
    fp16 = Fp(16)

    for constant in [
        PsConstantFunction(ConstantFunctions.E, fp16)(),
        PsConstantFunction(ConstantFunctions.PosInfinity, fp16)(),
        PsConstantExpr(PsConstant(3.5, fp16))
    ]:
        ctx = KernelCreationContext(default_dtype=Fp(32))
        typify = Typifier(ctx)

        expr = PsVecBroadcast(4, constant)
        expr = typify(expr)
        assert expr.dtype == PsVectorType(fp16, 4)


def test_typify_horizontal_vector_reductions():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)

    reduction_op = ReductionOp.Add
    stype = Fp(32)
    vtype = PsVectorType(stype, 4)

    def create_symb_expr(name, tpe):
        return PsExpression.make(ctx.get_symbol(name, tpe))

    # create valid horizontal and check if expression type is scalar
    result = typify(
        PsVecHorizontal(
            create_symb_expr("s1", stype), create_symb_expr("v1", vtype), ReductionOp.Add
        )
    )
    assert result.get_dtype() == stype

    # create invalid horizontal by using scalar type for expected vector type
    with pytest.raises(TypificationError):
        _ = typify(
            PsVecHorizontal(
                create_symb_expr("s2", stype), create_symb_expr("v2", stype), reduction_op
            )
        )

    # create invalid horizontal by using vector type for expected scalar type
    with pytest.raises(TypificationError):
        _ = typify(
            PsVecHorizontal(
                create_symb_expr("s3", vtype), create_symb_expr("v3", vtype), reduction_op
            )
        )

    # create invalid horizontal where base type of vector does not match with scalar type
    with pytest.raises(TypificationError):
        _ = typify(
            PsVecHorizontal(
                create_symb_expr("s4", Int(32)), create_symb_expr("v4", vtype), reduction_op
            )
        )


def test_inference_fails():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)

    x = PsExpression.make(PsConstant(42))

    with pytest.raises(TypificationError):
        typify(PsEq(x, x))

    with pytest.raises(TypificationError):
        typify(PsArrayInitList([x]))

    with pytest.raises(TypificationError):
        typify(PsCast(ctx.default_dtype, x))
