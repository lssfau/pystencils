import pytest
import sympy as sp
from operator import add, sub, mul

from pystencils import create_type, TypedSymbol, make_slice, Assignment
from pystencils.sympyextensions import mem_acc

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    Typifier,
)
from pystencils.backend.memory import BufferBasePtr
from pystencils.backend.constants import PsConstant
from pystencils.backend.ast.expressions import (
    PsExpression,
    PsCast,
    PsMemAcc,
    PsArrayInitList,
    PsSubscript,
    PsBufferAcc,
    PsSymbolExpr,
    PsTernary,
    PsBinOp,
)
from pystencils.backend.ast.structural import (
    PsStatement,
    PsAssignment,
    PsDeclaration,
    PsBlock,
    PsConditional,
    PsComment,
    PsPragma,
    PsLoop,
)
from pystencils.backend.ast.axes import PsLoopAxis
from pystencils.backend.ast.analysis import collect_undefined_symbols
from pystencils.types.quick import Fp, Ptr


def test_cloning():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)

    x, y, z, m = [PsExpression.make(ctx.get_symbol(name)) for name in "xyzm"]
    q = PsExpression.make(ctx.get_symbol("q", create_type("bool")))
    a, b, c = [
        PsExpression.make(ctx.get_symbol(name, ctx.index_dtype)) for name in "abc"
    ]
    c1 = PsExpression.make(PsConstant(3.0))
    c2 = PsExpression.make(PsConstant(-1.0))
    one_f = PsExpression.make(PsConstant(1.0))
    one_i = PsExpression.make(PsConstant(1))

    def check(orig, clone):
        assert not (orig is clone)
        assert type(orig) is type(clone)
        assert orig.structurally_equal(clone)

        if isinstance(orig, PsExpression):
            #   Regression: Expression data types used to not be cloned
            assert orig.dtype == clone.dtype

        for c1, c2 in zip(orig.children, clone.children, strict=True):
            check(c1, c2)

    for ast in [
        x,
        y,
        c1,
        x + y,
        x / y + c1,
        c1 + c2,
        PsStatement(x * y * z + c1),
        PsAssignment(y, x / c1),
        PsBlock([PsAssignment(x, c1 * y), PsAssignment(z, c2 + c1 * z)]),
        PsConditional(
            q, PsBlock([PsStatement(x + y)]), PsBlock([PsComment("hello world")])
        ),
        PsDeclaration(m, PsArrayInitList([[x, y, one_f + x], [one_f, c2, z]])),
        PsPragma("omp parallel for"),
        PsLoop(
            a,
            b,
            c,
            one_i,
            PsBlock(
                [
                    PsComment("Loop body"),
                    PsAssignment(x, y),
                    PsAssignment(x, y),
                    PsPragma("#pragma clang loop vectorize(enable)"),
                    PsStatement(
                        PsMemAcc(PsCast(Ptr(Fp(32)), z), one_i)
                        + PsCast(
                            Fp(32), PsSubscript(m, (one_i + one_i + one_i, b + one_i))
                        )
                    ),
                ]
            ),
        ),
    ]:
        ast = typify(ast)
        ast_clone = ast.clone()
        check(ast, ast_clone)


def test_children_leaves():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    #   Empty children
    for node in [
        factory.parse_sympy(sp.Symbol("x")),
        factory.parse_sympy(sp.Integer(14)),
        PsPragma("not a pragma"),
        PsComment("welcome to pystencils"),
    ]:
        assert not node.children


def test_children_api_expressions():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    x, y, z, w = sp.symbols("x, y, z, w")

    for op in [add, sub, mul]:
        expr: PsBinOp = factory.parse_sympy(op(x, y))
        c1, c2 = expr.children
        assert c1.structurally_equal(factory.parse_sympy(x))
        assert c2.structurally_equal(factory.parse_sympy(y))

        assert c1 == expr.operand1
        assert c2 == expr.operand2

        new_children = (factory.parse_sympy(z), factory.parse_sympy(w))
        expr.children = new_children
        assert expr.children == new_children
        assert (expr.operand1, expr.operand2) == new_children

        with pytest.raises(IndexError):
            expr.children = (factory.parse_sympy(z),) * 3

    #   Children of piecewise
    expr = factory.parse_sympy(sp.Piecewise((y, x < 0), (z, True)))
    c1, c2, c3 = expr.children

    assert isinstance(expr, PsTernary)
    assert c1.structurally_equal(factory.parse_sympy(x < 0))
    assert c2.structurally_equal(factory.parse_sympy(y))
    assert c3.structurally_equal(factory.parse_sympy(z))

    assert c1 == expr.condition
    assert c2 == expr.case_then
    assert c3 == expr.case_else

    new_children = (
        factory.parse_sympy(x >= 0),
        factory.parse_sympy(z),
        factory.parse_sympy(w),
    )
    expr.children = new_children
    assert expr.children == new_children
    assert (expr.condition, expr.case_then, expr.case_else) == new_children


def test_children_api_structural():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    x, y, z = sp.symbols("x, y, z")
    i, j, k, m, n = [TypedSymbol(name, ctx.index_dtype) for name in "ijkmn"]

    b_then = PsBlock([factory.parse_sympy(Assignment(y, 0))])
    b_else = PsBlock([factory.parse_sympy(Assignment(y, 1))])
    branch = PsConditional(factory.parse_sympy(x < 0), b_then, b_else)

    cond, b1, b2 = branch.children
    assert cond == branch.condition
    assert b1 == b_then
    assert b2 == b_else

    branch.children = (cond, b_else, b_then)
    assert branch.children == (cond, b_else, b_then)
    assert branch.branch_true == b_else
    assert branch.branch_false == b_then

    assert b_then.children == tuple(b_then.statements)

    loop = PsLoop(
        factory.parse_index(i),
        factory.parse_index(0),
        factory.parse_index(m),
        factory.parse_index(1),
        b_then,
    )

    assert loop.children == (loop.counter, loop.start, loop.stop, loop.step, loop.body)

    new_children = (
        factory.parse_index(j),
        factory.parse_index(2),
        factory.parse_index(n),
        factory.parse_index(3),
        b_else,
    )
    loop.children = new_children

    assert loop.children == new_children
    assert (loop.counter, loop.start, loop.stop, loop.step, loop.body) == new_children


def test_children_api_axes():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    x, y, z = sp.symbols("x, y, z")
    i, j, k, m, n = [TypedSymbol(name, ctx.index_dtype) for name in "ijkmn"]

    rang = factory.axis_range(i, make_slice[j:k:1])
    assert rang.children == (rang.counter, rang.start, rang.stop, rang.step)

    new_children = (
        factory.parse_index(j),
        factory.parse_index(0),
        factory.parse_index(m),
        factory.parse_index(2),
    )
    rang.children = new_children
    assert rang.children == new_children

    with pytest.raises(TypeError):
        rang.set_child(0, factory.parse_index(3 + i))

    with pytest.raises(TypeError):
        rang.set_child(0, PsBlock([]))

    cube = factory.axes_cube(
        (i, j, k),
        make_slice[0:n, 0:m, 0:3],
        PsBlock([factory.parse_sympy(Assignment(x, y + z))]),
    )

    for c in range(3):
        assert cube.children[c] == cube.ranges[c]

    assert cube.children[0].structurally_equal(factory.axis_range(i, make_slice[0:n]))
    assert cube.children[1].structurally_equal(factory.axis_range(j, make_slice[0:m]))
    assert cube.children[2].structurally_equal(factory.axis_range(k, make_slice[0:3]))
    assert cube.children[3] == cube.body

    new_children = (
        factory.axis_range(i, make_slice[0:3:1]),
        factory.axis_range(j, make_slice[0:3:1]),
        factory.axis_range(k, make_slice[0:3:1]),
        PsBlock([factory.parse_sympy(Assignment(x, y - z))]),
    )
    cube.children = new_children
    assert cube.children == new_children

    loop = PsLoopAxis(
        factory.axis_range(i, make_slice[j:k:1]),
        PsBlock([factory.parse_sympy(Assignment(x, y + z))]),
    )

    assert loop.children == (loop.range, loop.body)

    new_children = (
        factory.axis_range(k, make_slice[-1:3:1]),
        PsBlock([factory.parse_sympy(Assignment(x, y - z))]),
    )
    loop.children = new_children
    assert loop.children == new_children


def test_buffer_acc():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    from pystencils import fields

    f, g = fields("f, g(3): [2D]")
    a, b = [ctx.get_symbol(n, ctx.index_dtype) for n in "ab"]

    f_buf = ctx.get_buffer(f)

    f_acc = PsBufferAcc(
        f_buf.base_pointer,
        [PsExpression.make(i) for i in (a, b)] + [factory.parse_index(0)],
    )
    assert f_acc.buffer == f_buf
    assert f_acc.base_pointer.structurally_equal(PsSymbolExpr(f_buf.base_pointer))

    f_acc_clone = f_acc.clone()
    assert f_acc_clone is not f_acc

    assert f_acc_clone.buffer == f_buf
    assert f_acc_clone.base_pointer.structurally_equal(PsSymbolExpr(f_buf.base_pointer))
    assert len(f_acc_clone.index) == 3
    assert f_acc_clone.index[0].structurally_equal(PsSymbolExpr(ctx.get_symbol("a")))
    assert f_acc_clone.index[1].structurally_equal(PsSymbolExpr(ctx.get_symbol("b")))

    g_buf = ctx.get_buffer(g)

    g_acc = PsBufferAcc(
        g_buf.base_pointer,
        [PsExpression.make(i) for i in (a, b)] + [factory.parse_index(2)],
    )
    assert g_acc.buffer == g_buf
    assert g_acc.base_pointer.structurally_equal(PsSymbolExpr(g_buf.base_pointer))

    second_bptr = PsExpression.make(
        ctx.get_symbol("data_g_interior", g_buf.base_pointer.dtype)
    )
    second_bptr.symbol.add_property(BufferBasePtr(g_buf))
    g_acc.base_pointer = second_bptr

    assert g_acc.base_pointer == second_bptr
    assert g_acc.buffer == g_buf

    #   cannot change base pointer to different buffer
    with pytest.raises(ValueError):
        g_acc.base_pointer = PsExpression.make(f_buf.base_pointer)


def test_undefined_vars():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    x, y, z = sp.symbols("x, y, z")
    i, j, k = [TypedSymbol(name, ctx.index_dtype) for name in "ijk"]
    ptr = TypedSymbol("ptr", "float64 *")

    x_, y_, z_ = (ctx.get_symbol(name, ctx.default_dtype) for name in "xyz")
    i_, j_, k_ = (ctx.get_symbol(name, ctx.index_dtype) for name in "ijk")
    ptr_ = ctx.get_symbol(ptr.name, ptr.dtype)

    for ast, expected in [
        (factory.parse_sympy(x + y - z), {x_, y_, z_}),
        (factory.parse_sympy(sp.Integer(23)), set()),
        (
            PsBlock(
                [PsDeclaration(factory.parse_sympy(x), factory.parse_sympy(y - z))]
            ),
            {y_, z_},
        ),
        (
            PsBlock(
                [
                    PsDeclaration(factory.parse_sympy(y), factory.parse_sympy(z + 1)),
                    PsDeclaration(factory.parse_sympy(x), factory.parse_sympy(y - z)),
                ]
            ),
            {z_},
        ),
        (
            PsBlock(
                [
                    PsDeclaration(factory.parse_sympy(y), factory.parse_sympy(z + 1)),
                    PsAssignment(
                        factory.parse_sympy(mem_acc(ptr, i + j)), factory.parse_sympy(y)
                    ),
                ]
            ),
            {ptr_, i_, j_, z_},
        ),
        (
            factory.loop(
                "k",
                make_slice[0:j:2],
                PsBlock(
                    [
                        PsAssignment(
                            factory.parse_sympy(mem_acc(ptr, k)), factory.parse_sympy(y)
                        ),
                    ]
                ),
            ),
            {j_, ptr_, y_},
        ),
        (
            factory.axes_cube(
                (i, j),
                make_slice[0:k:2, i : i + 4 : 2],
                PsBlock(
                    [
                        PsDeclaration(
                            factory.parse_sympy(y), factory.parse_sympy(z + 1)
                        ),
                        PsAssignment(
                            factory.parse_sympy(mem_acc(ptr, i + 2 * j)),
                            factory.parse_sympy(y),
                        ),
                    ]
                ),
            ),
            {k_, z_, ptr_},
        ),
        (
            PsLoopAxis(
                factory.axis_range(j, make_slice[0:14:2]),
                PsBlock(
                    [
                        PsAssignment(
                            factory.parse_sympy(mem_acc(ptr, i + 2 * j)),
                            factory.parse_sympy(y),
                        ),
                    ]
                ),
            ),
            {i_, ptr_, y_},
        ),
    ]:
        undefs = collect_undefined_symbols(ast)
        assert undefs == expected
