import pytest

import sympy as sp
from pystencils.flow import block, tie, cases
from pystencils import fields, Field

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    FreezeFlowgraph,
    FreezeExpressions,
    create_full_iteration_space,
    AstFactory,
)
from pystencils.backend.ast.expressions import PsUndefined
from pystencils.backend.ast.structural import (
    PsBlock,
    PsDeclaration,
    PsAssignment,
    PsConditional,
    PsComment,
)
from pystencils.backend.kernelcreation import Typifier


def test_single_block():
    x, y, z, w = sp.symbols("x, y, z, w")
    f, g = fields("f(3), g(1): [2D]")

    @block
    def b(let):
        let[x] = w + f(2)
        let[y] = w + 2
        let.export[z] = y + x - 1
        let.store[f(0)] = x
        let.store[f(1)] = y
        let.store[g(0)] = z

    graph = tie(b)

    ctx = KernelCreationContext()

    for field in graph.fields_read | graph.fields_written:
        ctx.add_field(field)

    freeze_graph = FreezeFlowgraph(ctx)

    ispace = create_full_iteration_space(ctx, ghost_layers=0)
    ctx.set_iteration_space(ispace)

    ast = freeze_graph(graph)

    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    assert isinstance(ast, PsBlock)

    assert isinstance(ast.statements[0], PsComment)
    assert isinstance(ast.statements[-1], PsComment)

    statements = ast.statements[1:-1]

    assert len(statements) == 6

    assert isinstance(statements[0], PsDeclaration)
    assert statements[0].lhs.symbol == ctx.get_symbol("x")
    assert statements[0].rhs.structurally_equal(typify(freeze(w + f(2))))

    assert isinstance(statements[1], PsDeclaration)
    assert statements[1].lhs.symbol == ctx.get_symbol("y")
    assert statements[1].rhs.structurally_equal(typify(freeze(w + 2)))

    assert isinstance(statements[2], PsDeclaration)
    assert statements[2].lhs.symbol == ctx.get_symbol("z")
    assert statements[2].rhs.structurally_equal(typify(freeze(y + x - 1)))

    assert isinstance(statements[3], PsAssignment)
    assert statements[3].lhs.buffer == ctx.get_buffer(f)
    assert statements[3].rhs.structurally_equal(typify(freeze(x)))

    assert isinstance(statements[4], PsAssignment)
    assert statements[4].lhs.buffer == ctx.get_buffer(f)
    assert statements[4].rhs.structurally_equal(typify(freeze(y)))

    assert isinstance(statements[5], PsAssignment)
    assert statements[5].lhs.buffer == ctx.get_buffer(g)
    assert statements[5].rhs.structurally_equal(typify(freeze(z)))


def test_diamond_graph():
    x, y, z, v, w = sp.symbols("x, y, z, v, w")
    f = fields("f: [2D]")

    @block
    def block1(let):
        let[x] = v + 1
        let.export[y] = x + 1

    @block(preds=[block1])
    def block2(let):
        let[x] = y + 1
        let.export[z] = x + 1

    @block(preds=[block1])
    def block3(let):
        let[x] = y + 2
        let.export[w] = x + 1

    @block(preds=[block2, block3])
    def block4(let):
        let[x] = z + w
        let.store[f()] = x

    graph = tie(block4)

    ctx = KernelCreationContext()

    for field in graph.fields_read | graph.fields_written:
        ctx.add_field(field)

    ispace = create_full_iteration_space(ctx, ghost_layers=0)
    ctx.set_iteration_space(ispace)

    freeze_graph = FreezeFlowgraph(ctx)
    ast = freeze_graph(graph)

    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    assert isinstance(ast, PsBlock)

    x0, x1, x2 = sp.symbols("x__0, x__1, x__2")
    expected_decls = [
        (x, v + 1),
        (y, x + 1),
        (x0, y + 1),
        (z, x0 + 1),
        (x1, y + 2),
        (w, x1 + 1),
        (x2, z + w),
    ]

    statements = [s for s in ast.statements if not isinstance(s, PsComment)]
    assert len(statements) == 8

    for decl, (lhs, rhs) in zip(statements, expected_decls):
        assert isinstance(decl, PsDeclaration)
        assert decl.lhs.symbol == ctx.get_symbol(lhs.name)
        assert decl.rhs.structurally_equal(typify(freeze(rhs)))

    assert isinstance(statements[7], PsAssignment)
    assert statements[7].lhs.buffer == ctx.get_buffer(f)
    assert statements[7].rhs.structurally_equal(typify(freeze(x2)))


def test_parallel_graph():
    x, y, z, v, w = sp.symbols("x, y, z, v, w")
    f, g = fields("f, g: [2D]")

    @block
    def block1a(let):
        let[x] = v + 1
        let.export[y] = x + 1

    @block(preds=[block1a])
    def block1b(let):
        let[z] = y + 1
        let.store[f()] = z + 1

    @block
    def block2a(let):
        let[x] = v + 2
        let.export[y] = x + 2

    @block(preds=[block2a])
    def block2b(let):
        let[z] = y + 2
        let.store[g()] = z + 2

    graph = tie(block1b, block2b)

    ctx = KernelCreationContext()

    for field in graph.fields_read | graph.fields_written:
        ctx.add_field(field)

    ispace = create_full_iteration_space(ctx, ghost_layers=0)
    ctx.set_iteration_space(ispace)

    freeze_graph = FreezeFlowgraph(ctx)
    ast = freeze_graph(graph)

    x0, y0, z0 = sp.symbols("x__0, y__0, z__0")
    expected_asms = [
        (x, v + 1),
        (y, x + 1),
        (z, y + 1),
        (f(), z + 1),
        (x0, v + 2),
        (y0, x0 + 2),
        (z0, y0 + 2),
        (g(), z0 + 2),
    ]
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    statements = [s for s in ast.statements if not isinstance(s, PsComment)]

    for asm, (lhs, rhs) in zip(statements, expected_asms, strict=True):
        match lhs:
            case Field.Access():
                assert isinstance(asm, PsAssignment)
                assert asm.lhs.buffer == ctx.get_buffer(lhs.field)
            case _:
                assert isinstance(asm, PsDeclaration)
                assert asm.lhs.symbol == ctx.get_symbol(lhs.name)

        assert asm.rhs.structurally_equal(typify(freeze(rhs)))


def test_skewed_diamond_graph():
    x, y, z, v, w = sp.symbols("x, y, z, v, w")
    f = fields("f: [2D]")

    @block
    def block1(let):
        let.export[x] = v + 1

    @block(preds=[block1])
    def block2(let):
        let.export[y] = x + 1

    @block(preds=[block2])
    def block3(let):
        let.export[z] = y + v

    @block(preds=[block2, block3])
    def block4(let):
        let.store[f()] = y + z

    graph = tie(block4)

    ctx = KernelCreationContext()

    for field in graph.fields_read | graph.fields_written:
        ctx.add_field(field)

    ispace = create_full_iteration_space(ctx, ghost_layers=0)
    ctx.set_iteration_space(ispace)

    freeze_graph = FreezeFlowgraph(ctx)
    ast = freeze_graph(graph)

    expected_asms = [(x, v + 1), (y, x + 1), (z, y + v), (f(), y + z)]

    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    statements = [s for s in ast.statements if not isinstance(s, PsComment)]

    for asm, (lhs, rhs) in zip(statements, expected_asms, strict=True):
        match lhs:
            case Field.Access():
                assert isinstance(asm, PsAssignment)
                assert asm.lhs.buffer == ctx.get_buffer(lhs.field)
            case _:
                assert isinstance(asm, PsDeclaration)
                assert asm.lhs.symbol == ctx.get_symbol(lhs.name)

        assert asm.rhs.structurally_equal(typify(freeze(rhs)))


def test_freeze_cases():
    x, y, z, v, w = sp.symbols("x, y, z, v, w")
    f, g = fields("f, g: [2D]")

    @cases
    def cases1(_cs):
        @_cs.case(x > 0)
        def branch1(_eq):
            _eq.export[y] = 0

        @_cs.case(sp.Eq(x, 0))
        def branch2(_eq):
            _eq.export[y] = 1

        @_cs.case(x < 0)
        def branch3(_eq):
            _eq.export[y] = 2

    graph = tie(cases1)

    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    freeze_graph = FreezeFlowgraph(ctx)

    block = freeze_graph(graph)

    assert len(block.statements) == 4

    export_decl = block.statements[1]
    assert isinstance(export_decl, PsDeclaration)
    assert export_decl.lhs.symbol == ctx.get_symbol("y")
    assert isinstance(export_decl.rhs, PsUndefined)

    outer_cond = block.statements[2]
    assert isinstance(outer_cond, PsConditional)
    assert outer_cond.condition.structurally_equal(factory.parse_sympy(x > 0))

    export_y_0 = outer_cond.branch_true.statements[-1]
    assert isinstance(export_y_0, PsAssignment)
    assert export_y_0.rhs.symbol == ctx.get_symbol("y__0")
    assert export_y_0.lhs.symbol == ctx.get_symbol("y")

    inner_cond = outer_cond.branch_false.statements[0]
    assert isinstance(inner_cond, PsConditional)
    assert inner_cond.condition.structurally_equal(factory.parse_sympy(sp.Eq(x, 0)))

    export_y_1 = inner_cond.branch_true.statements[-1]
    assert isinstance(export_y_1, PsAssignment)
    assert export_y_1.rhs.symbol == ctx.get_symbol("y__1")
    assert export_y_1.lhs.symbol == ctx.get_symbol("y")

    export_y_2 = inner_cond.branch_false.statements[-1]
    assert isinstance(export_y_2, PsAssignment)
    assert export_y_2.rhs.symbol == ctx.get_symbol("y__2")
    assert export_y_2.lhs.symbol == ctx.get_symbol("y")
