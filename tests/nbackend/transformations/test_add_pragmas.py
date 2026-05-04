import sympy as sp
from itertools import product

from pystencils import make_slice, fields, Assignment
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    FullIterationSpace,
)

from pystencils.backend.ast import dfs_preorder
from pystencils.backend.ast.structural import (
    PsBlock,
    PsPragma,
    PsLoop,
    PsComment,
    PsConditional,
)
from pystencils.backend.transformations import InsertPragmasAtLoops, LoopPragma


def test_insert_pragmas():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    f, g = fields("f, g: [3D]")
    ispace = FullIterationSpace.create_from_slice(
        ctx, make_slice[:, :, :], archetype_field=f
    )
    ctx.set_iteration_space(ispace)

    stencil = list(product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
    loop_body = PsBlock(
        [factory.parse_sympy(Assignment(f.center(0), sum(g.neighbors(stencil))))]
    )
    loops = factory.loops_from_ispace(ispace, loop_body)

    pragmas = (
        LoopPragma("omp parallel for", 0),
        LoopPragma("some nonsense pragma", 1),
        LoopPragma("omp simd", -1),
    )
    add_pragmas = InsertPragmasAtLoops(ctx, pragmas)
    ast = add_pragmas(loops)

    assert isinstance(ast, PsBlock)

    first_pragma = ast.statements[0]
    assert isinstance(first_pragma, PsPragma)
    assert first_pragma.text == pragmas[0].text

    assert ast.statements[1] == loops
    second_pragma = loops.body.statements[0]
    assert isinstance(second_pragma, PsPragma)
    assert second_pragma.text == pragmas[1].text

    second_loop = list(dfs_preorder(ast, lambda node: isinstance(node, PsLoop)))[1]
    assert isinstance(second_loop, PsLoop)
    third_pragma = second_loop.body.statements[0]
    assert isinstance(third_pragma, PsPragma)
    assert third_pragma.text == pragmas[2].text


def test_insert_pragmas_versioned_loops():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    loop_body_1 = PsBlock([PsComment("Loop body 1")])
    loop_body_2 = PsBlock([PsComment("Loop body 2")])

    iloop_1 = factory.loop("i1", make_slice[1:17], loop_body_1)
    iloop_2 = factory.loop("i2", make_slice[3:21], loop_body_2)

    cond = PsConditional(
        factory.parse_sympy(sp.Symbol("x") < sp.Symbol("y")),
        PsBlock([iloop_1]),
        PsBlock([iloop_2]),
    )
    jloop = factory.loop("j", make_slice[5:19], PsBlock([cond]))

    pragma = LoopPragma("omp simd", -1)
    add_pragmas = InsertPragmasAtLoops(ctx, (pragma,))
    _ = add_pragmas(jloop)

    p1 = cond.branch_true.statements[0]
    assert isinstance(p1, PsPragma) and p1.text == pragma.text

    assert cond.branch_false is not None
    p2 = cond.branch_false.statements[0]
    assert isinstance(p2, PsPragma) and p2.text == pragma.text
