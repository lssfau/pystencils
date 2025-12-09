#  type: ignore
import sympy as sp

from pystencils import (
    Field,
    Assignment,
    AddAugmentedAssignment,
    make_slice,
    DEFAULTS,
    TypedSymbol,
)

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    FullIterationSpace,
)
from pystencils.backend.transformations import CanonicalizeSymbols
from pystencils.backend.ast.structural import PsConditional, PsBlock
from pystencils.backend.ast.expressions import PsSymbolExpr
from pystencils.backend.ast import dfs_preorder
from pystencils.backend.ast.axes import PsLoopAxis


def test_deduplication():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    canonicalize = CanonicalizeSymbols(ctx)

    f = Field.create_fixed_size("f", (5, 5), memory_strides=(5, 1))
    x, y, z = sp.symbols("x, y, z")

    ispace = FullIterationSpace.create_from_slice(ctx, make_slice[:, :], f)
    ctx.set_iteration_space(ispace)

    ctr_1 = DEFAULTS.spatial_counters[1]

    then_branch = PsBlock(
        [
            factory.parse_sympy(Assignment(x, y)),
            factory.parse_sympy(Assignment(f.center(0), x)),
        ]
    )

    else_branch = PsBlock(
        [
            factory.parse_sympy(Assignment(x, z)),
            factory.parse_sympy(Assignment(f.center(0), x)),
        ]
    )

    ast = PsConditional(
        factory.parse_sympy(ctr_1),
        then_branch,
        else_branch,
    )

    ast = factory.loops_from_ispace(ispace, PsBlock([ast]))

    ast = canonicalize(ast)

    assert canonicalize.get_last_live_symbols() == {
        ctx.find_symbol("y"),
        ctx.find_symbol("z"),
        ctx.get_buffer(f).base_pointer,
    }

    assert ctx.find_symbol("x") is not None
    assert ctx.find_symbol("x__0") is not None

    assert then_branch.statements[0].declared_symbol.name == "x__0"
    assert then_branch.statements[1].rhs.symbol.name == "x__0"

    assert else_branch.statements[0].declared_symbol.name == "x"
    assert else_branch.statements[1].rhs.symbol.name == "x"

    assert ctx.find_symbol("x").dtype.const
    assert ctx.find_symbol("x__0").dtype.const
    assert ctx.find_symbol("y").dtype.const
    assert ctx.find_symbol("z").dtype.const


def test_do_not_constify():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    canonicalize = CanonicalizeSymbols(ctx)

    x, z = sp.symbols("x, z")

    ast = factory.loop(
        "i",
        make_slice[:10],
        PsBlock(
            [
                factory.parse_sympy(Assignment(x, z)),
                factory.parse_sympy(AddAugmentedAssignment(z, 1)),
            ]
        ),
    )

    ast = canonicalize(ast)

    assert ctx.find_symbol("x").dtype.const
    assert not ctx.find_symbol("z").dtype.const


def test_loop_counters():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    canonicalize = CanonicalizeSymbols(ctx)

    f = Field.create_generic("f", 2, index_shape=(1,))
    g = Field.create_generic("g", 2, index_shape=(1,))
    ispace = FullIterationSpace.create_from_slice(
        ctx, make_slice[:, :], archetype_field=f
    )
    ctx.set_iteration_space(ispace)

    asm = Assignment(f.center(0), 2 * g.center(0))

    body = PsBlock([factory.parse_sympy(asm)])

    loops = factory.loops_from_ispace(ispace, body)

    loops_clone = loops.clone()
    loops_clone2 = loops.clone()

    ast = PsBlock([loops, loops_clone, loops_clone2])

    ast = canonicalize(ast)

    assert loops_clone2.counter.symbol.name == "ctr_0"
    assert not loops_clone2.counter.symbol.get_dtype().const
    assert loops_clone.counter.symbol.name == "ctr_0__0"
    assert not loops_clone.counter.symbol.get_dtype().const
    assert loops.counter.symbol.name == "ctr_0__1"
    assert not loops.counter.symbol.get_dtype().const


def test_axes():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    canonicalize = CanonicalizeSymbols(ctx)

    x, y, z = sp.symbols("x, y, z")
    i, j, k, m, n = [TypedSymbol(name, ctx.index_dtype) for name in "ijkmn"]

    cube = factory.axes_cube(
        (i, j, k),
        make_slice[0:m, 0:i, i:j],
        PsBlock(
            [
                PsLoopAxis(
                    factory.axis_range(n, make_slice[i : j + k]),
                    PsBlock([factory.parse_sympy(Assignment(x, y + z))]),
                )
            ]
        ),
    )

    cube_clone = cube.clone()
    cube_clone2 = cube.clone()

    ast = PsBlock([cube, cube_clone, cube_clone2])
    canonicalize(ast)

    for node0, node1, node2 in zip(
        dfs_preorder(ast.statements[0]),
        dfs_preorder(ast.statements[1]),
        dfs_preorder(ast.statements[2]),
        strict=True,
    ):
        assert type(node0) is type(node1) and type(node0) is type(node2)

        if isinstance(node2, PsSymbolExpr):
            if node2.symbol.name in ("i", "j", "k", "n", "x"):
                assert node1.symbol.name == (node2.symbol.name + "__0")
                assert node0.symbol.name == (node2.symbol.name + "__1")
            else:
                assert node0.symbol == node1.symbol
                assert node0.symbol == node2.symbol
