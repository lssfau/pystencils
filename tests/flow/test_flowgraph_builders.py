import pytest

import sympy as sp
import pystencils as ps

from pystencils import TypedSymbol, fields
from pystencils.sympyextensions.reduction import ReductionOp
from pystencils.flow import block, tie, cases
from pystencils.flow.flowgraph import (
    EquationsBlock,
    Let,
    Export,
    Store,
    Reduce,
    Cases,
    Flowgraph,
    Bottom,
)

from pystencils.flow.canonicalize_flowgraph import CanonicalizationError


def test_block():
    x, y, z, v, w = sp.symbols("x, y, z, v, w")
    f, g = fields("f(3), g(1) : [2D]")
    q = ps.TypedSymbol("q", "float32")

    @block
    def block1(let):
        let[x] = w + 1
        let.let[v] = x / 3
        let.export[y] = x
        let.store[f(0)] = w + 2
        let["z":"float32"] = x / 2
        let.reduce["p":"float32", "+"] = z - 1
        let.reduce[q, "*"] = z + 2

    assert isinstance(block1, EquationsBlock)
    assert block1.assignments == (
        Let(x, w + 1),
        Let(v, x / 3),
        Export(y, x),
        Store(f(0), w + 2),
        Let(TypedSymbol("z", "float32"), x / 2),
        Reduce(TypedSymbol("p", "float32"), z - 1, ReductionOp.Add),
        Reduce(q, z + 2, ReductionOp.Mul),
    )


def test_invalid_setters():
    x, y, z, w = sp.symbols("x, y, z, w")

    with pytest.raises(ValueError) as exc:

        @ps.flow.block
        def invaid_reduce(_b):
            _b.reduce[x, "+"] = y

    assert "Can only reduce onto typed symbols" in str(exc.value)

    with pytest.raises(ValueError) as exc:

        @ps.flow.block
        def invaid_reduce2(_b):
            _b.reduce["x", "+"] = y

    assert "Can only reduce onto typed symbols" in str(exc.value)

    with pytest.raises(ValueError) as exc:

        @ps.flow.block
        def invaid_reduce3(_b):
            _b.reduce["x":"float32", "invalid-op"] = y

    with pytest.raises(ValueError) as exc:

        @ps.flow.block
        def duplicate_assignment(_b):
            _b.let[x] = y
            _b.let[x] = z

    assert "Duplicate assignment left-hand side" in str(exc.value)

    with pytest.raises(ValueError) as exc:

        @ps.flow.block
        def duplicate_assignment2(_b):
            _b.let[x] = y
            _b.export[x] = z

    assert "Duplicate assignment left-hand side" in str(exc.value)


def test_multi_assignment():
    x, y, z, v, w = sp.symbols("x, y, z, v, w")
    f = ps.grids.TensorField("f", 3, (3,))
    g = ps.grids.TensorField("g", 3, ())
    h = ps.grids.TensorField("h", 3, (3,))
    r = ps.symbols("r", "float32")
    arr1, arr2 = ps.symbols(
        "arr1, arr2",
        ps.types.PsNamedArrayType(
            "std::array< float64, 3 >", ps.create_type("float64"), (3,)
        ),
    )

    @ps.flow.block
    def block1(_eq):
        _eq[x, y, z] = 1, 2, 3
        _eq.export[v, "r":"float32"] = sp.ImmutableMatrix([4, w + 1])
        _eq.store[f(1), g()] = x, y + f(0)
        #   LHS is symbol, RHS is tuple -> create an array
        _eq.let[arr1] = 4, 5, 6
        _eq.export[arr2] = 7, 8, 9

        with pytest.raises(TypeError):
            #   RHS not iterable
            _eq["s1", "s2"] = x

        with pytest.raises(TypeError):
            #   RHS not iterable
            _eq[h(0), h(1)] = x

        with pytest.raises(ValueError):
            #   Too few items on RHS
            _eq.store[h(0), h(1), h(2)] = sp.Matrix([x, y])

    assert block1 == EquationsBlock(
        [
            Let(x, 1),
            Let(y, 2),
            Let(z, 3),
            Export(v, 4),
            Export(r, w + 1),
            Store(f(1), x),
            Store(g(), y + f(0)),
            Let(arr1, sp.Tuple(4, 5, 6)),
            Export(arr2, sp.Tuple(7, 8, 9)),
        ],
        predecessors=[],
        name="block1",
    )


def test_predecessors():
    x, y, z, w = sp.symbols("x, y, z, w")
    f, g = fields("f(3), g(1) : [2D]")

    @block
    def block1(let):
        let[x] = w + 1
        let.export[y] = x

    @block
    def block2(let):
        let[x] = w + 2
        let.export[z] = x

    @block(preds=[block1, block2])
    def block3(let):
        let.store[f(0)] = y
        let.store[f(1)] = z

    assert block3.predecessors == frozenset({block1, block2})

    @block(preds=[block1, block1])
    def block4(let):
        let.store[f(0)] = y

    assert block4.predecessors == frozenset({block1})


def test_cases():
    x, y, z, w = sp.symbols("x, y, z, w")
    f = ps.fields("f: [2D]")

    @cases
    def cs_block(cs):
        # Cases from decorated function(s)
        @cs.case(x < y)
        def _(let):
            let[z] = y - x
            let.export[w] = z

        @cs.case(x > y)
        def _(let):
            let[z] = x - y
            let.export[w] = z

        # Case from subgraph
        @block
        def subgraph(let):
            let.export[w] = sp.Integer(0)

        cs.case(sp.Eq(x, y), tie(subgraph))

    sg0, sg1, sg2 = cs_block.subgraphs

    assert isinstance(sg0, Flowgraph)
    assert isinstance(sg1, Flowgraph)
    assert isinstance(sg2, Flowgraph)
    assert list(sg0.bottom.predecessors)[0].assignments == (
        Let(z, y - x),
        Export(w, z),
    )
    assert list(sg1.bottom.predecessors)[0].assignments == (
        Let(z, x - y),
        Export(w, z),
    )
    assert list(sg2.bottom.predecessors)[0].assignments == (Export(w, sp.Integer(0)),)

    with pytest.raises(ValueError) as exc:

        @cases
        def cs_block(cs):
            # Case from subgraph without tie
            @block
            def subgraph(let):
                let.store[f()] = sp.Integer(0)

            cs.case(sp.Eq(x, y), subgraph)

    assert "Subgraph passed to `case` was not a valid flowgraph" in str(exc.value)


def test_cases_predecessors():
    x, y, z, w = sp.symbols("x, y, z, w")

    @cases
    def cs_block0(cs):
        @cs.case(x < y - 1)
        def _(let):
            let[z] = y - x + 1
            let.export[w] = z

        @cs.case(x >= y - 1)
        def _(let):
            let[z] = y - x - 1
            let.export[w] = z

    # Cases with predecessors
    @cases(preds=([cs_block0]))
    def cs_block1(cs):
        @cs.case(x < y)
        def _(let):
            let[z] = y - x
            let.export[w] = z

        @cs.case(x >= y)
        def _(let):
            let[z] = x - y
            let.export[w] = z

    assert isinstance(cs_block1, Cases)
    assert cs_block1.conditions == (x < y, sp.true)
    assert cs_block1.free_symbols == {x, y}
    assert cs_block1.exports == {w}
    assert len(cs_block1.predecessors) == 1

    (pred_csblock0,) = list(cs_block1.predecessors)

    assert isinstance(pred_csblock0, Cases)
    assert pred_csblock0.conditions == (x < y - 1, True)
    assert pred_csblock0.free_symbols == {x, y}
    assert pred_csblock0.exports == {w}
    assert pred_csblock0.predecessors == frozenset()
    assert len(pred_csblock0.subgraphs) == 2
    case00, case01 = pred_csblock0.subgraphs

    assert isinstance(case00, Flowgraph)
    assert case00.free_symbols == {x, y}
    assert case00.exports == pred_csblock0.exports
    assert list(case00.bottom.predecessors)[0].assignments == (
        Let(z, y - x + 1),
        Export(w, z),
    )

    assert isinstance(case01, Flowgraph)
    assert case01.free_symbols == {x, y}
    assert case01.exports == pred_csblock0.exports
    assert list(case01.bottom.predecessors)[0].assignments == (
        Let(z, y - x - 1),
        Export(w, z),
    )


def test_tie():
    x, y, z = sp.symbols("x, y, z")
    f, g = ps.fields("f, g: [2D]")

    @ps.flow.block
    def block1(_b):
        _b.export[x] = y
        _b.store[f()] = x

    @ps.flow.block
    def block2(_b):
        _b.export[z] = 2
        _b.store[g()] = z

    graph = ps.flow.tie(block1, block2)

    assert isinstance(graph, Flowgraph)
    assert len(graph.bottom.predecessors) == 2

    b1, b2 = sorted(graph.bottom.predecessors, key=lambda n: n.name)

    assert isinstance(b1, EquationsBlock) and b1.assignments == block1.assignments
    assert isinstance(b2, EquationsBlock) and b2.assignments == block2.assignments

    assert graph.exports == {x, z}
    assert graph.free_symbols == set([y])
    assert graph.effects == {Store(f(), x), Store(g(), z)}

    bot = Bottom([block1])
    graph = ps.flow.tie(bot)

    assert len(graph.bottom.predecessors) == 1
    b1 = list(graph.bottom.predecessors)[0]
    assert isinstance(b1, EquationsBlock) and b1.assignments == block1.assignments


def test_tie_errors():
    x, y, z = sp.symbols("x, y, z")
    f, g = ps.fields("f, g: [2D]")

    #   Duplicate exports
    @ps.flow.block
    def block1(_b):
        _b.export[x] = y

    @ps.flow.block
    def block2(_b):
        _b.export[x] = 2

    with pytest.raises(CanonicalizationError):
        _ = ps.flow.tie(block1, block2)

    #   Duplicate effects
    @ps.flow.block
    def block3(_b):
        _b.store[f()] = y

    @ps.flow.block
    def block4(_b):
        _b.store[f()] = z

    with pytest.raises(CanonicalizationError) as exc:
        _ = ps.flow.tie(block3, block4)

    assert "Ambiguous side effect" in str(exc.value)

    #   Duplicate reductions
    @ps.flow.block
    def block5(_b):
        _b.reduce["x":"float64", "+"] = y

    @ps.flow.block
    def block6(_b):
        _b.reduce["x":"float64", "+"] = z

    with pytest.raises(CanonicalizationError) as exc:
        _ = ps.flow.tie(block5, block6)

    assert "Ambiguous side effect" in str(exc.value)

    #   Unexpected bottom node
    with pytest.raises(ValueError) as exc:
        _ = ps.flow.tie(block5, Bottom([]))

    assert "Cannot tie a Bottom node into a subgraph with other nodes" in str(exc.value)


def test_guarded_block():
    x, y, z = sp.symbols("x, y, z")
    f, g = ps.fields("f, g: [2D]")

    @ps.flow.guarded_block(x < 0)
    def myblock(_eq):
        _eq.store[f()] = x + g()

    assert isinstance(myblock, Cases)
    assert myblock.conditions == (x < 0,)
    assert myblock.subgraphs == (
        tie(
            EquationsBlock([Store(f(), x + g())], predecessors=[], name="myblock"),
            name="myblock",
        ),
    )
    assert not myblock.predecessors

    @ps.flow.block
    def block1(_eq):
        _eq.export[x] = y

    @ps.flow.guarded_block(x < 0, preds=[block1])
    def block2(_eq):
        _eq.store[f()] = x + g()

    assert isinstance(block2, Cases)
    assert block2.conditions == (x < 0,)
    assert block2.subgraphs == (
        tie(
            EquationsBlock([Store(f(), x + g())], predecessors=[], name="block2"),
            name="block2",
        ),
    )
    assert block2.predecessors == {block1}

    @ps.flow.block
    def block3(_eq):
        _eq.export[z] = 3

    @ps.flow.guarded_block(x < 0, preds=[block1])
    def block4(_eq):
        _eq.connect(block3)
        _eq.store[f()] = x + z

    assert isinstance(block4, Cases)
    assert block4.conditions == (x < 0,)
    assert block4.subgraphs == (
        tie(
            EquationsBlock([Store(f(), x + z)], predecessors=[], name="block4"),
            name="block4",
        ),
    )
    assert block4.predecessors == {block1, block3}

    with pytest.raises(ValueError):

        @ps.flow.guarded_block(x < 0)
        def invalid(_eq):
            _eq.export[z] = 2
