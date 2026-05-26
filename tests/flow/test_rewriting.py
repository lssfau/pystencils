import pytest

import sympy as sp
import pystencils as ps


def test_substitute_assignments():
    x, y, z, w = sp.symbols("x, y, z, w")
    q = ps.TypedSymbol("q", "float32")
    r = ps.TypedSymbol("r", "float32")
    f = ps.fields("f(3): [2D]")

    asm = ps.flow.Let(x, y + z)

    result = asm.subs({y: w + 3})
    assert result == ps.flow.Let(x, w + z + 3)

    result = asm.subs({x: w})
    assert result == ps.flow.Let(w, y + z)

    asm = ps.flow.Export(x, y + z)

    result = asm.subs({z: w, y: z - 1}, simultaneous=True)
    assert result == ps.flow.Export(x, w + z - 1)

    result = asm.subs({x: y, y: z, z: w}, simultaneous=True)
    assert result == ps.flow.Export(y, z + w)

    asm = ps.flow.Store(f(2), f(0) + f(1))

    result = asm.subs({f(0): f(1), f(2): f(0)}, simultaneous=True)
    assert result == ps.flow.Store(f(0), 2 * f(1))

    asm = ps.flow.Reduce(q, w + z, ps.sympyextensions.ReductionOp.Add)

    result = asm.subs({q: r, z: x + 2}, simultaneous=True)
    assert result == ps.flow.Reduce(r, w + x + 2, ps.sympyextensions.ReductionOp.Add)


def test_invalid_assignment_substitutions():
    x, y, z, w = sp.symbols("x, y, z, w")
    q = ps.TypedSymbol("q", "float32")
    f = ps.fields("f(3): [2D]")

    asm = ps.flow.Let(x, y + z)

    with pytest.raises(TypeError):
        asm.subs({x: f(2)})

    asm = ps.flow.Export(x, y + z)

    with pytest.raises(TypeError):
        asm.subs({x: f(2)})

    asm = ps.flow.Store(f(0), f(1))
    with pytest.raises(TypeError):
        asm.subs({f(0): q})

    asm = ps.flow.Reduce(q, w + z, ps.sympyextensions.ReductionOp.Add)
    with pytest.raises(TypeError):
        asm.subs({q: f(1)})
    with pytest.raises(TypeError):
        asm.subs({q: x})


def test_substitute_in_block():
    x, y, z, w = sp.symbols("x, y, z, w")
    foo = sp.Function("foo")
    bar = sp.Function("bar")

    @ps.flow.block
    def pred(_eq):
        _eq.export[w] = 3.4

    @ps.flow.block(name="test")
    def block(_eq):
        _eq.connect(pred)
        _eq.let[x] = bar(foo(y))
        _eq.export[z] = x - w

    result = block.subs({x: w, foo(y): foo(w), w: bar(x)}, simultaneous=True)

    @ps.flow.block(name="test")
    def expected(_eq):
        _eq.connect(pred)
        _eq.let[x] = bar(foo(w))
        _eq.export[z] = x - bar(x)

    assert result == expected


def test_substitute_in_cases():
    x, y, z, w = sp.symbols("x, y, z, w")
    f = ps.fields("f(3): [2D]")
    foo = sp.Function("foo")
    bar = sp.Function("bar")

    @ps.flow.block
    def pred(_eq):
        _eq.export[x] = 3.4

    @ps.flow.cases(name="test", preds=[pred])
    def distinction(_cs):
        @_cs.case(foo() > 0)
        def case1(_eq):
            _eq.export[w] = f(0) + f(1) + f(2)

        @_cs.case(bar(x) > 0)
        def case2(_eq):
            _eq.export[w] = foo() - bar(x)

        @_cs.case(True)
        def case3(_eq):
            _eq.export[w] = bar(y)

    result = distinction.subs(
        {foo(): x / 2, bar(y): f(2), f(2): z, w: f(0)}, simultaneous=True
    )

    @ps.flow.cases(name="test", preds=[pred])
    def expected(_cs):
        @_cs.case(x / 2 > 0)
        def case1(_eq):
            _eq.export[w] = f(0) + f(1) + z

        @_cs.case(bar(x) > 0)
        def case2(_eq):
            _eq.export[w] = x / 2 - bar(x)

        @_cs.case(True)
        def case3(_eq):
            _eq.export[w] = f(2)

    assert result == expected


def test_flowgraph_substitutions():
    x, y, z, w = sp.symbols("x, y, z, w")
    f = ps.fields("f(3): [2D]")
    foo = sp.Function("foo")
    bar = sp.Function("bar")

    @ps.flow.block
    def block1(_eq):
        _eq.export[x] = foo(z)
        _eq.export[y] = bar(w) + x

    @ps.flow.block(preds=[block1])
    def block2(_eq):
        _eq.export[w] = x + y - bar(z)

    graph = ps.flow.tie(block2, name="mygraph")

    result = graph.subs({z: f(1), bar(w): foo(w), x: 3})

    @ps.flow.block(name="block1")
    def block1_expect(_eq):
        _eq.export[x] = foo(f(1))
        _eq.export[y] = foo(w) + x

    @ps.flow.block(preds=[block1_expect, ps.flow.Top()], name="block2")
    def block2_expect(_eq):
        _eq.export[w] = x + y - bar(f(1))

    expected = ps.flow.tie(block2_expect, name="mygraph")
    assert result == expected


def test_subgraph_substitutions():
    x, y, z, w = sp.symbols("x, y, z, w")
    f = ps.fields("f(3): [2D]")
    foo = sp.Function("foo")
    bar = sp.Function("bar")

    @ps.flow.block
    def block1(_eq):
        _eq.export[x] = foo(z)
        _eq.export[y] = bar(w) + x

    @ps.flow.block(preds=[block1])
    def block2(_eq):
        _eq.export[w] = x + y - bar(z)

    graph = ps.flow.subgraph(block2, name="mygraph")

    result = graph.subs({z: f(1), bar(w): foo(w), x: 3})

    @ps.flow.block(name="block1")
    def block1_expect(_eq):
        _eq.export[x] = foo(f(1))
        _eq.export[y] = foo(w) + x

    @ps.flow.block(preds=[block1_expect, ps.flow.Top()], name="block2")
    def block2_expect(_eq):
        _eq.export[w] = x + y - bar(f(1))

    expected = ps.flow.subgraph(block2_expect, name="mygraph")
    assert result == expected
