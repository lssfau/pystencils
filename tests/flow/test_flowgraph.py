import pytest

import sympy as sp
import pystencils as ps

from pystencils.flow.flowgraph import (
    Let,
    Export,
    Store,
    Reduce,
    EquationsBlock,
    Bottom,
    Top,
    Cases,
    Flowgraph,
    Subgraph,
)
from pystencils.flow import tie
from pystencils import TypedSymbol, fields
from pystencils.sympyextensions import ReductionOp, mem_acc


def test_assignment_api():
    x, y, z, w = sp.symbols("x, y, z, w")
    q, r = TypedSymbol("q", "float32"), TypedSymbol("r", "float32")

    let1 = Let(x, y + z)

    assert let1.lhs == x
    assert let1.rhs == y + z

    let2 = Let(q, w + r)
    assert let2.lhs == q
    assert let2.rhs == w + r

    export1 = Export(x, y + z)

    assert export1.lhs == x
    assert export1.rhs == y + z

    export2 = Export(q, w + r)
    assert export2.lhs == q
    assert export2.rhs == w + r

    reduce1 = Reduce(r, q + 2, ReductionOp.Mul)
    assert reduce1.lhs == r
    assert reduce1.rhs == q + 2
    assert reduce1.op == ReductionOp.Mul


def test_invalid_assignments():
    x, y, z, w = sp.symbols("x, y, z, w")
    q, r = TypedSymbol("q", "float32"), TypedSymbol("r", "float32")
    f, g = fields("f(3), g(1) : [2D]")

    with pytest.raises(TypeError):
        _ = Let(f(2), w)

    with pytest.raises(TypeError):
        _ = Let(mem_acc(z, 3), w)

    with pytest.raises(TypeError):
        _ = Export(f(2), w)

    with pytest.raises(TypeError):
        _ = Export(mem_acc(z, 3), w)

    with pytest.raises(TypeError):
        _ = Store(x, w)

    with pytest.raises(TypeError):
        _ = Reduce(x, w, ReductionOp.Sub)


def test_assignment_equality():
    x, y, z, w = sp.symbols("x, y, z, w")
    q, r = TypedSymbol("q", "float32"), TypedSymbol("r", "float32")

    asm = Let(r, q + 2)
    assert asm == asm

    assert Let(x, z + w) == Let(x, z + w)
    assert Let(x, z + w) != Export(x, z + w)

    assert hash(Let(x, z + w)) == hash(Let(x, z + w))
    assert hash(Let(x, z + w)) != hash(Export(x, z + w))

    assert Reduce(q, r, ReductionOp.Min) == Reduce(q, r, ReductionOp.Min)
    assert Reduce(q, r, ReductionOp.Min) != Reduce(q, r, ReductionOp.Max)

    assert hash(Reduce(q, r, ReductionOp.Min)) == hash(Reduce(q, r, ReductionOp.Min))
    assert hash(Reduce(q, r, ReductionOp.Min)) != hash(Reduce(q, r, ReductionOp.Max))


def test_equationsblock_api():
    x, y, z, w = sp.symbols("x, y, z, w")
    q, r = TypedSymbol("q", "float32"), TypedSymbol("r", "float32")
    f, g = fields("f(3), g(1) : [2D]")

    asms = (Let(x, z / w), Let(y, x * 3 + 5), Export(q, x + 2 * w))

    block = EquationsBlock(asms, ())
    assert block.assignments == asms
    assert block.free_symbols == {z, w}
    assert block.exports == {q}
    assert block.predecessors == frozenset()

    asms2 = [Let(z, r + 3), Store(g[-1, 2](0), z)]
    block2 = EquationsBlock(asms2, (block,))

    assert block2.assignments == tuple(asms2)
    assert block2.free_symbols == {r}
    assert block2.exports == set()
    assert block2.predecessors == frozenset({block})

    #   Assignments given in the wrong order shall be ordered topologically
    asms3 = (Let(w, r + 3 - z), Let(r, y), Let(x, r / 2), Let(z, sp.Rational(3, 5)))
    block3 = EquationsBlock(asms3, ())
    lhss_sorted = [a.lhs for a in block3.assignments]
    assert lhss_sorted.index(w) > lhss_sorted.index(r)
    assert lhss_sorted.index(w) > lhss_sorted.index(z)
    assert lhss_sorted.index(x) > lhss_sorted.index(r)

    asms4 = (
        Reduce(q, z + x, ReductionOp.Min),
        Let(r, y),
        Store(f(1), z + r),
        Let(z, sp.Rational(3, 5)),
        Export(x, sp.sqrt(5)),
    )
    block4 = EquationsBlock(asms4, ())
    lhss_sorted = [a.lhs for a in block4.assignments]
    assert lhss_sorted.index(q) > lhss_sorted.index(z)
    assert lhss_sorted.index(q) > lhss_sorted.index(x)
    assert lhss_sorted.index(f(1)) > lhss_sorted.index(z)
    assert lhss_sorted.index(f(1)) > lhss_sorted.index(r)


def test_equationsblock_errors():
    x, y, z, w = sp.symbols("x, y, z, w")
    q, r = TypedSymbol("q", "float32"), TypedSymbol("r", "float32")
    f, g = fields("f(3), g(1) : [2D]")

    #   SSA form violations

    with pytest.raises(ValueError):
        EquationsBlock((Let(x, y), Let(x, z)), ())

    with pytest.raises(ValueError):
        EquationsBlock((Export(x, y), Let(x, z)), ())

    with pytest.raises(ValueError):
        EquationsBlock((Let(x, y), Store(f(1), z), Store(f(1), w)), ())

    #   Cyclic assignments
    with pytest.raises(ValueError):
        EquationsBlock((Let(x, y), Let(y, z), Let(z, x)), ())

    #   Bottom as a predecessor
    with pytest.raises(ValueError):
        EquationsBlock([Let(x, y)], (Bottom([]),))


def test_cases_api():
    x, y, z, w = sp.symbols("x, y, z, w")
    q = TypedSymbol("q", "float32")

    # Two cases
    conditions: list[sp.Basic] = []
    conditions.append(z > w)
    conditions.append(z <= w)

    subgraphs: list[Flowgraph] = []
    subgraphs.append(
        tie(
            EquationsBlock((Let(x, z / w), Let(y, x * 3 + 5), Export(q, x + 2 * y)), ())
        )
    )
    subgraphs.append(
        tie(
            EquationsBlock((Let(x, w / z), Let(y, x * 3 + 5), Export(q, x + 2 * y)), ())
        )
    )

    cases1 = Cases(zip(conditions, subgraphs), ())

    assert cases1.conditions == tuple(conditions[:-1]) + (sp.true,)
    assert cases1.subgraphs == tuple(subgraphs)
    assert cases1.free_symbols == {w, z}
    assert cases1.exports == {q}
    assert cases1.predecessors == frozenset()

    # Three cases
    conditions = []
    conditions.append(z > w)
    conditions.append(z < w)
    conditions.append(sp.Eq(z, w))

    subgraphs.append(
        tie(
            EquationsBlock(
                (Let(x, sp.Integer(1)), Let(y, x * 3 + 5), Export(q, x + 2 * y)), ()
            )
        )
    )

    cases2 = Cases(zip(conditions, subgraphs), ())

    assert cases2.conditions == tuple(conditions[:-1] + [sp.true])
    assert cases2.subgraphs == tuple(subgraphs)
    assert cases2.free_symbols == {w, z}
    assert cases2.exports == {q}
    assert cases2.predecessors == frozenset()


def test_cases_errors():
    x, y, z = sp.symbols("x, y, z")
    p, q, r = (
        TypedSymbol("p", "float32"),
        TypedSymbol("q", "float32"),
        TypedSymbol("r", "float32"),
    )

    # Incomplete cases
    with pytest.raises(ValueError) as rexc:
        conditions = tuple([y < z, y > z])
        subgraphs = tuple(
            [
                tie(EquationsBlock([Export(q, x + 2 * y)], ())),
                tie(EquationsBlock([Export(q, x + 2 * z)], ())),
            ]
        )
        Cases(
            zip(conditions, subgraphs),
            predecessors=(),
        )
    assert "Case distinction with exports must be complete" in str(rexc.value)

    # Incomplete cases
    with pytest.raises(ValueError) as rexc:
        Cases(
            [
                (
                    y < z,
                    tie(
                        EquationsBlock((Export(q, x + 2 * y), Export(r, x + 2 * z)), ())
                    ),
                )
            ],
            predecessors=(),
        )
    assert "Case distinction with exports must be complete" in str(rexc.value)

    # Non-matching exports
    with pytest.raises(ValueError) as rexc:
        conditions = tuple([sp.true, sp.false])
        subgraphs = tuple(
            [
                tie(EquationsBlock((Export(q, x + 2 * y), Export(r, x + 2 * z)), ())),
                tie(EquationsBlock(([Export(q, x + 2 * z)]), ())),
            ]
        )
        Cases(
            zip(conditions, subgraphs),
            predecessors=(),
        )
    assert (
        "All branches in a case distinction must have the same set of exports"
        in str(rexc.value)
    )

    # Non-matching exports
    with pytest.raises(ValueError) as rexc:
        conditions = [y < z, y > z, sp.true]
        subgraphs = tuple(
            [
                tie(EquationsBlock((Export(q, x + 2 * y), Export(r, x + 2 * z)), ())),
                tie(EquationsBlock(((Export(q, x + 2 * z), Export(r, x + 2 * y))), ())),
                tie(
                    EquationsBlock(
                        (
                            (
                                Export(q, x + 2 * z),
                                Export(r, x + 2 * z),
                                Export(p, y + z),
                            )
                        ),
                        (),
                    )
                ),
            ]
        )
        Cases(
            zip(conditions, subgraphs),
            predecessors=(),
        )

    assert (
        "All branches in a case distinction must have the same set of exports"
        in str(rexc.value)
    )

    # Cases with exported symbols used in conditions
    with pytest.raises(ValueError) as rexc:
        conditions = [q < z, q >= z]
        subgraphs = tuple(
            [
                tie(EquationsBlock((Export(q, x + 2 * y), Export(r, x + 2 * z)), ())),
                tie(EquationsBlock(((Export(q, x + 2 * z), Export(r, x + 2 * y))), ())),
            ]
        )
        Cases(
            zip(conditions, subgraphs),
            predecessors=(),
        )
    assert "Symbol name conflicts" in str(rexc.value)


def test_atoms():
    x, y, z, v, w = sp.symbols("x, y, z, v, w")
    f, g, h = ps.fields("f, g(2), h(3): [2D]")

    @ps.flow.block
    def block1(_eq):
        _eq.let[x] = 3 + v
        _eq.store[f()] = x - g(1)

    assert block1.atoms(sp.Symbol) == {x, v, f(), g(1)}

    assert block1.assignments[0].atoms() == {x, v, sp.sympify(3)}
    assert block1.assignments[0].atoms(sp.Symbol) == {x, v}
    assert block1.assignments[0].atoms(ps.Field.Access) == set()

    assert block1.assignments[1].atoms(sp.Symbol) == {x, f(), g(1)}
    assert block1.assignments[1].atoms(ps.Field.Access) == {f(), g(1)}

    @ps.flow.block(preds=[block1])
    def block2(_eq):
        _eq.export[z] = w

    assert block2.atoms() == {z, w}

    graph = ps.flow.tie(block1, block2)
    assert graph.atoms(sp.Symbol) == {x, v, z, w, f(), g(1)}

    subgr = Subgraph(graph, [])
    assert subgr.atoms() == graph.atoms()

    @ps.flow.cases
    def cases1(_cs):
        @_cs.case(y > 3)
        def case1(_eq):
            _eq.store[h(0)] = y

        @_cs.case(y < -2)
        def case2(_eq):
            _eq.store[h(0)] = x

    assert cases1.atoms() == {x, y, sp.Number(3), sp.Number(-2), h(0)}


def test_flowgraph_api():
    graph = Flowgraph(Bottom([]), name="my_graph")

    assert graph.bottom == Bottom([Top()])
    assert graph.name == "my_graph"

    x, y, z, v, w = sp.symbols("x, y, z, v, w")
    f, g, h = ps.fields("f, g(2), h(3): [2D]")

    @ps.flow.block
    def block1(_eq):
        _eq.store[f()] = g(1)
        _eq.export[x] = h(1) + y

    @ps.flow.block
    def block2(_eq):
        _eq.store[g(0)] = x
        _eq.export[v] = 2
        _eq.reduce["q":"float64", "max"] = z

    graph = ps.flow.tie(block1, block2, name="thegraph")

    assert graph.free_symbols == {x, y, z}
    assert graph.exports == {x, v}
    assert graph.fields_read == {g, h}
    assert graph.fields_written == {f, g}
    assert graph.effects == {
        Store(f(), g(1)),
        Store(g(0), x),
        Reduce(TypedSymbol("q", "float64"), z, ReductionOp.Max),
    }

    graph2 = Flowgraph(Bottom([block1, block2]), name="thegraph")
    assert graph == graph2


def test_printing():
    x, y, z, v, w = sp.symbols("x, y, z, v, w")
    f, g, h = ps.fields("f, g(2), h(3): [2D]")
    r = ps.TypedSymbol("r", "float64")

    let = Let(x, y + z)
    assert str(let) == "let x = y + z"

    export = Export(x, y + z)
    assert str(export) == "export x = y + z"

    store = Store(f(), y + z)
    assert str(store) == "store f[0,0] = y + z"

    reduce = Reduce(r, y + z, ReductionOp.Min)
    assert str(reduce) == "reduce(min) r: float64 = y + z"

    @ps.flow.cases
    def distinction(_cs):
        @_cs.case(y > 0)
        def case1(_eq):
            _eq.store[h(0)] = y
            _eq.export[z] = 1

        @_cs.case(y <= 0)
        def case2(_eq):
            _eq.store[h(0)] = x
            _eq.export[z] = 2

    @ps.flow.block(preds=[distinction])
    def block1(_eq):
        _eq.let[v] = z - 2
        _eq.reduce["q": "float64", "max"] = r
        _eq.store[f()] = z

    graph = ps.flow.tie(block1, distinction, name="lorem-ipsum")

    graph_str = str(graph)

    assert "graph lorem-ipsum" in graph_str
    assert "reduce (q: max)" in graph_str
    assert "store f[0,0]" in graph_str
    assert "store h[0,0](0)" in graph_str


def test_graphviz():
    pytest.importorskip("graphviz")

    x, y, z, v, w = sp.symbols("x, y, z, v, w")
    f, g, h = ps.fields("f, g(2), h(3): [2D]")
    r = ps.TypedSymbol("r", "float64")

    @ps.flow.cases
    def distinction(_cs):
        @_cs.case(y > 0)
        def case1(_eq):
            _eq.store[h(0)] = y
            _eq.export[z] = 1

        @_cs.case(y <= 0)
        def case2(_eq):
            _eq.store[h(0)] = x
            _eq.export[z] = 2

    @ps.flow.block(preds=[distinction])
    def block1(_eq):
        _eq.let[v] = z - 2
        _eq.reduce["q": "float64", "max"] = r
        _eq.store[f()] = z

    graph = ps.flow.tie(block1, distinction, name="lorem-ipsum")
    _ = ps.flow.to_dot(graph)
