import pytest
import sympy as sp
import pystencils as ps

from pystencils import fields, TypedSymbol
from pystencils.flow.flowgraph import (
    EquationsBlock,
    Top,
    Bottom,
    Let,
    Export,
)
from pystencils.grids import TensorField
from pystencils.flow.canonicalize_flowgraph import CanonicalizeFlowgraph, CanonicalizationError


def test_canonicalize():
    x, y, z, w = sp.symbols("x, y, z, w")

    orig_graph = EquationsBlock([Let(x, y + 1), Let(z, y + 2), Let(w, y + 3)], ())

    canon = CanonicalizeFlowgraph()
    canon_result = canon(orig_graph)
    bottom = canon_result.bottom

    assert isinstance(bottom, Bottom)
    assert len(bottom.predecessors) == 1

    ssa_block = list(bottom.predecessors)[0]
    assert isinstance(ssa_block, EquationsBlock)
    assert ssa_block.assignments == orig_graph.assignments
    assert ssa_block.predecessors == frozenset({Top()})

    orig_graph = EquationsBlock(
        [Export(x, y + z + w)],
        [
            EquationsBlock([Export(y, w + 1)], ()),
            EquationsBlock([Export(z, w + 2)], ()),
            EquationsBlock([Export(w, z + 3)], ()),
        ],
    )

    canon_result = canon(orig_graph)

    assert canon_result.free_symbols == frozenset([z, w])
    assert canon_result.exports == frozenset([x])
    assert not canon_result.effects
    assert not canon_result.fields_written
    assert not canon_result.fields_read

    bottom = canon_result.bottom

    assert isinstance(bottom, Bottom)
    assert len(bottom.predecessors) == 1

    x_block = list(bottom.predecessors)[0]
    assert isinstance(x_block, EquationsBlock)
    assert x_block.assignments == orig_graph.assignments

    assert len(x_block.predecessors) == 3
    assert Top() not in x_block.predecessors

    for pred in x_block.predecessors:
        assert pred.predecessors == frozenset({Top()})

    block_0 = EquationsBlock([Export(w, z + 3)], (), name="block_0")
    block_1a = EquationsBlock([Export(y, w + 1)], (block_0,), name="block_1a")
    block_1b = EquationsBlock([Export(z, w + 2)], (block_0,), name="block_1b")
    block_2 = EquationsBlock([Let(x, y + z + w)], (block_1a, block_1b, block_0), name="block_2")

    canon_result = canon(block_2)
    bottom = canon_result.bottom

    assert isinstance(bottom, Bottom)
    assert len(bottom.predecessors) == 1

    block_2_c = list(bottom.predecessors)[0]
    assert isinstance(block_2_c, EquationsBlock)
    assert block_2_c.assignments == block_2.assignments
    assert len(block_2_c.predecessors) == 3

    block_2_c_preds = sorted(block_2_c.predecessors, key=lambda n: n.name)

    block_0_c = block_2_c_preds[0]
    assert isinstance(block_0_c, EquationsBlock)
    assert block_0_c.assignments == block_0.assignments

    block_1a_c = block_2_c_preds[1]
    assert isinstance(block_1a_c, EquationsBlock)
    assert block_1a_c.assignments == block_1a.assignments

    block_1b_c = block_2_c_preds[2]
    assert isinstance(block_1b_c, EquationsBlock)
    assert block_1b_c.assignments == block_1b.assignments

    assert block_1a_c.predecessors == frozenset((block_0_c,))
    assert block_1b_c.predecessors == frozenset((block_0_c,))
    assert block_0_c.predecessors == frozenset((Top(),))


def test_free_symbols():
    x, y, z, v, w = sp.symbols("x, y, z, v, w")
    f, g, h = ps.fields("f, g(2), h(3): [2D]")

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
        _eq.store[f()] = z

    canon = CanonicalizeFlowgraph()
    canon_result = canon(Bottom([block1, distinction]))

    assert canon_result.free_symbols == {x, y}


def test_field_collection():
    x, y, z, w = sp.symbols("x, y, z, w")
    f, g, h = fields("f(2), g, h(3): [2D]")

    canon = CanonicalizeFlowgraph()

    @ps.flow.block
    def block1(_b):
        _b.let[x] = f(1)
        _b.export[y] = x + 1

    @ps.flow.block(preds=[block1])
    def block2(_b):
        _b.store[f(0)] = y
        _b.export[z] = g()

    @ps.flow.block(preds=[block2])
    def block3(_b):
        _b.store[h(1)] = z
        _b.store[h(2)] = z + 1

    bot = Bottom([block2, block3])

    canon_result = canon(bot)

    assert canon_result.fields_read == {f, g}
    assert canon_result.fields_written == {f, h}

    @ps.flow.cases
    def cs(cases):
        @cases.case(f(0) > 0)
        def _(_c):
            _c.store[h(1)] = g()

        @cases.case(f(0) < 0)
        def _(_c):
            _c.store[h(1)] = 3.2

    canon_result = canon(cs)

    assert canon_result.fields_read == {f, g}
    assert canon_result.fields_written == {h}


def test_field_collection_new_fields():
    x, y, z, w = sp.symbols("x, y, z, w")
    f = TensorField("f", 2, (2,))
    g = TensorField("g", 2)
    h = TensorField("h", 2, (3,))

    canon = CanonicalizeFlowgraph()

    @ps.flow.block
    def block1(_b):
        _b.let[x] = f(1)
        _b.export[y] = x + 1

    @ps.flow.block(preds=[block1])
    def block2(_b):
        _b.store[f(0)] = y
        _b.export[z] = g()

    @ps.flow.block(preds=[block2])
    def block3(_b):
        _b.store[h(1)] = z
        _b.store[h(2)] = z + 1

    bot = Bottom([block2, block3])

    canon_result = canon(bot)

    assert canon_result.fields_read == {f, g}
    assert canon_result.fields_written == {f, h}

    @ps.flow.cases
    def cs(cases):
        @cases.case(f(0) > 0)
        def _(_c):
            _c.store[h(1)] = g()

        @cases.case(f(0) < 0)
        def _(_c):
            _c.store[h(1)] = 3.2

    canon_result = canon(cs)

    assert canon_result.fields_read == {f, g}
    assert canon_result.fields_written == {h}


def test_malformed_graphs():
    x, y, z, w = sp.symbols("x, y, z, w")
    f, g = fields("f(2), g: [2D]")
    p = TypedSymbol("p", "float32")
    canon = CanonicalizeFlowgraph()

    #   Node imports symbol `y` from two predecessors
    graph = EquationsBlock(
        [Let(x, y + z + w)],
        [
            EquationsBlock([Export(y, w + 1)], ()),
            EquationsBlock([Export(z, w + 2)], ()),
            EquationsBlock([Export(y, z + 3)], ()),
        ],
    )

    with pytest.raises(CanonicalizationError):
        canon(graph)

    #   Duplicate reduction onto the same target
    @ps.flow.block
    def block1(_eq):
        _eq.reduce[p, "+"] = x

    @ps.flow.block(preds=[block1])
    def block2(_eq):
        _eq.reduce[p, "*"] = y

    with pytest.raises(CanonicalizationError) as exc:
        canon(block2)

    assert "Ambiguous side effect" in str(exc.value)

    #   Duplicate store onto the same target
    @ps.flow.block
    def block3(_eq):
        _eq.store[f(1)] = x

    @ps.flow.block(preds=[block3])
    def block4(_eq):
        _eq.export[y] = z

    @ps.flow.block(preds=[block4])
    def block5(_eq):
        _eq.store[f(1)] = y

    with pytest.raises(CanonicalizationError) as exc:
        canon(block5)

    assert "Ambiguous side effect" in str(exc.value)
