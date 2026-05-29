import pytest

import pystencils as ps
import sympy as sp
import pickle


def test_tensor_field_api():
    f = ps.grids.TensorField("f", 2, (2, 2))

    assert f.name == "f"
    assert f.dtype == ps.DynamicType.NUMERIC_TYPE
    assert f.layout == ps.grids.MemoryLayout.NUMPY
    assert f.spatial_rank == 2
    assert f.tensor_rank == 2
    assert f.tensor_shape == (2, 2)

    f2 = ps.grids.TensorField("f", 2, (2, 2))
    assert f == f2
    assert hash(f) == hash(f2)

    f3 = ps.grids.TensorField("f", 2, (2, 2), layout="fzyx")
    assert f3.layout == ps.grids.MemoryLayout.FZYX
    assert f3 != f
    assert hash(f3) != hash(f)

    c = ps.grids.TensorField(
        "c", 3, (), dtype="float", layout=ps.grids.MemoryLayout.ZYXF
    )
    assert c.name == "c"
    assert c.dtype == ps.create_type("float")
    assert c.spatial_rank == 3
    assert c.tensor_rank == 0
    assert c.tensor_shape == ()
    assert c.layout == ps.grids.MemoryLayout.ZYXF

    #   Degenerate shapes
    with pytest.raises(ValueError):
        _ = ps.grids.TensorField("f", 2, (2, 1))

    with pytest.raises(ValueError):
        _ = ps.grids.TensorField("f", 2, (1,))

    with pytest.raises(ValueError):
        _ = ps.grids.TensorField("f", 2, (3, 1, 3))

    #   Invalid spatial rank
    with pytest.raises(ValueError):
        _ = ps.grids.TensorField("f", 0, (3, 3))

    with pytest.raises(ValueError):
        _ = ps.grids.TensorField("f", -2, (3, 3))


def test_iteration_limits():
    shape_symbols = tuple(
        ps.TypedSymbol(ps.DEFAULTS.field_shape_name("f", c), ps.DynamicType.INDEX_TYPE)
        for c in range(3)
    )

    args = [
        dict(spatial_rank=1, tensor_shape=(), layout="leftmost"),
        dict(spatial_rank=1, tensor_shape=(), layout="rightmost"),
        dict(spatial_rank=1, tensor_shape=(), layout="zyxf"),
        dict(spatial_rank=3, tensor_shape=(), layout="leftmost"),
        dict(spatial_rank=3, tensor_shape=(), layout="rightmost"),
        dict(spatial_rank=3, tensor_shape=(), layout="zyxf"),
        dict(spatial_rank=2, tensor_shape=(2, 2), layout="leftmost"),
        dict(spatial_rank=2, tensor_shape=(2, 3), layout="rightmost"),
        dict(spatial_rank=2, tensor_shape=(5,), layout="zyxf"),
    ]

    expected_limits = [
        ((shape_symbols[0],), (0,)),
        ((shape_symbols[0],), (0,)),
        ((shape_symbols[0],), (0,)),
        (shape_symbols, (2, 1, 0)),
        (shape_symbols, (0, 1, 2)),
        (shape_symbols, (2, 1, 0)),
        (shape_symbols[:2], (1, 0)),
        (shape_symbols[:2], (0, 1)),
        (shape_symbols[:2], (1, 0)),
    ]

    for kwargs, (bounds, order) in zip(args, expected_limits, strict=True):
        f = ps.grids.TensorField("f", **kwargs)
        ilims = f.get_iteration_limits()

        assert ilims.bounds == bounds
        assert ilims.loop_order == order


def test_accesses():
    f = ps.grids.TensorField("f", 2, ())

    acc = f()
    assert acc.field == f

    assert isinstance(acc.offsets, sp.Tuple)
    assert acc.offsets == (0, 0)

    assert isinstance(acc.indices, sp.Tuple)
    assert acc.indices == ()

    acc = f[-1, 1]()
    assert acc.field == f

    assert isinstance(acc.offsets, sp.Tuple)
    assert acc.offsets == (-1, 1)

    assert isinstance(acc.indices, sp.Tuple)
    assert acc.indices == ()

    with pytest.raises(IndexError):
        _ = f(1)

    with pytest.raises(IndexError):
        _ = f(2, 3)

    with pytest.raises(IndexError):
        _ = f[0]()

    with pytest.raises(IndexError):
        _ = f[1, 2, 3]()

    g = ps.grids.TensorField("g", 3, (3,))

    acc = g(0)
    assert acc.field == g
    assert acc.offsets == (0, 0, 0)
    assert acc.indices == (0,)

    acc = g(2)
    assert acc.field == g
    assert acc.offsets == (0, 0, 0)
    assert acc.indices == (2,)

    with pytest.raises(IndexError):
        _ = g()

    with pytest.raises(IndexError):
        _ = g(
            3,
        )

    with pytest.raises(IndexError):
        _ = g(3)

    i, j, k = sp.symbols("i, j, k")

    acc = g[-2, -1, 0](i)
    assert acc.offsets == (-2, -1, 0)
    assert acc.indices == (i,)

    acc = g[i, j, k](0)
    assert acc.offsets == (i, j, k)
    assert acc.indices == (0,)

    with pytest.raises(IndexError):
        _ = g[i, j](k)


def test_arithmetic_with_accesses():
    c = ps.grids.TensorField("c", 2, ())

    term = c() + c[-1, 0]() + c[1, 0]() + c[0, -1]() + c[0, 1]()
    assert set(term.args) == {c[0, 0](), c[-1, 0](), c[1, 0](), c[0, -1](), c[0, 1]()}


def test_accesses_pickle():
    g = ps.grids.TensorField("g", 3, (3,))
    acc = g[-1, 0, 1](2)

    dump = pickle.dumps(acc)
    acc_restored = pickle.loads(dump)

    assert acc_restored.field == g
    assert acc_restored.offsets == (-1, 0, 1)
    assert acc_restored.indices == (2,)
    assert acc_restored == acc
