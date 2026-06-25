import pytest

import pystencils as ps
import sympy as sp
import numpy as np
import pickle

from pystencils.grids import (
    TensorField,
    MemoryLayout,
    Patch,
    PatchGrid,
    VariablePlacement,
)


def test_tensor_field_api():
    f = TensorField("f", 2, (2, 2))

    assert f.name == "f"
    assert f.dtype == ps.DynamicType.NUMERIC_TYPE
    assert f.layout == MemoryLayout.NUMPY
    assert f.spatial_rank == 2
    assert f.tensor_rank == 2
    assert f.tensor_shape == (2, 2)
    assert f.ghost_layers == 0
    assert f.grid is None

    f2 = TensorField("f", 2, (2, 2))
    assert f == f2
    assert hash(f) == hash(f2)

    f3 = TensorField("f", 2, (2, 2), layout="fzyx")
    assert f3.layout == MemoryLayout.FZYX
    assert f3 != f
    assert hash(f3) != hash(f)

    c = TensorField("c", 3, (), dtype="float", layout=MemoryLayout.ZYXF, ghost_layers=2)
    assert c.name == "c"
    assert c.dtype == ps.create_type("float")
    assert c.spatial_rank == 3
    assert c.tensor_rank == 0
    assert c.tensor_shape == ()
    assert c.layout == MemoryLayout.ZYXF
    assert c.ghost_layers == 2
    assert c.grid is None

    p = Patch("p", (1, 1, 1))
    g = TensorField("g", p.vertices, (3,))
    assert g.grid == PatchGrid(p, VariablePlacement.VERTICES)
    assert g.spatial_rank == 3

    g = TensorField("g", p.cells, ())
    assert g.grid == PatchGrid(p, VariablePlacement.CELLS)
    assert g.spatial_rank == 3

    #   Degenerate shapes
    with pytest.raises(ValueError):
        _ = TensorField("f", 2, (2, 1))

    with pytest.raises(ValueError):
        _ = TensorField("f", 2, (1,))

    with pytest.raises(ValueError):
        _ = TensorField("f", 2, (3, 1, 3))

    #   Invalid spatial rank
    with pytest.raises(ValueError):
        _ = TensorField("f", 0, (3, 3))

    with pytest.raises(ValueError):
        _ = TensorField("f", -2, (3, 3))


def test_iteration_limits():
    shape_symbols = tuple(
        ps.TypedSymbol(ps.DEFAULTS.field_shape_name("f", c), ps.DynamicType.INDEX_TYPE)
        for c in range(3)
    )

    args = [
        dict(rank_or_grid=1, tensor_shape=(), layout="leftmost"),
        dict(rank_or_grid=1, tensor_shape=(), layout="rightmost"),
        dict(rank_or_grid=1, tensor_shape=(), layout="zyxf"),
        dict(rank_or_grid=3, tensor_shape=(), layout="leftmost"),
        dict(rank_or_grid=3, tensor_shape=(), layout="rightmost"),
        dict(rank_or_grid=3, tensor_shape=(), layout="zyxf"),
        dict(rank_or_grid=2, tensor_shape=(2, 2), layout="leftmost"),
        dict(rank_or_grid=2, tensor_shape=(2, 3), layout="rightmost"),
        dict(rank_or_grid=2, tensor_shape=(5,), layout="zyxf"),
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
        f = TensorField("f", **kwargs)
        ilims = f.get_iteration_limits()

        assert ilims.bounds == bounds
        assert ilims.loop_order == order


def test_accesses():
    f = TensorField("f", 2, ())

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

    g = TensorField("g", 3, (3,))

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
    c = TensorField("c", 2, ())

    term = c() + c[-1, 0]() + c[1, 0]() + c[0, -1]() + c[0, 1]()
    assert set(term.args) == {c[0, 0](), c[-1, 0](), c[1, 0](), c[0, -1](), c[0, 1]()}


def test_accesses_pickle():
    g = TensorField("g", 3, (3,))
    acc = g[-1, 0, 1](2)

    dump = pickle.dumps(acc)
    acc_restored = pickle.loads(dump)

    assert acc_restored.field == g
    assert acc_restored.offsets == (-1, 0, 1)
    assert acc_restored.indices == (2,)
    assert acc_restored == acc


def test_create_ndarray(array_module):
    xp = array_module

    try:
        import tests.dpctl_compat as dpnp

        if array_module is dpnp:
            pytest.xfail("CreateNdArray not yet supported on DPCTL (issue #138)")
    except ImportError:
        pass

    f = TensorField("f", 3, (), dtype="float32")
    f_arr = f.create_ndarray(xp, (5, 6, 7))
    f_type = np.dtype(np.float32)

    assert isinstance(f_arr, xp.ndarray)
    assert f_arr.shape == (5, 6, 7)
    assert f_arr.dtype == f_type
    assert f_arr.strides == (
        6 * 7 * f_type.itemsize,
        7 * f_type.itemsize,
        f_type.itemsize,
    )

    f = TensorField("f", 3, (), dtype="float32", layout=MemoryLayout.FORTRAN)
    f_arr = f.create_ndarray(xp, (5, 6, 7))
    f_type = np.dtype(np.float32)

    assert f_arr.shape == (5, 6, 7)
    assert f_arr.dtype == f_type
    assert f_arr.strides == (
        f_type.itemsize,
        5 * f_type.itemsize,
        5 * 6 * f_type.itemsize,
    )

    f = TensorField("f", 2, (2,), dtype="float64", layout=MemoryLayout.ZYXF)
    f_arr = f.create_ndarray(xp, (5, 6))
    f_type = np.dtype(np.float64)

    assert f_arr.shape == (5, 6, 2)
    assert f_arr.dtype == f_type
    assert f_arr.strides == (
        2 * f_type.itemsize,
        2 * 5 * f_type.itemsize,
        f_type.itemsize,
    )

    f = TensorField("f", 2, (2, 2), dtype="float16", layout=MemoryLayout.FZYX)
    f_arr = f.create_ndarray(xp, (5, 6))
    f_type = np.dtype(np.float16)

    assert f_arr.shape == (5, 6, 2, 2)
    assert f_arr.dtype == f_type
    assert f_arr.strides == (
        f_type.itemsize,
        5 * f_type.itemsize,
        6 * 5 * f_type.itemsize,
        2 * 6 * 5 * f_type.itemsize,
    )

    f = TensorField(
        "f", 2, (2,), ghost_layers=1, dtype="float64", layout=MemoryLayout.NUMPY
    )
    f_arr = f.create_ndarray(xp, (5, 6))
    f_type = np.dtype(np.float64)

    assert f_arr.shape == (7, 8, 2)
    assert f_arr.dtype == f_type
    assert f_arr.strides == (
        2 * 8 * f_type.itemsize,
        2 * f_type.itemsize,
        f_type.itemsize,
    )

    f = TensorField(
        "f", 3, (3,), ghost_layers=1, dtype="float32", layout=MemoryLayout.FZYX
    )
    f_arr = f.create_ndarray(xp, (5, 6, 7))
    f_type = np.dtype(np.float32)

    assert f_arr.shape == (7, 8, 9, 3)
    assert f_arr.dtype == f_type
    assert f_arr.strides == (
        f_type.itemsize,
        7 * f_type.itemsize,
        7 * 8 * f_type.itemsize,
        7 * 8 * 9 * f_type.itemsize,
    )


def test_view_ndarray(array_module):
    xp = array_module

    rng = np.random.default_rng(42)

    f = TensorField(
        "f", 2, (2,), ghost_layers=0, dtype="float64", layout=MemoryLayout.NUMPY
    )

    f_arr = xp.zeros((5, 6, 2))
    f_arr[:] = xp.array(rng.random(f_arr.shape, dtype=np.float64))

    f_view = f.view_ndarray(f_arr)
    xp.testing.assert_array_equal(f_view, f_arr)

    f_view[3, 4, 1] = 14.2
    assert f_arr[3, 4, 1] == 14.2

    f = TensorField(
        "f", 2, (2,), ghost_layers=1, dtype="float64", layout=MemoryLayout.FORTRAN
    )

    f_arr = xp.zeros((5 + 2, 6 + 2, 2), order="F")
    f_arr[:] = xp.array(rng.random(f_arr.shape, dtype=np.float64))

    f_view = f.view_ndarray(f_arr)
    xp.testing.assert_array_equal(f_view, f_arr[1:-1, 1:-1, :])

    f_view[2, 2, 1] = 14.2
    assert f_arr[3, 3, 1] == 14.2
