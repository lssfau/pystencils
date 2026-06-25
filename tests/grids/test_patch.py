import pytest

import sympy as sp
import pystencils as ps
import numpy as np

from pystencils.grids import Patch, TensorField, PatchData
from pystencils.sympyextensions import TypedSymbol, DynamicType, tcast


def test_patch_api():
    patch_a = Patch("A", (1, 1, 1))

    assert patch_a.name == "A"
    assert patch_a.x_min == sp.ImmutableMatrix([0, 0, 0])
    assert patch_a.x_max == sp.ImmutableMatrix([1, 1, 1])
    assert patch_a.extents == sp.ImmutableMatrix([1, 1, 1])

    assert len(patch_a.num_vertices) == 3
    for nv, nc in zip(patch_a.num_vertices, patch_a.num_cells):
        assert isinstance(nv, TypedSymbol)
        assert nv.dtype == DynamicType.INDEX_TYPE
        assert nc == nv - 1

    assert patch_a.spacing == sp.ImmutableMatrix(
        [1 / tcast.auto(n - 1) for n in patch_a.num_vertices]
    )

    patch_b = Patch("B", (-1, 0, 1), (2, 3, 4), num_vertices=(6, 12, 24))

    assert patch_b.x_min == sp.ImmutableMatrix([-1, 0, 1])
    assert patch_b.x_max == sp.ImmutableMatrix([2, 3, 4])
    assert patch_b.extents == sp.ImmutableMatrix([3, 3, 3])
    assert patch_b.spacing == sp.ImmutableMatrix(
        [sp.Rational(3, 5), sp.Rational(3, 11), sp.Rational(3, 23)]
    )

    with pytest.raises(ValueError):
        _ = Patch("C", (1, 1, 1), num_cells=(3, 2, 3), num_vertices=(4, 5, 1))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_patch_vertex_geometry_fixed(dtype):
    S = Patch("S", (-1, 0, 1), (3, 4, 5), num_vertices=(10, 12, 14))

    f = TensorField("f", S.vertices, (3,), dtype=dtype)

    @ps.flow.operator
    def set_vertex_coords(_eq):
        v = S.vertex()
        for i in range(3):
            _eq.store[f(i)] = v[i]

    S_data = PatchData(S, fields=[f])
    set_vertex_coords(S_data)

    xs = np.linspace(-1, 3, 10, dtype=dtype)
    ys = np.linspace(0, 4, 12, dtype=dtype)
    zs = np.linspace(1, 5, 14, dtype=dtype)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    f_expected = np.stack((xx, yy, zz), axis=-1)

    rtol = np.finfo(dtype).resolution

    np.testing.assert_allclose(S_data[f], f_expected, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_patch_vertex_geometry_symbolic(dtype):
    Nv = ps.symbols("N_:3", dtype=ps.index_t)
    L = ps.symbols("L_:3", dtype=dtype)

    S = Patch("S", L, num_vertices=Nv)

    f = TensorField("f", S.vertices, (3,), dtype=dtype)

    @ps.flow.operator
    def set_vertex_coords(_eq):
        v = S.vertex()
        for i in range(3):
            _eq.store[f(i)] = v[i]

    S_data = PatchData(S, {L: (4, 5, 6), Nv: (4, 14, 9)}, fields=[f])
    set_vertex_coords(S_data)

    xs = np.linspace(0, 4, 4, dtype=dtype)
    ys = np.linspace(0, 5, 14, dtype=dtype)
    zs = np.linspace(0, 6, 9, dtype=dtype)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    f_expected = np.stack((xx, yy, zz), axis=-1)

    rtol = np.finfo(dtype).resolution

    np.testing.assert_allclose(S_data[f], f_expected, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_patch_cell_geometry_fixed(dtype):
    S = Patch("S", (-1, 0, 1), (0, 4, 4), num_cells=(5, 10, 12))

    f = TensorField("f", S.cells, (3,), dtype=dtype)

    @ps.flow.operator
    def set_cell_coords(_eq):
        c = S.cell_center()
        for i in range(3):
            _eq.store[f(i)] = c[i]

    S_data = PatchData(S, fields=[f])
    set_cell_coords(S_data)

    xs = np.linspace(-1, 0, 6, dtype=dtype)[:-1] + 1.0 / 10.0
    ys = np.linspace(0, 4, 11, dtype=dtype)[:-1] + 1.0 / 5.0
    zs = np.linspace(1, 4, 13, dtype=dtype)[:-1] + 1.0 / 8.0

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    f_expected = np.stack((xx, yy, zz), axis=-1)

    rtol = np.finfo(dtype).resolution

    np.testing.assert_allclose(S_data[f], f_expected, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_patch_cell_geometry_symbolic(dtype):
    Nc = ps.symbols("N_:3", dtype=ps.index_t)
    L = ps.symbols("L_:3", dtype=dtype)
    S = Patch("S", L, num_cells=Nc)

    f = TensorField("f", S.cells, (3,), dtype=dtype)

    @ps.flow.operator
    def set_cell_coords(_eq):
        c = S.cell_center()
        for i in range(3):
            _eq.store[f(i)] = c[i]

    S_data = PatchData(S, {L: (4, 5, 3), Nc: (9, 6, 11)}, fields=[f])
    set_cell_coords(S_data)

    xs = np.linspace(0, 4, 10, dtype=dtype)[:-1] + 2.0 / 9.0
    ys = np.linspace(0, 5, 7, dtype=dtype)[:-1] + 5.0 / 12.0
    zs = np.linspace(0, 3, 12, dtype=dtype)[:-1] + 3.0 / 22.0

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    f_expected = np.stack((xx, yy, zz), axis=-1)

    rtol = np.finfo(dtype).resolution

    np.testing.assert_allclose(S_data[f], f_expected, rtol=rtol)
