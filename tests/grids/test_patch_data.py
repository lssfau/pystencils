import pytest

import numpy as np
import sympy as sp
import pystencils as ps

from pystencils.grids import Patch, PatchData, TensorField


def test_patch_data_api():
    patch = Patch("P", (3, 4, 5), num_vertices=(10, 12, 14))

    patch_data = PatchData(patch)

    assert patch_data.patch == patch
    assert patch_data.dimensionality == 3
    np.testing.assert_array_equal(patch_data.x_min, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(patch_data.x_max, [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(patch_data.extents, [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(
        patch_data.spacing, [3.0 / 9.0, 4.0 / 11.0, 5.0 / 13.0]
    )
    assert patch_data.num_vertices == (10, 12, 14)
    assert patch_data.num_cells == (9, 11, 13)
    assert patch_data.target == ps.Target.GenericCPU
    assert patch_data.array_module is np
    assert patch_data.default_dtype == np.dtype(np.float64)

    L = ps.symbols("L_:3")
    Nv = ps.symbols("N_:3", dtype=ps.index_t)
    C = sp.Symbol("C")
    patch = Patch("P", L, num_vertices=Nv)

    patch_data = PatchData(
        patch, {L: (1.0, 1.5, 2.0), Nv: (8, 10, 6), C: 13.2}, default_dtype=np.float32
    )

    assert patch_data.patch == patch
    assert patch_data.dimensionality == 3
    assert patch_data.default_dtype == np.dtype(np.float32)

    np.testing.assert_array_equal(
        patch_data.x_min, np.array([0.0, 0.0, 0.0], dtype=np.float32), strict=True
    )
    np.testing.assert_array_equal(
        patch_data.x_max, np.array([1.0, 1.5, 2.0], dtype=np.float32), strict=True
    )
    np.testing.assert_array_equal(
        patch_data.extents, np.array([1.0, 1.5, 2.0], dtype=np.float32), strict=True
    )
    np.testing.assert_array_equal(
        patch_data.spacing,
        np.array([1.0 / 7.0, 1.5 / 9.0, 2.0 / 5.0], dtype=np.float32),
        strict=True,
    )
    assert patch_data.num_vertices == (8, 10, 6)
    assert patch_data.num_cells == (7, 9, 5)
    assert patch_data.target == ps.Target.GenericCPU
    assert patch_data.array_module is np

    assert patch_data[L] == (np.float32(1.0), np.float32(1.5), np.float32(2.0))
    assert all(isinstance(Lx, np.float32) for Lx in patch_data[L])

    assert patch_data[Nv] == (8, 10, 6)
    assert all(isinstance(Nvi, np.int64) for Nvi in patch_data[Nv])

    assert patch_data[C] == np.float32(13.2)
    assert isinstance(patch_data[C], np.float32)


def test_patch_data_variable_dtypes():
    patch = Patch("P", (1, 1, 1), num_vertices=(10, 10, 10))

    x, y = sp.symbols("x, y")
    q, w = ps.symbols("q, w", dtype=ps.numeric_t)
    i, j = ps.symbols("i, j", dtype=ps.index_t)
    a, b = ps.symbols("a, b", dtype="bool")

    c, d = ps.symbols("c, d", dtype=np.float16)

    patch_data = PatchData(
        patch,
        {x: 13.2, q: 5.1, i: 15, a: True, c: -3.2},
        default_dtype=np.float32,
        index_dtype=np.uint32,
    )

    assert isinstance(patch_data[x], np.float32)
    assert patch_data[x] == np.float32(13.2)

    assert isinstance(patch_data[q], np.float32)
    assert patch_data[q] == np.float32(5.1)

    assert isinstance(patch_data[i], np.uint32)
    assert patch_data[i] == np.uint32(15)

    assert isinstance(patch_data[a], np.bool_)
    assert patch_data[a] == np.bool_(True)

    assert isinstance(patch_data[c], np.float16)
    assert patch_data[c] == np.float16(-3.2)

    patch_data[y] = 5.4

    assert isinstance(patch_data[y], np.float32)
    assert patch_data[y] == np.float32(5.4)

    patch_data[w] = 0.7

    assert isinstance(patch_data[w], np.float32)
    assert patch_data[w] == np.float32(0.7)

    patch_data.set_data(j, 32)

    assert isinstance(patch_data[j], np.uint32)
    assert patch_data[j] == np.uint32(32)

    patch_data[b] = False

    assert isinstance(patch_data[b], np.bool_)
    assert patch_data[b] == np.bool_(False)

    patch_data.set_data(d, 1.03)

    assert isinstance(patch_data[d], np.float16)
    assert patch_data[d] == np.float16(1.03)


def test_patch_data_field_arrays(target, xp):
    if target == ps.Target.SYCL:
        pytest.xfail("CreateNdArray not yet available for SYCL (issue #138)")

    patch = Patch("P", (1, 1, 1), num_vertices=(5, 8, 12))

    f = TensorField("f", patch.vertices, (3,), dtype=ps.numeric_t)
    g = TensorField("g", patch.cells, (2, 2), dtype=np.float32)

    patch_data = PatchData(patch, target=target, fields=[f, g])

    if target == ps.Target.SYCL:
        import dpctl.tensor as dpnp

        assert patch_data.array_module is dpnp
        xp_ndarray = dpnp.usm_ndarray
    else:
        assert patch_data.array_module is xp
        xp_ndarray = xp.ndarray

    assert isinstance(patch_data[f], xp_ndarray)
    assert patch_data[f].shape == (5, 8, 12, 3)
    assert patch_data[f].dtype == patch_data.default_dtype

    assert isinstance(patch_data[g], xp_ndarray)
    assert patch_data[g].shape == (4, 7, 11, 2, 2)
    assert patch_data[g].dtype == np.dtype(np.float32)

    h = TensorField("h", patch.cells, (), ghost_layers=2)
    patch_data.add_field(h)

    assert isinstance(patch_data[h], xp_ndarray)
    assert patch_data[h].shape == (8, 11, 15)
    assert patch_data[h].dtype == patch_data.default_dtype

    patch_data[h][:] = -4.2

    h_arr = patch_data.asnumpy(h)
    assert isinstance(h_arr, np.ndarray)
    assert h_arr is not patch_data[h]
    np.testing.assert_array_equal(h_arr, -4.2)

    k = TensorField("k", patch.cells, (), ghost_layers=2)
    patch_data.add_field(k)

    xp.testing.assert_array_equal(patch_data[k], 0.0)

    h_old, k_old = patch_data[h], patch_data[k]
    patch_data.swap(h, k)
    assert patch_data[h] is k_old
    assert patch_data[k] is h_old

    # setting arrays explicitly is allowed; no shape checking takes place
    patch_data[k] = np.ones((2, 3, 4))
    np.testing.assert_array_equal(patch_data[k], np.ones((2, 3, 4)))


@pytest.mark.parametrize(
    "target", [t for t in ps.Target.available_targets() if not t.is_vector_cpu()]
)
def test_patch_data_reduction_vars(target, xp):
    if target == ps.Target.SYCL:
        pytest.xfail("Reduction variables not supported for SYCL, yet")

    patch = Patch("P", (1, 1, 1), num_vertices=(5, 8, 12))
    patch_data = PatchData(patch, target=target)

    p, q = ps.symbols("p, q", ps.numeric_t)
    r, s = ps.symbols("r, s", "float32")
    t = ps.symbols("t", ps.index_t)

    patch_data.add_reduction(p, "max")
    xp.testing.assert_array_equal(
        patch_data[p], xp.array([-np.inf], dtype=np.float64), strict=True
    )

    patch_data.add_reduction(q, "min")
    xp.testing.assert_array_equal(
        patch_data[q], xp.array([np.inf], dtype=np.float64), strict=True
    )

    patch_data.add_reduction(r, "+")
    xp.testing.assert_array_equal(
        patch_data[r], xp.array([0.0], dtype=np.float32), strict=True
    )

    patch_data.add_reduction(s, "*")
    xp.testing.assert_array_equal(
        patch_data[s], xp.array([1.0], dtype=np.float32), strict=True
    )

    patch_data.add_reduction(t, 142)
    xp.testing.assert_array_equal(
        patch_data[t], xp.array([142.0], dtype=np.int64), strict=True
    )


def test_patch_data_errors():
    patch = Patch("P", (1, 1, 1), num_vertices=(5, 8, 12))

    f = TensorField("f", patch.vertices, (3,), dtype=ps.numeric_t)
    g = TensorField("g", patch.cells, (2, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        _ = PatchData(patch, {"c": 1.3}, fields=[f, g])

    with pytest.raises(ValueError):
        _ = PatchData(patch, fields=[f, f])

    patch2 = Patch("Q", (1, 2, 3))
    j = TensorField("j", patch2.vertices, ())

    with pytest.raises(ValueError):
        _ = PatchData(patch, fields=[j])

    not_a_field = sp.Symbol("not_a_field")

    with pytest.raises(ValueError):
        _ = PatchData(patch, fields=[not_a_field])

    nonsense_type = ps.types.PsCustomType("nonsense")
    V = ps.symbols("V", dtype=nonsense_type)

    with pytest.raises(ValueError):
        _ = PatchData(patch, {V: 42})

    with pytest.raises(ValueError):
        _ = PatchData(patch, default_dtype=nonsense_type)

    with pytest.raises(ValueError):
        _ = PatchData(patch, index_dtype=np.float32)


def test_pyvista_bridge():
    pv = pytest.importorskip("pyvista")

    patch = Patch("P", (1.3, 2.4, 1.8), num_vertices=(5, 8, 12))

    f = TensorField("f", patch.vertices, (3,), dtype=ps.numeric_t)
    g = TensorField("g", patch.cells, (), dtype=np.float32, ghost_layers=1)

    patch_data = PatchData(patch, fields=[f, g])

    rng = np.random.default_rng(42)

    patch_data[f][:] = rng.random(patch_data.num_vertices + (3,))
    patch_data[g][1:-1, 1:-1, 1:-1] = rng.random(patch_data.num_cells)

    mesh = patch_data.viz().get_mesh()
    assert mesh.dimensionality == 3
    assert mesh.dimensions == (5, 8, 12)
    assert mesh.n_points == 5 * 8 * 12
    assert mesh.n_cells == 4 * 7 * 11
    assert mesh.bounds == pv.BoundsTuple(0.0, 1.3, 0.0, 2.4, 0.0, 1.8)

    np.testing.assert_array_equal(
        mesh.point_data["f"],
        patch_data[f].swapaxes(0, 2).reshape((5 * 8 * 12, 3)),
    )

    np.testing.assert_array_equal(
        mesh.cell_data["g"], patch_data[g][1:-1, 1:-1, 1:-1].flatten(order="F")
    )


def test_pyvista_bridge_2d():
    pv = pytest.importorskip("pyvista")

    patch = Patch("P", (4.0, 12.0), num_vertices=(10, 12))

    f = TensorField("f", patch.vertices, (2,), dtype=ps.numeric_t)
    g = TensorField("g", patch.cells, (2, 2), dtype=np.float32, ghost_layers=1)

    patch_data = PatchData(patch, fields=[f, g])

    rng = np.random.default_rng(42)

    patch_data[f][:] = rng.random(patch_data.num_vertices + (2,))
    patch_data[g][1:-1, 1:-1] = rng.random(patch_data.num_cells + (2, 2))

    mesh = patch_data.viz().get_mesh()
    assert mesh.dimensionality == 2
    assert mesh.dimensions == (10, 12, 1)
    assert mesh.n_points == 10 * 12
    assert mesh.n_cells == 9 * 11
    assert mesh.bounds == pv.BoundsTuple(0.0, 4.0, 0.0, 12.0, 0.0, 0.0)

    assert "f" in mesh.point_data
    assert "g" in mesh.cell_data

    np.testing.assert_array_equal(
        mesh.point_data["f"],
        patch_data[f].swapaxes(0, 1).reshape((10 * 12, 2)),
    )

    np.testing.assert_array_equal(
        mesh.cell_data["g"],
        patch_data[g][1:-1, 1:-1].swapaxes(0, 1).reshape((9 * 11, 4)),
    )
