import pytest
import sympy as sp
import numpy as np

import pystencils as ps
from pystencils.sympyextensions.fast_approximation import (
    fast_division,
    fast_inv_sqrt,
    fast_sqrt,
    insert_fast_divisions,
    insert_fast_sqrts,
)

from pystencils.backend.exceptions import MaterializationError

try:
    import cupy as cp
except ImportError:
    pytest.skip("cupy not available", allow_module_level=True)


def test_fast_sqrt():
    f, g = ps.fields("f, g: float32[2D]")

    expr_original = sp.sqrt(f[0, 0] + f[1, 0])

    #   Check that `insert_fast_sqrts` can handle lists
    assert len(insert_fast_sqrts([expr_original])[0].atoms(fast_sqrt)) == 1

    expr_substituted = insert_fast_sqrts(expr_original)
    assert len(expr_substituted.atoms(fast_sqrt)) == 1

    ker_fast_sqrt = ps.create_kernel(
        ps.Assignment(g[0, 0], expr_substituted), target=ps.Target.GPU
    ).compile()
    assert "__fsqrt_rn" in ker_fast_sqrt.code

    ker_sqrt = ps.create_kernel(
        ps.Assignment(g[0, 0], expr_original), target=ps.Target.GPU
    ).compile()

    rng = np.random.default_rng()
    f_arr = cp.asarray(5.0 * rng.random((16, 8), dtype=np.float32))
    g_arr = cp.zeros_like(f_arr)
    g_arr_ref = cp.zeros_like(f_arr)

    ker_sqrt(f=f_arr, g=g_arr_ref)
    ker_fast_sqrt(f=f_arr, g=g_arr)

    cp.testing.assert_allclose(g_arr, g_arr_ref, rtol=1e-6)


def test_fast_inverse_sqrt():
    f, g = ps.fields("f, g: float32[2D]")

    ac_original = ps.AssignmentCollection(
        [ps.Assignment(g(), 3 / sp.sqrt(f[0, 0] + f[1, 0]))]
    )
    ac_subsituted = insert_fast_sqrts(ac_original)
    assert len(ac_subsituted.atoms(fast_inv_sqrt)) == 1

    ker_fast_sqrt = ps.create_kernel(ac_subsituted, target=ps.Target.GPU).compile()
    assert "__frsqrt_rn" in ker_fast_sqrt.code

    ker_sqrt = ps.create_kernel(ac_original, target=ps.Target.GPU).compile()

    rng = np.random.default_rng()
    f_arr = cp.asarray(5.0 * rng.random((16, 8), dtype=np.float32))
    g_arr = cp.zeros_like(f_arr)
    g_arr_ref = cp.zeros_like(f_arr)

    ker_sqrt(f=f_arr, g=g_arr_ref)
    ker_fast_sqrt(f=f_arr, g=g_arr)

    cp.testing.assert_allclose(g_arr, g_arr_ref, rtol=1e-6)


def test_fast_divisions():
    f, g = ps.fields("f, g: float32[2D]")
    expr = f[0, 0] / f[1, 0]
    assert len(insert_fast_divisions(expr).atoms(fast_division)) == 1

    expr = 1 / f[0, 0] * 2 / f[0, 1]
    expr_substituted = insert_fast_divisions(expr)
    assert len(expr_substituted.atoms(fast_division)) == 1

    ker_fast_division = ps.create_kernel(
        ps.Assignment(g[0, 0], expr_substituted), target=ps.Target.GPU
    ).compile()

    ker_regular_division = ps.create_kernel(
        ps.Assignment(g[0, 0], expr), target=ps.Target.GPU
    ).compile()

    assert "__fdividef" in ker_fast_division.code

    rng = np.random.default_rng()
    f_arr = cp.asarray(5.0 * rng.random((16, 8), dtype=np.float32))
    g_arr = cp.zeros_like(f_arr)
    g_arr_ref = cp.zeros_like(f_arr)

    ker_regular_division(f=f_arr, g=g_arr_ref)
    ker_fast_division(f=f_arr, g=g_arr)

    cp.testing.assert_allclose(g_arr, g_arr_ref, rtol=1e-6)


def test_fast_approximations_fail():
    f, g = ps.fields("f, g: float32[2D]")
    expr = 1 / f[0, 0] * 2 / f[0, 1] + 2 / sp.sqrt(f[-1, 0])
    expr_substituted = insert_fast_divisions(insert_fast_sqrts(expr))
    
    assert len(expr_substituted.atoms(fast_division)) == 1
    assert len(expr_substituted.atoms(fast_inv_sqrt)) == 1

    with pytest.raises(MaterializationError):
        _ = ps.create_kernel(ps.Assignment(g(), expr_substituted), target=ps.Target.CPU)
