import sympy as sp
import pystencils as ps
import numpy as np
from pystencils.slicing import get_periodic_boundary_functor, remove_ghost_layers
import pytest


# ---------------------------------------------------------------------------
#   Symbolic discretization checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stencil", [ps.Staggering.D2Q5,
                                     ps.Staggering.D2Q9,
                                     ps.Staggering.D3Q7,
                                     ps.Staggering.D3Q19,
                                     ps.Staggering.D3Q27])
@pytest.mark.parametrize("derivative", [0, 1])
def test_flux_stencil(stencil, derivative):
    """The discrete flux/continuity reference the expected number of accesses."""
    L = (40, ) * stencil.D
    dh = ps.create_data_handling(L, periodicity=True, default_target=ps.Target.CPU)
    c = dh.add_array('c', values_per_cell=1)
    j = ps.StaggeredField("j", stencil)

    def gradient(f):
        return sp.Matrix([ps.fd.diff(f, i) for i in range(dh.dim)])

    eq = [sp.Matrix([sp.Symbol(f"a_{i}") * c.center for i in range(dh.dim)]),
          gradient(c)][derivative]
    disc = ps.fd.FVM1stOrder(c, flux=eq)

    # continuity: divergence over all full-stencil directions (Q - 1 edges)
    continuity_assignments = disc.discrete_continuity(j).assignments
    assert [len(a.rhs.atoms(ps.StaggeredFieldAccess)) for a in continuity_assignments] == \
           [stencil.Q - 1] * len(continuity_assignments)

    # flux: each edge flux averages exactly two cell values
    flux_assignments = disc.discrete_flux(j).assignments
    assert [len(a.rhs.atoms(ps.field.Field.Access)) for a in flux_assignments] == \
           [2] * len(flux_assignments)


@pytest.mark.parametrize("stencil", [ps.Staggering.D2Q5,
                                     ps.Staggering.D2Q9,
                                     ps.Staggering.D3Q7,
                                     ps.Staggering.D3Q19,
                                     ps.Staggering.D3Q27])
def test_source_stencil(stencil):
    """A source term adds the expected number of cell accesses to the continuity."""
    L = (40, ) * stencil.D
    dh = ps.create_data_handling(L, periodicity=True, default_target=ps.Target.CPU)
    c = dh.add_array('c', values_per_cell=1)
    j = ps.StaggeredField("j", stencil)

    continuity_ref = ps.fd.FVM1stOrder(c).discrete_continuity(j)

    for eq in [c.center] + [ps.fd.diff(c, i) for i in range(dh.dim)]:
        disc = ps.fd.FVM1stOrder(c, source=eq)
        rhs = disc.discrete_continuity(j).assignments[0].rhs
        rhs2 = continuity_ref.assignments[0].rhs
        diff = sp.simplify(rhs - rhs2)
        if type(eq) is ps.field.Field.Access:
            assert len(diff.atoms(ps.field.Field.Access)) == 1
        else:
            assert len(diff.atoms(ps.field.Field.Access)) == 2


@pytest.mark.parametrize("stencil", [ps.Staggering.D2Q5, ps.Staggering.D2Q9])
def test_ek(stencil):
    """discrete_flux/discrete_continuity match a hand-written EK discretization."""
    D = sp.Symbol("D")
    z = sp.Symbol("z")

    dh = ps.create_data_handling((40, 40), periodicity=True, default_target=ps.Target.CPU)
    c = dh.add_array('c', values_per_cell=1)
    j = ps.StaggeredField("j", stencil)
    Phi = dh.add_array('Φ', values_per_cell=1)

    def gradient(f):
        return sp.Matrix([ps.fd.diff(f, i) for i in range(dh.dim)])

    flux_eq = -D * gradient(c) + D * z * c.center * gradient(Phi)

    disc = ps.fd.FVM1stOrder(c, flux_eq)
    flux_assignments = disc.discrete_flux(j).assignments
    continuity_assignments = disc.discrete_continuity(j).assignments

    # manual discretization
    x_staggered = c[1, 0] - c[0, 0] - z * (c[1, 0] + c[0, 0]) / 2 * (Phi[1, 0] - Phi[0, 0])
    y_staggered = c[0, 1] - c[0, 0] - z * (c[0, 1] + c[0, 0]) / 2 * (Phi[0, 1] - Phi[0, 0])
    xy_staggered = (c[1, 1] - c[0, 0]) / sp.sqrt(2) - \
        z * (c[1, 1] + c[0, 0]) / 2 * (Phi[1, 1] - Phi[0, 0]) / sp.sqrt(2)
    xY_staggered = (c[1, -1] - c[0, 0]) / sp.sqrt(2) - \
        z * (c[1, -1] + c[0, 0]) / 2 * (Phi[1, -1] - Phi[0, 0]) / sp.sqrt(2)
    A0 = (1 + sp.sqrt(2) if j.stencil.Q == 9 else 1)

    divergence = sum([j.face[i] for i in range(1, j.stencil.Q)])

    @ps.flow.block
    def update_block(let):
        let.store[c.center] = c.center - divergence

    @ps.flow.block
    def flux_block(let):
        let.store[j["n"]] = -D * y_staggered / A0
        let.store[j["e"]] = -D * x_staggered / A0
        if j.stencil.Q == 9:
            let.store[j["ne"]] = -D * xy_staggered / A0
            let.store[j["se"]] = -D * xY_staggered / A0

    for a, b in zip(flux_block.assignments, flux_assignments):
        assert type(a.lhs) is type(b.lhs)
        assert sp.simplify(a.rhs - b.rhs) == 0
    for a, b in zip(update_block.assignments, continuity_assignments):
        assert type(a.lhs) is type(b.lhs)
        assert a.rhs == b.rhs


def test_fvm_staggered_simplification():
    """The generated flux kernel is free of the degenerate loop bound `x - 1 < x - 1`."""
    D = sp.Symbol("D")

    c = ps.fields("c: float64[2D]", layout='fzyx')
    j = ps.StaggeredField("j", ps.Staggering.D2Q9)

    grad_c = sp.Matrix([ps.fd.diff(c, i) for i in range(c.spatial_dimensions)])
    ek = ps.fd.FVM1stOrder(c, flux=-D * grad_c)

    code = ps.get_code_str(ps.create_kernel(ek.discrete_flux(j)))
    assert '_size_c_0 - 1 < _size_c_0 - 1' not in code


# ---------------------------------------------------------------------------
#   VOF advection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stencil", [ps.Staggering.D2Q9, ps.Staggering.D3Q27])
def test_vof_structure(stencil):
    """VOF writes one flux per stored edge, referencing the density and velocity."""
    dim = stencil.D
    c = ps.fields(f"c : double[{dim}D]", layout='fzyx')
    u = ps.fields(f"u({dim}) : double[{dim}D]", layout='fzyx')
    j = ps.StaggeredField("j", stencil)

    assignments = ps.fd.VOF(j, u, c).assignments

    # one store per stored edge (the staggered entries excluding the centre)
    assert len(assignments) == stencil.Q // 2
    for a in assignments:
        assert isinstance(a.lhs, ps.StaggeredFieldAccess)

    referenced = {f.field.name
                  for a in assignments
                  for f in a.rhs.atoms(ps.field.Field.Access)}
    assert c.name in referenced
    assert u.name in referenced


# ---------------------------------------------------------------------------
#   Kernel compilation + execution smoke test
# ---------------------------------------------------------------------------

def _fill_periodic_flux(j_array, dim):
    """Copy the periodic images into the staggered flux ghost layers."""
    for ax in range(dim):
        lo = [slice(None)] * j_array.ndim
        lo_src = [slice(None)] * j_array.ndim
        hi = [slice(None)] * j_array.ndim
        hi_src = [slice(None)] * j_array.ndim
        lo[ax], lo_src[ax] = 0, -2
        hi[ax], hi_src[ax] = -1, 1
        j_array[tuple(lo)] = j_array[tuple(lo_src)]
        j_array[tuple(hi)] = j_array[tuple(hi_src)]


def test_fvm_kernels_run():
    """The diffusion flux + continuity kernels compile, run, and conserve mass."""
    dim = 2
    L = (8, 8)
    stencil = ps.Staggering.D2Q9
    D = 0.1

    c = ps.fields(f"c : double[{dim}D]", layout='fzyx')
    j = ps.StaggeredField("j", stencil)
    grad_c = sp.Matrix([ps.fd.diff(c, i) for i in range(dim)])
    fvm = ps.fd.FVM1stOrder(c, flux=-D * grad_c)

    flux_kernel = ps.create_kernel(fvm.discrete_flux(j), ghost_layers=[(0, 1)] * dim).compile()
    pde_kernel = ps.create_kernel(fvm.discrete_continuity(j), ghost_layers=1).compile()
    pbc = get_periodic_boundary_functor(stencil, 1)

    c_arr = np.zeros([n + 2 for n in L])
    c_arr[L[0] // 2, L[1] // 2] = 1.0
    j_arr = np.zeros([n + 2 for n in L] + [stencil.Q // 2 + 1])
    pbc(c_arr)

    initial_mass = remove_ghost_layers(c_arr).sum()
    for _ in range(3):
        flux_kernel(c=c_arr, j=j_arr)
        _fill_periodic_flux(j_arr, dim)
        pde_kernel(c=c_arr, j=j_arr)
        pbc(c_arr)

    result = remove_ghost_layers(c_arr)
    assert np.all(np.isfinite(result))
    assert np.isclose(result.sum(), initial_mass)  # mass is conserved
