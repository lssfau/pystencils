import pytest
import sympy as sp

import pystencils as ps
from pystencils.grids.staggered_field import StaggeredField, StaggeredFieldAccess, Staggering
from pystencils.defaults import DEFAULTS


# ---------------------------------------------------------------------------
#   Staggering enum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["D2Q9", "D2Q5", "D3Q19", "D3Q27", "D3Q15", "D3Q7"])
def test_staggering_members(name):
    s = Staggering[name]
    assert s.name == name
    assert s.D == int(name[1])
    assert s.Q == int(name[3:])
    # positive entries = centre + stored edges = Q // 2 + 1
    assert len(s.entries) == s.Q // 2 + 1
    assert s.entries == s.staggered_entries
    assert s.entries[0] == (0,) * s.D


def test_staggering_full_stencil_covers_all_directions():
    s = Staggering.D2Q9
    full = [s[i] for i in range(s.Q)]
    assert len(full) == s.Q
    assert len(set(full)) == s.Q               # all distinct
    assert full[0] == (0, 0)                   # centre first
    for e in s.entries[1:]:                     # every edge and its inverse present
        assert e in full
        assert tuple(-c for c in e) in full


def test_staggering_index_lookups():
    s = Staggering.D2Q9
    for i, d in enumerate(s.staggered_entries):
        assert s.staggered_index(d) == i
        assert s.staggered_direction(i) == d


def test_staggering_inverse_maps_to_same_index():
    s = Staggering.D2Q9
    for d in s.staggered_entries[1:]:
        inv = tuple(-c for c in d)
        assert s.staggered_index(inv) == s.staggered_index(d)



# ---------------------------------------------------------------------------
#   StaggeredGrid construction & properties
# ---------------------------------------------------------------------------

def test_staggered_grid_construction():
    stencil = Staggering.D2Q9
    sg = StaggeredField("f", stencil)
    assert sg.name == "f"
    assert sg.stencil is stencil
    assert sg.dtype == ps.DynamicType.NUMERIC_TYPE


def test_staggered_grid_from_staggering():
    """A Staggering member is stored as-is on the field."""
    sg = StaggeredField("g", Staggering.D2Q9)
    assert sg.stencil is Staggering.D2Q9
    assert sg.stencil.Q == 9


def test_staggered_grid_explicit_dtype():
    sg = StaggeredField("h", Staggering.D2Q9, dtype="float32")
    assert sg.dtype == ps.types.create_numeric_type("float32")


def test_staggered_grid_equality_and_hash():
    a = StaggeredField("f", Staggering.D2Q9)
    b = StaggeredField("f", Staggering.D2Q9)
    assert a == b
    assert hash(a) == hash(b)

    c = StaggeredField("g", Staggering.D2Q9)
    assert a != c

    d = StaggeredField("f", Staggering.D2Q5)
    assert a != d

    assert a != "not a grid"


def test_staggered_grid_str_repr():
    sg = StaggeredField("f", Staggering.D2Q9)
    s = str(sg)
    assert "f" in s
    assert "D2Q9" in s

    r = repr(sg)
    assert "StaggeredField" in r


def test_staggered_grid_latex():
    sg = StaggeredField("my_field", Staggering.D2Q9)
    latex = sg._repr_latex_()
    assert "$" in latex
    assert "my-field" in latex  # underscores replaced with hyphens


# ---------------------------------------------------------------------------
#   Buffer spec
# ---------------------------------------------------------------------------

def test_buffer_spec():
    sg = StaggeredField("f", Staggering.D2Q9)
    spec = sg._buffer_spec
    assert "f" in spec.base_ptr_name
    assert spec.rank == 3  # 2 spatial + 1 pdf dimension
    # Q=9 => Q//2+1 = 5 pdf values in the staggered half
    assert spec.shape[-1] == 5


def test_buffer_spec_3d():
    sg = StaggeredField("f", Staggering.D3Q19)
    spec = sg._buffer_spec
    assert spec.rank == 4  # 3 spatial + 1 pdf dimension
    assert spec.shape[-1] == 19 // 2 + 1


# ---------------------------------------------------------------------------
#   Stored-edge access:  u[...]
# ---------------------------------------------------------------------------

def test_edge_access_basic():
    sg = StaggeredField("f", Staggering.D2Q9)
    acc = sg["e"]
    assert isinstance(acc, StaggeredFieldAccess)
    assert acc.staggered_grid is sg
    assert acc.offsets == sp.Tuple(0, 0)
    assert int(acc.index) == sg.stencil.staggered_index((1, 0))


def test_edge_access_with_offset():
    sg = StaggeredField("f", Staggering.D2Q9)
    acc = sg[1, -1, "e"]
    assert acc.offsets == sp.Tuple(1, -1)
    assert int(acc.index) == sg.stencil.staggered_index((1, 0))


def test_edge_access_by_index():
    sg = StaggeredField("f", Staggering.D2Q9)
    acc = sg[2]
    assert acc.index == sp.Integer(2)
    assert tuple(int(o) for o in acc.offsets) == (0, 0)

    acc2 = sg[1, -1, 2]
    assert acc2.index == sp.Integer(2)
    assert tuple(int(o) for o in acc2.offsets) == (1, -1)


def test_edge_access_all_stored_directions():
    sg = StaggeredField("f", Staggering.D2Q9)
    for i, d in enumerate(sg.stencil.staggered_entries):
        dir_str = ps.stencil.offset_to_direction_string(d)
        acc = sg[dir_str]
        assert isinstance(acc, StaggeredFieldAccess)
        assert int(acc.index) == i
        assert tuple(int(o) for o in acc.offsets) == (0, 0)


def test_edge_access_rejects_non_stored_direction():
    sg = StaggeredField("f", Staggering.D2Q9)
    # "w" is not a stored edge (only "e" is stored) -> must use .face
    with pytest.raises(ValueError):
        sg["w"]


def test_edge_access_rejects_out_of_range_index():
    sg = StaggeredField("f", Staggering.D2Q9)
    with pytest.raises(IndexError):
        sg[99]


def test_edge_access_direction_property():
    sg = StaggeredField("f", Staggering.D2Q9)
    acc = sg[0]
    assert acc.direction == sg.stencil.staggered_direction(0)


# ---------------------------------------------------------------------------
#   Flux access:  u.face[...]
# ---------------------------------------------------------------------------

def test_face_stored_direction_is_plain_access():
    sg = StaggeredField("f", Staggering.D2Q9)
    assert sg.face["e"] == sg["e"]


def test_face_reversed_direction_negates_neighbour():
    sg = StaggeredField("f", Staggering.D2Q9)
    # "w" -> -(east edge of the left neighbour); "s" -> -(north edge below)
    assert sg.face["w"] == -sg[-1, 0, "e"]
    assert sg.face["s"] == -sg[0, -1, "n"]


def test_face_by_full_direction_index():
    sg = StaggeredField("f", Staggering.D2Q5)
    stencil = sg.stencil
    for idx in range(1, stencil.Q):
        ci = stencil[idx]
        flux = sg.face[idx]
        if ci in stencil.staggered_entries:
            assert isinstance(flux, StaggeredFieldAccess)
            assert tuple(int(o) for o in flux.offsets) == (0, 0)
        else:
            assert isinstance(flux, sp.Mul)
            (acc,) = [a for a in flux.args if isinstance(a, StaggeredFieldAccess)]
            assert tuple(int(o) for o in acc.offsets) == ci


def test_face_offset_composes_with_reflection():
    sg = StaggeredField("f", Staggering.D2Q9)
    # offset (1, 0) + reflected "w" shift (-1, 0) -> home cell east edge
    assert sg.face[1, 0, "w"] == -sg[0, 0, "e"]


# ---------------------------------------------------------------------------
#   get_buffer_indices / get_buffer_spec
# ---------------------------------------------------------------------------

def test_get_buffer_indices():
    sg = StaggeredField("f", Staggering.D2Q9)
    acc = sg[2, -3, 1]
    indices = acc.get_buffer_indices()
    ctr0, ctr1 = DEFAULTS.spatial_counters[:2]
    assert indices == (ctr0 + 2, ctr1 + (-3), sp.Integer(1))


def test_get_buffer_spec():
    sg = StaggeredField("f", Staggering.D2Q9)
    acc = sg[0]
    assert acc.get_buffer_spec() is sg._buffer_spec


# ---------------------------------------------------------------------------
#   Printing
# ---------------------------------------------------------------------------

def test_sympystr():
    sg = StaggeredField("f", Staggering.D2Q9)
    acc = sg[1, 0, 2]
    s = sp.printing.sstr(acc)
    assert "f" in s
    assert "1" in s


def test_latex_printing():
    sg = StaggeredField("f", Staggering.D2Q9)
    acc = sg[1]
    latex = sp.latex(acc)
    assert "f" in latex


# ---------------------------------------------------------------------------
#   SymPy expression integration
# ---------------------------------------------------------------------------

def test_sympy_expr_arithmetic():
    sg = StaggeredField("f", Staggering.D2Q9)
    a = sg[0]
    b = sg[1, 0, 0]
    assert isinstance(a + b, sp.Expr)
    assert isinstance(2 * a - b, sp.Expr)


def test_sympy_subs():
    sg = StaggeredField("f", Staggering.D2Q9)
    a = sg[0]
    x = sp.Symbol("x")
    assert (x * a).subs(x, 3) == 3 * a


def test_sympy_equality():
    sg = StaggeredField("f", Staggering.D2Q9)
    assert sg[1] == sg[1]


def test_sympy_hash_consistency():
    sg = StaggeredField("f", Staggering.D2Q9)
    a = sg[1]
    b = sg[1]
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


# ---------------------------------------------------------------------------
#   3D stencils
# ---------------------------------------------------------------------------

def test_3d_staggered_grid():
    sg = StaggeredField("f3d", Staggering.D3Q19)
    assert sg.stencil.D == 3
    acc = sg[0]
    indices = acc.get_buffer_indices()
    assert len(indices) == 4  # 3 spatial + 1 edge index


def test_3d_edge_access():
    sg = StaggeredField("f3d", Staggering.D3Q19)
    acc = sg[1, -1, 2, "e"]
    assert acc is not None
    assert tuple(int(o) for o in acc.offsets) == (1, -1, 2)




# ---------------------------------------------------------------------------
#   Vector staggered grid
# ---------------------------------------------------------------------------

def test_vector_grid_buffer_spec():
    sg = StaggeredField("j", Staggering.D2Q5, index_shape=(3,))
    spec = sg.get_buffer_spec()
    # shape: (size_x, size_y, Q//2+1, 3)
    assert spec.rank == 4
    assert spec.shape[2] == 3  # Q//2+1 for D2Q5
    assert spec.shape[3] == 3  # index_shape
    assert len(spec.strides) == 4


def test_vector_grid_access_buffer_indices():
    sg = StaggeredField("j", Staggering.D2Q5, index_shape=(2,))
    acc = sg[1](0)
    assert acc.vector_index == sp.Integer(0)
    indices = acc.get_buffer_indices()
    # spatial (2) + edge index + vector_index
    assert len(indices) == 4
    assert indices[-1] == sp.Integer(0)

    acc1 = sg[1](1)
    assert acc1.vector_index == sp.Integer(1)
    assert acc1.get_buffer_indices()[-1] == sp.Integer(1)


def test_vector_grid_scalar_access_unchanged():
    sg = StaggeredField("j", Staggering.D2Q5)
    acc = sg[1]
    assert acc.vector_index is None
    assert len(acc.get_buffer_indices()) == 3


def test_vector_grid_index_error_on_scalar():
    sg = StaggeredField("j", Staggering.D2Q5)
    acc = sg[1]
    with pytest.raises(IndexError):
        _ = acc(0)


def test_vector_face_and_component():
    sg = StaggeredField("j", Staggering.D2Q5, index_shape=(2,))
    # a reflected flux is -1 * (stored-edge access)
    flux = sg.face["s"]
    assert isinstance(flux, sp.Mul)
    assert flux == -sg[0, -1, "n"]
    # the stored edge supports component selection, and remains writable
    comp = sg[0, -1, "n"](1)
    assert comp.vector_index == sp.Integer(1)
    from pystencils.flow.flowgraph import Store
    store = Store(sg[1](0), flux)
    assert store.rhs.args  # non-trivial RHS


def test_bare_subscript_and_component():
    """u[...] gives the stored edge; (k) selects the component (the user's example)."""
    sg = StaggeredField("u", Staggering.D2Q9, index_shape=(2,))
    c = sg[1, -1, "e"](1)
    assert c.vector_index == sp.Integer(1)
    assert tuple(int(o) for o in c.offsets) == (1, -1)


def test_current_cell_direction_access():
    """A bare direction accesses the current cell, no offset needed: u['e'](0)."""
    sg = StaggeredField("u", Staggering.D2Q9, index_shape=(2,))
    acc = sg["e"](0)
    assert tuple(int(o) for o in acc.offsets) == (0, 0)
    assert acc.vector_index == sp.Integer(0)
    indices = acc.get_buffer_indices()
    ctr0, ctr1 = DEFAULTS.spatial_counters[:2]
    assert indices[0] == ctr0 and indices[1] == ctr1


def test_direction_string_case_insensitive():
    """Lowercase and uppercase direction strings are equivalent."""
    sg = StaggeredField("u", Staggering.D2Q9, index_shape=(2,))
    assert sg["e"](0) == sg["E"](0)
    assert sg["ne"](1) == sg["NE"](1)
    assert sg.face["w"] == sg.face["W"]
    assert sg[1, -1, "se"](0) == sg[1, -1, "SE"](0)


@pytest.mark.parametrize("stencil", [Staggering.D2Q5, Staggering.D2Q9])
def test_vector_fvm_discrete_flux(stencil):
    dim = 2
    c = ps.fields(f"c(2) : double[{dim}D]", layout='fzyx')
    j = StaggeredField("j", stencil, index_shape=(2,))
    D = sp.Symbol("D")
    flux_eq = sp.Matrix([[-D * ps.fd.diff(c(k), d) for k in range(2)] for d in range(dim)])
    fvm = ps.fd.FVM1stOrder(c, flux=flux_eq)
    flux_block = fvm.discrete_flux(j)
    n_half = stencil.Q // 2
    # One assignment per staggered direction per species component
    assert len(flux_block.assignments) == n_half * 2
    kernel = ps.create_kernel(flux_block, ghost_layers=0)
    assert kernel is not None

