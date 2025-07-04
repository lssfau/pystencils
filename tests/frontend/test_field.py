import numpy as np
import pytest
import sympy as sp

import pystencils as ps
from pystencils import DEFAULTS, DynamicType, create_type, fields
from pystencils.field import (
    Field,
    FieldType,
    layout_string_to_tuple,
    spatial_layout_string_to_tuple,
)


def test_field_basic():
    f = Field.create_generic("f", spatial_dimensions=2)
    assert FieldType.is_generic(f)
    assert f.dtype == DynamicType.NUMERIC_TYPE
    assert f["E"] == f[1, 0]
    assert f["N"] == f[0, 1]
    assert "_" in f.center._latex("dummy")

    assert (
        f.index_to_physical(index_coordinates=sp.Matrix([0, 0]), staggered=False)[0]
        == 0
    )
    assert (
        f.index_to_physical(index_coordinates=sp.Matrix([0, 0]), staggered=False)[1]
        == 0
    )

    assert (
        f.physical_to_index(physical_coordinates=sp.Matrix([0, 0]), staggered=False)[0]
        == 0
    )
    assert (
        f.physical_to_index(physical_coordinates=sp.Matrix([0, 0]), staggered=False)[1]
        == 0
    )

    f1 = f.new_field_with_different_name("f1")
    assert f1.ndim == f.ndim
    assert f1.values_per_cell() == f.values_per_cell()

    f = Field.create_fixed_size("f", (10, 10), strides=(10, 1), dtype=np.float64)
    assert f.spatial_strides == (10, 1)
    assert f.index_strides == ()
    assert f.center_vector == sp.Matrix([f.center])
    assert f.dtype == create_type("float64")

    f1 = f.new_field_with_different_name("f1")
    assert f1.ndim == f.ndim
    assert f1.values_per_cell() == f.values_per_cell()
    assert f1.dtype == create_type("float64")

    f = Field.create_fixed_size("f", (8, 8, 2, 2), index_dimensions=2)
    assert f.center_vector == sp.Matrix([[f(0, 0), f(0, 1)], [f(1, 0), f(1, 1)]])
    field_access = f[1, 1]
    assert field_access.nr_of_coordinates == 2
    assert field_access.offset_name == "NE"
    neighbor = field_access.neighbor(coord_id=0, offset=-2)
    assert neighbor.offsets == (-1, 1)
    assert "_" in neighbor._latex("dummy")
    assert f.dtype == DynamicType.NUMERIC_TYPE

    f = Field.create_fixed_size("f", (8, 8, 2, 2, 2), index_dimensions=3)
    assert f.center_vector == sp.Array(
        [[[f(i, j, k) for k in range(2)] for j in range(2)] for i in range(2)]
    )
    assert f.dtype == DynamicType.NUMERIC_TYPE

    f = Field.create_generic("f", spatial_dimensions=5, index_dimensions=2)
    field_access = f[1, -1, 2, -3, 0](1, 0)
    assert field_access.offsets == (1, -1, 2, -3, 0)
    assert field_access.index == (1, 0)
    assert f.dtype == DynamicType.NUMERIC_TYPE


def test_field_description_parsing():
    f, g = fields("f(1), g(3): [2D]")

    assert f.dtype == g.dtype == DynamicType.NUMERIC_TYPE
    assert f.spatial_dimensions == g.spatial_dimensions == 2
    assert f.index_shape == (1,)
    assert g.index_shape == (3,)

    f = fields("f: dyn[3D]")
    assert f.dtype == DynamicType.NUMERIC_TYPE

    idx = fields("idx: dynidx[3D]")
    assert idx.dtype == DynamicType.INDEX_TYPE

    h = fields("h: float32[3D]")

    assert h.index_shape == ()
    assert h.spatial_dimensions == 3
    assert h.index_dimensions == 0
    assert h.dtype == create_type("float32")

    f: Field = fields("f(5, 5) : double[20, 20]")
    
    assert f.dtype == create_type("float64")
    assert f.spatial_shape == (20, 20)
    assert f.index_shape == (5, 5)
    assert f.neighbor_vector((1, 1)).shape == (5, 5)


def test_error_handling():
    struct_dtype = np.dtype(
        [("a", np.int32), ("b", np.float64), ("c", np.uint32)], align=True
    )
    Field.create_generic(
        "f", spatial_dimensions=2, index_dimensions=0, dtype=struct_dtype
    )
    with pytest.raises(ValueError) as e:
        Field.create_generic(
            "f", spatial_dimensions=2, index_dimensions=1, dtype=struct_dtype
        )
    assert "index dimension" in str(e.value)

    arr = np.array([[[(1,) * 3, (2,) * 3, (3,) * 3]] * 2], dtype=struct_dtype)
    Field.create_from_numpy_array("f", arr, index_dimensions=0)
    with pytest.raises(ValueError) as e:
        Field.create_from_numpy_array("f", arr, index_dimensions=1)
    assert "Structured arrays" in str(e.value)

    arr = np.zeros([3, 3, 3])
    Field.create_from_numpy_array("f", arr, index_dimensions=2)
    with pytest.raises(ValueError) as e:
        Field.create_from_numpy_array("f", arr, index_dimensions=3)
    assert "Too many" in str(e.value)

    Field.create_fixed_size(
        "f", (3, 2, 4), index_dimensions=0, dtype=struct_dtype, layout="reverse_numpy"
    )
    with pytest.raises(ValueError) as e:
        Field.create_fixed_size(
            "f",
            (3, 2, 4),
            index_dimensions=1,
            dtype=struct_dtype,
            layout="reverse_numpy",
        )
    assert "Structured arrays" in str(e.value)

    f = Field.create_fixed_size("f", (10, 10))
    with pytest.raises(ValueError) as e:
        f[1]
    assert "Wrong number of spatial indices" in str(e.value)

    f = Field.create_generic("f", spatial_dimensions=2, index_shape=(3,))
    with pytest.raises(ValueError) as e:
        f(3)
    assert "out of bounds" in str(e.value)

    f = Field.create_fixed_size("f", (10, 10, 3, 4), index_dimensions=2)
    with pytest.raises(ValueError) as e:
        f(3, 0)
    assert "out of bounds" in str(e.value)

    with pytest.raises(ValueError) as e:
        f(1, 0)(1, 0)
    assert "Indexing an already indexed" in str(e.value)

    with pytest.raises(ValueError) as e:
        f(1)
    assert "Wrong number of indices" in str(e.value)

    with pytest.raises(ValueError) as e:
        Field.create_generic("f", spatial_dimensions=2, layout="wrong")
    assert "Unknown layout descriptor" in str(e.value)

    assert layout_string_to_tuple("fzyx", dim=4) == (3, 2, 1, 0)
    with pytest.raises(ValueError) as e:
        layout_string_to_tuple("wrong", dim=4)
    assert "Unknown layout descriptor" in str(e.value)


def test_decorator_scoping():
    dst = fields("dst : double[2D]")

    def f1():
        a = sp.Symbol("a")

        def f2():
            b = sp.Symbol("b")

            @ps.kernel
            def decorated_func():
                dst[0, 0] @= a + b

            return decorated_func

        return f2

    assert f1()(), ps.Assignment(dst[0, 0], sp.Symbol("a") + sp.Symbol("b"))


def test_string_creation():
    x, y, z = fields("  x(4),    y(3,5) z : double[  3,  47]")
    assert x.index_shape == (4,)
    assert y.index_shape == (3, 5)
    assert z.spatial_shape == (3, 47)


def test_itemsize():

    x = fields("x: float32[1d]")
    y = fields("y:  float64[2d]")
    i = fields("i:  int16[1d]")

    assert x.itemsize == 4
    assert y.itemsize == 8
    assert i.itemsize == 2


def test_spatial_memory_layout_descriptors():
    assert (
        spatial_layout_string_to_tuple("AoS", 3)
        == spatial_layout_string_to_tuple("aos", 3)
        == spatial_layout_string_to_tuple("ZYXF", 3)
        == spatial_layout_string_to_tuple("zyxf", 3)
        == (2, 1, 0)
    )
    assert (
        spatial_layout_string_to_tuple("SoA", 3)
        == spatial_layout_string_to_tuple("soa", 3)
        == spatial_layout_string_to_tuple("FZYX", 3)
        == spatial_layout_string_to_tuple("fzyx", 3)
        == spatial_layout_string_to_tuple("f", 3)
        == spatial_layout_string_to_tuple("F", 3)
        == (2, 1, 0)
    )
    assert (
        spatial_layout_string_to_tuple("c", 3)
        == spatial_layout_string_to_tuple("C", 3)
        == (0, 1, 2)
    )

    assert spatial_layout_string_to_tuple("C", 5) == (0, 1, 2, 3, 4)

    with pytest.raises(ValueError):
        spatial_layout_string_to_tuple("aos", -1)

    with pytest.raises(ValueError):
        spatial_layout_string_to_tuple("aos", 4)


def test_memory_layout_descriptors():
    assert (
        layout_string_to_tuple("AoS", 4)
        == layout_string_to_tuple("aos", 4)
        == layout_string_to_tuple("ZYXF", 4)
        == layout_string_to_tuple("zyxf", 4)
        == (2, 1, 0, 3)
    )
    assert (
        layout_string_to_tuple("SoA", 4)
        == layout_string_to_tuple("soa", 4)
        == layout_string_to_tuple("FZYX", 4)
        == layout_string_to_tuple("fzyx", 4)
        == layout_string_to_tuple("f", 4)
        == layout_string_to_tuple("F", 4)
        == (3, 2, 1, 0)
    )
    assert (
        layout_string_to_tuple("c", 4)
        == layout_string_to_tuple("C", 4)
        == (0, 1, 2, 3)
    )

    assert layout_string_to_tuple("C", 5) == (0, 1, 2, 3, 4)

    with pytest.raises(ValueError):
        layout_string_to_tuple("aos", -1)

    with pytest.raises(ValueError):
        layout_string_to_tuple("aos", 5)


def test_staggered():

    # D2Q5
    j1, j2, j3 = fields(
        "j1(2), j2(2,2), j3(2,2,2) : double[2D]", field_type=FieldType.STAGGERED
    )

    assert j1[0, 1](1) == j1.staggered_access((0, sp.Rational(1, 2)))
    assert j1[0, 1](1) == j1.staggered_access(np.array((0, sp.Rational(1, 2))))
    assert j1[1, 1](1) == j1.staggered_access((1, sp.Rational(1, 2)))
    assert j1[0, 2](1) == j1.staggered_access((0, sp.Rational(3, 2)))
    assert j1[0, 1](1) == j1.staggered_access("N")
    assert j1[0, 0](1) == j1.staggered_access("S")
    assert j1.staggered_vector_access("N") == sp.Matrix([j1.staggered_access("N")])
    assert j1.staggered_stencil_name == "D2Q5"

    assert j1.physical_coordinates[0] == DEFAULTS.spatial_counters[0]
    assert j1.physical_coordinates[1] == DEFAULTS.spatial_counters[1]
    assert j1.physical_coordinates_staggered[0] == DEFAULTS.spatial_counters[0] + 0.5
    assert j1.physical_coordinates_staggered[1] == DEFAULTS.spatial_counters[1] + 0.5
    assert (
        j1.index_to_physical(index_coordinates=sp.Matrix([0, 0]), staggered=True)[0]
        == 0.5
    )
    assert (
        j1.index_to_physical(index_coordinates=sp.Matrix([0, 0]), staggered=True)[1]
        == 0.5
    )
    assert (
        j1.physical_to_index(physical_coordinates=sp.Matrix([0, 0]), staggered=True)[0]
        == -0.5
    )
    assert (
        j1.physical_to_index(physical_coordinates=sp.Matrix([0, 0]), staggered=True)[1]
        == -0.5
    )

    assert j2[0, 1](1, 1) == j2.staggered_access((0, sp.Rational(1, 2)), 1)
    assert j2[0, 1](1, 1) == j2.staggered_access("N", 1)
    assert j2.staggered_vector_access("N") == sp.Matrix(
        [j2.staggered_access("N", 0), j2.staggered_access("N", 1)]
    )

    assert j3[0, 1](1, 1, 1) == j3.staggered_access((0, sp.Rational(1, 2)), (1, 1))
    assert j3[0, 1](1, 1, 1) == j3.staggered_access("N", (1, 1))
    assert j3.staggered_vector_access("N") == sp.Matrix(
        [[j3.staggered_access("N", (i, j)) for j in range(2)] for i in range(2)]
    )

    # D2Q9
    k1, k2 = fields("k1(4), k2(2) : double[2D]", field_type=FieldType.STAGGERED)

    assert k1[1, 1](2) == k1.staggered_access("NE")
    assert k1[0, 0](2) == k1.staggered_access("SW")
    assert k1[0, 0](3) == k1.staggered_access("NW")

    a = k1.staggered_access("NE")
    assert a._staggered_offset(a.offsets, a.index[0]) == [
        sp.Rational(1, 2),
        sp.Rational(1, 2),
    ]
    a = k1.staggered_access("SW")
    assert a._staggered_offset(a.offsets, a.index[0]) == [
        sp.Rational(-1, 2),
        sp.Rational(-1, 2),
    ]
    a = k1.staggered_access("NW")
    assert a._staggered_offset(a.offsets, a.index[0]) == [
        sp.Rational(-1, 2),
        sp.Rational(1, 2),
    ]

    # sign reversed when using as flux field
    r = fields("r(2) : double[2D]", field_type=FieldType.STAGGERED_FLUX)
    assert r[0, 0](0) == r.staggered_access("W")
    assert -r[1, 0](0) == r.staggered_access("E")


# test_staggered()
