"""Tests  (un)packing (from)to buffers."""

import pytest
import numpy as np

import pystencils as ps
from pystencils import Assignment, Field, FieldType, create_kernel
from pystencils.field import create_numpy_array_with_layout, layout_string_to_tuple
from pystencils.slicing import (
    get_ghost_region_slice,
    get_slice_before_ghost_layer,
)
from pystencils.stencil import direction_string_to_offset


@pytest.mark.parametrize("shape", [(32, 10), (10, 8, 6)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_full_scalar_field(shape, dtype):
    """Tests fully (un)packing a scalar field (from)to a buffer."""
    rank = len(shape)
    src_field, dst_field = ps.fields(f"src, dst: {dtype}[{rank}D]")
    buffer = Field.create_generic(
        "buffer", spatial_dimensions=1, field_type=FieldType.BUFFER, dtype=dtype
    )

    pack_eqs = [Assignment(buffer.center(), src_field.center())]
    pack_code = create_kernel(pack_eqs)
    ps.show_code(pack_code)

    rng = np.random.default_rng(0x5EED)
    src_arr = rng.random(shape, dtype=dtype)
    dst_arr = np.zeros_like(src_arr)
    buffer_arr = np.zeros(np.prod(dst_arr.shape), dtype=dtype)

    pack_kernel = pack_code.compile()
    pack_kernel(buffer=buffer_arr, src=src_arr)

    unpack_eqs = [Assignment(dst_field.center(), buffer.center())]

    unpack_code = create_kernel(unpack_eqs)

    unpack_kernel = unpack_code.compile()
    unpack_kernel(dst=dst_arr, buffer=buffer_arr)

    np.testing.assert_equal(src_arr, dst_arr)


@pytest.mark.parametrize("shape", [(32, 10), (10, 8, 6)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("direction", ["N", "S", "NW", "SW", "TNW", "B"])
def test_field_slice(shape, dtype, direction):
    """Tests (un)packing slices of a scalar field (from)to a buffer."""
    rank = len(shape)
    src_field, dst_field = ps.fields(f"src, dst: {dtype}[{rank}D]")
    buffer = Field.create_generic(
        "buffer", spatial_dimensions=1, field_type=FieldType.BUFFER, dtype=dtype
    )

    rng = np.random.default_rng(0x5EED)
    src_arr = rng.random(shape, dtype=dtype)
    dst_arr = np.zeros_like(src_arr)
    buffer_arr = np.zeros(np.prod(dst_arr.shape), dtype=dtype)

    slice_dir = direction_string_to_offset(direction, dim=rank)
    pack_slice = get_slice_before_ghost_layer(slice_dir)
    unpack_slice = get_ghost_region_slice(slice_dir)

    pack_eqs = [Assignment(buffer.center(), src_field.center())]

    pack_code = create_kernel(pack_eqs)

    pack_kernel = pack_code.compile()
    pack_kernel(buffer=buffer_arr, src=src_arr[pack_slice])

    # Unpack into ghost layer of dst_field in N direction
    unpack_eqs = [Assignment(dst_field.center(), buffer.center())]

    unpack_code = create_kernel(unpack_eqs)

    unpack_kernel = unpack_code.compile()
    unpack_kernel(buffer=buffer_arr, dst=dst_arr[unpack_slice])

    np.testing.assert_equal(src_arr[pack_slice], dst_arr[unpack_slice])


@pytest.mark.parametrize("shape", [(32, 10), (10, 8, 6)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_all_cell_values(shape, dtype):
    """Tests (un)packing all cell values of the a field (from)to a buffer."""
    values_per_cell = 19

    rank = len(shape)
    src_field, dst_field = ps.fields(
        f"src({values_per_cell}), dst({values_per_cell}): {dtype}[{rank}D]"
    )
    buffer = Field.create_generic(
        "buffer",
        spatial_dimensions=1,
        index_shape=(values_per_cell,),
        field_type=FieldType.BUFFER,
        dtype=dtype,
    )

    rng = np.random.default_rng(0x5EED)
    src_arr = rng.random(shape + (values_per_cell,), dtype=dtype)
    dst_arr = np.zeros_like(src_arr)
    buffer_arr = np.zeros((np.prod(dst_arr.shape), values_per_cell), dtype=dtype)

    pack_eqs = []
    for idx in range(values_per_cell):
        eq = Assignment(buffer(idx), src_field(idx))
        pack_eqs.append(eq)

    pack_code = create_kernel(pack_eqs)
    pack_kernel = pack_code.compile()
    pack_kernel(buffer=buffer_arr, src=src_arr)

    unpack_eqs = []

    for idx in range(values_per_cell):
        eq = Assignment(dst_field(idx), buffer(idx))
        unpack_eqs.append(eq)

    unpack_code = create_kernel(unpack_eqs)
    unpack_kernel = unpack_code.compile()
    unpack_kernel(buffer=buffer_arr, dst=dst_arr)

    np.testing.assert_equal(src_arr, dst_arr)


@pytest.mark.parametrize("shape", [(32, 10), (10, 8, 6)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_subset_cell_values(shape, dtype):
    """Tests (un)packing a subset of cell values of the a field (from)to a buffer."""
    values_per_cell = 19
    indices_to_pack = [1, 5, 7, 8, 10, 12, 13]

    rank = len(shape)
    src_field, dst_field = ps.fields(
        f"src({values_per_cell}), dst({values_per_cell}): {dtype}[{rank}D]"
    )
    buffer = Field.create_generic(
        "buffer",
        spatial_dimensions=1,
        index_shape=(len(indices_to_pack),),
        field_type=FieldType.BUFFER,
        dtype=dtype,
    )

    rng = np.random.default_rng(0x5EED)
    src_arr = rng.random(shape + (values_per_cell,), dtype=dtype)
    dst_arr = np.zeros_like(src_arr)
    buffer_arr = np.zeros((np.prod(dst_arr.shape), len(indices_to_pack)), dtype=dtype)

    pack_eqs = []
    for buffer_idx, cell_idx in enumerate(indices_to_pack):
        eq = Assignment(buffer(buffer_idx), src_field(cell_idx))
        pack_eqs.append(eq)

    pack_code = create_kernel(pack_eqs)
    pack_kernel = pack_code.compile()
    pack_kernel(buffer=buffer_arr, src=src_arr)

    unpack_eqs = []

    for buffer_idx, cell_idx in enumerate(indices_to_pack):
        eq = Assignment(dst_field(cell_idx), buffer(buffer_idx))
        unpack_eqs.append(eq)

    unpack_code = create_kernel(unpack_eqs)
    unpack_kernel = unpack_code.compile()
    unpack_kernel(buffer=buffer_arr, dst=dst_arr)

    mask_arr = np.ma.masked_where((src_arr - dst_arr) != 0, src_arr)
    np.testing.assert_equal(dst_arr, mask_arr.filled(int(0)))


@pytest.mark.parametrize("shape", [(32, 10), (10, 8, 6)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("layout", ["numpy", "fzyx", "zyxf", "reverse_numpy"])
def test_field_layouts(shape, dtype, layout):
    values_per_cell = 27
    rank = len(shape)

    rng = np.random.default_rng(0x5EED)
    src_arr = create_numpy_array_with_layout(
        shape + (values_per_cell,),
        layout_string_to_tuple(layout, rank + 1),
        dtype=dtype,
    )
    src_arr[:] = rng.random(shape + (values_per_cell,), dtype=dtype)
    dst_arr = np.zeros_like(src_arr)
    buffer_arr = np.zeros((np.prod(dst_arr.shape), values_per_cell), dtype=dtype)

    src_field = Field.create_from_numpy_array("src", src_arr, index_dimensions=1)
    dst_field = Field.create_from_numpy_array("dst", dst_arr, index_dimensions=1)
    buffer = Field.create_generic(
        "buffer",
        spatial_dimensions=1,
        index_shape=(values_per_cell,),
        field_type=FieldType.BUFFER,
        dtype=dtype,
    )

    pack_eqs = []
    for idx in range(values_per_cell):
        eq = Assignment(buffer(idx), src_field(idx))
        pack_eqs.append(eq)

    pack_code = create_kernel(pack_eqs)
    pack_kernel = pack_code.compile()
    pack_kernel(buffer=buffer_arr, src=src_arr)

    unpack_eqs = []

    for idx in range(values_per_cell):
        eq = Assignment(dst_field(idx), buffer(idx))
        unpack_eqs.append(eq)

    unpack_code = create_kernel(unpack_eqs)
    unpack_kernel = unpack_code.compile()
    unpack_kernel(buffer=buffer_arr, dst=dst_arr)


@pytest.mark.parametrize("shape", [(32, 10), (10, 8, 6)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_iteration_slices(shape, dtype):
    values_per_cell = 19

    rank = len(shape)
    src_field, dst_field = ps.fields(
        f"src({values_per_cell}), dst({values_per_cell}): {dtype}[{rank}D]"
    )
    buffer = Field.create_generic(
        "buffer",
        spatial_dimensions=1,
        index_shape=(values_per_cell,),
        field_type=FieldType.BUFFER,
        dtype=dtype,
    )

    rng = np.random.default_rng(0x5EED)
    src_arr = rng.random(shape + (values_per_cell,), dtype=dtype)
    dst_arr = np.zeros_like(src_arr)
    buffer_arr = np.zeros((np.prod(dst_arr.shape), values_per_cell), dtype=dtype)

    pack_eqs = []
    for idx in range(values_per_cell):
        eq = Assignment(buffer(idx), src_field(idx))
        pack_eqs.append(eq)

    #   Pack only the leftmost slice, only every second cell
    pack_slice = (slice(None, None, 2),) * (rank - 1) + (slice(0, 1, None),)

    #   Fill the entire array with data
    src_arr[(slice(None, None, 1),) * rank] = np.arange(values_per_cell)
    dst_arr.fill(0)

    config = ps.CreateKernelConfig(iteration_slice=pack_slice)

    pack_code = create_kernel(pack_eqs, config=config)
    pack_kernel = pack_code.compile()
    pack_kernel(buffer=buffer_arr, src=src_arr)

    unpack_eqs = []

    for idx in range(values_per_cell):
        eq = Assignment(dst_field(idx), buffer(idx))
        unpack_eqs.append(eq)

    config = ps.CreateKernelConfig(iteration_slice=pack_slice)

    unpack_code = create_kernel(unpack_eqs, config=config)
    unpack_kernel = unpack_code.compile()
    unpack_kernel(buffer=buffer_arr, dst=dst_arr)

    #   Check if only every second entry of the leftmost slice has been copied
    np.testing.assert_equal(dst_arr[pack_slice], src_arr[pack_slice])
    np.testing.assert_equal(dst_arr[(slice(1, None, 2),) * (rank - 1) + (0,)], 0)
    np.testing.assert_equal(
        dst_arr[(slice(None, None, 1),) * (rank - 1) + (slice(1, None),)], 0
    )
