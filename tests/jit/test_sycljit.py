from itertools import product

import numpy as np
import pytest
import sympy as sp

from pystencils import (
    DEFAULTS, AddReductionAssignment, Assignment, CreateKernelConfig, Field, Target, TypedSymbol,
    create_kernel, fields, make_slice)

try:
    import dpctl
    import dpctl.tensor as dpt
except ImportError:
    pytest.skip(reason="DPCTL is not available", allow_module_level=True)


@pytest.fixture(params=[d for d in dpctl.get_devices() if d.has_aspect_gpu or d.has_aspect_cpu])
def queue(request) -> dpctl.SyclQueue:
    return dpctl.SyclQueue(request.param)


@pytest.mark.parametrize("order", ["C", "F"])
def test_basic_sycl_kernel(queue, order, sycl_jit):
    dtype = "float32"
    f, g = fields(f"f, g: {dtype}[2D]")
    asm = Assignment(g.center(), 2.0 * f.center())
    config = CreateKernelConfig(target=Target.SYCL)
    ker = create_kernel(asm, config=config)
    kfunc = sycl_jit.compile(ker)

    rng = np.random.default_rng(0x5eed)
    f_np_arr = rng.random(size=(34, 26), dtype=dtype)

    if order == "F":
        f_np_arr = np.asfortranarray(f_np_arr)

    # not using from_numpy, it misses the order kwarg https://github.com/IntelPython/dpctl/issues/2057
    f_arr = dpt.asarray(f_np_arr, order=order, dtype=dtype, sycl_queue=queue)
    g_arr = dpt.zeros_like(f_arr, dtype=dtype, device=queue, order=order)

    queue.wait()
    kfunc(f=f_arr, g=g_arr, queue=queue)
    queue.wait()

    np.testing.assert_almost_equal(dpt.asnumpy(g_arr), 2.0 * f_np_arr)


def test_invalid_args(queue, sycl_jit):

    f, g = fields("f, g: float[2D]")
    asm = Assignment(f.center(), 2.0 * g.center())

    config = CreateKernelConfig(target=Target.SYCL, default_dtype="float")
    ker = create_kernel(asm, config=config)
    kfunc = sycl_jit.compile(ker)

    f_arr = dpt.zeros((34, 26), dtype="float32", device=queue)
    g_arr = dpt.zeros_like(f_arr, device=queue)

    #   Missing Arguments
    with pytest.raises(KeyError):
        kfunc(f=f_arr, queue=queue)

    with pytest.raises(KeyError):
        kfunc(g=g_arr, queue=queue)

    with pytest.raises(Exception):
        kfunc(f=f_arr, g=g_arr, queue=6)

    #   Extra arguments are ignored
    kfunc(f=f_arr, g=g_arr, x=2.1, queue=queue)
    queue.wait()


def test_argument_type_error(queue, sycl_jit):

    # queue = dpctl.SyclQueue(device)
    dtype = "float32"
    f, g = fields(f"f, g: {dtype}[2D]")
    c = sp.Symbol("c")
    asm = Assignment(f.center(), c * g.center())
    config = CreateKernelConfig(target=Target.SYCL, default_dtype=dtype)
    ker = create_kernel(asm, config=config)
    kfunc = sycl_jit.compile(ker)

    arr_fp32 = dpt.zeros((23, 12), dtype=np.dtype("float32"), sycl_queue=queue)

    arr_i16 = dpt.zeros((23, 12), dtype="int16", sycl_queue=queue)
    arr_i32 = dpt.zeros((23, 12), dtype="int32", sycl_queue=queue)

    with pytest.raises(TypeError):
        kfunc(f=arr_fp32, g=arr_i16, c=2.0, queue=queue)

    with pytest.raises(TypeError):
        kfunc(f=arr_i16, g=arr_fp32, c=2.0, queue=queue)

    with pytest.raises(TypeError):
        kfunc(f=arr_i32, g=arr_i32, c=2.0, queue=queue)

    with pytest.raises(TypeError):
        kfunc(f=arr_i16, g=arr_i16, c=[2.0], queue=queue)

    #   Wrong scalar types are OK, though
    kfunc(f=arr_fp32, g=arr_fp32, c=np.float16(1.0), queue=queue)


def test_shape_check(queue, sycl_jit):

    # queue = dpctl.SyclQueue(device)
    dtype = "float32"
    f, g = fields(f"f, g: {dtype}[2D]")
    asm = Assignment(g.center(), 2.0 * f.center())
    config = CreateKernelConfig(target=Target.SYCL)
    ker = create_kernel(asm, config=config)
    kfunc = sycl_jit.compile(ker)

    f_arr = dpt.zeros((41, 32), dtype=dtype, device=queue)
    g_arr = dpt.zeros_like(f_arr, dtype=dtype, device=queue)
    kfunc(f=f_arr, g=g_arr, queue=queue)
    queue.wait()

    #   Trivial scalar dimensions are OK
    f_arr = dpt.zeros((41, 32, 1), dtype=dtype, device=queue)
    g_arr = dpt.zeros_like(f_arr, dtype=dtype, device=queue)
    kfunc(f=f_arr, g=g_arr, queue=queue)
    queue.wait()

    with pytest.raises(ValueError):
        f_arr = dpt.zeros((41, 32), dtype=dtype, device=queue)
        g_arr = dpt.zeros((40, 32), dtype=dtype, device=queue)
        kfunc(f=f_arr, g=g_arr, queue=queue)

    with pytest.raises(ValueError):
        f_arr = dpt.zeros((41, 1), dtype=dtype, device=queue)
        g_arr = dpt.zeros((40, 32), dtype=dtype, device=queue)
        kfunc(f=f_arr, g=g_arr, queue=queue)

    with pytest.raises(ValueError):
        f_arr = dpt.zeros((41, 1), dtype=dtype, device=queue)
        g_arr = dpt.zeros((40,), dtype=dtype, device=queue)
        kfunc(f=f_arr, g=g_arr, queue=queue)


def test_fixed_shape(queue, sycl_jit):

    dtype = "float32"
    a = np.zeros((12, 23), dtype=dtype)
    b = np.zeros((13, 21), dtype=dtype)

    f = Field.create_from_numpy_array("f", a)
    g = Field.create_from_numpy_array("g", a)

    asm = Assignment(f.center(), 2.0 * g.center())
    config = CreateKernelConfig(
        target=Target.SYCL,
        default_dtype=np.dtype(dtype)
    )
    ker = create_kernel(asm, config=config)

    kfunc = sycl_jit.compile(ker)
    a_dpt = dpt.from_numpy(a, sycl_queue=queue)
    a_dpt_copy = dpt.from_numpy(a, sycl_queue=queue)
    b_dpt = dpt.from_numpy(b, sycl_queue=queue)

    kfunc(f=a_dpt, g=a_dpt_copy, queue=queue)
    queue.wait()

    with pytest.raises(ValueError):
        kfunc(f=b_dpt, g=a_dpt, queue=queue)

    with pytest.raises(ValueError):
        kfunc(f=a_dpt, g=b_dpt, queue=queue)


def test_scalar_field(queue, sycl_jit):
    dtype = "float32"
    f, g = fields(f"f(1), g: {dtype}[2D]")
    asm = Assignment(g(), f(0))

    config = CreateKernelConfig(target=Target.SYCL, default_dtype=dtype)
    ker = create_kernel(asm, config=config)
    kfunc = sycl_jit.compile(ker)

    spatial_shape = (31, 29)
    #   Both implicit and explicit scalar fields must be accepted
    for ishape_f, ishape_g in product(((), (1,)), ((), (1,))):
        queue.wait()
        rng = np.random.default_rng(0x5eed)
        f_arr = rng.random(size=spatial_shape + ishape_f, dtype=dtype)
        f_arr_dpt = dpt.from_numpy(f_arr, sycl_queue=queue)
        g_arr_dpt = dpt.zeros(spatial_shape + ishape_g, dtype=dtype, sycl_queue=queue)
        queue.wait()

        kfunc(f=f_arr_dpt, g=g_arr_dpt, queue=queue)
        queue.wait()

        g_arr = dpt.asnumpy(g_arr_dpt)
        queue.wait()
        np.testing.assert_allclose(
            g_arr.flatten(),
            f_arr.flatten(),
            err_msg=f"{f_arr_dpt.shape} and {g_arr_dpt.shape} failed")


def test_basic_sycl_kernel_with_scalar(queue, sycl_jit):
    dtype = "float32"
    c = sp.Symbol("c")
    f, g = fields(f"f, g: {dtype}[2D]")
    asm = Assignment(g.center(), c * f.center())
    config = CreateKernelConfig(target=Target.SYCL, default_dtype=dtype)
    ker = create_kernel(asm, config=config)
    kfunc = sycl_jit.compile(ker)

    rng = np.random.default_rng(0x5eed)
    f_np_arr = rng.random(size=(34, 26), dtype=dtype)
    f_arr = dpt.from_numpy(f_np_arr, sycl_queue=queue)
    g_arr = dpt.zeros_like(f_arr, dtype=dtype, sycl_queue=queue)

    kfunc(f=f_arr, g=g_arr, c=np.float32(2.0), queue=queue)
    queue.wait()

    np.testing.assert_almost_equal(dpt.asnumpy(g_arr), 2.0 * f_np_arr)


def test_invalid_queue(sycl_jit):
    dtype = "float32"
    c = sp.Symbol("c")
    f, g = fields(f"f, g: {dtype}[2D]")
    asm = Assignment(g.center(), c * f.center())
    config = CreateKernelConfig(target=Target.SYCL, default_dtype=dtype)
    ker = create_kernel(asm, config=config)
    kfunc = sycl_jit.compile(ker)
    queue = dpctl.SyclQueue(property="in_order")
    other_queue = dpctl.SyclQueue(property="enable_profiling")

    rng = np.random.default_rng(0x5eed)
    f_np_arr = rng.random(size=(34, 26), dtype=dtype)
    f_arr = dpt.from_numpy(f_np_arr, sycl_queue=queue)
    g_arr = dpt.zeros_like(f_arr, dtype=dtype, sycl_queue=queue)

    kfunc(f=f_arr, g=g_arr, c=np.float32(2.0))
    queue.wait()

    with pytest.raises(ValueError):
        kfunc(f=f_arr, g=g_arr, c=np.float32(2.0), queue=other_queue)

    wrong_f_arr = dpt.from_numpy(f_np_arr, sycl_queue=other_queue)
    with pytest.raises(ValueError):
        kfunc(f=wrong_f_arr, g=g_arr, c=np.float32(2.0))


def test_basic_sycl_kernel_manual_block_size(queue, sycl_jit):
    dtype = "float32"
    f, g = fields(f"f, g: {dtype}[2D]")
    asm = Assignment(g.center(), 2.0 * f.center())
    config = CreateKernelConfig(target=Target.SYCL)
    config.sycl.automatic_block_size = False
    config.gpu.manual_launch_grid = True
    ker = create_kernel(asm, config=config)
    kfunc = sycl_jit.compile(ker)

    rng = np.random.default_rng(0x5eed)
    f_np_arr = rng.random(size=(34, 26), dtype=dtype)
    f_arr = dpt.from_numpy(f_np_arr, device=queue)
    for bs in [(1, 1), (2, 2), (1, 2), (2, 1)]:
        grid = tuple(g//b for b, g in zip(bs, f_np_arr.shape))
        queue.wait()
        g_arr = dpt.zeros_like(f_arr, dtype=dtype, device=queue)
        queue.wait()
        kfunc.launch_config.block_size = bs
        kfunc.launch_config.grid_size = grid
        kfunc(f=f_arr, g=g_arr, queue=queue)
        queue.wait()
        np.testing.assert_almost_equal(dpt.asnumpy(g_arr), 2.0 * f_np_arr)


@pytest.mark.parametrize("assume_warp_aligned_block_size", [True, False])
def test_dynamic_launch_config(queue, sycl_jit, assume_warp_aligned_block_size):

    shape = (34, 26)
    dtype = "float32"
    f, g = fields(f"f, g: float[{shape[0]},{shape[1]}]")
    asm = Assignment(f.center(), 2.0 * g.center())

    config = CreateKernelConfig(target=Target.SYCL, default_dtype=dtype)
    config.sycl.automatic_block_size = False
    config.gpu.indexing_scheme = "linear3d"
    config.gpu.warp_size = 32
    config.gpu.assume_warp_aligned_block_size = assume_warp_aligned_block_size
    ker = create_kernel(asm, config=config)
    kfunc = sycl_jit.compile(ker)

    f_arr = dpt.zeros(shape, dtype=dtype, sycl_queue=queue)
    g_arr = dpt.zeros_like(f_arr, sycl_queue=queue)
    kfunc(f=f_arr, g=g_arr, queue=queue)

    kfunc.launch_config.fit_block_size((4, 4))
    kfunc(f=f_arr, g=g_arr, queue=queue)

    kfunc.launch_config.trim_block_size((4, 2))
    kfunc(f=f_arr, g=g_arr, queue=queue)

    queue.wait()


def test_automatic_launch_config(queue, sycl_jit):
    shape = (34, 26)
    dtype = "float32"
    f, g = fields(f"f, g: float[{shape[0]},{shape[1]}]")
    asm = Assignment(f.center(), 2.0 * g.center())

    config = CreateKernelConfig(target=Target.SYCL, default_dtype=dtype)
    config.sycl.automatic_block_size = False
    config.gpu.indexing_scheme = "blockwise4D"
    ker = create_kernel(asm, config=config)
    kfunc = sycl_jit.compile(ker)

    f_arr = dpt.zeros(shape, dtype=dtype, sycl_queue=queue)
    g_arr = dpt.zeros_like(f_arr, sycl_queue=queue)
    kfunc(f=f_arr, g=g_arr, queue=queue)

    queue.wait()


@pytest.mark.parametrize(
    "islice",
    [
        make_slice[1:-1, 1:-1],
        make_slice[3, 2:-2],
        make_slice[2:-2:2, ::3],
        make_slice[10:, :-5:2],
        make_slice[-5:-1, -1],
        make_slice[-3, -1],
    ],
)
def test_numerical_slices(sycl_jit, queue, islice):
    dtype= "float32"
    shape = (64, 64)

    f_arr = dpt.zeros(shape, dtype=dtype, device=queue)
    expected = np.zeros(shape, dtype=dtype)
    expected[islice] = 1.0

    f = Field.create_from_numpy_array("f", expected)

    update = Assignment(f.center(), 1)

    config = CreateKernelConfig(
        target=Target.SYCL,
        default_dtype=dtype,
        iteration_slice=islice,
    )

    kernel = create_kernel(update, config=config)

    kfunc = sycl_jit.compile(kernel)

    queue.wait()
    kfunc(f=f_arr, queue=queue)
    queue.wait()

    np.testing.assert_array_equal(dpt.asnumpy(f_arr), expected)


def test_triangle_pattern(queue, sycl_jit):
    shape = (16, 16)
    dtype = "float32"

    f_arr = dpt.zeros(shape, dtype=dtype, device=queue)

    expected = np.zeros(shape, dtype=dtype)
    f = Field.create_from_numpy_array("f", expected)
    for r in range(shape[0]):
        expected[r, r:] = 1.0

    update = Assignment(f.center(), 1)

    #   Have NumPy data layout -> X is slowest coordinate, Y is fastest
    slow_counter = DEFAULTS.spatial_counters[0]
    islice = make_slice[:, slow_counter:]

    config = CreateKernelConfig(
        target=Target.SYCL,
        default_dtype=dtype,
        iteration_slice=islice,
    )

    kernel = create_kernel(update, config=config)
    kfunc = sycl_jit.compile(kernel)

    queue.wait()
    kfunc(f=f_arr, queue=queue)
    queue.wait()

    np.testing.assert_array_equal(dpt.asnumpy(f_arr), expected)


def test_fixed_index_shape(queue, sycl_jit):
    dtype = "float32"
    f, g = fields(f"f(3), g(2, 2): {dtype}[2D]")

    asm = Assignment(f.center(1), g.center(0, 0) + g.center(0, 1) + g.center(1, 0) + g.center(1, 1))

    config = CreateKernelConfig(target=Target.SYCL, default_dtype=dtype,)
    ker = create_kernel(asm, config=config)
    kfunc = sycl_jit.compile(ker)

    f_arr = dpt.zeros((12, 14, 3), dtype=dtype, sycl_queue=queue)
    g_arr = dpt.ones((12, 14, 2, 2), dtype=dtype, sycl_queue=queue)
    kfunc(f=f_arr, g=g_arr, queue=queue)
    expected = np.zeros((12, 14, 3), dtype=dtype)
    expected[:, :, 1] = 4
    queue.wait()
    np.testing.assert_array_equal(dpt.asnumpy(f_arr), expected)

    with pytest.raises(ValueError):
        f_arr = dpt.zeros((12, 14, 2), dtype=dtype, sycl_queue=queue)
        g_arr = dpt.zeros((12, 14, 2, 2), dtype=dtype, sycl_queue=queue)
        kfunc(f=f_arr, g=g_arr, queue=queue)

    with pytest.raises(ValueError):
        f_arr = dpt.zeros((12, 14, 3), dtype=dtype, sycl_queue=queue)
        g_arr = dpt.zeros((12, 14, 4), dtype=dtype, sycl_queue=queue)
        kfunc(f=f_arr, g=g_arr, queue=queue)

    with pytest.raises(ValueError):
        f_arr = dpt.zeros((12, 14, 3), dtype=dtype, sycl_queue=queue)
        g_arr = dpt.zeros((12, 14, 1, 3), dtype=dtype, sycl_queue=queue)
        kfunc(f=f_arr, g=g_arr, queue=queue)


@pytest.mark.skip("Reductions not yet implemented")
def test_reductions(queue, sycl_jit):
    dtype = "float32"
    r = TypedSymbol("r", dtype)
    x = fields(f"x: {dtype}[3D]")

    assign_sum = AddReductionAssignment(r, x.center())

    config = CreateKernelConfig(target=Target.SYCL, default_dtype=dtype,)
    ker = create_kernel(assign_sum, config=config)

    kfunc = sycl_jit.compile(ker)
    x_array = dpt.ones((4, 4, 4), dtype=dtype, sycl_queue=queue)
    reduction_result = dpt.zeros((1,), dtype=dtype, sycl_queue=queue)

    kfunc(x=x_array, r_local=reduction_result, queue=queue)

    assert reduction_result[0] == (4 * 4 * 4)
