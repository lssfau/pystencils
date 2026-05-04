import numpy as np
import sympy as sp

from pystencils import Assignment, CreateKernelConfig, Target, create_kernel, fields
from pystencils.jit import NoJit

try:
    import dpctl
    import dpctl.tensor as dpt
    HAVE_DPCTL = True
except ImportError:
    HAVE_DPCTL = False


def test_sycl_kernel_static(sycl_jit):
    dtype = "float32"
    src, dst = fields(f"src, dst: {dtype}[2D]")
    asm = Assignment(dst.center(), sp.sin(src.center()) + sp.cos(src.center()))

    cfg = CreateKernelConfig(target=Target.SYCL, jit=NoJit)
    kernel = create_kernel(asm, cfg)
    code_string = kernel.get_c_code()

    assert "sycl::id< 2 >" in code_string
    assert "sycl::sin(" in code_string
    assert "sycl::cos(" in code_string

    if HAVE_DPCTL:
        kfunc = sycl_jit.compile(kernel)
        queue = dpctl.SyclQueue()

        rng = np.random.default_rng(0x5eed)
        src_np_arr = rng.random(size=(34, 26), dtype=dtype) * np.pi

        src_arr = dpt.from_numpy(src_np_arr, sycl_queue=queue)
        dst_arr = dpt.zeros_like(src_arr, sycl_queue=queue)
        kfunc(src=src_arr, dst=dst_arr, queue=queue)
        queue.wait()

        def f(x):
            return np.sin(x) + np.cos(x)
        expected = f(src_np_arr)

        np.testing.assert_almost_equal(dpt.asnumpy(dst_arr), expected, decimal=5)


def test_sycl_kernel_manual_block_size(sycl_jit):

    dtype = "float32"
    src, dst = fields(f"src, dst: {dtype}[2D]")
    asm = Assignment(dst.center(), sp.sin(src.center()) + sp.cos(src.center()))

    cfg = CreateKernelConfig(target=Target.SYCL, jit=NoJit)
    cfg.sycl.automatic_block_size = False
    cfg.gpu.manual_launch_grid = True
    kernel = create_kernel(asm, cfg)

    code_string = kernel.get_c_code()

    assert "sycl::nd_item< 2 >" in code_string

    if HAVE_DPCTL:
        kfunc = sycl_jit.compile(kernel)
        queue = dpctl.SyclQueue()

        rng = np.random.default_rng(0x5eed)
        src_np_arr = rng.random(size=(34, 26), dtype=dtype) * np.pi

        src_arr = dpt.from_numpy(src_np_arr, sycl_queue=queue)
        dst_arr = dpt.zeros_like(src_arr, sycl_queue=queue)
        kfunc.launch_config.block_size = (1, 1)
        kfunc.launch_config.grid_size = src_np_arr.shape
        kfunc(src=src_arr, dst=dst_arr, queue=queue)
        queue.wait()

        def f(x):
            return np.sin(x) + np.cos(x)
        expected = f(src_np_arr)

        np.testing.assert_almost_equal(dpt.asnumpy(dst_arr), expected, decimal=5)
