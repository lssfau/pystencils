import numpy as np
import sympy as sp
import pystencils as ps


def test_kernel_decorator_config():
    config = ps.CreateKernelConfig()
    a, b, c = ps.fields(a=np.ones(100), b=np.ones(100), c=np.ones(100))

    @ps.kernel_config(config)
    def test():
        a[0] @= b[0] + c[0]

    ps.create_kernel(**test)


def test_kernel_decorator2():
    h = sp.symbols("h")
    dtype = "float64"

    src, dst = ps.fields(f"src, src_tmp: {dtype}[3D]")

    @ps.kernel
    def kernel_func():
        dst[0, 0, 0] @= (src[1, 0, 0] + src[-1, 0, 0]
                         + src[0, 1, 0] + src[0, -1, 0]
                         + src[0, 0, 1] + src[0, 0, -1]) / (6 * h ** 2)

    # assignments = ps.assignment_from_stencil(stencil, src, dst, normalization_factor=2)
    ast = ps.create_kernel(kernel_func)

    _ = ps.get_code_str(ast)
