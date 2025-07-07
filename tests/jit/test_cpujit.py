import pytest


from itertools import product
import sympy as sp
import numpy as np
from pystencils import create_kernel, Assignment, fields, Field, FieldType, make_slice
from pystencils.jit import CpuJit


@pytest.fixture
def cpu_jit(tmp_path) -> CpuJit:
    return CpuJit(objcache=tmp_path, emit_warnings=True)


def test_basic_cpu_kernel(cpu_jit):
    f, g = fields("f, g: [2D]")
    asm = Assignment(g.center(), 2.0 * f.center())
    ker = create_kernel(asm)
    kfunc = cpu_jit.compile(ker)

    rng = np.random.default_rng(0x5eed)
    f_arr = rng.random(size=(34, 26), dtype="float64")
    g_arr = np.zeros_like(f_arr)

    kfunc(f=f_arr, g=g_arr)

    np.testing.assert_almost_equal(g_arr, 2.0 * f_arr)


def test_invalid_args(cpu_jit):
    f, g = fields("f, g: [2D]")
    asm = Assignment(f.center(), 2.0 * g.center())
    ker = create_kernel(asm)
    kfunc = cpu_jit.compile(ker)

    f_arr = np.zeros((34, 26), dtype="float64")
    g_arr = np.zeros_like(f_arr)

    #   Missing Arguments
    with pytest.raises(KeyError):
        kfunc(f=f_arr)

    with pytest.raises(KeyError):
        kfunc(g=g_arr)

    #   Extra arguments are ignored
    kfunc(f=f_arr, g=g_arr, x=2.1)


def test_argument_type_error(cpu_jit):
    f, g = fields("f, g: [2D]")
    c = sp.Symbol("c")
    asm = Assignment(f.center(), c * g.center())
    ker = create_kernel(asm)
    kfunc = cpu_jit.compile(ker)

    arr_fp16 = np.zeros((23, 12), dtype="float16")
    arr_fp32 = np.zeros((23, 12), dtype="float32")
    arr_fp64 = np.zeros((23, 12), dtype="float64")

    with pytest.raises(TypeError):
        kfunc(f=arr_fp32, g=arr_fp64, c=2.0)

    with pytest.raises(TypeError):
        kfunc(f=arr_fp64, g=arr_fp32, c=2.0)

    with pytest.raises(TypeError):
        kfunc(f=arr_fp16, g=arr_fp16, c=2.0)

    with pytest.raises(TypeError):
        kfunc(f=arr_fp64, g=arr_fp64, c=[2.0])

    #   Wrong scalar types are OK, though
    kfunc(f=arr_fp64, g=arr_fp64, c=np.float16(1.0))


def test_shape_check(cpu_jit):
    f, g = fields("f, g: [2D]")
    asm = Assignment(g.center(), 2.0 * f.center())
    ker = create_kernel(asm)
    kfunc = cpu_jit.compile(ker)

    f_arr = np.zeros((41, 32))
    g_arr = np.zeros_like(f_arr)
    kfunc(f=f_arr, g=g_arr)

    #   Trivial scalar dimensions are OK
    f_arr = np.zeros((41, 32, 1))
    g_arr = np.zeros_like(f_arr)
    kfunc(f=f_arr, g=g_arr)

    with pytest.raises(ValueError):
        f_arr = np.zeros((41, 32))
        g_arr = np.zeros((40, 32))
        kfunc(f=f_arr, g=g_arr)

    with pytest.raises(ValueError):
        f_arr = np.zeros((41, 1))
        g_arr = np.zeros((40, 32))
        kfunc(f=f_arr, g=g_arr)

    with pytest.raises(ValueError):
        f_arr = np.zeros((41, 1))
        g_arr = np.zeros((40,))
        kfunc(f=f_arr, g=g_arr)


def test_fixed_shape(cpu_jit):
    a = np.zeros((12, 23), dtype="float64")
    b = np.zeros((13, 21), dtype="float64")
    
    f = Field.create_from_numpy_array("f", a)
    g = Field.create_from_numpy_array("g", a)

    asm = Assignment(f.center(), 2.0 * g.center())
    ker = create_kernel(asm)
    kfunc = cpu_jit.compile(ker)

    kfunc(f=a, g=a)

    with pytest.raises(ValueError):
        kfunc(f=b, g=a)

    with pytest.raises(ValueError):
        kfunc(f=a, g=b)


def test_fixed_index_shape(cpu_jit):
    f, g = fields("f(3), g(2, 2): [2D]")

    asm = Assignment(f.center(1), g.center(0, 0) + g.center(0, 1) + g.center(1, 0) + g.center(1, 1))
    ker = create_kernel(asm)
    kfunc = cpu_jit.compile(ker)

    f_arr = np.zeros((12, 14, 3))
    g_arr = np.zeros((12, 14, 2, 2))
    kfunc(f=f_arr, g=g_arr)

    with pytest.raises(ValueError):
        f_arr = np.zeros((12, 14, 2))
        g_arr = np.zeros((12, 14, 2, 2))
        kfunc(f=f_arr, g=g_arr)

    with pytest.raises(ValueError):
        f_arr = np.zeros((12, 14, 3))
        g_arr = np.zeros((12, 14, 4))
        kfunc(f=f_arr, g=g_arr)

    with pytest.raises(ValueError):
        f_arr = np.zeros((12, 14, 3))
        g_arr = np.zeros((12, 14, 1, 3))
        kfunc(f=f_arr, g=g_arr)


def test_scalar_field(cpu_jit):
    f, g = fields("f(1), g: [2D]")
    asm = Assignment(g(), f(0))
    ker = create_kernel(asm)
    kfunc = cpu_jit.compile(ker)

    spatial_shape = (31, 29)
    #   Both implicit and explicit scalar fields must be accepted
    for ishape_f, ishape_g in product(((), (1,)), ((), (1,))):
        rng = np.random.default_rng(0x5eed)
        f_arr = rng.random(size=spatial_shape + ishape_f, dtype="float64")
        g_arr = np.zeros(spatial_shape + ishape_g)
        
        kfunc(f=f_arr, g=g_arr)

        np.testing.assert_allclose(f_arr.flatten(), g_arr.flatten())


def test_only_custom_fields(cpu_jit):
    f = fields("f: [1D]", field_type=FieldType.CUSTOM)
    asm = Assignment(f(), 2 * f())
    ker = create_kernel(asm, iteration_slice=make_slice[0:12])
    kfunc = cpu_jit.compile(ker)

    f_arr = np.ones(12)
    kfunc(f=f_arr)
    np.testing.assert_allclose(f_arr, np.array([2.0] * 12))
