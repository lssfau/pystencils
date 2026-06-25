import pytest

import sympy as sp
import pystencils as ps

from pystencils.flow import Operator, operator


def test_operator_api():
    x, y = sp.symbols("x, y")
    f = ps.grids.TensorField("f", 2, ())

    @ps.flow.block
    def block1(_b):
        _b.store[f()] = x + y

    op = Operator(block1)
    assert op.graph == ps.flow.tie(block1)
    assert op.kernel is None
    assert op.func is None

    op.generate_code()

    with pytest.raises(AttributeError):
        _ = op.config

    with pytest.raises(AttributeError):
        op.config = ps.CreateKernelConfig(target=ps.Target.HIP)

    assert op.func is None

    assert isinstance(op.kernel, ps.Kernel)

    with pytest.raises(RuntimeError):
        op.generate_code()

    op.clear()
    op.config.default_dtype = "float32"
    op.compile_code()

    with pytest.raises(RuntimeError):
        op.compile_code()

    assert isinstance(op.func, ps.jit.KernelWrapper)

    op.clear_compiled_code()

    assert op.func is None

    with pytest.raises(AttributeError):
        _ = op.config


def test_operator_args_from_patch_data(gen_config, xp):
    if gen_config.get_target() == ps.Target.SYCL:
        pytest.xfail("CreateNdArray does not support SYCL yet (issue #138)")

    patch = ps.grids.Patch("P", (1, 1), num_cells=(8, 8))
    f = ps.grids.TensorField("f", patch.cells, ())
    g = ps.grids.TensorField("g", patch.cells, ())
    c = sp.Symbol("c")

    pdata = ps.grids.PatchData(
        patch, {c: 3.2}, fields=[f, g], target=gen_config.get_target()
    )

    @operator(config=gen_config)
    def scale(_eq):
        _eq.store[g()] = c * f()

    pdata[f][:] = 1.0
    scale(pdata)
    xp.testing.assert_array_equal(pdata[g], 3.2)

    pdata[f][:] = 1.0
    scale(pdata, c=-1.2)
    xp.testing.assert_array_equal(pdata[g], -1.2)
