import pytest
import numpy as np
import sympy as sp
import pystencils as ps

from pystencils import (
    Field,
    Target,
    CreateKernelConfig,
    create_type,
)
from pystencils.flow import block, tie, cases, operator
from pystencils.grids import TensorField
from pystencils.codegen.properties import FieldBasePtr
from pystencils.types import PsPointerType
from pystencils.sympyextensions import convolve

from pystencils import Kernel
from pystencils.backend.emission import emit_code


def inspect_kernel(kernel: Kernel, gen_config: CreateKernelConfig):
    code = emit_code(kernel)

    px = (
        "pd"
        if gen_config.get_option("default_dtype") == create_type("float64")
        else "ps"
    )

    match gen_config.target:
        case Target.X86_SSE:
            assert f"_mm_loadu_{px}" in code
            assert f"_mm_storeu_{px}" in code
        case Target.X86_AVX:
            assert f"_mm256_loadu_{px}" in code
            assert f"_mm256_storeu_{px}" in code
        case Target.X86_AVX512:
            assert f"_mm512_loadu_{px}" in code
            assert f"_mm512_storeu_{px}" in code


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_filter_kernel(gen_config, xp, dtype):
    weight = sp.Symbol("weight")
    stencil = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    src = TensorField("src", 2, ghost_layers=1, dtype=dtype)
    dst = TensorField("dst", 2, ghost_layers=1, dtype=dtype)

    cfg = gen_config.copy()
    cfg.default_dtype = dtype

    if cfg.get_target().is_cpu():
        cfg.cpu.openmp.enable = True

    @operator(config=cfg)
    def update_rule(let):
        let.store[dst()] = weight * convolve(stencil, src)

    update_rule.compile_code()

    for param in update_rule.kernel.parameters:
        if pprop := param.get_properties(FieldBasePtr):
            field: TensorField = pprop.pop().field
            assert isinstance(param.dtype, PsPointerType)

            if field == src:
                assert param.dtype.base_type.const
            else:
                assert not param.dtype.base_type.const

    inspect_kernel(update_rule.kernel, cfg)

    shape = (42, 31)
    src_arr = xp.ones(shape, dtype=dtype)
    dst_arr = xp.zeros_like(src_arr)

    update_rule(src=src_arr, dst=dst_arr, weight=2.0)

    expected = np.zeros(shape)
    expected[1:-1, 1:-1].fill(18.0)

    xp.testing.assert_allclose(dst_arr, xp.array(expected))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_filter_kernel_fixedsize(gen_config, xp, dtype):
    weight = sp.Symbol("weight")
    stencil = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    shape = (42, 31)
    src_arr = xp.ones(shape, dtype=dtype)
    dst_arr = xp.zeros_like(src_arr)

    src = Field.create_fixed_size("src", shape)
    dst = Field.create_fixed_size("dst", shape)

    cfg = gen_config.copy()
    cfg.default_dtype = dtype

    if cfg.get_target().is_cpu():
        cfg.cpu.openmp.enable = True

    @operator(config=cfg)
    def update_rule(let):
        let.store[dst()] = weight * convolve(stencil, src)

    update_rule.compile_code()

    for param in update_rule.kernel.parameters:
        if pprop := param.get_properties(FieldBasePtr):
            field: Field = pprop.pop().field
            assert isinstance(param.dtype, PsPointerType)

            if field == src:
                assert param.dtype.base_type.const
            else:
                assert not param.dtype.base_type.const

    inspect_kernel(update_rule.kernel, cfg)

    update_rule(src=src_arr, dst=dst_arr, weight=2.0)

    expected = np.zeros(shape)
    expected[1:-1, 1:-1].fill(18.0)

    xp.testing.assert_allclose(dst_arr, xp.array(expected))


def test_diamond_graph_codegen(gen_config, xp):
    x, y, z, v, w = sp.symbols("x, y, z, v, w")
    f = TensorField("f", 2)
    g = TensorField("g", 2)

    @operator(config=gen_config)
    def ker():
        @block
        def block1(let):
            let[x] = g() + v
            let.export[y] = x + 1

        @block(preds=[block1])
        def block2(let):
            let[x] = y + 1
            let.export[z] = x + 1

        @block(preds=[block1])
        def block3(let):
            let[x] = y + 2
            let.export[w] = x + 1

        @block(preds=[block2, block3])
        def block4(let):
            let[x] = z + w
            let.store[f()] = x

        return block4

    rng = np.random.default_rng(seed=0x5005)

    g_arr = xp.array(rng.random((18, 23)))
    f_arr = xp.zeros_like(g_arr)
    v_val = 31

    f_arr_expected = 2 * (g_arr + v_val) + 7

    ker(f=f_arr, g=g_arr, v=v_val)

    xp.testing.assert_allclose(f_arr, f_arr_expected)


@pytest.mark.parametrize(
    "target", [t for t in Target.available_targets() if not t.is_vector_cpu()]
)
def test_parallel_graphs_codegen(gen_config, xp):
    x, y, z, v, w = sp.symbols("x, y, z, v, w")
    f = TensorField("f", 2, (2,))
    g = TensorField("g", 2, (2,))

    @operator(config=gen_config)
    def ker():
        @block
        def block1a(let):
            let[x] = g(0) + v
            let.export[y] = x + 1

        @block(preds=[block1a])
        def block1b(let):
            let[z] = y + 1
            let.store[f(0)] = z + 1

        @block
        def block2a(let):
            let[x] = g(1) + 2 * v
            let.export[y] = x + 2

        @block(preds=[block2a])
        def block2b(let):
            let[z] = y + 2
            let.store[f(1)] = z + 2

        return block1b, block2b

    rng = np.random.default_rng(seed=0x5005)

    g_arr = xp.array(rng.random((18, 23, 2)))
    f_arr = xp.zeros_like(g_arr)
    v_val = 31

    f_arr_expected = xp.zeros_like(f_arr)
    f_arr_expected[:, :, 0] = g_arr[:, :, 0] + v_val + 3
    f_arr_expected[:, :, 1] = g_arr[:, :, 1] + 2 * v_val + 6

    ker(f=f_arr, g=g_arr, v=v_val)

    xp.testing.assert_allclose(f_arr, f_arr_expected)


@pytest.mark.parametrize(
    "target",
    [t for t in Target.available_targets() if t not in [Target.HIP, Target.SYCL]],
)
def test_reductions(gen_config, xp):
    f = TensorField("f", 2)
    g = TensorField("g", 2)

    q = ps.TypedSymbol("q", ps.DynamicType.NUMERIC_TYPE)
    r = ps.TypedSymbol("r", ps.DynamicType.NUMERIC_TYPE)

    @operator(config=gen_config)
    def ker():
        @block
        def block1(_eq):
            _eq.reduce[q, "max"] = f()

        @block
        def block2(_eq):
            _eq.reduce[r, "+"] = g()

        return block1, block2

    rng = np.random.default_rng(seed=0x5005)

    f_arr = xp.array(rng.random((18, 23)))
    g_arr = xp.array(rng.random((18, 23)))

    q_slot = xp.zeros((1,))
    r_slot = xp.zeros((1,))

    ker(f=f_arr, g=g_arr, q=q_slot, r=r_slot)

    q_desired = xp.max(f_arr)
    r_desired = xp.sum(g_arr)

    xp.testing.assert_allclose(q_slot[0], q_desired)
    xp.testing.assert_allclose(r_slot[0], r_desired)


@pytest.mark.parametrize(
    "target", [t for t in Target.available_targets() if not t.is_vector_cpu()]
)
def test_simple_cases_graph_codegen(gen_config, xp):
    x, y = sp.symbols("x, y")
    u, v = sp.symbols("u, v")
    f = TensorField("f", 2, (2,))
    g = TensorField("g", 2, (2,))
    h = TensorField("h", 2, (2,))

    @operator(config=gen_config)
    def ker():
        @cases
        def cs_block1(cs):
            @cs.case(u > 0)
            def _(let):
                let.export[x] = f(0) + u

            @cs.case(True)
            def _(let):
                let.export[x] = f(1) - 2 * u

        @cases
        def cs_block2(cs):
            @cs.case(v > 0)
            def _(let):
                let.export[y] = g(0) + v

            @cs.case(True)
            def _(let):
                let.export[y] = g(1) - v - 1

        @block(preds=[cs_block1, cs_block2])
        def block3(let):
            let.store[h(0)] = x
            let.store[h(1)] = y

        return block3

    rng = np.random.default_rng(seed=0x5005)

    f_arr = xp.array(rng.random((18, 23, 2)))
    g_arr = xp.array(rng.random((18, 23, 2)))

    # u <= 0 (default), v > 0
    h_arr = xp.zeros_like(g_arr)
    u_val = -2
    v_val = 31

    h_arr_expected = xp.zeros_like(h_arr)
    h_arr_expected[:, :, 0] = f_arr[:, :, 1] - 2 * u_val
    h_arr_expected[:, :, 1] = g_arr[:, :, 0] + v_val

    ker(f=f_arr, g=g_arr, h=h_arr, u=u_val, v=v_val)

    xp.testing.assert_allclose(h_arr, h_arr_expected)

    # u > 0, v <= 0 (default)
    h_arr = xp.zeros_like(g_arr)
    u_val = 22
    v_val = 0

    h_arr_expected = xp.zeros_like(h_arr)
    h_arr_expected[:, :, 0] = f_arr[:, :, 0] + u_val
    h_arr_expected[:, :, 1] = g_arr[:, :, 1] - v_val - 1

    ker(f=f_arr, g=g_arr, h=h_arr, u=u_val, v=v_val)

    xp.testing.assert_allclose(h_arr, h_arr_expected)


@pytest.mark.parametrize(
    "target", [t for t in Target.available_targets() if not t.is_vector_cpu()]
)
def test_nested_cases_graph_codegen(gen_config, xp):
    x, y, z, w = sp.symbols("x, y, z, w")
    u, v = sp.symbols("u, v")
    f = TensorField("f", 2, (2,))
    g = TensorField("g", 2, (2,))
    h = TensorField("h", 2, (2,))
    k = TensorField("k", 2, (2,))

    @operator(config=gen_config)
    def ker():

        @block
        def block1_pred(let):
            let[x] = g(0) + v
            let.export[y] = x + 1

        @block(preds=[block1_pred])
        def block1(let):
            let[x] = y + 1
            let.export[z] = x

        @block
        def block2_pred(let):
            let[x] = g(1) + 2 * v
            let.export[y] = x + 2

        @block(preds=[block2_pred])
        def block2(let):
            let[x] = y + 2
            let.store[k(0)] = x - v - 3
            let.store[k(1)] = x + 2

        @cases
        def cs_block_nested(cs):
            @cs.case(u > 0)
            def _(let):
                let.export[w] = f(0) + u

            @cs.case(sp.true)
            def _(let):
                let.export[w] = f(1) - 2 * u

        @cases
        def cs_block(cs):
            cs.case(v > 0, tie(cs_block_nested))

            @cs.case(sp.true)
            def _(let):
                let.export[w] = g(1) - v - 1

        @block(preds=[block1, cs_block])
        def block3(let):
            let.store[h(0)] = w
            let.store[h(1)] = z

        return block2, block3

    rng = np.random.default_rng(seed=0x5005)

    f_arr = xp.array(rng.random((18, 23, 2)))
    g_arr = xp.array(rng.random((18, 23, 2)))
    k_arr = xp.zeros_like(g_arr)

    k_arr_expected = xp.zeros_like(k_arr)
    h_arr_expected = xp.zeros_like(g_arr)

    # u > 0, v > 0
    h_arr = xp.zeros_like(g_arr)
    u_val = 2
    v_val = 31

    k_arr_expected[:, :, 0] = g_arr[:, :, 1] + v_val + 3
    k_arr_expected[:, :, 1] = g_arr[:, :, 1] + 2 * v_val + 6

    h_arr_expected[:, :, 0] = f_arr[:, :, 0] + u_val
    h_arr_expected[:, :, 1] = g_arr[:, :, 0] + v_val + 2

    ker(f=f_arr, g=g_arr, h=h_arr, k=k_arr, u=u_val, v=v_val)

    xp.testing.assert_allclose(h_arr, h_arr_expected)

    # u <= 0 (default), v > 0
    h_arr = xp.zeros_like(g_arr)
    u_val = 0
    v_val = 31

    k_arr_expected[:, :, 0] = g_arr[:, :, 1] + v_val + 3
    k_arr_expected[:, :, 1] = g_arr[:, :, 1] + 2 * v_val + 6

    h_arr_expected[:, :, 0] = f_arr[:, :, 1] - 2 * u_val
    h_arr_expected[:, :, 1] = g_arr[:, :, 0] + v_val + 2

    ker(f=f_arr, g=g_arr, h=h_arr, k=k_arr, u=u_val, v=v_val)

    xp.testing.assert_allclose(h_arr, h_arr_expected)

    # v <= 0 (default)
    h_arr = xp.zeros_like(g_arr)
    u_val = 0
    v_val = 0

    k_arr_expected[:, :, 0] = g_arr[:, :, 1] + v_val + 3
    k_arr_expected[:, :, 1] = g_arr[:, :, 1] + 2 * v_val + 6

    h_arr_expected[:, :, 0] = g_arr[:, :, 1] - v_val - 1
    h_arr_expected[:, :, 1] = g_arr[:, :, 0] + v_val + 2

    ker(f=f_arr, g=g_arr, h=h_arr, k=k_arr, u=u_val, v=v_val)

    xp.testing.assert_allclose(h_arr, h_arr_expected)


@pytest.mark.parametrize("target", [Target.GenericCPU])
def test_conditional_reductions_and_writes(gen_config, xp):
    f = TensorField("f", 2)
    g = TensorField("g", 2)
    h = TensorField("h", 2)

    q = ps.TypedSymbol("q", ps.DynamicType.NUMERIC_TYPE)
    r = ps.TypedSymbol("r", ps.DynamicType.NUMERIC_TYPE)

    @operator(config=gen_config)
    def ker():
        @cases
        def block1(_cs):
            @_cs.case(f() <= 0)
            def case1(_eq):
                _eq.reduce[q, "+"] = g()
                _eq.store[h()] = -g()

            @_cs.case(f() >= 1)
            def case2(_eq):
                _eq.reduce[r, "+"] = g()
                _eq.store[h()] = 2 * g()

        return block1

    rng = np.random.default_rng(seed=0x5005)

    f_arr = xp.zeros((10, 20))
    f_arr[:, :10] = xp.array(-2 + 2 * rng.random((10, 10)))
    f_arr[:, 10:] = xp.array(1 + 2 * rng.random((10, 10)))
    g_arr = xp.array(rng.random((10, 20)))
    h_arr = xp.zeros_like(g_arr)

    q_slot = xp.zeros((1,))
    r_slot = xp.zeros((1,))

    ker(f=f_arr, g=g_arr, h=h_arr, q=q_slot, r=r_slot)

    q_desired = xp.sum(g_arr[:, :10])
    r_desired = xp.sum(g_arr[:, 10:])

    h_desired = xp.zeros_like(g_arr)
    h_desired[:, :10] = -g_arr[:, :10]
    h_desired[:, 10:] = 2 * g_arr[:, 10:]

    xp.testing.assert_allclose(q_slot[0], q_desired)
    xp.testing.assert_allclose(r_slot[0], r_desired)
    xp.testing.assert_allclose(h_arr, h_desired)


def test_subgraph(gen_config, xp):
    x, y, z, v = sp.symbols("x, y, z, v")
    f = TensorField("f", 2)
    g = TensorField("g", 2)
    h = TensorField("h", 2)

    @operator(config=gen_config)
    def ker():

        @ps.flow.block
        def block1(_eq):
            _eq.export[z] = h()

        @ps.flow.block
        def block2(_eq):
            _eq.let[x] = g() * v
            _eq.export[y] = x + z

        subgr = ps.flow.subgraph(block2, preds=[block1])

        @ps.flow.block(preds=[subgr])
        def block3(_eq):
            _eq.store[f()] = y

        return block3

    rng = np.random.default_rng(seed=0x5005)

    g_arr = xp.array(rng.random((18, 23)))
    h_arr = xp.array(rng.random((18, 23)))
    f_arr = xp.zeros_like(g_arr)
    v_val = 2.3

    f_arr_expected = xp.zeros_like(f_arr)
    f_arr_expected[:, :] = g_arr * v_val + h_arr

    ker(f=f_arr, g=g_arr, h=h_arr, v=v_val)

    xp.testing.assert_allclose(f_arr, f_arr_expected)
