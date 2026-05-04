from typing import cast
import pytest

import pystencils as ps

if not any(t.is_gpu() for t in ps.Target.available_targets()):
    pytest.skip("No GPU available", allow_module_level=True)

import cupy as cp

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    FullIterationSpace,
)
from pystencils.backend.platforms import CudaPlatform, HipPlatform
from pystencils.backend.ast.structural import PsBlock
from pystencils.backend.transformations import (
    AxisExpansion,
    MaterializeAxes,
    ReductionsToMemory,
    SelectFunctions,
    LowerToC,
)
from pystencils.backend.transformations.axis_expansion import AxisExpansionStrategy
from pystencils.backend.ast import dfs_preorder
from pystencils.backend.ast.structural import PsConditional

from pystencils.codegen.driver import KernelFactory
from pystencils.codegen.gpu_indexing import ManualLaunchConfiguration, GpuIndexing
from pystencils.jit.gpu_cupy import CupyJit, CupyKernelWrapper


def _make_kernel(
    ctx: KernelCreationContext,
    target: ps.Target,
    body: PsBlock,
    ae_strategy: AxisExpansionStrategy,
):
    factory = AstFactory(ctx)
    cube = factory.cube_from_ispace(ctx.get_full_iteration_space(), body)

    match target:
        case ps.Target.CUDA:
            platform = CudaPlatform(ctx)
        case ps.Target.HIP:
            platform = HipPlatform(ctx)
        case _:
            assert False, "unexpected target"

    loops = ae_strategy(cube)

    materialize_axes = MaterializeAxes(ctx)
    ast = materialize_axes(loops)

    r_to_mem = ReductionsToMemory(ctx, ctx.reduction_data.values())
    ast = r_to_mem(ast)

    lower = LowerToC(ctx)
    ast = cast(PsBlock, lower(ast))

    select_functions = SelectFunctions(platform)
    ast = cast(PsBlock, select_functions(ast))

    def launch_config_factory() -> ManualLaunchConfiguration:
        return ManualLaunchConfiguration(GpuIndexing.get_hardware_properties(target))

    kfactory = KernelFactory(ctx)
    kernel = kfactory.create_gpu_kernel(
        platform, ast, "kernel", target, CupyJit(), launch_config_factory
    )

    return kernel


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_threads_only(rank: int):
    f, g = ps.fields(f"f, g: [{rank}D]")

    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, 0, f)
    ctx.set_iteration_space(ispace)

    body = PsBlock([factory.parse_sympy(ps.Assignment(g(), 2 * f()))])

    ae = AxisExpansion(ctx)
    strategy = ae.create_strategy(
        [ae.gpu_thread(dim) for dim in ["x", "y", "z"][:rank]][::-1]
    )

    target = ps.Target.auto_gpu()
    kernel = _make_kernel(ctx, target, body, strategy)
    kfunc: CupyKernelWrapper = kernel.compile()

    shape = (29, 7, 3)[:rank][::-1]
    rng = cp.random.default_rng(seed=514)
    f_arr = rng.random(shape)
    g_arr = cp.zeros_like(f_arr)
    expected = 2 * f_arr

    kfunc.launch_config.block_size = (32, 8, 4)[:rank] + (1,) * (3 - rank)
    kfunc.launch_config.grid_size = (1, 1, 1)

    kfunc(f=f_arr, g=g_arr)
    cp.testing.assert_allclose(g_arr, expected)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_blocks_and_threads(rank: int):
    f, g = ps.fields(f"f, g: [{rank}D]")

    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, 0, f)
    ctx.set_iteration_space(ispace)

    body = PsBlock([factory.parse_sympy(ps.Assignment(g(), 2 * f()))])

    ae = AxisExpansion(ctx)
    steps = [ae.gpu_block_x_thread("x"), ae.gpu_thread("y"), ae.gpu_block("y")]
    strategy = ae.create_strategy(steps[:rank][::-1])

    target = ps.Target.auto_gpu()
    kernel = _make_kernel(ctx, target, body, strategy)
    kfunc: CupyKernelWrapper = kernel.compile()

    shape = (49, 15, 9)[:rank][::-1]
    rng = cp.random.default_rng(seed=514)
    f_arr = rng.random(shape)
    g_arr = cp.zeros_like(f_arr)
    expected = 2 * f_arr

    kfunc.launch_config.block_size = (64, 16, 1)[:rank] + (1,) * (3 - rank)
    kfunc.launch_config.grid_size = (1, 10, 1) if rank == 3 else (1, 1, 1)

    kfunc(f=f_arr, g=g_arr)
    cp.testing.assert_allclose(g_arr, expected)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_blocks_x_threads(rank: int):
    f, g = ps.fields(f"f, g: [{rank}D]")

    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, 0, f)
    ctx.set_iteration_space(ispace)

    body = PsBlock([factory.parse_sympy(ps.Assignment(g(), 2 * f()))])

    ae = AxisExpansion(ctx)
    strategy = ae.create_strategy(
        [ae.gpu_block_x_thread(dim) for dim in ["x", "y", "z"][:rank]][::-1]
    )

    target = ps.Target.auto_gpu()
    kernel = _make_kernel(ctx, target, body, strategy)
    kfunc: CupyKernelWrapper = kernel.compile()

    shape = (125, 61, 7)[:rank][::-1]
    rng = cp.random.default_rng(seed=514)
    f_arr = rng.random(shape)
    g_arr = cp.zeros_like(f_arr)
    expected = 2 * f_arr

    kfunc.launch_config.block_size = (16, 2, 2)[:rank] + (1,) * (3 - rank)
    kfunc.launch_config.grid_size = (8, 32, 4)

    kfunc(f=f_arr, g=g_arr)
    cp.testing.assert_allclose(g_arr, expected)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_with_reductions(rank: int):
    f = ps.fields(f"f: [{rank}D]")
    w = ps.TypedSymbol("w", ps.DynamicType.NUMERIC_TYPE)

    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, 0, f)
    ctx.set_iteration_space(ispace)

    body = PsBlock([factory.parse_sympy(ps.MaxReductionAssignment(w, f()))])

    ae = AxisExpansion(ctx)
    strategy = ae.create_strategy(
        [ae.gpu_block_x_thread(dim) for dim in ["x", "y", "z"][:rank]][::-1]
    )

    target = ps.Target.auto_gpu()
    kernel = _make_kernel(ctx, target, body, strategy)
    kfunc: CupyKernelWrapper = kernel.compile()

    shape = (29, 3, 2)[:rank][::-1]
    rng = cp.random.default_rng(seed=514)
    f_arr = rng.random(shape)
    w_arr = cp.zeros((1,))
    expected = cp.max(f_arr)

    kfunc.launch_config.block_size = (16, 2, 2)[:rank] + (1,) * (3 - rank)
    kfunc.launch_config.grid_size = (2, 2, 1)[:rank] + (1,) * (3 - rank)

    kfunc(f=f_arr, w=w_arr)
    cp.testing.assert_allclose(w_arr, expected)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_combined_thread_guards(rank: int):
    f, g = ps.fields(f"f, g: [{rank}D]")

    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, 0, f)
    ctx.set_iteration_space(ispace)

    body = PsBlock([factory.parse_sympy(ps.Assignment(g(), 2 * f()))])

    ae = AxisExpansion(ctx)
    steps = [ae.gpu_block("y"), ae.gpu_thread("y"), ae.gpu_block_x_thread("x")]
    strategy = ae.create_strategy(steps[:rank])

    target = ps.Target.auto_gpu()
    kernel = _make_kernel(ctx, target, body, strategy)

    conditionals: list[PsConditional] = list(
        filter(lambda n: isinstance(n, PsConditional), dfs_preorder(kernel.body))
    )
    assert len(conditionals) == 1
