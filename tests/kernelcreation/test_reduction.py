import pytest
import numpy as np

import pystencils as ps
from pystencils import Target
from pystencils.sympyextensions import ReductionOp, reduction_assignment

INIT_W = 5
INIT_ARR = 2
SIZE = 15
SOLUTION = {
    ReductionOp.Add: INIT_W + INIT_ARR * SIZE,
    ReductionOp.Sub: INIT_W - INIT_ARR * SIZE,
    ReductionOp.Mul: INIT_W * INIT_ARR**SIZE,
    ReductionOp.Min: min(INIT_W, INIT_ARR),
    ReductionOp.Max: max(INIT_W, INIT_ARR),
}


# get AST for kernel with reduction assignment
def get_reduction_assign_ast(dtype, op, config):
    x = ps.fields(f"x: {dtype}[1d]")
    w = ps.TypedSymbol("w", dtype)

    red_assign = reduction_assignment(w, op, x.center())

    return ps.create_kernel([red_assign], config, default_dtype=dtype)


def get_cpu_array(op, dtype):
    # increase difficulty of min/max tests by using range of values
    match op:
        case ReductionOp.Min:
            return np.linspace(INIT_ARR, INIT_ARR + SIZE, SIZE, dtype=dtype)
        case ReductionOp.Max:
            return np.linspace(INIT_ARR - SIZE, INIT_ARR, SIZE, dtype=dtype)
        case _:
            return np.full((SIZE,), INIT_ARR, dtype=dtype)


@pytest.mark.parametrize(
    "target", [Target.GenericCPU] + Target.available_vector_cpu_targets()
)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize(
    "op",
    [
        ReductionOp.Add,
        ReductionOp.Sub,
        ReductionOp.Mul,
        ReductionOp.Min,
        ReductionOp.Max,
    ],
)
def test_reduction_cpu(target, dtype, op):
    config = ps.CreateKernelConfig(target=target)
    config.cpu.openmp.enable = True

    if target.is_vector_cpu():
        config.cpu.vectorize.enable = True
        config.cpu.vectorize.assume_inner_stride_one = True

    ast_reduction = get_reduction_assign_ast(dtype, op, config)
    kernel_reduction = ast_reduction.compile()

    array = get_cpu_array(op, dtype)
    reduction_array = np.full((1,), INIT_W, dtype=dtype)

    kernel_reduction(x=array, w=reduction_array)
    assert np.allclose(reduction_array, SOLUTION[op])


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize(
    "op",
    [
        ReductionOp.Add,
        ReductionOp.Sub,
        ReductionOp.Mul,
        ReductionOp.Min,
        ReductionOp.Max,
    ],
)
@pytest.mark.parametrize("assume_warp_aligned_block_size", [True, False])
@pytest.mark.parametrize("use_block_fitting", [True, False])
@pytest.mark.parametrize("warp_size", [32, None])
@pytest.mark.skipif(
    Target.CUDA not in Target.available_targets(), reason="CUDA is not available"
)
def test_reduction_gpu(
    dtype: str,
    op: str,
    assume_warp_aligned_block_size: bool,
    use_block_fitting: bool,
    warp_size: int | None,
):
    import cupy as cp

    cfg = ps.CreateKernelConfig(target=ps.Target.GPU)
    cfg.gpu.assume_warp_aligned_block_size = assume_warp_aligned_block_size
    if warp_size:
        cfg.gpu.warp_size = warp_size

    ast_reduction = get_reduction_assign_ast(dtype, op, cfg)
    kernel_reduction = ast_reduction.compile()

    if use_block_fitting and warp_size:
        kernel_reduction.launch_config.fit_block_size((warp_size, 1, 1))

    array = get_cpu_array(op, dtype)
    reduction_array = np.full((1,), INIT_W, dtype=dtype)

    array_gpu = cp.asarray(array)
    reduction_array_gpu = cp.asarray(reduction_array)

    kernel_reduction(x=array_gpu, w=reduction_array_gpu)
    assert np.allclose(reduction_array_gpu.get(), SOLUTION[op])
