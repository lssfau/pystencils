import pytest
import numpy as np

import pystencils as ps
from pystencils import Target
from pystencils.sympyextensions import ReductionOp, reduction_assignment

init_w = 5.0
arr_size = 16


def init_arr(op):
    match op:
        case ReductionOp.Mul:
            return 0.99  # avoid value overflow for large array sizes
        case _:
            return 2.0


def get_expected_solution(op, array):
    match op:
        case ReductionOp.Add:
            return init_w + np.sum(array)
        case ReductionOp.Sub:
            return init_w - np.sum(array)
        case ReductionOp.Mul:
            return init_w * np.prod(array)
        case ReductionOp.Min:
            return min(init_w, np.min(array))
        case ReductionOp.Max:
            return max(init_w, np.max(array))


# get AST for kernel with reduction assignment
def get_reduction_assign_ast(dtype, op, dims, config):
    x = ps.fields(f"x: {dtype}[{dims}d]")
    w = ps.TypedSymbol("w", dtype)

    red_assign = reduction_assignment(w, op, x.center())

    return ps.create_kernel([red_assign], config, default_dtype=dtype)


def get_cpu_array(dtype, op, dims):
    shape = (arr_size,) * dims

    # increase difficulty of min/max tests by using range of values
    match op:
        case ReductionOp.Min | ReductionOp.Max:
            lo = init_arr(op) - arr_size
            mi = init_arr(op)
            hi = init_arr(op) + arr_size

            if op is ReductionOp.Min:
                return np.random.randint(mi, hi, size=shape).astype(dtype)
            else:
                return np.random.randint(lo, mi, size=shape).astype(dtype)
        case _:
            return np.full(shape, init_arr(op), dtype=dtype)


@pytest.mark.parametrize(
    "target", (Target.GenericCPU,) + Target.available_vector_cpu_targets()
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
@pytest.mark.parametrize("dims", [1, 2, 3])
def test_reduction_cpu(
        target: ps.Target,
        dtype: str,
        op: str,
        dims: int):
    config = ps.CreateKernelConfig(target=target)
    config.cpu.openmp.enable = True

    if target.is_vector_cpu():
        config.cpu.vectorize.enable = True
        config.cpu.vectorize.assume_inner_stride_one = True

    ast_reduction = get_reduction_assign_ast(dtype, op, dims, config)
    kernel_reduction = ast_reduction.compile()

    array = get_cpu_array(dtype, op, dims)
    reduction_array = np.full((1,), init_w, dtype=dtype)

    kernel_reduction(x=array, w=reduction_array)
    assert np.allclose(reduction_array, get_expected_solution(op, array))


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
@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("assume_warp_aligned_block_size", [True, False])
@pytest.mark.parametrize("use_block_fitting", [True, False])
@pytest.mark.parametrize("warp_size", [32, None])
@pytest.mark.skipif(
    Target.CUDA not in Target.available_targets(), reason="CUDA is not available"
)
def test_reduction_gpu(
    dtype: str,
    op: str,
    dims: int,
    assume_warp_aligned_block_size: bool,
    use_block_fitting: bool,
    warp_size: int | None,
):
    import cupy as cp

    cfg = ps.CreateKernelConfig(target=ps.Target.GPU)
    cfg.gpu.assume_warp_aligned_block_size = assume_warp_aligned_block_size
    if warp_size:
        cfg.gpu.warp_size = warp_size

    ast_reduction = get_reduction_assign_ast(dtype, op, dims, cfg)
    kernel_reduction = ast_reduction.compile()

    if use_block_fitting and warp_size:
        kernel_reduction.launch_config.fit_block_size((warp_size, 1, 1))

    array = get_cpu_array(dtype, op, dims)
    reduction_array = np.full((1,), init_w, dtype=dtype)

    array_gpu = cp.asarray(array)
    reduction_array_gpu = cp.asarray(reduction_array)

    kernel_reduction(x=array_gpu, w=reduction_array_gpu)
    assert np.allclose(reduction_array_gpu.get(), get_expected_solution(op, array))
