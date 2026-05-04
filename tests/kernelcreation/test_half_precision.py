import pytest
import platform
import sys

import numpy as np
import pystencils as ps

from pystencils.jit.cpu import CpuJit, ClangInfo, AppleClangInfo


@pytest.fixture
def cpujit(target: ps.Target, tmp_path) -> CpuJit:
    #   Set target in compiler info such that `-march` is set accordingly
    if sys.platform == "darwin":
        cinfo = AppleClangInfo(target=target)
    else:
        cinfo = ClangInfo(target=target)

    return CpuJit(
        compiler_info=cinfo,
        objcache=tmp_path
    )


@pytest.mark.parametrize('target', (ps.Target.CPU, ps.Target.CurrentGPU))
def test_half_precison(target, cpujit):
    if target == ps.Target.CPU:
        if not platform.machine() in ['arm64', 'aarch64']:
            pytest.xfail("skipping half precision test on non arm platform")

    if target.is_gpu():
        pytest.importorskip("cupy")

    dh = ps.create_data_handling(domain_size=(10, 10), default_target=target)

    f1 = dh.add_array("f1", values_per_cell=1, dtype=np.float16)
    dh.fill("f1", 1.0, ghost_layers=True)
    f2 = dh.add_array("f2", values_per_cell=1, dtype=np.float16)
    dh.fill("f2", 2.0, ghost_layers=True)

    f3 = dh.add_array("f3", values_per_cell=1, dtype=np.float16)
    dh.fill("f3", 0.0, ghost_layers=True)

    up = ps.Assignment(f3.center, f1.center + 2.1 * f2.center)

    config = ps.CreateKernelConfig(target=dh.default_target, default_dtype=np.float32)
    if target.is_cpu():
        config.jit = cpujit
    ast = ps.create_kernel(up, config=config)

    kernel = ast.compile()

    dh.run_kernel(kernel)
    dh.all_to_cpu()

    assert np.all(dh.cpu_arrays[f3.name] == 5.2)
    assert dh.cpu_arrays[f3.name].dtype == np.float16
