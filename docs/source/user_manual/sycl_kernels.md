---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
mystnb:
  execution_mode: cache
---

```{code-cell} ipython3
:tags: [remove-cell]

import sympy as sp
import pystencils as ps
import numpy as np
```

(guide_syclkernels)=
# Pystencils for SYCL

Pystencils offers code generation for SYCL targets {any}`Target.SYCL`.
The Pystencils Jit uses [dpctl](https://github.com/IntelPython/dpctl) to manage devices, Queues and USM allocations.
For more information on dpctl, refer to [their documenation](https://intelpython.github.io/dpctl/)
It can be installed via pip to our virtual environment
```bash
pip install dpctl
```

:::{note}
It is possible to target Nvidia and AMD GPUs via Intel oneAPI. From version 2025.3 the corresponding
plugin needs to be build from source [oneAPI docs](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2025.html#inpage-nav-4-5-1)
For previous versions there were the pre-built plugins from [Codeplay](https://developer.codeplay.com/products/oneapi/nvidia/home/index.html).
:::

## Setup
To list all available device you can run

```{code-cell} ipython3
:tags: [raises-exception]
import dpctl
dpctl.get_devices()
```

To launch a kernel on a device a {any}`dpctl.SyclQueue` is needed

```{code-cell} ipython3
q = dpctl.SyclQueue()
q.get_sycl_device()
```

:::{tip}
If `dpctl` can not find any devices take care that `intel-cmplr-lib-rt`, `intel-cmplr-lib-ur`, `intel-cmplr-lic-rt`, and `intel-sycl-rt` package are installed in the version that corresponds with the oneAPI installation.
Some times it also helps to add the `lib` folder of our environment to our `LD_LIBRARY_PATH` (see
the `noxfile.py` for reference)
:::

Then we need to create some array as {any}`dpctl.tensor.usm_ndarray` allocations with {any}`dpctl.tensor`

```{code-cell} ipython3
import dpctl.tensor as dpt
f_arr = dpt.ones((10, 10), dtype=np.float32, sycl_queue=q)
g_arr = dpt.zeros((10, 10), dtype=np.float32, sycl_queue=q)
f_arr
```


## Executing range kernels

A simple SYCL kernel can be executed like this:
The SYCL kernels are launched with in {any}`parallel_for <sycl:handler-parallel_for>`.
To create a {any}`sycl:range` based kernel use {any}`Target.SYCL` in {any}`CreateKernelConfig`

```{code-cell} ipython3
dtype = "float32"
f, g = ps.fields(f"f, g: {dtype}[2D]")
asm = ps.Assignment(g.center(), 2.0 * f.center())
config = ps.CreateKernelConfig(target=ps.Target.SYCL)
ker = ps.create_kernel(asm, config=config)
kfunc = ker.compile()
kfunc(f=f_arr, g=g_arr, queue=q)
q.wait()
g_arr
```

## Executing ND-range kernels

It is also possible to create ND-range kernels
To create a kernel launched with {any}`sycl:nd_range` the {any}`SyclOptions.automatic_block_size`
needs to be set to `false`.

```{code-cell} ipython3
config = ps.CreateKernelConfig(target=ps.Target.SYCL)
config.sycl.automatic_block_size = False
config.gpu.manual_launch_grid = True
ker = ps.create_kernel(asm, config=config)
kfunc = ker.compile()
```
The indexing scheme and the launch configuration work in the same way as for the [GPU Targets](indexing_and_launch_config).
```{code-cell} ipython3
kfunc.launch_config.block_size = (2, 2)
kfunc.launch_config.grid_size = (2, 3)
kfunc(f=g_arr, g=f_arr, queue=q)
q.wait()
f_arr
```

## Note on targeting AMD GPUs

:::{warning}
Targeting AMD GPUs via dpctl is currently experimental and not tested thoroughly.
:::

The SYCL compiler uses the `-fsycl-target` flag to add support for different targets in the produced
binary.
The `spirv64` target is for CPUs or Intel GPUs, the `nvptx64-nvidia-cuda` is for Nvida GPUs, and `amdgcn-amd-amdhsa` is for AMD GPUs.
However, for AMD GPUs the exact architecture needs to be specified (for more info [see here](https://developer.codeplay.com/products/oneapi/amd/2025.2.0/guides/get-started-guide-amd.html#use-dpc-to-target-amd-gpus))
The {any}`jit.sycl.SYCLClangInfo` it tries to figure out the correct flags for each device that it can find
via the SYCL runtime.
If the detection of an AMD GPU does not work {any}`jit.sycl.SYCLClangInfo.amd_offload_architecutres` allows
to specify the correct architecture name manually.
