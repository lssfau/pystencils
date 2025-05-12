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
:tags: [remove-cell, raises-exception]

import sympy as sp
import pystencils as ps
import numpy as np
import cupy as cp

from enum import Enum
```

(guide_reductions)=
# Reductions in Pystencils

Reductions play a vital role in numerical simulations as they allow aggregating data across multiple elements, 
such as computing sums, products over an array or finding its minima or maxima.

## Specifying Assignments with Reductions

In pystencils, reductions are made available via specialized assignments, namely `ReductionAssignment`.
Here is a snippet creating a reduction assignment for adding up all elements of a field:

```{code-cell} ipython3
r = ps.TypedSymbol("r", "double")
x = ps.fields(f"x: double[3D]", layout="fzyx")

assign_sum = ps.AddReductionAssignment(r, x.center())
```

For each point in the iteration space, the left-hand side symbol `r` accumulates the contents of the 
right-hand side `x.center()`. In our case, the `AddReductionAssignment` denotes an accumulation via additions.

**Pystencils requires type information about the reduction symbols and thus requires `r` to be a `TypedSymbol`.**

The following reduction assignment classes are available in pystencils:    
* `AddReductionAssignment`: Builds sum over elements
* `SubReductionAssignment`: Builds difference over elements
* `MulReductionAssignment`: Builds product over elements
* `MinReductionAssignment`: Finds minimum element
* `MaxReductionAssignment`: Finds maximum element

:::{note}
Alternatívely, you can also make use of the `reduction_assignment` function
to specify reduction assignments:
:::

```{code-cell} ipython3
from pystencils.sympyextensions import reduction_assignment
from pystencils.sympyextensions.reduction import ReductionOp

assign_sum = reduction_assignment(r, ReductionOp.Add, x.center())
```

For other reduction operations, the following enums can be passed to `reduction_assignment`.

```{code-cell} python3
class ReductionOp(Enum):
    Add = "+"
    Sub = "-"
    Mul = "*"
    Min = "min"
    Max = "max"
```

## Generating and Running Reduction Kernels

With the assignments being fully assembled, we can finally invoke the code generator and 
create the kernel object via the {any}`create_kernel` function.

### CPU Platforms

For this example, we assume a kernel configuration for CPU platforms with no optimizations explicitly enabled.

```{code-cell} ipython3
cfg = ps.CreateKernelConfig(target=ps.Target.CurrentCPU)
kernel = ps.create_kernel(assign_sum, cfg)

ps.inspect(kernel)
```

:::{note}
The generated reduction kernels may vary vastly for different platforms and optimizations.
You can find a  detailed description of configuration choices and their impact on the generated code below.
:::

The kernel can be compiled and run immediately.

To execute the kernel on CPUs, not only a {any}`numpy.ndarray` has to be passed for each field
but also one for exporting reduction results. 
The export mechanism can be seen in the previously generated code snippet. 
Here, the kernel obtains a pointer with the name of the reduction symbol (here: `r`).
This pointer is used for exporting the reduction result back from the kernel.
Please note that the **values passed via pointer will not be overwritten** 
but will be incorporated in the reduction computation.
Since our reduction result is a single scalar value, it is sufficient to set up an array comprising a singular value.

```{code-cell} ipython3
    kernel_func = kernel.compile()

    x_array = np.ones((4, 4, 4), dtype="float64")
    reduction_result = np.zeros((1,), dtype="float64")

    kernel_func(x=x_array, r=reduction_result)
    
    reduction_result[0]
```

### GPU Platforms

Please note that **reductions are currently only supported for CUDA platforms**.
Similar to the CPU section, a base variant for NVIDIA GPUs without 
explicitly employing any optimizations is shown:

```{code-cell} ipython3
    cfg = ps.CreateKernelConfig(target=ps.Target.CUDA)

    kernel_gpu = ps.create_kernel(assign_sum, cfg)

    ps.inspect(kernel_gpu)
```

The steps for running the generated code on NVIDIA GPUs are identical but the fields and the write-back pointer 
now require device memory, i.e. instances of {any}`cupy.ndarray`.

## Optimizations for Reductions

Going beyond the aforementioned basic kernel configurations,
we now demonstrate optimization strategies for different platforms 
that can be applied to reduction kernels and show what impact they have.

### CPU Platforms

For CPU platforms, standard optimizations are employing SIMD vectorization and shared-memory parallelism using OpenMP.
The supported SIMD instruction sets for reductions are:
* SSE3
* AVX/AVX2
* AVX512

Below you can see that an AVX vectorization was employed by using the target `Target.X86_AVX`.
**Note that reductions require `assume_inner_stride_one` to be enabled.**
This is due to the fact that other inner strides would require masked SIMD operations 
which are not supported yet.

```{code-cell} ipython3
# configure SIMD vectorization
cfg = ps.CreateKernelConfig(
  target=ps.Target.X86_AVX,
)
cfg.cpu.vectorize.enable = True
cfg.cpu.vectorize.assume_inner_stride_one = True

# configure OpenMP parallelization
cfg.cpu.openmp.enable = True
cfg.cpu.openmp.num_threads = 8

kernel_cpu_opt = ps.create_kernel(assign_sum, cfg)

ps.inspect(kernel_cpu_opt)
```

### GPU Platforms

As evident from the generated kernel for the base variant, atomic operations are employed 
for updating the pointer holding the reduction result.
Using the *explicit warp-level instructions* provided by CUDA allows us to achieve higher performance compared to
only using atomic operations.
To generate kernels with warp-level reductions, the generator expects that CUDA block sizes are divisible by 
the hardware's warp size.
**Similar to the SIMD configuration, we assure the code generator that the configured block size fulfills this
criterion by enabling `assume_warp_aligned_block_size`.**
While the default block sizes provided by the code generator already fulfill this criterion,
we employ a block fitting algorithm to obtain a block size that is also optimized for the kernel's iteration space.

You can find more detailed information about warp size alignment in {ref}`gpu_codegen`.

```{code-cell} ipython3
    cfg = ps.CreateKernelConfig(target=ps.Target.CUDA)
    cfg.gpu.assume_warp_aligned_block_size = True

    kernel_gpu_opt = ps.create_kernel(assign_sum, cfg)
    
    kernel_func = kernel_gpu_opt.compile()
    kernel_func.launch_config.fit_block_size((32, 1, 1))

    ps.inspect(kernel_gpu_opt)
```
