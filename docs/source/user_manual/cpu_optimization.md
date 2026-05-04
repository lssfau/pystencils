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
import matplotlib.pyplot as plt
```

(guide_cpu_optimization)=
# Optimizing CPU Kernels

Pystencils is capable of automatically performing various optimizations on CPU kernels in order to
accelerate them.
This guide introduces these optimizations and explains how they can be activated and controlled
in the code generator.
They will be illustrated using the following simple stencil kernel:

```{code-cell} ipython3
src, dst = ps.fields("src, dst: [2D]")
asm = ps.Assignment(dst(), src[-1, 0])
```

## Parallel Execution with OpenMP

Pystencils can add OpenMP instrumentation to kernel loop nests to have them execute in parallel.
OpenMP can be enabled by setting the `cpu.openmp.enable` code generator option to `True`:

```{code-cell} ipython3
cfg = ps.CreateKernelConfig()
cfg.cpu.openmp.enable = True
```

This will cause pystencils to parallelize the outermost kernel loop
with `static` scheduling:

```{code-cell} ipython3
ker = ps.create_kernel(asm, cfg)
ps.inspect(ker, show_cpp=True)
```

The `cpu.openmp` category allows us to customize parallelization behaviour through OpenMP clauses.
Use the `collapse`, `schedule` and `num_threads` options to set values for the respective clauses.

## Tiling and Blocking

Loop tiling and blocking can be effective optimizations for improving cache reuse and temporal locality in stencil kernels.
Pystencils allows generating block loops by setting the desired block sizes in the code generator config:

```{code-cell} ipython3
cfg = ps.CreateKernelConfig()
cfg.cpu.loop_blocking = (16, ps.TypedSymbol("k", ps.DynamicType.INDEX_TYPE))
```

The blocking spec must be a tuple with as many block sizes as there are iteration space dimensions.
Block sizes must be either integers or SymPy expressions of the kernel's {any}`index type <DynamicType.INDEX_TYPE>`.
They can also be `None`, in which case the corresponding dimension will not be blocked.

The above configuration leads to the following kernel:

```{code-cell} ipython3
ker = ps.create_kernel(asm, cfg)
ps.inspect(ker, show_cpp=True)
```

It now contains four loops: two block loops, and two inner loops.
The outer block loop has a fixed size of $16$, while the second loop's block size depends on the kernel parameter $k$.

Here's a few more things to consider:
 - The block sizes refer to the *logical* dimensions of the iteration space, ordered $x$-$y$-$z$;
   block loops will be ordered according to the same loop reordering rules as the primary iteration loops.
 - When enabling OpenMP and blocking together, the outermost block loop will be parallelized.
   To parallelize multiple block loops together, consider adding a `collapse` clause to the OpenMP options.

## Vectorization

To fully utilize the SIMD capabilities of modern CPUs, pystencils can vectorize
generated kernels using SIMD intrinsics.
To enable vectorization, set the `target` option to a SIMD-capable CPU architecture,
and set `cpu.vectorize.enable` to `True`.
The following snippet, for example, enables vectorization for the `AVX512` architecture:

```{code-cell} ipython3
cfg = ps.CreateKernelConfig()
cfg.target = ps.Target.X86_AVX512
cfg.cpu.vectorize.enable = True
```

This will lead to pystencils vectorizing the innermost loop.
When inspecting the generated code, we will see that
both a vectorized SIMD-, and scalar remainder loop have been generated:

```{code-cell} ipython3
ker = ps.create_kernel(asm, cfg)
ps.inspect(ker, show_cpp=True)
```

This vectorized kernel is far from ideal, though.
The kernel accesses memory through *gather* and *scatter* instructions instead of
using contiguous, packed loads and stores.
Since pystencils does not know the access strides of the kernel's fields beforehand,
it cannot safely emit packed memory accesses.
Instead, it must access memory in a strided manner, using stride parameters known only at runtime.

:::{note}
Scatter instructions where only introduced into x86-64 with the AVX512 extension.
Code generation in the above example would therefore have failed for any older x86
architecture level (SSE or AVX).
:::

### Packed Memory Accesses

To use contiguous memory accesses, all fields must be set up to use a *structure-of-arrays* layout,
with unit stride in the fastest coordinate.
It is the responsibility of the user and runtime system to make sure that all arrays
passed to the kernel fulfill this requirement.
If it is fulfilled, we can instruct the code generator to emit packed memory accesses
by setting `cpu.vectorize.assume_inner_stride_one` to `True`:

```{code-cell} ipython3
cfg.cpu.vectorize.assume_inner_stride_one = True

ker = ps.create_kernel(asm, cfg)
ps.inspect(ker, show_cpp=True)
```

### Strided Iteration Spaces

Even with the unit stride assumption, packed loads and stores can only be used
if the iteration space itself is contiguous in the fastest coordinate.
Kernels operating on a striped iteration space (see [](#section_strided_iteration))
still have to use strided accesses.
Nevertheless, setting the unit stride assumption - or otherwise fixing field strides beforehand - 
will lead to more compact code, as stride computations can be done fully at compile-time,
as the following example shows:

```{code-cell} ipython3
# Only processes every third element in the fastest coordinate
cfg.iteration_slice = ps.make_slice[:, ::3]

cfg.cpu.vectorize.assume_inner_stride_one = True

ker = ps.create_kernel(asm, cfg)
ps.inspect(ker, show_cpp=True)
```
