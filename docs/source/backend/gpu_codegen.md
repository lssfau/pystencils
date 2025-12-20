---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
mystnb:
  execution_mode: cache
---

(gpu_codegen)=
# Walkthrough: Code Generation for GPUs

This document shall give an overview of the faclities of the pystencils code generation backend
for the generation of GPU kernels for the CUDA and HIP platforms.
It complements the previous [](page_backend_walkthrough), so if you haven't read that, we advise you take
a look at it first.

## Preparation: Setup and Parsing

```{code-cell} ipython3
:tags: [remove-cell]

import sympy as sp
import pystencils as ps
```

```{code-cell} ipython3
:tags: [remove-cell]

f = ps.fields("f: [3D]", layout="fzyx")
x, c = sp.symbols("x, c")
wmax = ps.TypedSymbol("wmax", ps.DynamicType.NUMERIC_TYPE)

assignments = [
    ps.Assignment(x, f()),
    ps.Assignment(f(), c * x),
    ps.MaxReductionAssignment(wmax, x)
]
```

We'll be using the same kernel as in the [walkthrough](page_backend_walkthrough) for demonstration:

```{code-cell} ipython3
:tags: [remove-input]
assignments
```

The setup and parsing steps are the same as in [](walkthrough_context_setup) and [](walkthrough_parsing),
and we arrive at the same IR for the kernel body:

```{code-cell} ipython3
:tags: [remove-input]

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    FullIterationSpace
)

ctx = KernelCreationContext()
ispace = FullIterationSpace.create_with_ghost_layers(ctx, 0, f)
ctx.set_iteration_space(ispace)

from pystencils.backend.kernelcreation import AstFactory

factory = AstFactory(ctx)

from pystencils.backend.ast.structural import PsBlock

body = PsBlock([
    factory.parse_sympy(asm) for asm in assignments
])

ps.inspect(body)
```

## Iteration Spaces via GPU Block/Thread Indexing

For GPU, same as for CPU, we continue by manifesting the kernel's index space
as an [axes cube](PsAxesCube) via the [ast factory](AstFactory), and canonicalizing it:

```{code-cell} ipython3
cube = factory.cube_from_ispace(ispace, body)

from pystencils.backend.transformations import CanonicalizeSymbols

canonicalize = CanonicalizeSymbols(ctx)
cube = canonicalize(cube)

ps.inspect(cube)
```

Now, instead of expanding this cube into a loop nest, we use GPU block and thread axes.
We have three available axis types:
 - {any}`gpu_block <AxisExpansion.gpu_block>` maps the cube's leading axis
   onto the current GPU block index (`blockIdx` in CUDA, HIP);
 - {any}`gpu_thread <AxisExpansion.gpu_thread>` parallelizes the next axis
   using the current GPU thread index (`threadIdx` in CUDA, HIP); and
 - {any}`gpu_block_x_thread <AxisExpansion.gpu_block_x_thread>` maps the leading axis
    onto the typical combination `blockIdx * blockDim + threadIdx`.

These axes may be freely combined with target-agnostic axes, like {any}`loop <AxisExpansion.loop>`,
to construct a wide range of GPU iteration strategies.

Here's a sample expansion strategy using all three axis types:

```{code-cell} ipython3
from pystencils.backend.transformations import AxisExpansion

ae = AxisExpansion(ctx)
strategy = ae.create_strategy(
    [
        ae.gpu_block("y"),
        ae.gpu_thread("y"),
        ae.gpu_block_x_thread("x")
    ]
)

kernel_ast = strategy(cube)
ps.inspect(kernel_ast)
```

Next, we invoke {any}`MaterializeAxes` to turn the GPU iteration axes into actual code:

```{code-cell} ipython3
from pystencils.backend.transformations import MaterializeAxes

mat_axes = MaterializeAxes(ctx)
kernel_ast = mat_axes(kernel_ast)

ps.inspect(kernel_ast)
```

The resulting code uses IR functions `blockIdx.Y()`, `threadIdx.Y()`, etc., to define
the axis counters. Also, kernel guards are introduced to make sure the kernel body runs only
if the indices are inside the iteration space.
Furthermore, you can see that the reduction to `w` is realized as an abstract write-back function
within the iteration guard.

## Lowering

At this point, the kernel AST yet contains IR features (GPU index functions, the reduction write-back)
that must still be mapped onto target-specific implementations.
To perform these lowerings, we must first set up a GPU platform object.
Let's use the CUDA platform in this case:

```{code-cell} ipython3
from pystencils.backend.platforms import CudaPlatform

platform = CudaPlatform(
  ctx,
  assume_warp_aligned_block_size=True,
  warp_size=32
)
```

We have specified two optional parameters here: the GPU's warp size (which is always 32 for CUDA),
and the assumption that the total size of a thread block at runtime will always be a multiple of two.
These assumptions allow the platform to emit optimized code for the reduction onto `w`, as we will see below.

So, let us run the {any}`SelectFunctions` and {any}`LowerToC` passes:

```{code-cell} ipython3
:tags: [hide-output]

from pystencils.backend.transformations import SelectFunctions, LowerToC

lower_to_c = LowerToC(ctx)
kernel_ast = lower_to_c(kernel_ast)

select_functions = SelectFunctions(platform)
kernel_ast = select_functions(kernel_ast)

ps.inspect(kernel_ast)
```

Here you can see that the GPU indexing intrinsics were replaced
by the corresponding CUDA thread and block index variables.
Also, the reduction write-back was materialized using a *warp-level reduction* construct: 
First, shuffle instructions are used to perform a tree-reduction within the current warp,
such that afterward the warp's leading thread holds the warp's reduction result,
and only it needs to perform an atomic memory update.

## Wrap Up the Kernel

At the end, the finished kernel must be wrapped in a `GpuKernel` object, to be given to the JIT
compiler and runtime system.
We're using the {any}`KernelFactory` and {any}`GpuIndexing` classes from the `codegen` module for this.
We define a factory for the kernel's launch configurations using {any}`ManualLaunchConfiguration`,
and set up an instance of the {any}`CupyJit` just-in-time compiler.

```{code-cell} ipython3
from pystencils.codegen.driver import KernelFactory
from pystencils.codegen.gpu_indexing import GpuIndexing, ManualLaunchConfiguration
from pystencils.jit.gpu_cupy import CupyJit

def launch_config_factory() -> ManualLaunchConfiguration:
    return ManualLaunchConfiguration(GpuIndexing.get_hardware_properties(target))

kfactory = KernelFactory(ctx)
kernel = kfactory.create_gpu_kernel(
    platform,
    kernel_ast,
    "kernel",
    ps.Target.CUDA,
    CupyJit(),
    launch_config_factory
)

ps.inspect(kernel, show_cpp=True)
```

(gpu_launch_config)=
## Note on Launch Configurations

A GPU kernel's launch configuration can be pre-determined wholly or partly by the code generator,
or kept fully in the hands of the runtime system. These different types of launch configurations
are implemented by a set of classes in the `pystencils.codegen.gpu_indexing` module.
Since a concrete launch configuration is not specific to the kernel itself, but to the kernels'
invocation site, the code generator only attaches a *factory function* for launch configurations
to `GpuKernel`. It is up to the runtime system to locally instantiate and configure a launch configuration.
To determine the actual launch grid, the launch configuration must be evaluated at the kernel's call site
by passing the required parameters to `GpuLaunchConfiguration.evaluate`

The {any}`CupyJit`, for instance, will create the launch configuration object while preparing the JIT-compiled
kernel wrapper object. The launch config is there exposed to the user, who may modify some of its properties.
These depend on the type of the launch configuration:
while the {any}`AutomaticLaunchConfiguration` permits no modification and computes grid and block size directly from kernel
parameters,
the {any}`ManualLaunchConfiguration` requires the user to manually specifiy both grid and block size.
The {any}`DynamicBlockSizeLaunchConfiguration` dynamically computes the grid size from either the default block size
or a computed block size. Computing block sizes can be signaled by the user via the `trim_block_size` or 
`fit_block_size` member functions. These function receive an initial block size as an argument and adapt it.
The `trim_block_size` function trims the initial block size with the sizes of the iteration space, i.e. it takes 
the minimum value of both sizes per dimension. The `fit_block_size` performs a block fitting algorithm that adapts 
the initial block size by incrementally enlarging the trimmed block size until it is large enough 
and aligns with the warp size.

The `evaluate` method can only be used from within a Python runtime environment.
When exporting pystencils CUDA kernels for external use in C++ projects,
equivalent C++ code evaluating the launch config must be generated.
This is the task of, e.g., [pystencils-sfg](https://pycodegen.pages.i10git.cs.fau.de/pystencils-sfg/).
