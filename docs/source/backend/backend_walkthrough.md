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

(page_backend_walkthrough)=
# Walkthrough: The Kernel Translation Process

+++

This guide is meant to give an overview of pystencils' code generation toolkit
by walking through the kernel translation procedure step by step.
We will cover the various steps of the pipeline, which are
 - Setup of the context;
 - Parsing of the symbolic kernel from SymPy into the intermediate representation (IR);
 - Materialization of the kernel's iteration space to an iteration strategy
   using the iteration axes system;
 - A selection of optimizing transformations;
 - Lowering of the IR to C++ code;
 - And finally, packaging of the kernel as an output object.

In the process, a number of core concepts and classes of pystencils' backend will be introduced.
This guide effectively shows an abridged version of the translation pipeline implemented in
the {any}`DefaultKernelCreationDriver`.

```{code-cell} ipython3
import sympy as sp
import pystencils as ps
```

For illustration, we're going to define a very simple kernel which scales a scalar field $f$
by a factor $c$ and computes the maximum value of $f$ in the process:

```{code-cell} ipython3
f = ps.fields("f: [3D]", layout="fzyx")
x, c = sp.symbols("x, c")
wmax = ps.TypedSymbol("wmax", ps.DynamicType.NUMERIC_TYPE)

assignments = [
    ps.Assignment(x, f()),
    ps.Assignment(f(), c * x),
    ps.MaxReductionAssignment(wmax, x)
]

assignments
```

(walkthrough_context_setup)=
## Context Setup

+++

Before kernel translation can begin, we need to instantiate the backend's context objects:
 - The {any}`KernelCreationContext` manages all global information about the kernel, and primarily serves
   as a symbol table for the kernel's variables, memory buffers, and reduction targets.
   The context object is conventionally called `ctx`.
 - The {any}`IterationSpace` defines the index space on which the kernel is executed;
   it is attached to `ctx` and required during parsing of field accesses and later
   to materialize the index space to an iteration strategy.

Import the required classes from `pystencils.backend.kernelcreation` and initialize them:

```{code-cell} ipython3
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    FullIterationSpace
)

ctx = KernelCreationContext()
ispace = FullIterationSpace.create_with_ghost_layers(ctx, 0, f)
ctx.set_iteration_space(ispace)
```

Here, we created a {any}`FullIterationSpace` object, which represents a dense index space.
Using {any}`create_with_ghost_layers <FullIterationSpace.create_with_ghost_layers>`,
we initialized the index space according to the rank and memory layout of the field $f$;
it is now a 3D index space with dimensions ordered such that the shortest-stride dimension
of $f$ is mapped by the fastest-increasing coordinate of `ispace`.

(walkthrough_parsing)=
## Parsing of the Kernel Body

Next, the kernel's symbolic representation from above must be translated into pystencil's intermediate representation.
The IR has two layers:
 - the IR's [atomic objects](ir_objects), which are *constants*, *symbols*, *memory buffers* and *functions*;
 - and the [abstract syntax tree (AST)](pystencils.backend.ast), which represents the kernel's generated code.

While constants and functions are transient and created on-demand, symbols and buffers must be unique and are
therefore managed by the kernel creation context.
Symbols (`PsSymbol`) are the backend analouge to SymPy's symbols;
in turn, buffers represent the memory regions behind the frontend's fields.

Generating an IR syntax tree from the symbolic kernel is a two-stage process,
constisting of the [freeze](FreezeExpressions) and [typify](Typifier) steps.
During *freeze*, symbolic equations become IR AST nodes, on which *typify* computes and applies data types.
The backend's {any}`AstFactory` class offers a convenient interface to accelerate this process:

```{code-cell} ipython3
from pystencils.backend.kernelcreation import AstFactory

factory = AstFactory(ctx)
```

To create IR objects from symbolic forms we will use {any}`AstFactory` in several places throughout this guide.

The kernel body will be a {any}`PsBlock`, with declarations and assignments parsed from SymPy:

```{code-cell} ipython3
from pystencils.backend.ast.structural import PsBlock

body = PsBlock([
    factory.parse_sympy(asm) for asm in assignments
])

ps.inspect(body)
```

We can use `ps.inspect` to print a text representation of our abstract syntax trees.
Observe that symbols and constants have all been assigned data types by the {any}`Typifier`.
This is according to the first rule for the [canonical form](ast-canonical-form) of abstract syntax trees.

:::{admonition} Canonical Form I: Data Types
:class: important

Each symbol, constant, and expression node inside an AST must be annotated with a data type.
This is ensured by running the {any}`Typifier` on all newly created syntax trees.

:::


## Iteration Space Materialization

Now that our kernel body is complete, we need to turn its index space into syntax structure.
This we will achieve via the [iteration axes system](ast-iteration-axes), which is part of the AST
class hierarchy.

### The Main Iteration Cube

At first, we manifest the entire iteration space as an [axes cube](PsAxesCube),
which represents abstract iteration over all iteration dimensions in their required order.
We use the AST factory to create the cube:

```{code-cell} ipython3
cube = factory.cube_from_ispace(ispace, body)

ps.inspect(cube)
```

Observe the order of the cube's coordinates, which are listed slowest-to-fastest.
According to the `fzyx` memory layout previously specified for the field $f$,
the $x$ coordinate is the fastest while the $z$ coordinate is the slowest.
This is reflected in the cube, where `ctr_2`, the $z$ iteration counter, is listed first.

### Symbol Canonicalization

Now we have reached an important point: All symbols required by the kernel have been introduced
and defined in its AST. This includes, in our case, all numerical symbols, field buffers, and the
iteration counters.
At this point, we should *canonicalize* the symbol declarations.

:::{admonition} Canonical Form II: Symbol Declarations
:class: important

For an AST to be in canonical form,
- Each symbol has at most one declaration;
- Each symbol that is never written to apart from its declaration has a ``const`` type; and
- Each symbol whose type is *not* ``const`` has at least one non-declaring assignment.

This form is achieved by running the {any}`CanonicalizeSymbols` pass on the AST.

:::

```{code-cell} ipython3
from pystencils.backend.transformations import CanonicalizeSymbols

canonicalize = CanonicalizeSymbols(ctx)
cube = canonicalize(cube)

ps.inspect(cube)
```

### Axis Expansion

Now, we will turn the iteration cube into a tree of nested *iteration axes* using the {any}`AxisExpansion`
transformer.
This is where our kernel's iteration strategy is introduced and all decisions concerning loop structure, loop
tiling and blocking, parallelization and vectorization, as well as GPU thread-block indexing are made.

We would like to create a CPU kernel with the outermost loop parallelized using OpenMP, and the innermost
loop vectorized with four SIMD lanes.
We express this as an *expansion strategy*:

```{code-cell} ipython3
from pystencils.backend.transformations import AxisExpansion

ae = AxisExpansion(ctx)
strategy = ae.create_strategy(
    [
        ae.parallel_loop(num_threads=8, schedule="static,16"),
        ae.loop(),
        ae.peel_for_divisibility(4),
        [
            ae.block_loop(4, assume_divisible=True),
            ae.simd(4)
        ],
        [
            ae.loop()
        ]
    ]
)
```

Each step in this strategy modifies or peels off one dimension of the iteration cube,
from slowest to fastest.
Let's briefly take this apart:
 - The first expansion, `parallel_loop()`, introduces a loop parallelized by OpenMP for the
   cube's leading dimension and strips that dimension from the cube.
 - The next expansion, `loop()`, turns the now-leading cube dimension (`ctr_1`) into a plain loop.
 - The only remaining dimension (`ctr_0`) is now *peeled* (`peel_for_divisibility()`);
   the cube split into two sub-cubes.
   Iteration limits of the first sub-cube are selected such that its iteration count is divisible by four,
   while the second sub-cube holds all remaining iterations.
 - We then have a branch in the strategy.
   The first sub-strategy introduces a blocked loop with block size four. The remaining four iterations per block
   are then vectorized using the `simd()` expansion.
   The second sub-strategy merely produces a remainder loop.

Let's apply the strategy to our iteration cube, and observe how it is replaced by iteration axes according to the
strategy definition:

```{code-cell} ipython3
:tags: [hide-output]

kernel_ast = strategy(cube)
ps.inspect(kernel_ast)
```

We now have our entire iteration strategy represented by [axis nodes](PsIterationAxis) in the AST.
These are still fully general, and will be gradually lowered to target-specific implementations.

### Axis-Invariant Code Motion

Before we take the first lowering step, let's optimize our kernel a bit.
At this point, it is advisable to perform an *axis-invariant code motion* pass.
This pass detects declarations that are independent of their surrounding iteration axes
and moves them as far outward as possible to avoid computing them multiple times.

```{code-cell} ipython3
:tags: [hide-output]

from pystencils.backend.transformations import HoistIterationInvariantDeclarations

hoist = HoistIterationInvariantDeclarations(ctx)
kernel_ast = hoist(kernel_ast)

ps.inspect(kernel_ast)
```

As we can see, the declaration of `ctr_0__rem_start` was moved outside of the axes tree.

### Axis Materialization

In the next step, we will materialize the abstract iteration axes to more concrete IR code.
Loop axes will become loops, OpenMP directives will be introduced for parallelization,
and SIMD axes will be manifested as vectorized arithmetic.
During this process, also modulo variables for the kernel's reductions (i.e. `wmax = max(x)`)
are introduced.

Let us thus invoke the axes materializer:

```{code-cell} ipython3
:tags: [hide-output]

from pystencils.backend.transformations import MaterializeAxes

mat_axes = MaterializeAxes(ctx)
kernel_ast = mat_axes(kernel_ast)

ps.inspect(kernel_ast)
```

### Reductions to Memory

Our reduction to `wmax` is not complete yet; the accumulated result from the kernel's modulo variables must still
be written back to the reductions' target memory location.
We invoke the `ReductionsToMemory` pass for this:

```{code-cell} ipython3
:tags: [hide-output]

from pystencils.backend.transformations import ReductionsToMemory

reduce_to_memory = ReductionsToMemory(ctx, ctx.reduction_data.values())
kernel_ast = reduce_to_memory(kernel_ast)

ps.inspect(kernel_ast)
```

## Optimization Passes

At this point, we can run another set of optimization passes.
To illustrate, we will run the {any}`EliminateConstants` pass to simplify any constant
subexpressions in the AST.
First, however, we should run another symbol canonicalization pass for good measure:

```{code-cell} ipython3
:tags: [hide-output]

from pystencils.backend.transformations import EliminateConstants

kernel_ast = canonicalize(kernel_ast)

elim_constants = EliminateConstants(ctx, extract_constant_exprs=True)
kernel_ast = elim_constants(kernel_ast)

ps.inspect(kernel_ast)
```

## Lowering

By now, our kernel has already taken on a very concrete form, but still contains various IR constructs
that are not yet valid C code.
Most importantly, its vectorized arithmetic still needs to be turned into architecture-specific intrinsics,
and it contains arithmetic functions (`max`) that must be mapped onto a platform-dependent implementation.

Let's go through the required lowering passes one by one.

### Select Vector Intrinsics

First, we will lower vectorized operations and functions to a target architecture's vector intrinsics.
We will use an x86 AVX512 architecture, and thus need to set up the corresponding [platform](Platform)
object:

```{code-cell} ipython3
from pystencils.backend.platforms import X86VectorCpu, X86VectorArch

platform = X86VectorCpu(ctx, X86VectorArch.AVX512)
```

Now, we invoke its intrinsics selector on our kernel AST:

```{code-cell} ipython3
:tags: [hide-output]

select_intrin = platform.get_intrinsic_selector()
kernel_ast = select_intrin(kernel_ast)

ps.inspect(kernel_ast)
```

### Lowering of Buffer Accesses and Functions

The IR memory buffer accesses from the vectorized code have now already been lowered to memory access intrinsics,
but there are still buffer accesses left in the remainder loop.
These need to be linearized to raw C pointer arithmetic.
We use the `LowerToC` pass for this:

```{code-cell} ipython3
:tags: [hide-output]

from pystencils.backend.transformations import LowerToC

lower_to_c = LowerToC(ctx)
kernel_ast = lower_to_c(kernel_ast)

ps.inspect(kernel_ast)
```

Finally, all that remains is to map IR functions to target-specific library functions
using the `SelectFunctions` pass.

```{code-cell} ipython3
:tags: [hide-output]

from pystencils.backend.transformations import SelectFunctions

select_functions = SelectFunctions(platform)
kernel_ast = select_functions(kernel_ast)

ps.inspect(kernel_ast)
```

## Wrapping Up

There it is - our finished kernel.
We can now use `ps.inspect` also in C++ mode, since no more non-C++-concepts are left in the AST.
This would previously have failed with a printing error:

```{code-cell} ipython3
ps.inspect(kernel_ast, show_cpp=True)
```

To make the kernel available to the runtime system, JIT compiler,
or [pystencils-sfg](https://pycodegen.pages.i10git.cs.fau.de/pystencils-sfg/),
we need to wrap the AST inside a {any}`Kernel <pystencils.codegen.Kernel>` object.
We use the `KernelFactory` from the `codegen` module for this task.
To create the kernel, we have to specify its platform, AST, name, target, and a
JIT compiler (use {any}`no_jit` if not applicable):

```{code-cell} ipython3
from pystencils.codegen.driver import KernelFactory
from pystencils.jit.cpu import CpuJit, CompilerInfo

kfactory = KernelFactory(ctx)
ker = kfactory.create_generic_kernel(
    platform,
    kernel_ast,
    "my_kernel",
    ps.Target.X86_AVX512,
    CpuJit(CompilerInfo.get_default()),
)

print(ker)
```

This concludes the walkthrough of the kernel creation pipeline.
