# pystencils v2.0-dev Documentation

:::{note}
  You are currently viewing the documentation pages for the development revision |release|
  of pystencils 2.0.
  These pages have been generated from the branch 
  [v2.0-dev](https://i10git.cs.fau.de/pycodegen/pystencils/-/tree/v2.0-dev?ref_type=heads)

  Pystencils 2.0 is currently under development. 
  It marks a complete re-design of the package's internal structure;
  furthermore, it will bring a set of new features and API changes.
  Be aware that many features are still missing or might have brand-new bugs in this version. Likewise, these documentation pages are still incomplete.
:::

Welcome to the documentation and reference guide of *pystencils*!
*Pystencils* offers a symbolic language and code generator for the development of high-performing
numerical kernels for both CPU and GPU targets. 
Its features include:

- **Symbolic Algebra:** Design numerical methods on a mathematical level using the full power
  of the [SymPy](https://sympy.org) computer algebra system.
  Make use of pystencils' discretization engines to automatically derive finite difference- and finite volume-methods,
  and take control of numerical precision using the [versatile type system](#page_type_system).
- **Kernel Description:** Derive and optimize stencil-based update rules using a symbolic abstraction
  of numerical [fields](#page_symbolic_language).
- **Code Generation:** [Generate and compile](#guide_kernelcreation) high-performance parallel kernels for CPUs and GPUs.
  Accelerate your kernels on multicore CPUs using the automatic OpenMP parallelization
  and make full use of your cores' SIMD units through the highly configurable vectorizer.
- **Rapid Prototyping:** Run your numerical solvers on [NumPy](https://numpy.org) and [CuPy](https://cupy.dev) arrays
  and test them interactively inside [Jupyter](https://jupyter.org) notebooks.
  Quickly set up numerical schemes, apply initial and boundary conditions, evaluate them on model problems
  and rapidly visualize the results using matplotlib or VTK.
- **Framework Integration:** Export your kernels and use them inside HPC frameworks
  such as [waLBerla] to build massively parallel simulations.

Topics
------

:::{toctree}
:maxdepth: 1
:caption: Getting Started

installation
tutorials/index
:::

:::{toctree}
:maxdepth: 1
:caption: User Manual

user_manual/symbolic_language
user_manual/kernelcreation
user_manual/cpu_optimization
user_manual/gpu_kernels
user_manual/WorkingWithTypes
user_manual/reductions
user_manual/random_numbers
:::

:::{toctree}
:maxdepth: 1
:caption: API Reference

api/symbolic/index
api/types
api/codegen
api/jit
:::

:::{toctree}
:maxdepth: 1
:caption: Topics

contributing/index
migration
backend/index
:::

Projects using pystencils
-------------------------

- [lbmpy](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/)
- [walberla]
- [HyTeG Operator Generator (HOG)](https://hyteg.pages.i10git.cs.fau.de/hog/)


[walberla]: https://walberla.net
