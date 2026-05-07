# pystencils

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

## Table of Contents

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
user_manual/sycl_kernels
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

## Projects using pystencils

- [lbmpy](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/)
- [walberla]
- [HyTeG Operator Generator (HOG)](https://hyteg.pages.i10git.cs.fau.de/hog/)


[walberla]: https://walberla.net

## Cite Us

When using *pystencils* in your published works, please cite the following articles:

**Overview:**
  - M. Bauer et al, Code Generation for Massively Parallel Phase-Field Simulations. Association for Computing Machinery, 2019. https://doi.org/10.1145/3295500.3356186

**Performance Modelling:**
  - D. Ernst et al, Analytical performance estimation during code generation on modern GPUs. Journal of Parallel and Distributed Computing, 2023. https://doi.org/10.1016/j.jpdc.2022.11.003
