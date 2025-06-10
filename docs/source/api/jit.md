# JIT Compilation

## Base Infrastructure

```{eval-rst}
.. module:: pystencils.jit

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

    KernelWrapper
    JitBase
    NoJit

.. autodata:: no_jit
```

## CPU Just-In-Time Compiler

The CPU JIT compiler 
- embeds a kernel's code into a prepared C++ frame, which includes routines
  that map NumPy arrays and Python scalars to kernel arguments,
  and perform shape and type checks on these arguments;
- invokes a host C++ compiler to compile and link the generated code as a
  Python extension module;
- dynamically loads that module and exposes the compiled kernel to the user.

```{eval-rst}
.. module:: pystencils.jit.cpu

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  CpuJit
```

### Compiler Infos

The properties of the host compiler are defined in a `CompilerInfo` object.
To select a custom host compiler and customize its options, set up and pass
a custom compiler info object to `CpuJit`.

```{eval-rst}
.. module:: pystencils.jit.cpu.compiler_info

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  CompilerInfo
  GccInfo
  ClangInfo
  AppleClangInfo
```

### Implementation Details

```{eval-rst}
.. currentmodule:: pystencils.jit.cpu

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  cpujit.ExtensionModuleBuilderBase
  default_module_builder.DefaultExtensionModuleBuilder
  default_module_builder.DefaultCpuKernelWrapper
```

## CuPy-based GPU JIT

```{eval-rst}
.. module:: pystencils.jit.gpu_cupy

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  CupyJit
  CupyKernelWrapper
  LaunchGrid
```
