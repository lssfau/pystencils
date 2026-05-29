# New-Style Algebraic Fields [Experimental]

:::{caution}
The `pystencils.grids` module is meant to replace the legacy fields module (`pystencils.field`) in the near future.
It is still under active development and considered *experimental*.
:::

```{eval-rst}
.. module:: pystencils.grids
```

## Tensor Fields

```{eval-rst}
.. module:: pystencils.grids.tensor_field

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  TensorField
  TensorFieldAccess
```

## Helper Classes

```{eval-rst}
.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  MemoryLayout
```

## Infrastructure and Protocols


```{eval-rst}
.. module:: pystencils.grids.protocols

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  IField
  IFieldAccess
  FieldBufferSpec
  IterationLimits
```