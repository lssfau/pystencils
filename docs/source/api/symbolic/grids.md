# Patches and Algebraic Fields [Experimental]

:::{caution}
The `pystencils.grids` module is meant to replace the legacy fields module (`pystencils.field`) in the near future.
It is still under active development and considered *experimental*.
:::

```{eval-rst}
.. module:: pystencils.grids
```

## Patches and Patch Data

```{eval-rst}
.. module:: pystencils.grids.patch

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  Patch
  PatchGrid
  VariablePlacement

.. module:: pystencils.grids.patch_data

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  PatchData
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
  CreateNdArray
  ViewNdArray
```