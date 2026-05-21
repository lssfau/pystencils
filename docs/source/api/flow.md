# Flowgraphs [Experimental]

## Primary APIs

```{eval-rst}

.. module:: pystencils.flow

.. autofunction:: block

.. autofunction:: cases

.. autofunction:: tie

.. autofunction:: subgraph

```

## Flowgraph Nodes

### Inheritance Diagram

:::{inheritance-diagram} pystencils.flow.flowgraph
:top-classes: pystencils.flow.flowgraph.FlowgraphAssignment pystencils.flow.flowgraph.FlowgraphNode
:parts: 1
:::

### Classes

```{eval-rst}
.. module:: pystencils.flow.flowgraph

.. autodata:: SymbolicMemoryLoc

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  Flowgraph
  FlowgraphAssignment
  Let
  Export
  Store
  Reduce
  FlowgraphNode
  Top
  Bottom
  EquationsBlock
  Subgraph
  Cases
```

## Builders

```{eval-rst}
.. module:: pystencils.flow.builders

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  EquationsBlockBuilder
  CasesBuilder
```

## Canonicalization

```{eval-rst}
.. module:: pystencils.flow.canonicalize_flowgraph

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  CanonicalizeFlowgraph
  CanonicalizationResult
  CanonicalizationError
```