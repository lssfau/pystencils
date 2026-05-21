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

(guide_eqgraphs)=
# Kernel Flowgraphs with `ps.flow` [Experimental]

:::{caution}

`ps.flow` is under active development and considered *experimental*.
Its APIs may be unstable and can still change without warning.
Use at your own discretion.

:::

This page introduces `pystencils.flow`, the novel kernel language of pystencils.
It is built around the idea of declaratively expressing kernels as *flowgraphs*.
A flowgraph comprises one or more *blocks* of equations,
which are connected via edges to form a directed acyclic graph.
The equations in each block define variables (*symbols* in pystencils jargon);
the values of symbols stream from one block to another along the graph's edges.
At the top, parameters enter the flowgraph while the results of the computation
leave the graph at the bottom.
Equations may read values from fields,
and the graph may update field values as well as perform reductions.
These side effects also count as "results" of the graph and must therefore occur only at the bottom.

In the following, we will present how flowgraphs are constructed, from a single block of equations
to complex structures including case distinctions and subgraphs.

## Equation Blocks

The equation block, or just *block*, is the basic component of each flowgraph.
All equations that make up a kernel's computations are defined inside and grouped into blocks.

Blocks can be defined using the {any}`ps.flow.block <pystencils.flow.block>` decorator.
Equations are then written using a special syntax:

```{code-cell} ipython3
:tags: [hide-output]

x, y, z, v, w = sp.symbols("x, y, z, v, w")
f, g = ps.fields("f(1), g(2): [2D]")

@ps.flow.block
def example(_eq):
    #   Define subexpressions
    _eq.let[x] = z + 1
    _eq.export[y] = x / 2 * v

    #   Store a value to a field
    _eq.store[f(0)] = y / z

    #   Perform a reduction onto a target symbol
    q = ps.TypedSymbol("q", "float64")
    _eq.reduce[q, "max"] = 2 * y

example
```

```{code-cell} ipython3
:tags: [remove-cell]

from IPython.display import Code
from myst_nb import glue

ker = ps.create_kernel(example)
glue("code-example1", Code(ker.get_c_code()))
```

:::{dropdown} Generated Code
```{glue:} code-example1
```
:::

The function decorated by `ps.flow.block` must take a single parameter.
The name of that parameter is arbitrary; for convention's sake, we are going to call it `_eq`.
The object `_eq` is a {any}`builder <EquationsBlockBuilder>` which we use to assemble the block's equations.
There are four types of equations:

 - {any}`let <EquationsBlockBuilder.let>` defines *private subexpressions*.
   The left-hand side symbol (passed in `[]`) is associated with the given right-hand side term
   and refers to that term in the current block.
 - {any}`export <EquationsBlockBuilder.export>` creates *exported subexpressions*;
   the left-hand symbol of a public subexpression is published to successor blocks in the flowgraph.
   Exports allow data to move from one block to another in the graph.
 - {any}`store <EquationsBlockBuilder.store>` equations assign a value to a memory location, typically
   an entry of a field.
   This constitutes a side effect of the kernel.
 - {any}`reduce <EquationsBlockBuilder.reduce>` equations perform a global reduction onto a target symbol;
   the values assigned to the left-hand side symbol in each kernel iteration will be reduced by the given
   commutative reduction operation.
   Reductions, like stores, constitute side effects.

The order in which subexpressions are defined does not matter; `ps.flow` will
reorder equations such that a symbol's definition always comes before its first use.
If dependency cycles are detected (e.g. we have cyclically dependent equations `x = f(y)` and `y = g(x)`),
the block is invalid and an error is raised.
Similarily, errors will be thrown if any left-hand side is assigned more than once.

## Building Flowgraphs

Multiple blocks can be combined into a [directed acyclic graph (DAG)](https://en.wikipedia.org/wiki/Directed_acyclic_graph).
This can be a useful tool to build large and complex kernels out of distinct, self-contained components.

### Connecting Blocks

Blocks are connected by adding existing blocks as *predecessors* when creating new blocks,
via the `preds` keyword argument to the `ps.flow.block` decorator:

```{code-cell} ipython3
:tags: [hide-output]

@ps.flow.block(preds=[example])
def store_g0(_eq):
    _eq.store[g(0)] = y
    _eq.export[z] = y + 2

store_g0
```

```{code-cell} ipython3
:tags: [remove-cell]

ker = ps.create_kernel(store_g0)
glue("code-example2", Code(ker.get_c_code()))
```

:::{dropdown} Generated Code
```{glue:} code-example2
```
:::

By adding the previously defined block `example` as a predecessor to `store_g0`,
equations inside `store_g0` get access to all variables [exported](#EquationsBlockBuilder.export) by `example`.

Alternatively to the `preds` argument, predecessors can also be added
using the `connect <EquationsBlockBuilder.connect>` method inside the block.
Both variants can also be combined:

```{code-cell} ipython3
:tags: [hide-output]

@ps.flow.block(preds=[example])
def store_g1(_eq):
    _eq.connect(store_g0)
    _eq.store[g(1)] = y + z

store_g1
```

```{code-cell} ipython3
:tags: [remove-cell]

ker = ps.create_kernel(store_g1)
glue("code-example3", Code(ker.get_c_code()))
```

:::{dropdown} Generated Code
```{glue:} code-example3
```
:::

:::{note}

Flowgraph nodes only store their *predecessors*; they don't know about their *successors*.
Also, they are *immutable*; once created, they can no longer be modified.

:::

### Rules for Symbol Definitions

There are a few rules governing symbol definitions and exports:

 - **Block-Local Subexpression Names:** Symbols assigned with `let` and `export` are block-local;
   the same symbol can be used for different subexpressions in different blocks without name conflicts.
 - **Name Hiding:** If a node defines a symbol with `let` or `export`, that definition hides any definition
   of the same symbol `export`ed from a predecessor.
 - **Unique Imports:** If a node has more than one predecessor, the sets of exported symbols of all predecessors must be *disjoint*.
   In other words: for each imported symbol `x`, there must be a *unique* predecessor exporting it.
 - **Only Direct Imports:** Symbols are only imported from direct predecessors; not from transitive ones.


### Tying Up Graphs

Once you have defined all your nodes, they must be *tied up* into a flowgraph:

```{code-cell} ipython3
:tags: [hide-output]

graph = ps.flow.tie(store_g1, name="my-graph")
graph
```

The function {any}`ps.flow.tie <pystencils.flow.tie>` creates a {any}`Flowgraph` object from one or more nodes and all their predecessors.
Once tied up, the flowgraph can no longer be extended.
`Flowgraph` objects can either be [compiled to a kernel](#section:psflow-generating-code)
or be used as a ([guarded](#section:psflow-cases) or unguarded) [subgraph](section:psflow-subgraphs).

:::{dropdown} Technical Details of `tie`

The `tie` operation combines one or more output nodes and all their (transitive) predecessors into a DAG,
and transforms that graph into *canonical form*.
First and foremost, it adds an explicit *bottom* node $\bot$, which denotes the foot of the DAG.
On the other end, it will add a special *top* node $\top$ to mark the head of the graph.
All nodes that contain {term}`free symbols <Free Symbol>` that are not imported from some of their predecessors will be connected
to $\top$, and will implicitly import these missing symbols from the "outside world" through $\top$.

:::

### Visualization with GraphViz

We can view a graphical representation of our graph using `ps.flow.to_dot`
(requires the [graphviz](https://pypi.org/project/graphviz/) package).
The visualization shows the graph's nodes (including $\top$ and $\bot$),
their connecting edges, and the flow of variables along these edges.

```{code-cell} ipython3
:tags: [hide-output]

ps.flow.to_dot(graph)
```

The object returned by `to_dot` is a {any}`graphviz.Digraph`;
you can view it visually in Jupyter, but also export to a text or image file using the API of `Digraph`.

(section:psflow-generating-code)=
### Generating Code

To generate a kernel implementation for a flowgraph, pass it to `ps.create_kernel`:

```{code-cell} ipython3
:tags: [hide-output]

ker = ps.create_kernel(graph)
ps.inspect(ker)
```

:::{note}
You don't have to `tie` your graph before passing it to `create_kernel`; it suffices to pass in its output node.
If your graph has more than one output node, however, `ps.flow.tie` must happen beforehand.
:::

(section:psflow-subgraphs)=
## Subgraphs

A closed `Flowgraph` can be embedded into a larger graph using the {any}`Subgraph` node.
To the outside, a `Subgraph` acts like an `EquationsBlock`, except that it contains an entire graph
instead of just a set of equations.
Like a block, a subgraph node may have predecessors and may be connected to successors.
The free symbols (i.e. parameters) of that graph become free symbols of the `Subgraph` node and must either be imported
from a predecessor node, or become parameters of the enclosing graph.
If the `Flowgraph` {term}`exports` symbols, these are exported through the `Subgraph` node to its successors.

Create a subgraph node from an existing flowgraph using {any}`ps.flow.subgraph <pystencils.flow.subgraph>`.
Predecessors are connected via the `preds` argument:

```{code-cell} ipython3
:tags: [remove-cell]

@ps.flow.block
def blockX(_eq):
    _eq.export[z] = 42

@ps.flow.block
def blockY(_eq):
    _eq.export[v] = -3

```

```{code-cell} ipython3
:tags: [hide-output]

# blockX and blockY are equation blocks
subgr = ps.flow.subgraph(graph, preds=[blockY, blockX])
subgr
```

(section:psflow-cases)=
## Conditionals and Case Distinctions

`ps.flow` supports conditional evaluation through *case distinctions*.
A case distinction is a special kind of node that has multiple branches, each with a boolean condition.
Similar to the embedding of [subgraphs](#section:psflow-subgraphs), every branch encapsulates a {any}`Flowgraph`
that will be evaluated if the branch condition matches.
Case distinctions can be used to guard parts of a kernel, or to implement conditional data flow.

We can create case distinctions using the {any}`ps.flow.cases <pystencils.flow.cases>` decorator.
Predecessors can be added to the case distinction using the `preds` argument.
The object `cs` passed to the decorated function is uses to declare one or more cases.
Each case has a condition (a boolean expression) and a subgraph.
You can either set a predefined `Flowgraph` for a case,
or use `cs.case` as a decorator to define the guarded subgraph in-line.

```{code-cell} ipython3
:tags: [remove-cell]

@ps.flow.block()
def blockZ(_eq):
    _eq.export[x] = 15

@ps.flow.block()
def blockW(_eq):
    _eq.export[w] = -4.2

some_graph = ps.flow.tie(blockZ, name="some-subgraph")
```

```{code-cell} ipython3
:tags: [hide-output]

t = sp.symbols("t")

@ps.flow.cases(preds=[blockX, blockY])
def my_case_distinction(cs):
    # Existing flowgraph as a case
    cs.case(t < 0, some_graph)

    # Inline-defined subgraph
    @cs.case(t > 0)
    def if_t_larger_zero(_eq):
        _eq.export[x] = v + z

    # Fallback case
    @cs.case(True, preds=[blockW])
    def otherwise(_eq):
        _eq.export[x] = w

my_case_distinction
```

The `cs.case` decorator works the same as the {any}`ps.flow.block <pystencils.flow.block>` decorator.
As shown above in the fallback case, you can add predecessors to a case using the `preds` keyword;
these predecessors - and all of their transitive predecessors - will be guarded by the case's condition.
Take care which nodes you add as predecessors to the case distinction as a whole, and which you add to individual cases,
depending on which equations you want to put below guards.

Here's a few more rules concerning case distinctions:
 - **Case Selection at Runtime:** The cases' conditions are evaluated in order.
   At kernel runtime, the first case whose condition matches will be evaluated.
 - **Exporting from Cases:** All cases must export the exact same symbols; these are re-exported by the case distinction.
 - **Completeness:** If the case distinction exports at least one symbol, its cases must be *complete*;
   i.e. one of them must always evaluate to `True`.
   The easiest way to accomplish this is using a final fallback case with `True` as label.
   If there are no exported symbols, the case distinction may remain incomplete.

## Glossary

::::{dropdown} Glossary
:::{glossary}

DAG
  Directed acyclic graph; a graph with directed edges that contains no cycles.

Root
  In a DAG, a node is a *root* if it has no predecessors.
  A `ps.flow` flowgraph has only a single root, which is $\top$.

Sink
  In a DAG, a node is a *sink* if it has no successors.
  A `ps.flow` flowgraph has only a single sink, which is $\bot$.

Top
  The Top ($\top$) node of a flowgraph marks the beginning of its data flow;
  it is where data enters the graph.

Output Node
  In a `ps.flow` flowgraph, an output node is a node connected to $\bot$.
  Its `export`ed symbols are exported outside of the graph when it is used as a {term}`subgraph`.

Free Symbol
  The free symbols of a flowgraph node are those symbols used by expressions inside the node
  that are not also defined (using `let` or `export`) inside that node.

  The free symbols of a flowgraph as a whole are symbols that are *free* in at least one of its nodes
  without being imported from a predecessor node.
  These symbols are also called the graph's {term}`parameters <Parameter>`.

Exports
  The exported symbols of an {any}`EquationsBlock` are those defined using `export` equations.
  The exported symbols of a flowgraph are the symbols exported by predecessors of its $\bot$ node
  (i.e. its {term}`output nodes <Output Node>`.)

Parameter
  See {term}`Free Symbol`

Canonical Form
  A flowgraph is in canonical form if
   - it has a unique $\top$ node as its only {term}`root`, and a unique $\bot$ node as its only {term}`sink`;
   - at each node, every free symbol of that node must be imported from exactly one of the node's predecessors
     (symbols not provided by any other block must be imported from $\top$)
   - Each memory location and reduction target may be written to *at most once*
  
  These conditions are enforced by {any}`pystencils.flow.tie`.
  See also {any}`CanonicalizeFlowgraph`.

Subgraph
  A flowgraph embedded into a larger, enclosing graph via the {any}`Subgraph <pystencils.flow.flowgraph.Subgraph>` node.

:::
::::