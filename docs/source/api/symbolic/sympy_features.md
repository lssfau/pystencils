# Supported Subset of SymPy

This page lists the parts of [SymPy](https://sympy.org)'s symbolic algebra toolbox
that pystencils is able to parse and translate into code.
This includes a list of restrictions and caveats.

## sympy.core

:::{list-table}

*   - Symbols
    - {any}`sp.Symbol <sympy.core.symbol.Symbol>`
    - Represent untyped variables
*   - Integers
    - {any}`sp.Integer <sympy.core.numbers.Integer>`
    - 
*   - Rational Numbers
    - {any}`sp.Rational <sympy.core.numbers.Rational>`
    - Non-integer rationals are interpreted using the division operation
      of their context's data type.
*   - Arbitrary-Precision Floating Point
    - {any}`sp.Float <sympy.core.numbers.Float>`
    - Will initially be narrowed to double-precision (aka. the Python {any}`float` type),
*   - Transcendentals: $\pi$ and $e$
    - {any}`sp.pi <sympy.core.numbers.Pi>`, {any}`sp.E <sympy.core.numbers.Exp1>`
    - Only valid in floating-point contexts.
      Will be rounded to the nearest number representable in the
      respective data type (e.g. `pi = 3.1415927` for `float32`).
*   - Infinities ($\pm \infty$)
    - {any}`sp.Infinity <sympy.core.numbers.Infinity>`,
      {any}`sp.NegativeInfinity <sympy.core.numbers.NegativeInfinity>`
    - Only valid in floating point contexts.
*   - Arithmetic
    - {any}`sp.Add <sympy.core.add.Add>`,
      {any}`sp.Mul <sympy.core.mul.Mul>`,
      {any}`sp.Pow <sympy.core.power.Pow>`,
    - Integer powers up to $8$ will be expanded by pairwise multiplication.
      Negative integer powers will be replaced by divisions.
      Square root powers with a numerator $\le 8$ will be replaced
      by (products of) the `sqrt` function.
*   - Relations (`==`, `<=`, `>`, ...)
    - {any}`sp.Relational <sympy.core.relational.Relational>`
    - Result has type `boolean` and can only be used in boolean contexts.
*   - (Nested) Tuples
    - {any}`sp.Tuple <sympy.core.containers.Tuple>`
    - Tuples of expressions are interpreted as array literals.
      Tuples that contain further nested tuples must have a uniform, cuboid structure,
      i.e. represent a proper n-dimensional array,
      to be parsed as multidimensional array literals; otherwise an error is raised.
:::

## sympy.functions

:::{list-table}
*   - [Trigonometry](https://docs.sympy.org/latest/modules/functions/elementary.html#trigonometric)
    - {any}`sp.sin <sympy.functions.elementary.trigonometric.sin>`,
      {any}`sp.asin <sympy.functions.elementary.trigonometric.asin>`,
      ...
    - Only valid in floating-point contexts
*   - Hyperbolic Functions
    - {any}`sp.sinh <sympy.functions.elementary.hyperbolic.sinh>`,
      {any}`sp.cosh <sympy.functions.elementary.hyperbolic.cosh>`,
    - Only valid in floating-point contexts
*   - Exponentials
    - {any}`sp.exp <sympy.functions.elementary.exponential.exp>`,
      {any}`sp.log <sympy.functions.elementary.exponential.log>`,
    - Only valid in floating-point contexts
*   - Absolute
    - {any}`sp.Abs <sympy.functions.elementary.complexes.Abs>`
    -
*   - Rounding
    - {any}`sp.floor <sympy.functions.elementary.integers.floor>`,
      {any}`sp.ceiling <sympy.functions.elementary.integers.ceiling>`
    - Result will have the same data type as the arguments, so in order to
      get an integer, a type cast is additionally required (see {any}`tcast <pystencils.sympyextensions.tcast>`)
*   - Min/Max
    - {any}`sp.Min <sympy.functions.elementary.miscellaneous.Min>`,
      {any}`sp.Max <sympy.functions.elementary.miscellaneous.Max>`
    - 
*   - Piecewise Functions
    - {any}`sp.Piecewise <sympy.functions.elementary.piecewise.Piecewise>`
    - Cases of the piecewise function must be exhaustive; i.e. end with a default case.
:::

## sympy.logic

:::{list-table}
*   - Boolean atoms
    - {any}`sp.true <sympy.logic.boolalg.BooleanTrue>`,
      {any}`sp.false <sympy.logic.boolalg.BooleanFalse>`
    -
*   - Basic boolean connectives
    - {any}`sp.And <sympy.logic.boolalg.And>`,
      {any}`sp.Or <sympy.logic.boolalg.Or>`,
      {any}`sp.Not <sympy.logic.boolalg.Not>`
    - 
:::

## sympy.tensor

:::{list-table}
*   - Indexed Objects
    - {any}`sp.Indexed <sympy.tensor.indexed.Indexed>`
    - Base of the indexed object must have a {any}`PsArrayType` of the correct dimensionality.
      Currently, only symbols ({any}`sp.Symbol <sympy.core.symbol.Symbol>` or {any}`TypedSymbol`)
      can be used as the base of an `Indexed`.
:::
