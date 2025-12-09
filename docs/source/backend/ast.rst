********************
Abstract Syntax Tree
********************

.. _ast-canonical-form:

Canonical Form
==============

Many transformations in pystencils' backend require that their input AST is in *canonical form*.
This means that:

- Each symbol, constant, and expression node is annotated with a data type;
- Each symbol has at most one declaration;
- Each symbol that is never written to apart from its declaration has a ``const`` type; and
- Each symbol whose type is *not* ``const`` has at least one non-declaring assignment.

The first requirement can be ensured by running the `Typifier` on each newly constructed subtree.
The other three requirements are ensured by the `CanonicalizeSymbols` pass,
which should be run first before applying any optimizing transformations.
All transformations in this module retain canonicality of the AST.

Canonicality allows transformations to forego various checks that would otherwise be necessary
to prove their legality.


.. automodule:: pystencils.backend.ast

API Documentation
=================

Inheritance Diagram
-------------------

.. inheritance-diagram:: pystencils.backend.ast.astnode.PsAstNode pystencils.backend.ast.structural pystencils.backend.ast.axes pystencils.backend.ast.expressions pystencils.backend.ast.vector pystencils.backend.extensions.foreign_ast
    :top-classes: pystencils.types.PsAstNode
    :parts: 1

Base Classes
------------

.. module:: pystencils.backend.ast.astnode

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    PsAstNode
    PsAstNodeChildrenMixin
    PsLeafMixIn


Structural Nodes
----------------

.. module:: pystencils.backend.ast.structural

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    PsBlock
    PsStatement
    PsAssignment
    PsDeclaration
    PsLoop
    PsConditional
    PsEmptyLeafMixIn
    PsPragma
    PsComment

.. _ast-iteration-axes:

Iteration Axes System
---------------------

.. module:: pystencils.backend.ast.axes

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    PsAxisRange
    PsAxesCube
    PsIterationAxis
    PsLoopAxis
    PsParallelLoopAxis
    PsSimdAxis


Expressions
-----------

.. module:: pystencils.backend.ast.expressions

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    PsExpression
    PsLvalue
    PsSymbolExpr
    PsConstantExpr
    PsLiteralExpr
    PsBufferAcc
    PsSubscript
    PsMemAcc
    PsLookup
    PsCall
    PsTernary
    PsNumericOpTrait
    PsIntOpTrait
    PsBoolOpTrait
    PsUnOp
    PsNeg
    PsAddressOf
    PsCast
    PsBinOp
    PsAdd
    PsSub
    PsMul
    PsDiv
    PsIntDiv
    PsRem
    PsLeftShift
    PsRightShift
    PsBitwiseAnd
    PsBitwiseXor
    PsBitwiseOr
    PsAnd
    PsOr
    PsNot
    PsRel
    PsEq
    PsNe
    PsGe
    PsLe
    PsGt
    PsLt
    PsArrayInitList


SIMD Nodes
----------

.. module:: pystencils.backend.ast.vector

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    PsVectorOp
    PsVecBroadcast
    PsVecMemAcc


Utility
-------

.. currentmodule:: pystencils.backend.ast

.. autosummary::
    :toctree: generated
    :nosignatures:

    expressions.evaluate_expression
    dfs_preorder
    dfs_postorder
    util.determine_memory_object
