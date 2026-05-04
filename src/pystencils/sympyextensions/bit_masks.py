import sympy as sp


# noinspection PyPep8Naming
class bit_conditional(sp.Function):
    """Evaluates a bit condition on an integer mask, and returns the value of one of two expressions,
    depending on whether the bit is set.

    Semantics:

    .. code-block:: none

        #   Three-argument version
        flag_cond(bitpos, mask, expr) = expr if (bitpos is set in mask) else 0

        #   Four-argument version
        flag_cond(bitpos, mask, expr_then, expr_else) = expr_then if (bitpos is set in mask) else expr_else

    The ``bitpos`` and ``mask`` arguments must both be of the same integer type.
    When in doubt, fix the type using `tcast`.
    """

    nargs = (3, 4)
