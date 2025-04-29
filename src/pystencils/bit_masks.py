from .sympyextensions.bit_masks import bit_conditional
from warnings import warn


class flag_cond(bit_conditional):
    def __new__(cls, *args, **kwargs):
        warn(
            "flag_cond is deprecated and will be removed in pystencils 2.1. "
            "Use `pystencils.sympyextensions.bit_conditional` instead.",
            FutureWarning
        )
        return bit_conditional.__new__(cls, *args, **kwargs)
