from __future__ import annotations
from typing import TYPE_CHECKING

import sympy as sp
import numpy as np

if TYPE_CHECKING:
    from ..field import Field


def convolve(mask, field: Field) -> sp.Expr:
    """Convolve a scalar field with a mask."""
    mask = np.array(mask)
    expr = 0
    offset = tuple(s // 2 for s in mask.shape)

    fa = field.center()

    for index, factor in np.ndenumerate(mask):
        shift = tuple(i - o for i, o in zip(index, offset))
        expr += factor * fa.get_shifted(*shift)

    return expr
