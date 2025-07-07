import numpy as np
import sympy as sp

from warnings import warn

import pystencils as ps
from pystencils.jupyter import make_imshow_animation, display_animation, set_display_mode
import pystencils.plot as plt

warn(
    "Importing `pystencils.session` is deprecated and the module will be removed in pystencils 2.1. "
    "Use `import pystencils as ps` instead.",
    FutureWarning
)

__all__ = ['sp', 'np', 'ps', 'plt', 'make_imshow_animation', 'display_animation', 'set_display_mode']
