from .codegen import Target as _Target

from warnings import warn

warn(
    "Importing anything from `pystencils.enums` is deprecated and the module will be removed in pystencils 2.1. "
    "Import from `pystencils` instead.",
    FutureWarning,
    stacklevel=2,
)

Target = _Target
