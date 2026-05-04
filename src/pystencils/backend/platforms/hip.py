from __future__ import annotations

from .generic_gpu import GenericGpu


class HipPlatform(GenericGpu):
    """Platform for the HIP GPU target."""

    @property
    def required_headers(self) -> set[str]:
        return super().required_headers | {'"pystencils_runtime/hip.h"'}
