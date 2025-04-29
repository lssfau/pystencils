from __future__ import annotations

from .generic_gpu import GenericGpu


class CudaPlatform(GenericGpu):
    """Platform for the CUDA GPU taret."""

    @property
    def required_headers(self) -> set[str]:
        return super().required_headers | {'"pystencils_runtime/cuda.cuh"'}
