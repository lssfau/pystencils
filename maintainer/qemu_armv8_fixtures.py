import pytest

from pystencils import Target
from pystencils.jit.cpu import ClangInfo


@pytest.fixture
def compiler_info(target: Target):
    #   GCC 13 segfaults in various cases with SVE intrinsics -> use clang instead
    return ClangInfo(target=target)
