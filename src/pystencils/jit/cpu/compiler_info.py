from __future__ import annotations
from typing import Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ...codegen.target import Target


@dataclass
class CompilerInfo(ABC):
    """Base class for compiler infos."""

    openmp: bool = True
    """Enable/disable OpenMP compilation"""

    optlevel: str | None = "fast"
    """Compiler optimization level"""

    cxx_standard: str = "c++14"
    """C++ language standard to be compiled with"""

    target: Target = Target.CurrentCPU
    """Hardware target to compile for.
    
    The value of ``target`` is used to set the ``-march`` compiler
    option (or equivalent).
    `Target.CurrentCPU` translates to ``-march=native``.
    """

    @abstractmethod
    def cxx(self) -> str:
        """Path to the executable of this compiler"""

    @abstractmethod
    def cxxflags(self) -> list[str]:
        """Compiler flags affecting C++ compilation"""

    @abstractmethod
    def linker_flags(self) -> list[str]:
        """Flags affecting linkage of the extension module"""

    @abstractmethod
    def include_flags(self, include_dirs: Sequence[str]) -> list[str]:
        """Convert a list of include directories into corresponding compiler flags"""

    @abstractmethod
    def restrict_qualifier(self) -> str:
        """*restrict* memory qualifier recognized by this compiler"""

    @staticmethod
    def get_default() -> CompilerInfo:
        import platform

        sysname = platform.system()
        match sysname.lower():
            case "linux":
                #   Use GCC on Linux
                return GccInfo()
            case "darwin":
                return AppleClangInfo()
            case _:
                raise RuntimeError(
                    f"Cannot determine compiler information for platform {sysname}"
                )


class _GnuLikeCliCompiler(CompilerInfo):
    def cxxflags(self) -> list[str]:
        flags = ["-DNDEBUG", f"-std={self.cxx_standard}", "-fPIC"]

        if self.optlevel is not None:
            flags.append(f"-O{self.optlevel}")

        if self.openmp:
            flags.append("-fopenmp")

        match self.target:
            case Target.CurrentCPU:
                flags.append("-march=native")
            case Target.X86_SSE:
                flags += ["-march=x86-64-v2"]
            case Target.X86_AVX:
                flags += ["-march=x86-64-v3"]
            case Target.X86_AVX512:
                flags += ["-march=x86-64-v4"]
            case Target.X86_AVX512_FP16:
                flags += ["-march=x86-64-v4", "-mavx512fp16"]

        return flags

    def linker_flags(self) -> list[str]:
        return ["-shared"]

    def include_flags(self, include_dirs: Sequence[str]) -> list[str]:
        return [f"-I{d}" for d in include_dirs]

    def restrict_qualifier(self) -> str:
        return "__restrict__"


class GccInfo(_GnuLikeCliCompiler):
    """Compiler info for the GNU Compiler Collection C++ compiler (``g++``)."""

    def cxx(self) -> str:
        return "g++"


@dataclass
class ClangInfo(_GnuLikeCliCompiler):
    """Compiler info for the LLVM C++ compiler (``clang``)."""

    def cxx(self) -> str:
        return "clang++"


@dataclass
class AppleClangInfo(ClangInfo):
    """Compiler info for the Apple Clang compiler."""

    def cxxflags(self) -> list[str]:
        return super().cxxflags() + ["-Xclang"]
