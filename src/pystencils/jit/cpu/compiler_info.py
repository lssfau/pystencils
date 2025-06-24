from __future__ import annotations
from typing import Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

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
    def get_default(**kwargs) -> CompilerInfo:
        """Create a default compiler info object for the current runtime environment.
        
        Args:
            kwargs: Are forwarded to the constructor of the selected `CompilerInfo` subclass.
        """
        import platform

        sysname = platform.system()
        match sysname.lower():
            case "linux":
                #   Use GCC on Linux
                return GccInfo(**kwargs)
            case "darwin":
                return AppleClangInfo(**kwargs)
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
    
    def cxxflags(self):
        flags = super().cxxflags()
        if self.optlevel == "fast":
            #   clang deprecates -Ofast
            flags.remove("-Ofast")
            flags += ["-O3", "-ffast-math"]
        return flags


@dataclass
class AppleClangInfo(ClangInfo):
    """Compiler info for the Apple Clang compiler."""

    def cxxflags(self) -> list[str]:
        flags = super().cxxflags()

        if self.openmp:
            #   AppleClang requires the `-Xclang -fopenmp` in exactly that order for OpenMP to work
            flags.remove("-fopenmp")
            flags += ["-Xclang", "-fopenmp"]

        return flags
    
    def linker_flags(self):
        ldflags = super().linker_flags()
        
        #   Link against libpython
        import sysconfig
        libpython_file = Path(sysconfig.get_config_var("LIBRARY")).with_suffix(".dylib")
        libpython_dir = Path(sysconfig.get_config_var("LIBDIR"))
        libpython = libpython_dir / libpython_file

        ldflags += [str(libpython)]

        #   Find an appropriate OpenMP dylib to link against
        omp_candidates = [
            Path("/opt/local/lib/libomp/libomp.dylib"),
            Path("/usr/local/lib/libomp.dylib"),
            Path("/opt/homebrew/lib/libomp.dylib"),
        ]
        for omplib in omp_candidates:
            if omplib.exists():
                ldflags.append(str(omplib))
                break

        return ldflags
