from pathlib import Path

from ...codegen.config import _AUTO_TYPE, AUTO
from ..cpu.compiler_info import CompilerInfo
from ..cpu.cpujit import CpuJit, ExtensionModuleBuilderBase
from .sycl_compiler_info import SYCLIcpxInfo
from .sycl_module_builder import SyclExtensionModuleBuilder


class SYCLJit(CpuJit):

    def __init__(
        self,
        compiler_info: CompilerInfo | None = None,
        objcache: str | Path | _AUTO_TYPE | None = AUTO,
        *,
        module_builder: ExtensionModuleBuilderBase | None = None,
        emit_warnings: bool = False
    ):

        if compiler_info is None:
            compiler_info = SYCLIcpxInfo(optlevel="3")

        if module_builder is None:
            module_builder = SyclExtensionModuleBuilder(compiler_info)
        super().__init__(compiler_info, objcache, module_builder=module_builder, emit_warnings=emit_warnings)
