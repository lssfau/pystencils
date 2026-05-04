from __future__ import annotations

from types import ModuleType
from typing import cast, Sequence
from pathlib import Path
from dataclasses import dataclass, field
from textwrap import indent

from pystencils.jit.jit import KernelWrapper

from ...types import (
    PsPointerType,
    PsType,
    deconstify,
)
from ...field import Field, FieldType
from ...sympyextensions import DynamicType
from ...codegen import Kernel, Parameter
from ...codegen.properties import FieldBasePtr, FieldShape, FieldStride

from .compiler_info import CompilerInfo
from .cpujit import ExtensionModuleBuilderBase
from ..error import JitError

import numpy as np


_module_template = Path(__file__).parent / "cpu_kernel_module.tmpl.cpp"


class DefaultExtensionModuleBuilder(ExtensionModuleBuilderBase):

    @dataclass
    class ParamExtraction:
        argstruct_members: list[str] = field(default_factory=list)

        kernel_kwarg_refs: list[str] = field(default_factory=list)
        array_proxy_defs: list[str] = field(default_factory=list)

        extract_kernel_args: list[str] = field(default_factory=list)
        precondition_checks: list[str] = field(default_factory=list)
        kernel_invocation_args: list[str] = field(default_factory=list)

        def substitutions(self) -> dict[str, str]:
            t = "    "
            tt = 2 * t

            return {
                "argstruct_members": indent(
                    "\n".join(self.argstruct_members), prefix=t
                ),
                "kernel_kwarg_refs": indent(
                    "\n".join(self.kernel_kwarg_refs), prefix=tt
                ),
                "array_proxy_defs": indent("\n".join(self.array_proxy_defs), prefix=tt),
                "extract_kernel_args": indent(
                    "\n".join(self.extract_kernel_args), prefix=tt
                ),
                "precondition_checks": indent(
                    "\n".join(self.precondition_checks), prefix=tt
                ),
                "kernel_invocation_args": ", ".join(self.kernel_invocation_args),
            }

        def add_array_for_field(self, ptr_param: Parameter):
            field: Field = ptr_param.fields[0]

            ptr_type = ptr_param.dtype
            assert isinstance(ptr_type, PsPointerType)

            if isinstance(field.dtype, DynamicType):
                elem_type = ptr_type.base_type
            else:
                elem_type = field.dtype

            parg_name = self.add_kwarg(field.name)
            self._init_array_proxy(field.name, elem_type, len(field.shape), parg_name)

        def add_pointer_param(self, ptr_param: Parameter):
            ptr_type = ptr_param.dtype
            assert isinstance(ptr_type, PsPointerType)
            elem_type = deconstify(ptr_type.base_type)

            parg_name = self.add_kwarg(ptr_param.name)
            proxy_name = self._init_array_proxy(ptr_param.name, elem_type, 1, parg_name)
            self._add_kernel_argument(
                ptr_param, f"{proxy_name}.data< {elem_type.c_string()} >()"
            )

        def _array_proxy_name(self, name: str) -> str:
            return f"array_proxy_{name}"

        def _init_array_proxy(
            self,
            name: str,
            dtype: PsType,
            ndim: int,
            pyobj: str,
            itemsize: int | None = None,
        ) -> str:
            proxy_name = self._array_proxy_name(name)
            elem_type = deconstify(dtype)
            typeno = self._typeno(elem_type)

            if itemsize is None:
                itemsize = dtype.itemsize

            if itemsize is None:
                raise JitError(
                    f"Cannot set up array proxy for data type with unknown size: {dtype}"
                )

            proxy_ctor_args = [f'"{name}"', pyobj, str(ndim), typeno, str(itemsize)]

            self.array_proxy_defs.append(
                f"ArrayProxy {proxy_name} = ArrayProxy::fromPyObject( {', '.join(proxy_ctor_args)} ) ;"
            )
            return proxy_name

        def add_kwarg(self, name: str) -> str:
            kwarg_name = f"_ref_{name}"
            self.kernel_kwarg_refs.append(
                f'PyObject * {kwarg_name} = getKwarg(kwargs, "{name}");'
            )
            return kwarg_name

        def _add_kernel_argument(self, param: Parameter, extraction: str):
            self.argstruct_members.append(
                f"{deconstify(param.dtype).c_string()} {param.name};"
            )
            self.kernel_invocation_args.append(f"kernel_args.{param.name}")
            self.extract_kernel_args.append(f"{param.name} = {extraction};")

        def add_field_base_pointer(self, param: Parameter, ptr_prop: FieldBasePtr):
            field_name = ptr_prop.field.name
            proxy_name = self._array_proxy_name(field_name)

            ptr_type = param.dtype
            assert isinstance(ptr_type, PsPointerType)
            elem_type = deconstify(ptr_type.base_type)

            self._add_kernel_argument(
                param, f"{proxy_name}.data< {elem_type.c_string()} >()"
            )

        def add_scalar_param(self, param: Parameter):
            parg_name = self.add_kwarg(param.name)
            stype = deconstify(param.dtype)
            typeno = self._typeno(stype)

            self._add_kernel_argument(
                param,
                f'scalarFromPyObject< {stype.c_string()}, {typeno} > ({parg_name}, "{param.name}")',
            )

        def add_shape_param(self, param: Parameter, shape_prop: FieldShape):
            field_name = shape_prop.field.name
            proxy_name = self._array_proxy_name(field_name)
            stype = deconstify(param.dtype)

            self._add_kernel_argument(
                param,
                f"{proxy_name}.shape< {stype.c_string()} >({shape_prop.coordinate})",
            )

        def add_stride_param(self, param: Parameter, stride_prop: FieldStride):
            field_name = stride_prop.field.name
            proxy_name = self._array_proxy_name(field_name)
            stype = deconstify(param.dtype)

            self._add_kernel_argument(
                param,
                f"{proxy_name}.stride< {stype.c_string()} > ({stride_prop.coordinate})",
            )

        def check_same_shape(self, fields: Sequence[Field]):
            if len(fields) > 1:
                check_args = ", ".join("&" + self._array_proxy_name(f.name) for f in fields)
                rank = fields[0].spatial_dimensions
                self.precondition_checks.append(
                    f"checkSameShape({{ {check_args} }}, {rank});"
                )

        def check_fixed_shape_and_strides(self, field: Field):
            proxy_name = self._array_proxy_name(field.name)
            expect_shape = (
                "("
                + ", ".join(
                    (str(s) if isinstance(s, int) else "*") for s in field.shape
                )
                + ")"
            )
            field_shape = field.spatial_shape
            #   Scalar fields may omit their trivial index dimension
            if field.index_shape not in ((), (1,)):
                field_shape += field.index_shape
                scalar_field = False
            else:
                scalar_field = True

            for coord, size in enumerate(field_shape):
                if isinstance(size, int):
                    self.precondition_checks.append(
                        f'checkFieldShape("{expect_shape}", {proxy_name}, {coord}, {size});'
                    )

            if scalar_field:
                self.precondition_checks.append(
                    f'checkTrivialIndexShape("{expect_shape}", {proxy_name}, {len(field_shape)});'
                )

            expect_strides = (
                "("
                + ", ".join(
                    (str(s) if isinstance(s, int) else "*") for s in field.strides
                )
                + ")"
            )
            for coord, stride in enumerate(field.strides[: len(field_shape)]):
                if isinstance(stride, int):
                    self.precondition_checks.append(
                        f'checkFieldStride("{expect_strides}", {proxy_name}, {coord}, {stride});'
                    )

        @staticmethod
        def _typeno(dtype: PsType):
            if dtype.numpy_dtype is None:
                raise JitError(f"Cannot get typeno for non-numpy type {dtype}")
            npname = dtype.numpy_dtype.name.upper()
            if npname.startswith("VOID"):
                npname = "VOID"  # for struct types
            return f"NPY_{npname}"

    @staticmethod
    def include_dirs() -> list[str]:
        return [np.get_include()]

    def __init__(self, compiler_info: CompilerInfo):
        self._compiler_info = compiler_info

    def render_module(self, kernel: Kernel, module_name: str) -> str:
        extr = self._handle_params(kernel)
        kernel_def = self._get_kernel_definition(kernel)
        includes = [f"#include {h}" for h in sorted(kernel.required_headers)]

        from string import Template

        templ = Template(_module_template.read_text())
        code_str = templ.substitute(
            includes="\n".join(includes),
            restrict_qualifier=self._compiler_info.restrict_qualifier(),
            module_name=module_name,
            kernel_name=kernel.name,
            kernel_definition=kernel_def,
            **extr.substitutions(),
        )
        return code_str

    def get_wrapper(
        self, kernel: Kernel, extension_module: ModuleType
    ) -> KernelWrapper:
        return DefaultCpuKernelWrapper(kernel, extension_module)

    def _get_kernel_definition(self, kernel: Kernel) -> str:
        from ...backend.emission import CAstPrinter

        printer = CAstPrinter()

        return printer(kernel)

    def _handle_params(self, kernel: Kernel) -> ParamExtraction:
        parameters = kernel.parameters
        extr = self.ParamExtraction()

        for param in parameters:
            if param.get_properties(FieldBasePtr):
                extr.add_array_for_field(param)

        for param in parameters:
            if ptr_props := param.get_properties(FieldBasePtr):
                extr.add_field_base_pointer(param, cast(FieldBasePtr, ptr_props.pop()))
            elif shape_props := param.get_properties(FieldShape):
                extr.add_shape_param(param, cast(FieldShape, shape_props.pop()))
            elif stride_props := param.get_properties(FieldStride):
                extr.add_stride_param(param, cast(FieldStride, stride_props.pop()))
            elif isinstance(param.dtype, PsPointerType):
                extr.add_pointer_param(param)
            else:
                extr.add_scalar_param(param)

        fields = kernel.get_fields()
        for f in fields:
            extr.check_fixed_shape_and_strides(f)

        extr.check_same_shape([f for f in fields if f.field_type == FieldType.GENERIC])

        return extr


class DefaultCpuKernelWrapper(KernelWrapper):
    def __init__(self, kernel: Kernel, jit_module: ModuleType):
        super().__init__(kernel)
        self._module = jit_module
        self._invoke = getattr(jit_module, "invoke")

    def __call__(self, **kwargs) -> None:
        return self._invoke(**kwargs)
