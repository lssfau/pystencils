from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from textwrap import indent
from types import ModuleType
from typing import Sequence, cast

import numpy as np
import sympy as sp

from pystencils.jit.jit import KernelWrapper

from ...codegen import GpuKernel, Kernel, Parameter
from ...codegen.gpu_indexing import GpuLaunchConfiguration, HardwareProperties
from ...codegen.properties import (
    FieldBasePtr,
    FieldShape,
    FieldStride,
    SYCLItem,
    SYCLNDItem,
)
from ...field import Field, FieldType
from ...grids import IField
from ...sympyextensions import DynamicType
from ...types import PsPointerType, PsType, deconstify
from ..cpu.compiler_info import CompilerInfo
from ..cpu.cpujit import ExtensionModuleBuilderBase
from ..error import JitError

try:
    import dpctl
    import dpctl.tensor as dpt

    HAVE_DPCTL = True
except ImportError:
    HAVE_DPCTL = False


_module_template = Path(__file__).parent / "sycl_kernel_module.tmpl.cpp"


class SyclExtensionModuleBuilder(ExtensionModuleBuilderBase):
    @dataclass
    class ParamExtraction:
        argstruct_members: list[str] = field(default_factory=list)

        kernel_kwarg_refs: list[str] = field(default_factory=list)
        array_proxy_defs: list[str] = field(default_factory=list)
        queue_checks: list[str] = field(default_factory=list)

        extract_kernel_args: list[str] = field(default_factory=list)
        precondition_checks: list[str] = field(default_factory=list)
        kernel_lambda_caption: list[str] = field(default_factory=list)
        kernel_invocation_args: list[str] = field(default_factory=list)
        launch_config_args: list[str] = field(default_factory=list)
        extra_substitutions: dict[str, str] = field(default_factory=dict)

        def substitutions(self) -> dict[str, str]:
            t = "    "
            tt = 2 * t
            ttt = 3 * t

            return {
                "argstruct_members": indent(
                    "\n".join(self.argstruct_members), prefix=t
                ),
                "kernel_kwarg_refs": indent(
                    "\n".join(self.kernel_kwarg_refs), prefix=tt
                ),
                "array_proxy_defs": indent("\n".join(self.array_proxy_defs), prefix=tt),
                "queue_checks": indent("\n".join(self.queue_checks), prefix=tt),
                "extract_kernel_args": indent(
                    "\n".join(self.extract_kernel_args), prefix=tt
                ),
                "precondition_checks": indent(
                    "\n".join(self.precondition_checks), prefix=tt
                ),
                "kernel_invocation_args": ", ".join(self.kernel_invocation_args),
                "kernel_lambda_caption": ", ".join(self.kernel_lambda_caption),
                "launch_config_args": indent(
                    "\n".join(self.launch_config_args), prefix=ttt
                ),
            }

        def add_array_for_field(self, ptr_param: Parameter):
            field: Field | IField = ptr_param.fields[0]

            ptr_type = ptr_param.dtype
            assert isinstance(ptr_type, PsPointerType)

            if isinstance(field.dtype, DynamicType):
                elem_type = ptr_type.base_type
            else:
                elem_type = field.dtype

            parg_name = self.add_kwarg(field.name)
            rank = (
                len(field.shape)
                if isinstance(field, Field)
                else field.get_buffer_spec().rank
            )
            self._init_array_proxy(field.name, elem_type, rank, parg_name)

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
            self.queue_checks.append(f"checkQueue( q, {proxy_name}.get_queue() );")
            if "array_queue" not in self.extra_substitutions:
                self.extra_substitutions["array_queue"] = f"{proxy_name}.get_queue()"
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
            self.kernel_lambda_caption.append(
                f"{param.name} = kernel_args.{param.name}"
            )
            self.kernel_invocation_args.append(f"{param.name}")
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

        def add_sycl_range_args(self, field: Field | IField, ghost_layers):
            match field:
                case Field():
                    rank = field.spatial_dimensions
                    spatial_bounds = field.spatial_shape
                case IField():
                    ilimits = field.get_iteration_limits()
                    rank = ilimits.rank
                    spatial_bounds = ilimits.bounds

            if isinstance(ghost_layers, int):
                gl = (ghost_layers,) * rank
            else:
                gl = ghost_layers

            if "sycl_range_arg" not in self.extra_substitutions:
                shape_str = [
                    f"{s}" if isinstance(s, sp.Symbol) else f"{s}"
                    for s in spatial_bounds
                ]

                shape_args = [
                    f"static_cast<size_t>({s} - {g})" for s, g in zip(shape_str, gl)
                ]
                self.extra_substitutions["sycl_range_arg"] = f"{', '.join(shape_args)}"

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

        def add_item_param(self, param: Parameter, item_prop: SYCLItem):
            self.kernel_invocation_args.append(f"{param.name}")
            self.extra_substitutions["sycl_item"] = (
                f"{param.dtype.c_string()} {param.name}"
            )
            self.extra_substitutions["sycl_range"] = f"sycl::range<{item_prop.rank}>"

        def add_nditem_param(self, param: Parameter, item_prop: SYCLNDItem):
            self.kernel_invocation_args.append(f"{param.name}")
            self.extra_substitutions["sycl_item"] = (
                f"{param.dtype.c_string()} {param.name}"
            )
            self.extra_substitutions["sycl_range"] = f"sycl::nd_range<{item_prop.rank}>"
            self.extra_substitutions["sycl_range_arg"] = "grid * block, block"
            self.launch_config_args.append(
                f'auto grid  = getRangefromPyObject<{item_prop.rank}>(kwargs, "grid");'
            )
            self.launch_config_args.append(
                f'auto block = getRangefromPyObject<{item_prop.rank}>(kwargs, "block");'
            )
            self.kernel_lambda_caption.append("grid")
            self.kernel_lambda_caption.append("block")

        def check_same_shape(self, fields: Sequence[Field | IField]):
            if len(fields) > 1:
                check_args = ", ".join(
                    "&" + self._array_proxy_name(f.name) for f in fields
                )
                rep_field = fields[0]
                if isinstance(rep_field, Field):
                    rank = rep_field.spatial_dimensions
                else:
                    rank = rep_field.get_iteration_limits().rank

                self.precondition_checks.append(
                    f"checkSameShape({{ {check_args} }}, {rank});"
                )

        def check_fixed_shape_and_strides(self, field: Field | IField):
            proxy_name = self._array_proxy_name(field.name)
            if isinstance(field, Field):
                field_shape = field.spatial_shape

                #   Scalar fields may omit their trivial index dimension
                if field.index_shape not in ((), (1,)):
                    field_shape += field.index_shape
                    scalar_field = False
                else:
                    scalar_field = True

                field_strides = field.strides[: len(field_shape)]
            else:
                bspec = field.get_buffer_spec()
                field_shape = bspec.shape
                field_strides = bspec.strides

                #   New-style fields cannot have trivial index shape by design
                #   -> can omit scalar-field check
                scalar_field = False

            expect_shape = (
                "("
                + ", ".join(
                    (str(s) if isinstance(s, int) else "*") for s in field_shape
                )
                + ")"
            )

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
                    (str(s) if isinstance(s, int) else "*") for s in field_strides
                )
                + ")"
            )
            for coord, stride in enumerate(field_strides):
                if isinstance(stride, int):
                    self.precondition_checks.append(
                        f'checkFieldStride("{expect_strides}", {proxy_name}, {coord}, {stride});'
                    )

        @staticmethod
        def _typeno(dtype: PsType):
            # dpctl typenum is consitent with numpy dtype c-api
            # https://intelpython.github.io/dpctl/latest/api_reference/dpctl_capi.html
            if dtype.numpy_dtype is None:
                raise JitError(f"Cannot get typeno for non-numpy type {dtype}")
            npname = dtype.numpy_dtype.name.upper()
            # npname = dtype.c_string().upper()
            if npname.startswith("VOID"):
                npname = "VOID"  # for struct types
            return f"NPY_{npname}"

    @staticmethod
    def include_dirs() -> list[str]:
        return [np.get_include(), dpctl.get_include()]

    def __init__(self, compiler_info: CompilerInfo):
        if not HAVE_DPCTL:
            raise JitError("dpctl is not installed")
        self._compiler_info = compiler_info

    def render_module(self, kernel: Kernel, module_name: str) -> str:
        extr = self._handle_params(kernel)
        kernel_def = self._get_kernel_definition(kernel)
        includes = [f"#include {h}" for h in sorted(kernel.required_headers)]

        import hashlib
        from string import Template

        # each kernel needs a unique name so the sycl runtime can distiguies it
        kernel_class_name = hashlib.sha256(kernel_def.encode("utf-8")).hexdigest()

        templ = Template(_module_template.read_text())
        code_str = templ.substitute(
            includes="\n".join(includes),
            restrict_qualifier=self._compiler_info.restrict_qualifier(),
            module_name=module_name,
            kernel_name=kernel.name,
            kernel_definition=kernel_def,
            kernel_class_name=f"{module_name}_{kernel.name}_{kernel_class_name}",
            **extr.substitutions(),
            **extr.extra_substitutions,
        )
        return code_str

    def get_wrapper(
        self, kernel: Kernel, extension_module: ModuleType
    ) -> KernelWrapper:
        return SyclKernelWrapper(kernel, extension_module)

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
                extr.add_sycl_range_args(
                    param.fields[0], kernel.metadata.get("ghost_layers", 0)
                )

        for param in parameters:
            if ptr_props := param.get_properties(FieldBasePtr):
                extr.add_field_base_pointer(param, cast(FieldBasePtr, ptr_props.pop()))
            elif shape_props := param.get_properties(FieldShape):
                extr.add_shape_param(param, cast(FieldShape, shape_props.pop()))
            elif stride_props := param.get_properties(FieldStride):
                extr.add_stride_param(param, cast(FieldStride, stride_props.pop()))
            elif item_props := param.get_properties(SYCLItem):
                extr.add_item_param(param, cast(SYCLItem, item_props.pop()))
            elif item_props := param.get_properties(SYCLNDItem):
                extr.add_nditem_param(param, cast(SYCLNDItem, item_props.pop()))
            elif isinstance(param.dtype, PsPointerType):
                extr.add_pointer_param(param)
            else:
                extr.add_scalar_param(param)

        fields = kernel.get_fields()
        for f in fields:
            extr.check_fixed_shape_and_strides(f)

        extr.check_same_shape(
            [
                f
                for f in fields
                if isinstance(f, IField) or f.field_type == FieldType.GENERIC
            ]
        )

        return extr


class SyclKernelWrapper(KernelWrapper):
    def __init__(self, kernel: Kernel, jit_module: ModuleType):
        super().__init__(kernel)
        self._module = jit_module
        self._invoke = getattr(jit_module, "invoke")
        self._launch_config: GpuLaunchConfiguration | None = None
        if isinstance(kernel, GpuKernel):
            self._launch_config = kernel.get_launch_configuration()

    @property
    def launch_config(self) -> GpuLaunchConfiguration:
        assert self._launch_config
        return self._launch_config

    @staticmethod
    def _find_queue(**kwargs) -> dpctl.SyclDevice:
        queue: dpctl.SyclQueue = kwargs.get("queue", None)
        if not queue:
            for kwarg in kwargs.values():
                if isinstance(kwarg, dpt.usm_ndarray):
                    queue = kwarg.sycl_queue
                    break
        if not queue:
            raise ValueError("Did not find a sycl device")
        device = queue.get_sycl_device()
        return device

    @staticmethod
    def get_hardware_properties(device: dpctl.SyclDevice) -> HardwareProperties:
        return HardwareProperties(
            None, device.max_work_group_size, device.max_work_item_sizes3d
        )

    def __call__(self, **kwargs) -> None:
        if self._launch_config:
            device = self._find_queue(**kwargs)
            hw_props = self.get_hardware_properties(device)
            self._launch_config.hardware_properties = hw_props
            block, grid = self._launch_config.evaluate()
            return self._invoke(block=block, grid=grid, **kwargs)
        else:
            return self._invoke(**kwargs)
