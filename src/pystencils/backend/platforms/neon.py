from __future__ import annotations

from typing import Sequence, cast, Callable

from pystencils.backend.constants import PsConstant

from ...types import (
    PsVectorType,
    PsScalarType,
    PsCustomType,
    PsIntegerType,
    PsSignedIntegerType,
    PsUnsignedIntegerType,
    PsIeeeFloatType,
    PsPointerType,
    constify,
    PsVoidType,
)
from ...types.quick import SInt, UInt, Fp

from ..kernelcreation import KernelCreationContext, Typifier
from ..functions import CFunction, PsMathFunction, MathFunctions
from ..ast.vector import PsVecBroadcast, PsVecMemAcc, PsVecHorizontal, ReductionOp
from ..ast.expressions import (
    PsCall,
    PsExpression,
    PsUnOp,
    PsBinOp,
    PsAdd,
    PsSub,
    PsMul,
    PsDiv,
    PsAddressOf,
    PsMemAcc,
    PsCast,
)
from .generic_cpu import GenericVectorCpu
from ..transformations.select_intrinsics import SelectIntrinsics, SelectionContext
from ..exceptions import MaterializationError


class NeonCpu(GenericVectorCpu):
    """Platform modeling the Aarch64 Neon vector architecture.

    All intrinsics information is extracted from
    https://developer.arm.com/architectures/instruction-sets/intrinsics/.

    Args:
        ctx: The kernel creation context
        enable_fp16: Whether to enable support for 16-bit floating arithmetic
    """

    def __init__(self, ctx: KernelCreationContext, enable_fp16: bool = False):
        super().__init__(ctx)

        self._enable_fp16 = enable_fp16

    @property
    def required_headers(self) -> set[str]:
        headers = {"<arm_neon.h>", '"pystencils_runtime/neon.hpp"'}
        return super().required_headers | headers

    def get_intrinsic_selector(
        self, use_builtin_convertvector: bool = False
    ) -> SelectIntrinsics:
        return SelectIntrinsicsNeon(
            self._ctx,
            self._enable_fp16,
            use_builtin_convertvector=use_builtin_convertvector,
        )


class ArmCommonIntrinsics:
    def _cast_fp16_ptr(self, expr: PsExpression) -> PsExpression:
        ptr_type: PsPointerType = cast(PsPointerType, expr.get_dtype())
        if (
            isinstance(ptr_type.base_type, PsIeeeFloatType)
            and ptr_type.base_type.width == 16
        ):
            target_ptr_type = PsPointerType(
                PsCustomType("float16_t", const=ptr_type.base_type.const)
            )
            return PsCast(target_ptr_type, expr)
        else:
            return expr

    def _op_type_suffix(self, sctype: PsScalarType) -> str:
        match sctype:
            case PsUnsignedIntegerType(w):
                return f"u{w}"
            case PsSignedIntegerType(w):
                return f"s{w}"
            case PsIeeeFloatType(w):
                return f"f{w}"
            case _:
                raise MaterializationError(
                    f"Invalid base type for SVE vector operation: {sctype}"
                )


class SelectIntrinsicsNeon(ArmCommonIntrinsics, SelectIntrinsics):
    def __init__(
        self,
        ctx: KernelCreationContext,
        enable_fp16: bool,
        use_builtin_convertvector: bool = False,
    ):
        super().__init__(ctx, use_builtin_convertvector=use_builtin_convertvector)
        self._enable_fp16 = enable_fp16

        self._typify = Typifier(ctx)

    def type_intrinsic(
        self, vector_type: PsVectorType, sc: SelectionContext
    ) -> PsCustomType:
        return self._vtype_intrin(vector_type)

    def _vtype_intrin(self, vector_type: PsVectorType) -> PsCustomType:
        sctype = vector_type.scalar_type

        if not isinstance(sctype, PsIntegerType | PsIeeeFloatType):
            raise MaterializationError(
                f"Don't know Neon intrinsic for vector type {vector_type}."
            )

        if sctype.is_float() and sctype.width == 16 and not self._enable_fp16:
            raise MaterializationError(
                f"Failed to select intrinsic for vector type {vector_type}: Fp16 support is disabled"
            )

        base_typename = str(vector_type.scalar_type)
        lanes = vector_type.vector_entries

        return PsCustomType(f"{base_typename}x{lanes}_t")

    def _check_fp16_support(self, vtype: PsVectorType, expr: PsExpression | PsConstant):
        if (
            vtype.scalar_type.is_float()
            and vtype.scalar_type.width == 16
            and not self._enable_fp16
        ):
            raise MaterializationError(
                "Unable to select intrinsics: Fp16 support for the Neon platform is disabled.\n"
                f"    At: {expr}"
            )

    def constant_intrinsic(self, c: PsConstant, sc: SelectionContext) -> PsExpression:
        vtype = cast(PsVectorType, c.get_dtype())
        self._check_fp16_support(vtype, c)
        args = [PsExpression.make(PsConstant(ci, vtype.scalar_type)) for ci in c.value]
        return self._vset(vtype)(*args)

    def op_intrinsic(
        self,
        expr: PsUnOp | PsBinOp,
        operands: Sequence[PsExpression],
        sc: SelectionContext,
    ) -> PsExpression:
        vtype: PsVectorType
        if isinstance(expr, PsVecHorizontal):
            # return type of expression itself is scalar, but input argument to intrinsic is a vector
            vtype = cast(PsVectorType, expr.vector_operand.get_dtype())
        else:
            vtype = cast(PsVectorType, expr.get_dtype())

        self._check_fp16_support(vtype, expr)
        intrin_func = self._op_intrin(expr, vtype)
        return intrin_func(*operands)

    def math_func_intrinsic(
        self, expr: PsCall, operands: Sequence[PsExpression], sc: SelectionContext
    ) -> PsExpression:
        vtype: PsVectorType = cast(PsVectorType, expr.get_dtype())
        func = expr.function
        assert isinstance(func, PsMathFunction)

        self._check_fp16_support(vtype, expr)
        intrin_func = self._math_intrin(func, vtype)
        return intrin_func(*operands)

    def vector_load(self, acc: PsVecMemAcc, sc: SelectionContext) -> PsExpression:
        vtype = cast(PsVectorType, acc.dtype)
        self._check_fp16_support(vtype, acc)

        if acc.stride is not None:
            raise MaterializationError(
                f"Unable to materialize strided memory access: {acc}"
            )

        addr: PsExpression = self._cast_fp16_ptr(
            self._typify(PsAddressOf(PsMemAcc(acc.pointer, acc.offset)))
        )

        ld_intrin = self._vld1(vtype)
        return ld_intrin(addr)

    def vector_store(
        self, acc: PsVecMemAcc, arg: PsExpression, sc: SelectionContext
    ) -> PsExpression:
        vtype = cast(PsVectorType, acc.dtype)
        self._check_fp16_support(vtype, acc)

        if acc.stride is not None:
            raise MaterializationError(
                f"Unable to materialize strided memory access: {acc}"
            )

        addr: PsExpression = self._cast_fp16_ptr(
            self._typify(PsAddressOf(PsMemAcc(acc.pointer, acc.offset)))
        )
        st_intrin = self._vst1(vtype)
        return st_intrin(addr, arg)

    def _q(self, vtype: PsVectorType) -> str:
        if vtype.width == 128:
            return "q"
        elif vtype.width == 64:
            return ""
        else:
            raise MaterializationError(
                f"Unable to materialize operation on vector type {vtype} "
                f"of width {vtype.width} bits"
            )

    def _vset(self, vtype: PsVectorType) -> CFunction:
        sctype = vtype.scalar_type
        suffix = self._op_type_suffix(sctype)
        vtype_intrin = self._vtype_intrin(vtype)
        return CFunction(
            f"pystencils::runtime::neon::vset{self._q(vtype)}_{suffix}",
            (sctype,) * vtype.vector_entries,
            vtype_intrin,
        )

    def _vld1(self, vtype: PsVectorType) -> CFunction:
        sctype = vtype.scalar_type
        ptr_type = PsPointerType(constify(sctype))
        suffix = self._op_type_suffix(sctype)
        rtype = self._vtype_intrin(vtype)

        return CFunction(f"vld1{self._q(vtype)}_{suffix}", (ptr_type,), rtype)

    def _vst1(self, vtype: PsVectorType) -> CFunction:
        sctype = vtype.scalar_type
        ptr_type = PsPointerType(constify(sctype))
        suffix = self._op_type_suffix(sctype)
        vtype_intrin = self._vtype_intrin(vtype)

        return CFunction(
            f"vst1{self._q(vtype)}_{suffix}", (ptr_type, vtype_intrin), PsVoidType()
        )

    def _op_intrin(
        self, op: PsUnOp | PsBinOp, vtype: PsVectorType
    ) -> Callable[..., PsExpression]:
        sctype = vtype.scalar_type
        vtype_suffix = self._op_type_suffix(sctype)

        vtype_intrin = self._vtype_intrin(vtype)
        nargs = 1 if isinstance(op, PsUnOp) else 2
        atypes = (vtype_intrin,) * nargs
        q = self._q(vtype)

        opstr: str
        match op:
            case PsVecBroadcast() if sctype == PsIeeeFloatType(16):
                #   The type float16_t expected by Neon float16 intrinsics
                #   is an alias of `__fp16` instead of C _Float16 type
                #   So we need to cast here.
                #   See also https://clang.llvm.org/docs/LanguageExtensions.html#half-precision-floating-point
                ifunc = CFunction(f"vdup{q}_n_{vtype_suffix}", (sctype,), vtype_intrin)

                def intrin(arg: PsExpression) -> PsCall:
                    cast_arg = PsCast(PsCustomType("float16_t"), arg)
                    return ifunc(cast_arg)

                return intrin

            case PsVecHorizontal(_, _, reduction_op):
                actual_op = (
                    ReductionOp.Add if reduction_op == ReductionOp.Sub else reduction_op
                )

                scalar_op: Callable[[PsExpression, PsExpression], PsExpression]
                match actual_op:
                    case ReductionOp.Add:
                        hzop = "vaddv"
                        scalar_op = PsAdd
                    case ReductionOp.Mul:
                        hzop = "pystencils::runtime::neon::vmulv"
                        scalar_op = PsMul
                    case ReductionOp.Min:
                        hzop = "vminv"
                        scalar_op = PsMathFunction(MathFunctions.Min)
                    case ReductionOp.Max:
                        hzop = "vmaxv"
                        scalar_op = PsMathFunction(MathFunctions.Max)
                    case _:
                        assert False, "unreachable code"

                hz_reduce_func = CFunction(
                    f"{hzop}{q}_{vtype_suffix}", (vtype_intrin,), sctype
                )

                def hreduce(scalar: PsExpression, vector: PsExpression) -> PsExpression:
                    return self._typify(scalar_op(scalar, hz_reduce_func(vector)))

                return hreduce

            case PsVecBroadcast():
                opstr = f"vdup{q}_n_{vtype_suffix}"
            case PsAdd():
                opstr = f"vadd{q}_{vtype_suffix}"
            case PsSub():
                opstr = f"vsub{q}_{vtype_suffix}"
            case PsMul():
                opstr = f"vmul{q}_{vtype_suffix}"
            case PsDiv() if vtype.is_float():
                opstr = f"vdiv{q}_{vtype_suffix}"

            case PsCast(target_type, arg):
                atype = arg.dtype

                if not (
                    isinstance(target_type, PsVectorType)
                    and isinstance(atype, PsVectorType)
                    and target_type.vector_entries == atype.vector_entries
                ):
                    raise MaterializationError(
                        f"Unable to select intrinsics for vector cast from {atype} to {target_type}"
                    )

                assert target_type == vtype

                if atype.width not in (64, 128):
                    raise MaterializationError(
                        f"Unable to materialize operation on vector type {atype}"
                    )

                match (target_type.scalar_type, atype.scalar_type):
                    case (
                        [SInt(w1), Fp(w2)]
                        | [UInt(w1), Fp(w2)]
                        | [Fp(w1), SInt(w2)]
                        | [Fp(w1), UInt(w2)]
                    ) if w1 == w2 and w1 in (
                        16,
                        32,
                        64,
                    ):
                        pass  # OK

                    case (
                        [Fp(16), Fp(32)]
                        | [Fp(32), Fp(16)]
                        | [Fp(32), Fp(64)]
                        | [Fp(64), Fp(32)]
                    ):
                        q = ""

                    case _:
                        raise MaterializationError(
                            f"Unable to select intrinsic for type cast from {atype} to {target_type}"
                        )

                atype_suffix = self._op_type_suffix(atype.scalar_type)
                opstr = f"vcvt{q}_{vtype_suffix}_{atype_suffix}"

            case _:
                raise MaterializationError(
                    f"Unable to select operation intrinsic for {type(op)}"
                )

        return CFunction(opstr, atypes, vtype_intrin)

    def _math_intrin(self, func: PsMathFunction, vtype: PsVectorType) -> CFunction:
        sctype = vtype.scalar_type
        suffix = self._op_type_suffix(sctype)

        rtype = self._vtype_intrin(vtype)
        atypes = (sctype,) * func.arg_count
        q = self._q(vtype)

        opstr: str
        match func.func:
            case MathFunctions.Abs if vtype.is_float() or vtype.is_sint():
                opstr = "abs"
            case MathFunctions.Min if vtype.is_float() or (
                sctype.is_int() and sctype.width <= 32
            ):
                opstr = "min"
            case MathFunctions.Max if vtype.is_float() or (
                sctype.is_int() and sctype.width <= 32
            ):
                opstr = "max"
            case MathFunctions.Sqrt if vtype.is_float():
                opstr = "sqrt"
            case _:
                raise MaterializationError(
                    f"Unable to select intrinsic for function {func}"
                )

        return CFunction(f"v{opstr}{q}_{suffix}", atypes, rtype)
