from __future__ import annotations

from typing import cast, Callable, Sequence
from functools import reduce

from ...types import (
    PsType,
    PsVectorType,
    PsCustomType,
    PsIntegerType,
    PsUnsignedIntegerType,
    PsSignedIntegerType,
    PsIeeeFloatType,
    PsPointerType,
    PsVoidType,
    constify,
)
from ..exceptions import MaterializationError

from ..memory import PsSymbol
from ..constants import PsConstant
from ..functions import CFunction, PsMathFunction, MathFunctions
from .generic_cpu import GenericVectorCpu
from .neon import ArmCommonIntrinsics
from ..kernelcreation import KernelCreationContext, Typifier, AstFactory
from ..transformations.select_intrinsics import SelectIntrinsics, SelectionContext
from ..ast.expressions import (
    PsExpression,
    PsAddressOf,
    PsMemAcc,
    PsUnOp,
    PsBinOp,
    PsAdd,
    PsSub,
    PsMul,
    PsDiv,
    PsCast,
    PsCall,
)
from ..ast.structural import PsDeclaration, PsBlock
from ..ast.vector import PsVecMemAcc, PsVecBroadcast, PsVecHorizontal, ReductionOp


class SveCpu(GenericVectorCpu):
    """Platform modeling the Aarch64 SVE vector architecture.

    All intrinsics information is extracted from
    https://developer.arm.com/architectures/instruction-sets/intrinsics/.

    Args:
        ctx: The kernel creation context
    """

    def __init__(self, ctx: KernelCreationContext):
        super().__init__(ctx)

    @property
    def required_headers(self) -> set[str]:
        headers = {"<arm_sve.h>", '"pystencils_runtime/sve.hpp"'}
        return super().required_headers | headers

    def get_intrinsic_selector(
        self, use_builtin_convertvector: bool = False
    ) -> SelectIntrinsics:
        return SelectIntrinsicsSve(
            self._ctx,
            use_builtin_convertvector=use_builtin_convertvector,
        )


class SveSelectionContext(SelectionContext):

    def __init__(self, visitor: SelectIntrinsicsSve):
        super().__init__(visitor)

        self._fixed_width_predicates: dict[tuple[int, int], PsSymbol] = dict()

    def lane_predicate(self, vtype: PsVectorType) -> PsExpression:
        sctype_width = vtype.scalar_type.width
        lanes = vtype.vector_entries
        key = (sctype_width, lanes)

        if key not in self._fixed_width_predicates:
            mask_symb = self._visitor._ctx.get_new_symbol(
                f"__mask_b{sctype_width}x{lanes}", SelectIntrinsicsSve.svbool_t
            )
            self._fixed_width_predicates[key] = mask_symb

        return PsExpression.make(self._fixed_width_predicates[key])

    def get_fixed_width_predicate_decls(self) -> list[PsDeclaration]:
        decls: list[PsDeclaration] = []

        for (sctype_width, lanes), symb in self._fixed_width_predicates.items():
            rhs = self._k_lanes_predicate(lanes, sctype_width)
            decls.append(PsDeclaration(PsExpression.make(symb), rhs))

        return decls

    def _k_lanes_predicate(self, lanes: int, width: int) -> PsExpression:
        atype = PsUnsignedIntegerType(32)
        zero = PsExpression.make(PsConstant(0, atype))
        end = PsExpression.make(PsConstant(lanes, atype))
        func = CFunction(
            f"svwhilelt_b{width}_u32", (atype, atype), SelectIntrinsicsSve.svbool_t
        )
        return func(zero, end)


class SelectIntrinsicsSve(ArmCommonIntrinsics, SelectIntrinsics):

    svbool_t = PsCustomType("svbool_t")

    def __init__(
        self,
        ctx: KernelCreationContext,
        use_builtin_convertvector: bool = False,
    ):
        super().__init__(ctx, use_builtin_convertvector=use_builtin_convertvector)

        self._factory = AstFactory(self._ctx)
        self._typify = Typifier(ctx)

    def __call__(self, node: PsBlock) -> PsBlock:
        sc = SveSelectionContext(self)
        node = cast(PsBlock, self.visit(node, sc))
        node.statements = sc.get_fixed_width_predicate_decls() + node.statements
        return node

    def type_intrinsic(
        self, vector_type: PsVectorType, sc: SelectionContext
    ) -> PsCustomType:
        return self._vtype_intrin(vector_type)

    def _vtype_intrin(self, vector_type: PsVectorType) -> PsCustomType:
        sctype = vector_type.scalar_type

        if not isinstance(sctype, PsIntegerType | PsIeeeFloatType):
            raise MaterializationError(
                f"Don't know SVE intrinsic for vector type {vector_type}."
            )

        base_typename = str(vector_type.scalar_type)

        return PsCustomType(f"sv{base_typename}_t")

    def constant_intrinsic(self, c: PsConstant, sc: SelectionContext) -> PsExpression:
        vtype = cast(PsVectorType, c.get_dtype())
        args = [PsExpression.make(PsConstant(ci, vtype.scalar_type)) for ci in c.value]
        insr = self._insr(vtype)
        undef = self._undef(vtype)
        return reduce(insr, args[::-1], undef)

    def op_intrinsic(
        self,
        expr: PsUnOp | PsBinOp,
        operands: Sequence[PsExpression],
        sc: SelectionContext,
    ) -> PsExpression:
        vtype: PsVectorType
        predicate: PsExpression

        if isinstance(expr, PsVecHorizontal):
            # return type of expression itself is scalar, but input argument to intrinsic is a vector
            vtype = cast(PsVectorType, expr.vector_operand.get_dtype())
            sve_sc = cast(SveSelectionContext, sc)
            predicate = sve_sc.lane_predicate(vtype)
        else:
            vtype = cast(PsVectorType, expr.get_dtype())
            predicate = self._ptrue(vtype.scalar_type.width)

        if isinstance(expr, PsVecBroadcast):
            return self._broadcast(vtype)(*operands)
        else:
            intrin_func = self._op_intrin(expr, vtype)
            return intrin_func(predicate, *operands)

    def math_func_intrinsic(
        self, expr: PsCall, operands: Sequence[PsExpression], sc: SelectionContext
    ) -> PsExpression:
        vtype: PsVectorType = cast(PsVectorType, expr.get_dtype())
        func = expr.function
        assert isinstance(func, PsMathFunction)

        intrin_func = self._math_intrin(func, vtype)
        return intrin_func(self._ptrue(vtype.scalar_type.width), *operands)

    def vector_load(self, acc: PsVecMemAcc, sc: SelectionContext) -> PsExpression:
        vtype = cast(PsVectorType, acc.dtype)

        addr: PsExpression = self._cast_fp16_ptr(
            self._typify(PsAddressOf(PsMemAcc(acc.pointer, acc.offset)))
        )

        sve_sc = cast(SveSelectionContext, sc)
        pred = sve_sc.lane_predicate(vtype)

        if acc.stride is not None:
            if vtype.scalar_type.width not in (32, 64):
                raise MaterializationError(
                    f"SVE does not support gather-loads for type {vtype}"
                )

            gather_intrin = self._svld1_gather(vtype)
            scalar_idx_type = PsSignedIntegerType(vtype.scalar_type.width)
            svindex = self._svindex(scalar_idx_type)
            zero = PsExpression.make(PsConstant(0, scalar_idx_type))
            step = PsCast(scalar_idx_type, acc.stride)
            return gather_intrin(pred, addr, svindex(zero, step))
        else:
            ld_intrin = self._svld1(vtype)
            return ld_intrin(pred, addr)

    def vector_store(
        self, acc: PsVecMemAcc, arg: PsExpression, sc: SelectionContext
    ) -> PsExpression:
        vtype = cast(PsVectorType, acc.dtype)

        addr: PsExpression = self._cast_fp16_ptr(
            self._typify(PsAddressOf(PsMemAcc(acc.pointer, acc.offset)))
        )

        sve_sc = cast(SveSelectionContext, sc)
        pred = sve_sc.lane_predicate(vtype)

        if acc.stride is not None:
            if vtype.scalar_type.width not in (32, 64):
                raise MaterializationError(
                    f"SVE does not support scatter-stores for type {vtype}"
                )

            scatter_intrin = self._svst1_scatter(vtype)
            scalar_idx_type = PsSignedIntegerType(vtype.scalar_type.width)
            svindex = self._svindex(scalar_idx_type)
            zero = PsExpression.make(PsConstant(0, scalar_idx_type))
            step = PsCast(scalar_idx_type, acc.stride)
            return scatter_intrin(pred, addr, svindex(zero, step), arg)
        else:
            st_intrin = self._svst1(vtype)
            return st_intrin(pred, addr, arg)

    def _insr(self, vtype: PsVectorType) -> CFunction:
        suffix = self._op_type_suffix(vtype.scalar_type)
        atypes = (vtype, vtype.scalar_type)
        return CFunction(f"svinsr_n_{suffix}", atypes, vtype)

    def _undef(self, vtype: PsVectorType) -> PsExpression:
        suffix = self._op_type_suffix(vtype.scalar_type)
        return CFunction(f"svundef_{suffix}", (), vtype)()

    def _ptrue(self, width: int) -> PsExpression:
        return CFunction(f"svptrue_b{width}", (), self.svbool_t)()

    def _svindex(self, itype: PsIntegerType) -> CFunction:
        suffix = self._op_type_suffix(itype)
        rtype = self._vtype_intrin(PsVectorType(itype, 1))
        return CFunction(f"svindex_{suffix}", (itype, itype), rtype)

    def _svld1(self, vtype: PsVectorType) -> CFunction:
        sctype = vtype.scalar_type
        ptr_type = PsPointerType(constify(sctype))
        suffix = self._op_type_suffix(sctype)
        rtype = self._vtype_intrin(vtype)

        return CFunction(
            f"svld1_{suffix}",
            (
                self.svbool_t,
                ptr_type,
            ),
            rtype,
        )

    def _svld1_gather(self, vtype: PsVectorType) -> CFunction:
        sctype = vtype.scalar_type
        ptr_type = PsPointerType(constify(sctype))

        scalar_idx_type = PsSignedIntegerType(sctype.width)
        idx_type_intrin = self._vtype_intrin(
            PsVectorType(scalar_idx_type, vtype.vector_entries)
        )
        idx_suffix = self._op_type_suffix(scalar_idx_type)
        atypes = (self.svbool_t, ptr_type, idx_type_intrin)

        v_suffix = self._op_type_suffix(sctype)
        rtype = self._vtype_intrin(vtype)

        return CFunction(f"svld1_gather_{idx_suffix}index_{v_suffix}", atypes, rtype)

    def _svst1(self, vtype: PsVectorType) -> CFunction:
        sctype = vtype.scalar_type
        ptr_type = PsPointerType(constify(sctype))
        suffix = self._op_type_suffix(sctype)
        vtype_intrin = self._vtype_intrin(vtype)

        return CFunction(
            f"svst1_{suffix}", (self.svbool_t, ptr_type, vtype_intrin), PsVoidType()
        )

    def _svst1_scatter(self, vtype: PsVectorType) -> CFunction:
        sctype = vtype.scalar_type
        ptr_type = PsPointerType(constify(sctype))

        scalar_idx_type = PsSignedIntegerType(sctype.width)
        idx_type_intrin = self._vtype_intrin(
            PsVectorType(scalar_idx_type, vtype.vector_entries)
        )
        idx_suffix = self._op_type_suffix(scalar_idx_type)
        atypes = (self.svbool_t, ptr_type, idx_type_intrin, vtype)

        v_suffix = self._op_type_suffix(sctype)

        return CFunction(
            f"svst1_scatter_{idx_suffix}index_{v_suffix}", atypes, PsVoidType()
        )

    def _broadcast(self, vtype: PsVectorType) -> Callable[..., PsExpression]:
        sctype = vtype.scalar_type
        vtype_suffix = self._op_type_suffix(sctype)
        vtype_intrin = self._vtype_intrin(vtype)

        ifunc = CFunction(f"svdup_n_{vtype_suffix}", (sctype,), vtype_intrin)

        if sctype == PsIeeeFloatType(16):
            #   The type float16_t expected by Neon and SVE float16 intrinsics
            #   is an alias of `__fp16` instead of C _Float16 type
            #   So we need to cast here.
            #   See also https://clang.llvm.org/docs/LanguageExtensions.html#half-precision-floating-point

            def intrin(arg: PsExpression) -> PsExpression:
                cast_arg = PsCast(PsCustomType("float16_t"), arg)
                return ifunc(cast_arg)

            return intrin
        else:
            return ifunc

    def _op_intrin(
        self, op: PsUnOp | PsBinOp, vtype: PsVectorType
    ) -> Callable[..., PsExpression]:
        sctype = vtype.scalar_type
        vtype_suffix = self._op_type_suffix(sctype)

        vtype_intrin = self._vtype_intrin(vtype)
        nargs = 1 if isinstance(op, PsUnOp) else 2
        atypes: tuple[PsType, ...] = (self.svbool_t,) + (vtype_intrin,) * nargs

        match op:
            case PsAdd():
                opstr = f"svadd_{vtype_suffix}_x"
            case PsSub():
                opstr = f"svsub_{vtype_suffix}_x"
            case PsMul():
                opstr = f"svmul_{vtype_suffix}_x"
            case PsDiv() if vtype.is_float():
                opstr = f"svdiv_{vtype_suffix}_x"

            case PsVecHorizontal(_, _, reduction_op):
                actual_op = (
                    ReductionOp.Add if reduction_op == ReductionOp.Sub else reduction_op
                )

                scalar_op: Callable[[PsExpression, PsExpression], PsExpression]
                match actual_op:
                    case ReductionOp.Add:
                        hzop = "svaddv"
                        scalar_op = PsAdd
                    case ReductionOp.Mul:
                        raise MaterializationError(
                            "Horizontal multiplication is not available on SVE"
                        )
                    case ReductionOp.Min:
                        hzop = "svminv"
                        scalar_op = PsMathFunction(MathFunctions.Min)
                    case ReductionOp.Max:
                        hzop = "svmaxv"
                        scalar_op = PsMathFunction(MathFunctions.Max)
                    case _:
                        assert False, "unreachable code"

                hz_reduce_func = CFunction(
                    f"{hzop}_{vtype_suffix}", (self.svbool_t, vtype_intrin), sctype
                )

                def hreduce(
                    predicate: PsExpression, scalar: PsExpression, vector: PsExpression
                ) -> PsExpression:
                    return self._typify(
                        scalar_op(scalar, hz_reduce_func(predicate, vector))
                    )

                return hreduce

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

                if target_type.scalar_type.width != atype.scalar_type.width:
                    raise MaterializationError(
                        f"Cannot materialize vector type cast from {atype} to {target_type}"
                    )

                atype_suffix = self._op_type_suffix(atype.scalar_type)
                atypes = (self.svbool_t, self._vtype_intrin(atype))
                opstr = f"svcvt_{vtype_suffix}_{atype_suffix}_x"

            case _:
                raise MaterializationError(
                    f"Unable to select operation intrinsic for {type(op)}"
                )

        return CFunction(opstr, atypes, vtype_intrin)

    def _math_intrin(self, func: PsMathFunction, vtype: PsVectorType) -> CFunction:
        sctype = vtype.scalar_type
        suffix = self._op_type_suffix(sctype)

        rtype = self._vtype_intrin(vtype)
        atypes = (self.svbool_t,) + (sctype,) * func.arg_count

        opstr: str
        match func.func:
            case MathFunctions.Abs if vtype.is_float() or vtype.is_sint():
                opstr = "abs"
            case MathFunctions.Min if vtype.is_float() or vtype.is_int():
                opstr = "min"
            case MathFunctions.Max if vtype.is_float() or vtype.is_int():
                opstr = "max"
            case MathFunctions.Sqrt if vtype.is_float():
                opstr = "sqrt"
            case _:
                raise MaterializationError(
                    f"Unable to select intrinsic for function {func}"
                )

        return CFunction(f"sv{opstr}_{suffix}_x", atypes, rtype)
