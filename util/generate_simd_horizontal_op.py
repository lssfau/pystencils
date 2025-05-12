from enum import Enum

FCT_QUALIFIERS = "inline"


class InstructionSets(Enum):
    SSE3 = "SSE3"
    AVX = "AVX"
    AVX512 = "AVX512"
    NEON = "NEON"

    def __str__(self):
        return self.value


class ReductionOps(Enum):
    Add = ("add", "+")
    Mul = ("mul", "*")
    Min = ("min", "min")
    Max = ("max", "max")

    def __init__(self, op_name, op_str):
        self.op_name = op_name
        self.op_str = op_str


class ScalarTypes(Enum):
    Double = "double"
    Float = "float"

    def __str__(self):
        return self.value


class VectorTypes(Enum):
    SSE3_128d = "__m128d"
    SSE3_128 = "__m128"

    AVX_256d = "__m256d"
    AVX_256 = "__m256"
    AVX_128 = "__m128"

    AVX_512d = "__m512d"
    AVX_512 = "__m512"

    NEON_64x2 = "float64x2_t"
    NEON_32x4 = "float32x4_t"

    def __str__(self):
        return self.value


class Variable:
    def __init__(self, name: str, dtype: ScalarTypes | VectorTypes):
        self._name = name
        self._dtype = dtype

    def __str__(self):
        return f"{self._dtype} {self._name}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> ScalarTypes | VectorTypes:
        return self._dtype


def get_intrin_from_vector_type(vtype: VectorTypes) -> InstructionSets:
    match vtype:
        case VectorTypes.SSE3_128 | VectorTypes.SSE3_128d:
            return InstructionSets.SSE3
        case VectorTypes.AVX_256 | VectorTypes.AVX_256d:
            return InstructionSets.AVX
        case VectorTypes.AVX_512 | VectorTypes.AVX_512d:
            return InstructionSets.AVX512
        case VectorTypes.NEON_32x4 | VectorTypes.NEON_64x2:
            return InstructionSets.NEON


def intrin_prefix(instruction_set: InstructionSets, double_prec: bool):
    match instruction_set:
        case InstructionSets.SSE3:
            return "_mm"
        case InstructionSets.AVX:
            return "_mm256"
        case InstructionSets.AVX512:
            return "_mm512"
        case InstructionSets.NEON:
            return "vgetq" if double_prec else "vget"
        case _:
            raise ValueError(f"Unknown instruction set {instruction_set}")


def intrin_suffix(instruction_set: InstructionSets, double_prec: bool):
    if instruction_set in [InstructionSets.SSE3, InstructionSets.AVX, InstructionSets.AVX512]:
        return "pd" if double_prec else "ps"
    elif instruction_set in [InstructionSets.NEON]:
        return "f64" if double_prec else "f32"
    else:
        raise ValueError(f"Unknown instruction set {instruction_set}")


def generate_hadd_intrin(instruction_set: InstructionSets, double_prec: bool, v: str):
    return f"{intrin_prefix(instruction_set, double_prec)}_hadd_{intrin_suffix(instruction_set, double_prec)}({v}, {v})"


def generate_shuffle_intrin(instruction_set: InstructionSets, double_prec: bool, v: str, offset):
    return f"_mm_shuffle_{intrin_suffix(instruction_set, double_prec)}({v}, {v}, {offset})"


def generate_op_intrin(instruction_set: InstructionSets, double_prec: bool, reduction_op: ReductionOps, a: str, b: str):
    return f"_mm_{reduction_op.op_name}_{intrin_suffix(instruction_set, double_prec)}({a}, {b})"


def generate_cvts_intrin(double_prec: bool, v: str):
    convert_suffix = "f64" if double_prec else "f32"
    intrin_suffix = "d" if double_prec else "s"
    return f"_mm_cvts{intrin_suffix}_{convert_suffix}({v})"


def generate_fct_name(instruction_set: InstructionSets, double_prec: bool, op: ReductionOps):
    prefix = intrin_prefix(instruction_set, double_prec)
    suffix = intrin_suffix(instruction_set, double_prec)
    return f"{prefix}_horizontal_{op.op_name}_{suffix}"


def generate_fct_decl(instruction_set: InstructionSets, op: ReductionOps, svar: Variable, vvar: Variable):
    double_prec = svar.dtype is ScalarTypes.Double
    return f"{FCT_QUALIFIERS} {svar.dtype} {generate_fct_name(instruction_set, double_prec, op)}({svar}, {vvar}) {{ \n"


# SSE & AVX provide horizontal add 'hadd' intrinsic that allows for specialized handling
def generate_simd_horizontal_add(scalar_var: Variable, vector_var: Variable):
    reduction_op = ReductionOps.Add
    instruction_set = get_intrin_from_vector_type(vector_var.dtype)
    double_prec = scalar_var.dtype is ScalarTypes.Double

    sname = scalar_var.name
    vtype = vector_var.dtype
    vname = vector_var.name

    simd_op = lambda a, b: generate_op_intrin(instruction_set, double_prec, reduction_op, a, b)
    hadd = lambda var: generate_hadd_intrin(instruction_set, double_prec, var)
    cvts = lambda var: generate_cvts_intrin(double_prec, var)

    # function body
    body = f"\t{vtype} _v = {vname};\n"
    match instruction_set:
        case InstructionSets.SSE3:
            if double_prec:
                body += f"\treturn {sname} + {cvts(hadd('_v'))};\n"
            else:
                body += f"\t{vtype} _h = {hadd('_v')};\n" \
                        f"\treturn {sname} + {cvts(simd_op('_h', '_mm_movehdup_ps(_h)'))};\n"

        case InstructionSets.AVX:
            if double_prec:
                body += f"\t{vtype} _h = {hadd('_v')};\n" \
                        f"\treturn {sname} + {cvts(simd_op('_mm256_extractf128_pd(_h,1)', '_mm256_castpd256_pd128(_h)'))};\n"
            else:
                add_i = "_mm_hadd_ps(_i,_i)"
                body += f"\t{vtype} _h = {hadd('_v')};\n" \
                        f"\t__m128  _i = {simd_op('_mm256_extractf128_ps(_h,1)', '_mm256_castps256_ps128(_h)')};\n" \
                        f"\treturn {sname} + {cvts(add_i)};\n"

        case _:
            raise ValueError(f"No specialized version of horizontal_add available for {instruction_set}")

    # function decl
    decl = generate_fct_decl(instruction_set, reduction_op, scalar_var, vector_var)

    return decl + body + "}\n"


def generate_simd_horizontal_op(reduction_op: ReductionOps, scalar_var: Variable, vector_var: Variable):
    instruction_set = get_intrin_from_vector_type(vector_var.dtype)
    double_prec = scalar_var.dtype is ScalarTypes.Double

    # generate specialized version for add operation
    if reduction_op == ReductionOps.Add and instruction_set in [InstructionSets.SSE3, InstructionSets.AVX]:
        return generate_simd_horizontal_add(scalar_var, vector_var)

    sname = scalar_var.name
    stype = scalar_var.dtype
    vtype = vector_var.dtype
    vname = vector_var.name

    opname = reduction_op.op_name
    opstr = reduction_op.op_str

    reduction_function = f"f{opname}" \
        if reduction_op in [ReductionOps.Max, ReductionOps.Min] else None

    simd_op = lambda a, b: generate_op_intrin(instruction_set, double_prec, reduction_op, a, b)
    cvts = lambda var: generate_cvts_intrin(double_prec, var)
    shuffle = lambda var, offset: generate_shuffle_intrin(instruction_set, double_prec, var, offset)

    # function body
    body = f"\t{vtype} _v = {vname};\n" if instruction_set != InstructionSets.AVX512 else ""
    match instruction_set:
        case InstructionSets.SSE3:
            if double_prec:
                body += f"\t{stype} _r = {cvts(simd_op('_v', shuffle('_v', 1)))};\n"
            else:
                body += f"\t{vtype} _h = {simd_op('_v', shuffle('_v', 177))};\n" \
                        f"\t{stype} _r = {cvts(simd_op('_h', shuffle('_h', 10)))};\n"

        case InstructionSets.AVX:
            if double_prec:
                body += f"\t__m128d _w = {simd_op('_mm256_extractf128_pd(_v,1)', '_mm256_castpd256_pd128(_v)')};\n" \
                        f"\t{stype} _r = {cvts(simd_op('_w', '_mm_permute_pd(_w,1)'))}; \n"
            else:
                body += f"\t__m128 _w = {simd_op('_mm256_extractf128_ps(_v,1)', '_mm256_castps256_ps128(_v)')};\n" \
                        f"\t__m128 _h = {simd_op('_w', shuffle('_w', 177))};\n" \
                        f"\t{stype} _r = {cvts(simd_op('_h', shuffle('_h', 10)))};\n"

        case InstructionSets.AVX512:
            suffix = intrin_suffix(instruction_set, double_prec)
            body += f"\t{stype} _r = _mm512_reduce_{opname}_{suffix}({vname});\n"

        case InstructionSets.NEON:
            if double_prec:
                body += f"\t{stype} _r = vgetq_lane_f64(_v,0);\n"
                if reduction_function:
                    body += f"\t_r = {reduction_function}(_r, vgetq_lane_f64(_v,1));\n"
                else:
                    body += f"\t_r {opstr}= vgetq_lane_f64(_v,1);\n"
            else:
                body += f"\tfloat32x2_t _w = v{opname}_f32(vget_high_f32(_v), vget_low_f32(_v));\n" \
                        f"\t{stype} _r = vgetq_lane_f32(_w,0);\n"
                if reduction_function:
                    body += f"\t_r = {reduction_function}(_r, vget_lane_f32(_w,1));\n"
                else:
                    body += f"\t_r {opstr}= vget_lane_f32(_w,1);\n"

        case _:
            raise ValueError(f"Unsupported instruction set {instruction_set}")

    # finalize reduction
    if reduction_function:
        body += f"\treturn {reduction_function}(_r, {sname});\n"
    else:
        body += f"\treturn {sname} {opstr} _r;\n"

    # function decl
    decl = generate_fct_decl(instruction_set, reduction_op, scalar_var, vector_var)

    return decl + body + "}\n"


stypes = {
    True: ScalarTypes.Double,
    False: ScalarTypes.Float
}

vtypes_for_instruction_set = {
    InstructionSets.SSE3: {
        True: VectorTypes.SSE3_128d,
        False: VectorTypes.SSE3_128
    },
    InstructionSets.AVX: {
        True: VectorTypes.AVX_256d,
        False: VectorTypes.AVX_256
    },
    InstructionSets.AVX512: {
        True: VectorTypes.AVX_512d,
        False: VectorTypes.AVX_512
    },
    InstructionSets.NEON: {
        True: VectorTypes.NEON_64x2,
        False: VectorTypes.NEON_32x4
    },
}

guards_for_instruction_sets = {
    InstructionSets.SSE3: "__SSE3__",
    InstructionSets.AVX: "__AVX__",
    InstructionSets.AVX512: '__AVX512F__',
    InstructionSets.NEON: '_M_ARM64',
}

code = """#pragma once

#include <cmath>

"""

for instruction_set in InstructionSets:
    code += f"#if defined({guards_for_instruction_sets[instruction_set]})\n"

    if instruction_set in [InstructionSets.SSE3, InstructionSets.AVX, InstructionSets.AVX512]:
        code += "#include <immintrin.h>\n\n"
    elif instruction_set == InstructionSets.NEON:
        code += "#include <arm_neon.h>\n\n"
    else:
        ValueError(f"Missing header include for instruction set {instruction_set}")

    for reduction_op in ReductionOps:
        for double_prec in [True, False]:
            scalar_var = Variable("dst", stypes[double_prec])
            vector_var = Variable("src", vtypes_for_instruction_set[instruction_set][double_prec])

            code += generate_simd_horizontal_op(reduction_op, scalar_var, vector_var) + "\n"

    code += "#endif\n\n"

print(code)
