import pytest

from pystencils import (
    fields,
    TypedSymbol,
    AddReductionAssignment,
    Assignment,
    KernelConstraintsError,
    AssignmentCollection,
    FieldType,
)
from pystencils.backend.kernelcreation import KernelCreationContext, KernelAnalysis
from pystencils.types import constify
from pystencils.types.quick import Fp


def test_invalid_reduction_symbol_reassign():
    dtype = Fp(64)
    ctx = KernelCreationContext(default_dtype=dtype)
    analysis = KernelAnalysis(ctx)

    x = fields(f"x: [1d]")
    w = TypedSymbol("w", dtype)

    # illegal reassign to already locally defined symbol (here: reduction symbol)
    with pytest.raises(KernelConstraintsError):
        analysis([AddReductionAssignment(w, 3 * x.center()), Assignment(w, 0)])


def test_invalid_reduction_symbol_reference():
    dtype = Fp(64)
    ctx = KernelCreationContext(default_dtype=dtype)
    analysis = KernelAnalysis(ctx)

    x = fields(f"x: [1d]")
    v = TypedSymbol("v", dtype)
    w = TypedSymbol("w", dtype)

    # do not allow reduction symbol to be referenced on rhs of other assignments
    with pytest.raises(KernelConstraintsError):
        analysis([AddReductionAssignment(w, 3 * x.center()), Assignment(v, w)])


def test_readonly_fields_are_const():
    f, g, h, i, j = fields("f, g, h, i, j: double[2D]")
    buf1, buf2 = fields("buf1(1), buf2(1): double[1D]", field_type=FieldType.BUFFER)

    asms = AssignmentCollection(
        [
            Assignment(f(), 2 * g()),
            Assignment(i(), h() * (1 / j())),
            Assignment(buf1(0), g() * buf2(0)),
        ]
    )

    ctx = KernelCreationContext()
    analysis = KernelAnalysis(ctx)

    analysis(asms)

    for field in ctx.fields:
        buf = ctx.get_buffer(field)
        if field not in (f, i, buf1):
            assert buf.element_type == constify(field.dtype)
        else:
            assert buf.element_type == field.dtype
            assert not buf.element_type.const
