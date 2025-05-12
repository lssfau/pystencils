import pytest

from pystencils import fields, TypedSymbol, AddReductionAssignment, Assignment, KernelConstraintsError
from pystencils.backend.kernelcreation import KernelCreationContext, KernelAnalysis
from pystencils.sympyextensions import mem_acc
from pystencils.types.quick import Ptr, Fp


def test_invalid_reduction_symbol_reassign():
    dtype = Fp(64)
    ctx = KernelCreationContext(default_dtype=dtype)
    analysis = KernelAnalysis(ctx)

    x = fields(f"x: [1d]")
    w = TypedSymbol("w", dtype)

    # illegal reassign to already locally defined symbol (here: reduction symbol)
    with pytest.raises(KernelConstraintsError):
        analysis([
            AddReductionAssignment(w, 3 * x.center()),
            Assignment(w, 0)
        ])

def test_invalid_reduction_symbol_reference():
    dtype = Fp(64)
    ctx = KernelCreationContext(default_dtype=dtype)
    analysis = KernelAnalysis(ctx)

    x = fields(f"x: [1d]")
    v = TypedSymbol("v", dtype)
    w = TypedSymbol("w", dtype)

    # do not allow reduction symbol to be referenced on rhs of other assignments
    with pytest.raises(KernelConstraintsError):
        analysis([
            AddReductionAssignment(w, 3 * x.center()),
            Assignment(v, w)
        ])