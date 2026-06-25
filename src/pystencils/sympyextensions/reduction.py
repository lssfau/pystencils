from enum import Enum

import sympy as sp
from sympy.codegen.ast import AssignmentBase

from . import TypedSymbol


class ReductionOp(Enum):
    Add = "+"
    Sub = "-"
    Mul = "*"
    Div = "/"
    Min = "min"
    Max = "max"

    @property
    def neutral_element(self) -> sp.Number:
        match self:
            case ReductionOp.Add | ReductionOp.Sub:
                return sp.Integer(0)
            case ReductionOp.Mul:
                return sp.Integer(1)
            case ReductionOp.Min:
                return sp.core.numbers.Infinity()
            case ReductionOp.Max:
                return sp.core.numbers.NegativeInfinity()
            case _:
                raise ValueError(f"No neutral element for reduction: {self}")


class ReductionAssignment(AssignmentBase):
    """
    Base class for reduced assignments.

    Attributes:
    ===========

    reduction_op : ReductionOp
       Enum for binary operation being applied in the assignment, such as "Add" for "+", "Sub" for "-", etc.
    """

    _reduction_op = None  # type: ReductionOp

    @property
    def reduction_op(self):
        return self._reduction_op

    @reduction_op.setter
    def reduction_op(self, op):
        self._reduction_op = op

    @classmethod
    def _check_args(cls, lhs, rhs):
        super()._check_args(lhs, rhs)

        if not isinstance(lhs, TypedSymbol):
            raise TypeError(
                f"lhs of needs to be a TypedSymbol. Got {type(lhs)} instead."
            )


class AddReductionAssignment(ReductionAssignment):
    reduction_op = ReductionOp.Add


class SubReductionAssignment(ReductionAssignment):
    reduction_op = ReductionOp.Sub


class MulReductionAssignment(ReductionAssignment):
    reduction_op = ReductionOp.Mul


class MinReductionAssignment(ReductionAssignment):
    reduction_op = ReductionOp.Min


class MaxReductionAssignment(ReductionAssignment):
    reduction_op = ReductionOp.Max


# Mapping from ReductionOp enum to ReductionAssigment classes
_reduction_assignment_classes = {
    cls.reduction_op: cls
    for cls in [
        AddReductionAssignment,
        SubReductionAssignment,
        MulReductionAssignment,
        MinReductionAssignment,
        MaxReductionAssignment,
    ]
}


def reduction_assignment(lhs, op: ReductionOp, rhs):
    if op not in _reduction_assignment_classes:
        raise ValueError("Unrecognized operator %s" % op)
    return _reduction_assignment_classes[op](lhs, rhs)
