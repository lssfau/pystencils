from .ast.expressions import PsExpression, PsCall, PsAdd, PsSub, PsMul, PsDiv
from .exceptions import PsInternalCompilerError
from .functions import PsMathFunction, MathFunctions
from ..sympyextensions.reduction import ReductionOp


def reduction_op_to_expr(op: ReductionOp, op1, op2) -> PsExpression:
    match op:
        case ReductionOp.Add:
            return PsAdd(op1, op2)
        case ReductionOp.Sub:
            return PsSub(op1, op2)
        case ReductionOp.Mul:
            return PsMul(op1, op2)
        case ReductionOp.Div:
            return PsDiv(op1, op2)
        case ReductionOp.Min:
            return PsCall(PsMathFunction(MathFunctions.Min), [op1, op2])
        case ReductionOp.Max:
            return PsCall(PsMathFunction(MathFunctions.Max), [op1, op2])
        case _:
            raise PsInternalCompilerError(
                f"Found unsupported operation type for reduction assignments: {op}."
            )
