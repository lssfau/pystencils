from ..ast.structural import PsAssignment, PsStructuralNode
from ..exceptions import MaterializationError
from ..platforms import Platform
from ..ast import PsAstNode
from ..ast.expressions import PsCall, PsExpression
from ..functions import PsMathFunction, PsConstantFunction, PsReductionWriteBack


class SelectFunctions:
    """Traverse the AST to replace all instances of `PsMathFunction` by their implementation
    provided by the given `Platform`."""

    def __init__(self, platform: Platform):
        self._platform = platform

    def __call__(self, node: PsAstNode) -> PsAstNode:
        return self.visit(node)

    def visit(self, node: PsAstNode) -> PsAstNode:
        node.children = [self.visit(c) for c in node.children]

        if isinstance(node, PsAssignment):
            call = node.rhs
            if isinstance(call, PsCall) and isinstance(
                call.function, PsReductionWriteBack
            ):
                ptr_expr, symbol_expr = call.args
                op = call.function.reduction_op
                reduction_func = self._platform.resolve_reduction(ptr_expr, symbol_expr, op)

                match reduction_func:
                    case PsStructuralNode():
                        return reduction_func
                    case _:
                        raise MaterializationError(
                            f"Unexpected return type for resolved function {call.function.name} in SelectFunctions."
                        )
            else:
                return node
        elif isinstance(node, PsCall) and isinstance(
                node.function, (PsMathFunction | PsConstantFunction)
        ):
            resolved_func = self._platform.select_function(node)
            assert isinstance(resolved_func, PsExpression)

            return resolved_func
        else:
            return node
