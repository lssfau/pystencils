from __future__ import annotations
from enum import Enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..ast.structural import (
    PsAstNode,
    PsBlock,
    PsStatement,
    PsDeclaration,
    PsAssignment,
    PsLoop,
    PsConditional,
    PsComment,
    PsPragma,
)

from ..ast.expressions import (
    PsExpression,
    PsAdd,
    PsAddressOf,
    PsArrayInitList,
    PsBinOp,
    PsBitwiseAnd,
    PsBitwiseOr,
    PsBitwiseXor,
    PsCall,
    PsCast,
    PsConstantExpr,
    PsMemAcc,
    PsDiv,
    PsRem,
    PsIntDiv,
    PsLeftShift,
    PsLookup,
    PsMul,
    PsNeg,
    PsRightShift,
    PsSub,
    PsSymbolExpr,
    PsLiteralExpr,
    PsTernary,
    PsAnd,
    PsOr,
    PsNot,
    PsEq,
    PsNe,
    PsGt,
    PsLt,
    PsGe,
    PsLe,
    PsSubscript,
)

from ..extensions.foreign_ast import PsForeignExpression

from ..memory import PsSymbol
from ..constants import PsConstant
from ...types import PsType

if TYPE_CHECKING:
    from ...codegen import Kernel


class EmissionError(Exception):
    """Indicates a fatal error during code printing"""


class LR(Enum):
    Left = 0
    Right = 1
    Middle = 2


class Ops(Enum):
    """Operator precedence and associativity in C/C++.

    See also https://en.cppreference.com/w/cpp/language/operator_precedence
    """

    Call = (2, LR.Left)
    Subscript = (2, LR.Left)
    Lookup = (2, LR.Left)

    Neg = (3, LR.Right)
    Not = (3, LR.Right)
    AddressOf = (3, LR.Right)
    Deref = (3, LR.Right)
    Cast = (3, LR.Right)

    Mul = (5, LR.Left)
    Div = (5, LR.Left)
    Rem = (5, LR.Left)

    Add = (6, LR.Left)
    Sub = (6, LR.Left)

    LeftShift = (7, LR.Left)
    RightShift = (7, LR.Left)

    RelOp = (9, LR.Left)  # >=, >, <, <=

    EqOp = (10, LR.Left)  # == and !=

    BitwiseAnd = (11, LR.Left)

    BitwiseXor = (12, LR.Left)

    BitwiseOr = (13, LR.Left)

    LogicAnd = (14, LR.Left)

    LogicOr = (15, LR.Left)

    Ternary = (16, LR.Right)

    Weakest = (17, LR.Middle)

    def __init__(self, pred: int, assoc: LR) -> None:
        self.precedence = pred
        self.assoc = assoc


class PrinterCtx:
    def __init__(self) -> None:
        self.operator_stack = [Ops.Weakest]
        self.branch_stack = [LR.Middle]
        self.indent_level = 0

    def push_op(self, operator: Ops, branch: LR):
        self.operator_stack.append(operator)
        self.branch_stack.append(branch)

    def pop_op(self) -> None:
        self.operator_stack.pop()
        self.branch_stack.pop()

    def switch_branch(self, branch: LR):
        self.branch_stack[-1] = branch

    @property
    def current_op(self) -> Ops:
        return self.operator_stack[-1]

    @property
    def current_branch(self) -> LR:
        return self.branch_stack[-1]

    def parenthesize(self, expr: str, next_operator: Ops) -> str:
        if next_operator.precedence > self.current_op.precedence:
            return f"({expr})"
        elif (
            next_operator.precedence == self.current_op.precedence
            and self.current_branch != self.current_op.assoc
        ):
            return f"({expr})"

        return expr

    def indent(self, line: str) -> str:
        return " " * self.indent_level + line


class BasePrinter(ABC):
    """Base code printer.

    The base printer is capable of printing syntax tree nodes valid in all output dialects.
    It is specialized in `CAstPrinter` for the C output language,
    and in `IRAstPrinter` for debug-printing the entire IR.
    """

    def __init__(self, indent_width=3, func_prefix: str | None = None):
        self._indent_width = indent_width
        self._func_prefix = func_prefix

    def __call__(self, obj: PsAstNode | Kernel) -> str:
        from ...codegen import Kernel
        if isinstance(obj, Kernel):
            sig = self.print_signature(obj)
            body_code = self.visit(obj.body, PrinterCtx())
            return f"{sig}\n{body_code}"
        else:
            return self.visit(obj, PrinterCtx())

    def visit(self, node: PsAstNode, pc: PrinterCtx) -> str:
        match node:
            case PsBlock(statements):
                if not statements:
                    return pc.indent("{ }")

                pc.indent_level += self._indent_width
                interior = "\n".join(self.visit(stmt, pc) for stmt in statements) + "\n"
                pc.indent_level -= self._indent_width
                return pc.indent("{\n") + interior + pc.indent("}")

            case PsStatement(expr):
                return pc.indent(f"{self.visit(expr, pc)};")

            case PsDeclaration(lhs, rhs):
                lhs_symb = node.declared_symbol
                lhs_code = self._symbol_decl(lhs_symb)
                rhs_code = self.visit(rhs, pc)

                return pc.indent(f"{lhs_code} = {rhs_code};")

            case PsAssignment(lhs, rhs):
                lhs_code = self.visit(lhs, pc)
                rhs_code = self.visit(rhs, pc)
                return pc.indent(f"{lhs_code} = {rhs_code};")

            case PsLoop(ctr, start, stop, step, body):
                ctr_symbol = ctr.symbol

                ctr_decl = self._symbol_decl(ctr_symbol)
                start_code = self.visit(start, pc)
                stop_code = self.visit(stop, pc)
                step_code = self.visit(step, pc)
                body_code = self.visit(body, pc)

                code = (
                    f"for({ctr_decl} = {start_code};"
                    + f" {ctr.symbol.name} < {stop_code};"
                    + f" {ctr.symbol.name} += {step_code})\n"
                    + body_code
                )
                return pc.indent(code)

            case PsConditional(condition, branch_true, branch_false):
                cond_code = self.visit(condition, pc)
                then_code = self.visit(branch_true, pc)

                code = f"if({cond_code})\n{then_code}"

                if branch_false is not None:
                    else_code = self.visit(branch_false, pc)
                    code += f"\nelse\n{else_code}"

                return pc.indent(code)

            case PsComment(lines):
                lines_list = list(lines)
                lines_list[0] = "/* " + lines_list[0]
                for i in range(1, len(lines_list)):
                    lines_list[i] = "   " + lines_list[i]
                lines_list[-1] = lines_list[-1] + " */"
                return pc.indent("\n".join(lines_list))

            case PsPragma(text):
                return pc.indent("#pragma " + text)

            case PsSymbolExpr(symbol):
                return symbol.name

            case PsConstantExpr(constant):
                return self._constant_literal(constant)

            case PsLiteralExpr(lit):
                return lit.text

            case PsMemAcc(base, offset):
                pc.push_op(Ops.Subscript, LR.Left)
                base_code = self.visit(base, pc)
                pc.pop_op()

                pc.push_op(Ops.Weakest, LR.Middle)
                index_code = self.visit(offset, pc)
                pc.pop_op()

                return pc.parenthesize(f"{base_code}[{index_code}]", Ops.Subscript)

            case PsSubscript(base, indices):
                pc.push_op(Ops.Subscript, LR.Left)
                base_code = self.visit(base, pc)
                pc.pop_op()

                pc.push_op(Ops.Weakest, LR.Middle)
                indices_code = "".join(
                    "[" + self.visit(idx, pc) + "]" for idx in indices
                )
                pc.pop_op()

                return pc.parenthesize(base_code + indices_code, Ops.Subscript)

            case PsLookup(aggr, member_name):
                pc.push_op(Ops.Lookup, LR.Left)
                aggr_code = self.visit(aggr, pc)
                pc.pop_op()

                return pc.parenthesize(f"{aggr_code}.{member_name}", Ops.Lookup)

            case PsCall(function, args):
                pc.push_op(Ops.Weakest, LR.Middle)
                args_string = ", ".join(self.visit(arg, pc) for arg in args)
                pc.pop_op()

                return pc.parenthesize(f"{function.name}({args_string})", Ops.Call)

            case PsBinOp(op1, op2):
                op_char, op = self._char_and_op(node)

                pc.push_op(op, LR.Left)
                op1_code = self.visit(op1, pc)
                pc.switch_branch(LR.Right)
                op2_code = self.visit(op2, pc)
                pc.pop_op()

                return pc.parenthesize(f"{op1_code} {op_char} {op2_code}", op)

            case PsNeg(operand):
                pc.push_op(Ops.Neg, LR.Right)
                operand_code = self.visit(operand, pc)
                pc.pop_op()

                return pc.parenthesize(f"-{operand_code}", Ops.Neg)

            case PsNot(operand):
                pc.push_op(Ops.Not, LR.Right)
                operand_code = self.visit(operand, pc)
                pc.pop_op()

                return pc.parenthesize(f"!{operand_code}", Ops.Not)

            case PsAddressOf(operand):
                pc.push_op(Ops.AddressOf, LR.Right)
                operand_code = self.visit(operand, pc)
                pc.pop_op()

                return pc.parenthesize(f"&{operand_code}", Ops.AddressOf)

            case PsCast(target_type, operand):
                pc.push_op(Ops.Cast, LR.Right)
                operand_code = self.visit(operand, pc)
                pc.pop_op()

                type_str = self._type_str(target_type)
                return pc.parenthesize(f"({type_str}) {operand_code}", Ops.Cast)

            case PsTernary(cond, then, els):
                pc.push_op(Ops.Ternary, LR.Left)
                cond_code = self.visit(cond, pc)
                pc.switch_branch(LR.Middle)
                then_code = self.visit(then, pc)
                pc.switch_branch(LR.Right)
                else_code = self.visit(els, pc)
                pc.pop_op()

                return pc.parenthesize(
                    f"{cond_code} ? {then_code} : {else_code}", Ops.Ternary
                )

            case PsArrayInitList(_):

                def print_arr(item) -> str:
                    if isinstance(item, PsExpression):
                        return self.visit(item, pc)
                    else:
                        #   it's a subarray
                        entries = ", ".join(print_arr(i) for i in item)
                        return "{ " + entries + " }"

                pc.push_op(Ops.Weakest, LR.Middle)
                arr_str = print_arr(node.items_grid)
                pc.pop_op()
                return arr_str

            case PsForeignExpression(children):
                pc.push_op(Ops.Weakest, LR.Middle)
                foreign_code = node.get_code(self.visit(c, pc) for c in children)
                pc.pop_op()
                return foreign_code

            case _:
                raise NotImplementedError(
                    f"BasePrinter does not know how to print {type(node)}"
                )

    def print_signature(self, func: Kernel) -> str:
        params_str = ", ".join(
            f"{self._type_str(p.dtype)} {p.name}" for p in func.parameters
        )

        from ...codegen import GpuKernel
        
        sig_parts = [self._func_prefix] if self._func_prefix is not None else []
        if isinstance(func, GpuKernel) and func.target.is_gpu():
            sig_parts.append("__global__")
        sig_parts += ["void", func.name, f"({params_str})"]
        signature = " ".join(sig_parts)
        return signature

    @abstractmethod
    def _symbol_decl(self, symb: PsSymbol) -> str:
        pass

    @abstractmethod
    def _constant_literal(self, constant: PsConstant) -> str:
        pass

    @abstractmethod
    def _type_str(self, dtype: PsType | None) -> str:
        """Return a valid string representation of the given type"""

    def _char_and_op(self, node: PsBinOp) -> tuple[str, Ops]:
        match node:
            case PsAdd():
                return ("+", Ops.Add)
            case PsSub():
                return ("-", Ops.Sub)
            case PsMul():
                return ("*", Ops.Mul)
            case PsDiv() | PsIntDiv():
                return ("/", Ops.Div)
            case PsRem():
                return ("%", Ops.Rem)
            case PsLeftShift():
                return ("<<", Ops.LeftShift)
            case PsRightShift():
                return (">>", Ops.RightShift)
            case PsBitwiseAnd():
                return ("&", Ops.BitwiseAnd)
            case PsBitwiseXor():
                return ("^", Ops.BitwiseXor)
            case PsBitwiseOr():
                return ("|", Ops.BitwiseOr)
            case PsAnd():
                return ("&&", Ops.LogicAnd)
            case PsOr():
                return ("||", Ops.LogicOr)
            case PsEq():
                return ("==", Ops.EqOp)
            case PsNe():
                return ("!=", Ops.EqOp)
            case PsGt():
                return (">", Ops.RelOp)
            case PsGe():
                return (">=", Ops.RelOp)
            case PsLt():
                return ("<", Ops.RelOp)
            case PsLe():
                return ("<=", Ops.RelOp)
            case _:
                assert False
