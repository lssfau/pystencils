from __future__ import annotations
from typing import TYPE_CHECKING

from pystencils.backend.constants import PsConstant
from pystencils.backend.emission.base_printer import PrinterCtx
from pystencils.backend.memory import PsSymbol
from pystencils.types.meta import PsType, deconstify

from .base_printer import BasePrinter, Ops, LR

from ..ast import PsAstNode
from ..ast.expressions import PsBufferAcc
from ..ast.vector import PsVecMemAcc, PsVecBroadcast, PsVecHorizontal
from ..ast.axes import (
    PsAxisRange,
    PsAxesCube,
    PsLoopAxis,
    PsSimdAxis,
    PsIterationAxis,
    PsParallelLoopAxis,
)

if TYPE_CHECKING:
    from ...codegen import Kernel


def emit_ir(ir: PsAstNode | Kernel):
    """Emit the IR as C-like pseudo-code for inspection."""
    ir_printer = IRAstPrinter()
    return ir_printer(ir)


class IRAstPrinter(BasePrinter):
    """Print the IR AST as pseudo-code.

    This printer produces a complete pseudocode representation of a pystencils AST.
    Other than the `CAstPrinter`, the `IRAstPrinter` is capable of emitting code for
    each node defined in `ast <pystencils.backend.ast>`.
    It is furthermore configurable w.r.t. the level of detail it should emit.

    Args:
        indent_width: Number of spaces with which to indent lines in each nested block.
        annotate_constants: If ``True`` (the default), annotate all constant literals with their data type.
    """

    def __init__(self, indent_width=3, annotate_constants: bool = True):
        super().__init__(indent_width)
        self._annotate_constants = annotate_constants

    def visit(self, node: PsAstNode, pc: PrinterCtx) -> str:
        match node:
            case PsBufferAcc(ptr, indices):
                pc.push_op(Ops.Subscript, LR.Left)
                base_code = self.visit(ptr, pc)
                pc.pop_op()

                pc.push_op(Ops.Weakest, LR.Middle)
                indices_code = ", ".join(self.visit(idx, pc) for idx in indices)
                pc.pop_op()

                return pc.parenthesize(
                    base_code + "[" + indices_code + "]", Ops.Subscript
                )

            case PsVecMemAcc(ptr, offset, lanes, stride):
                pc.push_op(Ops.Subscript, LR.Left)
                ptr_code = self.visit(ptr, pc)
                pc.pop_op()

                pc.push_op(Ops.Weakest, LR.Middle)
                offset_code = self.visit(offset, pc)
                pc.pop_op()

                stride_code = "" if stride is None else f", stride={stride}"

                code = f"vec_memacc< {lanes}{stride_code} >({ptr_code}, {offset_code})"
                return pc.parenthesize(code, Ops.Subscript)

            case PsVecBroadcast(lanes, operand):
                pc.push_op(Ops.Weakest, LR.Middle)
                operand_code = self.visit(operand, pc)
                pc.pop_op()

                return pc.parenthesize(
                    f"vec_broadcast<{lanes}>({operand_code})", Ops.Weakest
                )

            case PsVecHorizontal(scalar_operand, vector_operand, reduction_op):
                pc.push_op(Ops.Weakest, LR.Middle)
                scalar_operand_code = self.visit(scalar_operand, pc)
                vector_operand_code = self.visit(vector_operand, pc)
                pc.pop_op()

                return pc.parenthesize(
                    f"vec_horizontal_{reduction_op.name.lower()}({scalar_operand_code, vector_operand_code})",
                    Ops.Weakest,
                )

            case PsAxisRange(ctr, start, stop, step):
                ctr_code = self.visit(ctr, pc)
                start_code = self.visit(start, pc)
                stop_code = self.visit(stop, pc)
                step_code = self.visit(step, pc)

                return f"range({ctr_code} : [{start_code} : {stop_code} : {step_code}])"

            case PsAxesCube(ranges, body):
                pc.indent_level += self._indent_width
                ranges_code = ",\n".join(pc.indent(self.visit(r, pc)) for r in ranges)
                pc.indent_level -= self._indent_width

                body_code = self.visit(body, pc)
                code = pc.indent("axes-cube(\n")
                code += ranges_code + "\n"
                code += pc.indent(")\n")
                code += body_code

                return code

            case PsIterationAxis(rang, body):
                range_code = self.visit(rang, pc)
                body_code = self.visit(body, pc)
                code = f"{self._axis_key(node)}({range_code})\n{body_code}"
                return pc.indent(code)

            case _:
                return super().visit(node, pc)

    def _symbol_decl(self, symb: PsSymbol):
        return f"{symb.name}: {self._type_str(symb.dtype)}"

    def _constant_literal(self, constant: PsConstant) -> str:
        if self._annotate_constants:
            return f"[{constant.value}: {self._deconst_type_str(constant.dtype)}]"
        else:
            return str(constant.value)

    def _type_str(self, dtype: PsType | None):
        if dtype is None:
            return "<untyped>"
        else:
            return str(dtype)

    def _deconst_type_str(self, dtype: PsType | None):
        if dtype is None:
            return "<untyped>"
        else:
            return str(deconstify(dtype))

    def _axis_key(self, node: PsIterationAxis):
        match node:
            case PsLoopAxis():
                return "loop-axis"
            case PsParallelLoopAxis():
                directives = ", ".join(
                    f"{k}({v})"
                    for k, v in zip(
                        ["num_threads", "schedule", "collapse"],
                        [node.num_threads, node.schedule, node.collapse],
                    )
                    if v is not None
                )

                if directives:
                    directives = f"< {directives} >"

                return f"parallel-loop-axis{directives}"
            case PsSimdAxis():
                return "simd-axis"
            case _:
                raise NotImplementedError(
                    f"Don't know how to print axis of type{type(node)}"
                )
