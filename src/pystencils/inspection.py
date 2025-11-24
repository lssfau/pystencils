from .backend.ast import PsAstNode
from .backend.emission import CAstPrinter, IRAstPrinter
from .codegen import Kernel
from .jit import KernelWrapper


def inspect(obj: PsAstNode | Kernel | KernelWrapper, *, show_cpp: bool = False):
    """Print the IR or C++ code of an abstract syntax tree or kernel object,
    if possible with syntax highlighting."""

    try:
        from IPython.display import display, Code
        
        def do_print(code):
            display(Code(code, language="C++"))

    except ImportError:
        def do_print(code):
            print(code)

    match obj:
        case PsAstNode():
            printer = CAstPrinter() if show_cpp else IRAstPrinter()
            do_print(printer(obj))
        case Kernel():
            do_print(obj.get_c_code() if show_cpp else obj.get_ir_code())
        case KernelWrapper(ker):
            do_print(ker.get_c_code() if show_cpp else ker.get_ir_code())
        case _:
            raise ValueError(f"Cannot inspect object of type {type(obj)}")
