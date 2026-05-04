from warnings import warn
from typing import Any, Dict, Optional

import sympy as sp

from .codegen import Kernel
from .jit import KernelWrapper


def to_dot(expr: sp.Expr, graph_style: Optional[Dict[str, Any]] = None, short=True):
    """Show a sympy or pystencils AST as dot graph"""
    try:
        import graphviz
    except ImportError:
        print("graphviz is not installed. Visualizing the AST is not available")
        return

    graph_style = {} if graph_style is None else graph_style

    # if isinstance(expr, Node):
    #     from pystencils.backends.dot import print_dot
    #     return graphviz.Source(print_dot(expr, short=short, graph_attr=graph_style))
    if isinstance(expr, sp.Basic):
        from sympy.printing.dot import dotprint

        return graphviz.Source(dotprint(expr, graph_attr=graph_style))
    else:
        #  TODO Implement dot / graphviz exporter for new backend AST
        raise NotImplementedError(
            "Printing of AST nodes for the new backend is not implemented yet"
        )


def get_code_str(ker: KernelWrapper | Kernel):
    warn(
        "`get_code_str` is deprecated and will be removed with pystencils 2.1\nUse `ker.get_c_code()` instead.",
        UserWarning,
    )
    return ker.get_c_code() if isinstance(ker, Kernel) else ker.kernel.get_c_code()


def show_code(ast: KernelWrapper | Kernel):
    warn(
        "`ps.show_code` is deprecated and will be removed with pystencils 2.1\nUse `ps.inspect()` instead.",
        UserWarning,
    )

    from .inspection import inspect

    inspect(ast)
