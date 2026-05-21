import sympy as sp
from typing import Sequence
from dataclasses import dataclass
from contextlib import contextmanager

from ..sympyextensions.typed_sympy import TypedSymbol
from ..field import Field

from .flowgraph import (
    Flowgraph,
    FlowgraphNode,
    FlowgraphAssignment,
    EquationsBlock,
    Cases,
    Let,
    Export,
    Store,
    Reduce,
    Bottom,
    Top,
    Subgraph,
)


@dataclass
class PrintingContext:
    indent_depth: int = 0
    indent_text: str = "· "

    @contextmanager
    def indent(self):
        self.indent_depth += 1
        try:
            yield None
        finally:
            self.indent_depth -= 1

    def indent_line(self, line: str) -> str:
        return self.indent_text * self.indent_depth + line


class PlainTextPrinter:

    def print(self, node: Flowgraph | FlowgraphAssignment | FlowgraphNode) -> str:
        return self._visit(node, PrintingContext())

    def _visit(
        self, node: Flowgraph | FlowgraphAssignment | FlowgraphNode, pc: PrintingContext
    ) -> str:
        match node:
            case Flowgraph():
                return self._graph(node, pc)
            case Top() | Bottom():
                return node.name
            case EquationsBlock():
                return self._block(node, pc)
            case Cases():
                return self._cases(node, pc)
            case Subgraph():
                return self._subgraph(node, pc)
            case Let():
                return self._let(node)
            case Export():
                return self._export(node)
            case Store():
                return self._store(node)
            case Reduce():
                return self._reduce(node)
            case _:
                assert False, f"TODO: {type(node)}"

    def _symb_with_type(self, symb: sp.Basic | Field):
        match symb:
            case Field():
                return repr(symb)
            case Field.Access():
                return str(symb)
            case TypedSymbol():
                return f"{symb.name}: {symb.dtype}"
            case _:
                return str(symb)

    def _let(self, let: Let) -> str:
        return f"let {self._symb_with_type(let.lhs)} = {let.rhs}"

    def _export(self, export: Export) -> str:
        return f"export {self._symb_with_type(export.lhs)} = {export.rhs}"

    def _store(self, store: Store) -> str:
        return f"store {self._symb_with_type(store.lhs)} = {store.rhs}"

    def _reduce(self, reduce: Reduce) -> str:
        return f"reduce({reduce.op.value}) {self._symb_with_type(reduce.lhs)} = {reduce.rhs}"

    def _assignment_line(self, asm: FlowgraphAssignment, pc: PrintingContext) -> str:
        return pc.indent_line(f"{self._symb_with_type(asm.lhs)} = {asm.rhs}\n")

    def _import_line(
        self, pred: FlowgraphNode, symbols: Sequence[sp.Symbol], pc: PrintingContext
    ) -> str:
        pred_name = pred.name

        imports = sorted(symbols, key=str)
        batches = [imports[b : b + 4] for b in range(0, len(imports), 4)]  # noqa: E203

        if len(batches) == 1:
            symbols_list = ", ".join(str(s) for s in batches[0])
            return pc.indent_line(f"import {symbols_list} from {pred_name}\n")
        else:
            code = pc.indent_line("import (\n")
            with pc.indent():
                for batch in batches:
                    symbols_list = ", ".join(str(s) for s in batch)
                    code += pc.indent_line(f"{symbols_list},\n")
            code += pc.indent_line(f") from {pred_name}\n")
            return code

    def _import_lines(self, node: FlowgraphNode, pc: PrintingContext) -> str:
        code = ""
        for pred in sorted(node.predecessors, key=lambda n: n.name):
            imports = sorted(node.free_symbols & pred.exports, key=str)
            if imports:
                code += self._import_line(pred, imports, pc)

        if code:
            code += pc.indent_line("\n")

        return code

    def _export_line(self, node: FlowgraphNode, pc: PrintingContext) -> str:
        exports = sorted(node.exports, key=str)
        batches = [exports[b : b + 4] for b in range(0, len(exports), 4)]  # noqa: E203

        if len(batches) == 1:
            symbols_list = ", ".join(str(s) for s in batches[0])
            return pc.indent_line(f"export {symbols_list}\n")
        else:
            code = pc.indent_line("export (\n")
            with pc.indent():
                for batch in batches:
                    symbols_list = ", ".join(str(s) for s in batch)
                    code += pc.indent_line(f"{symbols_list},\n")
            code += pc.indent_line(")\n")
            return code

    def _store_line(self, node: FlowgraphNode, pc: PrintingContext) -> str:
        stores = sorted((str(s.lhs) for s in node.effects if isinstance(s, Store)))
        symbols_list = ", ".join(stores)
        return pc.indent_line(f"store {symbols_list}\n")

    def _reduce_line(self, node: FlowgraphNode, pc: PrintingContext) -> str:
        reductions: list[Reduce] = sorted(
            (s for s in node.effects if isinstance(s, Reduce)), key=lambda s: str(s.lhs)
        )
        reduce_list = ", ".join(f"({str(s.lhs)}: {s.op.value})" for s in reductions)
        return pc.indent_line(f"reduce {reduce_list}\n")

    def _block(self, block: EquationsBlock, pc: PrintingContext) -> str:
        block_name = block.name

        code = pc.indent_line(f"block {block_name} =\n")

        with pc.indent():
            code += self._block_body(block, pc)

        return code

    def _block_body(self, block: EquationsBlock, pc: PrintingContext) -> str:
        code = ""

        code += self._import_lines(block, pc)

        code += pc.indent_line("let\n")

        with pc.indent():
            code += "".join(self._assignment_line(asm, pc) for asm in block.assignments)

        code += pc.indent_line("in\n")

        with pc.indent():
            if block.exports:
                code += self._export_line(block, pc)
            if any(isinstance(e, Store) for e in block.effects):
                code += self._store_line(block, pc)
            if any(isinstance(e, Reduce) for e in block.effects):
                code += self._reduce_line(block, pc)
        return code

    def _cases(self, cases: Cases, pc: PrintingContext) -> str:
        node_name = cases.name

        code = pc.indent_line(f"cases {node_name} =\n")

        with pc.indent():
            code += self._import_lines(cases, pc)

            exports = ", ".join(sorted(str(s) for s in cases.exports))
            code += pc.indent_line(f"export {exports} from\n")

            with pc.indent():
                for cond, subgr in cases.branches:
                    case_label = "_" if cond is sp.true else str(cond)

                    subgr_nodes = [
                        n for n in subgr.walk() if not isinstance(n, Top | Bottom)
                    ]
                    if len(subgr_nodes) == 1 and isinstance(
                        subgr_nodes[0], EquationsBlock
                    ):
                        code += pc.indent_line(f"{case_label} =>\n")
                        with pc.indent():
                            code += self._block_body(subgr_nodes[0], pc)
                    else:
                        subgr_name = subgr.name

                        code += pc.indent_line(f"{case_label} =>\n")
                        with pc.indent():
                            code += pc.indent_line(f"subgraph {subgr_name} =\n")
                            with pc.indent():
                                code += self._subgraph_body(subgr, pc)

        return code

    def _subgraph(self, subgr: Subgraph, pc: PrintingContext) -> str:
        node_name = subgr.name

        code = pc.indent_line(f"subgraph {node_name} =\n")

        with pc.indent():
            code += self._import_lines(subgr, pc)
            code += self._subgraph_body(subgr.graph, pc)

        return code

    def _subgraph_body(self, graph: Flowgraph, pc: PrintingContext) -> str:
        code = ""

        if (block := self._is_trivial_graph(graph)) is not None:
            code += self._block_body(block, pc)
        else:
            code += self._graph_let(graph, pc)
            code += self._graph_exports(graph, pc)

        return code

    def _is_trivial_graph(self, graph: Flowgraph) -> EquationsBlock | None:
        inner_nodes = [n for n in graph.walk() if not isinstance(n, Top | Bottom)]
        if len(inner_nodes) == 1 and isinstance(inner_nodes[0], EquationsBlock):
            return inner_nodes[0]
        else:
            return None

    def _graph_let(self, graph: Flowgraph, pc: PrintingContext) -> str:
        code = pc.indent_line("let\n")

        with pc.indent():
            for node in graph.list_topological():
                if isinstance(node, Top | Bottom):
                    continue
                code += self._visit(node, pc) + pc.indent_line("\n")

        return code

    def _graph_tie(self, graph: Flowgraph, pc: PrintingContext) -> str:
        code = pc.indent_line("in\n")

        with pc.indent():
            pred_names = ", ".join(
                sorted(pred.name for pred in sorted(graph.bottom.predecessors, key=lambda n: n.name))
            )
            code += pc.indent_line(f"{pred_names}\n")

        return code

    def _graph_exports(self, graph: Flowgraph, pc: PrintingContext) -> str:
        code = ""
        if graph.exports:
            code += pc.indent_line("in\n")

            with pc.indent():
                for node in sorted(graph.bottom.predecessors, key=lambda n: n.name):
                    exports = ", ".join(sorted(str(e) for e in node.exports))
                    if exports:
                        code += pc.indent_line(f"export {exports} from {node.name}\n")
        return code

    def _graph(self, graph: Flowgraph, pc: PrintingContext) -> str:
        params = [
            self._symb_with_type(s) for s in sorted(graph.fields, key=lambda f: f.name)
        ] + [self._symb_with_type(s) for s in sorted(graph.free_symbols, key=str)]
        params_list = ", ".join(params)

        code = pc.indent_line(f"graph {graph.name} ({params_list}) =\n")

        with pc.indent():
            # code += pc.indent_line(f"parameters {params_list}\n")

            if (block := self._is_trivial_graph(graph)) is not None:
                code += self._block_body(block, pc)
            else:
                code += self._graph_let(graph, pc)
                if graph.exports:
                    code += self._graph_exports(graph, pc)

        return code


class GraphvizPrinter:
    COLOR_SCHEME = "paired10"
    BLOCK_COLOR = "1"
    CASES_COLOR = "3"
    COND_COLOR = "4"
    BOTTOM_COLOR = "9"
    TOP_COLOR = "9"

    def __init__(self):
        self._pretty_printer = PlainTextPrinter()

    def to_digraph(self, graph: FlowgraphNode | Flowgraph):
        import graphviz
        import html

        dot = graphviz.Digraph(node_attr=dict(colorscheme=self.COLOR_SCHEME))

        for node in graph.walk():
            name = str(hash(node))
            match node:
                case Top():
                    dot.node(
                        name,
                        label="⊤",
                        shape="circle",
                        **self._node_attrs(node),
                    )
                case Bottom():
                    dot.node(
                        name,
                        label="⊥",
                        shape="circle",
                        **self._node_attrs(node),
                    )
                case _:
                    label = node.name
                    dot.node(
                        name,
                        label,
                        shape="rectangle",
                        labeljust="left",
                        **self._node_attrs(node),
                    )

        ctr = 0
        for node in graph.walk():
            n2 = str(hash(node))
            for pred in node.predecessors:
                n1 = str(hash(pred))

                if isinstance(pred, Top):
                    parameters = node.free_symbols - set().union(
                        *(p.exports for p in node.predecessors)
                    )
                    imports = sorted(parameters, key=str)
                elif isinstance(node, Bottom):
                    imports = sorted(pred.exports, key=str)
                else:
                    imports = sorted(node.free_symbols & pred.exports, key=str)

                batchsize = 2

                label = ",<br/>".join(
                    ", ".join(
                        html.escape(s.name)
                        for s in imports[i : i + batchsize]  # noqa: E203
                    )  # noqa: E203
                    for i in range(0, len(imports), batchsize)
                )
                if label:
                    label = f'<<font face="mono">{label}</font>>'
                dot.edge(n1, n2, label, **self._edge_attrs(ctr))
                ctr += 1

        return dot

    def _node_attrs(self, node: FlowgraphNode) -> dict[str, str]:
        match node:
            case Bottom():
                return dict(style="filled", color=self.BOTTOM_COLOR)
            case Top():
                return dict(style="filled", color=self.BOTTOM_COLOR)
            case EquationsBlock():
                return dict(style="filled", color=self.BLOCK_COLOR)
            case Cases():
                return dict(style="filled", color=self.CASES_COLOR)
            case _:
                return dict()

    def _edge_attrs(self, counter: int) -> dict[str, str]:
        color = str(1 + counter % 7)
        return dict(colorscheme="dark28", color=color, fontcolor=color)


def to_dot(graph: FlowgraphNode):
    return GraphvizPrinter().to_digraph(graph)
