from __future__ import annotations

import sympy as sp

from dataclasses import dataclass

from ...field import Field
from ...flow.flowgraph import (
    FlowgraphNode,
    EquationsBlock,
    Bottom,
    Top,
    Cases,
    Flowgraph,
    Subgraph,
)
from ...types import PsType
from ...sympyextensions.typed_sympy import TypedSymbol

from ..ast.structural import (
    PsBlock,
    PsAssignment,
    PsExpression,
    PsConditional,
    PsStructuralNode,
    PsDeclaration,
    PsComment,
)
from ..ast.expressions import PsUndefined
from ..exceptions import PsInternalCompilerError
from ..memory import PsSymbol
from .context import KernelCreationContext
from .freeze import FreezeExpressions
from . import Typifier


class NodeSymbolTable:
    def __init__(self, ctx: KernelCreationContext) -> None:
        self._ctx = ctx
        self._symbol_map: dict[str, str] = dict()

    def add_mapping(self, sympy_name: str, backend_name: str):
        self._symbol_map[sympy_name] = backend_name

    def get_local_name(self, sympy_name: str | sp.Symbol) -> str:
        return self._symbol_map[
            sympy_name if isinstance(sympy_name, str) else sympy_name.name
        ]

    def get_symbol(self, name: str, dtype: PsType | None = None) -> PsSymbol:
        local_name = self._symbol_map.get(name)
        if local_name is not None:
            return self._ctx.get_symbol(local_name, dtype)
        else:
            return self._ctx.get_symbol(name, dtype)

    def declare_symbol(self, name: str, dtype: PsType | None = None) -> PsSymbol:
        symb = self._ctx.get_new_symbol(name, dtype)
        self._symbol_map[name] = symb.name
        return symb

    def get_local_symbol(self, name: str, dtype: PsType | None = None) -> PsSymbol:
        if (local_name := self._symbol_map.get(name)) is not None:
            return self._ctx.get_symbol(local_name, dtype)

        raise KeyError(f"Symbol {name} not declared locally")

    @property
    def symbol_map(self) -> dict[str, str]:
        return self._symbol_map


@dataclass
class NodeInfo:
    node: FlowgraphNode
    #   Use a list of successors for deterministic traversal order,
    #   despite worse asymptotic complexity
    successors: list[FlowgraphNode]
    sym_table: NodeSymbolTable
    ast: PsBlock | None = None


class FlowgraphInfo:
    """Augment a flowgraph with extra information required during translation to the backend IR.

    For each node, the flowgraph info holds:
     - A list of its successors
     - A symbol table
     - After freezing, its backend AST

    Args:
        ctx: The current kernel creation context
        flowgraph: A flowgraph in canonical form; i.e. with explicit bottom and top node
    """

    def __init__(self, ctx: KernelCreationContext, flowgraph: Flowgraph) -> None:
        self._nodes: dict[FlowgraphNode, NodeInfo] = {
            node: NodeInfo(node, list(), NodeSymbolTable(ctx))
            for node in flowgraph.walk()
        }

        self._nodes_linearized = flowgraph.list_topological()

        for node in flowgraph.walk():
            for pred in node.predecessors:
                if node not in self._nodes[pred].successors:
                    self._nodes[pred].successors.append(node)

    @property
    def node_infos(self) -> dict[FlowgraphNode, NodeInfo]:
        return self._nodes

    @property
    def nodes_linearized(self) -> tuple[FlowgraphNode, ...]:
        return self._nodes_linearized


class FreezeFlowgraph:
    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx
        self._typify = Typifier(self._ctx)

    def __call__(self, graph: Flowgraph) -> PsBlock:
        graph_info = FlowgraphInfo(self._ctx, graph)
        self._init_symbol_table_top(graph_info)
        return self._process_subgraph(graph, graph_info)

    def _process_subgraph(
        self, subgraph: Flowgraph, graph_info: FlowgraphInfo
    ) -> PsBlock:
        """Process all nodes of a subgraph in linearized order, by filling in their symbol tables
        from their predecessors and then freezing their code.

        Before calling this function, the symbol table of the graph's Top node must already be initialized.
        """

        #   Assemble symbol tables and freeze all nodes
        for node in graph_info.nodes_linearized:
            if not isinstance(node, Top):
                self._process_node(graph_info.node_infos[node], graph_info)

        self._init_symbol_table_bottom(
            graph_info.node_infos[subgraph.bottom], graph_info
        )

        #   Assemble kernel body by linearizing the graph
        body: PsBlock = self._linearize_graph(graph_info)
        return body

    def _process_node(self, node_info: NodeInfo, graph: FlowgraphInfo):
        self._init_symbol_table(node_info, graph)
        if not isinstance(node_info.node, Bottom):
            self._freeze_node(node_info)
            self._typify_node(node_info)

    def _init_symbol_table_top(self, graph: FlowgraphInfo):
        """Initialize the symbol table of the Top node to hold all unbound symbols of the graph"""
        top_info = graph.node_infos[Top()]
        parameters: set[sp.Symbol] = set()

        for inner_node in graph.node_infos:
            if not isinstance(inner_node, Top):
                imported_symbols: set[sp.Symbol] = set().union(
                    *(p.exports for p in inner_node.predecessors)
                )
                missing_symbols = inner_node.free_symbols - imported_symbols
                if missing_symbols:
                    if Top() not in inner_node.predecessors:
                        raise PsInternalCompilerError(
                            "Flowgraph not in canonical form: "
                            "Encountered node with non-imported free symbols not connected to Top."
                        )
                parameters |= missing_symbols

        for symb in parameters:
            assert not isinstance(
                symb, Field.Access
            )  # Field.Access inherits from sp.Symbol

            backend_symb = self._ctx.get_symbol(
                symb.name,
                (
                    self._ctx.resolve_dynamic_type(symb.dtype)
                    if isinstance(symb, TypedSymbol)
                    else None
                ),
            )
            top_info.sym_table.add_mapping(symb.name, backend_symb.name)

    def _init_symbol_table_bottom(self, bot_info: NodeInfo, graph: FlowgraphInfo):
        """Initialize the symbol table of the Bottom node to hold all symbols exported from the flowgraph"""
        for pred in bot_info.node.predecessors:
            pred_info = graph.node_infos[pred]
            for export in pred.exports:
                backend_name = pred_info.sym_table.get_local_name(export)
                bot_info.sym_table.add_mapping(export.name, backend_name)

    def _init_symbol_table(self, node_info: NodeInfo, graph: FlowgraphInfo):
        """Initialize the given node's symbol table by importing symbols from its predecessors"""
        for symb in node_info.node.free_symbols:
            assert not isinstance(symb, Field.Access)
            for pred in node_info.node.predecessors:
                pred_info = graph.node_infos[pred]
                if symb in pred.exports:
                    node_info.sym_table.add_mapping(
                        symb.name, pred_info.sym_table.get_local_name(symb.name)
                    )
                    break
            else:
                top_info = graph.node_infos[Top()]
                node_info.sym_table.add_mapping(
                    symb.name, top_info.sym_table.get_local_name(symb.name)
                )

    def _freeze_node(self, node_info: NodeInfo):
        """Freeze the given node to a backend AST fragment, and store it in the node info's ``ast`` member"""
        match node_info.node:
            case EquationsBlock(assignments):
                freeze = FreezeExpressions(self._ctx, node_info.sym_table)
                node_info.ast = PsBlock(freeze(asm) for asm in assignments)

            case Subgraph(subgraph):
                freeze = FreezeExpressions(self._ctx, node_info.sym_table)

                subgraph_info = FlowgraphInfo(self._ctx, subgraph)

                #   Pass parameters to the subgraph by forwarding the backend symbol names
                #   of their source symbols to the subgraphs' Top symbol table

                subgraph_top_stable = subgraph_info.node_infos[Top()].sym_table
                for sym in node_info.node.free_symbols:
                    subgraph_top_stable.add_mapping(
                        sym.name, node_info.sym_table.get_local_name(sym)
                    )

                subgraph_block = self._process_subgraph(subgraph, subgraph_info)

                #   Pass exports out of the subgraph by setting the exported symbol's
                #   backend names for their destination symbols in the current node's symbol table

                subgraph_bottom_stable = subgraph_info.node_infos[
                    subgraph.bottom
                ].sym_table

                for export in node_info.node.exports:
                    node_info.sym_table.add_mapping(
                        export.name, subgraph_bottom_stable.get_local_name(export)
                    )

                node_info.ast = subgraph_block

            case Cases():
                freeze = FreezeExpressions(self._ctx, node_info.sym_table)

                #   Create backend symbols for all exports
                export_symbols = [
                    node_info.sym_table.declare_symbol(export.name)
                    for export in node_info.node.exports
                ]

                subgraph_blocks: list[PsBlock] = []
                for subgraph in node_info.node.subgraphs:
                    #   Freeze subgraphs while forwarding free symbol mappings
                    subgraph_info = FlowgraphInfo(self._ctx, subgraph)

                    subgraph_top_stable = subgraph_info.node_infos[Top()].sym_table
                    for sym in node_info.node.free_symbols:
                        subgraph_top_stable.add_mapping(
                            sym.name, node_info.sym_table.get_local_name(sym)
                        )

                    subgraph_block = self._process_subgraph(subgraph, subgraph_info)

                    #   Forward exported symbols from subgraph
                    bottom_symtable = subgraph_info.node_infos[
                        subgraph.bottom
                    ].sym_table
                    for sym in subgraph.exports:
                        sym_in_subgraph = bottom_symtable.get_local_symbol(sym.name)
                        subgraph_block.statements.append(
                            self._typify(
                                PsAssignment(
                                    freeze(sym), PsExpression.make(sym_in_subgraph)
                                )
                            )
                        )

                    subgraph_blocks.append(subgraph_block)

                #   Types of exported symbols are now known from the subgraphs
                #   -> create their declarations to undefined values

                export_decls = [
                    PsDeclaration(PsExpression.make(sym), PsUndefined(sym.get_dtype()))
                    for sym in export_symbols
                ]

                conditions = node_info.node.conditions

                #   Assemble nested conditionals
                ast: PsBlock | None

                if node_info.node.is_complete:
                    ast = subgraph_blocks[-1]
                    subgraph_blocks = subgraph_blocks[:-1]
                    conditions = conditions[:-1]
                else:
                    ast = None

                for cond, sblock in zip(conditions[::-1], subgraph_blocks[::-1]):
                    ast = PsBlock([PsConditional(freeze(cond), sblock, ast)])

                assert ast is not None

                ast.statements = export_decls + ast.statements

                node_info.ast = ast

            case _:
                raise NotImplementedError(
                    f"Don't know how to freeze flowgraph node {type(node_info.node)}"
                )

    def _typify_node(self, node_info: NodeInfo):
        if node_info.ast:
            node_info.ast = self._typify(node_info.ast)

    def _linearize_graph(self, graph: FlowgraphInfo) -> PsBlock:
        all_statements: list[PsStructuralNode] = []

        for node in graph.nodes_linearized:
            node_info = graph.node_infos[node]
            if not isinstance(node, Top | Bottom):
                assert node_info.ast is not None
                node_type = type(node_info.node).__name__
                pre_comment = PsComment(f"(begin {node_type} {node_info.node.name})")
                post_comment = PsComment(f"(end {node_type} {node_info.node.name})")
                all_statements += (
                    [pre_comment] + node_info.ast.statements + [post_comment]
                )

        return PsBlock(all_statements)
