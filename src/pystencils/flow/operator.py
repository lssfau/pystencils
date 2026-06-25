from __future__ import annotations

from typing import overload, Callable, Iterable, TYPE_CHECKING, cast
from dataclasses import replace

import inspect

from .flowgraph import Flowgraph, FlowgraphNode
from .builders import EquationsBlockBuilder, block, tie
from ..codegen.config import CreateKernelConfig
from ..grids.patch_data import PatchData

if TYPE_CHECKING:
    from ..codegen import Kernel
    from ..jit import KernelWrapper


class Operator:
    def __init__(
        self,
        graph: FlowgraphNode | Flowgraph,
        *,
        config: CreateKernelConfig | None = None,
    ):
        if not isinstance(graph, Flowgraph):
            graph = tie(graph)

        self._graph = graph
        self._cfg = CreateKernelConfig() if config is None else config
        self._ker: Kernel | None = None
        self._func: KernelWrapper | None = None

        self._cfg.function_name = graph.name

    @property
    def graph(self) -> Flowgraph:
        return self._graph

    @property
    def config(self) -> CreateKernelConfig:
        if self._ker is not None:
            raise AttributeError(
                "Code generator configuration for this operator is inaccessible: Code was already generated.\n"
                "To modify the configuration and re-generate the code, call `clear()` first."
            )
        return self._cfg

    @config.setter
    def config(self, cfg: CreateKernelConfig):
        if self._ker is not None:
            raise AttributeError(
                "Code generator configuration for this operator cannot be set: Code was already generated.\n"
                "To modify the configuration and re-generate the code, call `clear()` first."
            )
        self._cfg = cfg

    @property
    def kernel(self) -> Kernel | None:
        """Kernel IR of this operator"""
        return self._ker

    @property
    def func(self) -> KernelWrapper | None:
        """Executable kernel function of this operator"""
        return self._func

    def generate_code(self):
        """Generate the intermediate representation of this operator's code.

        Raises:
            RuntimeError: If code was already generated
        """
        if self._ker is not None:
            raise RuntimeError("Code was already generated.")

        from ..codegen import create_kernel

        self._ker = create_kernel(self._graph, self._cfg)

    def clear(self):
        """Clear the generated IR and compiled module, if present."""
        self._ker = None
        self._func = None

    def compile_code(self):
        """Compile this operator's executable module.

        Internally calls `generate_code` if code was not already generated.

        Raises:
            RuntimeError: If the code was already compiled
        """
        if self._func is not None:
            raise RuntimeError("Code was already compiled.")

        if self._ker is None:
            self.generate_code()

        assert self._ker is not None
        self._func = self._ker.compile()

    def clear_compiled_code(self):
        """Clear the compiled code module. Does nothing if no compiled module is attached."""
        self._func = None

    def __call__(self, *pdata: PatchData, **kwargs):
        if self._func is None:
            self.compile_code()

        assert self._func is not None

        args = dict()
        for pd in pdata:
            args.update(pd.args)

        args.update(**kwargs)

        self._func(**args)

    def __str__(self) -> str:
        return str(self._graph)

    def _repr_markdown_(self) -> str:
        return f"```\n{str(self)}\n```"


_SingleBlockOperatorFunc = Callable[[EquationsBlockBuilder], None]
_ComplexGraphOperatorFunc = Callable[[], FlowgraphNode | tuple[FlowgraphNode, ...]]


@overload
def operator(
    *,
    preds: Iterable[FlowgraphNode] | None = None,
    name: str | None = None,
    config: CreateKernelConfig | None = None,
    **kwargs,
) -> Callable[
    [FlowgraphNode | _SingleBlockOperatorFunc | _ComplexGraphOperatorFunc],
    Operator,
]: ...  # noqa: E704


@overload
def operator(
    func: _SingleBlockOperatorFunc | _ComplexGraphOperatorFunc,
    /,
) -> Operator: ...  # noqa: E704


def operator(
    func: _SingleBlockOperatorFunc | _ComplexGraphOperatorFunc | None = None,
    /,
    config: CreateKernelConfig | None = None,
    preds: Iterable[FlowgraphNode] | None = None,
    name: str | None = None,
    **kwargs,
):
    if config is None:
        config = CreateKernelConfig()
    if kwargs:
        config = replace(config, **kwargs)

    def decorate(
        func: (
            Callable[[], FlowgraphNode | tuple[FlowgraphNode, ...]]
            | Callable[[EquationsBlockBuilder], None]
        ),
    ) -> Operator:
        nodes: tuple[FlowgraphNode, ...]
        op_name = name if name is not None else func.__name__

        if isinstance(func, FlowgraphNode):
            nodes = (func,)
        else:
            params = inspect.signature(func).parameters
            if params:
                func = cast(Callable[[EquationsBlockBuilder], None], func)
                graph = block(preds=preds, name=op_name)(func)
                nodes = (graph,)
            else:
                func = cast(
                    Callable[[], FlowgraphNode | tuple[FlowgraphNode, ...]], func
                )
                outp_nodes = func()
                if isinstance(outp_nodes, tuple):
                    nodes = outp_nodes
                else:
                    nodes = (outp_nodes,)

        return Operator(tie(*nodes, name=op_name), config=config)

    if func is None:
        return decorate
    else:
        return decorate(func)
