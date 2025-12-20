from __future__ import annotations

from typing import Sequence, cast
from abc import ABC

from ..memory import PsSymbol
from ..functions import GpuGridDimension
from .astnode import PsAstNode, PsAstNodeChildrenMixin
from .structural import PsStructuralNode, PsBlock, PsDeclaresSymbolTrait
from .expressions import PsExpression, PsSymbolExpr
from .util import failing_cast


class PsAxisRange(PsAstNodeChildrenMixin, PsDeclaresSymbolTrait, PsAstNode):
    """Iteration range of an axis."""

    __match_args__ = ("counter", "start", "stop", "step")
    _ast_children = (
        ("counter", PsSymbolExpr),
        ("start", PsExpression),
        ("stop", PsExpression),
        ("step", PsExpression),
    )

    def __init__(
        self,
        counter: PsSymbolExpr,
        start: PsExpression,
        stop: PsExpression,
        step: PsExpression,
    ):
        self._counter = counter
        self._start = start
        self._stop = stop
        self._step = step

    @property
    def counter(self) -> PsSymbolExpr:
        return self._counter

    @counter.setter
    def counter(self, expr: PsSymbolExpr):
        self._counter = expr

    @property
    def declared_symbol(self) -> PsSymbol:
        return self._counter.symbol

    @property
    def start(self) -> PsExpression:
        return self._start

    @start.setter
    def start(self, expr: PsExpression):
        self._start = expr

    @property
    def stop(self) -> PsExpression:
        return self._stop

    @stop.setter
    def stop(self, expr: PsExpression):
        self._stop = expr

    @property
    def step(self) -> PsExpression:
        return self._step

    @step.setter
    def step(self, expr: PsExpression):
        self._step = expr

    def clone(self) -> PsAxisRange:
        return PsAxisRange(
            cast(PsSymbolExpr, self._counter.clone()),
            self._start.clone(),
            self._stop.clone(),
            self._step.clone(),
        )


class PsAxesCube(PsStructuralNode):
    """Hypercube of multiple iteration ranges.

    .. note::
        The ranges of the axis cube are evaluated in order.
        Start, stop and step expressions of later ranges may depend
        on the counter variables of ranges listed before.
    """

    __match_args__ = ("ranges", "body")

    def __init__(self, ranges: Sequence[PsAxisRange], body: PsBlock):
        self._ranges = list(ranges)
        self._body = body

    @property
    def ranges(self) -> list[PsAxisRange]:
        return self._ranges

    @property
    def body(self) -> PsBlock:
        return self._body

    def get_children(self) -> tuple[PsAstNode, ...]:
        return tuple(self._ranges) + (self._body,)

    def set_child(self, idx: int, c: PsAstNode):
        idx = range(len(self._ranges) + 1)[idx]
        match idx:
            case _ if idx < len(self._ranges):
                self._ranges[idx] = failing_cast(PsAxisRange, c)
            case _ if idx == len(self._ranges):
                self._body = failing_cast(PsBlock, c)
            case _:
                assert False

    def _clone_structural(self):
        return PsAxesCube(
            [rang.clone() for rang in self._ranges], self._body._clone_structural()
        )


class PsIterationAxis(PsAstNodeChildrenMixin, PsStructuralNode):
    """Common base class for iteration axes."""

    __match_args__ = ("range", "body")
    _ast_children = (("range", PsAxisRange), ("body", PsBlock))

    can_hoist: bool = False
    """Whether or not this axis type is subject to iteration-invariant code motion.

    If set to `False`, the `HoistIterationInvariantDeclarations` pass will ignore this axis type.
    """

    def __init__(self, range: PsAxisRange, body: PsBlock):
        self._range = range
        self._body = body

    @property
    def range(self) -> PsAxisRange:
        return self._range

    @range.setter
    def range(self, r: PsAxisRange):
        self._range = r

    @property
    def body(self) -> PsBlock:
        return self._body

    @body.setter
    def body(self, block: PsBlock):
        self._body = block


class PsLoopAxis(PsIterationAxis):
    """Loop axis.

    Will be lowered to a `for-loop <PsLoop>`.
    """

    can_hoist: bool = True

    def _clone_structural(self) -> PsStructuralNode:
        return PsLoopAxis(self._range.clone(), self._body._clone_structural())


class PsSimdAxis(PsIterationAxis):
    """Axis modelling a SIMD block.

    A SIMD axis with k lanes translates to k iterations unrolled and packed
    into vectorized operations.

    The range of a SIMD axis is interpreted as follows:
    - ``counter`` is the scalar counter symbol that will be vectorized
    - ``start`` is the value of ``counter`` on the first SIMD lane
    - ``step`` is the counter's stride
    - ``stop`` must be equal to ``start + lanes * step``
    """

    __match_args__ = ("lanes", "range", "body")

    can_hoist: bool = True

    def __init__(self, lanes: int, range: PsAxisRange, body: PsBlock):
        self._lanes = lanes
        super().__init__(range, body)

    @property
    def lanes(self) -> int:
        return self._lanes

    @lanes.setter
    def lanes(self, n: int):
        self._lanes = n

    def _clone_structural(self) -> PsStructuralNode:
        return PsSimdAxis(
            self._lanes, self._range.clone(), self._body._clone_structural()
        )


class PsParallelLoopAxis(PsIterationAxis):
    """Parallel loop axis.

    Will be lowered to a loop parallelized using OpenMP directives.
    """

    can_hoist: bool = True

    def __init__(
        self,
        range: PsAxisRange,
        body: PsBlock,
        *,
        num_threads: int | None = None,
        schedule: str | None = None,
        collapse: int | None = None,
    ):
        super().__init__(range, body)
        self._num_threads = num_threads
        self._schedule = schedule
        self._collapse = collapse

    @property
    def num_threads(self) -> int | None:
        return self._num_threads

    @num_threads.setter
    def num_threads(self, num_threads: int | None):
        self._num_threads = num_threads

    @property
    def schedule(self) -> str | None:
        return self._schedule

    @schedule.setter
    def schedule(self, schedule: str | None):
        self._schedule = schedule

    @property
    def collapse(self) -> int | None:
        return self._collapse

    @collapse.setter
    def collapse(self, collapse: int | None):
        self._collapse = collapse

    def _clone_structural(self) -> PsStructuralNode:
        return PsParallelLoopAxis(
            self._range.clone(),
            self._body._clone_structural(),
            num_threads=self._num_threads,
            schedule=self._schedule,
            collapse=self._collapse,
        )


class PsGpuIndexingAxis(PsIterationAxis, ABC):
    """Common base class for GPU block+thread indexing axes."""

    __match_args__ = ("gpu_dimension", "range", "body")

    #   Mustn't perform code motion outside of GPU axes,
    #   as this would a) move code outside of the thread index guard, executing it unnecessarily
    #   and b) would introduce code fragments between nested GPU axes, s.t. `MaterializeAxes`
    #   can no longer combine their guards
    can_hoist: bool = False

    def __init__(self, gpu_dimension: GpuGridDimension, xrange: PsAxisRange, body: PsBlock):
        super().__init__(xrange, body)
        self._gpu_dimension = gpu_dimension

    @property
    def gpu_dimension(self) -> GpuGridDimension:
        return self._gpu_dimension

    @gpu_dimension.setter
    def gpu_dimension(self, d: GpuGridDimension):
        self._gpu_dimension = d

    def _clone_structural(self) -> PsStructuralNode:
        return type(self)(
            self._gpu_dimension, self._range.clone(), self._body._clone_structural()
        )


class PsGpuBlockAxis(PsGpuIndexingAxis):
    """Gpu block axis.

    Set the axis counter as an affine expression of the GPU block index
    in the given dimension; i.e. ``ctr = start + step * blockIdx.{c}``.
    Only execute the body if ``ctr < stop``.
    """


class PsGpuThreadAxis(PsGpuIndexingAxis):
    """Gpu thread axis.

    Set the axis counter as an affine expression of the block-local GPU thread index
    in the given dimension; i.e. ``ctr = start + step * threadIdx.{c}``.
    Only execute the body if ``ctr < stop``.
    """


class PsGpuBlockXThreadAxis(PsGpuIndexingAxis):
    """Gpu block x thread axis.

    Set the axis counter as an affine expression of the global GPU thread index
    in the given dimension; i.e. ``ctr = start + step * (blockIdx.{c} * blockDim.{c} + threadIdx.{c})``.
    Only execute the body if ``ctr < stop``.
    """
