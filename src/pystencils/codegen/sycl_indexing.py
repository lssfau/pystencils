
from ..backend.ast.expressions import PsExpression
from ..backend.kernelcreation import FullIterationSpace, KernelCreationContext, SparseIterationSpace
from .config import GpuIndexingScheme
from .errors import CodegenError
from .gpu_indexing import BaseIndexing


class SyclIndexing(BaseIndexing):
    def __init__(self,
                 ctx: KernelCreationContext,
                 scheme: GpuIndexingScheme,
                 warp_size: int | None,
                 manual_launch_grid: bool = False,
                 assume_warp_aligned_block_size: bool = False,
                 ) -> None:

        super().__init__(ctx,
                         scheme,
                         warp_size,
                         manual_launch_grid,
                         assume_warp_aligned_block_size,
                         None)

    def _get_work_items(self) -> tuple[PsExpression, ...]:
        """Return a tuple of expressions representing the number of work items
        in each dimension of the kernel's iteration space,
        ordered from slowest to fastes dimension (according to the sycl order).
        """
        ispace = self._ctx.get_iteration_space()
        match ispace:
            case FullIterationSpace():
                # do not invert
                dimensions = ispace.dimensions_in_loop_order()

                from ..backend.ast.analysis import collect_undefined_symbols as collect

                for i, dim in enumerate(dimensions):
                    symbs = collect(dim.start) | collect(dim.stop) | collect(dim.step)
                    for ctr in ispace.counters:
                        if ctr in symbs:
                            raise CodegenError(
                                "Unable to construct GPU launch grid constraints for this kernel: "
                                f"Limits in dimension {i} "
                                f"depend on another dimension's counter {ctr.name}"
                            )

                return tuple(ispace.actual_iterations(dim) for dim in dimensions)

            case SparseIterationSpace():
                return (self._ast_factory.parse_index(ispace.index_list.shape[0]),)

            case _:
                assert False, "unexpected iteration space"
