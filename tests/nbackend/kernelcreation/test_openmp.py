import pytest
from pystencils import (
    fields,
    Assignment,
    create_kernel,
    CreateKernelConfig,
    Target,
)

from pystencils.backend.ast import dfs_preorder
from pystencils.backend.ast.structural import PsPragma


@pytest.mark.parametrize("num_threads", [None, 1, 2, 4, 8])
@pytest.mark.parametrize("schedule", ["static", "static,16", "dynamic", "auto"])
@pytest.mark.parametrize("collapse", [None, 1, 2])
def test_openmp(num_threads, schedule, collapse):
    f, g = fields("f, g: [3D]")
    asm = Assignment(f.center(0), g.center(0))

    gen_config = CreateKernelConfig(target=Target.CPU)
    gen_config.cpu.openmp.enable = True
    gen_config.cpu.openmp.num_threads = num_threads
    gen_config.cpu.openmp.schedule = schedule
    gen_config.cpu.openmp.collapse = collapse

    kernel = create_kernel(asm, gen_config)
    ast = kernel.body

    parallel_pragma, for_pragma = [node for node in dfs_preorder(ast) if isinstance(node, PsPragma)]

    expected_tokens = {"omp", "parallel"}

    if num_threads is not None:
        expected_tokens.add(f"num_threads({num_threads})")

    assert set(parallel_pragma.text.split()) == expected_tokens

    expected_tokens = {"omp", "for", f"schedule({schedule})"}
    
    if collapse is not None:
        expected_tokens.add(f"collapse({collapse})")

    assert set(for_pragma.text.split()) == expected_tokens
