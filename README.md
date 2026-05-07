# pystencils

[![Docs](https://img.shields.io/badge/read-the_docs-brightgreen.svg)](https://pycodegen.pages.i10git.cs.fau.de/pystencils)
[![pypi-package](https://badge.fury.io/py/pystencils.svg)](https://badge.fury.io/py/pystencils)
[![pipeline status](https://i10git.cs.fau.de/pycodegen/pystencils/badges/master/pipeline.svg)](https://i10git.cs.fau.de/pycodegen/pystencils/commits/master)
[![coverage report](https://i10git.cs.fau.de/pycodegen/pystencils/badges/master/coverage.svg)](http://pycodegen.pages.i10git.cs.fau.de/pystencils/coverage_report)

Pystencils is a symbolic domain-specific language and metaprogramming toolkit
for writing high-performance numerical stencil kernels for a variety of hardware targets.

> [!note]
> This is the code of repository of *pystencils 2.x*, the near-complete rework of the pystencils package.
> The legacy package *pystencils 1.4* is being supported at the [`v1.x`](https://i10git.cs.fau.de/pycodegen/pystencils/-/tree/v1.x) branch of this repository.

## Installation

Pystencils can be installed from PyPI using `pip`, e.g.:

```bash
pip install pystencils~=2.0
```

For more information on installing *pystencils*, refer to the [Installation Guide](https://pycodegen.pages.i10git.cs.fau.de/pystencils/installation.html).

## Example

Here is a code snippet that computes the average of neighboring cells:
```python
import pystencils as ps
import numpy as np

f, g = ps.fields("f, g : [2D]")
stencil = ps.Assignment(g[0, 0],
                       (f[1, 0] + f[-1, 0] + f[0, 1] + f[0, -1]) / 4)
kernel = ps.create_kernel(stencil).compile()

f_arr = np.random.rand(1000, 1000)
g_arr = np.empty_like(f_arr)
kernel(f=f_arr, g=g_arr)
```

*pystencils* is mostly used for numerical simulations using finite difference or finite volume methods.
It comes with automatic finite difference discretization for PDEs:

```python
import pystencils as ps
import sympy as sp

c, v = ps.fields("c, v(2): [2D]")
adv_diff_pde = ps.fd.transient(c) - ps.fd.diffusion(c, sp.symbols("D")) + ps.fd.advection(c, v)
discretize = ps.fd.Discretization2ndOrder(dx=1, dt=0.01)
discretization = discretize(adv_diff_pde)
```

## Documentation

Here's an overview of our documentation ressources:

 - [pystencils User Manual (`master`)](https://pycodegen.pages.i10git.cs.fau.de/pystencils)
 - [pycodegen Index Page](https://pycodegen.pages.i10git.cs.fau.de/)
 
## Contributing

Please refer to [the contribution guide](https://pycodegen.pages.i10git.cs.fau.de/pystencils/contributing)
for instructions on how to start contributing to *pystencils*.

## Authors

Many thanks go to the [contributors](CITATION.cff) of pystencils.

### Please cite us

If you use pystencils in a publication, please cite the following articles:

Overview:
  - M. Bauer et al, Code Generation for Massively Parallel Phase-Field Simulations. Association for Computing Machinery, 2019. https://doi.org/10.1145/3295500.3356186

Performance Modelling:
  - D. Ernst et al, Analytical performance estimation during code generation on modern GPUs. Journal of Parallel and Distributed Computing, 2023. https://doi.org/10.1016/j.jpdc.2022.11.003
