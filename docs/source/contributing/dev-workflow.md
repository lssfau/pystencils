# Development Workflow

This page contains instructions on how to get started with developing pystencils.

## Prepare the Git Repository

The pystencils Git repository is hosted at [i10git.cs.fau.de](https://i10git.cs.fau.de), the GitLab instance of the
[Chair for Systems Simulation](https://www.cs10.tf.fau.de/) at [FAU Erlangen-Nürnberg](https://fau.de).
In order to contribute code to pystencils, you will need to acquire an account there; to do so,
please follow the instructions on the GitLab landing page.

### Create a Fork

Only the core developers of pystencils have write-access to the primary repository.
To contribute, you will therefore have to create a fork of that repository
by navigating to the [repository page](https://i10git.cs.fau.de/pycodegen/pystencils)
and selecting *Fork* there.
In this fork, you may freely create branches and develop code, which may later be merged to a primary branch
via merge requests.

### Create a Local Clone

Once you have a fork of the repository, you can clone it to your local machine using the git command-line.

:::{note}
To clone via SSH, which is recommended, you will first have to [register an SSH key](https://docs.gitlab.com/ee/user/ssh.html).
:::

Open up a shell and navigate to a directory you want to work in.
Then, enter

```bash
git clone git@i10git.cs.fau.de:<your-username>/pystencils.git
```

to clone your fork of pystencils.

:::{note}
To keep up to date with the upstream repository, you can add it as a secondary remote to your clone:
```bash
git remote add upstream git@i10git.cs.fau.de:pycodegen/pystencils.git
```
You can point your clone's `master` branch at the upstream master like this:
```bash
git pull --set-upstream upstream master
```
:::

## Set Up the Python Environment

### Prerequesites

To develop pystencils, you will need at least the following software installed on your machine:

- Python 3.10 or later: Since pystencils minimal supported version is Python 3.10, we recommend that you work with Python 3.10 directly.
- An up-to-date C++ compiler, used by pystencils to JIT-compile generated code
- [Nox](https://nox.thea.codes/en/stable/), which we use for test automation.
  Nox will be used extensively in the instructions on testing below.
- Optionally, for GPU development:
  - At least CUDA 11 for Nvidia GPUs, or
  - At least ROCm/HIP 6.1 for AMD GPUs.

### Virtual Environment Setup

Once you have all the prerequesites,
set up a [virtual environment](https://docs.python.org/3/library/venv.html) for development.
This ensures that your system's installation of Python is kept clean, and isolates your development environment
from outside influence.
Use the following commands to create a virtual environment at `.venv` and perform an editable install of pystencils into it:

```bash
python -m venv .venv
source .venv/bin/activate
export PIP_REQUIRE_VIRTUALENV=true
pip install -e .[dev]
```

:::{note}
Setting `PIP_REQUIRE_VIRTUALENV` ensures that pip refuses to install packages globally --
Consider setting this variable globally in your shell's configuration file.
:::

:::{admonition} Feature Groups
The above installation instructions assume that you will be running all code checking
and test tasks through `nox`.
If you need or want to run them manually, you will need to add one or more
of these feature groups to your installation:

 - `doc`, which contains all dependencies required to build this documentation;
 - `dev`, which adds `flake8` for code style checking,
   `mypy` for static type checking,
    and the `black` formatter;
 - `testsuite`, which adds `pytest` plus plugins and some more dependencies required
   for running the test suite.

Depending on your development focus, you might also need to add some of the user feature
groups listed in [the installation guide](#installation_guide).
:::

### Cupy for CUDA and HIP

When developing for NVidia or AMD GPUs, you will likely need an installation of [cupy](https://cupy.dev/).
Since cupy has to be built specifically against the libraries of a given CUDA or ROCm version,
it cannot be installed directly via dependency resolution from pystencils.
For instructions on how to install Cupy, refer to their [installation manual](https://docs.cupy.dev/en/stable/install.html).

### Test Your Setup

To check if your setup is complete, a good check is to invoke the pystencils test suite:

```bash
nox -s "testsuite(cpu)"
```

If this finishes without errors, you are ready to go! Create a new git branch to work on, open up an IDE, and start coding.
Make sure your IDE recognizes the virtual environment you created, though.

## Static Code Analysis

### PEP8 Code Style

We use [flake8](https://github.com/PyCQA/flake8/tree/main) to check our code for compliance with the
[PEP8](https://peps.python.org/pep-0008/) code style.
You can either run `flake8` directly, or through Nox, to analyze your code with respect to style errors:

::::{grid}
:::{grid-item}
```bash
nox -s lint
```
:::
:::{grid-item}
```bash
flake8 src/pystencils
```
:::
::::

### Static Type Checking

New code added to pystencils is required to carry type annotations,
and its types are checked using [mypy](https://mypy.readthedocs.io/en/stable/index.html#).
To discover type errors, run *mypy* either directly or via Nox:

::::{grid}
:::{grid-item}
```bash
nox -s typecheck
```
:::
:::{grid-item}
```bash
mypy src/pystencils
```
:::
::::

:::{note}
Type checking is currently restricted only to a few modules, which are listed in the `mypy.ini` file.
Most code in the remaining modules is significantly older and is not comprehensively type-annotated.
As more modules are updated with type annotations, this list will expand in the future.
If you think a new module is ready to be type-checked, add an exception clause to `mypy.ini`.
:::

## Running the Test Suite

Pystencils comes with an extensive and steadily growing suite of unit tests.
To run the full testsuite, invoke the Nox `testsuite` session:

```bash
nox -s testsuite
```

:::{seealso}
[](#testing_pystencils)
:::


## Building the Documentation

The pystencils documentation pages are written in MyST Markdown and ReStructuredText,
located at the `docs` folder, and built using Sphinx.
To build the documentation pages of pystencils, simply run the `docs` Nox session:
```bash
nox -s docs
```

This will emit the generated HTML pages to `docs/build/html`.
The `docs` session permits two parameters to customize its execution:
 - `--clean`: Clean the page generator's output before building
 - `--fail-on-warnings`: Have the build fail (finish with a nonzero exit code) if Sphinx emits any warnings.

You must pass any of these to the session command *after a pair of dashes* (`--`); e.g.:
```bash
nox -s docs -- --clean
```
