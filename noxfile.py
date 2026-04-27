from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Sequence

import nox

nox.options.sessions = ["lint", "typecheck", "testsuite"]


def get_cuda_version(session: nox.Session) -> None | tuple[int, ...]:
    query_args = ["nvcc", "--version"]

    try:
        query_result = subprocess.run(query_args, capture_output=True)
    except FileNotFoundError:
        return None

    matches = re.findall(r"release \d+\.\d+", str(query_result.stdout))
    if matches:
        match = matches[0]
        version_string = match.split()[-1]
        try:
            return tuple(int(v) for v in version_string.split("."))
        except ValueError:
            pass

    session.warn("nvcc was found, but I am unable to determine the CUDA version.")
    return None


def install_cupy(
    session: nox.Session, cupy_version: str, skip_if_no_cuda: bool = False
):
    if cupy_version is not None:
        cuda_version = get_cuda_version(session)
        if cuda_version is None or cuda_version[0] not in (11, 12):
            if skip_if_no_cuda:
                session.skip(
                    "No compatible installation of CUDA found - Need either CUDA 11 or 12"
                )
            else:
                session.warn(
                    "Running without cupy: no compatbile installation of CUDA found. Need either CUDA 11 or 12."
                )
                return

        cuda_major = cuda_version[0]
        cupy_package = f"cupy-cuda{cuda_major}x=={cupy_version}"
        session.install(cupy_package)


def get_dpcpp_version(session: nox.Session)  -> None | tuple[int, ...]:
    query_args = ["icpx", "--version"]
    try:
        query_result = subprocess.run(query_args, capture_output=True)
    except FileNotFoundError:
        return None
    match = re.search(r"\b\d+\.\d+\.\d+\b", str(query_result.stdout))
    if match:
        version_string = match.group(0).split()[-1]
        try:
            return tuple(int(v) for v in version_string.split("."))
        except ValueError:
            pass

    session.warn("icpx was found, but I am unable to determine the icpx version.")
    return None


def install_dpctl(session: nox.Session, skip_if_no_dpcpp: bool = False):
    dpcpp_version = get_dpcpp_version(session)
    dpctl_version = ""
    if dpcpp_version:
        session.log(f"Found dpctl version: {dpcpp_version}")
        # session.install(f"intel-sycl-rt=={dpcpp_version[0]}.{dpcpp_version[1]}.{dpcpp_version[2]}")
        if dpcpp_version[0] == 2025 and dpcpp_version[1] <= 2:
            # there was a change in the rt packages with 2025.2, so this the one to use, it works also with oAPI 2025.0
            dpcpp_version = (2025, 2, 0)
            dpctl_version = "0.20.2"
        session.install(f"intel-sycl-rt=={dpcpp_version[0]}.{dpcpp_version[1]}.{dpcpp_version[2]}")
    else:
        if skip_if_no_dpcpp:
            session.skip(
                "No compatible installation of Data-Parallel C++ found."
            )
        else:
            session.warn(
                "Running without dpctl: no installation of dpcpp found."
            )
            return

    if dpctl_version:
        session.install(f"dpctl=={dpctl_version}")
    else:
        session.install("dpctl")
    env = session.env.copy()
    try:
        session.run("python", "-c", "import dpctl; assert(len(dpctl.get_devices()) > 0)")
    except Exception:
        venv_lib = Path(session.virtualenv.location) / "lib"
        print(f"Adding {venv_lib} to LD_LIBRARY_PATH")
        session.env['LD_LIBRARY_PATH'] = f"{venv_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        session.env['LIBRARY_PATH'] = f"{venv_lib}:{env.get('LIBRARY_PATH', '')}"
        session.run("python", "-c", "import dpctl; assert(len(dpctl.get_devices()) > 0)")


def check_external_doc_dependencies(session: nox.Session):
    dot_args = ["dot", "--version"]
    try:
        _ = subprocess.run(dot_args, capture_output=True)
    except FileNotFoundError:
        session.error(
            "Unable to build documentation: "
            "Command `dot` from the `graphviz` package (https://www.graphviz.org/) is not available"
        )


def editable_install(session: nox.Session, opts: Sequence[str] = ()):
    if opts:
        opts_str = "[" + ",".join(opts) + "]"
    else:
        opts_str = ""
    session.install("-e", f".{opts_str}")


@nox.session(python="3.10", tags=["qa", "code-quality"])
def lint(session: nox.Session):
    """Lint code using flake8"""

    session.install("flake8")
    session.run("flake8", "src/pystencils")


@nox.session(python="3.10", tags=["qa", "code-quality"])
def typecheck(session: nox.Session):
    """Run MyPy for static type checking"""
    editable_install(session)
    session.install("mypy")
    session.run("mypy", "src/pystencils")


@nox.session(python=["3.10", "3.11", "3.12", "3.13"], tags=["test"])
@nox.parametrize("device_interface", ["cpu", "cupy12", "cupy13", "dpctl"], ids=["cpu", "cupy12", "cupy13", "dpctl"])
def testsuite(session: nox.Session, device_interface: str):
    """Run the pystencils test suite.

    **Positional Arguments:** Any positional arguments passed to nox after `--`
    are propagated to pytest.
    """

    if device_interface.startswith("cupy"):
        cupy_version = device_interface.removeprefix("cupy")
        install_cupy(session, cupy_version, skip_if_no_cuda=True)

    if device_interface == "dpctl":
        install_dpctl(session, skip_if_no_dpcpp=True)
        num_cores = 1
        session.run("python", "-c", "import dpctl; print(dpctl.get_devices())")
    else:
        num_cores = os.cpu_count()

    #   FIXME remove once https://github.com/bashtage/randomgen/issues/426 is resolved
    session.install("numpy<2.4")
    editable_install(session, ["alltrafos", "use_cython", "interactive", "testsuite"])

    session.run(
        "pytest",
        "-v",
        "-n",
        str(num_cores),
        "--cov",
        "-m",
        "not longrun",
        "--html",
        "test-report/index.html",
        "--junitxml=report.xml",
        *session.posargs,
    )


@nox.session
def coverage_report(session: nox.Session):
    session.install("coverage")

    session.run("coverage", "combine")
    session.run("coverage", "report", "--precision=2")
    session.run("coverage", "html")
    session.run("coverage", "xml")


@nox.session(tags=["minitest"])
@nox.parametrize("target", ["ARM_SVE"], ids=["SVE"])
def minitest_simd(session: nox.Session, target: str):
    """Run a reduced testsuite for only testing vectorization features for a specific target."""

    if session.venv_backend != "none":
        session.install(
            "pytest", "randomgen", "py-cpuinfo"
        )
        editable_install(session)

    test_files = [
        "tests/nbackend/test_vectorization.py",
    ]
    pytest_filter = (
        f"{target} and "
        "((test_update_kernel and 16bit) or "
        "(test_set and 32bit) or "
        "(test_strided_load and float and 32bit) or "
        "(test_strided_store and float and 64bit))"
    )

    session.run(
        "pytest",
        "-v",
        "-k",
        pytest_filter,
        *session.posargs,
        *test_files,
    )


@nox.session(python=["3.10"], tags=["docs"])
def docs(session: nox.Session):
    """Build the documentation pages"""
    check_external_doc_dependencies(session)
    install_cupy(session, "12.3")
    install_dpctl(session)
    editable_install(session, ["doc"])

    env = {}

    session_args = session.posargs
    if "--fail-on-warnings" in session_args:
        env["SPHINXOPTS"] = "-W --keep-going"

    session.chdir("docs")

    if "--clean" in session_args:
        session.run("make", "clean", external=True)

    session.run("make", "html", external=True, env=env)
