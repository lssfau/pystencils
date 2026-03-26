import subprocess
from dataclasses import dataclass, field

from ..cpu.compiler_info import ClangInfo
from ..jit import JitError
from ...codegen.target import Target


def _run_rocminfo():
    try:
        return subprocess.run(
            ["rocminfo"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout
    except FileNotFoundError:
        raise RuntimeError("rocminfo not found (ROCm not installed or not in PATH)")
    except subprocess.CalledProcessError:
        raise RuntimeError("rocminfo failed to run")


def _get_rocm_agents(rocminfo_output):

    agents = []

    cur_agent = dict()
    for line in rocminfo_output.split("\n"):
        stripped_line = line.strip()
        if stripped_line.startswith("Agent") and len(cur_agent) > 0:
            agents.append(cur_agent)
            cur_agent = dict()
        elif stripped_line.startswith("Name:"):
            if "name" in cur_agent:
                continue
            splitted = stripped_line.split(":")
            cur_agent.update({"name": splitted[-1].strip()})
        elif stripped_line.startswith("Marketing Name:"):
            splitted = stripped_line.split(":")
            cur_agent.update({"marketingname": splitted[-1].strip()})
    agents.append(cur_agent)

    return agents


def _get_amd_gfx_architecture(device_name: str) -> str:
    rocminfo_output = _run_rocminfo()
    agents = _get_rocm_agents(rocminfo_output)
    for agent in agents:
        if agent["marketingname"] == device_name:
            return agent["name"]
    raise JitError(f"Could not find gf architecture for {device_name}")


@dataclass
class SYCLClangInfo(ClangInfo):
    """Compiler info for the SYCL/oneAPI Clang compiler."""

    target: Target = Target.SYCL

    sycl_targets: list[str] = field(default_factory=list)
    """
    Arguments for *-fsycl-targets* can be *nvptx64-nvidia-cuda*, *amdgcn-amd-amdhsa*, and *spir64*
    """
    amd_offload_architecutres: list[str] = field(default_factory=list)
    """
    For targeting the AMD/HIP backend, the SYCL compiler needs the exact architecture specifier
    https://developer.codeplay.com/products/oneapi/amd/2025.2.0/guides/get-started-guide-amd.html#use-dpc-to-target-amd-gpus
    """

    def _figure_gpu_targets(self):
        import dpctl
        gpu_targets = set()

        # todo hip
        for dev in dpctl.get_devices():
            if dev.backend.name.lower() == 'cuda':
                gpu_targets.add("nvptx64-nvidia-cuda")
            elif dev.backend.name.lower() == 'hip':
                gpu_targets.add("amdgcn-amd-amdhsa")
                self.amd_offload_architecutres.append(_get_amd_gfx_architecture(dev.name))
        self.sycl_targets = [*gpu_targets]

    def _get_gpu_flags(self) -> list[str]:
        if self.amd_offload_architecutres and "amdgcn-amd-amdhsa" not in self.sycl_targets:
            self.sycl_targets.append("amdgcn-amd-amdhsa")
        # this is the spir64 target for some versions of icpx/the codeplay plugins
        # it was importatnt to have this at the last position
        targets = [*set(self.sycl_targets), "spir64"]
        flags = [f"-fsycl-targets={','.join(targets)}"]

        if len(self.amd_offload_architecutres) == 1:
            flags.append("-Xsycl-target-backend=amdgcn-amd-amdhsa")
            flags.extend(f"--offload-arch={self.amd_offload_architecutres[0]}")
        if len(self.amd_offload_architecutres) > 1:
            raise JitError(
                "Current Codeplay plugins support only one AMD GPU Target Architecture"
                " use ONEAPI_DEVICE_FILTER to hide unwanted devices")
        return flags

    def cxxflags(self):
        if not self.sycl_targets:
            self._figure_gpu_targets()

        flags = ["-fsycl", "-fno-sycl-dead-args-optimization"]
        flags += self._get_gpu_flags()
        flags += super().cxxflags()

        return flags


@dataclass
class SYCLIcpxInfo(SYCLClangInfo):
    """Compiler info for the oneAPI Icpx compiler."""

    def cxx(self) -> str:
        return "icpx"

    def cxxflags(self):
        flags = super().cxxflags()
        if self.openmp:
            flags.remove("-fopenmp")
            flags += ["-qopenmp"]
        return flags
