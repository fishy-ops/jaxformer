"""Build/load the custom CUDA attention kernels via torch.utils.cpp_extension.

JIT-compiled rather than packaged: the extension is tiny and this keeps iteration to
"edit .cu, re-run", with torch caching the build. The split of pybind (.cpp, compiled
by cl) from the kernel (.cu, compiled by nvcc) is deliberate and load-bearing on
Windows — see docs/windows_gpu_setup.md.

The environment (CUDA_HOME, vcvars, PATH) must already be set up; on the RTX 2070 box
that is what scripts/jf_env.bat does before invoking Python.
"""

from __future__ import annotations

import functools
import os

_CSRC = os.path.join(os.path.dirname(__file__), "csrc")


@functools.lru_cache(maxsize=None)
def load_v1(verbose: bool = False):
    """Compile (once per process) and return the v1 fp32 kernel module."""
    from torch.utils.cpp_extension import load

    return load(
        name="jf_fused_attn_v1",
        sources=[
            os.path.join(_CSRC, "bindings.cpp"),
            os.path.join(_CSRC, "fused_attn_v1.cu"),
        ],
        extra_cflags=["-O3"],
        # Turing is sm_75. Line info makes Nsight Compute source-correlate the kernel
        # in Phase 6 without the overhead of full device debug (-G).
        extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_75,code=sm_75"],
        verbose=verbose,
    )
