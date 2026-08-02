"""Minimal, deterministic launch target for Nsight Compute to profile.

Nsight Compute (`ncu`) wraps a process and instruments its kernel launches. This
script does the least that produces a representative launch: build the v1 kernel, then
issue a fixed number of identical forward passes at one shape. Warmups let the profiler
skip cold launches (``ncu --launch-skip``) and capture a single steady-state one
(``--launch-count 1``), so the report reflects the kernel's real behavior rather than
first-launch or allocator noise.

Shape and causality come from the environment so the same script can profile different
regimes without editing:

    PROFILE_T=2048 PROFILE_CAUSAL=1  ncu ... python -m bench.profile_target
"""

from __future__ import annotations

import os

import torch

from kernels.build import load_v1


def main() -> None:
    assert torch.cuda.is_available(), "profiling target requires CUDA"
    kernel = load_v1(verbose=False)

    B = int(os.environ.get("PROFILE_B", 4))
    H = int(os.environ.get("PROFILE_H", 8))
    T = int(os.environ.get("PROFILE_T", 2048))
    Dh = int(os.environ.get("PROFILE_DH", 64))
    causal = os.environ.get("PROFILE_CAUSAL", "1") == "1"
    warmup = int(os.environ.get("PROFILE_WARMUP", 12))

    torch.manual_seed(0)
    q = torch.randn(B, H, T, Dh, device="cuda")
    k = torch.randn(B, H, T, Dh, device="cuda")
    v = torch.randn(B, H, T, Dh, device="cuda")

    # Warmups (skipped by the profiler) then measured launches. ncu selects which one
    # to capture via --launch-skip/--launch-count; extra launches are harmless.
    for _ in range(warmup + 3):
        _ = kernel.forward(q, k, v, causal)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
