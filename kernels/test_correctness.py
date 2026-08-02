"""Correctness of the custom CUDA attention kernel vs a float64 reference.

Requires a CUDA GPU and the assembled toolchain, so it is skipped anywhere else. Run
on the RTX 2070 box with the environment sourced:

    cmd /c "call C:\\jaxformer\\jf_env.bat && C:\\jfvenv\\Scripts\\python.exe -m pytest kernels\\test_correctness.py -v"

The reference computes attention densely in float64 (materializing the T x T scores —
fine at test sizes) and the kernel's fp32 streaming output is compared against it. The
tolerance is set for fp32 accumulation, not the reference: the interesting failures are
tail handling (sequence lengths that are not multiples of the 64-wide tile) and the
causal mask, both of which are exercised explicitly.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from kernels.build import load_v1  # noqa: E402


def reference_attention(q, k, v, causal):
    """Dense float64 attention. q,k,v: (B, H, T, Dh)."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
    if causal:
        T = q.shape[2]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", attn, v)


@pytest.fixture(scope="module")
def kernel():
    return load_v1(verbose=True)


def _max_err(B, H, T, causal, kernel, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H, T, 64, device="cuda")
    k = torch.randn(B, H, T, 64, device="cuda")
    v = torch.randn(B, H, T, 64, device="cuda")

    out = kernel.forward(q, k, v, causal)
    ref = reference_attention(q.double(), k.double(), v.double(), causal).float()
    return (out - ref).abs().max().item()


# Sequence lengths deliberately include non-multiples of the 64-wide tile (1, 7, 65,
# 100, 200) to catch ragged-tail bugs, plus exact multiples (64, 128, 256).
@pytest.mark.parametrize("T", [1, 7, 64, 65, 100, 128, 200, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_matches_reference(T, causal, kernel):
    err = _max_err(2, 3, T, causal, kernel)
    assert err < 2e-3, f"T={T} causal={causal}: max abs err {err:.2e}"


def test_single_batch_single_head(kernel):
    assert _max_err(1, 1, 129, True, kernel) < 2e-3


def test_first_query_attends_only_to_itself(kernel):
    """Under causal masking, row 0's output must equal V[...,0,:] exactly-ish:
    softmax over a single visible key is 1, so out[0] == v[0]."""
    torch.manual_seed(1)
    q = torch.randn(1, 2, 32, 64, device="cuda")
    k = torch.randn(1, 2, 32, 64, device="cuda")
    v = torch.randn(1, 2, 32, 64, device="cuda")
    out = kernel.forward(q, k, v, True)
    assert torch.allclose(out[:, :, 0], v[:, :, 0], atol=1e-4)


if __name__ == "__main__":
    k = load_v1(verbose=True)
    print("built. running a sweep...")
    for causal in (True, False):
        for T in (1, 7, 64, 65, 100, 128, 200, 256, 512, 1024):
            e = _max_err(2, 3, T, causal, k)
            flag = "ok" if e < 2e-3 else "FAIL"
            print(f"  [{flag}] T={T:5d} causal={causal!s:5}  max_abs_err={e:.2e}")
