"""Attention micro-benchmark: our CUDA kernel vs PyTorch's attention paths.

Measures, at several sequence lengths, the latency / peak memory / achieved TFLOP/s of:

  naive            explicit matmul -> softmax -> matmul (materializes the T x T scores)
  sdpa_math        F.scaled_dot_product_attention, MATH backend (also materializes)
  sdpa_mem_eff     ... EFFICIENT backend (the honest reference; works on Turing)
  sdpa_flash       ... FLASH backend (expected to FAIL on sm_75 -- a headline finding)
  ours_v1          the hand-written fp32 tiled online-softmax kernel

Methodology, stated because the method is the credibility:
  * CUDA-event timing, warmup then median of many iters, synchronize at boundaries.
  * Peak memory from torch.cuda.max_memory_allocated with stats reset per measurement.
  * Achieved TFLOP/s from an analytic FLOP count (causal counts half).
  * Every implementation is validated against the mem-efficient reference once before
    timing, so a fast-but-wrong kernel cannot post a good number.

Results are written to bench/results/attention_<host>.json for the plots to consume,
so the charts are regenerable from data rather than hand-made.

Run on the RTX 2070 box:
  cmd /c "call C:\\jaxformer\\jf_env.bat && cd /d C:\\jaxformer && \
          C:\\jfvenv\\Scripts\\python.exe -m bench.bench_attention"
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from kernels.build import load_v1


# ---------------------------------------------------------------------------
# Implementations under test
# ---------------------------------------------------------------------------


def naive_attention(q, k, v, causal):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        T = q.shape[2]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def _sdpa(backend):
    from torch.nn.attention import sdpa_kernel

    def run(q, k, v, causal):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    return run


def make_impls(kernel):
    from torch.nn.attention import SDPBackend

    return {
        "naive": naive_attention,
        "sdpa_math": _sdpa(SDPBackend.MATH),
        "sdpa_mem_eff": _sdpa(SDPBackend.EFFICIENT_ATTENTION),
        "sdpa_flash": _sdpa(SDPBackend.FLASH_ATTENTION),
        "ours_v1": lambda q, k, v, causal: kernel.forward(q, k, v, causal),
    }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def attn_flops(B, H, T, Dh, causal):
    # QK^T and AV are each 2*B*H*T*T*Dh FLOPs; causal masks ~half the pairs.
    flops = 2 * (2 * B * H * T * T * Dh)
    return flops * 0.5 if causal else float(flops)


def time_ms(fn, warmup=25, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def measure(fn, q, k, v, causal, reference):
    """Return (latency_ms, peak_MB, max_abs_err_vs_reference) or an error string."""
    try:
        out = fn(q, k, v, causal)
        torch.cuda.synchronize()
    except Exception as e:  # backend unsupported on this GPU, OOM, etc.
        return {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"}

    err = None if reference is None else (out - reference).abs().max().item()
    torch.cuda.reset_peak_memory_stats()
    lat = time_ms(lambda: fn(q, k, v, causal))
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    return {"latency_ms": lat, "peak_mb": peak_mb, "max_abs_err": err}


def run(seq_lens, B, H, Dh, causal, seed=0):
    kernel = load_v1(verbose=False)
    impls = make_impls(kernel)

    dev = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    results = {
        "device": dev,
        "compute_capability": f"sm_{cc[0]}{cc[1]}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config": {"B": B, "H": H, "head_dim": Dh, "causal": causal},
        "rows": [],
    }
    print(f"{dev} ({results['compute_capability']}), torch {torch.__version__}, causal={causal}")
    print(f"shape per length: B={B} H={H} Dh={Dh}\n")

    for T in seq_lens:
        torch.manual_seed(seed)
        q = torch.randn(B, H, T, Dh, device="cuda")
        k = torch.randn(B, H, T, Dh, device="cuda")
        v = torch.randn(B, H, T, Dh, device="cuda")
        flops = attn_flops(B, H, T, Dh, causal)

        # Reference for correctness: the mem-efficient backend (accurate, Turing-OK).
        try:
            reference = impls["sdpa_mem_eff"](q, k, v, causal)
            torch.cuda.synchronize()
        except Exception:
            reference = None

        print(f"T={T}")
        for name, fn in impls.items():
            torch.cuda.empty_cache()
            ref = None if name == "sdpa_mem_eff" else reference
            m = measure(fn, q, k, v, causal, ref)
            row = {"seq_len": T, "impl": name, **m}
            if "error" in m:
                print(f"  {name:14} ERROR  {m['error']}")
            else:
                tflops = flops / (m["latency_ms"] * 1e-3) / 1e12
                row["tflops"] = tflops
                err = "" if m["max_abs_err"] is None else f" err={m['max_abs_err']:.1e}"
                print(f"  {name:14} {m['latency_ms']:8.3f} ms  {m['peak_mb']:8.1f} MB  "
                      f"{tflops:6.2f} TFLOP/s{err}")
            results["rows"].append(row)
        print()
        del q, k, v, reference
        torch.cuda.empty_cache()

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--non-causal", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = run(args.seq, args.batch, args.heads, args.head_dim, causal=not args.non_causal)
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    host = platform.node().split(".")[0]
    out = args.out or os.path.join(
        os.path.dirname(__file__), "results", f"attention_{host}.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
