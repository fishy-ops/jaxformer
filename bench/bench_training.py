"""Cross-hardware training + inference benchmark.

Measures, for the same ~55M architecture (parity-verified across the two frameworks),
three numbers per hardware target:

  * training throughput    tokens/sec of a full fwd + bwd + optimizer step
  * inference latency      ms per token, single-token decode with a KV cache
  * peak memory            MB

Each backend runs where it belongs and writes its own JSON; the results are combined
into one table. This is the "three-way hardware comparison" the project is named for.

  # M4 Pro CPU (JAX), on the dev machine:
  python -m bench.bench_training --backend jax_cpu

  # RTX 2070 Super (PyTorch), on the GPU box:
  python -m bench.bench_training --backend torch_cuda --dtype fp16

  # TPU v5e-8 (JAX): notebooks/kaggle_tpu_train.ipynb calls run_jax() there.

Each backend records the precision it ran in, because training throughput is
precision-dependent and the honest comparison uses each target's native dtype
(TPU bf16, Turing fp16, CPU fp32).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time


def _median_ms(times):
    return statistics.median(times) * 1e3


# ---------------------------------------------------------------------------
# JAX backend (CPU here; the same function runs on a TPU host)
# ---------------------------------------------------------------------------


def run_jax(batch, seq, train_steps, infer_tokens, seed=0):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    from jaxformer import sharding as shd
    from jaxformer.config import DEFAULT_MODEL, TrainConfig
    from jaxformer.sample import generate
    from jaxformer.train import init_train_state, make_train_step

    cfg = DEFAULT_MODEL
    tcfg = TrainConfig()
    devices = jax.devices()
    mesh = shd.make_mesh(devices)
    # On a single CPU device compute in fp32; a TPU host would pass bf16.
    dtype = jnp.bfloat16 if devices[0].platform == "tpu" else jnp.float32
    graphdef, state, tx = init_train_state(cfg, tcfg, mesh, compute_dtype=dtype)
    step = make_train_step(graphdef, tx, accum_steps=1)

    rng = np.random.default_rng(seed)
    batch_arr = shd.put_batch(
        jnp.asarray(rng.integers(0, cfg.vocab_size, (batch, seq + 1), np.int32)), mesh
    )

    state, _ = step(state, batch_arr)  # warmup / compile
    jax.block_until_ready(state)
    times = []
    for _ in range(train_steps):
        t0 = time.perf_counter()
        state, _ = step(state, batch_arr)
        jax.block_until_ready(state)
        times.append(time.perf_counter() - t0)
    tok_s = (batch * seq) / statistics.median(times)

    # Inference latency: prompt of 8, decode infer_tokens with the KV cache.
    model = nnx.merge(graphdef, state.params)
    prompt = list(range(8))
    generate(model, prompt, max_new_tokens=8, temperature=0.0, top_k=None)  # warmup
    t0 = time.perf_counter()
    generate(model, prompt, max_new_tokens=infer_tokens, temperature=0.0, top_k=None)
    infer_ms = (time.perf_counter() - t0) / infer_tokens * 1e3

    peak_mb = _peak_rss_mb()
    return {
        "backend": f"jax_{devices[0].platform}",
        "device": str(devices[0].device_kind),
        "n_devices": len(devices),
        "dtype": "bf16" if dtype == jnp.bfloat16 else "fp32",
        "train_tokens_per_sec": tok_s,
        "infer_ms_per_token": infer_ms,
        "peak_mb": peak_mb,
    }


# ---------------------------------------------------------------------------
# PyTorch / CUDA backend
# ---------------------------------------------------------------------------


def run_torch(batch, seq, train_steps, infer_tokens, dtype_name, seed=0):
    import torch

    from jaxformer.config import DEFAULT_MODEL, TrainConfig
    from torch_ref.train import build, train_step

    assert torch.cuda.is_available()
    torch.manual_seed(seed)
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    cfg, tcfg = DEFAULT_MODEL, TrainConfig()
    model, opt = build(cfg, tcfg, "cuda", dtype)

    batch_arr = torch.randint(0, cfg.vocab_size, (batch, seq + 1), device="cuda")

    for _ in range(3):
        train_step(model, opt, batch_arr)  # warmup
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(train_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_step(model, opt, batch_arr)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    tok_s = (batch * seq) / statistics.median(times)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    # Inference latency: prefill 8, decode infer_tokens with a KV cache.
    model.eval()
    with torch.no_grad():
        prompt = torch.randint(0, cfg.vocab_size, (1, 8), device="cuda")
        cache = model.init_cache(1, 8 + infer_tokens, dtype=dtype, device="cuda")
        logits, cache = model(prompt, cache=cache, start_pos=0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(infer_tokens):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            logits, cache = model(nxt, cache=cache, start_pos=8 + i)
        torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) / infer_tokens * 1e3

    return {
        "backend": "torch_cuda",
        "device": torch.cuda.get_device_name(0),
        "n_devices": 1,
        "dtype": dtype_name,
        "train_tokens_per_sec": tok_s,
        "infer_ms_per_token": infer_ms,
        "peak_mb": peak_mb,
    }


def _peak_rss_mb():
    import resource
    import sys

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes.
    return ru / 1e6 if sys.platform == "darwin" else ru / 1e3


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["jax_cpu", "torch_cuda"], required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--train-steps", type=int, default=None)
    ap.add_argument("--infer-tokens", type=int, default=64)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.backend == "jax_cpu":
        steps = args.train_steps if args.train_steps is not None else 3
        row = run_jax(args.batch, args.seq, steps, args.infer_tokens)
    else:
        steps = args.train_steps if args.train_steps is not None else 20
        row = run_torch(args.batch, args.seq, steps, args.infer_tokens, args.dtype)

    row["config"] = {"batch": args.batch, "seq": args.seq}
    row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(row, indent=2))

    host = platform.node().split(".")[0]
    out = args.out or os.path.join(
        os.path.dirname(__file__), "results", f"training_{args.backend}_{host}.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(row, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
