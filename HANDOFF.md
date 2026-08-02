# Handoff — updated 2026-08-02 (repo public, near-complete)

Working notes for resuming JaxFormer. Delete this file once the README exists.

Full plan: `~/.claude/plans/project-jaxformer-a-polymorphic-whale.md`

---

## Getting back in

```bash
cd ~/Developer/jaxformer
conda activate jaxformer          # python 3.11, package installed editable
pytest -q                         # 62 tests, ~19s, all should pass
```

If `import jaxformer` fails: `pip install -e ".[data,dev]"`.

Versions verified on this machine: jax 0.10.2, flax 0.12.8, optax 0.2.8,
tokenizers 0.23.1, datasets 5.0.1.

---

## State: near-complete. Only actual training runs (TPU notebook + convergence) remain

| Phase | Status |
|---|---|
| 0 — Mac environment | done, verified |
| 0 — Windows CUDA toolchain | **done, gate passes** — see `docs/windows_gpu_setup.md` |
| 1 — JAX model + tests | done, 13 tests |
| 1 — Sharding + training loop | done, 14 tests |
| 1 — Tokenizer + data pipeline | done, 19 tests |
| 1 — Sampler (`sample.py`) | done, 16 tests |
| 3 — PyTorch mirror + parity | **done**, 8 tests, parity max_abs ~1e-6 |
| 4 — CUDA kernel v1 (fp32) | **done**, correct vs float64 ref (~1e-6), 18 tests |
| 5 — CUDA kernel v2 (fp16 WMMA) | **done**, correct (~1e-3), 22 tests, profiling-driven 3.4x fix |
| 6 — Attention benchmark (fp32+fp16) | **done**, `bench/results/attention_AKPC*.json` |
| 6 — Nsight profiling (v1 + v2 before/after) | **done**, `bench/results/ncu_v{1,2}_2048.json` |
| 7 — README + charts | **done** |
| 6 — Cross-hardware training bench (CPU+GPU) | **done**, `bench/results/training_*.json` |
| 2 — Kaggle TPU run | notebook ready (`notebooks/kaggle_tpu_train.ipynb`); user runs it |
| 3 — Local GPU training to convergence | not started (throughput measured; needs corpus) |

**Repo is public: https://github.com/fishy-ops/jaxformer** (`origin/main`).

Only two things remain, both gated on running actual multi-hour/interactive jobs, not
on code: (a) run the Kaggle notebook to fill the TPU row of the cross-hardware table;
(b) prep the real corpus and run a training-to-convergence loss curve. Everything the
project is *about* — the pipeline, both CUDA kernels, the profiling-driven optimization,
the parity, the benchmarks — is done, tested, and pushed.

**Profiling gave v2 a concrete target, not a guess:** v1 sits at 12% occupancy,
register-limited (164 regs/thread from the per-thread q + accumulator arrays); DRAM at
1.5% so it's latency/occupancy-bound, not bandwidth; shared-STORE bank conflicts are
~10x the loads. So v2 = fp16 WMMA tensor cores (collapse the register arrays, halve
smem) + a padded shared layout (kill the store conflicts). The nearest cheap
"show-the-delta" win is the padded layout alone.

**68 tests on the Mac** (incl. parity), **40 kernel tests on the box** (v1 + v2).
Full results and the benchmark/profiling story live in the README now.

### What's actually verified

- Model builds at 55,325,184 params, matching the analytic budget exactly.
- Causal mask verified by perturbation, RoPE's relative-position property holds,
  KV-cache decode reproduces the full forward pass.
- 8-device training == 1-device training (atol 1e-5).
- Checkpoints round-trip bit-exactly and leave no `.tmp` directory.
- Corpus prep ran end to end against real fineweb-edu and decoded back to clean
  English with disjoint train/val. Token targets are respected: 402,166 against a
  400,000 request, overshoot bounded by one document.
- Generation matches an uncached full-forward reference token for token.

---

## Windows CUDA gate — CLEARED

The gate that blocked Phases 3–6 now passes: `scripts/probe_cuda_build.py` compiles
and runs a `.cu` on the RTX 2070 Super via `cpp_extension` and returns correct
results. Full recipe and the traps that cost the most time are in
**`docs/windows_gpu_setup.md`** — read it before touching the box.

The short version:

- **Reach it:** `ssh akpc` (alias in `~/.ssh/config` → `reach@100.109.40.37`, key
  auth). The account is `reach` (admin); the key lives in
  `%ProgramData%\ssh\administrators_authorized_keys`. Sessions are elevated.
- **Run anything:** SSH is non-interactive, so nothing is on PATH. Always
  `ssh akpc "cmd /c \"call C:\jaxformer\jf_env.bat && C:\jfvenv\Scripts\python.exe <script>\""`.
  `jf_env.bat` sources vcvars64 + exports `CUDA_HOME=C:\cudahome` + PATH.
- **CUDA_HOME is hand-assembled** at `C:\cudahome` (12.8.93). The NVIDIA installer
  is unusable — it force-installs the bundled 572.61 driver over the newer 596.36
  and aborts. `scripts/assemble_cudahome.ps1` unpacks the installer with 7-Zip and
  merges the toolkit payloads (no driver).
- **torch is pinned to 2.8.0+cu128.** 2.11.0 fails: nvcc + MSVC 19.44 choke on
  `compiled_autograd.h` (`C2872 'std': ambiguous`).
- Env: venv at `C:\jfvenv`, Python 3.13, ninja 1.13, 7-Zip 26.02, MSVC 14.44.

---

## Next steps (only the runs remain)

1. **TPU row of the cross-hardware table.** Open `notebooks/kaggle_tpu_train.ipynb`
   on a Kaggle TPU v5e-8, run it, download `training_jax_tpu_kaggle.json` into
   `bench/results/`, and `python -m bench.plots` to refresh the chart.

2. **Training to convergence (optional headline).** `python scripts/prepare_data.py
   --out-dir data` (multi-hour, ~2.2 GB, exits cleanly), upload `data/` as a private
   Kaggle Dataset, then use the notebook's optional training section. Sanity target:
   val loss ≈ 3.2–3.6 nats.

The GPU box (`ssh akpc`) is otherwise fully operational; see the section above and
`docs/windows_gpu_setup.md`.

---

## Decisions made this session, with reasons

Things that will look arbitrary later without the reasoning:

- **Kaggle TPU v5e-8, not Colab.** Colab retired free TPU v2-8 in Sept 2025; its
  free tier is now a single-chip v5e-1, which cannot demonstrate multi-device
  sharding. Kaggle gives 8 chips free, 20h/week.

- **`AxisType.Auto` in `sharding.py`.** JAX 0.10 defaults `make_mesh` to
  `Explicit` ("sharding in types"), which requires a `jax.set_mesh` context
  around every `jit` call and may postdate the Kaggle image. Auto behaves
  identically across versions. If you see *"Please enter your jit into a mesh
  context"*, this is why.

- **Flax NNX over linen.** So `torch_ref/model.py` can mirror it near
  line-for-line, which is what will make the parity test legible. NNX 0.12
  needs `nnx.List` for containers holding parameters, not a plain list.

- **Own 32k BPE, not GPT-2's.** At d_model=512 a 50k vocab puts ~25M params —
  45% of the model — in the embedding table. 32k also fits uint16, halving the
  corpus on disk to ~2.2 GB, which is what makes the Kaggle upload practical.

- **`os._exit(0)` at the end of `prepare_data.py`.** Not laziness. Abandoning a
  HuggingFace streaming iterator mid-document — which the target-token stop does,
  twice — wedges the interpreter at shutdown: 0% CPU, manifests already written,
  never exits. All durable output is on disk by that point. Without this the
  multi-hour background run never reports completion.

- **The 1e-5 tolerance in `test_sharded_step_matches_single_device`.** Not
  sloppiness: float32 all-reduce ordering genuinely differs by device count.
  Measured ~1 element in 45k differing at 5e-6. A tighter bound fails for
  arithmetic reasons, not correctness ones.

---

## Known facts about the target hardware

Established by research this session; they shape the kernel design, so don't
re-derive them:

- **FlashAttention-2 will not build for the 2070 Super.** It requires Ampere
  (sm_80+); Turing is sm_75. FA1 or the `flash-attention-turing` fork are the
  only FA options. PyTorch SDPA's `mem_efficient` backend *does* work on Turing
  and is the intended reference baseline. SDPA's `flash` backend failing on this
  GPU is a headline README finding, not a problem to hide.
- **Turing has no bf16 tensor-core support** — local training is fp16 + loss
  scaling, unlike the TPU run's bf16. Worth calling out in the writeup.
- **No `cp.async` on Turing**, 64 KB shared memory per block, 40 SMs, 8 GB VRAM,
  448 GB/s. Most FlashAttention tutorial code assumes Ampere pipelining and
  won't compile here.
- **`nvcuda::wmma` with `m16n16k16` half fragments is supported on sm_75** — the
  intended v2 path, and much less error-prone than hand-written
  `mma.sync.m16n8k8` PTX.

---

## Loose ends

- No README yet (Phase 7).
- No GitHub remote. Account is `fishy-ops`; plan defaults to a public repo named
  `jaxformer`, created at the end of Phase 1.
- `bench/`, `kernels/`, `torch_ref/`, `notebooks/` are empty directories.
- HuggingFace cache has grown to ~2 GB in `~/.cache/huggingface/hub` from the
  smoke runs. Only 42 GB free on this machine — worth watching during the real
  corpus prep.
