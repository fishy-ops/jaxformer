# Handoff — 2026-08-01

Working notes for resuming JaxFormer. Delete this file once the README exists.

Full plan: `~/.claude/plans/project-jaxformer-a-polymorphic-whale.md`

---

## Getting back in

```bash
cd ~/Developer/jaxformer
conda activate jaxformer          # python 3.11, package installed editable
pytest -q                         # 46 tests, ~14s, all should pass
```

If `import jaxformer` fails: `pip install -e ".[data,dev]"`.

Versions verified on this machine: jax 0.10.2, flax 0.12.8, optax 0.2.8,
tokenizers 0.23.1, datasets 5.0.1.

---

## State: Phase 1 essentially done, Phase 2+ not started

| Phase | Status |
|---|---|
| 0 — Mac environment | done, verified |
| 0 — Windows CUDA toolchain | **not started — blocked, see below** |
| 1 — JAX model + tests | done, 13 tests |
| 1 — Sharding + training loop | done, 14 tests |
| 1 — Tokenizer + data pipeline | done, 19 tests |
| 1 — Sampler (`sample.py`) | written, **no tests yet** |
| 2–7 | not started |

Two commits on `main`. No GitHub remote yet — deliberate, the plan creates the
repo once there's a working pipeline to show rather than an empty scaffold.

### What's actually verified

- Model builds at 55,325,184 params, matching the analytic budget exactly.
- Causal mask verified by perturbation, RoPE's relative-position property holds,
  KV-cache decode reproduces the full forward pass.
- 8-device training == 1-device training (atol 1e-5).
- Checkpoints round-trip bit-exactly and leave no `.tmp` directory.
- Corpus prep ran end to end against real fineweb-edu and decoded back to clean
  English with disjoint train/val.

---

## Blocker: the Windows CUDA gate

**This blocks Phases 3, 4, 5, and 6 — roughly half the project.** Nothing in
Phase 1/2 depends on it, which is why it hasn't stopped progress yet.

Tailscale is stopped on the Mac, so the RTX 2070 Super box was never reachable
this session. Two things need to happen, in this order:

1. **Reach the box.** Start Tailscale on both machines, and enable the built-in
   OpenSSH Server on Windows (Settings → Optional Features). Without SSH the
   fallback is a git round-trip: commit here, `git pull` on Windows, run a
   script, commit results back. Workable but slow to iterate on CUDA.

2. **Verify the toolchain.** In priority order — the first one is the real gate:
   - `torch.utils.cpp_extension.load` compiling a hello-world `.cu`. If this
     can't drive MSVC, everything downstream stalls. Fallback is CuPy
     `RawModule` (NVRTC, no MSVC dependency), which costs the "real build
     system" talking point but keeps the CUDA work alive.
   - CUDA Toolkit 12.x + **Visual Studio 2022 Build Tools** (nvcc needs MSVC —
     the Toolkit alone is not enough).
   - `nvcc --version`, `nvidia-smi`.
   - Nsight Compute installed with `ncu` on PATH.
   - PyTorch CUDA wheel (`torch.cuda.is_available()`).

These installs are large and slow. Worth kicking off before a work session
rather than during one.

---

## Next steps, in order

1. **Add sampler tests.** `sample.py` is the only untested module. Wants: greedy
   decoding is deterministic, `top_k` actually restricts the support, EOT stops
   generation, and generated ids stay in vocab range.

2. **Re-verify corpus prep against real data.** The per-document stop fix is
   unit-tested but the confirming real-data run was interrupted. Command:
   ```bash
   python scripts/prepare_data.py --out-dir /tmp/smoke2 --vocab-size 4096 \
     --target-tokens 400000 --val-tokens 100000 --shard-tokens 200000 \
     --tokenizer-bytes 8000000
   ```
   Expect train ≈ 400k tokens (previously 1.54M). Takes a few minutes, mostly
   HuggingFace stream startup.

3. **Real corpus prep.** `python scripts/prepare_data.py --out-dir data`.
   Multi-hour, ~2.2 GB output. Run it in the background. Then upload `data/` to
   Kaggle as a private Dataset.

4. **Phase 2 — Kaggle notebook.** `notebooks/kaggle_tpu_train.ipynb` as a thin
   driver over the pip-installed package. Confirm `jax.devices()` reports 8 TPU
   chips and note the image's JAX/Flax versions before the real run.

5. **Phase 3+ — gated on the Windows box.**

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

- `jaxformer/sample.py` has no tests.
- No README yet (Phase 7).
- No GitHub remote. Account is `fishy-ops`; plan defaults to a public repo named
  `jaxformer`, created at the end of Phase 1.
- `bench/`, `kernels/`, `torch_ref/`, `notebooks/` are empty directories.
- HuggingFace cache has grown to ~2 GB in `~/.cache/huggingface/hub` from the
  smoke runs. Only 42 GB free on this machine — worth watching during the real
  corpus prep.
