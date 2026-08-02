"""Short training run on the real corpus, to show the loss actually descends.

Not a convergence run — that's the multi-hour Kaggle job. This trains the real ~55M
model on the real fineweb-edu shards for a few hundred steps on whatever device JAX
finds (CPU on the dev machine), logging train and val loss so the descent from the
~ln(vocab) starting point is visible. The output feeds bench/plots.py.

    python -m scripts.train_smoke --steps 250 --batch 8 --seq 512
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import jax
import jax.numpy as jnp

from jaxformer.config import DEFAULT_MODEL, TrainConfig
from jaxformer.data import TokenDataset
from jaxformer.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", default=os.path.join("bench", "results", "train_smoke_loss.json"))
    args = ap.parse_args()

    # Same 55M model; cap the context to the smoke sequence length so the tokens/step
    # accounting the training loop prints stays honest. Param count is independent of it.
    model_cfg = dataclasses.replace(DEFAULT_MODEL, max_seq_len=args.seq)
    train_cfg = dataclasses.replace(
        TrainConfig(),
        total_steps=args.steps,
        warmup_steps=max(10, args.steps // 10),
        batch_size=args.batch,
        grad_accum_steps=1,
        log_every=10,
        eval_every=max(25, args.steps // 5),
        eval_steps=5,
        checkpoint_every=10**9,  # no checkpoints for a smoke
    )

    train_ds = TokenDataset(args.data_dir, "train")
    val_ds = TokenDataset(args.data_dir, "val")
    train_batches = train_ds.batches(args.batch, args.seq, seed=0)
    val_batches = val_ds.cycle_sequential(args.batch, args.seq)

    print(f"device: {jax.devices()[0].platform} x{len(jax.devices())}")

    val_log: list[dict] = []
    # fp32 on CPU (bf16 is emulated and slow there); vocab-uniform init loss ~= ln(vocab).
    _, _, logs = train(
        model_cfg, train_cfg, train_batches, val_batches,
        compute_dtype=jnp.float32,
        on_eval=lambda step, v: val_log.append({"step": step, "val_loss": v}),
    )

    import math

    out = {
        "device": str(jax.devices()[0].platform),
        "model": "jaxformer ~55M",
        "config": {"steps": args.steps, "batch": args.batch, "seq": args.seq},
        "init_loss_reference_ln_vocab": math.log(model_cfg.vocab_size),
        "train_log": [dataclasses.asdict(l) for l in logs],
        "val_log": val_log,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
