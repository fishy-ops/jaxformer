"""Real PyTorch training loop for the RTX 2070 Super (Turing) run.

The JAX pipeline can't run on this GPU (no Windows CUDA jaxlib), so the local training
run is PyTorch, on the parity-verified mirror. Turing has no bf16 tensor cores, so this
is the fp16 + loss-scaling path the README calls out: fp32 master weights, autocast
fp16 forward/backward, a GradScaler for gradient underflow.

Resumable and stoppable: it checkpoints (model + optimizer + step + RNG) periodically and
loads the latest checkpoint on start, and writes the loss history to JSON as it goes so
the curve can be plotted mid-run. That matters for a multi-hour run on a personal machine.

  python -m torch_ref.train_loop --data-dir C:\\jfdata --out C:\\jfrun \\
      --steps 30000 --batch 16 --seq 512 --log-every 50 --ckpt-every 2000
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch

from jaxformer.config import DEFAULT_MODEL, TrainConfig
from jaxformer.data import TokenDataset
from torch_ref.model import Transformer


def lr_at(step, base_lr, min_lr, warmup, total):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    if step >= total:
        return min_lr
    frac = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * frac))


def make_optimizer(model, tcfg):
    # Weight decay on matrices only (ndim >= 2), matching the JAX _decay_mask.
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tcfg.learning_rate, betas=(tcfg.b1, tcfg.b2), eps=tcfg.eps,
    )


def latest_ckpt(out_dir):
    cks = glob.glob(os.path.join(out_dir, "ckpt_*.pt"))
    return max(cks, key=lambda p: int(p.split("_")[-1].split(".")[0])) if cks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    cfg, tcfg = DEFAULT_MODEL, TrainConfig()
    model = Transformer(cfg).to(dev)  # fp32 master weights; autocast handles fp16 compute
    opt = make_optimizer(model, tcfg)
    scaler = torch.amp.GradScaler("cuda")

    train_ds = TokenDataset(args.data_dir, "train")
    val_ds = TokenDataset(args.data_dir, "val")
    train_batches = train_ds.batches(args.batch, args.seq, seed=0)
    val_iter = val_ds.cycle_sequential(args.batch, args.seq)

    log_path = os.path.join(args.out, "loss_log.json")
    train_log, val_log = [], []
    start_step = 0
    ck = latest_ckpt(args.out)
    if ck:
        state = torch.load(ck, map_location=dev)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        train_log = state.get("train_log", [])
        val_log = state.get("val_log", [])
        print(f"resumed from {ck} at step {start_step}")

    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = args.batch * args.seq
    print(f"device: {torch.cuda.get_device_name(0)}  params: {n_params:,}  "
          f"tokens/step: {tokens_per_step:,}  target: {args.steps * tokens_per_step / 1e6:.0f}M tokens")

    def to_dev(np_batch):
        return torch.from_numpy(np_batch.astype(np.int64)).to(dev, non_blocking=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(args.eval_steps):
            b = to_dev(next(val_iter))
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _ = model(b[:, :-1])
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), b[:, 1:].reshape(-1))
            losses.append(loss.item())
        model.train()
        return float(np.mean(losses))

    def save_ckpt(step):
        tmp = os.path.join(args.out, f"ckpt_{step}.pt.tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "step": step,
                    "train_log": train_log, "val_log": val_log}, tmp)
        os.replace(tmp, os.path.join(args.out, f"ckpt_{step}.pt"))
        # keep only the two most recent checkpoints
        cks = sorted(glob.glob(os.path.join(args.out, "ckpt_*.pt")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
        for old in cks[:-2]:
            os.remove(old)

    model.train()
    t0 = time.perf_counter()
    window_t, window_steps = t0, 0
    for step in range(start_step + 1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg.learning_rate, tcfg.min_learning_rate, args.warmup, args.steps)

        b = to_dev(next(train_batches))
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, _ = model(b[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), b[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        window_steps += 1

        if step % args.log_every == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            tps = window_steps * tokens_per_step / (now - window_t)
            row = {"step": step, "loss": float(loss.item()), "grad_norm": float(gnorm),
                   "lr": opt.param_groups[0]["lr"], "tokens_per_sec": tps}
            train_log.append(row)
            print(f"step {step:>6}  loss {row['loss']:.4f}  gnorm {row['grad_norm']:.2f}  "
                  f"lr {row['lr']:.2e}  {tps/1e3:.1f}k tok/s", flush=True)
            json.dump({"model": "jaxformer ~55M", "device": torch.cuda.get_device_name(0),
                       "dtype": "fp16", "config": {"steps": args.steps, "batch": args.batch, "seq": args.seq},
                       "init_loss_reference_ln_vocab": math.log(cfg.vocab_size),
                       "train_log": train_log, "val_log": val_log},
                      open(log_path, "w"), indent=2)
            window_t, window_steps = time.perf_counter(), 0

        if step % args.eval_every == 0:
            v = evaluate()
            val_log.append({"step": step, "val_loss": v})
            print(f"step {step:>6}  val_loss {v:.4f}", flush=True)
            window_t = time.perf_counter()

        if step % args.ckpt_every == 0:
            save_ckpt(step)

    save_ckpt(args.steps)
    print(f"done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
