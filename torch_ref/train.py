"""Minimal PyTorch training step for the cross-hardware benchmark.

This is not a second training pipeline — the real one is `jaxformer/train.py`. It exists
so the *same architecture* (parity-verified against Flax) can be measured for training
throughput and inference latency on the GPU, against the JAX path on TPU/CPU. Kept
deliberately small: AdamW with the same hyperparameters as the JAX side, next-token
cross-entropy, one fused step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from jaxformer.config import ModelConfig, TrainConfig
from torch_ref.model import Transformer


def build(cfg: ModelConfig, tcfg: TrainConfig, device, dtype=torch.float32):
    model = Transformer(cfg).to(device=device, dtype=dtype)
    # RMSNorm scales stay fp32 even in a half model, matching the Flax side.
    for m in model.modules():
        if m.__class__.__name__ == "RMSNorm":
            m.weight.data = m.weight.data.float()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.learning_rate,
        betas=(tcfg.b1, tcfg.b2),
        eps=tcfg.eps,
        weight_decay=tcfg.weight_decay,
    )
    return model, opt


def loss_fn(model: Transformer, batch: torch.Tensor) -> torch.Tensor:
    """`batch` is (B, T+1) int64: inputs are [:, :-1], labels the shifted [:, 1:]."""
    inputs, labels = batch[:, :-1], batch[:, 1:]
    logits, _ = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def train_step(model, opt, batch, grad_clip: float = 1.0) -> float:
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return float(loss.detach())
