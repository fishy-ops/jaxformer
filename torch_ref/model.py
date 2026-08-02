"""Decoder-only transformer in PyTorch — a weight-for-weight mirror of the Flax model.

This exists for one reason: to make "JAX on TPU vs PyTorch on GPU" a defensible
comparison rather than an apples-to-oranges one. ``parity.py`` loads a single set of
weights into both this module and ``jaxformer/model.py`` and asserts the logits agree
to < 1e-4; the benchmark table means nothing without that check passing first.

So every numerical convention here must match the Flax version exactly. The ones that
are easy to get subtly wrong:

* RoPE uses the LLaMA/HF ``rotate_half`` layout (split the head dim in half), never the
  GPT-J interleaved-pairs layout.
* ``nn.Linear`` stores its weight as ``(out, in)``; Flax stores ``(in, out)``. The
  parity loader transposes — this file just uses ``nn.Linear`` normally.
* RMSNorm reduces in float32 and keeps its scale in float32 regardless of compute dtype.
* Logits are returned in float32.

The module is intentionally close to line-for-line with ``jaxformer/model.py`` so the
two can be read side by side.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from jaxformer.config import ModelConfig

# A KV-cache entry is a mutable (k, v) pair of preallocated (B, H, max_len, Dh) tensors.
KVCache = tuple[torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------


def rope_tables(
    seq_len: int, head_dim: int, theta: float = 10000.0, *, device=None, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape ``(seq_len, head_dim)``, built in float32 then cast."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)  # (T, half)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``(B, H, T, Dh)`` by tables of shape ``(T, Dh)``."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer norm. No mean subtraction, no bias. float32 reduction."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight).to(orig_dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        d = cfg.d_model
        # Separate q/k/v/o, no bias — matches the Flax model and keeps the weights
        # directly inspectable when debugging the CUDA kernel that replaces this.
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, KVCache | None]:
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.head_dim

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)

        q = split(self.q_proj(x))
        k = split(self.k_proj(x))
        v = split(self.v_proj(x))

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Absolute positions, so full-sequence and single-token decode share one path.
        q_pos = start_pos + torch.arange(T, device=x.device)
        if cache is not None:
            k_cache, v_cache = cache
            k_cache[:, :, start_pos : start_pos + T] = k.to(k_cache.dtype)
            v_cache[:, :, start_pos : start_pos + T] = v.to(v_cache.dtype)
            k, v = k_cache, v_cache
            cache = (k_cache, v_cache)
        k_pos = torch.arange(k.shape[2], device=x.device)

        # A key is visible to a query iff it is at or before it; also masks the
        # unwritten tail of the preallocated cache for free during decode.
        mask = k_pos[None, :] <= q_pos[:, None]  # (T, k_len)

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k).float() * self.scale
        scores = scores.masked_fill(~mask[None, None], torch.finfo(torch.float32).min)
        weights = torch.softmax(scores, dim=-1).to(v.dtype)
        out = torch.einsum("bhqk,bhkd->bhqd", weights, v)

        out = out.transpose(1, 2).reshape(B, T, H * Dh)
        return self.o_proj(out), cache


class SwiGLU(nn.Module):
    """Gated feed-forward: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, cache = self.attn(
            self.attn_norm(x), cos, sin, cache=cache, start_pos=start_pos
        )
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, cache


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        """GPT-style init, matching the Flax model, for training from scratch.

        Without this a fresh torch model uses PyTorch's default (kaiming) init, whose
        much larger weights give a ~200-nat initial loss instead of ~ln(vocab). Parity
        never caught it because the parity loader overwrites these with Flax's weights;
        it only bites a from-scratch run.
        """
        cfg = self.cfg
        std = 0.02
        out_std = 0.02 / (2 * cfg.n_layers) ** 0.5  # GPT-2 depth scaling for residual outs
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        if not cfg.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        for blk in self.blocks:
            for lin in (blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj,
                        blk.mlp.gate_proj, blk.mlp.up_proj):
                nn.init.normal_(lin.weight, mean=0.0, std=std)
            for lin in (blk.attn.o_proj, blk.mlp.down_proj):  # feed the residual stream
                nn.init.normal_(lin.weight, mean=0.0, std=out_std)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        cache: list[KVCache] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        """``tokens`` is ``(B, T)`` int64. Returns float32 logits ``(B, T, vocab)``."""
        _, T = tokens.shape
        cfg = self.cfg

        # Without a cache the only positions needed are [0, T). With one, the cache
        # width bounds every decodable position, so it is a sufficient table length.
        table_len = T if cache is None else cache[0][0].shape[2]
        compute_dtype = self.embed.weight.dtype
        cos, sin = rope_tables(
            table_len, cfg.head_dim, cfg.rope_theta, device=tokens.device, dtype=compute_dtype
        )
        cos_t = cos[start_pos : start_pos + T]
        sin_t = sin[start_pos : start_pos + T]

        x = self.embed(tokens)
        new_cache: list[KVCache] | None = [] if cache is not None else None
        for i, block in enumerate(self.blocks):
            x, c = block(
                x, cos_t, sin_t, cache=None if cache is None else cache[i], start_pos=start_pos
            )
            if new_cache is not None:
                new_cache.append(c)

        x = self.final_norm(x)
        # Compute logits in fp32 with autocast disabled. Under fp16 autocast the tied
        # projection over a 32k vocab can overflow fp16's ~65504 max and blow up the
        # loss; the Flax side likewise returns fp32 logits. In an fp32 forward this
        # context is a no-op, so parity is unaffected.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            if cfg.tie_embeddings:
                logits = F.linear(xf, self.embed.weight.float())
            else:
                logits = self.lm_head(xf.to(self.lm_head.weight.dtype)).float()
        return logits, new_cache

    def init_cache(self, batch_size: int, max_len: int, dtype=torch.float32, device=None) -> list[KVCache]:
        cfg = self.cfg
        shape = (batch_size, cfg.n_heads, max_len, cfg.head_dim)
        return [
            (torch.zeros(shape, dtype=dtype, device=device), torch.zeros(shape, dtype=dtype, device=device))
            for _ in range(cfg.n_layers)
        ]


def count_params(model: Transformer) -> int:
    # Tied embeddings: embed.weight is the only copy, so a plain sum is already correct.
    return sum(p.numel() for p in model.parameters())
