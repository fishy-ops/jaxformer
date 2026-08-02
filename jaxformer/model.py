"""Decoder-only transformer in Flax NNX.

NNX (rather than linen) is deliberate: its object-oriented structure lets
``torch_ref/model.py`` be a near line-for-line mirror, which is what makes the
parity test in ``torch_ref/parity.py`` legible enough to trust. That test gates the
entire cross-hardware benchmark table, so the two implementations being visually
comparable is worth more here than linen's more established sharding idioms.

Conventions worth stating once, because the PyTorch mirror must match them exactly:

* RoPE uses the LLaMA/HF ``rotate_half`` layout (split the head dim in half), not the
  GPT-J interleaved-pairs layout. Both are "RoPE"; they are not interchangeable.
* Flax ``Linear`` kernels are ``(in_features, out_features)``. ``torch.nn.Linear``
  stores ``(out_features, in_features)``. The parity loader transposes.
* RMSNorm reduces in float32 regardless of compute dtype.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from jaxformer.config import ModelConfig


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------


def rope_tables(
    seq_len: int, head_dim: int, theta: float = 10000.0, dtype=jnp.float32
) -> tuple[jax.Array, jax.Array]:
    """Precompute cos/sin tables of shape ``(seq_len, head_dim)``.

    Built in float32 and cast at the end: computing the angles in bf16 loses enough
    precision at long positions to measurably degrade the model.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(pos, inv_freq)  # (T, half)
    # Duplicated to full head_dim so the rotate_half form applies elementwise.
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (T, head_dim)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _rotate_half(x: jax.Array) -> jax.Array:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate ``x`` of shape ``(B, H, T, Dh)`` by tables of shape ``(T, Dh)``."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class RMSNorm(nnx.Module):
    """Root-mean-square layer norm. No mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        del rngs  # weights are deterministic at init
        self.scale = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.eps)
        return (x * self.scale[...]).astype(orig_dtype)


class KVCache(NamedTuple):
    """Preallocated decode cache. Shapes ``(B, H, max_len, Dh)``."""

    k: jax.Array
    v: jax.Array


class CausalSelfAttention(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs, compute_dtype=jnp.float32):
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5

        # Separate q/k/v rather than a fused qkv matrix. A fused matrix is marginally
        # faster, but separate projections keep the PyTorch mirror trivial and make the
        # weights directly inspectable when debugging the CUDA kernel.
        kw = dict(
            use_bias=False,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        init = nnx.initializers.normal(stddev=0.02)
        # Output projection feeds the residual stream, so it gets the GPT-2 depth-scaled
        # init: without it, residual variance grows with depth and early training is unstable.
        out_init = nnx.initializers.normal(stddev=0.02 / (2 * cfg.n_layers) ** 0.5)

        self.q_proj = nnx.Linear(cfg.d_model, cfg.d_model, kernel_init=init, **kw)
        self.k_proj = nnx.Linear(cfg.d_model, cfg.d_model, kernel_init=init, **kw)
        self.v_proj = nnx.Linear(cfg.d_model, cfg.d_model, kernel_init=init, **kw)
        self.o_proj = nnx.Linear(cfg.d_model, cfg.d_model, kernel_init=out_init, **kw)

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        *,
        cache: KVCache | None = None,
        start_pos: int | jax.Array = 0,
    ) -> tuple[jax.Array, KVCache | None]:
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.head_dim

        def split(proj_out):
            return proj_out.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)  # (B, H, T, Dh)

        q = split(self.q_proj(x))
        k = split(self.k_proj(x))
        v = split(self.v_proj(x))

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Absolute positions, so full-sequence and single-token-decode share one path.
        q_pos = start_pos + jnp.arange(T)
        if cache is not None:
            k = jax.lax.dynamic_update_slice(cache.k, k.astype(cache.k.dtype), (0, 0, start_pos, 0))
            v = jax.lax.dynamic_update_slice(cache.v, v.astype(cache.v.dtype), (0, 0, start_pos, 0))
            cache = KVCache(k, v)
        k_pos = jnp.arange(k.shape[2])

        # A key is visible to a query iff it is at or before it. When decoding, this
        # also masks the unwritten tail of the preallocated cache for free.
        mask = k_pos[None, :] <= q_pos[:, None]  # (T, k_len)

        # Softmax in float32: bf16 exponentials lose too much mass in the tails.
        # Masking happens after the cast so the sentinel is a float32 value.
        scores = (jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale).astype(jnp.float32)
        scores = jnp.where(mask[None, None], scores, jnp.finfo(jnp.float32).min)
        weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)
        return self.o_proj(out), cache


class SwiGLU(nnx.Module):
    """Gated feed-forward: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs, compute_dtype=jnp.float32):
        kw = dict(use_bias=False, dtype=compute_dtype, param_dtype=jnp.float32, rngs=rngs)
        init = nnx.initializers.normal(stddev=0.02)
        out_init = nnx.initializers.normal(stddev=0.02 / (2 * cfg.n_layers) ** 0.5)
        self.gate_proj = nnx.Linear(cfg.d_model, cfg.d_ff, kernel_init=init, **kw)
        self.up_proj = nnx.Linear(cfg.d_model, cfg.d_ff, kernel_init=init, **kw)
        self.down_proj = nnx.Linear(cfg.d_ff, cfg.d_model, kernel_init=out_init, **kw)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nnx.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs, compute_dtype=jnp.float32):
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps, rngs=rngs)
        self.attn = CausalSelfAttention(cfg, rngs=rngs, compute_dtype=compute_dtype)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps, rngs=rngs)
        self.mlp = SwiGLU(cfg, rngs=rngs, compute_dtype=compute_dtype)

    def __call__(self, x, cos, sin, *, cache=None, start_pos=0):
        attn_out, cache = self.attn(
            self.attn_norm(x), cos, sin, cache=cache, start_pos=start_pos
        )
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, cache


class Transformer(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs, compute_dtype=jnp.float32):
        self.cfg = cfg
        self.compute_dtype = compute_dtype
        self.embed = nnx.Embed(
            cfg.vocab_size,
            cfg.d_model,
            embedding_init=nnx.initializers.normal(stddev=0.02),
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        # nnx.List, not a plain list: Flax 0.12 requires containers holding parameters
        # to be marked as data so they are traversed by nnx.state / nnx.split.
        self.blocks = nnx.List(
            [Block(cfg, rngs=rngs, compute_dtype=compute_dtype) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps, rngs=rngs)
        if not cfg.tie_embeddings:
            self.lm_head = nnx.Linear(
                cfg.d_model,
                cfg.vocab_size,
                use_bias=False,
                kernel_init=nnx.initializers.normal(stddev=0.02),
                dtype=compute_dtype,
                param_dtype=jnp.float32,
                rngs=rngs,
            )

    def __call__(
        self,
        tokens: jax.Array,
        *,
        cache: list[KVCache] | None = None,
        start_pos: int | jax.Array = 0,
    ) -> tuple[jax.Array, list[KVCache] | None]:
        """``tokens`` is ``(B, T)`` int32. Returns float32 logits ``(B, T, vocab)``."""
        B, T = tokens.shape
        cfg = self.cfg

        # RoPE tables are rebuilt each call rather than stored as buffers. They are
        # cheap relative to the matmuls and XLA constant-folds them under jit, which
        # keeps the module free of non-parameter state that sharding and checkpointing
        # would otherwise have to reason about.
        #
        # Without a cache the only positions needed are [0, T). With one, the cache
        # width already bounds every position that can be decoded, so it is always a
        # sufficient table length and start_pos may stay traced.
        table_len = T if cache is None else cache[0].k.shape[2]
        cos, sin = rope_tables(table_len, cfg.head_dim, cfg.rope_theta, dtype=self.compute_dtype)
        cos_t = jax.lax.dynamic_slice_in_dim(cos, start_pos, T, axis=0)
        sin_t = jax.lax.dynamic_slice_in_dim(sin, start_pos, T, axis=0)

        x = self.embed(tokens)
        new_cache = [] if cache is not None else None
        for i, block in enumerate(self.blocks):
            x, c = block(
                x, cos_t, sin_t, cache=None if cache is None else cache[i], start_pos=start_pos
            )
            if new_cache is not None:
                new_cache.append(c)

        x = self.final_norm(x)
        logits = self.embed.attend(x) if cfg.tie_embeddings else self.lm_head(x)
        # float32 logits: the cross-entropy and the sampler both want full precision,
        # and this is the one place where bf16 rounding is visibly harmful.
        return logits.astype(jnp.float32), new_cache

    def init_cache(self, batch_size: int, max_len: int, dtype=jnp.float32) -> list[KVCache]:
        cfg = self.cfg
        shape = (batch_size, cfg.n_heads, max_len, cfg.head_dim)
        return [
            KVCache(jnp.zeros(shape, dtype), jnp.zeros(shape, dtype))
            for _ in range(cfg.n_layers)
        ]


def count_params(model: Transformer) -> int:
    """Actual parameter count, for asserting against ``ModelConfig.param_count()``."""
    state = nnx.state(model, nnx.Param)
    return sum(int(x.size) for x in jax.tree.leaves(state))
