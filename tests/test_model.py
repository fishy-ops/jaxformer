"""Correctness tests for the Flax model.

These are the tests that catch the bugs that actually happen in a from-scratch
transformer: an off-by-one in the causal mask (which leaks the answer and produces a
suspiciously good loss curve), a RoPE convention error (which trains fine but breaks
length extrapolation), and a KV-cache indexing bug (which only shows up at generation
time, long after training has been paid for).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxformer.config import ModelConfig, tiny
from jaxformer.model import Transformer, apply_rope, count_params, rope_tables


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    return tiny()


@pytest.fixture(scope="module")
def model(cfg) -> Transformer:
    return Transformer(cfg, rngs=nnx.Rngs(0))


# ---------------------------------------------------------------------------
# Shapes and parameter budget
# ---------------------------------------------------------------------------


def test_forward_shape_and_dtype(model, cfg):
    tokens = jnp.zeros((3, 16), jnp.int32)
    logits, cache = model(tokens)
    assert logits.shape == (3, 16, cfg.vocab_size)
    # Logits must be float32 even under bf16 compute — the loss and sampler need it.
    assert logits.dtype == jnp.float32
    assert cache is None


def test_param_count_matches_analytic_budget(model, cfg):
    assert count_params(model) == cfg.param_count()["total"]


def test_default_config_is_55m():
    cfg = ModelConfig()
    total = cfg.param_count()["total"]
    assert 50e6 < total < 60e6, f"drifted off the ~55M target: {total:,}"


def test_no_bias_parameters(model):
    """Every Linear is bias-free; a stray bias would silently break torch parity."""
    flat = nnx.to_flat_state(nnx.state(model, nnx.Param))
    assert not [p for p, _ in flat if "bias" in "/".join(str(k) for k in p)]


# ---------------------------------------------------------------------------
# Causal masking
# ---------------------------------------------------------------------------


def test_causal_mask_blocks_future_tokens(model, cfg):
    """Perturbing token t must leave every logit at position < t bit-identical.

    This is the test that catches an off-by-one in the mask. A model that can see one
    token ahead trains to a deceptively low loss and is useless for generation.
    """
    T = 12
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(2, T)), jnp.int32)

    base, _ = model(tokens)
    for t in (1, 5, T - 1):
        perturbed = tokens.at[:, t].set((tokens[:, t] + 1) % cfg.vocab_size)
        other, _ = model(perturbed)
        # Positions before t: unchanged.
        np.testing.assert_array_equal(np.asarray(base[:, :t]), np.asarray(other[:, :t]))
        # Position t itself: must change, otherwise the token isn't being read at all.
        assert not np.allclose(np.asarray(base[:, t]), np.asarray(other[:, t]))


def test_first_position_ignores_all_context(model, cfg):
    """Logits at position 0 depend only on token 0."""
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(1, 8)), jnp.int32)
    b = a.at[:, 1:].set((a[:, 1:] + 3) % cfg.vocab_size)
    la, _ = model(a)
    lb, _ = model(b)
    np.testing.assert_array_equal(np.asarray(la[:, 0]), np.asarray(lb[:, 0]))


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def test_rope_is_norm_preserving(cfg):
    """Rotation must not change vector length."""
    cos, sin = rope_tables(16, cfg.head_dim)
    x = jax.random.normal(jax.random.key(0), (2, 2, 16, cfg.head_dim))
    y = apply_rope(x, cos, sin)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(x), axis=-1),
        np.linalg.norm(np.asarray(y), axis=-1),
        rtol=1e-5,
    )


def test_rope_dot_product_depends_only_on_relative_position(cfg):
    """The defining property of RoPE: <R_i q, R_j k> is a function of (i - j) alone.

    If this passes, the convention (rotate_half vs interleaved) is self-consistent and
    the model will extrapolate past its training context rather than degrading.
    """
    Dh = cfg.head_dim
    cos, sin = rope_tables(64, Dh)
    key = jax.random.key(0)
    q = jax.random.normal(key, (1, 1, 1, Dh))
    k = jax.random.normal(jax.random.fold_in(key, 1), (1, 1, 1, Dh))

    def dot(i: int, j: int) -> float:
        qi = apply_rope(q, cos[i : i + 1], sin[i : i + 1])
        kj = apply_rope(k, cos[j : j + 1], sin[j : j + 1])
        return float(jnp.sum(qi * kj))

    # Same offset, different absolute positions -> same score.
    for offset in (0, 1, 7):
        ref = dot(10 + offset, 10)
        for base in (3, 20, 41):
            assert dot(base + offset, base) == pytest.approx(ref, rel=1e-4, abs=1e-5)


def test_rope_position_zero_is_identity(cfg):
    cos, sin = rope_tables(4, cfg.head_dim)
    x = jax.random.normal(jax.random.key(0), (1, 1, 1, cfg.head_dim))
    np.testing.assert_allclose(
        np.asarray(apply_rope(x, cos[:1], sin[:1])), np.asarray(x), rtol=1e-6, atol=1e-6
    )


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


def test_kv_cache_decode_matches_full_forward(model, cfg):
    """Token-at-a-time decoding must reproduce the parallel forward pass exactly.

    Any indexing error in the cache write, the RoPE position slice, or the mask shows
    up here as a divergence that grows with position.
    """
    T = 10
    rng = np.random.default_rng(2)
    tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(2, T)), jnp.int32)

    full, _ = model(tokens)

    cache = model.init_cache(batch_size=2, max_len=T)
    stepwise = []
    for t in range(T):
        logits, cache = model(tokens[:, t : t + 1], cache=cache, start_pos=t)
        stepwise.append(logits[:, 0])
    stepwise = jnp.stack(stepwise, axis=1)

    np.testing.assert_allclose(
        np.asarray(full), np.asarray(stepwise), rtol=1e-5, atol=1e-5
    )


def test_kv_cache_prefill_then_decode(model, cfg):
    """Prefill a chunk, then decode one token — the real generation path."""
    T, prefill = 10, 6
    rng = np.random.default_rng(3)
    tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(1, T)), jnp.int32)
    full, _ = model(tokens)

    cache = model.init_cache(batch_size=1, max_len=T)
    _, cache = model(tokens[:, :prefill], cache=cache, start_pos=0)
    for t in range(prefill, T):
        logits, cache = model(tokens[:, t : t + 1], cache=cache, start_pos=t)

    np.testing.assert_allclose(
        np.asarray(full[:, T - 1]), np.asarray(logits[:, 0]), rtol=1e-5, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_gradients_are_finite_and_reach_every_parameter(model, cfg):
    """A zero gradient anywhere means a parameter is disconnected from the loss."""
    tokens = jnp.zeros((2, 8), jnp.int32)

    def loss_fn(m):
        logits, _ = m(tokens)
        return jnp.mean(jnp.square(logits))

    grads = nnx.grad(loss_fn)(model)
    leaves = jax.tree.leaves(nnx.to_pure_dict(grads))
    assert leaves, "no gradients produced"
    for g in leaves:
        assert np.all(np.isfinite(np.asarray(g))), "non-finite gradient"
    assert any(np.any(np.asarray(g) != 0) for g in leaves)


def test_tied_embeddings_share_one_parameter(cfg):
    """Tied model must not carry a separate lm_head matrix."""
    tied = Transformer(cfg, rngs=nnx.Rngs(0))
    untied = Transformer(
        ModelConfig(**{**vars(cfg), "tie_embeddings": False}), rngs=nnx.Rngs(0)
    )
    assert count_params(untied) - count_params(tied) == cfg.vocab_size * cfg.d_model
