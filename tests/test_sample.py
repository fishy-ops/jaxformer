"""Sampling and generation tests.

The decode loop's correctness against the parallel forward pass is already pinned by
``test_model.py::test_kv_cache_decode_matches_full_forward``. What is left to check is
everything layered on top of that: the sampling distribution, the stopping rule, and
the loop bookkeeping that advances ``start_pos``. Those are cheap to get subtly wrong
and expensive to notice — an off-by-one in the position counter produces text that is
merely bad rather than obviously broken.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxformer.config import ModelConfig, tiny
from jaxformer.model import Transformer
from jaxformer.sample import generate, generate_text, sample_token


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    return tiny()


@pytest.fixture(scope="module")
def model(cfg) -> Transformer:
    return Transformer(cfg, rngs=nnx.Rngs(0))


# ---------------------------------------------------------------------------
# sample_token
# ---------------------------------------------------------------------------


def test_greedy_picks_the_argmax():
    logits = jnp.array([[0.1, 5.0, 0.2, 0.3], [9.0, 0.0, 0.0, 0.0]])
    got = sample_token(logits, jax.random.key(0), temperature=0.0, top_k=None)
    np.testing.assert_array_equal(got, jnp.array([1, 0]))


def test_greedy_ignores_the_key():
    """Temperature 0 must be a function of the logits alone, or "greedy" is a lie."""
    logits = jax.random.normal(jax.random.key(1), (4, 32))
    a = sample_token(logits, jax.random.key(0), 0.0, None)
    b = sample_token(logits, jax.random.key(99), 0.0, None)
    np.testing.assert_array_equal(a, b)


def test_top_k_restricts_the_support():
    """Tokens outside the top-k must be unreachable, not merely unlikely."""
    logits = jnp.arange(64, dtype=jnp.float32)[None, :]  # token id == score
    keys = jax.random.split(jax.random.key(0), 200)
    drawn = {int(sample_token(logits, k, 1.0, 4)[0]) for k in keys}
    assert drawn <= {60, 61, 62, 63}


def test_top_k_of_one_is_greedy():
    logits = jax.random.normal(jax.random.key(2), (3, 50))
    got = sample_token(logits, jax.random.key(7), temperature=1.0, top_k=1)
    np.testing.assert_array_equal(got, jnp.argmax(logits, axis=-1))


def test_high_temperature_widens_the_distribution():
    """Sanity check that temperature is dividing, not multiplying."""
    logits = jnp.array([[3.0, 2.0, 1.0, 0.0]])
    keys = jax.random.split(jax.random.key(0), 300)
    cold = {int(sample_token(logits, k, 0.1, None)[0]) for k in keys}
    hot = {int(sample_token(logits, k, 100.0, None)[0]) for k in keys}
    assert len(hot) > len(cold)


def test_sampled_ids_are_in_range():
    logits = jax.random.normal(jax.random.key(3), (8, 17))
    got = sample_token(logits, jax.random.key(4), 1.0, 5)
    assert int(got.min()) >= 0 and int(got.max()) < 17


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def test_generate_returns_prompt_plus_new_tokens(model, cfg):
    prompt = [1, 2, 3]
    out = generate(model, prompt, max_new_tokens=5, temperature=0.0, top_k=None)
    assert out[:3] == prompt
    assert len(out) == 8


def test_generated_ids_stay_in_vocab_range(model, cfg):
    out = generate(model, [1, 2], max_new_tokens=20, temperature=1.0, top_k=10)
    assert all(0 <= t < cfg.vocab_size for t in out)


def test_greedy_generation_is_deterministic(model):
    kw = dict(max_new_tokens=12, temperature=0.0, top_k=None)
    assert generate(model, [4, 5, 6], seed=0, **kw) == generate(
        model, [4, 5, 6], seed=123, **kw
    )


def test_sampling_is_reproducible_under_a_seed(model):
    kw = dict(max_new_tokens=12, temperature=1.0, top_k=8)
    assert generate(model, [4, 5, 6], seed=42, **kw) == generate(
        model, [4, 5, 6], seed=42, **kw
    )
    assert generate(model, [4, 5, 6], seed=42, **kw) != generate(
        model, [4, 5, 6], seed=43, **kw
    )


def test_eot_stops_generation(model, cfg):
    """Stop at the first EOT, and keep it — the token is part of the output."""
    prompt = [1, 2]
    out = generate(model, prompt, max_new_tokens=30, temperature=0.0, top_k=None)
    stop = out[3]  # whatever greedy emits second; force that as the stop token
    stopped = generate(
        model, prompt, max_new_tokens=30, temperature=0.0, top_k=None, eot_id=stop
    )
    assert stopped[-1] == stop
    # Only the generated tail is constrained — the prompt may legitimately contain
    # the stop id, and an eot in the prompt must not suppress generation entirely.
    assert stop not in stopped[len(prompt) : -1]
    assert len(stopped) < len(out)


def test_generation_matches_a_manual_full_forward_decode(model, cfg):
    """The cached loop must agree with recomputing the full forward every step.

    This is the test that catches a mis-advanced ``start_pos``: the cache would still
    hold plausible values, so only a comparison against an uncached reference exposes
    it. Greedy so the comparison is exact.
    """
    prompt = [3, 1, 4, 1]
    n_new = 6
    got = generate(model, prompt, max_new_tokens=n_new, temperature=0.0, top_k=None)

    ref = list(prompt)
    for _ in range(n_new):
        logits, _ = model(jnp.asarray(ref, jnp.int32)[None])
        ref.append(int(jnp.argmax(logits[0, -1])))
    assert got == ref


def test_generate_accepts_a_presplit_model(model):
    """The graphdef/params fast path must produce identical output."""
    graphdef, params = nnx.split(model)
    kw = dict(max_new_tokens=8, temperature=0.0, top_k=None)
    assert generate(model, [2, 7], **kw) == generate(
        model, [2, 7], graphdef=graphdef, params=params, **kw
    )


def test_single_token_prompt_works(model):
    """Degenerate prefill: one token, so prefill and decode look alike."""
    out = generate(model, [5], max_new_tokens=4, temperature=0.0, top_k=None)
    assert len(out) == 5 and out[0] == 5


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------


def test_generate_text_round_trips_through_a_tokenizer(tmp_path, cfg):
    from jaxformer.tokenizer import load_tokenizer, train_tokenizer

    path = tmp_path / "tokenizer.json"
    docs = ["the model attends to every token " * 4] * 200
    train_tokenizer(docs, vocab_size=cfg.vocab_size, out_path=str(path))
    tok = load_tokenizer(str(path))

    m = Transformer(cfg, rngs=nnx.Rngs(0))
    text = generate_text(m, tok, "the model", max_new_tokens=6, temperature=0.0, top_k=None)
    assert isinstance(text, str)
    assert text.startswith("the model")


def test_empty_prompt_is_seeded_with_eot(tmp_path, cfg):
    """An empty prompt still needs one token to condition on.

    Asserted at the id level rather than on the decoded string: ``decode`` drops
    special tokens, so the seeded EOT is invisible in the output text.
    """
    from jaxformer.tokenizer import eot_id, load_tokenizer, train_tokenizer

    path = tmp_path / "tokenizer.json"
    train_tokenizer(["a token stream " * 8] * 200, vocab_size=cfg.vocab_size, out_path=str(path))
    tok = load_tokenizer(str(path))

    m = Transformer(cfg, rngs=nnx.Rngs(0))
    kw = dict(max_new_tokens=4, temperature=0.0, top_k=None)
    from_empty = generate_text(m, tok, "", **kw)
    from_eot = tok.decode(generate(m, [eot_id(tok)], **kw))
    assert from_empty == from_eot
