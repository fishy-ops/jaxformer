"""Autoregressive generation with a KV cache.

The decode step is jitted once and reused for every token. Two things make that work:
the cache is preallocated to its full width so shapes never change, and ``start_pos``
is passed as a traced array rather than a Python int, so advancing the position does
not trigger a recompile.

Correctness here is pinned by ``tests/test_model.py::test_kv_cache_decode_matches_full_forward``,
which asserts token-at-a-time decoding reproduces the parallel forward pass.
"""

from __future__ import annotations

import functools
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from jaxformer.model import Transformer


def make_decode_step(graphdef):
    """Build the jitted single-token decode step."""

    @jax.jit
    def step(params, cache, token, start_pos):
        model = nnx.merge(graphdef, params)
        logits, cache = model(token, cache=cache, start_pos=start_pos)
        return logits[:, -1], cache  # (B, vocab)

    return step


@functools.partial(jax.jit, static_argnames=("top_k",))
def sample_token(
    logits: jax.Array, key: jax.Array, temperature: float, top_k: int | None
) -> jax.Array:
    """Sample one token per batch row from ``(B, vocab)`` logits.

    ``temperature <= 0`` means greedy. Top-k is applied before the temperature
    division, which is the conventional order.
    """
    if top_k is not None:
        kth = jnp.sort(logits, axis=-1)[:, -top_k][:, None]
        logits = jnp.where(logits < kth, jnp.finfo(logits.dtype).min, logits)
    return jax.lax.cond(
        temperature > 0,
        lambda: jax.random.categorical(key, logits / jnp.maximum(temperature, 1e-6), axis=-1),
        lambda: jnp.argmax(logits, axis=-1),
    )


def generate(
    model: Transformer,
    prompt_ids: Sequence[int] | jax.Array,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 50,
    eot_id: int | None = None,
    seed: int = 0,
    params=None,
    graphdef=None,
) -> list[int]:
    """Generate a continuation of ``prompt_ids``. Returns prompt + new tokens.

    Pass ``graphdef``/``params`` to reuse an already-split model (and avoid a
    recompile per call); otherwise the model is split here.
    """
    if graphdef is None or params is None:
        graphdef, params = nnx.split(model)

    prompt = jnp.asarray(prompt_ids, jnp.int32).reshape(1, -1)
    n_prompt = prompt.shape[1]
    total = n_prompt + max_new_tokens

    cache = model.init_cache(batch_size=1, max_len=total)
    decode = make_decode_step(graphdef)

    # Prefill the whole prompt in one pass, then decode one token at a time.
    logits, cache = decode(params, cache, prompt, jnp.asarray(0, jnp.int32))

    key = jax.random.key(seed)
    out = list(map(int, prompt[0]))
    for i in range(max_new_tokens):
        key, sub = jax.random.split(key)
        token = sample_token(logits, sub, temperature, top_k)
        tid = int(token[0])
        out.append(tid)
        if eot_id is not None and tid == eot_id:
            break
        logits, cache = decode(
            params, cache, token.reshape(1, 1), jnp.asarray(n_prompt + i, jnp.int32)
        )
    return out


def generate_text(
    model: Transformer,
    tokenizer,
    prompt: str = "",
    **kwargs,
) -> str:
    """Convenience wrapper: text in, text out."""
    ids = tokenizer.encode(prompt).ids if prompt else []
    if not ids:
        # An empty prompt still needs one token to condition on; EOT is the natural
        # "start of document" signal since every document in the corpus ends with it.
        from jaxformer.tokenizer import eot_id as _eot

        ids = [_eot(tokenizer)]
    out = generate(model, ids, **kwargs)
    return tokenizer.decode(out)
