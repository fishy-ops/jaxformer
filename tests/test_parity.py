"""Flax <-> PyTorch parity and torch-model sanity.

These gate the benchmark table: if the two implementations disagree, comparing JAX on
TPU against PyTorch on GPU measures two different networks. torch is an optional dep,
so the whole module skips cleanly when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxformer.config import ModelConfig, tiny

pytest.importorskip("torch")

import torch  # noqa: E402

from torch_ref.model import Transformer as TorchTransformer  # noqa: E402
from torch_ref.model import count_params  # noqa: E402
from torch_ref.parity import build_paired_models, check_parity  # noqa: E402


def test_logits_match_flax_on_tiny():
    m = check_parity(tiny(), batch=2, seq_len=32)
    assert m["shape"] == (2, 32, tiny().vocab_size)
    # float32 matmul reordering across two frameworks lands well under 1e-4.
    assert m["max_abs"] < 1e-4


def test_logits_match_flax_on_a_deeper_config():
    """Depth and head count are where a transposed or mis-tied weight would show."""
    cfg = ModelConfig(vocab_size=1024, d_model=256, n_layers=4, n_heads=4, d_ff=704, max_seq_len=64)
    m = check_parity(cfg, batch=3, seq_len=48)
    assert m["max_abs"] < 1e-4


def test_strict_load_catches_a_missing_weight():
    """The loader must be strict: a silently-random torch weight is the failure mode
    the whole parity check exists to prevent, so a key mismatch has to raise."""
    from torch_ref.parity import flax_to_torch_state_dict
    from flax import nnx
    from jaxformer.model import Transformer as FlaxTransformer

    cfg = tiny()
    flax_model = FlaxTransformer(cfg, rngs=nnx.Rngs(0))
    torch_model = TorchTransformer(cfg)
    tsd = flax_to_torch_state_dict(flax_model)
    tsd.pop(next(iter(tsd)))  # drop one parameter
    with pytest.raises(RuntimeError):
        torch_model.load_state_dict(tsd, strict=True)


def test_param_count_matches_flax_and_budget():
    cfg = tiny()
    _, torch_model = build_paired_models(cfg)
    from jaxformer.model import Transformer as FlaxTransformer
    from jaxformer.model import count_params as flax_count
    from flax import nnx

    flax_n = flax_count(FlaxTransformer(cfg, rngs=nnx.Rngs(0)))
    assert count_params(torch_model) == flax_n


def test_kv_cache_decode_matches_full_forward():
    """Token-at-a-time decoding with the cache must reproduce the parallel forward."""
    cfg = tiny()
    _, model = build_paired_models(cfg)
    prompt = torch.tensor([[3, 1, 4, 1, 5]], dtype=torch.long)
    n_new = 6

    with torch.no_grad():
        full_ids = prompt.clone()
        # Reference: recompute the whole sequence each step, greedy.
        for _ in range(n_new):
            logits, _ = model(full_ids)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            full_ids = torch.cat([full_ids, nxt], dim=1)

        # Cached: prefill, then one token at a time.
        total = prompt.shape[1] + n_new
        cache = model.init_cache(batch_size=1, max_len=total)
        logits, cache = model(prompt, cache=cache, start_pos=0)
        cached_ids = prompt.clone()
        for i in range(n_new):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            cached_ids = torch.cat([cached_ids, nxt], dim=1)
            logits, cache = model(nxt, cache=cache, start_pos=prompt.shape[1] + i)

    assert torch.equal(full_ids, cached_ids)


def test_causal_mask_holds():
    """Perturbing a future token must not change an earlier position's logits."""
    cfg = tiny()
    _, model = build_paired_models(cfg)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    with torch.no_grad():
        base = model(ids)[0]
        ids2 = ids.clone()
        ids2[0, -1] = (ids2[0, -1] + 7) % cfg.vocab_size
        perturbed = model(ids2)[0]
    # Positions before the last must be identical; the last may differ.
    assert torch.allclose(base[:, :-1], perturbed[:, :-1], atol=1e-5)
    assert not torch.allclose(base[:, -1], perturbed[:, -1])
