"""Flax <-> PyTorch numerical parity.

The whole point of the hardware comparison is to run *the same model* on a TPU (JAX)
and a GPU (PyTorch). "The same model" is a claim that has to be proven, not asserted:
this module loads one set of weights into both implementations and checks the logits
agree. If they don't, the two are different networks and every number in the benchmark
table is meaningless.

The load is where the two frameworks' conventions collide, so it is the interesting
part:

* Flax ``Linear`` stores its kernel as ``(in, out)``; ``torch.nn.Linear`` stores
  ``(out, in)``. Every kernel is transposed.
* Flax ``Embed`` and torch ``Embedding`` both store ``(vocab, d_model)`` — no transpose.
* RMSNorm's parameter is called ``scale`` in Flax and ``weight`` in torch.

``load_state_dict(..., strict=True)`` is deliberate: if the key mapping is wrong for
even one parameter, it raises instead of silently leaving a torch tensor at its random
init, which would otherwise surface as a mysterious parity failure.
"""

from __future__ import annotations

import numpy as np

from jaxformer.config import ModelConfig


def flax_to_torch_state_dict(flax_model) -> dict:
    """Convert a Flax NNX model's parameters into a torch state_dict.

    Returns a plain ``{name: torch.Tensor}`` mapping keyed to match this package's
    ``torch_ref.model.Transformer`` parameter names.
    """
    import torch
    from flax import nnx

    flat = nnx.to_flat_state(nnx.state(flax_model, nnx.Param))

    tsd: dict[str, "torch.Tensor"] = {}
    for path, leaf in flat:
        # nnx.List may insert a container segment (e.g. an integer index or the
        # attribute name) into the path; join everything and let the suffix decide.
        parts = [str(p) for p in path]
        fkey = ".".join(parts)
        # leaf is an NNX Variable/VariableState; leaf[...] is the non-deprecated read.
        arr = np.asarray(leaf[...])

        if fkey.endswith(".embedding"):
            tkey, w = fkey[: -len(".embedding")] + ".weight", arr
        elif fkey.endswith(".scale"):
            tkey, w = fkey[: -len(".scale")] + ".weight", arr
        elif fkey.endswith(".kernel"):
            tkey, w = fkey[: -len(".kernel")] + ".weight", arr.T  # (in,out) -> (out,in)
        else:
            raise ValueError(f"unmapped Flax parameter {fkey!r}")

        # .copy() so the tensor owns writable memory (jax arrays are read-only).
        tsd[tkey] = torch.from_numpy(np.ascontiguousarray(w).copy())
    return tsd


def load_flax_weights_into_torch(flax_model, torch_model) -> None:
    """Copy Flax weights into an equivalent torch model, in float32, strictly."""
    tsd = flax_to_torch_state_dict(flax_model)
    # Match the torch model's dtype so a half-precision model still loads.
    ref_dtype = next(torch_model.parameters()).dtype
    tsd = {k: v.to(ref_dtype) for k, v in tsd.items()}
    missing, unexpected = torch_model.load_state_dict(tsd, strict=True)
    assert not missing and not unexpected  # strict=True already raises, belt and braces


def build_paired_models(cfg: ModelConfig, seed: int = 0):
    """Construct a Flax model and a torch model holding identical weights."""
    from flax import nnx

    from jaxformer.model import Transformer as FlaxTransformer
    from torch_ref.model import Transformer as TorchTransformer

    flax_model = FlaxTransformer(cfg, rngs=nnx.Rngs(seed))
    torch_model = TorchTransformer(cfg)
    torch_model.eval()
    load_flax_weights_into_torch(flax_model, torch_model)
    return flax_model, torch_model


def check_parity(
    cfg: ModelConfig | None = None,
    *,
    batch: int = 2,
    seq_len: int = 16,
    seed: int = 0,
) -> dict:
    """Run both models on the same tokens and return difference metrics.

    Returns a dict with ``max_abs``, ``max_rel``, and the logit shape. The caller
    decides the tolerance; ``parity.py`` as a script prints and thresholds at 1e-4.
    """
    import jax.numpy as jnp
    import torch

    from jaxformer.config import tiny

    if cfg is None:
        cfg = tiny()

    flax_model, torch_model = build_paired_models(cfg, seed=seed)

    rng = np.random.default_rng(seed)
    tokens_np = rng.integers(0, cfg.vocab_size, size=(batch, seq_len), dtype=np.int64)

    flax_logits = np.asarray(flax_model(jnp.asarray(tokens_np))[0])
    with torch.no_grad():
        torch_logits = torch_model(torch.from_numpy(tokens_np))[0].numpy()

    diff = np.abs(flax_logits - torch_logits)
    denom = np.maximum(np.abs(flax_logits), np.abs(torch_logits))
    rel = np.where(denom > 1e-6, diff / denom, 0.0)
    return {
        "max_abs": float(diff.max()),
        "max_rel": float(rel.max()),
        "shape": tuple(flax_logits.shape),
        "mean_abs": float(diff.mean()),
    }


def main() -> int:
    from jaxformer.config import ModelConfig, tiny

    # A deeper/wider config than tiny() exercises depth-dependent init and more heads,
    # which is where a transposed or mis-tied weight would show up.
    for name, cfg in [("tiny", tiny()), ("small", ModelConfig(vocab_size=1024, d_model=256, n_layers=4, n_heads=4, d_ff=704, max_seq_len=64))]:
        m = check_parity(cfg, batch=2, seq_len=min(32, cfg.max_seq_len))
        status = "PASS" if m["max_abs"] < 1e-4 else "FAIL"
        print(f"[{status}] {name:6} logits {m['shape']}  max_abs={m['max_abs']:.2e}  "
              f"max_rel={m['max_rel']:.2e}  mean_abs={m['mean_abs']:.2e}")
        if m["max_abs"] >= 1e-4:
            return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
