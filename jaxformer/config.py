"""Single source of truth for architecture and training hyperparameters.

Deliberately free of any framework import. Both the Flax model (``jaxformer.model``)
and the PyTorch mirror (``torch_ref.model``) consume these same objects, which is what
makes the cross-framework parity test in ``torch_ref/parity.py`` meaningful — there is
no second place for the two implementations to disagree about shapes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Decoder-only transformer, ~55M parameters.

    Modern-standard choices throughout: pre-norm, RMSNorm, RoPE, SwiGLU, tied
    embeddings, no biases anywhere.
    """

    vocab_size: int = 32768
    """Fits in uint16, which halves the on-disk size of the tokenized corpus.

    Also deliberately smaller than GPT-2's 50257: at d_model=512 a 50k vocab would
    put ~25M parameters (45% of the model) into the embedding table alone.
    """

    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8

    d_ff: int = 1408
    """SwiGLU uses three matrices rather than two, so ~8/3 * d_model keeps the
    parameter count level with a conventional 4x GELU MLP. 1408 = 2.75 * 512,
    rounded to a multiple of 128 for tensor-core friendliness."""

    max_seq_len: int = 1024
    """Training context. RoPE has no learned position table, so inference and the
    attention benchmarks can run past this without touching the weights."""

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} not divisible by n_heads={self.n_heads}"
            )
        if self.head_dim % 2:
            # RoPE rotates (even, odd) channel pairs.
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE")

    @property
    def head_dim(self) -> int:
        """64 by default — the tile width the CUDA kernel is built around."""
        return self.d_model // self.n_heads

    def param_count(self) -> dict[str, int]:
        """Analytic parameter budget. Asserted against the real model in tests."""
        embed = self.vocab_size * self.d_model
        attn = 4 * self.d_model * self.d_model  # q, k, v, o — no biases
        mlp = 3 * self.d_model * self.d_ff  # gate, up, down
        norms = 2 * self.d_model  # attn + mlp pre-norms, per layer
        per_layer = attn + mlp + norms
        total = embed + self.n_layers * per_layer + self.d_model  # + final norm
        if not self.tie_embeddings:
            total += embed
        return {
            "embedding": embed,
            "per_layer": per_layer,
            "layers": self.n_layers * per_layer,
            "total": total,
        }


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "HuggingFaceFW/fineweb-edu"
    subset: str = "sample-10BT"
    """Streamed, never fully downloaded — only ~42GB free on the dev machine."""

    tokenizer_train_bytes: int = 2_000_000_000
    """Text sampled to train the BPE. 2GB is ample for a 32k vocab."""

    target_tokens: int = 1_100_000_000
    """Chinchilla-optimal for 55M params (~20 tokens/param). ~2.2GB as uint16."""

    shard_tokens: int = 100_000_000
    """Tokens per binary shard, so a failed tokenization run loses at most one shard."""

    val_tokens: int = 5_000_000
    seed: int = 1337


@dataclass(frozen=True)
class TrainConfig:
    # Optimizer — the GPT-3/Chinchilla-standard small-model recipe.
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 2000
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Batching. tokens_per_step = batch_size * max_seq_len, split across devices
    # and further divided by grad_accum_steps into micro-batches.
    batch_size: int = 256
    grad_accum_steps: int = 1

    total_steps: int = 4200
    """4200 * 256 * 1024 ~= 1.1B tokens."""

    # bf16 compute with fp32 master weights. Note: on the RTX 2070 Super (Turing,
    # sm_75) there is no bf16 tensor-core support, so torch_ref/train.py overrides
    # this to fp16 + loss scaling. That divergence is itself a README finding.
    compute_dtype: str = "bfloat16"
    param_dtype: str = "float32"

    eval_every: int = 250
    eval_steps: int = 40
    checkpoint_every: int = 500
    log_every: int = 10
    seed: int = 1337

    def tokens_per_step(self, max_seq_len: int) -> int:
        return self.batch_size * max_seq_len


DEFAULT_MODEL = ModelConfig()
DEFAULT_DATA = DataConfig()
DEFAULT_TRAIN = TrainConfig()


def replace(cfg, **kwargs):
    """`dataclasses.replace` re-exported so callers need not import dataclasses."""
    return dataclasses.replace(cfg, **kwargs)


def tiny() -> ModelConfig:
    """A ~1M-param model for unit tests and CPU smoke runs.

    head_dim stays 64 so the CUDA kernel's tiling assumptions are exercised by the
    same tests that exercise the full model.
    """
    return ModelConfig(
        vocab_size=512, d_model=128, n_layers=2, n_heads=2, d_ff=352, max_seq_len=128
    )


if __name__ == "__main__":
    counts = DEFAULT_MODEL.param_count()
    for name, n in counts.items():
        print(f"{name:>12}: {n:>12,}")
    print(f"{'total (M)':>12}: {counts['total'] / 1e6:>12.1f}")
