"""Byte-level BPE tokenizer, trained from scratch on the target corpus.

Training our own rather than borrowing GPT-2's is a parameter-budget decision, not
a purity one: at d_model=512 a 50,257-entry vocabulary would put ~25M parameters —
45% of a 55M model — into the embedding table. A 32,768 vocabulary halves that, and
as a bonus fits in uint16 so the tokenized corpus is 2 bytes per token on disk.

Byte-level (rather than character-level) BPE means every possible byte sequence is
encodable, so there is no out-of-vocabulary case and no UNK token to reason about.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"
"""Document separator. Also the id the sampler treats as a stop token."""

SPECIAL_TOKENS = [EOT]


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    # add_prefix_space=False: matches GPT-2's handling so a leading space is part of
    # the following token rather than a separate one.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    out_path: str,
    *,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train and save a BPE tokenizer. ``texts`` may be a lazy iterator."""
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tok.save(out_path)
    return tok


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def eot_id(tok: Tokenizer) -> int:
    tid = tok.token_to_id(EOT)
    if tid is None:
        raise ValueError(f"tokenizer at hand has no {EOT} token")
    return tid


# ---------------------------------------------------------------------------
# Corpus streaming
# ---------------------------------------------------------------------------


def stream_corpus(
    dataset: str, subset: str, *, split: str = "train", text_key: str = "text"
) -> Iterator[str]:
    """Yield documents from a HuggingFace dataset without downloading it.

    ``streaming=True`` matters: fineweb-edu's sample-10BT subset is far larger than
    the free disk on the machine preparing the shards.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, name=subset, split=split, streaming=True)
    for record in ds:
        text = record.get(text_key)
        if text:
            yield text


def take_bytes(texts: Iterable[str], max_bytes: int) -> Iterator[str]:
    """Yield documents until roughly ``max_bytes`` of UTF-8 text have been seen."""
    seen = 0
    for text in texts:
        yield text
        seen += len(text.encode("utf-8"))
        if seen >= max_bytes:
            return


def tokenizer_metadata(tok: Tokenizer) -> dict:
    return {
        "vocab_size": tok.get_vocab_size(),
        "eot_id": eot_id(tok),
        "special_tokens": SPECIAL_TOKENS,
    }


def save_metadata(tok: Tokenizer, path: str) -> None:
    with open(path, "w") as f:
        json.dump(tokenizer_metadata(tok), f, indent=2)
