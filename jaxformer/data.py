"""Tokenized-corpus writer and loader.

The corpus is stored as flat uint16 binary shards — a single contiguous stream of
token ids with documents separated by the EOT token — plus a small JSON manifest.
No framework data pipeline is involved: the loader memory-maps the shards and slices
windows out of them, which is a few dozen lines and has no version-coupling to
TensorFlow or PyTorch. On a TPU host the OS page cache does the prefetching.

uint16 requires vocab_size <= 65536; that is enforced at write time. It halves the
corpus on disk relative to uint32 (2.2GB instead of 4.4GB for 1.1B tokens), which is
what makes it practical to upload as a Kaggle Dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

TOKEN_DTYPE = np.uint16
MANIFEST = "manifest.json"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


@dataclass
class ShardWriter:
    """Append token ids to fixed-size shards.

    Sharding is for crash tolerance more than anything: a tokenization run that dies
    partway through keeps every completed shard instead of restarting from zero.
    """

    out_dir: str
    split: str
    shard_tokens: int
    vocab_size: int

    def __post_init__(self) -> None:
        if self.vocab_size > np.iinfo(TOKEN_DTYPE).max + 1:
            raise ValueError(
                f"vocab_size {self.vocab_size} exceeds {TOKEN_DTYPE.__name__} range; "
                "widen TOKEN_DTYPE or shrink the vocabulary"
            )
        os.makedirs(self.out_dir, exist_ok=True)
        self._buf: list[np.ndarray] = []
        self._buf_len = 0
        self._shards: list[dict] = []
        self._total = 0

    def add(self, ids: Iterable[int]) -> None:
        arr = np.fromiter(ids, dtype=TOKEN_DTYPE)
        self._buf.append(arr)
        self._buf_len += arr.size
        self._total += arr.size
        while self._buf_len >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, n: int) -> None:
        joined = np.concatenate(self._buf) if len(self._buf) > 1 else self._buf[0]
        head, tail = joined[:n], joined[n:]
        name = f"{self.split}_{len(self._shards):04d}.bin"
        head.tofile(os.path.join(self.out_dir, name))
        self._shards.append({"file": name, "tokens": int(head.size)})
        self._buf = [tail] if tail.size else []
        self._buf_len = int(tail.size)

    def close(self) -> dict:
        if self._buf_len:
            self._flush(self._buf_len)
        manifest = {
            "split": self.split,
            "dtype": TOKEN_DTYPE.__name__,
            "vocab_size": self.vocab_size,
            "total_tokens": self._total,
            "shards": self._shards,
        }
        with open(os.path.join(self.out_dir, f"{self.split}_{MANIFEST}"), "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest


def tokenize_stream(
    documents: Iterable[str],
    tokenizer,
    writer: ShardWriter,
    *,
    target_tokens: int,
    batch_size: int = 1024,
    progress_every: int = 10_000_000,
) -> dict:
    """Encode documents into shards until ``target_tokens`` is reached.

    Documents are encoded in batches because ``encode_batch`` releases the GIL and
    parallelizes across cores; one-at-a-time encoding is roughly an order of
    magnitude slower and would dominate the wall clock of corpus prep.
    """
    from jaxformer.tokenizer import eot_id

    eot = eot_id(tokenizer)
    written = 0
    next_report = progress_every
    batch: list[str] = []

    def drain(batch: list[str]) -> int:
        """Write a batch of documents, stopping mid-batch once the target is met.

        The stop check is per-document, not per-batch. Checking only between batches
        overshoots by up to one batch — with 1024 documents of ~1.4k tokens each that
        is ~1.4M tokens, which swamps any target smaller than itself.
        """
        nonlocal written
        n = 0
        for enc in tokenizer.encode_batch(batch):
            # EOT terminates each document so the model learns document boundaries
            # rather than treating the corpus as one run-on text.
            writer.add(enc.ids + [eot])
            n += len(enc.ids) + 1
            if written + n >= target_tokens:
                break
        written += n
        return n

    for doc in documents:
        batch.append(doc)
        if len(batch) >= batch_size:
            drain(batch)
            batch = []
            if written >= next_report:
                print(f"  {written / 1e6:.0f}M tokens")
                next_report += progress_every
            if written >= target_tokens:
                break
    if batch and written < target_tokens:
        drain(batch)

    return writer.close()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class TokenDataset:
    """Memory-mapped view over a split's shards.

    Windows are sampled within a single shard rather than across the virtual
    concatenation. At 100M tokens per shard the boundary tokens lost are ~1e-5 of the
    corpus, and it keeps every read a single contiguous slice of one mapping.
    """

    def __init__(self, data_dir: str, split: str = "train"):
        manifest_path = os.path.join(data_dir, f"{split}_{MANIFEST}")
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        if self.manifest["dtype"] != TOKEN_DTYPE.__name__:
            raise ValueError(f"unexpected dtype {self.manifest['dtype']}")

        self.split = split
        self.vocab_size = self.manifest["vocab_size"]
        self.total_tokens = self.manifest["total_tokens"]
        self._shards = [
            np.memmap(os.path.join(data_dir, s["file"]), dtype=TOKEN_DTYPE, mode="r")
            for s in self.manifest["shards"]
        ]
        if not self._shards:
            raise ValueError(f"no shards listed in {manifest_path}")
        self._sizes = np.array([s.size for s in self._shards], dtype=np.int64)

    def __len__(self) -> int:
        return self.total_tokens

    def batches(
        self, batch_size: int, seq_len: int, *, seed: int = 0
    ) -> Iterator[np.ndarray]:
        """Yield ``(batch_size, seq_len + 1)`` int32 arrays, forever.

        The extra +1 column is the shifted target: the training step consumes
        ``batch[:, :-1]`` as input and ``batch[:, 1:]`` as labels.
        """
        window = seq_len + 1
        usable = self._sizes - window
        if np.all(usable < 0):
            raise ValueError(f"no shard is long enough for a {window}-token window")
        # Sample shards in proportion to how many windows each can supply, so every
        # token in the corpus is equally likely regardless of shard size.
        weights = np.maximum(usable, 0).astype(np.float64)
        weights /= weights.sum()

        rng = np.random.default_rng(seed)
        while True:
            picks = rng.choice(len(self._shards), size=batch_size, p=weights)
            out = np.empty((batch_size, window), dtype=np.int32)
            for i, s in enumerate(picks):
                start = rng.integers(0, usable[s] + 1)
                out[i] = self._shards[s][start : start + window]
            yield out

    def sequential(self, batch_size: int, seq_len: int) -> Iterator[np.ndarray]:
        """Deterministic non-overlapping pass, for validation.

        Evaluation must not resample randomly: a val loss computed on different
        windows each time is not comparable across steps.
        """
        window = seq_len + 1
        rows: list[np.ndarray] = []
        for shard in self._shards:
            for start in range(0, shard.size - window, seq_len):
                rows.append(np.asarray(shard[start : start + window], dtype=np.int32))
                if len(rows) == batch_size:
                    yield np.stack(rows)
                    rows = []

    def cycle_sequential(self, batch_size: int, seq_len: int) -> Iterator[np.ndarray]:
        """``sequential`` restarted forever, so eval loops never run dry."""
        while True:
            yielded = False
            for batch in self.sequential(batch_size, seq_len):
                yielded = True
                yield batch
            if not yielded:
                raise ValueError("dataset too small for one sequential batch")


def describe(data_dir: str, split: str = "train") -> str:
    ds = TokenDataset(data_dir, split)
    return (
        f"{split}: {ds.total_tokens:,} tokens in {len(ds.manifest['shards'])} shard(s), "
        f"vocab {ds.vocab_size:,}, {ds.total_tokens * 2 / 1e9:.2f} GB on disk"
    )
