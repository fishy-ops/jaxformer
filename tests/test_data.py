"""Tokenizer and corpus-shard tests.

All of these run offline against synthetic text. Corpus prep is a multi-hour
streaming job; the bugs in it (a shard boundary off by one, a non-deterministic
"deterministic" loader, silent uint16 overflow) are all reproducible at toy scale,
and finding them there costs seconds instead of a re-run.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from jaxformer.data import TOKEN_DTYPE, ShardWriter, TokenDataset, tokenize_stream
from jaxformer.tokenizer import EOT, eot_id, load_tokenizer, train_tokenizer


def synthetic_docs(n: int = 400) -> list[str]:
    """Repetitive pseudo-text — BPE needs recurring substrings to learn merges."""
    rng = np.random.default_rng(0)
    words = ["the", "model", "attention", "kernel", "tensor", "gradient", "shard", "token"]
    return [
        " ".join(rng.choice(words, size=int(rng.integers(20, 60)))) + "."
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    train_tokenizer(synthetic_docs(), vocab_size=500, out_path=str(path))
    return load_tokenizer(str(path))


def test_tokenizer_round_trips_text(tokenizer):
    text = "the attention kernel shards every gradient tensor."
    assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_tokenizer_is_byte_level_so_nothing_is_oov(tokenizer):
    """Byte-level BPE must encode text it never saw, including non-ASCII."""
    for text in ("éàü 你好", "\U0001f680 emoji", "\x00\x01raw bytes"):
        assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_eot_token_exists_and_is_stable(tokenizer):
    assert tokenizer.token_to_id(EOT) is not None
    assert eot_id(tokenizer) == tokenizer.token_to_id(EOT)


def test_eot_is_atomic(tokenizer):
    """EOT must encode to exactly one id, or document boundaries blur."""
    assert tokenizer.encode(EOT).ids == [eot_id(tokenizer)]


def test_vocab_size_is_respected(tokenizer):
    assert tokenizer.get_vocab_size() <= 500


# ---------------------------------------------------------------------------
# Shard writing
# ---------------------------------------------------------------------------


def test_shards_split_at_the_requested_size(tmp_path):
    w = ShardWriter(str(tmp_path), "train", shard_tokens=100, vocab_size=256)
    for _ in range(25):
        w.add(range(10))  # 250 tokens total
    manifest = w.close()

    assert manifest["total_tokens"] == 250
    assert [s["tokens"] for s in manifest["shards"]] == [100, 100, 50]
    for s in manifest["shards"]:
        assert os.path.getsize(tmp_path / s["file"]) == s["tokens"] * 2  # uint16


def test_shard_contents_are_the_concatenated_stream(tmp_path):
    w = ShardWriter(str(tmp_path), "train", shard_tokens=64, vocab_size=1024)
    expected = list(range(500))
    w.add(expected)
    w.close()

    ds = TokenDataset(str(tmp_path), "train")
    joined = np.concatenate([np.asarray(s) for s in ds._shards])
    np.testing.assert_array_equal(joined, np.array(expected, dtype=TOKEN_DTYPE))


def test_writer_rejects_vocab_too_large_for_uint16(tmp_path):
    with pytest.raises(ValueError, match="exceeds uint16"):
        ShardWriter(str(tmp_path), "train", shard_tokens=10, vocab_size=70_000)


def test_manifest_is_valid_json_with_expected_keys(tmp_path):
    w = ShardWriter(str(tmp_path), "val", shard_tokens=32, vocab_size=256)
    w.add(range(100))
    w.close()
    with open(tmp_path / "val_manifest.json") as f:
        m = json.load(f)
    assert {"split", "dtype", "vocab_size", "total_tokens", "shards"} <= m.keys()
    assert m["dtype"] == "uint16"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path):
    w = ShardWriter(str(tmp_path), "train", shard_tokens=500, vocab_size=1024)
    w.add(range(2000))
    w.close()
    return TokenDataset(str(tmp_path), "train")


def test_batch_shape_and_dtype(dataset):
    batch = next(dataset.batches(batch_size=4, seq_len=16, seed=0))
    # seq_len + 1: the training step slices inputs and shifted labels out of this.
    assert batch.shape == (4, 17)
    assert batch.dtype == np.int32


def test_batches_are_deterministic_under_a_seed(dataset):
    a = next(dataset.batches(4, 16, seed=7))
    b = next(dataset.batches(4, 16, seed=7))
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_batches(dataset):
    a = next(dataset.batches(4, 16, seed=1))
    b = next(dataset.batches(4, 16, seed=2))
    assert not np.array_equal(a, b)


def test_windows_are_contiguous_runs_of_the_corpus(dataset):
    """Each row must be consecutive token ids — this corpus counts up from 0."""
    batch = next(dataset.batches(8, 16, seed=3))
    for row in batch:
        np.testing.assert_array_equal(np.diff(row), np.ones(16, dtype=np.int32))


def test_sequential_is_reproducible_and_ordered(dataset):
    a = [b.copy() for _, b in zip(range(3), dataset.sequential(2, 16))]
    b = [x.copy() for _, x in zip(range(3), dataset.sequential(2, 16))]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    # First window starts at the very beginning of shard 0.
    assert a[0][0][0] == 0


def test_cycle_sequential_never_runs_dry(dataset):
    it = dataset.cycle_sequential(2, 16)
    got = [next(it) for _ in range(200)]
    assert len(got) == 200


def test_loader_rejects_windows_longer_than_any_shard(tmp_path):
    w = ShardWriter(str(tmp_path), "train", shard_tokens=32, vocab_size=256)
    w.add(range(64))
    w.close()
    ds = TokenDataset(str(tmp_path), "train")
    with pytest.raises(ValueError, match="long enough"):
        next(ds.batches(2, seq_len=512))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_tokenize_stream_writes_a_loadable_corpus(tmp_path, tokenizer):
    writer = ShardWriter(
        str(tmp_path), "train", shard_tokens=1000, vocab_size=tokenizer.get_vocab_size()
    )
    manifest = tokenize_stream(
        synthetic_docs(200), tokenizer, writer, target_tokens=5000, batch_size=32
    )
    assert manifest["total_tokens"] > 0

    ds = TokenDataset(str(tmp_path), "train")
    assert ds.total_tokens == manifest["total_tokens"]
    batch = next(ds.batches(2, 16, seed=0))
    assert batch.max() < tokenizer.get_vocab_size()

    # Every document is EOT-terminated, so the marker must appear in the stream.
    joined = np.concatenate([np.asarray(s) for s in ds._shards])
    assert (joined == eot_id(tokenizer)).sum() >= 1


def test_target_tokens_is_respected_within_one_document(tmp_path, tokenizer):
    """Stopping must be per-document, not per-batch.

    Checking the target only between batches overshoots by up to a full batch. With
    the real corpus that is 1024 documents of ~1.4k tokens — enough to turn a 5M-token
    validation split into a 6.4M-token one, or to make a small smoke run 4x too big.
    """
    target = 2000
    writer = ShardWriter(
        str(tmp_path), "train", shard_tokens=10_000, vocab_size=tokenizer.get_vocab_size()
    )
    manifest = tokenize_stream(
        synthetic_docs(400), tokenizer, writer, target_tokens=target, batch_size=64
    )
    longest = max(len(tokenizer.encode(d).ids) for d in synthetic_docs(400)) + 1
    assert target <= manifest["total_tokens"] <= target + longest


def test_tokenized_ids_stay_in_vocab_range(tmp_path, tokenizer):
    """uint16 wraps silently; an id above vocab_size would corrupt the corpus."""
    writer = ShardWriter(
        str(tmp_path), "train", shard_tokens=10_000, vocab_size=tokenizer.get_vocab_size()
    )
    tokenize_stream(synthetic_docs(100), tokenizer, writer, target_tokens=3000)
    ds = TokenDataset(str(tmp_path), "train")
    joined = np.concatenate([np.asarray(s) for s in ds._shards])
    assert joined.min() >= 0
    assert joined.max() < tokenizer.get_vocab_size()
