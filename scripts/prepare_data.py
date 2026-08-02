#!/usr/bin/env python
"""Build the tokenizer and the token shards.

Run once on the dev machine; upload the output directory to Kaggle as a private
Dataset. The whole job streams — nothing but the shards themselves touches disk.

    python scripts/prepare_data.py --out-dir data
    python scripts/prepare_data.py --out-dir data/smoke --target-tokens 2000000 \
        --tokenizer-bytes 20000000    # ~2 min, for verifying the path end to end

Validation shards are drawn from the head of the stream and training shards from the
continuation, so the two are disjoint by construction rather than by a random split
that could leak documents across the boundary.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from jaxformer.config import DEFAULT_DATA, DEFAULT_MODEL
from jaxformer.data import ShardWriter, describe, tokenize_stream
from jaxformer.tokenizer import (
    load_tokenizer,
    save_metadata,
    stream_corpus,
    take_bytes,
    train_tokenizer,
)


def main() -> None:
    d, m = DEFAULT_DATA, DEFAULT_MODEL
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="data")
    p.add_argument("--dataset", default=d.dataset)
    p.add_argument("--subset", default=d.subset)
    p.add_argument("--vocab-size", type=int, default=m.vocab_size)
    p.add_argument("--target-tokens", type=int, default=d.target_tokens)
    p.add_argument("--val-tokens", type=int, default=d.val_tokens)
    p.add_argument("--shard-tokens", type=int, default=d.shard_tokens)
    p.add_argument("--tokenizer-bytes", type=int, default=d.tokenizer_train_bytes)
    p.add_argument(
        "--reuse-tokenizer",
        action="store_true",
        help="skip training if tokenizer.json already exists in --out-dir",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok_path = os.path.join(args.out_dir, "tokenizer.json")

    # --- tokenizer -------------------------------------------------------
    if args.reuse_tokenizer and os.path.exists(tok_path):
        print(f"reusing tokenizer at {tok_path}")
        tok = load_tokenizer(tok_path)
    else:
        print(f"training {args.vocab_size}-token BPE on ~{args.tokenizer_bytes / 1e9:.1f}GB")
        t0 = time.perf_counter()
        tok = train_tokenizer(
            take_bytes(stream_corpus(args.dataset, args.subset), args.tokenizer_bytes),
            vocab_size=args.vocab_size,
            out_path=tok_path,
        )
        print(f"  done in {time.perf_counter() - t0:.0f}s -> {tok_path}")
    save_metadata(tok, os.path.join(args.out_dir, "tokenizer_meta.json"))

    vocab = tok.get_vocab_size()
    if vocab != args.vocab_size:
        # BPE can fall short of the requested size on a small corpus. The model's
        # embedding table is sized from ModelConfig, so a mismatch here would either
        # waste parameters or produce out-of-range ids.
        print(f"WARNING: tokenizer has {vocab} tokens, requested {args.vocab_size}")

    # --- shards ----------------------------------------------------------
    stream = stream_corpus(args.dataset, args.subset)

    print(f"\nwriting validation shards ({args.val_tokens / 1e6:.0f}M tokens)")
    t0 = time.perf_counter()
    tokenize_stream(
        stream,
        tok,
        ShardWriter(args.out_dir, "val", args.shard_tokens, vocab),
        target_tokens=args.val_tokens,
    )
    print(f"  {describe(args.out_dir, 'val')}  [{time.perf_counter() - t0:.0f}s]")

    # Same generator object, so training data continues where validation stopped.
    print(f"\nwriting training shards ({args.target_tokens / 1e9:.2f}B tokens)")
    t0 = time.perf_counter()
    tokenize_stream(
        stream,
        tok,
        ShardWriter(args.out_dir, "train", args.shard_tokens, vocab),
        target_tokens=args.target_tokens,
    )
    print(f"  {describe(args.out_dir, 'train')}  [{(time.perf_counter() - t0) / 60:.1f}min]")

    print(f"\nready. upload {args.out_dir}/ to Kaggle as a private Dataset.")


if __name__ == "__main__":
    main()
    # Hard exit rather than falling off the end of main().
    #
    # Streaming a HuggingFace dataset and abandoning the iterator mid-document (which
    # is exactly what the target-token stop does, twice) leaves the interpreter wedged
    # at shutdown: measured 0% CPU with the process never exiting, both manifests
    # already on disk. Every durable output is written by this point, so there is
    # nothing to lose by skipping teardown.
    #
    # This matters because the real run is multi-hour and backgrounded — a job that
    # never exits never reports completion, and looks indistinguishable from one still
    # working. Flush explicitly first, since os._exit skips buffer flushing too.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
