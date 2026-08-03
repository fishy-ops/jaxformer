"""Generate text from a trained torch_ref checkpoint, to see what the model produces.

Loads a checkpoint saved by torch_ref/train_loop.py plus the corpus tokenizer, and
samples a few continuations with the KV cache. At a val loss around 4.5 (this undertrained
55M model) the output is English-shaped -- real words, local structure -- rather than
coherent, which is the honest expectation for the compute budget.

  python scripts/gen_sample.py --ckpt C:\\jfrun\\ckpt_15000.pt --tokenizer C:\\jfdata\\tokenizer.json
"""

from __future__ import annotations

import argparse

import torch
from tokenizers import Tokenizer

from jaxformer.config import DEFAULT_MODEL
from torch_ref.model import Transformer


@torch.no_grad()
def generate(model, ids, max_new, temperature, top_k, eot_id, device):
    model.eval()
    total = len(ids) + max_new
    cache = model.init_cache(1, total, dtype=torch.float32, device=device)
    logits, cache = model(torch.tensor([ids], device=device), cache=cache, start_pos=0)
    out = list(ids)
    for i in range(max_new):
        lg = logits[:, -1].float()
        if top_k:
            v, _ = torch.topk(lg, top_k)
            lg[lg < v[:, [-1]]] = -float("inf")
        if temperature > 0:
            nxt = torch.multinomial(torch.softmax(lg / temperature, dim=-1), 1)
        else:
            nxt = lg.argmax(-1, keepdim=True)
        tid = int(nxt)
        out.append(tid)
        if eot_id is not None and tid == eot_id:
            break
        logits, cache = model(nxt, cache=cache, start_pos=len(ids) + i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")

    model = Transformer(DEFAULT_MODEL).to(dev)
    state = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(state["model"])
    print(f"loaded step {state['step']} checkpoint\n")

    prompts = ["The history of", "Scientists have discovered", "In order to learn"]
    torch.manual_seed(0)
    for p in prompts:
        ids = tok.encode(p).ids
        out = generate(model, ids, args.max_new, args.temperature, args.top_k, eot, dev)
        print(f"--- prompt: {p!r}")
        print(tok.decode(out).strip(), "\n")


if __name__ == "__main__":
    main()
