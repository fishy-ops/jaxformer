"""Render the benchmark JSON to charts. No hand-made images: every figure is
regenerable from bench/results/*.json so the numbers in the README and the pictures
can never drift apart.

    python -m bench.plots            # reads bench/results/attention_*.json
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG_DIR = os.path.join(os.path.dirname(HERE), "docs", "figures")

# Stable per-impl styling so the three charts read consistently.
STYLE = {
    "naive":        ("Naive matmul",        "#9e9e9e", "o", "-"),
    "sdpa_math":    ("SDPA math",           "#c62828", "s", "-"),
    "sdpa_mem_eff": ("SDPA mem-efficient",  "#2e7d32", "D", "-"),
    "sdpa_flash":   ("SDPA flash",          "#ef6c00", "x", ":"),
    "ours_v1":      ("Ours v1 (fp32)",      "#1565c0", "^", "-"),
    "ours_v2":      ("Ours v2 (fp16 WMMA)", "#6a1b9a", "P", "-"),
}
ORDER = ["naive", "sdpa_math", "sdpa_mem_eff", "sdpa_flash", "ours_v1", "ours_v2"]


def _series(rows, impl, key):
    """(seq_lens, values) for one impl/metric, skipping rows that errored."""
    pts = [(r["seq_len"], r.get(key)) for r in rows if r["impl"] == impl and key in r and "error" not in r]
    pts.sort()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return xs, ys


def _failed_lengths(rows, impl):
    return sorted(r["seq_len"] for r in rows if r["impl"] == impl and "error" in r)


def _line_chart(rows, key, ylabel, title, path, logy=True):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for impl in ORDER:
        label, color, marker, ls = STYLE[impl]
        xs, ys = _series(rows, impl, key)
        if xs:
            ax.plot(xs, ys, marker=marker, linestyle=ls, color=color, label=label, linewidth=2, markersize=6)
    # Annotate backends that failed on this GPU (flash on Turing).
    failed = _failed_lengths(rows, "sdpa_flash")
    if failed and key == "latency_ms":
        ax.annotate("SDPA flash: no Turing kernel\n(fails at every length)",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8,
                    color=STYLE["sdpa_flash"][1], va="bottom")
    ax.set_xlabel("sequence length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    xs_all = sorted({r["seq_len"] for r in rows})
    ax.set_xticks(xs_all)
    ax.set_xticklabels([str(x) for x in xs_all])
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def render(json_path):
    with open(json_path) as f:
        data = json.load(f)
    rows = data["rows"]
    dev = data["device"]
    cc = data.get("compute_capability", "")
    dtype = data.get("dtype", "fp32")
    cfg = data["config"]
    causal = "causal" if cfg.get("causal", True) else "noncausal"
    subtitle = f"{dev} ({cc}), {dtype}, B={cfg['B']} H={cfg['H']} Dh={cfg['head_dim']}, {causal}"
    tag = f"_{dtype}"  # keep fp32 and fp16 figures distinct

    os.makedirs(FIG_DIR, exist_ok=True)
    _line_chart(rows, "latency_ms", "latency (ms, log)",
                f"Attention latency\n{subtitle}",
                os.path.join(FIG_DIR, f"attn_latency{tag}.png"))
    _line_chart(rows, "peak_mb", "peak memory (MB, log)",
                f"Attention peak memory -- O(T) vs O(T^2)\n{subtitle}",
                os.path.join(FIG_DIR, f"attn_memory{tag}.png"))
    _line_chart(rows, "tflops", "achieved TFLOP/s",
                f"Attention throughput\n{subtitle}",
                os.path.join(FIG_DIR, f"attn_tflops{tag}.png"), logy=False)


def render_training():
    """Grouped bars: training throughput and inference latency per hardware target."""
    paths = sorted(glob.glob(os.path.join(HERE, "results", "training_*.json")))
    if not paths:
        return
    rows = [json.load(open(p)) for p in paths]
    # Stable order: CPU, GPU, TPU as available.
    order = {"jax_cpu": 0, "torch_cuda": 1, "jax_tpu": 2}
    rows.sort(key=lambda r: order.get(r["backend"], 9))
    labels = [f"{r['device'].split('(')[0].strip()}\n({r['backend']}, {r['dtype']})" for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = range(len(rows))
    colors = ["#1565c0", "#2e7d32", "#6a1b9a"][: len(rows)]

    ax1.bar(x, [r["train_tokens_per_sec"] for r in rows], color=colors)
    ax1.set_title("Training throughput")
    ax1.set_ylabel("tokens / sec")
    for i, r in enumerate(rows):
        ax1.text(i, r["train_tokens_per_sec"], f"{r['train_tokens_per_sec']:,.0f}",
                 ha="center", va="bottom", fontsize=9)

    ax2.bar(x, [r["infer_ms_per_token"] for r in rows], color=colors)
    ax2.set_title("Inference latency (single-token decode)")
    ax2.set_ylabel("ms / token")
    for i, r in enumerate(rows):
        ax2.text(i, r["infer_ms_per_token"], f"{r['infer_ms_per_token']:.1f}",
                 ha="center", va="bottom", fontsize=9)

    for ax in (ax1, ax2):
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
    cfg = rows[0].get("config", {})
    fig.suptitle(f"Cross-hardware: same ~55M model  (batch={cfg.get('batch')}, seq={cfg.get('seq')})")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "training_hardware.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def render_loss_curve():
    """Train/val loss from the real-data runs, vs tokens seen: does the model learn,
    and does more compute help? Overlays every train_*_loss.json on a shared token axis."""
    runs = []
    for name, color in [("train_smoke_loss.json", "#1565c0"), ("train_gpu_loss.json", "#6a1b9a")]:
        p = os.path.join(HERE, "results", name)
        if os.path.exists(p):
            runs.append((json.load(open(p)), color))
    if not runs:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ln_vocab = None
    for d, color in runs:
        cfg = d.get("config", {})
        tok = cfg.get("batch", 1) * cfg.get("seq", 1)
        dev = d.get("device", "").split("(")[0].strip() or d.get("device", "")
        label = f"{dev} ({d.get('dtype','')}, {cfg.get('seq')} ctx)"
        vl = d.get("val_log") or []
        if vl:
            ax.plot([v["step"] * tok / 1e6 for v in vl], [v["val_loss"] for v in vl],
                    color=color, marker="D", markersize=5, linewidth=2, label=f"{label} — val")
        tr = d.get("train_log") or []
        ax.plot([l["step"] * tok / 1e6 for l in tr], [l["loss"] for l in tr],
                color=color, alpha=0.35, linewidth=1, label=f"{label} — train")
        ln_vocab = d.get("init_loss_reference_ln_vocab", ln_vocab)
    if ln_vocab:
        ax.axhline(ln_vocab, color="#9e9e9e", ls="--", lw=1,
                   label=f"random init ≈ ln(vocab) = {ln_vocab:.1f}")

    ax.set_xlabel("tokens seen (millions)")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_xscale("log")
    ax.set_title("Training the ~55M model on real fineweb-edu (more compute → lower loss)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "train_loss.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    paths = sorted(glob.glob(os.path.join(HERE, "results", "attention_*.json")))
    if not paths:
        raise SystemExit("no bench/results/attention_*.json found")
    for p in paths:
        render(p)
    render_training()
    render_loss_curve()


if __name__ == "__main__":
    main()
