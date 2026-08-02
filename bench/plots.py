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


def main():
    paths = sorted(glob.glob(os.path.join(HERE, "results", "attention_*.json")))
    if not paths:
        raise SystemExit("no bench/results/attention_*.json found")
    for p in paths:
        render(p)


if __name__ == "__main__":
    main()
