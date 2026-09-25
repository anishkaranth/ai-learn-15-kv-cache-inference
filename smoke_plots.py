"""Matplotlib SVG plots + RESULTS.md writer for the KV-cache smoke run."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams["svg.hashsalt"] = "ai-learn-15"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
_META = {"Date": None}


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata=_META)
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    out.mkdir(exist_ok=True)
    names = []
    rows = m["benchmark"]
    L = [r["total_len"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(L, [r["time_no_cache_s"] for r in rows], "o-", color="#e76f51", label="no cache")
    axes[0].plot(L, [r["time_cache_s"] for r in rows], "o-", color="#2a9d8f", label="KV cache")
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xlabel("final sequence length"); axes[0].set_ylabel("generation wall time (s)")
    axes[0].set_title("Greedy generation time"); axes[0].legend()
    axes[1].plot(L, [r["speedup"] for r in rows], "o-", color="#264653")
    for x, r in zip(L, rows):
        axes[1].annotate(f"{r['speedup']:.1f}x", (x, r["speedup"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("final sequence length"); axes[1].set_ylabel("speedup (no-cache / cache)")
    axes[1].set_title("KV-cache speedup vs length")
    names.append(_save(fig, out / "speedup_vs_length.svg"))

    st = m["per_step"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(st["positions"], st["no_cache_ms"], color="#e76f51", label="no cache (re-run prefix)")
    ax.plot(st["positions"], st["cache_ms"], color="#2a9d8f", label="KV cache (1 token)")
    ax.set_xlabel("decode position"); ax.set_ylabel("step latency (ms)")
    ax.set_title(f"Per-step latency (length {st['positions'][-1] + 1})"); ax.legend()
    names.append(_save(fig, out / "per_step_latency.svg"))

    mem = m["memory"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    T = [r["seq_len"] for r in mem]
    ax.plot(T, [r["kv_cache_bytes"] / 1024 for r in mem], "o-", color="#2a9d8f", label="KV cache (measured)")
    ax.plot(T, [r["no_cache_attn_scores_bytes"] / 1024 for r in mem], "s--", color="#e76f51", label="attn scores per layer, no-cache step")
    ax.axhline(m["model"]["param_bytes"] / 1024, color="#264653", ls=":", label="model params")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("sequence length"); ax.set_ylabel("KiB"); ax.set_title("Memory vs sequence length"); ax.legend(fontsize=8)
    names.append(_save(fig, out / "memory_vs_length.svg"))
    return names


def write_results_md(out: Path, m: Dict[str, Any], plots: List[str]) -> None:
    c = m["config"]; mo = m["model"]
    L = ["# Results -- ai-learn-15-kv-cache-inference", "",
         f"**Seed:** `{m['seed']}` | layers={c['n_layers']} heads={c['n_heads']} d_model={c['d_model']} d_ff={c['d_ff']} vocab={c['vocab_size']} | "
         f"params={mo['n_params']:,} ({mo['param_bytes'] / 1024:.1f} KiB float32) | prompt={m['prompt_len']} tokens, greedy decoding", "",
         "## Equivalence and speed (real smoke run)", "",
         "| final len | new tokens | tokens identical | max abs logit diff | tokens fed (no cache) | tokens fed (cache) | no cache (s) | cache (s) | speedup |",
         "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|"]
    for r in m["benchmark"]:
        L.append(f"| {r['total_len']} | {r['n_new']} | {'yes' if r['tokens_identical'] else 'NO'} | {r['max_abs_logit_diff']:.2e} | "
                 f"{r['tokens_processed_no_cache']:,} | {r['tokens_processed_cache']:,} | {r['time_no_cache_s']:.4f} | {r['time_cache_s']:.4f} | {r['speedup']:.2f}x |")
    L += ["", f"All runs identical: **{m['summary']['all_identical']}**. Worst max abs logit diff: **{m['summary']['worst_max_abs_diff']:.2e}** "
          "(float32 rounding: the two paths do the same math in a different order).", "",
          "## Memory table", "",
          "| seq len | KV cache bytes (measured) | formula 2·L·H·T·d_head·4 | match | bytes/token | KV / params | no-cache attn scores per layer (bytes) |",
          "|---:|---:|---:|:---:|---:|---:|---:|"]
    for r in m["memory"]:
        L.append(f"| {r['seq_len']} | {r['kv_cache_bytes']:,} | {r['formula_bytes']:,} | {'yes' if r['match'] else 'NO'} | {r['kv_per_token_bytes']:,} | {r['kv_vs_params']:.3f} | {r['no_cache_attn_scores_bytes']:,} |")
    ref = m["reference_formula"]
    L += ["", f"Same formula applied to a GPT-2-small-shaped model ({ref['n_layers']} layers, {ref['n_heads']} heads, d_head {ref['d_head']}, fp16, T={ref['seq_len']}, batch={ref['batch']}): "
          f"**{ref['kv_bytes'] / 2**20:.1f} MiB** of KV cache (analytic, not measured).", "",
          "## Plots", ""] + [f"![{p}]({p})" for p in plots] + ["", f"Wall time: {m['runtime_s']:.2f}s on CPU.", ""]
    (out / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
