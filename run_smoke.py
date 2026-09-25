#!/usr/bin/env python3
"""KV-cache smoke: equivalence, speedup vs length, per-step latency, memory table -> results/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from generate import compare
from memory import kv_bytes, memory_table
from model import Config, KVCache, forward_cached, forward_full, init_params, n_params
from smoke_plots import make_plots, write_results_md

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
PROMPT_LEN = 8
LENGTHS = [16, 32, 64, 128, 256]
MEM_LENGTHS = [64, 128, 256, 512, 1024]


def _compact(js: str) -> str:
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", js)


def per_step_latency(prompt, total, params, cfg):
    """Time every decode step of one run for both paths (teacher-forced on the same tokens)."""
    toks, _, _ = __import__("generate").generate_no_cache(prompt, total - len(prompt), params, cfg)
    pos, nc, kc = [], [], []
    cache = KVCache(cfg, total)
    forward_cached(toks[:len(prompt)], params, cfg, cache)
    for t in range(len(prompt), total):
        a = time.perf_counter(); forward_full(toks[: t + 1], params, cfg); b = time.perf_counter()
        forward_cached([toks[t]], params, cfg, cache); c = time.perf_counter()
        pos.append(t); nc.append(b - a); kc.append(c - b)
    return {"positions": pos, "no_cache_ms": [round(x, 6) for x in nc], "cache_ms": [round(x, 6) for x in kc]}


def main() -> None:
    t0 = time.perf_counter()
    cfg = Config()
    params = init_params(cfg, SEED)
    rng = np.random.default_rng(SEED)
    prompt = [int(x) for x in rng.integers(0, cfg.vocab_size, PROMPT_LEN)]
    np_ = n_params(params)
    compare(prompt, 8, params, cfg)  # warm-up
    bench = []
    for T in LENGTHS:
        r = compare(prompt, T - PROMPT_LEN, params, cfg)
        r["speedup"] = r["time_no_cache_s"] / r["time_cache_s"]
        bench.append(r)
    steps = per_step_latency(prompt, 128, params, cfg)
    mem = memory_table(cfg, MEM_LENGTHS, np_ * 4)
    ref = {"n_layers": 12, "n_heads": 12, "d_head": 64, "seq_len": 1024, "batch": 1, "bytes_per": 2}
    ref["kv_bytes"] = kv_bytes(ref["n_layers"], ref["n_heads"], ref["d_head"], ref["seq_len"], ref["batch"], ref["bytes_per"])
    runtime = time.perf_counter() - t0
    for r in bench:
        for k in ("time_no_cache_s", "time_cache_s", "speedup"):
            r[k] = round(r[k], 5)
    m = {
        "project": "ai-learn-15-kv-cache-inference", "seed": SEED, "prompt_len": PROMPT_LEN, "prompt": prompt,
        "config": cfg.__dict__, "model": {"n_params": np_, "param_bytes": np_ * 4, "dtype": "float32"},
        "benchmark": bench, "per_step": steps, "memory": mem, "reference_formula": ref,
        "summary": {"all_identical": all(r["tokens_identical"] for r in bench),
                    "worst_max_abs_diff": max(r["max_abs_logit_diff"] for r in bench),
                    "speedup_at_max_len": bench[-1]["speedup"], "max_len": bench[-1]["total_len"],
                    "work_ratio_at_max_len": round(bench[-1]["tokens_processed_no_cache"] / bench[-1]["tokens_processed_cache"], 2),
                    "median_step_ms_no_cache_last16": round(float(np.median(steps["no_cache_ms"][-16:])) * 1e3, 4),
                    "median_step_ms_cache_last16": round(float(np.median(steps["cache_ms"][-16:])) * 1e3, 4)},
        "runtime_s": round(runtime, 3),
    }
    RESULTS.mkdir(exist_ok=True)
    plots = make_plots(RESULTS, m)
    m["plots"] = plots
    (RESULTS / "metrics.json").write_text(_compact(json.dumps(m, indent=1)), encoding="utf-8")
    shot = {"project": m["project"], "seed": SEED, "config": cfg.__dict__, "prompt_len": PROMPT_LEN, "lengths": LENGTHS,
            "summary": m["summary"],
            "speedup_by_len": {str(r["total_len"]): r["speedup"] for r in bench},
            "max_abs_diff_by_len": {str(r["total_len"]): r["max_abs_logit_diff"] for r in bench},
            "kv_bytes_by_len": {str(r["seq_len"]): r["kv_cache_bytes"] for r in mem},
            "runtime_s": m["runtime_s"],
            "pass": bool(m["summary"]["all_identical"] and m["summary"]["worst_max_abs_diff"] < 1e-3 and bench[-1]["speedup"] > 1)}
    (RESULTS / "JSON.shot").write_text(json.dumps(shot, indent=2), encoding="utf-8")
    write_results_md(RESULTS, m, plots)
    for r in bench:
        print(f"  len={r['total_len']:4d} identical={r['tokens_identical']} maxdiff={r['max_abs_logit_diff']:.2e} "
              f"no_cache={r['time_no_cache_s']:.3f}s cache={r['time_cache_s']:.3f}s speedup={r['speedup']:.1f}x")
    print(f"pass={shot['pass']} runtime {runtime:.2f}s | wrote {RESULTS}")


if __name__ == "__main__":
    main()
