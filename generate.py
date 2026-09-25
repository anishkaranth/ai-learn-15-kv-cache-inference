"""Greedy generation with and without a KV cache + equivalence checks."""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from model import Config, KVCache, forward_cached, forward_full


def generate_no_cache(prompt: List[int], n_new: int, params: Dict, cfg: Config):
    """Each step re-runs the whole sequence through the model (O(T^2) total token work)."""
    toks = list(prompt)
    step_logits = []
    tokens_processed = 0
    for _ in range(n_new):
        logits = forward_full(toks, params, cfg)
        tokens_processed += len(toks)
        last = logits[-1]
        step_logits.append(last)
        toks.append(int(np.argmax(last)))
    return toks, np.stack(step_logits), tokens_processed


def generate_with_cache(prompt: List[int], n_new: int, params: Dict, cfg: Config):
    """Prefill the prompt once, then feed exactly one new token per step."""
    cache = KVCache(cfg, len(prompt) + n_new)
    toks = list(prompt)
    step_logits = []
    logits = forward_cached(toks, params, cfg, cache)  # prefill
    tokens_processed = len(toks)
    last = logits[-1]
    for i in range(n_new):
        step_logits.append(last)
        nxt = int(np.argmax(last))
        toks.append(nxt)
        if i < n_new - 1:
            last = forward_cached([nxt], params, cfg, cache)[-1]
            tokens_processed += 1
    return toks, np.stack(step_logits), tokens_processed, cache


def compare(prompt: List[int], n_new: int, params: Dict, cfg: Config) -> Dict:
    t0 = time.perf_counter()
    a_toks, a_log, a_work = generate_no_cache(prompt, n_new, params, cfg)
    t1 = time.perf_counter()
    b_toks, b_log, b_work, cache = generate_with_cache(prompt, n_new, params, cfg)
    t2 = time.perf_counter()
    return {
        "prompt_len": len(prompt), "n_new": n_new, "total_len": len(a_toks),
        "tokens_identical": a_toks == b_toks,
        "n_token_mismatches": int(sum(x != y for x, y in zip(a_toks, b_toks))),
        "max_abs_logit_diff": float(np.max(np.abs(a_log - b_log))),
        "time_no_cache_s": t1 - t0, "time_cache_s": t2 - t1,
        "tokens_processed_no_cache": a_work, "tokens_processed_cache": b_work,
        "cache_bytes": cache.used_bytes(), "generated": b_toks[len(prompt):],
    }
