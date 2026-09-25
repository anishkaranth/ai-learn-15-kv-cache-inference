"""KV-cache memory accounting: measured bytes of our cache + the analytic formula."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from model import Config, KVCache


def kv_bytes(n_layers: int, n_heads: int, d_head: int, seq_len: int, batch: int = 1, bytes_per: int = 4) -> int:
    """2 (K and V) x layers x heads x seq x d_head x batch x bytes-per-element."""
    return 2 * n_layers * n_heads * seq_len * d_head * batch * bytes_per


def memory_table(cfg: Config, lengths: List[int], param_bytes: int) -> List[Dict]:
    rows = []
    for T in lengths:
        c = KVCache(cfg, T)
        c.length = T
        measured = c.used_bytes()
        formula = kv_bytes(cfg.n_layers, cfg.n_heads, cfg.d_head, T, bytes_per=np.dtype(c.k.dtype).itemsize)
        # attention-score matrix a no-cache step materialises at length T (H x T x T per layer, float32)
        rows.append({"seq_len": T, "kv_cache_bytes": measured, "formula_bytes": formula,
                     "match": measured == formula, "kv_per_token_bytes": measured // T,
                     "kv_vs_params": round(measured / param_bytes, 4),
                     "no_cache_attn_scores_bytes": cfg.n_heads * T * T * 4})
    return rows
