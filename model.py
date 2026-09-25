"""A tiny decoder-only transformer in NumPy (random weights, seed 42).

Two forward paths share the same weights:
  * forward_full(tokens)            -> logits for every position (causal mask), no cache
  * forward_cached(new_tokens, cache) -> logits for only the new tokens, reading/writing a KV cache
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

DTYPE = np.float32


@dataclass
class Config:
    vocab_size: int = 128
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    max_len: int = 1024

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


def init_params(cfg: Config, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    s = 0.08

    def w(*shape):
        return (rng.standard_normal(shape) * s).astype(DTYPE)

    layers = []
    for _ in range(cfg.n_layers):
        layers.append({
            "ln1_g": np.ones(cfg.d_model, DTYPE), "ln1_b": np.zeros(cfg.d_model, DTYPE),
            "W_qkv": w(cfg.d_model, 3 * cfg.d_model), "W_o": w(cfg.d_model, cfg.d_model),
            "ln2_g": np.ones(cfg.d_model, DTYPE), "ln2_b": np.zeros(cfg.d_model, DTYPE),
            "W_1": w(cfg.d_model, cfg.d_ff), "b_1": np.zeros(cfg.d_ff, DTYPE),
            "W_2": w(cfg.d_ff, cfg.d_model), "b_2": np.zeros(cfg.d_model, DTYPE),
        })
    # larger embedding scale so greedy decoding is not degenerate
    return {"tok_emb": (rng.standard_normal((cfg.vocab_size, cfg.d_model)) * 1.0).astype(DTYPE),
            "pos_emb": (rng.standard_normal((cfg.max_len, cfg.d_model)) * 0.5).astype(DTYPE),
            "W_out": (rng.standard_normal((cfg.d_model, cfg.vocab_size)) * 0.5).astype(DTYPE),
            "layers": layers, "lnf_g": np.ones(cfg.d_model, DTYPE), "lnf_b": np.zeros(cfg.d_model, DTYPE)}


def n_params(params: Dict) -> int:
    tot = params["tok_emb"].size + params["pos_emb"].size + params["W_out"].size + params["lnf_g"].size + params["lnf_b"].size
    for L in params["layers"]:
        tot += sum(v.size for v in L.values())
    return int(tot)


def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _split_heads(x, cfg):  # (T, D) -> (H, T, dh)
    T = x.shape[0]
    return x.reshape(T, cfg.n_heads, cfg.d_head).transpose(1, 0, 2)


def _merge_heads(x):  # (H, T, dh) -> (T, D)
    H, T, dh = x.shape
    return x.transpose(1, 0, 2).reshape(T, H * dh)


def _mlp(x, L):
    h = layer_norm(x, L["ln2_g"], L["ln2_b"])
    return gelu(h @ L["W_1"] + L["b_1"]) @ L["W_2"] + L["b_2"]


def _logits(x, params):
    return layer_norm(x, params["lnf_g"], params["lnf_b"]) @ params["W_out"]


def forward_full(tokens: List[int], params: Dict, cfg: Config) -> np.ndarray:
    """No-cache path: recompute Q, K, V for *all* positions. Returns (T, vocab) logits."""
    T = len(tokens)
    x = params["tok_emb"][tokens] + params["pos_emb"][:T]
    mask = np.triu(np.full((T, T), -1e9, DTYPE), k=1)
    scale = DTYPE(1.0 / np.sqrt(cfg.d_head))
    for L in params["layers"]:
        h = layer_norm(x, L["ln1_g"], L["ln1_b"])
        q, k, v = np.split(h @ L["W_qkv"], 3, axis=-1)
        q, k, v = _split_heads(q, cfg), _split_heads(k, cfg), _split_heads(v, cfg)
        att = softmax(q @ k.transpose(0, 2, 1) * scale + mask)
        x = x + _merge_heads(att @ v) @ L["W_o"]
        x = x + _mlp(x, L)
    return _logits(x, params)


class KVCache:
    """Pre-allocated per-layer K and V buffers of shape (H, max_len, d_head)."""

    def __init__(self, cfg: Config, max_len: int):
        shape = (cfg.n_layers, cfg.n_heads, max_len, cfg.d_head)
        self.k = np.zeros(shape, DTYPE)
        self.v = np.zeros(shape, DTYPE)
        self.length = 0

    def used_bytes(self) -> int:
        return int(2 * self.k[:, :, : self.length].nbytes)

    def allocated_bytes(self) -> int:
        return int(self.k.nbytes + self.v.nbytes)


def forward_cached(tokens: List[int], params: Dict, cfg: Config, cache: KVCache) -> np.ndarray:
    """Cached path: process only the *new* tokens, append their K/V, attend over cache[:end].

    Works for a multi-token prefill (tokens = the prompt) or a single decode step.
    Returns logits for the new positions only.
    """
    start = cache.length
    n = len(tokens)
    end = start + n
    x = params["tok_emb"][tokens] + params["pos_emb"][start:end]
    # new query i (absolute pos start+i) may attend to keys 0..start+i
    mask = np.triu(np.full((n, end), -1e9, DTYPE), k=start + 1)
    scale = DTYPE(1.0 / np.sqrt(cfg.d_head))
    for li, L in enumerate(params["layers"]):
        h = layer_norm(x, L["ln1_g"], L["ln1_b"])
        q, k, v = np.split(h @ L["W_qkv"], 3, axis=-1)
        q = _split_heads(q, cfg)
        cache.k[li, :, start:end] = _split_heads(k, cfg)
        cache.v[li, :, start:end] = _split_heads(v, cfg)
        K = cache.k[li, :, :end]
        V = cache.v[li, :, :end]
        att = softmax(q @ K.transpose(0, 2, 1) * scale + mask)
        x = x + _merge_heads(att @ V) @ L["W_o"]
        x = x + _mlp(x, L)
    cache.length = end
    return _logits(x, params)
