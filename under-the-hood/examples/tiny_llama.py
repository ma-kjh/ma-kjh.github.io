#!/usr/bin/env python3
"""An executable, deliberately tiny Llama-style decoder for learning.

Run: python tiny_llama.py --verify
     python tiny_llama.py --export-trace trace.json

Requires NumPy only. All weights are deterministic RANDOM, UNTRAINED float64
arrays. The vocabulary is a hand-written list of 32 labels, not a tokenizer.
Outputs demonstrate computation and have no language-model quality meaning.

Compared with Llama 3.2 1B: this has 2 layers, width 32, 4 query / 2 KV heads,
head dimension 8, and an FFN width of 96. It uses basic, unscaled RoPE with
theta=10000, not the Llama 3 frequency-scaling recipe. There is no padding,
training, dropout, sampling policy, quantization, or production GPU kernel.
We retain the main pre-norm decoder flow, RMSNorm, RoPE, GQA, SwiGLU, residual
paths, final norm, and a tied embedding/output projection.

References (architecture, not pretrained equivalence):
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json
https://arxiv.org/abs/2305.13245
https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


VOCAB = [
    "<bos>", "I", "love", "learning", "about", "attention", ".", "How",
    "does", "a", "model", "use", "context", "?", "the", "key", "value",
    "query", "cache", "token", "is", "shared", "across", "heads", "hello",
    "world", "<eos>", "and", "memory", "position", "new", "old",
]


@dataclass(frozen=True)
class Config:
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 8
    hidden_dim: int = 96
    vocab_size: int = 32
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        if self.n_kv_heads < 1 or self.n_heads % self.n_kv_heads:
            raise ValueError("query-head count must be divisible by KV-head count")
        if self.head_dim % 2 or min(self.n_layers, self.hidden_dim, self.vocab_size) < 1:
            raise ValueError("head_dim must be even and dimensions must be positive")


@dataclass
class KVCache:
    """A layer's stored K/V: [batch, KV heads, cached positions, head dimension]."""

    k: np.ndarray  # Keys AFTER RoPE; no repetition to the number of query heads.
    v: np.ndarray  # Values are not rotated.
    positions: np.ndarray  # [cached positions]; shared by this unpadded batch.

    @property
    def nbytes(self) -> int:
        """K/V tensor payload only; excludes position metadata and object overhead."""
        return self.k.nbytes + self.v.nbytes


@dataclass
class ForwardResult:
    logits: np.ndarray  # [batch, NEW positions, vocabulary]
    cache: list[KVCache] | None
    trace: dict[str, Any] | None


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """1. Normalize each position across its feature dimension, then scale."""
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def apply_rope(x: np.ndarray, positions: np.ndarray, theta: float = 10000.0) -> np.ndarray:
    """2. Rotate Q or K [B,H,T,dh] using ABSOLUTE positions [T].

    This uses Hugging Face Llama's split-half convention: channel i pairs with
    channel i+dh/2. Interleaved-pair RoPE is equivalent after a corresponding
    weight/channel permutation, but those layouts must not be mixed.
    """
    if x.ndim != 4 or x.shape[-1] % 2 or positions.shape != (x.shape[-2],):
        raise ValueError("expected x=[B,H,T,even dh] and positions=[T]")
    d = x.shape[-1]
    inv_freq = theta ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    angles = positions[:, None] * inv_freq[None, :]  # [T,dh/2]
    angles = np.concatenate([angles, angles], axis=-1)[None, None, :, :]
    rotated_half = np.concatenate([-x[..., d // 2 :], x[..., : d // 2]], axis=-1)
    return x * np.cos(angles) + rotated_half * np.sin(angles)


def grouped_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    query_positions: np.ndarray,
    key_positions: np.ndarray,
    return_weights: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """3. Causal GQA for prefill, single-token decode, or a multi-token chunk.

    q:[B,Hq,Tq,dh], k/v:[B,Hkv,Tk,dh]. Query heads are grouped in contiguous
    blocks. Reshaping q and using einsum shares K/V logically; the stored cache
    is never expanded. Masking compares absolute positions, so cached chunks
    cannot accidentally see future tokens inside their current chunk.
    """
    b, hq, tq, dh = q.shape
    bk, hkv, tk, dkey = k.shape
    if bk != b or dkey != dh or v.shape != k.shape or hq % hkv:
        raise ValueError("incompatible query/key/value shapes")
    if query_positions.shape != (tq,) or key_positions.shape != (tk,):
        raise ValueError("position lengths must match the query/key lengths")
    groups = hq // hkv
    q_grouped = q.reshape(b, hkv, groups, tq, dh)
    scores = np.einsum("bhgtd,bhsd->bhgts", q_grouped, k) / np.sqrt(dh)
    visible = key_positions[None, :] <= query_positions[:, None]  # [Tq,Tk]
    if not np.all(visible.any(axis=-1)):
        raise ValueError("every query must have at least one visible key")
    scores = np.where(visible[None, None, None, :, :], scores, -np.inf)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    out = np.einsum("bhgts,bhsd->bhgtd", weights, v).reshape(b, hq, tq, dh)
    if return_weights:
        return out, weights.reshape(b, hq, tq, tk)
    return out


def silu(x: np.ndarray) -> np.ndarray:
    """4. SiLU(x)=x*sigmoid(x), with a stable sigmoid for either sign."""
    return x * np.exp(-np.logaddexp(0.0, -x))


def decoder_block(
    x: np.ndarray,
    w: dict[str, np.ndarray],
    config: Config,
    cache: KVCache | None = None,
    trace: bool = False,
) -> tuple[np.ndarray, KVCache, dict[str, Any] | None]:
    """5. One pre-norm block: RMSNorm -> GQA -> residual -> RMSNorm -> SwiGLU -> residual."""
    b, t, _ = x.shape
    past = 0 if cache is None else cache.k.shape[2]
    positions = np.arange(past, past + t, dtype=np.int64)
    normed = rms_norm(x, w["input_norm"], config.norm_eps)

    # Projections: [B,T,D] @ [D,H*dh] -> [B,H,T,dh].
    q_raw = (normed @ w["q_proj"]).reshape(b, t, config.n_heads, config.head_dim).transpose(0, 2, 1, 3)
    k_raw = (normed @ w["k_proj"]).reshape(b, t, config.n_kv_heads, config.head_dim).transpose(0, 2, 1, 3)
    v_new = (normed @ w["v_proj"]).reshape(b, t, config.n_kv_heads, config.head_dim).transpose(0, 2, 1, 3)
    q = apply_rope(q_raw, positions, config.rope_theta)
    k_new = apply_rope(k_raw, positions, config.rope_theta)
    k = k_new if cache is None else np.concatenate([cache.k, k_new], axis=2)
    v = v_new if cache is None else np.concatenate([cache.v, v_new], axis=2)
    all_positions = positions if cache is None else np.concatenate([cache.positions, positions])
    updated_cache = KVCache(k.copy(), v.copy(), all_positions.copy())

    attended, weights = grouped_attention(q, k, v, positions, all_positions, return_weights=True)
    joined = attended.transpose(0, 2, 1, 3).reshape(b, t, config.d_model)
    attention_output = joined @ w["o_proj"]
    after_attention = x + attention_output
    ffn_input = rms_norm(after_attention, w["post_attention_norm"], config.norm_eps)
    # SwiGLU: the activated gate and the up branch are multiplied elementwise.
    gate = silu(ffn_input @ w["gate_proj"])
    up = ffn_input @ w["up_proj"]
    ffn_output = (gate * up) @ w["down_proj"]
    output = after_attention + ffn_output

    details = None
    if trace:
        details = {
            "positions": positions.tolist(),
            "key_positions": all_positions.tolist(),
            "shapes": {
                "input": list(x.shape), "normed": list(normed.shape),
                "q": list(q.shape), "new_k": list(k_new.shape), "new_v": list(v_new.shape),
                "cached_k": list(k.shape), "cached_v": list(v.shape),
                "attention_weights": list(weights.shape), "joined_heads": list(joined.shape),
                "gate": list(gate.shape), "up": list(up.shape), "ffn_output": list(ffn_output.shape),
                "output": list(output.shape),
            },
            "cache_payload_bytes": updated_cache.nbytes,
            "causal_mask": (all_positions[None, :] <= positions[:, None]).astype(int).tolist(),
            "attention_weights": np.round(weights[0], 8).tolist(),
            "first_query_before_rope": np.round(q_raw[0, 0, 0], 8).tolist(),
            "first_query_after_rope": np.round(q[0, 0, 0], 8).tolist(),
            "first_input_values": np.round(x[0, 0, :8], 8).tolist(),
            "first_output_values": np.round(output[0, 0, :8], 8).tolist(),
            "residual_rms": {
                "input": float(np.sqrt(np.mean(x * x))),
                "after_attention": float(np.sqrt(np.mean(after_attention * after_attention))),
                "output": float(np.sqrt(np.mean(output * output))),
            },
        }
    return output, updated_cache, details


class TinyLlama:
    """6. Embed -> N decoder blocks -> final RMSNorm -> vocabulary logits."""

    def __init__(self, config: Config | None = None, seed: int = 7) -> None:
        self.config = Config() if config is None else config
        self.seed = seed
        c = self.config
        rng = np.random.default_rng(seed)
        matrix = lambda n, m: rng.normal(0.0, 1.0 / np.sqrt(n), size=(n, m))
        self.embedding = rng.normal(0.0, 0.2, size=(c.vocab_size, c.d_model))
        self.layers = []
        for _ in range(c.n_layers):
            self.layers.append({
                "input_norm": np.ones(c.d_model),
                "q_proj": matrix(c.d_model, c.n_heads * c.head_dim),
                "k_proj": matrix(c.d_model, c.n_kv_heads * c.head_dim),
                "v_proj": matrix(c.d_model, c.n_kv_heads * c.head_dim),
                "o_proj": matrix(c.n_heads * c.head_dim, c.d_model),
                "post_attention_norm": np.ones(c.d_model),
                "gate_proj": matrix(c.d_model, c.hidden_dim),
                "up_proj": matrix(c.d_model, c.hidden_dim),
                "down_proj": matrix(c.hidden_dim, c.d_model),
            })
        self.final_norm = np.ones(c.d_model)
        self.lm_head = self.embedding.T  # Tied weights; [D,vocabulary].

    def forward(
        self,
        token_ids: np.ndarray | list[int] | list[list[int]],
        cache: list[KVCache] | None = None,
        use_cache: bool = False,
        trace: bool = False,
    ) -> ForwardResult:
        """7. Process only NEW token IDs; prior context is supplied via cache.

        A cached call returns logits for its new positions only. Cached objects
        are not mutated, so alternative continuations can reuse a prefix safely.
        All batches here are unpadded and share the same position indices.
        """
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2 or min(ids.shape) < 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("token_ids must be a nonempty integer array [B,T] or [T]")
        if np.any(ids < 0) or np.any(ids >= self.config.vocab_size):
            raise ValueError("token id outside vocabulary")
        if cache is not None:
            if len(cache) != self.config.n_layers:
                raise ValueError("one cache entry is required for each decoder layer")
            for layer_cache in cache:
                expected = (ids.shape[0], self.config.n_kv_heads, len(layer_cache.positions), self.config.head_dim)
                if layer_cache.k.shape != expected or layer_cache.v.shape != expected:
                    raise ValueError("cache batch/head/dimension mismatch")
                if not np.array_equal(layer_cache.positions, np.arange(len(layer_cache.positions))):
                    raise ValueError("this model expects a contiguous prefix starting at zero")
                if not np.array_equal(layer_cache.positions, cache[0].positions):
                    raise ValueError("all layer caches must contain the same prefix positions")
        x = self.embedding[ids]  # [B,T,D].
        new_cache, layer_traces = [], []
        for i, weights in enumerate(self.layers):
            x, layer_cache, details = decoder_block(x, weights, self.config, None if cache is None else cache[i], trace)
            new_cache.append(layer_cache)
            if details is not None:
                details["layer"] = i
                layer_traces.append(details)
        x = rms_norm(x, self.final_norm, self.config.norm_eps)
        logits = x @ self.lm_head  # [B,T,V]; position t predicts token t+1.
        details = None if not trace else {
            "token_ids": ids.tolist(),
            "tokens": [[VOCAB[i] if i < len(VOCAB) else str(i) for i in row] for row in ids.tolist()],
            "input_shape": list(ids.shape), "embedding_shape": [*ids.shape, self.config.d_model],
            "layers": layer_traces, "final_hidden_shape": list(x.shape), "logits_shape": list(logits.shape),
            "cache_payload_bytes": sum(item.nbytes for item in new_cache),
        }
        return ForwardResult(logits, new_cache if use_cache else None, details)


def verify() -> dict[str, Any]:
    """8. Check distinct correctness properties against uncached/reference paths."""
    model = TinyLlama()
    c = model.config
    ids = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=np.int64)
    full = model.forward(ids).logits
    tolerance = 1e-10
    checks = {}

    # A two-item batch guards against silently mixing batch/head dimensions.
    cache, single_parts = None, []
    for t in range(ids.shape[1]):
        result = model.forward(ids[:, t : t + 1], cache=cache, use_cache=True)
        cache = result.cache
        single_parts.append(result.logits)
    single_error = float(np.max(np.abs(full - np.concatenate(single_parts, axis=1))))
    assert single_error < tolerance, f"single-token cache mismatch: {single_error}"
    checks["cached_single_vs_full_max_abs_error"] = single_error

    cache, chunk_parts = None, []
    for start, end in [(0, 2), (2, 5), (5, 6)]:
        result = model.forward(ids[:, start:end], cache=cache, use_cache=True)
        cache = result.cache
        chunk_parts.append(result.logits)
    chunk_error = float(np.max(np.abs(full - np.concatenate(chunk_parts, axis=1))))
    assert chunk_error < tolerance, f"chunked cache mismatch: {chunk_error}"
    checks["cached_chunked_vs_full_max_abs_error"] = chunk_error

    altered = ids.copy()
    altered[:, 3:] = (altered[:, 3:] + 11) % c.vocab_size
    future_error = float(np.max(np.abs(full[:, :3] - model.forward(altered).logits[:, :3])))
    assert future_error < tolerance, f"future tokens leaked: {future_error}"
    checks["future_change_earlier_logits_max_abs_error"] = future_error

    rng = np.random.default_rng(17)
    q = rng.normal(size=(2, c.n_heads, 3, c.head_dim))
    k = rng.normal(size=(2, c.n_kv_heads, 5, c.head_dim))
    v = rng.normal(size=k.shape)
    qpos, kpos = np.arange(2, 5), np.arange(5)
    actual, weights = grouped_attention(q, k, v, qpos, kpos, True)
    repeated_k = np.repeat(k, c.n_heads // c.n_kv_heads, axis=1)
    repeated_v = np.repeat(v, c.n_heads // c.n_kv_heads, axis=1)
    scores = q @ repeated_k.swapaxes(-1, -2) / np.sqrt(c.head_dim)
    scores = np.where(kpos[None, :] <= qpos[:, None], scores, -np.inf)
    reference_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    reference_weights /= reference_weights.sum(axis=-1, keepdims=True)
    reference = reference_weights @ repeated_v
    gqa_error = float(np.max(np.abs(actual - reference)))
    assert gqa_error < tolerance, f"GQA differs from expanded reference: {gqa_error}"
    assert np.max(np.abs(weights[..., 0, 3:])) == 0.0
    assert np.max(np.abs(weights.sum(axis=-1) - 1)) < tolerance
    checks["gqa_vs_explicit_head_repeat_max_abs_error"] = gqa_error
    checks["masked_attention_weight_max"] = float(np.max(np.abs(weights[..., 0, 3:])))

    rope_input = rng.normal(size=(1, 1, 1, c.head_dim))
    at_zero = apply_rope(rope_input, np.array([0]))
    at_one = apply_rope(rope_input, np.array([1]))
    assert np.allclose(at_zero, rope_input, atol=tolerance, rtol=0)
    position_change = float(np.max(np.abs(at_one - at_zero)))
    assert position_change > 1e-3, "RoPE must depend on absolute position"
    norm_error = float(np.max(np.abs(np.linalg.norm(at_one, axis=-1) - np.linalg.norm(rope_input, axis=-1))))
    assert norm_error < tolerance, "RoPE rotation must preserve vector norm"
    checks["rope_position_0_vs_1_max_abs_difference"] = position_change
    checks["rope_norm_preservation_max_abs_error"] = norm_error

    assert cache is not None
    expected_shape = (2, c.n_kv_heads, 6, c.head_dim)
    assert all(item.k.shape == expected_shape and item.v.shape == expected_shape for item in cache)
    expected_bytes = 2 * 2 * c.n_layers * 6 * c.n_kv_heads * c.head_dim * np.dtype(np.float64).itemsize
    actual_bytes = sum(item.nbytes for item in cache)
    assert actual_bytes == expected_bytes
    checks["cache_shape_per_tensor"] = list(expected_shape)
    checks["cache_payload_bytes_batch_2_length_6"] = actual_bytes
    checks["expected_cache_payload_bytes"] = expected_bytes

    prefix = model.forward(ids[:, :2], use_cache=True)
    before = [item.k.copy() for item in prefix.cache]
    model.forward(ids[:, 2:4], cache=prefix.cache, use_cache=True)
    assert all(np.array_equal(old, item.k) for old, item in zip(before, prefix.cache))
    checks["prefix_cache_unchanged_after_continuation"] = True
    return {"passed": True, "dtype": "float64", "tolerance": tolerance, "checks": checks}


def export_trace() -> dict[str, Any]:
    """9. Export real measured values for an interactive educational walkthrough."""
    model = TinyLlama()
    prompt = [1, 2, 3, 4, 5, 6]  # I love learning about attention .
    prefill = model.forward(prompt, use_cache=True, trace=True)
    next_id = int(prefill.logits[0, -1].argmax())
    decoded = model.forward([next_id], cache=prefill.cache, use_cache=True, trace=True)
    chunk = model.forward([next_id, 28], cache=prefill.cache, use_cache=True, trace=True)
    records = []
    for name, result in [("prefill", prefill), ("single_token_decode", decoded), ("chunked_decode", chunk)]:
        detail = result.trace
        detail["name"] = name
        last = result.logits[0, -1]
        probabilities = np.exp(last - last.max())
        probabilities /= probabilities.sum()
        top_ids = np.argsort(last)[-5:][::-1]
        detail["last_position_top5"] = [
            {"token_id": int(i), "token": VOCAB[i], "logit": float(last[i]), "probability": float(probabilities[i])}
            for i in top_ids
        ]
        records.append(detail)
    return {
        "schema_version": 1,
        "model": "tiny-llama-style-educational",
        "weights": "deterministic random untrained",
        "tokenizer": "hand-written 32-label vocabulary; not a real tokenizer",
        "numpy_version": np.__version__, "seed": model.seed, "dtype": "float64",
        "config": asdict(model.config), "vocabulary": VOCAB,
        "cache_bytes_formula": "2 * batch * layers * cached_tokens * kv_heads * head_dim * bytes_per_element",
        "verification": verify(), "passes": records,
        "notes": [
            "All numbers are measured by this NumPy reference; they are not pretrained-model results.",
            "Prefill computes the distribution for the first generated token.",
            "The single-token and chunked demonstrations each start from the same six-token prefix.",
            "Chunked decode uses the greedy next ID followed by a supplied token ID for demonstration.",
            "Cache byte counts include K/V payload, excluding metadata and temporary attention arrays.",
            "Keys are cached after RoPE; only two KV heads are stored for four query heads.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="run meaningful numerical correctness checks")
    parser.add_argument("--trace", action="store_true", help="print measured walkthrough JSON")
    parser.add_argument("--export-trace", type=Path, help="write measured walkthrough JSON to this file")
    args = parser.parse_args()
    if args.trace or args.export_trace:
        payload = export_trace()
        if args.export_trace:
            args.export_trace.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Wrote validated trace: {args.export_trace}")
        if args.trace:
            print(json.dumps(payload, indent=2, allow_nan=False))
    if args.verify or not (args.trace or args.export_trace):
        print(json.dumps(verify(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
