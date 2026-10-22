#!/usr/bin/env python3
"""Inference-only GPU probe of a REAL pretrained Llama-architecture checkpoint.

This runner has not been executed merely by being included in the tutorial.
Its output is experimental evidence only after a successful local invocation.

Environment: Python >=3.10; a CUDA-compatible PyTorch installation;
             transformers>=4.45,<6, safetensors, sentencepiece.
Example (choose an available GPU yourself):
  python probe_pretrained.py --device cuda:0 --output pretrained_trace.json

Default model is the public TinyLlama/TinyLlama-1.1B-Chat-v1.0 checkpoint.
Weights download to the normal Hugging Face cache or --cache-dir, never to
this repository. It uses eager attention to expose attention matrices, and
FP32 by default for a more stringent numerical cache-equivalence check.
This is a short correctness/shape probe, not a throughput benchmark.

Sources:
https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/main/config.json
https://huggingface.co/docs/transformers/model_doc/llama
https://huggingface.co/docs/transformers/cache_explanation
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import re
import sys


def finite_nonnegative(value):
    """Reject NaN/inf tolerances before a model download or CUDA initialization."""
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("tolerances must be finite and nonnegative")
    return result


def cache_tensors(cache):
    """Read current and older Transformers cache representations without mutation."""
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(layer[0], layer[1]) for layer in cache]


def run(args):
    # Set deterministic cuBLAS workspace before CUDA is initialized.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not re.fullmatch(r"cuda:\d+", args.device):
        raise ValueError("--device must explicitly name a GPU, for example cuda:2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")
    index = int(args.device.split(":")[1])
    if index >= torch.cuda.device_count():
        raise ValueError("requested device is outside the process's visible CUDA devices")
    if not 8 <= args.max_tokens <= 64:
        raise ValueError("--max-tokens must be between 8 and 64 for this small attention probe")
    if args.cache_dir:
        cache_path = args.cache_dir.expanduser().resolve()
        repo_path = Path(__file__).resolve().parents[2]
        if cache_path == repo_path or repo_path in cache_path.parents:
            raise ValueError("put --cache-dir outside the website repository")

    torch.cuda.set_device(index)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this GPU does not support bfloat16; choose float32 or float16")
    default_tolerances = {
        "float32": (1e-3, 1e-4),
        "float16": (2e-2, 2e-3),
        "bfloat16": (1e-1, 1e-2),
    }
    atol, rtol = default_tolerances[args.dtype]
    atol = atol if args.atol is None else args.atol
    rtol = rtol if args.rtol is None else args.rtol
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("tolerances must be finite and nonnegative")
    common = {"revision": args.revision, "trust_remote_code": False}
    if args.cache_dir:
        common["cache_dir"] = str(args.cache_dir.expanduser())
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", **{dtype_kw: dtype}, **common
    ).to(device).eval()
    model.requires_grad_(False)
    if model.config.model_type != "llama":
        raise ValueError("this architecture-specific probe expects model_type='llama'")
    inputs = tokenizer(args.text, return_tensors="pt", add_special_tokens=True)
    ids = inputs["input_ids"][:, :args.max_tokens].to(device)
    length = ids.shape[1]
    if length < 8:
        raise ValueError("use a prompt that tokenizes to at least eight tokens")

    # Capture one real full pass. The first block illustrates the repeated pattern.
    captured = {}
    handles = []
    names = [
        "model.embed_tokens", "model.layers.0.input_layernorm",
        "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj", "model.layers.0.self_attn.o_proj",
        "model.layers.0.post_attention_layernorm", "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj", "model.norm", "lm_head",
    ]
    for name in names:
        module = model.get_submodule(name)
        def record(_module, hook_inputs, output, label=name):
            captured[label] = {
                "input_shape": list(hook_inputs[0].shape),
                "output_shape": list(output.shape), "dtype": str(output.dtype),
            }
        handles.append(module.register_forward_hook(record))

    with torch.inference_mode():
        try:
            full = model(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=False, output_attentions=True, output_hidden_states=True,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        if full.attentions is None or any(a is None for a in full.attentions):
            raise RuntimeError("attention weights were unavailable despite requesting eager attention")
        reference_logits = full.logits.float().cpu()
        attention_shapes = [list(a.shape) for a in full.attentions]
        attention_example = full.attentions[0][0, 0].float().cpu().tolist()
        masked_mass = max(float(a.triu(diagonal=1).abs().max().item()) for a in full.attentions)
        hidden_shapes = [list(h.shape) for h in full.hidden_states]
        del full

        def cached_pass(chunks):
            past, pieces, steps = None, [], []
            for start, end in chunks:
                result = model(
                    input_ids=ids[:, start:end],
                    attention_mask=torch.ones((1, end), device=device, dtype=torch.long),
                    position_ids=torch.arange(start, end, device=device)[None, :],
                    past_key_values=past, use_cache=True, return_dict=True,
                )
                past = result.past_key_values
                pieces.append(result.logits.float().cpu())
                pairs = cache_tensors(past)
                steps.append({
                    "new_token_range": [start, end],
                    "new_positions": list(range(start, end)),
                    "input_shape": [1, end - start],
                    "logits_shape": list(result.logits.shape),
                    "first_layer_k_shape": list(pairs[0][0].shape),
                    "kv_payload_bytes": sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in pairs),
                })
            return torch.cat(pieces, dim=1), past, steps

        single_logits, single_cache, single_steps = cached_pass([(i, i + 1) for i in range(length)])
        del single_cache
        boundaries = sorted(set([0, 3, min(7, length - 1), length]))
        chunks = list(zip(boundaries[:-1], boundaries[1:]))
        chunk_logits, chunk_cache, chunk_steps = cached_pass(chunks)

        # A real no-future-information check on an independently changed suffix.
        changed = ids.clone()
        split = length // 2
        changed[:, split:] = (changed[:, split:] + 17) % model.config.vocab_size
        changed_logits = model(input_ids=changed, use_cache=False, return_dict=True).logits.float().cpu()
        early_error = float((changed_logits[:, :split] - reference_logits[:, :split]).abs().max())

    def comparison(actual):
        delta = (actual - reference_logits).abs()
        return {
            "passed": bool(torch.allclose(actual, reference_logits, atol=atol, rtol=rtol)),
            "max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean()),
            "argmax_agreement_fraction": float((actual.argmax(-1) == reference_logits.argmax(-1)).float().mean()),
        }
    single_check, chunk_check = comparison(single_logits), comparison(chunk_logits)
    pairs = cache_tensors(chunk_cache)
    actual_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in pairs)
    config = model.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    expected_bytes = 2 * config.num_hidden_layers * length * config.num_key_value_heads * head_dim * pairs[0][0].element_size()
    expected_shape = [1, config.num_key_value_heads, length, head_dim]
    cache_shapes_ok = len(pairs) == config.num_hidden_layers and all(list(k.shape) == expected_shape and list(v.shape) == expected_shape for k, v in pairs)
    last = reference_logits[0, -1]
    top = torch.topk(last.softmax(-1), 5)
    passed = single_check["passed"] and chunk_check["passed"] and early_error <= atol and masked_mass == 0 and cache_shapes_ok and actual_bytes == expected_bytes
    return {
        "schema_version": 1, "status": "passed" if passed else "failed",
        "executed": True, "kind": "pretrained_gpu_inference_probe",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model, "requested_revision": args.revision,
        "resolved_revision": getattr(config, "_commit_hash", None),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
            "device": args.device, "device_name": torch.cuda.get_device_name(device),
            "dtype": args.dtype, "attention_backend": "eager", "seed": 7,
            "tf32": False, "deterministic_algorithms": True,
        },
        "config": {
            "hidden_size": config.hidden_size, "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads, "num_key_value_heads": config.num_key_value_heads,
            "head_dim": head_dim, "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size, "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": getattr(config, "rope_theta", None),
            "rope_scaling": getattr(config, "rope_scaling", None),
            "rope_parameters": getattr(config, "rope_parameters", None),
            "tie_word_embeddings": config.tie_word_embeddings,
        },
        "input": {"text": args.text, "token_ids": ids[0].tolist(), "tokens": tokenizer.convert_ids_to_tokens(ids[0].tolist()), "sequence_length": length},
        "shape_trace": captured, "hidden_state_shapes": hidden_shapes,
        "attention_shapes_by_layer": attention_shapes,
        "first_layer_first_head_attention": attention_example,
        "cache": {
            "layers": len(pairs), "per_layer_k_shape": expected_shape,
            "kv_payload_bytes": actual_bytes, "formula_bytes": expected_bytes,
            "compact_kv_heads_confirmed": cache_shapes_ok,
        },
        "verification": {
            "atol": atol, "rtol": rtol, "single_vs_full": single_check,
            "chunked_vs_full": chunk_check, "future_change_earlier_logits_max_abs_error": early_error,
            "future_masked_attention_max": masked_mass,
        },
        "single_token_decode_steps": single_steps, "chunked_decode_steps": chunk_steps,
        "last_position_top5": [
            {"token_id": int(i), "token": tokenizer.convert_ids_to_tokens(int(i)), "probability": float(p)}
            for i, p in zip(top.indices, top.values)
        ],
        "notes": [
            "Fixed input text is processed for parity checks; this is not a generated-answer benchmark.",
            "Logits at position t describe the prediction for the next token.",
            "Logical K/V payload excludes allocator overhead, weights, and temporary attention matrices.",
            "No timing or throughput claims are made; eager attention is chosen for inspection.",
            "Floating-point cached/full differences are evaluated at the recorded dtype and tolerances.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", required=True, help="explicit visible GPU index, e.g. cuda:2")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--revision", default="main", help="use an immutable model commit for exact reproduction")
    parser.add_argument("--cache-dir", type=Path, help="optional weights cache OUTSIDE this repository")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--text", default="The bank beside the river was quiet, while the central bank announced a new policy.")
    parser.add_argument("--atol", type=finite_nonnegative)
    parser.add_argument("--rtol", type=finite_nonnegative)
    parser.add_argument("--output", type=Path, default=Path("pretrained_trace.json"))
    args = parser.parse_args()
    try:
        report = run(args)
    except ModuleNotFoundError as error:
        print(f"Missing dependency: {error.name}. Install a CUDA-compatible torch and transformers>=4.45,<6, safetensors, sentencepiece.", file=sys.stderr)
        raise SystemExit(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "verification": report["verification"]}, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
