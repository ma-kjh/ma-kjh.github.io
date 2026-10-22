# Under the Hood

A focused, executable guide to representations, attention, and KV caching. Each article connects the mechanism to equations, tensor shapes, readable code, and checks that can be reproduced.

Intended published URL: https://ma-kjh.github.io/under-the-hood/ (publication pending).

This directory adds a static learning section to the existing academic website. Its generated pages are ordinary HTML. The original Jekyll site, posts, layouts, and whole-site deployment remain in place; this section does not replace them.

## What has been verified

| Component | Status |
| --- | --- |
| NumPy decoder | Executed with NumPy 2.3.5 and float64; numerical checks passed |
| Interactive trace | Exported from the executed NumPy decoder; deterministic random, untrained weights |
| Article snippets | Attention and cache examples executed, including a cached multi-token chunk |
| Pretrained GPU probe | Syntax and command-line help checked; pretrained inference has **not** been run |

The GPU probe has no published experimental results in this version. PyTorch and a CUDA execution target were unavailable for that validation. The included script is a reproducible next experiment; its presence is not evidence of a successful GPU run.

## Files and generated routes

| Path | Purpose |
| --- | --- |
| `_content/foundations.html` | Article body: vectors, embeddings, and contextual representations |
| `_content/attention.html` | Article body: attention and the complete small decoder |
| `_content/kv-cache.html` | Article body: cached inference, masking, and memory |
| `_content/modern-attention.html` | Article body: architectural variations and implementation context |
| `_tools/build_site.py` | Static page builder, shared page shell, and topic/article metadata |
| `assets/` | Styling and interactive walkthrough code |
| `examples/tiny_llama.py` | Independently written NumPy decoder and numerical verification |
| `examples/trace.json` | Measured NumPy walkthrough data and verification results |
| `examples/probe_pretrained.py` | Optional inference probe for a real pretrained checkpoint on CUDA |
| `examples/requirements-numpy.txt` | Dependencies for the small reference model |
| `examples/requirements-gpu.txt` | Dependencies for the optional pretrained probe |

The builder produces nine pages:

- `index.html`
- `topics/{representations,attention,kv-cache,modern-attention}/index.html`
- `articles/{foundations,attention,kv-cache,modern-attention}/index.html`

Those paths are relative to `under-the-hood/`.

## Build and preview the section

Use Python 3.10 or newer. From the `under-the-hood/` directory:

```bash
python _tools/build_site.py
```

The equivalent command from the repository root is:

```bash
python under-the-hood/_tools/build_site.py
```

The builder resolves paths relative to its own location. No Jekyll installation is needed to generate these nine pages. For a local preview, run this from the **repository root**:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open [http://127.0.0.1:8000/under-the-hood/](http://127.0.0.1:8000/under-the-hood/). Serving from the root preserves the section’s absolute asset and download URLs. Use the repository’s existing Jekyll workflow when previewing or building the entire academic website.

### Editing an article

Edit the corresponding `_content/*.html` body fragment, then rebuild. Each fragment uses section IDs and `h2` headings, HTML tables, and `pre/code` elements. Escape HTML-sensitive characters inside code, such as `&lt;` and `&amp;`. Math uses `\(...\)` for inline expressions and `\[...\]` for display expressions.

Keep IDs unique within an article and keep internal reference links synchronized with their targets. Topic titles, descriptions, routes, and the shared page structure belong in the builder’s metadata and shell. Changes to generated page files will be overwritten by the next build.

When numerical behavior changes, regenerate `examples/trace.json` before building. Its attention matrices and cache sizes come from the actual reference calculation. The browser presents those exported measurements; it does not load pretrained model weights.

## Run the NumPy reference

The following commands assume the current directory is `under-the-hood/`. Install the small dependency set in your Python environment:

```bash
python -m pip install -r examples/requirements-numpy.txt
python examples/tiny_llama.py --verify --export-trace examples/trace.json
python _tools/build_site.py
```

The combined verification/export command is supported by the script. Exporting a trace also invokes verification. Run Python normally: `python -O` disables the assertions used by this educational checker.

The model uses seed 7, two layers, float64 arrays, and a manually labeled vocabulary. Its principal checks are:

- Cached single-token decoding and cached multi-token chunks agree with full-prefix logits within `1e-10` maximum absolute error.
- Changing a future suffix does not alter earlier logits.
- Compact grouped attention agrees with an explicit repeated-head reference.
- Masked attention entries are zero and each attention row sums to one.
- RoPE depends on position and preserves vector norms.
- Cache tensors have the expected compact KV-head shape and payload size.
- Continuing a prefix does not mutate the supplied prefix cache.

The bundled trace records these measured results:

| Check | Result |
| --- | ---: |
| Single-token cached/full maximum absolute logit error | `2.3314683517128287e-15` |
| Chunked cached/full maximum absolute logit error | `1.942890293094024e-15` |
| Grouped/explicit-repeat attention maximum absolute error | `4.440892098500626e-16` |
| Future-suffix change: earlier-logit maximum absolute error | `0.0` |
| Cache payload, batch 2 and six positions | `6144` bytes |
| Interactive prefill payload, batch 1 and six positions | `3072` bytes |

The checker uses a two-item batch to expose batch/head-axis mistakes; the interactive walkthrough uses one item. Payload counts cover K and V arrays, excluding position metadata, model weights, temporary arrays, and allocation overhead. Small numerical differences between NumPy/BLAS environments can occur; the assertions and their tolerance are the reproducibility criterion.

## Educational model and pretrained target

The GPU runner defaults to [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0). It is a trained Llama-architecture checkpoint. Its dimensions below are taken from the publisher’s [configuration](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/config.json).

| Property | NumPy educational model | Default pretrained target |
| --- | ---: | ---: |
| Weights | Fixed random initialization | Trained checkpoint |
| Decoder layers | 2 | 22 |
| Hidden width | 32 | 2048 |
| Query heads | 4 | 32 |
| KV heads | 2 | 4 |
| Head width | 8 | 64 |
| Feed-forward width | 96 | 5632 |
| Vocabulary size | 32 hand-written labels | 32000 tokenizer entries |
| RoPE base | 10000 | 10000 |
| RMSNorm epsilon | `1e-5` | `1e-5` |
| Tied embedding/output weights | Yes | No |
| Execution precision used by these examples | float64 | float32 by default |

Both follow the RMSNorm, RoPE, GQA, SwiGLU, residual, and final-logit structure explained in the articles. The NumPy version reduces dimensions and simplifies tokenization, storage, and execution; it does not reproduce a pretrained checkpoint’s outputs. Its split-half RoPE convention follows the Hugging Face Llama layout. Importing weights with another channel-pairing convention requires the corresponding conversion.

The pretrained configuration records bfloat16 as its stored dtype, while the probe explicitly loads float32 by default. Its short fixed input and eager attention are chosen for inspecting shapes and cache equivalence. It does not measure throughput, train a model, or evaluate generated-answer quality.

## Optional pretrained GPU experiment

### Install a CUDA-compatible environment

Use Python 3.10+ and choose a CUDA-enabled PyTorch wheel compatible with the selected machine. The [official PyTorch installation selector](https://pytorch.org/get-started/locally/) supplies the appropriate wheel command. Current [Transformers installation guidance](https://huggingface.co/docs/transformers/installation) tests Python 3.10+ and PyTorch 2.6+.

For example, **only when the selector recommends the CUDA 12.8 wheel channel**:

```bash
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r examples/requirements-gpu.txt
```

The requirements preserve the runner’s declared `transformers>=4.45,<6` range. That range is an intended compatibility range, not a claim that every release combination was executed. The script handles the Transformers 4/5 dtype keyword difference and several cache representations. Record the resolved package versions and model revision from each successful report.

### Select a GPU and run

Inspect the device list and current usage first:

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Visible devices:', torch.cuda.device_count())"
```

Choose an available, allocated device, then pass its explicit visible index as `cuda:N`. The example below uses `cuda:2`; replace it with the index you selected. If `CUDA_VISIBLE_DEVICES` remaps devices, the index is relative to that visible list.

From `under-the-hood/`:

```bash
python examples/probe_pretrained.py \
  --device cuda:2 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --revision fe8a4ea1ffedaf415f4da2f062534de366a451e6 \
  --dtype float32 \
  --max-tokens 24 \
  --output results/pretrained_trace.json
```

The script defaults to the same model, revision `main`, float32, eager attention, and a 24-token cap. The example pins the immutable revision used for the configuration table. The default prompt is a short sentence about a river bank and a central bank. It must tokenize to at least eight tokens; `--max-tokens` accepts 8–64.

Weights download into the normal Hugging Face cache. An optional `--cache-dir` must point outside the website repository. The runner uses `trust_remote_code=False` and validates the Llama model type. Keep its output separate from `examples/trace.json`: the pretrained report uses a different schema.

### Read the report

A completed run records model/revision information, package and CUDA versions, GPU name, input token IDs, module and attention shapes, compact cache tensors, payload bytes, and numerical comparisons. The verification checks full-prefix versus single-token and chunked execution, future-token isolation, causal attention zeros, and cache shapes/size.

| Precision | Default absolute tolerance | Default relative tolerance |
| --- | ---: | ---: |
| float32 | `1e-3` | `1e-4` |
| float16 | `2e-2` | `2e-3` |
| bfloat16 | `1e-1` | `1e-2` |

The runner disables TF32 and requests deterministic algorithms. Floating-point equivalence is evaluated at the recorded precision and tolerances; changing tolerances changes the claim being tested.

A completed passing probe exits with code 0. A completed failed verification writes `"status": "failed"` and exits with code 1. Missing dependencies produce code 2; other setup or execution errors can terminate before a report is written. Treat a GPU experiment as completed only when an actual report has been produced and its verification inspected.

## Primary implementation references

- [Hugging Face Llama implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [Transformers cache explanation](https://huggingface.co/docs/transformers/cache_explanation)
- [TinyLlama checkpoint configuration](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/config.json)
- [GQA paper](https://aclanthology.org/2023.emnlp-main.298/)
- [RoFormer / RoPE paper](https://arxiv.org/abs/2104.09864)

Each article includes the primary references for its own architectural claims.
