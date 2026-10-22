#!/usr/bin/env python3
"""Build the standalone Under the Hood notebook using Python's standard library."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "/under-the-hood"
REPO = "https://github.com/ma-kjh/ma-kjh.github.io"
ORIGIN = "https://ma-kjh.github.io"
DATE = "22 October 2026"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


ARTICLES = [
    dict(slug="foundations", topic="representations", title="Before attention: how words become vectors",
         short="From word2vec to attention",
         deck="Build an embedding table, calculate a complete skip-gram training step, and discover why a decoder needs more than one fixed vector.",
         label="01 / Foundations", level="Start here", minutes="14",
         tags="word2vec embeddings CBOW skip-gram RNN Bahdanau history vectors",
         concepts=["Embedding lookup", "CBOW and skip-gram", "Softmax and negative sampling", "Static vs. contextual states", "The encoder–decoder bottleneck"]),
    dict(slug="attention", topic="attention", title="Inside attention, one tensor at a time",
         short="A tiny Llama-style decoder",
         deck="Follow a complete decoder through Q, K, V, rotary positions, grouped-query attention, residual connections, and next-token logits. Every shape has a job.",
         label="02 / Mechanism", level="Deep dive", minutes="22",
         tags="attention query key value QKV softmax mask Llama RoPE RMSNorm SwiGLU NumPy",
         concepts=["Queries, keys, and values", "Causal softmax", "RoPE and absolute positions", "Grouped-query attention", "RMSNorm, residuals, and SwiGLU"]),
    dict(slug="kv-cache", topic="kv-cache", title="KV cache: what can a model safely reuse?",
         short="Prefill, decode, and the KV cache",
         deck="Derive why old keys and values remain valid, inspect what each layer stores, and prove that cached decoding agrees with a full forward pass.",
         label="03 / Inference", level="Deep dive", minutes="14",
         tags="KV cache inference prefill decode memory causal chunk positions caching",
         concepts=["Prefix invariance", "Prefill and decoding", "A cache for every layer", "Chunked causal masks", "Memory accounting and equivalence"]),
    dict(slug="modern-attention", topic="modern-attention", title="Why modern attention has so many names",
         short="GQA, MLA, and modern attention",
         deck="Separate head sharing, position, compressed memory, sparse selection, and GPU execution. Then read DeepSeek’s recent designs with those distinctions in mind.",
         label="04 / Research", level="Research map", minutes="13",
         tags="modern GQA MQA MHA MLA DeepSeek FlashAttention NSA DSA CSA HCA RoPE",
         concepts=["MHA → MQA → GQA", "Learned latent memory in MLA", "Exact FlashAttention", "NSA and DSA selection", "CSA, HCA, and cross-layer reuse"]),
]
BY_SLUG = {a["slug"]: a for a in ARTICLES}

TOPICS = [
    dict(slug="representations", title="Word representations", number="01", question="How do words become useful vectors?",
         description="From a row in an embedding table to a representation that depends on context.",
         lead="Start with the objects attention operates on. Learn how word vectors are trained, what a static lookup can express, and why sequence models need a way to retrieve relevant context.",
         main="foundations", related=["attention"], icon="vectors",
         papers=[("Bengio et al. · 2003", "A Neural Probabilistic Language Model", "https://jmlr.org/papers/v3/bengio03a.html"),
                 ("Mikolov et al. · 2013", "Efficient Estimation of Word Representations", "https://arxiv.org/abs/1301.3781"),
                 ("Bahdanau et al. · 2014/15", "Jointly Learning to Align and Translate", "https://arxiv.org/abs/1409.0473")]),
    dict(slug="attention", title="Attention mechanisms", number="02", question="How does a token decide what to read?",
         description="Queries, keys, values, and the exact path through a small Llama-style decoder.",
         lead="Turn the attention equation into an actual computation. Track every dimension, inspect a causal attention matrix, and read the same implementation that produced the interactive trace.",
         main="attention", related=["foundations", "modern-attention"], icon="attention",
         papers=[("Vaswani et al. · 2017", "Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
                 ("Su et al. · 2021", "RoFormer: Rotary Position Embedding", "https://arxiv.org/abs/2104.09864"),
                 ("Meta · reference implementation", "Llama decoder source", "https://github.com/meta-llama/llama3/blob/main/llama/model.py")]),
    dict(slug="kv-cache", title="KV cache", number="03", question="What can a model safely reuse?",
         description="Follow prefill and decoding. See which tensors stay, which grow, and why.",
         lead="Caching follows from causality. Once you understand why an old state cannot see a newly appended token, you can derive the cache layout, position handling, and memory cost yourself.",
         main="kv-cache", related=["attention", "modern-attention"], icon="cache",
         papers=[("Hugging Face · documentation", "Understanding the KV cache", "https://huggingface.co/docs/transformers/cache_explanation"),
                 ("Ainslie et al. · 2023", "Grouped-query attention", "https://arxiv.org/abs/2305.13245"),
                 ("Kwon et al. · 2023", "PagedAttention and vLLM", "https://arxiv.org/abs/2309.06180")]),
    dict(slug="modern-attention", title="Modern architectures", number="04", question="What changes when context gets expensive?",
         description="Understand GQA, latent memory, sparse attention, and efficient GPU kernels.",
         lead="A new attention name can describe a different storage layout, a different set of visible entries, or a faster way to execute the same mathematics. Compare the design choice before comparing the headline.",
         main="modern-attention", related=["attention", "kv-cache"], icon="modern",
         papers=[("Ainslie et al. · 2023", "GQA", "https://arxiv.org/abs/2305.13245"),
                 ("DeepSeek-AI · 2024", "DeepSeek-V2: multi-head latent attention", "https://arxiv.org/abs/2405.04434"),
                 ("Dao et al. · 2022", "FlashAttention", "https://arxiv.org/abs/2205.14135"),
                 ("DeepSeek-AI · 2026", "DeepSeek-V4.1-Flash", "https://arxiv.org/abs/2609.19969")]),
]
BY_TOPIC = {t["slug"]: t for t in TOPICS}

def header() -> str:
    return f'''<a class="skip" href="#main">Skip to content</a>
<header class="topbar"><div class="shell">
  <a class="brand" href="/">Jeonghyeon Kim</a>
  <button class="menu-button" data-menu aria-expanded="false" aria-controls="site-nav">Menu</button>
  <nav class="topnav" id="site-nav" aria-label="Main navigation">
    <a class="active" href="{BASE}/">Under the Hood</a>
    <a href="/publications">Publications</a>
    <a href="{REPO}/tree/main/under-the-hood">GitHub</a>
  </nav>
</div></header>'''


def footer() -> str:
    return f'''<footer class="footer"><div class="shell">
  <p>© 2026 Jeonghyeon Kim</p>
  <div class="footer-links"><a href="/">About</a><a href="{BASE}/examples/tiny_llama.py" download>Code</a><a href="#top">Back to top</a></div>
</div></footer>'''


def document(title: str, description: str, path: str, body: str, *, article: bool = False) -> str:
    math = '''<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>''' if article else ""
    structured = {
        "@context": "https://schema.org",
        "@type": "TechArticle" if article else "CollectionPage",
        "headline": title, "description": description,
        "url": ORIGIN + path,
        "author": {"@type": "Person", "name": "Jeonghyeon Kim", "url": ORIGIN},
        "inLanguage": "en", "dateModified": "2026-10-22"
    }
    return f'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} · Jeonghyeon Kim</title>
<meta name="description" content="{esc(description)}">
<meta name="theme-color" content="#ffffff">
<link rel="canonical" href="{ORIGIN}{path}">
<meta property="og:type" content="{'article' if article else 'website'}">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}">
<meta property="og:url" content="{ORIGIN}{path}">
<link rel="icon" href="{BASE}/assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="{BASE}/assets/research.css">
{math}
<script defer src="{BASE}/assets/research.js"></script>
<script type="application/ld+json">{json.dumps(structured, ensure_ascii=False)}</script>
</head><body id="top">
{header()}
{body}
{footer()}
</body></html>
'''


def article_row(slug: str, number: str | None = None) -> str:
    a = BY_SLUG[slug]
    return f'''<a class="article-row" href="{BASE}/articles/{slug}/">
      <span class="row-number">{number or a["label"][:2]}</span>
      <div><h3>{esc(a["title"])}</h3><p>{esc(a["deck"])}</p>
        <div class="row-meta"><span>{a["minutes"]} min read</span><span>Equations and code</span></div>
      </div><span class="row-end" aria-hidden="true">↗</span>
    </a>'''


def reading_path() -> str:
    blurbs = ["What a vector represents, and how it learns.",
              "From a token ID to the next-token logits.",
              "Reuse the prefix. Verify every new output.",
              "Compare what each architecture changes."]
    return '<div class="reading-path">' + "".join(
        f'''<a class="path-item" href="{BASE}/articles/{a["slug"]}/">
        <span class="path-no">{i + 1:02d}</span><span class="path-arrow" aria-hidden="true">→</span>
        <h3>{esc(a["short"])}</h3><p>{blurbs[i]}</p></a>'''
        for i, a in enumerate(ARTICLES)) + "</div>"


def hub() -> str:
    rows = "".join(f'''<section class="index-row" data-search="{esc(t["title"] + " " + t["description"] + " " + BY_SLUG[t["main"]]["tags"])}">
      <span class="index-number">{t["number"]}</span>
      <div class="index-content"><h3><a href="{BASE}/topics/{t["slug"]}/">{t["title"]}</a></h3>
        <p>{t["description"]}</p>
        <div class="index-links"><a href="{BASE}/topics/{t["slug"]}/">Topic guide</a><a href="{BASE}/articles/{t["main"]}/">{esc(BY_SLUG[t["main"]]["short"])} →</a></div>
      </div>
    </section>''' for t in TOPICS)
    body = f'''<main id="main" class="shell">
<section class="notebook-intro" aria-labelledby="notebook-title">
  <h1 id="notebook-title">Under the Hood</h1>
  <p class="notebook-description">Notes on how language models work.</p>
  <p>From word vectors to attention and efficient inference, with the equations, tensor shapes, and code worked out along the way. If you are starting from scratch, read the topics in order.</p>
</section>
<section class="section" id="topics" aria-labelledby="topics-title">
  <div class="section-head"><h2 id="topics-title">Contents</h2>
    <div class="search-wrap"><label class="sr-only" for="question-search">Search topics and concepts</label><span class="search-icon" aria-hidden="true">⌕</span><input type="search" id="question-search" placeholder="Find a topic, e.g. GQA" autocomplete="off"></div>
  </div>
  <div class="notebook-index">{rows}</div>
  <div id="no-results" class="no-results" hidden>No matching topic. Try “attention”, “vectors”, “GQA”, or “cache”.</div>
  <div id="search-status" class="search-results" aria-live="polite"></div>
</section>
<section class="section notebook-columns" aria-label="History and experiments">
  <div><h2>How we got here</h2>
    <ol class="chronology">
      <li><span class="year">2003–2013</span><p>Learned word representations and word2vec: words become reusable vectors. <a href="{BASE}/articles/foundations/">Start with embeddings</a>.</p></li>
      <li><span class="year">2014–2015</span><p>Encoder–decoder attention learns to retrieve source states while generating a translation. <a href="{BASE}/articles/foundations/#sequence-bottleneck">The fixed-vector bottleneck</a>.</p></li>
      <li><span class="year">2017</span><p>The Transformer makes attention a central operation in sequence modeling. <a href="{BASE}/articles/attention/">Work through Q, K, and V</a>.</p></li>
      <li><span class="year">2019–2024</span><p>MQA, GQA, FlashAttention, and MLA address different costs of long-context inference. <a href="{BASE}/articles/modern-attention/">Compare the mechanisms</a>.</p></li>
      <li><span class="year">2025–2026</span><p>Sparse selection, compressed context, and cross-layer reuse extend the design space. <a href="{BASE}/articles/modern-attention/#recent-research">Read the recent work</a>.</p></li>
    </ol>
  </div>
  <div><h2>Code and experiments</h2>
    <ul class="resource-list">
      <li><a href="{BASE}/examples/tiny_llama.py" download>NumPy decoder</a><p>Two layers, four query heads, two KV heads. A complete forward pass you can read and run.</p></li>
      <li><a href="{BASE}/articles/attention/#implementation-source">Line-by-line walkthrough</a><p>Follow the implementation alongside the shapes it computes.</p></li>
      <li><a href="{BASE}/examples/trace.json" download>Measured attention and cache trace</a><p>The values behind the interactive figures, including numerical checks.</p></li>
      <li><a href="{BASE}/articles/attention/#pretrained-lab">Pretrained TinyLlama experiment</a><p>A prepared GPU probe for the 1.1B checkpoint. The GPU run is still pending.</p></li>
    </ul>
    <div class="notebook-note"><p>The NumPy example uses deterministic random weights to make the mechanics easy to inspect. The pretrained experiment is separate.</p><p>Last updated {DATE}. Primary references are linked in each article.</p></div>
  </div>
</section>
</main>'''
    return document("Under the Hood", "Notes on how language models work: word vectors, attention, KV caching, and modern architectures, with equations, tensor shapes, and executable code.", BASE + "/", body)


def topic_page(t: dict) -> str:
    a = BY_SLUG[t["main"]]
    papers = "".join(f'<li><span class="paper-meta">{esc(who)}</span><a href="{url}">{esc(title)}</a></li>' for who, title, url in t["papers"])
    concepts = "".join(f"<li>{esc(c)}</li>" for c in a["concepts"])
    body = f'''<main id="main" class="shell">
  <section class="topic-hero"><div class="breadcrumbs"><a href="{BASE}/">Under the Hood</a><span>/</span><span>{t["title"]}</span></div>
    <h1>{t["title"]}</h1><p class="question">{t["question"]}</p><p class="lead">{t["lead"]}</p>
    <p><a href="{BASE}/articles/{t["main"]}/">Read the full explanation →</a></p>
  </section>
  <div class="topic-detail-grid"><div>
    <section class="section"><div class="section-head"><h2>The article</h2></div>
      <div class="article-list">{article_row(t["main"])}</div>
    </section>
    <section class="section"><div class="section-head"><h2>Related reading</h2></div>
      <div class="article-list">{"".join(article_row(s) for s in t["related"])}</div>
    </section>
  </div><aside class="topic-aside" aria-label="Topic resources"><h3>What this covers</h3><ul>{concepts}</ul><h3>Original papers and sources</h3><ul>{papers}</ul>
    <div class="notebook-note"><p>The core articles share one small NumPy decoder. Its random weights make the computation inspectable.</p><a href="{BASE}/examples/tiny_llama.py" download>Download the code</a></div>
  </aside></div>
</main>'''
    return document(t["title"] + " · Under the Hood", t["description"], f'{BASE}/topics/{t["slug"]}/', body)


SOURCE_APPENDIX = r'''
<section id="implementation-source">
  <h2>Read the implementation, line by line</h2>
  <p>The source reader below loads the actual <a href="/under-the-hood/examples/tiny_llama.py" download>tiny_llama.py</a> used to generate these diagrams. Select an operation to inspect its numbered source lines and the role of each step. The complete implementation includes both decoder layers and a cache for each layer.</p>
  <div data-widget="source-lab"></div>
  <p>Reproduce the reference calculation locally:</p>
  <pre><code class="language-bash">python -m pip install numpy
python tiny_llama.py --verify --export-trace trace.json</code></pre>
  <p>The verification compares cached single-token and chunked outputs with full-sequence outputs, compares grouped attention with explicitly repeated K/V, checks future masking, and counts the cache payload. The tests use more than one batch item so that accidental broadcasting is less likely to hide an error.</p>
  <div class="verifications">
    <div class="verification"><span>Cached vs. full logits</span><strong>2.33 × 10⁻¹⁵</strong><small>Maximum absolute error · single-token test</small></div>
    <div class="verification"><span>Future-token leakage</span><strong>0.0</strong><small>Earlier logits after changing a future suffix</small></div>
    <div class="verification"><span>Six-token KV payload</span><strong>3,072 bytes</strong><small>Batch 1 · 2 layers · float64 · excludes metadata</small></div>
  </div>
  <p>These are measured CPU results for the deterministic NumPy model, not accuracy or speed benchmarks. See <a href="/under-the-hood/examples/trace.json" download>trace.json</a> for the full verification record and the data behind the widgets.</p>
</section>
<section id="pretrained-lab">
  <h2>Take the same questions to a pretrained model</h2>
  <p>For a realistic small model, the accompanying <a href="/under-the-hood/examples/probe_pretrained.py" download>GPU probe</a> uses the public <a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama-1.1B-Chat-v1.0</a> checkpoint. Its <a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/main/config.json">published configuration</a> has 22 layers, hidden width 2,048, 32 query heads, four KV heads, head dimension 64, and a 5,632-channel feed-forward expansion.</p>
  <p>This is a separate pretrained checkpoint. The educational class named <code>TinyLlama</code> above has independent random weights, a 32-label vocabulary, and much smaller dimensions. The downloaded model has a real tokenizer and trained parameters.</p>
  <div class="callout"><div class="callout-label">Experiment status</div><p>The pretrained runner is prepared and syntax-checked. A GPU run has not yet been completed, so this notebook does not report pretrained trace values or performance measurements.</p></div>
  <p>Install a CUDA-compatible PyTorch build for your machine, then download <a href="/under-the-hood/examples/requirements-gpu.txt" download>requirements-gpu.txt</a> and <a href="/under-the-hood/examples/requirements-numpy.txt" download>requirements-numpy.txt</a> into the same directory as the probe. The GPU requirements file includes the NumPy requirements file. Choose an available GPU explicitly before running:</p>
  <pre><code class="language-bash">nvidia-smi
python -m pip install -r requirements-gpu.txt
# Replace cuda:0 with the GPU you intend to use.
python probe_pretrained.py \
  --device cuda:0 \
  --dtype float32 \
  --output pretrained_trace.json</code></pre>
  <p>The probe runs inference with eager attention so that attention probabilities can be inspected. It records the actual projection and block shapes, cache tensor sizes, software versions, and GPU metadata. It then compares cached and full-forward logits. Eager attention exposes the computation for study; this experiment is not a throughput benchmark.</p>
  <p>For this checkpoint, ordinary K/V payload per token at batch one is \(2\times22\times4\times64=11{,}264\) scalars. That is 45,056 bytes in float32, or 22,528 bytes in a two-byte format. These are configuration-based calculations, separate from measured total GPU memory.</p>
</section>
'''


def article_page(a: dict, index: int) -> str:
    content = (ROOT / "_content" / f'{a["slug"]}.html').read_text()
    if a["slug"] == "attention":
        reference_start = content.index('<section id="attention-references">')
        content = content[:reference_start] + SOURCE_APPENDIX + "\n" + content[reference_start:]

    headings: list[tuple[str, str]] = []

    def heading_match(match: re.Match) -> str:
        section_start, sid, title = match.group(1), match.group(2), match.group(3)
        hid = "heading-" + sid
        headings.append((hid, re.sub("<[^>]+>", "", title)))
        return f'{section_start}<h2 id="{hid}">{title}</h2>'

    content = re.sub(r'(<section id="([^"]+)">\s*)<h2>(.*?)</h2>',
                     heading_match, content, flags=re.S)
    toc = '<ol class="toc-list">' + "".join(f'<li><a href="#{hid}">{esc(title)}</a></li>' for hid, title in headings) + "</ol>"
    previous = ARTICLES[index - 1] if index > 0 else None
    following = ARTICLES[index + 1] if index + 1 < len(ARTICLES) else None
    prev_link = f'<a href="{BASE}/articles/{previous["slug"]}/"><span class="direction">← Previous article</span><span class="next-title">{previous["short"]}</span></a>' if previous else f'<a href="{BASE}/"><span class="direction">← The notebook</span><span class="next-title">Browse all topics</span></a>'
    next_link = f'<a href="{BASE}/articles/{following["slug"]}/"><span class="direction">Next article →</span><span class="next-title">{following["short"]}</span></a>' if following else f'<a href="{BASE}/articles/attention/#pretrained-lab"><span class="direction">Next experiment →</span><span class="next-title">Run the pretrained model probe</span></a>'
    t = BY_TOPIC[a["topic"]]
    body = f'''<div class="progress" id="reading-progress" aria-hidden="true"></div>
<main id="main" class="article-shell">
  <header class="article-top"><div class="breadcrumbs"><a href="{BASE}/">Under the Hood</a><span>/</span><a href="{BASE}/topics/{t["slug"]}/">{t["title"]}</a></div>
    <div class="eyebrow">{a["label"]}</div><h1>{esc(a["title"])}</h1><p class="article-deck">{esc(a["deck"])}</p>
    <div class="article-meta"><span class="author">Jeonghyeon Kim</span><time datetime="2026-10-22">{DATE}</time><span>{a["minutes"]} min read</span></div>
  </header>
  <div class="article-grid"><aside class="toc" aria-label="On this page"><p class="toc-title">On this page</p>{toc}
    <div class="toc-tools"><a href="{BASE}/examples/tiny_llama.py" download>↓ Download NumPy source</a><a href="{BASE}/examples/trace.json" download>↓ Download measured trace</a><a href="{REPO}/tree/main/under-the-hood">↗ View on GitHub</a></div></aside>
    <article class="article-body"><details class="mobile-toc"><summary>On this page</summary>{toc}</details>
    {content}<nav class="article-next" aria-label="Article sequence">{prev_link}{next_link}</nav>
    </article>
  </div>
</main>'''
    return document(a["title"], a["deck"], f'{BASE}/articles/{a["slug"]}/', body, article=True)


def write(relative: str, value: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    print(path.relative_to(ROOT.parent))


def main() -> None:
    write("index.html", hub())
    for topic in TOPICS:
        write(f'topics/{topic["slug"]}/index.html', topic_page(topic))
    for i, article in enumerate(ARTICLES):
        write(f'articles/{article["slug"]}/index.html', article_page(article, i))


if __name__ == "__main__":
    main()
