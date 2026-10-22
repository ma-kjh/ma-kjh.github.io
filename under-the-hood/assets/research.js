/* Under the Hood: progressive enhancement and measured educational traces. */
(() => {
  "use strict";

  const TRACE_URL = "/under-the-hood/examples/trace.json";
  const SOURCE_URL = "/under-the-hood/examples/tiny_llama.py";
  const PASS_LABELS = {
    prefill: "Prefill · 6 new tokens",
    single_token_decode: "Decode · 1 new token",
    chunked_decode: "Chunked decode · 2 new tokens"
  };
  let tracePromise = null;
  let sourcePromise = null;

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const shape = value => `[${value.join(", ")}]`;
  const number = value => Number(value).toLocaleString("en-US");
  const bytes = value => {
    if (value < 1024) return `${number(value)} B`;
    if (value < 1024 ** 2) return `${(value / 1024).toFixed(2)} KiB`;
    return `${(value / 1024 ** 2).toFixed(2)} MiB`;
  };

  async function getTrace() {
    if (!tracePromise) {
      tracePromise = fetch(TRACE_URL, { credentials: "same-origin" })
        .then(response => {
          if (!response.ok) throw new Error(`Trace request returned ${response.status}`);
          return response.json();
        })
        .then(trace => {
          if (trace.schema_version !== 1 || !trace.config || !Array.isArray(trace.passes) || !trace.passes.length) {
            throw new Error("The trace has an unsupported format");
          }
          return trace;
        })
        .catch(error => { tracePromise = null; throw error; });
    }
    return tracePromise;
  }

  async function getSource() {
    if (!sourcePromise) {
      sourcePromise = fetch(SOURCE_URL, { credentials: "same-origin" })
        .then(response => {
          if (!response.ok) throw new Error(`Source request returned ${response.status}`);
          return response.text();
        })
        .then(source => {
          if (!source.includes("class TinyLlama:") || !source.includes("def grouped_attention(")) {
            throw new Error("The requested Python source could not be read");
          }
          return source;
        })
        .catch(error => { sourcePromise = null; throw error; });
    }
    return sourcePromise;
  }

  function labHeading(root, title, badge, eyebrow = "Interactive notebook") {
    const heading = el("div", "lab-heading");
    const titles = el("div");
    titles.append(el("div", "eyebrow", eyebrow), el("h3", "", title));
    heading.append(titles, el("span", "pill outline", badge));
    root.append(heading);
  }

  function makeSelect(parent, label, options, value) {
    const wrapper = el("label", "", label);
    const select = el("select");
    options.forEach(([optionValue, text]) => {
      const option = el("option", "", text);
      option.value = String(optionValue);
      select.append(option);
    });
    if (value !== undefined) select.value = String(value);
    wrapper.append(select);
    parent.append(wrapper);
    return select;
  }

  function setOptions(select, options, value) {
    select.replaceChildren();
    options.forEach(([optionValue, text]) => {
      const option = el("option", "", text);
      option.value = String(optionValue);
      select.append(option);
    });
    select.value = String(value);
  }

  function fact(label, value) {
    const box = el("div", "lab-fact");
    box.append(el("span", "", label), el("strong", "", value));
    return box;
  }

  function contextTokens(trace, pass) {
    if (pass.name === "prefill") return pass.tokens[0];
    const prefill = trace.passes.find(item => item.name === "prefill");
    return [...prefill.tokens[0], ...pass.tokens[0]];
  }

  function drawTokens(container, tokens, newStart) {
    container.replaceChildren();
    tokens.forEach((token, position) => {
      const cached = position < newStart;
      const item = el("span", cached ? "cached" : "new-token", `${position} · ${token}`);
      item.title = cached ? "K/V reused from the prompt" : "Processed in this forward pass";
      container.append(item);
    });
  }

  function attentionLab(root, trace) {
    labHeading(root, "Inspect one attention head", "Measured NumPy trace", "Attention lab");
    const controls = el("div", "lab-controls");
    const phase = makeSelect(controls, "Forward pass", trace.passes.map((p, i) => [i, PASS_LABELS[p.name] || p.name]), 0);
    const layer = makeSelect(controls, "Layer", trace.passes[0].layers.map((_, i) => [i, `Layer ${i}`]), 0);
    const head = makeSelect(controls, "Query head", Array.from({ length: trace.config.n_heads }, (_, i) => [i, `Head ${i}`]), 0);
    const query = makeSelect(controls, "Query position", [], 0);
    const tokenFlow = el("div", "token-flow");
    const legend = el("p", "lab-note", "Dark token labels are processed now; pale labels reuse prompt K/V. In the grid, rows are queries and columns are keys. Stripes mark future positions.");
    const heatmap = el("div", "heatmap");
    const grid = el("div", "heatmap-grid");
    heatmap.append(grid);
    const readout = el("div", "lab-readout");
    readout.setAttribute("aria-live", "polite");
    const facts = el("div", "lab-facts");
    const shapes = el("p", "lab-note");
    root.append(controls, tokenFlow, legend, heatmap, readout, facts, shapes);
    root.append(el("p", "lab-note", "Both decode options continue from the same six-token prompt cache; they are separate branches, not consecutive steps. Values come from deterministic random, untrained weights, and show the calculation rather than learned word meanings. Each row is rounded to eight decimal places in the saved trace."));
    let selectedKey = null;

    function fillQueries() {
      const pass = trace.passes[Number(phase.value)];
      const item = pass.layers[Number(layer.value)];
      const tokens = contextTokens(trace, pass);
      setOptions(query, item.positions.map((p, i) => [i, `${p} · ${tokens[p]}`]), item.positions.length - 1);
      selectedKey = null;
    }

    function render() {
      const pass = trace.passes[Number(phase.value)];
      const layerIndex = Number(layer.value);
      const headIndex = Number(head.value);
      const item = pass.layers[layerIndex];
      const rowIndex = Number(query.value);
      const values = item.attention_weights[headIndex];
      const row = values[rowIndex];
      const tokens = contextTokens(trace, pass);
      const queryPosition = item.positions[rowIndex];
      const keys = item.key_positions;
      const kvHead = Math.floor(headIndex / (trace.config.n_heads / trace.config.n_kv_heads));
      const visibleCount = item.causal_mask[rowIndex].filter(Boolean).length;
      drawTokens(tokenFlow, tokens, item.positions[0]);
      grid.style.gridTemplateColumns = `80px repeat(${keys.length}, minmax(42px, 1fr))`;
      grid.style.minWidth = `${80 + keys.length * 46}px`;
      grid.setAttribute("aria-label", `Attention weights for layer ${layerIndex}, query head ${headIndex}`);
      grid.replaceChildren(el("span", "axis", "query / key"));
      keys.forEach(position => grid.append(el("span", "axis", `${position} · ${tokens[position]}`)));

      values.forEach((weights, r) => {
        const absoluteQuery = item.positions[r];
        grid.append(el("span", "axis", `${absoluteQuery} · ${tokens[absoluteQuery]}`));
        weights.forEach((weight, k) => {
          const masked = !item.causal_mask[r][k];
          const cell = el("button", `${masked ? "masked " : ""}${r === rowIndex ? "selected-row" : ""}`, masked ? "—" : `${(weight * 100).toFixed(1)}%`);
          cell.type = "button";
          cell.setAttribute("aria-pressed", String(r === rowIndex && k === selectedKey));
          cell.setAttribute("aria-label", `Query ${absoluteQuery} ${tokens[absoluteQuery]}, key ${keys[k]} ${tokens[keys[k]]}: ${masked ? "masked future position; weight zero" : `attention weight ${weight.toFixed(8)}`}`);
          cell.title = masked ? "Causal mask: this key is in the future" : `Attention weight: ${weight.toFixed(8)}`;
          if (!masked) {
            cell.style.backgroundColor = `rgba(36, 92, 67, ${0.10 + 0.80 * weight})`;
            cell.style.color = weight > 0.55 ? "#ffffff" : "#163b27";
          }
          cell.addEventListener("click", () => {
            query.value = String(r);
            selectedKey = k;
            render();
            const replacement = grid.querySelectorAll("button")[r * keys.length + k];
            if (replacement) replacement.focus({ preventScroll: true });
          });
          grid.append(cell);
        });
      });
      const total = row.reduce((sum, weight) => sum + weight, 0);
      let detail = `Query ${queryPosition} “${tokens[queryPosition]}” can read ${visibleCount} of ${keys.length} keys. Row sum: ${total.toFixed(8)}. Q head ${headIndex} shares KV head ${kvHead}.`;
      if (selectedKey !== null && selectedKey < keys.length) {
        detail += item.causal_mask[rowIndex][selectedKey]
          ? ` Selected key ${keys[selectedKey]} “${tokens[keys[selectedKey]]}”: attention weight ${row[selectedKey].toFixed(8)}.`
          : ` Selected key ${keys[selectedKey]} is a future position, so its weight is exactly zero.`;
      }
      readout.textContent = detail;
      facts.replaceChildren(fact("New queries", item.positions.length), fact("Available keys", keys.length), fact("Stored KV heads", trace.config.n_kv_heads));
      shapes.textContent = `Q ${shape(item.shapes.q)} · K/V cache ${shape(item.shapes.cached_k)} · attention weights ${shape(item.shapes.attention_weights)}. Tensor order: batch, head, position, feature (or key position for weights).`;
    }
    phase.addEventListener("change", () => { fillQueries(); render(); });
    layer.addEventListener("change", render);
    head.addEventListener("change", render);
    query.addEventListener("change", () => { selectedKey = null; render(); });
    fillQueries();
    render();
  }

  function cacheLab(root, trace) {
    labHeading(root, "Follow the cache through a pass", "Actual tensor sizes", "KV cache lab");
    const controls = el("div", "lab-controls");
    const phase = makeSelect(controls, "Forward pass", trace.passes.map((p, i) => [i, PASS_LABELS[p.name] || p.name]), 0);
    const flow = el("div", "token-flow");
    const readout = el("div", "lab-readout");
    readout.setAttribute("aria-live", "polite");
    const facts = el("div", "lab-facts");
    const shapes = el("p", "lab-note");
    const checks = el("div", "verifications");
    const verification = trace.verification.checks;
    checks.append(
      el("span", "verification", `Single/full error: ${verification.cached_single_vs_full_max_abs_error.toExponential(2)}`),
      el("span", "verification", `Chunk/full error: ${verification.cached_chunked_vs_full_max_abs_error.toExponential(2)}`),
      el("span", "verification", `Future leakage: ${verification.future_change_earlier_logits_max_abs_error}`)
    );
    root.append(controls, flow, readout, facts, shapes, checks);
    root.append(el("p", "lab-note", "Both decode examples branch from the same six-token prompt. The two-token chunk is a separate demonstration, not the next step after the one-token decode. Its second token is supplied explicitly. All weights are untrained."));
    function render() {
      const pass = trace.passes[Number(phase.value)];
      const layer = pass.layers[0];
      const previous = layer.positions[0];
      const current = layer.positions.length;
      const total = layer.key_positions.length;
      drawTokens(flow, contextTokens(trace, pass), previous);
      readout.textContent = pass.name === "prefill"
        ? `Process all ${current} prompt tokens and save their K/V at every layer. The last position's logits already provide the distribution for the first generated token.`
        : `Reuse ${previous} prompt positions. Process ${current} new ${current === 1 ? "token" : "tokens"} at absolute ${current === 1 ? "position" : "positions"} ${layer.positions.join(", ")}. The returned cache contains ${total} positions; the logits cover only the ${current} new ${current === 1 ? "position" : "positions"}.`;
      facts.replaceChildren(fact("Cached positions", total), fact("K/V payload", bytes(pass.cache_payload_bytes)), fact("New positions", current));
      shapes.textContent = `Input IDs ${shape(pass.input_shape)} · one layer's K/V ${shape(layer.shapes.cached_k)} · logits ${shape(pass.logits_shape)}. Total payload: 2 × 1 batch × ${trace.config.n_layers} layers × ${total} positions × ${trace.config.n_kv_heads} KV heads × ${trace.config.head_dim} features × 8 bytes = ${number(pass.cache_payload_bytes)} bytes. Metadata, weights, and temporary attention arrays are excluded.`;
    }
    phase.addEventListener("change", render);
    render();
  }

  function headLab(root, trace) {
    const config = trace.config;
    labHeading(root, "How much memory do the heads need?", "Calculated K/V payload", "Head sharing lab");
    const controls = el("div", "lab-controls");
    const design = makeSelect(controls, "KV heads", [[1, "1 · MQA"], [2, "2 · GQA"], [4, "4 · MHA"]], config.n_kv_heads);
    const tokenLabel = el("label", "", "Cached tokens");
    const length = el("input");
    length.type = "number";
    length.min = "1";
    length.max = "131072";
    length.step = "1";
    length.value = "6";
    tokenLabel.append(length);
    controls.append(tokenLabel);
    const precision = makeSelect(controls, "Storage precision", [[8, "float64 · 8 bytes"], [4, "float32 · 4 bytes"], [2, "FP16 / BF16 · 2 bytes"]], 8);
    const bars = el("div", "head-bars");
    const readout = el("div", "lab-readout");
    readout.setAttribute("aria-live", "polite");
    const facts = el("div", "lab-facts");
    const invalid = el("p", "lab-note");
    invalid.setAttribute("role", "status");
    root.append(controls, invalid, bars, readout, facts);
    root.append(el("p", "lab-note", `The calculator holds batch = 1, layers = ${config.n_layers}, query heads = ${config.n_heads}, and head dimension = ${config.head_dim} fixed. It calculates storage only; changing a selector does not retrain a model or predict its quality. The measured NumPy model uses two KV heads and float64.`));
    function render() {
      const t = Number(length.value);
      if (!Number.isInteger(t) || t < 1 || t > 131072 || length.value.trim() === "") {
        length.setAttribute("aria-invalid", "true");
        invalid.textContent = "Enter a whole number from 1 to 131,072. The previous valid calculation remains visible.";
        return;
      }
      length.removeAttribute("aria-invalid");
      invalid.textContent = "";
      const selected = Number(design.value);
      const perElement = Number(precision.value);
      const payload = kvHeads => 2 * config.n_layers * t * kvHeads * config.head_dim * perElement;
      const maximum = payload(4);
      bars.replaceChildren();
      [["MHA", 4], ["GQA", 2], ["MQA", 1]].forEach(([name, kvHeads]) => {
        const row = el("div", "head-bar");
        const label = el("span", "", name);
        const track = el("div", "bar-track");
        const fill = el("div", "bar-fill");
        fill.style.width = `${100 * payload(kvHeads) / maximum}%`;
        track.append(fill);
        const value = el("span", "", bytes(payload(kvHeads)));
        row.title = `${name}: ${kvHeads} KV heads, ${number(payload(kvHeads))} bytes`;
        if (kvHeads === selected) {
          label.textContent += " •";
          row.style.fontWeight = "600";
          row.setAttribute("aria-current", "true");
          fill.style.backgroundColor = "var(--green)";
        }
        row.append(label, track, value);
        bars.append(row);
      });
      const name = selected === 1 ? "MQA" : selected === 4 ? "MHA" : "GQA";
      readout.textContent = `${name}: 2 (K and V) × 1 batch × ${config.n_layers} layers × ${number(t)} tokens × ${selected} KV heads × ${config.head_dim} features × ${perElement} bytes = ${number(payload(selected))} bytes.`;
      facts.replaceChildren(fact("K/V payload", bytes(payload(selected))), fact("Saved vs MHA", `${100 * (1 - selected / 4)}%`), fact("Q heads per KV", config.n_heads / selected));
    }
    design.addEventListener("change", render);
    length.addEventListener("input", render);
    precision.addEventListener("change", render);
    render();
  }

  const SOURCE_STEPS = [
    { symbol: "rms_norm", label: "1 · RMSNorm", note: "Each token is normalized across its feature dimension. The learned scale then adjusts each feature. This operation does not mix different token positions." },
    { symbol: "apply_rope", label: "2 · Rotary positions", note: "Absolute positions control the rotations applied to Q and K. This file uses the Hugging Face Llama split-half convention: channel i pairs with i + head_dim/2. The convention must match the projection-weight layout when importing a checkpoint." },
    { symbol: "grouped_attention", label: "3 · GQA and the causal mask", note: "Query heads are grouped around shared K/V heads. The einsum uses compact K/V tensors directly. Comparing key_positions <= query_positions makes the mask work for both a full prompt and a cached multi-token chunk." },
    { symbol: "decoder_block", label: "4 · A complete decoder block", note: "Follow both residual paths: normalize → project Q/K/V → RoPE → append cache → attention → output projection → add the input. Then normalize → SwiGLU → add the second residual. Every layer owns its own K/V cache." },
    { symbol: "TinyLlama", label: "5 · The complete model", note: "The constructor creates deterministic random weights. The forward method looks up embeddings, applies two decoder blocks, normalizes the final hidden states, and projects to vocabulary logits. Logits at position t predict the token at position t + 1." },
    { symbol: "verify", label: "6 · Numerical verification", note: "These checks compare independently executed full, single-token cached, and chunked paths. They also test future-token leakage, GQA against explicitly repeated heads, RoPE rotation, compact cache shapes and bytes, and prefix-cache immutability." }
  ];

  function sourceBlock(source, symbol) {
    const lines = source.replace(/\r\n/g, "\n").split("\n");
    const pattern = new RegExp(`^(?:def|class) ${symbol}(?:\\(|:)`);
    const start = lines.findIndex(line => pattern.test(line));
    if (start === -1) throw new Error(`Source symbol ${symbol} was not found`);
    let end = start + 1;
    while (end < lines.length && !/^(?:def |class |@dataclass|if __name__)/.test(lines[end])) end += 1;
    while (end > start && lines[end - 1].trim() === "") end -= 1;
    return { start, end, lines: lines.slice(start, end) };
  }

  function sourceLab(root, source) {
    root.classList.add("source-walkthrough");
    labHeading(root, "Read the code that produced the trace", "Actual source file", "Source walkthrough");
    const controls = el("div", "lab-controls");
    const step = makeSelect(controls, "Code step", SOURCE_STEPS.map((item, i) => [i, item.label]), 0);
    const previous = el("button", "button secondary", "← Previous");
    const next = el("button", "button secondary", "Next →");
    previous.type = next.type = "button";
    controls.append(previous, next);
    const note = el("div", "source-step-note");
    note.setAttribute("aria-live", "polite");
    const location = el("p", "lab-note");
    const sourceLink = el("a", "", "Open the complete tiny_llama.py");
    sourceLink.href = SOURCE_URL;
    const pre = el("pre");
    const code = el("code", "language-python");
    pre.append(code);
    root.append(controls, note, location, pre, sourceLink);
    function render() {
      const index = Number(step.value);
      const entry = SOURCE_STEPS[index];
      const block = sourceBlock(source, entry.symbol);
      note.textContent = entry.note;
      location.textContent = `tiny_llama.py · lines ${block.start + 1}–${block.end}`;
      code.replaceChildren();
      block.lines.forEach((line, i) => {
        const row = el("span", "source-line");
        const lineNumber = el("span", "source-number", String(block.start + i + 1));
        lineNumber.setAttribute("aria-hidden", "true");
        row.append(lineNumber, el("span", "", line || "\u200b"));
        code.append(row);
      });
      pre.copyText = block.lines.join("\n");
      pre.scrollTop = 0;
      pre.scrollLeft = 0;
      previous.disabled = index === 0;
      next.disabled = index === SOURCE_STEPS.length - 1;
    }
    step.addEventListener("change", render);
    previous.addEventListener("click", () => { step.value = String(Number(step.value) - 1); render(); });
    next.addEventListener("click", () => { step.value = String(Number(step.value) + 1); render(); });
    render();
    addCodeCopy(root);
  }

  async function startWidget(root) {
    root.classList.add("lab");
    root.setAttribute("aria-busy", "true");
    root.replaceChildren(el("p", "lab-note", "Loading the saved computation…"));
    try {
      const type = root.dataset.widget;
      if (type === "source-lab") {
        const source = await getSource();
        root.replaceChildren();
        sourceLab(root, source);
      } else {
        const trace = await getTrace();
        root.replaceChildren();
        const builders = { "attention-lab": attentionLab, "cache-lab": cacheLab, "head-lab": headLab };
        if (!builders[type]) throw new Error("This notebook widget is not recognized");
        builders[type](root, trace);
      }
    } catch (_error) {
      root.replaceChildren();
      const message = el("p", "lab-readout", "This interactive example could not load its data. The article remains available; you can retry or open the accompanying file.");
      message.setAttribute("role", "alert");
      const retry = el("button", "button secondary", "Try again");
      retry.type = "button";
      retry.addEventListener("click", () => startWidget(root));
      const file = el("a", "text-link", "Open the data file ↗");
      file.href = root.dataset.widget === "source-lab" ? SOURCE_URL : TRACE_URL;
      const actions = el("div", "lab-controls");
      actions.append(retry, file);
      root.append(message, actions);
    } finally {
      root.removeAttribute("aria-busy");
    }
  }

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      try { await navigator.clipboard.writeText(text); return; } catch (_error) { /* use fallback */ }
    }
    const textarea = el("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.append(textarea);
    const previouslyFocused = document.activeElement;
    textarea.select();
    let success = false;
    try { success = document.execCommand("copy"); } finally {
      textarea.remove();
      if (previouslyFocused && previouslyFocused.focus) previouslyFocused.focus();
    }
    if (!success) throw new Error("Clipboard access unavailable");
  }

  function addCodeCopy(scope = document) {
    scope.querySelectorAll("pre").forEach(pre => {
      if (pre.dataset.copyReady === "true") return;
      const code = pre.querySelector("code");
      if (!code) return;
      pre.dataset.copyReady = "true";
      let wrapper = pre.parentElement;
      if (!wrapper.classList.contains("code-wrap")) {
        wrapper = el("div", "code-wrap");
        pre.before(wrapper);
        wrapper.append(pre);
      }
      const button = el("button", "code-copy", "Copy");
      button.type = "button";
      button.setAttribute("aria-label", "Copy this code block");
      button.setAttribute("aria-live", "polite");
      button.addEventListener("click", async () => {
        try {
          await copyText(typeof pre.copyText === "string" ? pre.copyText : code.textContent);
          button.textContent = "Copied";
        } catch (_error) {
          button.textContent = "Select to copy";
          button.setAttribute("aria-label", "Clipboard unavailable. Select this code and copy it manually.");
        }
        window.setTimeout(() => {
          button.textContent = "Copy";
          button.setAttribute("aria-label", "Copy this code block");
        }, 2200);
      });
      wrapper.append(button);
    });
  }

  function wrapTables() {
    document.querySelectorAll(".article-body table").forEach(table => {
      if (table.parentElement.classList.contains("table-wrap")) return;
      const wrapper = el("div", "table-wrap");
      wrapper.tabIndex = 0;
      wrapper.setAttribute("role", "region");
      const caption = table.querySelector("caption");
      wrapper.setAttribute("aria-label", caption ? caption.textContent : "Scrollable comparison table");
      table.before(wrapper);
      wrapper.append(table);
    });
  }

  function setupMenu() {
    const button = document.querySelector("[data-menu]");
    const nav = document.getElementById("site-nav");
    if (!button || !nav) return;
    const setOpen = open => {
      nav.classList.toggle("open", open);
      button.setAttribute("aria-expanded", String(open));
      button.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    };
    button.setAttribute("aria-controls", "site-nav");
    setOpen(false);
    button.addEventListener("click", () => setOpen(button.getAttribute("aria-expanded") !== "true"));
    nav.querySelectorAll("a").forEach(link => link.addEventListener("click", () => setOpen(false)));
    document.addEventListener("keydown", event => {
      if (event.key === "Escape" && nav.classList.contains("open")) { setOpen(false); button.focus(); }
    });
    document.addEventListener("click", event => {
      if (!nav.contains(event.target) && !button.contains(event.target)) setOpen(false);
    });
    const desktop = window.matchMedia("(min-width: 541px)");
    const onResize = event => { if (event.matches) setOpen(false); };
    if (desktop.addEventListener) desktop.addEventListener("change", onResize);
    else if (desktop.addListener) desktop.addListener(onResize);
  }

  function setupSearch() {
    const input = document.getElementById("question-search");
    if (!input) return;
    const cards = Array.from(document.querySelectorAll("[data-search]"));
    const status = document.getElementById("search-status");
    const empty = document.getElementById("no-results");
    if (status) { status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite"); }
    const normalize = text => text.normalize("NFKD").toLocaleLowerCase().replace(/[\u0300-\u036f]/g, "");
    function filter() {
      const terms = normalize(input.value).trim().split(/\s+/).filter(Boolean);
      let count = 0;
      cards.forEach(card => {
        const haystack = normalize(`${card.dataset.search || ""} ${card.textContent}`);
        const show = terms.every(term => haystack.includes(term));
        card.hidden = !show;
        card.style.display = show ? "" : "none";
        if (show) count += 1;
      });
      if (status) status.textContent = terms.length ? `${count} ${count === 1 ? "topic" : "topics"} found` : `${count} research ${count === 1 ? "topic" : "topics"}`;
      if (empty) { empty.hidden = count !== 0; empty.style.display = count === 0 ? "block" : "none"; }
    }
    input.addEventListener("input", filter);
    input.addEventListener("keydown", event => { if (event.key === "Escape") { input.value = ""; filter(); } });
    filter();
  }

  function setupReading() {
    const article = document.querySelector(".article-body");
    if (!article) return;
    const bar = document.getElementById("reading-progress");
    const headings = Array.from(article.querySelectorAll("h2[id]"));
    const links = Array.from(document.querySelectorAll('.toc-list a[href^="#"], .mobile-toc a[href^="#"]'));
    let pending = false;
    function update() {
      pending = false;
      const header = document.querySelector(".topbar");
      const offset = (header ? header.getBoundingClientRect().height : 80) + 30;
      if (bar) {
        const rect = article.getBoundingClientRect();
        const start = rect.top + window.scrollY - offset;
        const end = rect.bottom + window.scrollY - window.innerHeight;
        const fraction = end > start ? (window.scrollY - start) / (end - start) : window.scrollY >= start ? 1 : 0;
        const percent = Math.min(100, Math.max(0, fraction * 100));
        bar.style.width = `${percent}%`;
        bar.setAttribute("aria-valuenow", String(Math.round(percent)));
      }
      let active = headings[0];
      headings.forEach(heading => { if (heading.getBoundingClientRect().top <= offset + 4) active = heading; });
      links.forEach(link => {
        const matched = Boolean(active && decodeURIComponent(link.hash.slice(1)) === active.id);
        link.classList.toggle("active", matched);
        if (matched) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    }
    const schedule = () => { if (!pending) { pending = true; window.requestAnimationFrame(update); } };
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule, { passive: true });
    window.addEventListener("load", schedule);
    update();
  }

  function renderMath() {
    if (typeof window.renderMathInElement !== "function") return;
    const scope = document.querySelector(".article-body") || document.body;
    if (scope.dataset.mathRendered === "true") return;
    window.renderMathInElement(scope, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false,
      strict: "ignore",
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"]
    });
    scope.dataset.mathRendered = "true";
  }

  function boot() {
    setupMenu();
    setupSearch();
    setupReading();
    wrapTables();
    addCodeCopy(document.querySelector(".article-body") || document);
    renderMath();
    window.addEventListener("load", renderMath, { once: true });
    document.querySelectorAll("[data-widget]").forEach(root => startWidget(root));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot, { once: true });
  else boot();
})();
