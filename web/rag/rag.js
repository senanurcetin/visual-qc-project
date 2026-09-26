// Steel QC RAG map: data loading, answer panel, 2D map, metric cards and lazy 3D.
// The 3D renderer (rag3d.js + Three.js) is only imported when the map scrolls into view
// and the device can use it; phones and reduced-motion users get the 2D canvas map.

export const TOPIC_COLORS = ["#f472b6", "#fb923c", "#a3e635", "#60a5fa", "#c084fc", "#2dd4bf"];

const $ = (id) => document.getElementById(id);
const pct = (v, digits = 0) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);
const num = (v, digits = 2) => (v == null ? "—" : Number(v).toFixed(digits));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const state = { data: null, mode: null, active: 0, renderer: null, loading3d: false, visible: false };

function supportsWebGL() {
    try {
        const c = document.createElement("canvas");
        return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl")));
    } catch { return false; }
}

const narrow = window.matchMedia("(max-width: 767px)");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

// ── Rendering helpers ───────────────────────────────────────────────────────
function renderKpis(d) {
    const s = d.summary;
    const items = [
        ["Passages indexed", s.corpus.passages, `${s.corpus.articles} articles`],
        ["Right passage in top 4", pct(s.retrieval.hit_at_4), `TF-IDF ${pct(s.tfidf_baseline.hit_at_4)}`],
        ["Answer sentences supported", pct(d.generation.rag.sentence_support_rate), `closed-book ${pct(d.generation.closed_book.sentence_support_rate)}`],
        ["Embedding dimensions", s.dimensions, "shown in 3D"],
    ];
    $("kpis").innerHTML = items.map(([k, v, sub]) =>
        `<div class="kpi"><dt>${k}</dt><dd>${v}<small>${sub}</small></dd></div>`).join("");
}

function renderSteps(d) {
    const s = d.summary;
    const c = d.corpus;
    $("step-chunk").textContent = `${c.articles.length} Wikipedia articles on steel making, surface defects, corrosion, inspection and SPC are split into ${c.passages} section-aware passages of ~140 words.`;
    $("step-embed").textContent = `${s.embedding_model.split("/").pop()} turns each passage into a ${s.dimensions}-dimensional vector; passages with similar meaning land close together. The vectors are stored in a Chroma vector database.`;
    $("step-answer").textContent = `The question is embedded the same way, Chroma returns the ${d.samples.top_k} nearest passages, and ${s.generator.split("/").pop()} answers from them with citations. Everything runs locally; no paid API.`;
}

function renderLegend(d) {
    $("legend").innerHTML = d.map.topics.map((t, i) =>
        `<li><span class="dot" style="background:${TOPIC_COLORS[i]}"></span>${esc(t)}</li>`).join("");
}

function renderQuestions(d) {
    const box = $("questions");
    box.innerHTML = d.samples.samples.map((s, i) =>
        `<button type="button" class="q-btn" data-i="${i}" aria-pressed="${i === state.active}">${esc(s.question)}</button>`).join("");
    box.addEventListener("click", (e) => {
        const btn = e.target.closest(".q-btn");
        if (btn) selectQuestion(Number(btn.dataset.i));
    });
}

function answerHtml(text) {
    return esc(text).replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, '<span class="cite">[$1]</span>');
}

function claimsHtml(f) {
    if (f.abstained) return `<p class="card-note">The model declined: the retrieved passages do not state an answer to this question. Compare the control below, where the same model answers anyway without sources.</p>`;
    return `<ul class="claims">${f.sentences.map((c) => `
        <li class="claim ${c.supported ? "ok" : "no"}">${esc(c.text)}
            <span class="claim-meta">${c.supported ? `supported by [${c.best_passage}]` : "not supported by the retrieved passages"} · entailment ${num(c.entailment)}</span>
        </li>`).join("")}</ul>`;
}

function renderAnswer() {
    const d = state.data;
    const s = d.samples.samples[state.active];
    const f = s.faithfulness;
    const cf = s.closed_book_faithfulness;
    const supported = f.sentences.filter((c) => c.supported).length;
    $("answer").innerHTML = `
        <h3>Answer</h3>
        <p class="answer-text">${answerHtml(s.answer)}</p>
        <div class="subhead"><span>Claim check</span><span class="badge">${f.abstained ? "declined" : `${supported}/${f.sentences.length} supported`}</span></div>
        ${claimsHtml(f)}
        <div class="subhead"><span>Retrieved passages</span><span>cosine similarity</span></div>
        <ul class="sources">${s.retrieved.map((r) => `
            <li><details class="src" data-index="${r.index}">
                <summary><span class="rank">[${r.rank}]</span>
                    <span class="src-title"><span class="dot" style="background:${TOPIC_COLORS[d.map.topics.indexOf(r.topic)]}"></span>
                    ${esc(r.article)}${r.section === "Introduction" ? "" : ` — ${esc(r.section)}`}</span>
                    <span class="sim">${num(r.similarity, 3)}</span></summary>
                <p>${esc(r.text)}</p>
            </details></li>`).join("")}</ul>
        <details class="control">
            <summary>Control: the same model without retrieval (${cf.sentences.filter((c) => c.supported).length}/${cf.sentences.length} sentences supported)</summary>
            <p>${esc(s.closed_book_answer)}</p>
        </details>`;
    $("answer").querySelectorAll(".src").forEach((el) => {
        el.addEventListener("toggle", () => state.renderer?.focus?.(el.open ? Number(el.dataset.index) : null));
    });
}

function renderHonesty() {
    const d = state.data;
    const s = d.samples.samples[state.active];
    const k = d.samples.top_k;
    const view = state.mode === "3d" ? "3D" : "2D";
    const overlap = state.mode === "3d" ? s.overlap_3d : s.overlap_2d;
    const fid = d.fidelity;
    const mean = state.mode === "3d" ? fid.query_neighbor_overlap.mean_overlap_3d : fid.query_neighbor_overlap.mean_overlap_2d;
    $("honesty").innerHTML = `<strong>Read this map with care.</strong> Glowing passages are the top ${k} results of the full ${d.summary.dimensions}-dimensional cosine search in Chroma, not the dots that happen to look closest here. Squeezing ${d.summary.dimensions} dimensions into ${view} distorts distances: for this question, ${overlap} of the ${k} dots nearest the question in this ${view} view are real results (average across all ${fid.query_neighbor_overlap.queries} evaluation questions: ${num(mean, 1)} of ${k}). Local neighbourhoods between passages hold up well (UMAP trustworthiness ${num(fid.trustworthiness[state.mode === "3d" ? "umap_3d" : "umap_2d"])}); it is the question-to-passage distances that are unreliable.`;
}

function bars(rows, max = 1) {
    return rows.map(([label, value, color, fmt]) => `
        <div class="bar-row"><span>${label}</span>
            <span class="bar-track"><span class="bar-fill" style="width:${Math.max(0, Math.min(1, value / max)) * 100}%;background:${color}"></span></span>
            <span class="bar-val">${fmt ? fmt(value) : pct(value)}</span></div>`).join("");
}

function renderCards(d) {
    const r = d.retrieval;
    $("card-retrieval").innerHTML = `
        <h3>Retrieval</h3>
        <p class="card-note">${r.generated_questions} generated questions with a known source passage, ${r.curated_questions} hand-written questions labelled by article.</p>
        <div class="key"><span><i style="background:#60a5fa"></i>Hit@1</span><span><i style="background:#6ee7b7"></i>Hit@4</span><span><i style="background:#c084fc"></i>MRR@10</span></div>
        <div class="bars">${r.methods.map((m) => `
            <div><div class="bar-group-label"><span>${esc(m.label)}</span>${m.selected ? '<span class="tag">selected</span>' : ""}</div>
            ${bars([["Hit@1", m.hit_at_1, "#60a5fa"], ["Hit@4", m.hit_at_4, "#6ee7b7"], ["MRR@10", m.mrr_at_10, "#c084fc", (v) => num(v)]])}</div>`).join("")}
        </div>
        <p class="card-foot">Hand-written questions: the right article is in the top 5 for ${pct(d.summary.retrieval.curated_article_hit_at_5)} with the selected model.</p>`;

    const g = d.generation;
    const promptLabel = { v1_basic: "Basic", v2_strict: "Strict" };
    const groups = Object.entries(g.generators).map(([gkey, spec]) => {
        const rows = [["No RAG", g.closed_book_by_generator[gkey].sentence_support_rate, "#8da4c4"]];
        Object.values(g.by_config).filter((c) => c.generator === gkey).forEach((c) => {
            const chosen = `${c.generator}/${c.prompt}` === g.selected_config;
            rows.push([`${promptLabel[c.prompt] || c.prompt}${chosen ? " ✓" : ""}`, c.sentence_support_rate, chosen ? "#6ee7b7" : "#3b82f6"]);
        });
        return `<div><div class="bar-group-label"><span>${esc(spec.label)}</span>${gkey === g.selected_generator ? '<span class="tag">selected</span>' : ""}</div>${bars(rows)}</div>`;
    });
    $("card-faithfulness").innerHTML = `
        <h3>Faithfulness</h3>
        <p class="card-note">Share of answer sentences entailed by the retrieved passages (${g.rag.questions} questions, NLI threshold ${g.entailment_threshold}). “No RAG” is the same model answering without retrieval, scored against the same passages.</p>
        <div class="key"><span><i style="background:#8da4c4"></i>No retrieval</span><span><i style="background:#3b82f6"></i>RAG</span><span><i style="background:#6ee7b7"></i>RAG, used on this page</span></div>
        <div class="bars">${groups.join("")}</div>
        <p class="card-foot">Basic prompt: answer from the passages and cite them. Strict prompt: additionally forbids background knowledge. The model and prompt shown on this page were chosen by mean faithfulness, not by eye.</p>`;

    const f = d.fidelity;
    $("card-fidelity").innerHTML = `
        <h3>Projection fidelity</h3>
        <p class="card-note">How much does squeezing ${d.summary.dimensions} dimensions into 3 or 2 distort the picture?</p>
        <div class="bars">
            <div><div class="bar-group-label"><span>Trustworthiness (k=${f.neighbors_k})</span></div>
            ${bars([["UMAP 3D", f.trustworthiness.umap_3d, "#6ee7b7", (v) => num(v)], ["UMAP 2D", f.trustworthiness.umap_2d, "#60a5fa", (v) => num(v)], ["PCA 3D", f.trustworthiness.pca_3d, "#8da4c4", (v) => num(v)]])}</div>
            <div><div class="bar-group-label"><span>Real top-${f.query_neighbor_overlap.k} among nearest on screen</span></div>
            ${bars([["3D", f.query_neighbor_overlap.mean_overlap_3d, "#fbbf24", (v) => `${num(v, 1)}/${f.query_neighbor_overlap.k}`], ["2D", f.query_neighbor_overlap.mean_overlap_2d, "#fbbf24", (v) => `${num(v, 1)}/${f.query_neighbor_overlap.k}`]], f.query_neighbor_overlap.k)}</div>
        </div>
        <p class="card-foot">Passage-to-passage neighbourhoods survive the projection; question-to-passage distances do not. That is why highlights come from the full-dimensional search.</p>`;
}

function renderMethod(d) {
    const s = d.summary;
    const g = d.generation;
    $("method").innerHTML = `
        <article class="card"><h3>Method</h3><ul>
            <li>Knowledge base: ${d.corpus.articles.length} English Wikipedia articles (${esc(d.corpus.license)}), pinned to revision ids, max ${d.corpus.max_passages_per_article} passages per article.</li>
            <li>Retrievers compared: TF-IDF keyword baseline and three sentence-embedding models; the best MRR@10 is indexed in Chroma (HNSW, cosine). Chroma matched exact search on ${pct(s.chroma_exact_agreement_top_k)} of queries.</li>
            <li>Answers: ${esc(s.generator)} run locally with greedy decoding; the example questions are precomputed, so no API key sits in this page.</li>
            <li>Faithfulness: ${esc(s.nli_model)} scores each answer sentence against 1–3 sentence windows of the retrieved passages; a sentence counts as supported at entailment ≥ ${g.entailment_threshold}.</li>
            <li>Map: UMAP (cosine, 15 neighbours, min_dist 0.4) fitted on passage vectors; questions are placed with the same fitted transform.</li>
        </ul></article>
        <article class="card"><h3>Limits</h3><ul>
            <li>Generated evaluation questions come from a single passage each, which makes retrieval easier than real user questions; the hand-written set is small (${d.retrieval.curated_questions} questions).</li>
            <li>Faithfulness is not correctness: a supported sentence can still be an incomplete answer, and the NLI model has its own errors, especially on paraphrase.</li>
            <li>Small local generators (1.5B–3B parameters) still add background knowledge the sources do not state; the claim check makes that visible instead of hiding it.</li>
            <li>The knowledge base is encyclopaedic, not a plant's own SOPs, so this shows the method rather than a deployable assistant.</li>
        </ul></article>`;
    $("footer").innerHTML = `Passages are from English Wikipedia and are available under <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>. Source articles: ${d.corpus.articles.map((a) => `<a href="${esc(a.url)}">${esc(a.title)}</a>`).join(", ")}.`;
}

// ── Tooltip shared by both renderers ────────────────────────────────────────
export function showTooltip(index, x, y) {
    const tip = $("tooltip");
    if (index == null) { tip.hidden = true; return; }
    const m = state.data.map;
    const p = m.points[index];
    const frame = $("map-frame").getBoundingClientRect();
    tip.innerHTML = `<strong>${esc(m.articles[p[6]])}</strong><div class="tt-meta">${esc(m.sections[index])} · ${esc(m.topics[p[5]])}</div>${esc(m.previews[index])}`;
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${Math.min(Math.max(8, x + 14), frame.width - w - 8)}px`;
    tip.style.top = `${Math.min(Math.max(8, y + 14), frame.height - h - 8)}px`;
}

// ── 2D canvas renderer ──────────────────────────────────────────────────────
function createMap2D(frame, data) {
    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "2D map of passage embeddings coloured by topic, with the selected question and its retrieved passages highlighted");
    frame.appendChild(canvas);
    frame.classList.add("is-2d");
    const ctx = canvas.getContext("2d");
    const pts = data.map.points;
    let sample = null, focused = null, screen = [], dims = { w: 0, h: 0 };

    function layout() {
        const rect = frame.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        dims = { w: rect.width, h: rect.height };
        const pad = 24, size = Math.min(rect.width, rect.height) / 2 - pad;
        const cx = rect.width / 2, cy = rect.height / 2;
        const project = (x, y) => [cx + x * size, cy - y * size];
        screen = pts.map((p) => project(p[3], p[4]));
        return project;
    }

    let project = layout();

    function draw() {
        ctx.clearRect(0, 0, dims.w, dims.h);
        const retrieved = new Set(sample ? sample.retrieved.map((r) => r.index) : []);
        pts.forEach((p, i) => {
            if (retrieved.has(i)) return;
            ctx.globalAlpha = sample ? 0.35 : 0.8;
            ctx.fillStyle = TOPIC_COLORS[p[5]];
            ctx.beginPath(); ctx.arc(screen[i][0], screen[i][1], 2.4, 0, Math.PI * 2); ctx.fill();
        });
        ctx.globalAlpha = 1;
        if (!sample) return;
        const [qx, qy] = project(sample.query_2d[0], sample.query_2d[1]);
        ctx.strokeStyle = "rgba(230, 239, 255, 0.75)"; ctx.lineWidth = 1.2;
        sample.retrieved.forEach((r) => {
            ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(screen[r.index][0], screen[r.index][1]); ctx.stroke();
        });
        sample.retrieved.forEach((r) => {
            const [x, y] = screen[r.index];
            const big = focused === r.index;
            ctx.fillStyle = TOPIC_COLORS[pts[r.index][5]];
            ctx.beginPath(); ctx.arc(x, y, big ? 9 : 6.5, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2; ctx.stroke();
            ctx.fillStyle = "#07111f"; ctx.font = "600 10px 'IBM Plex Mono', monospace";
            ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(r.rank), x, y + 0.5);
        });
        ctx.save(); ctx.translate(qx, qy); ctx.rotate(Math.PI / 4);
        ctx.fillStyle = "#ffffff"; ctx.fillRect(-6, -6, 12, 12); ctx.restore();
        ctx.fillStyle = "#ffffff"; ctx.font = "600 12px Inter, sans-serif"; ctx.textAlign = "left";
        ctx.fillText("question", qx + 11, qy - 10);
    }

    function nearest(x, y) {
        let best = null, bestD = 100; // 10px radius, squared
        screen.forEach(([sx, sy], i) => {
            const dd = (sx - x) ** 2 + (sy - y) ** 2;
            if (dd < bestD) { bestD = dd; best = i; }
        });
        return best;
    }

    const onMove = (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left, y = e.clientY - rect.top;
        showTooltip(nearest(x, y), x, y);
    };
    const onLeave = () => showTooltip(null);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    const ro = new ResizeObserver(() => { project = layout(); draw(); });
    ro.observe(frame);

    return {
        setQuery(s) { sample = s; focused = null; draw(); },
        focus(i) { focused = i; draw(); },
        destroy() { ro.disconnect(); canvas.remove(); frame.classList.remove("is-2d"); showTooltip(null); },
    };
}

// ── Mode handling and lazy 3D ───────────────────────────────────────────────
async function mount(mode) {
    const frame = $("map-frame");
    state.renderer?.destroy();
    state.renderer = null;
    state.mode = mode;
    document.querySelectorAll("#mode-toggle button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    const placeholder = $("map-placeholder");
    if (mode === "3d") {
        placeholder.hidden = false;
        placeholder.textContent = "Loading 3D map…";
        try {
            state.loading3d = true;
            const { createMap3D } = await import("./rag3d.js");
            if (state.mode !== "3d") return;
            state.renderer = createMap3D(frame, state.data, { colors: TOPIC_COLORS, reducedMotion: reducedMotion.matches, onHover: showTooltip });
        } catch (err) {
            console.warn("3D map unavailable, falling back to 2D", err);
            return mount("2d");
        } finally {
            state.loading3d = false;
        }
    } else {
        state.renderer = createMap2D(frame, state.data);
    }
    placeholder.hidden = true;
    state.renderer.setQuery(state.data.samples.samples[state.active]);
    renderHonesty();
}

function selectQuestion(i) {
    state.active = i;
    document.querySelectorAll(".q-btn").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.i) === i)));
    renderAnswer();
    renderHonesty();
    state.renderer?.setQuery(state.data.samples.samples[i]);
}

function preferredMode() {
    return narrow.matches || reducedMotion.matches || !supportsWebGL() ? "2d" : "3d";
}

async function init() {
    const res = await fetch(document.body.dataset.src);
    if (!res.ok) throw new Error(`Could not load RAG artifacts (${res.status})`);
    state.data = await res.json();
    const d = state.data;
    renderKpis(d); renderSteps(d); renderLegend(d); renderQuestions(d); renderAnswer(); renderCards(d); renderMethod(d);
    state.mode = preferredMode();
    renderHonesty();

    const toggle = $("mode-toggle");
    if (supportsWebGL() && !narrow.matches) {
        toggle.hidden = false;
        toggle.addEventListener("click", (e) => {
            const btn = e.target.closest("button");
            if (btn && btn.dataset.mode !== state.mode && state.visible) mount(btn.dataset.mode);
            else if (btn) { state.mode = btn.dataset.mode; renderHonesty(); }
        });
    }
    narrow.addEventListener("change", () => { if (narrow.matches && state.mode === "3d" && state.visible) mount("2d"); });

    // Mount (and only then fetch Three.js) once the map is about to be seen.
    const io = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting) && !state.visible) {
            state.visible = true;
            io.disconnect();
            mount(state.mode);
        }
    }, { rootMargin: "200px" });
    io.observe($("map-frame"));
}

init().catch((err) => {
    console.error(err);
    $("map-placeholder").textContent = err.message.startsWith("Could not load")
        ? `${err.message}. Run python analysis/run_rag_case_study.py.`
        : "The RAG map failed to render. See the browser console for details.";
});
