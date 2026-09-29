"""Generate README charts for the RAG knowledge-assistant layer from its JSON artifacts."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "rag-knowledge-assistant"
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

PAL = {"primary":"#2563EB","accent":"#16A34A","warn":"#D97706","danger":"#DC2626",
       "neutral":"#6B7280","highlight":"#7C3AED","bg":"#F8FAFC","grid":"#E2E8F0"}
TOPIC_COLORS = ["#DB2777", "#EA580C", "#65A30D", "#2563EB", "#9333EA", "#0D9488"]

def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))

def save(fig, name):
    p = ASSETS / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved -> {p.relative_to(ROOT)}")

def style(ax):
    ax.set_facecolor(PAL["bg"])
    ax.yaxis.grid(True, color=PAL["grid"], linewidth=0.8); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ── 1. Retrieval benchmark ────────────────────────────────────────────────────
rb = load("retrieval-benchmark.json")
methods = rb["methods"]
names = [m["label"].replace(" (", "\n(") + (" *" if m["selected"] else "") for m in methods]
series = [("Hit@1", "hit_at_1", PAL["primary"]), (f"Hit@{rb['top_k']}", "hit_at_4", PAL["accent"]),
          ("MRR@10", "mrr_at_10", PAL["highlight"])]
x = np.arange(len(methods)); w = 0.26
fig, ax = plt.subplots(figsize=(9.5, 5), facecolor=PAL["bg"]); style(ax)
for k, (label, key, color) in enumerate(series):
    vals = [m[key] for m in methods]
    bars = ax.bar(x + (k - 1) * w, vals, w, label=label, color=color, alpha=0.85)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.2f}", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5); ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=10)
ax.set_title(f"Retrieval Benchmark — {rb['generated_questions']} questions with a known source passage\n(* = indexed in Chroma)",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=9, framealpha=0.7, loc="upper left")
save(fig, "rag-retrieval-benchmark.png")

# ── 2. Faithfulness ───────────────────────────────────────────────────────────
ge = load("generation-eval.json")
gens = list(ge["generators"])
variants = [("No retrieval", None, PAL["neutral"]), ("RAG, basic prompt", "v1_basic", PAL["primary"]),
            ("RAG, strict prompt", "v2_strict", PAL["highlight"])]
x = np.arange(len(gens)); w = 0.26
fig, ax = plt.subplots(figsize=(9, 5), facecolor=PAL["bg"]); style(ax)
for k, (label, prompt, color) in enumerate(variants):
    vals = [ge["closed_book_by_generator"][g]["sentence_support_rate"] if prompt is None
            else ge["by_config"][f"{g}/{prompt}"]["sentence_support_rate"] for g in gens]
    bars = ax.bar(x + (k - 1) * w, vals, w, label=label, color=color, alpha=0.85)
    for g, b in zip(gens, bars, strict=False):
        chosen = prompt is not None and f"{g}/{prompt}" == ge["selected_config"]
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.0%}" + (" *" if chosen else ""),
                ha="center", fontsize=9, fontweight="bold" if chosen else "normal")
ax.set_xticks(x); ax.set_xticklabels([ge["generators"][g]["label"] for g in gens], fontsize=9.5)
ax.set_ylim(0, 1.05); ax.set_ylabel("Answer sentences supported by retrieved passages", fontsize=9.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_title(f"Faithfulness — NLI entailment check, {ge['rag']['questions']} questions\n(* = generator and prompt used on the /rag page)",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=9, framealpha=0.7, loc="upper left")
save(fig, "rag-faithfulness.png")

# ── 3. Embedding map (2D projection with one showcase query) ─────────────────
em = load("embedding-map.json")
sq = load("sample-queries.json")
pts = np.array([p[:5] for p in em["points"]], dtype=float)
topic = np.array([p[5] for p in em["points"]])
sample = sq["samples"][0]
fig, ax = plt.subplots(figsize=(9, 7.5), facecolor=PAL["bg"]); ax.set_facecolor(PAL["bg"])
for t, name in enumerate(em["topics"]):
    m = topic == t
    ax.scatter(pts[m, 3], pts[m, 4], s=9, color=TOPIC_COLORS[t], alpha=0.55, label=name, linewidths=0)
qx, qy = sample["query_2d"]
for r in sample["retrieved"]:
    px, py = pts[r["index"], 3], pts[r["index"], 4]
    ax.plot([qx, px], [qy, py], color="#0F172A", linewidth=1, alpha=0.7)
    ax.scatter([px], [py], s=90, color=TOPIC_COLORS[topic[r["index"]]], edgecolors="#0F172A", linewidths=1.4, zorder=3)
    ax.annotate(str(r["rank"]), (px, py), xytext=(5, 5), textcoords="offset points", fontsize=8, fontweight="bold")
ax.scatter([qx], [qy], s=120, marker="D", color="#0F172A", zorder=4, label="Question")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title(f"Embedding map — {len(pts)} passages, UMAP 2D of {em['dimensions']}-d vectors\n"
             f"Q: “{sample['question']}” → top-{sq['top_k']} from full-dimensional search",
             fontsize=11, fontweight="bold", pad=12)
leg = ax.legend(fontsize=8.5, framealpha=0.8, loc="lower left")
for handle in leg.legend_handles:
    handle.set_sizes([36])
save(fig, "rag-embedding-map.png")

print("\nRAG charts generated")
