"""Generate all visualization assets for visual-qc-project from JSON artifacts."""
import json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "neu-cls-case-study"
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

PAL = {"primary":"#2563EB","accent":"#16A34A","warn":"#D97706","danger":"#DC2626",
       "neutral":"#6B7280","highlight":"#7C3AED","bg":"#F8FAFC","grid":"#E2E8F0"}

def save(fig, name):
    p = ASSETS / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved -> {p.relative_to(ROOT)}")

# ── 1. Benchmark comparison ───────────────────────────────────────────────────
data = json.loads((DATA/"benchmark-comparison.json").read_text())
labels = {"dummy_baseline":"Dummy\nBaseline","logistic_regression":"Logistic\nRegression","random_forest":"Random\nForest *"}
models = [labels[d["model"]] for d in data]
acc = [d["accuracy"] for d in data]
f1  = [d["macro_f1"] for d in data]
prec= [d["macro_precision"] for d in data]
x = np.arange(len(models)); w = 0.28
fig, ax = plt.subplots(figsize=(9,5), facecolor=PAL["bg"]); ax.set_facecolor(PAL["bg"])
b1 = ax.bar(x-w, acc,  w, label="Accuracy",    color=PAL["primary"],   alpha=0.85)
b2 = ax.bar(x,   f1,   w, label="Macro F1",    color=PAL["accent"],    alpha=0.85)
b3 = ax.bar(x+w, prec, w, label="Macro Prec.", color=PAL["highlight"], alpha=0.85)
for bar in list(b1)+list(b2)+list(b3):
    h=bar.get_height()
    if h>0.01: ax.text(bar.get_x()+bar.get_width()/2, h+0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9); ax.set_ylim(0,1.12)
ax.set_title("Model Benchmark — NEU-CLS Steel Defect Dataset\n(* = selected model)", fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=9, framealpha=0.7); ax.yaxis.grid(True, color=PAL["grid"], linewidth=0.8)
ax.set_axisbelow(True); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
save(fig, "model-comparison.png")

# ── 2. Confusion matrix ───────────────────────────────────────────────────────
cm_data = json.loads((DATA/"confusion-matrix.json").read_text())
labels_list = cm_data["labels"]
matrix = np.array(cm_data["matrix"])
short = [l.replace("_"," ").replace("-","‑") for l in labels_list]

fig, ax = plt.subplots(figsize=(7,6), facecolor=PAL["bg"]); ax.set_facecolor(PAL["bg"])
im = ax.imshow(matrix, cmap=plt.cm.Blues, vmin=0, vmax=matrix.max())
for i in range(len(labels_list)):
    for j in range(len(labels_list)):
        v = matrix[i,j]
        color = "white" if v > matrix.max()*0.6 else "#1E293B"
        ax.text(j, i, str(v), ha="center", va="center", fontsize=9, fontweight="bold" if i==j else "normal", color=color)
ax.set_xticks(range(len(labels_list))); ax.set_yticks(range(len(labels_list)))
ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
ax.set_yticklabels(short, fontsize=8)
ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("Actual", fontsize=10)
ax.set_title("Confusion Matrix — Random Forest on HOG Features\n(NEU-CLS holdout, n=360)", fontsize=11, fontweight="bold", pad=12)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
save(fig, "confusion-matrix.png")

# ── 3. Per-class precision / recall / F1 ─────────────────────────────────────
cls_data = json.loads((DATA/"class-metrics.json").read_text())
cls_names = [d["class_name"].replace("_"," ") for d in cls_data]
prec_v = [d["precision"] for d in cls_data]
rec_v  = [d["recall"]    for d in cls_data]
f1_v   = [d["f1"]        for d in cls_data]
x = np.arange(len(cls_names)); w=0.26
fig, ax = plt.subplots(figsize=(10,5), facecolor=PAL["bg"]); ax.set_facecolor(PAL["bg"])
ax.bar(x-w, prec_v, w, label="Precision", color=PAL["primary"],   alpha=0.85)
ax.bar(x,   f1_v,   w, label="F1",        color=PAL["accent"],    alpha=0.85)
ax.bar(x+w, rec_v,  w, label="Recall",    color=PAL["highlight"], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(cls_names, fontsize=9)
ax.set_ylim(0.75, 1.05); ax.set_ylabel("Score", fontsize=10)
ax.set_title("Per-Class Metrics — Random Forest on NEU-CLS", fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=9, framealpha=0.7); ax.yaxis.grid(True, color=PAL["grid"], linewidth=0.8)
ax.set_axisbelow(True); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
save(fig, "class-metrics.png")

# ── 4. Pareto — error distribution ───────────────────────────────────────────
pareto = json.loads((DATA/"pareto.json").read_text())
p_items = pareto["pareto"]
cls_p = [d["class_name"].replace("_"," ") for d in p_items]
err_p = [d["error_count"] for d in p_items]
cum_p = [d["cumulative_share"]*100 for d in p_items]

fig, ax1 = plt.subplots(figsize=(8,5), facecolor=PAL["bg"]); ax1.set_facecolor(PAL["bg"])
ax2 = ax1.twinx()
bars = ax1.bar(cls_p, err_p, color=PAL["danger"], alpha=0.8)
for bar, v in zip(bars, err_p):
    ax1.text(bar.get_x()+bar.get_width()/2, v+0.1, str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.plot(cls_p, cum_p, "o-", color=PAL["primary"], linewidth=2, markersize=6, label="Cumulative %")
ax2.axhline(80, color=PAL["neutral"], linestyle="--", linewidth=1, label="80% line")
ax1.set_ylabel("Error count", fontsize=10); ax2.set_ylabel("Cumulative share (%)", fontsize=10)
ax2.set_ylim(0,115)
ax1.set_title("Pareto — Error Distribution by Defect Class\n(scratches + pitted_surface = 64% of all errors)", fontsize=11, fontweight="bold", pad=12)
ax2.legend(fontsize=9, framealpha=0.7, loc="center right")
ax1.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
save(fig, "pareto-errors.png")

# ── 5. Review queue curve ─────────────────────────────────────────────────────
rq = json.loads((DATA/"review-queue.json").read_text())
budgets = rq["review_budgets"]
fracs   = [0.0] + [b["review_fraction"] for b in budgets]
capture = [0.0] + [b["error_capture_rate"] for b in budgets]
rand_l  = [b["review_fraction"] for b in budgets]
fig, ax = plt.subplots(figsize=(8,5), facecolor=PAL["bg"]); ax.set_facecolor(PAL["bg"])
ax.plot(fracs, capture, "o-", color=PAL["primary"], linewidth=2.2, markersize=7, label="Model (entropy-ranked queue)")
ax.plot([0]+rand_l, [0]+rand_l, "--", color=PAL["neutral"], linewidth=1.5, label="Random review baseline")
for b in budgets:
    ax.annotate(f"{b['error_capture_rate']:.0%} (×{b['yield_lift_vs_random']:.1f})",
                xy=(b["review_fraction"], b["error_capture_rate"]),
                xytext=(b["review_fraction"]+0.005, b["error_capture_rate"]-0.06),
                fontsize=8, color="#1E293B")
ax.set_xlim(-0.01, 0.26); ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("Fraction of samples reviewed", fontsize=10)
ax.set_ylabel("Error capture rate", fontsize=10)
ax.set_title("Review Queue — Error Capture vs Inspection Budget", fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=9, framealpha=0.7)
ax.yaxis.grid(True, color=PAL["grid"], linewidth=0.8); ax.xaxis.grid(True, color=PAL["grid"], linewidth=0.8)
ax.set_axisbelow(True); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
save(fig, "review-queue-curve.png")

print("\nAll 5 charts generated ✅")
