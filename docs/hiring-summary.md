# Hiring Summary — Visual QC Project

## One-line summary

Computer vision analytics case study: steel surface defect classification on NEU-CLS (1,800 images, 6 classes), with per-class cost weighting, Pareto error analysis, entropy-based review queue, and operator-facing Flask dashboard.

## Headline metrics

| Metric | Value |
|--------|-------|
| Final model | Random Forest on HOG + Gabor + grid features |
| Accuracy | **0.9389** |
| Macro F1 | **0.9392** |
| Macro Precision | **0.9421** |
| Error rate | 6.1% (22 / 360 holdout samples) |
| Review queue | Top 20% entropy captures **81.8%** of errors — 4.1× yield vs random |
| Pareto insight | 2 classes (scratches + pitted_surface) = 64% of all errors |

## Skills demonstrated

| Skill | Evidence |
|-------|----------|
| **Computer vision** | HOG descriptors + Gabor filter bank + grid statistics pipeline |
| **Multi-class classification** | 6-class balanced problem, macro metrics (not accuracy-only) |
| **Error cost analysis** | Per-class FN costs ($100–$500) used to prioritise review routing |
| **Pareto analysis** | Identifies top 2 error classes driving 64% of all defects |
| **Review queue design** | Entropy-ranked routing captures 82% of errors at 20% inspection budget |
| **Business framing** | Model output → quality decision routing, not just label prediction |
| **Python stack** | scikit-image, scikit-learn, matplotlib, Flask, SQLite |
| **CI** | GitHub Actions: syntax check + unit tests on every push |

## Interview-ready talking points

1. Accuracy alone is misleading for quality inspection — a model that is always right on easy classes but consistently misses high-cost inclusions is worse than its accuracy score suggests. Per-class F1 and cost-weighted evaluation are the right metrics here.
2. The Pareto chart shows two classes (scratches, pitted_surface) account for 64% of errors — this directly informs where to focus model improvement or operator training.
3. Entropy-based review queue design: instead of routing all 360 holdout samples to human review, reviewing only the top 20% highest-entropy samples catches 82% of all errors. This is the business value of calibrated uncertainty.
4. The confused pairs (scratches → inclusion, inclusion → pitted_surface) have a visual explanation — both pairs involve elongated or textural surface patterns that share low-frequency HOG features. A Gabor bank adds some separation, but deep features would likely resolve this.
