# Hiring Summary — Visual QC Project

Live demo: https://visual-qc-project-pearl.vercel.app (HMI + 3D line twin) · https://visual-qc-project-pearl.vercel.app/rag (RAG map)

## One-line summary

Computer vision analytics case study: steel surface defect classification on NEU-CLS (1,800 images, 6 classes), with per-class cost weighting, Pareto error analysis, entropy-based review queue, and operator-facing Flask dashboard. An AI engineering layer adds an embedding-based RAG assistant over a steel QC knowledge base, with measured retrieval and answer faithfulness.

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
| RAG retrieval | Exact source passage in top 4 for **94.8%** of 77 questions (TF-IDF baseline 89.6%) |
| RAG faithfulness | **75.7%** of answer sentences entailed by the retrieved passages (same model without retrieval: 20.7%) |

## Skills demonstrated

| Skill | Evidence |
|-------|----------|
| **Computer vision** | HOG descriptors + Gabor filter bank + grid statistics pipeline |
| **Multi-class classification** | 6-class balanced problem, macro metrics (not accuracy-only) |
| **Error cost analysis** | Per-class FN costs ($100–$500) used to prioritise review routing |
| **Pareto analysis** | Identifies top 2 error classes driving 64% of all misclassifications |
| **Review queue design** | Entropy-ranked routing captures 82% of errors at 20% inspection budget |
| **Business framing** | Model output → quality decision routing, not just label prediction |
| **Python stack** | scikit-image, scikit-learn, matplotlib, Flask, SQLite |
| **Embedding retrieval** | 4-way retriever benchmark (TF-IDF, MiniLM, BGE-small, BGE-base); winner served from Chroma |
| **LLM evaluation** | Sentence-level NLI faithfulness with a no-retrieval control; generator and prompt chosen by the metric |
| **Honest visualisation** | 3D embedding map whose highlights come from the real search, with the projection's distortion quantified on the page |
| **Production engineering** | Postgres persistence with versioned migrations (tested on SQLite and a real Postgres in CI), per-visitor quotas, retention and rate limits on public write endpoints, token-protected export |
| **Model calibration and uncertainty** | Temperature scaling, ECE, bootstrap confidence intervals, identical-holdout comparison enforced by a test; a calibrated CNN pipeline verified end to end on synthetic data, plus a fine-tuned ResNet-18 baseline that scores 1.000 on real NEU-CLS (random and blocked splits), read as a saturated benchmark |
| **Serving** | ONNX export with parity check (2e-6), `/api/classify` on ONNX Runtime, Docker + Postgres compose stack |
| **Performance** | Found and fixed a per-poll cost that grew with uptime (~330 ms → ~0.4 ms), with brute-force equivalence tests |
| **CI** | GitHub Actions: ruff, mypy, 80% coverage floor, Postgres service, CPU-torch pipeline smoke test, Playwright e2e; Dependabot with a guard against a breaking OpenCV major |

## Interview-ready talking points

1. Accuracy alone is misleading for quality inspection — a model that is always right on easy classes but consistently misses high-cost inclusions is worse than its accuracy score suggests. Per-class F1 and cost-weighted evaluation are the right metrics here.
2. The Pareto chart shows two classes (scratches, pitted_surface) account for 64% of errors — this directly informs where to focus model improvement or operator training.
3. Entropy-based review queue design: instead of routing all 360 holdout samples to human review, reviewing only the top 20% highest-entropy samples catches 82% of all errors. This is the business value of calibrated uncertainty.
4. The confused pairs (scratches → inclusion, inclusion → pitted_surface) have a visual explanation — both pairs involve elongated or textural surface patterns that share low-frequency HOG features. A Gabor bank adds some separation, but deep features would likely resolve this.
5. Retrieval needs a baseline. On questions that reuse the source passage's wording, TF-IDF (MRR@10 0.773) beats a small embedding model (MiniLM, 0.720); BGE-base wins (0.833) and is what the vector database serves.
6. Measure prompts per model. A strict "no background knowledge" prompt lowered the 1.5B generator's grounded-sentence rate (52.7% → 48.9%) but raised the 3B generator's (61.7% → 75.7%). Four configurations were scored and the best one ships; the page also shows the claim check sentence by sentence, including unsupported ones.
