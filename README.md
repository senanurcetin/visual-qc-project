# Visual QC Project

[![CI](https://github.com/senanurcetin/visual-qc-project/actions/workflows/ci.yml/badge.svg)](https://github.com/senanurcetin/visual-qc-project/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Computer vision analytics case study** — steel surface defect classification on the NEU-CLS dataset, combined with an operator-facing Flask QA dashboard. Covers EDA, feature engineering (HOG), model benchmarking, per-class metrics, Pareto error analysis, and confidence-based review queue design.

**AI engineering layer** — an embedding-based RAG assistant over a steel quality-engineering knowledge base, with a retriever benchmark, a Chroma vector index, NLI-scored answer faithfulness, and a 3D map of the embedding space that shows which passages each answer came from ([details](#ai-engineering-layer--steel-qc-knowledge-assistant-rag)).

**Live demo:** [visual-qc-project-pearl.vercel.app](https://visual-qc-project-pearl.vercel.app) — HMI with 3D line twin · [RAG map](https://visual-qc-project-pearl.vercel.app/rag) · [CV case study](https://visual-qc-project-pearl.vercel.app/case-study)

Portfolio entry: [senanur-cetin.vercel.app/projects/visual-qc-project](https://senanur-cetin.vercel.app/projects/visual-qc-project)

Short video: [`docs/assets/visual-qc-dashboard.webm`](docs/assets/visual-qc-dashboard.webm)

![Visual QC dashboard](docs/assets/visual-qc-dashboard.png)

---

## Problem

Steel surface inspection needs more than a classifier. A useful system must route uncertain cases, preserve a production log, and export reviewable evidence for quality teams. The model output is a decision surface — not just a label.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [NEU-CLS Steel Surface Defect Dataset](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) |
| Images | 1,800 grayscale images |
| Classes | 6 defect types, perfectly balanced (300 per class) |
| Evaluation | Deterministic 80/20 stratified holdout (seed=42) |
| Feature pipeline | 64×64 HOG descriptors + Gabor filter bank + grid statistics |

### Defect Classes

| Class | Description |
|-------|-------------|
| Crazing | Micro-crack patterns — surface fatigue and coating stress |
| Inclusion | Foreign material trapped in the strip surface |
| Patches | Irregular patches — often require operator review before shipment |
| Pitted surface | Localized pitting — corrosion risk |
| Rolled-in scale | Scale rolled into strip — persistent texture distortion |
| Scratches | Linear scoring defects — commonly drive rework or rejection |

---

## Model Benchmark

![Model comparison](docs/assets/model-comparison.png)

| Model | Accuracy | Macro F1 | Macro Precision |
|-------|----------|----------|-----------------|
| Dummy baseline | 0.167 | 0.048 | 0.028 |
| Logistic regression | 0.869 | 0.868 | 0.872 |
| **Random Forest (selected)** | **0.939** | **0.939** | **0.942** |

Random forest on HOG + Gabor + grid features selected for highest accuracy and balanced per-class performance.

---

## Per-Class Metrics

![Class metrics](docs/assets/class-metrics.png)

| Class | Precision | Recall | F1 | FN Cost |
|-------|-----------|--------|----|---------|
| Crazing | 0.967 | 0.983 | 0.975 | $100 |
| Inclusion | 0.836 | 0.933 | 0.882 | $500 ← highest cost |
| Patches | 0.966 | 0.950 | 0.958 | $200 |
| Pitted surface | 0.900 | 0.900 | 0.900 | $350 |
| Rolled-in scale | 0.984 | 1.000 | 0.992 | $400 |
| Scratches | 1.000 | 0.867 | 0.929 | $150 |

Inclusion has the lowest precision/recall and the highest false-negative cost — the primary target for the review queue.

---

## Confusion Matrix

![Confusion matrix](docs/assets/confusion-matrix.png)

22 total errors out of 360 holdout samples (6.1% error rate). Main confusion hotspot: **scratches → inclusion** (8 cases). Second: **inclusion → pitted_surface** (4 cases).

---

## Pareto Error Analysis

![Pareto errors](docs/assets/pareto-errors.png)

| Class | Errors | Share | Cumulative |
|-------|--------|-------|-----------|
| Scratches | 8 | 36.4% | 36.4% |
| Pitted surface | 6 | 27.3% | 63.6% |
| Inclusion | 4 | 18.2% | 81.8% |
| Patches | 3 | 13.6% | 95.5% |
| Crazing | 1 | 4.5% | 100% |

**Two classes (scratches + pitted_surface) account for 64% of all errors** — focusing review effort on these two classes addresses most of the classifier's misclassifications.

---

## Review Queue — Confidence-Based Routing

![Review queue curve](docs/assets/review-queue-curve.png)

Samples ranked by prediction entropy (high entropy = low confidence = route to human review):

| Budget | Samples reviewed | Errors captured | Yield lift vs random |
|--------|-----------------|-----------------|---------------------|
| 10% (36 samples) | 36 | 45.5% | **4.5×** |
| 15% (54 samples) | 54 | 50.0% | 3.3× |
| 20% (72 samples) | 72 | 81.8% | **4.1×** |

Reviewing the top 20% highest-entropy samples captures 82% of the classifier's misclassifications (18 of 22) at 4.1× the yield of random review.

---

## AI Engineering Layer — Steel QC Knowledge Assistant (RAG)

![Embedding map](docs/assets/rag-embedding-map.png)

A retrieval-augmented assistant for process questions a quality team asks (*"When is a process out of statistical control?"*, *"How is mill scale removed before painting?"*). Everything runs locally; no paid API and no key in the page. Full method: [`docs/rag-case-study.md`](docs/rag-case-study.md). Live page: `/rag`.

| Stage | Implementation |
|-------|----------------|
| Knowledge base | 43 Wikipedia articles (CC BY-SA, pinned revisions) in 6 topics → 728 section-aware passages |
| Embeddings | `BAAI/bge-base-en-v1.5` (768-d), chosen from a 4-way benchmark |
| Vector DB | Chroma (HNSW, cosine) — top-4 identical to exact search on 100% of queries |
| Generation | Local Qwen2.5 models; generator and prompt chosen by measured faithfulness |
| Faithfulness | NLI cross-encoder checks every answer sentence against the retrieved passages |
| Visual | UMAP 3D map (Three.js, lazy-loaded) with a 2D fallback for phones and reduced motion |

### Retrieval benchmark

![Retrieval benchmark](docs/assets/rag-retrieval-benchmark.png)

77 generated questions with a known source passage; the exact passage must come back.

| Retriever | Hit@1 | Hit@4 | MRR@10 |
|-----------|-------|-------|--------|
| TF-IDF keyword baseline | 0.688 | 0.896 | 0.773 |
| MiniLM-L6 (384-d) | 0.584 | 0.844 | 0.720 |
| BGE-small (384-d) | 0.740 | 0.922 | 0.828 |
| **BGE-base (768-d), selected** | 0.727 | **0.948** | **0.833** |

The keyword baseline beats the small MiniLM model: generated questions reuse the passage's terms. Without a baseline that would have gone unnoticed.

### Faithfulness

![Faithfulness](docs/assets/rag-faithfulness.png)

An NLI cross-encoder checks each answer sentence against 1–3 sentence windows of the four retrieved passages (99 questions). The control is the same model answering without retrieval.

| Generator | No retrieval | RAG, basic prompt | RAG, strict prompt |
|-----------|-------------|-------------------|--------------------|
| Qwen2.5-1.5B (fp16) | 14.2% | 52.7% | 48.9% |
| Qwen2.5-3B (4-bit NF4) | 20.7% | 61.7% | **75.7%** (used on `/rag`) |

*Share of answer sentences entailed by the retrieved passages.* The strict "no background knowledge" prompt hurts the 1.5B model and helps the 3B model, so the configuration on the page was picked by the metric, not by eye. The selected setup declines 2% of questions instead of answering beyond its sources.

### Honest 3D

The glowing passages on the map always come from the full 768-dimensional Chroma search. The projection keeps passage neighbourhoods (UMAP trustworthiness 0.99 vs 0.91 for PCA) but not question-to-passage distances: on average only 1.6 of the 4 points nearest a question in 3D are real results. The page prints that overlap under the map for every example question.

---

## Dashboard — 3D Line Twin

The operator dashboard renders the simulated inspection line in 3D (Three.js): steel plates leave a feed hood, pass under a QC gantry with a camera and laser scan sheet, and are either stacked on an OK pallet or pushed into a reject bin by a pneumatic pusher. Rejected plates carry a procedural texture for one of the six NEU-CLS defect classes, and the event historian logs that class.

- The line is a pure function of a small control state kept in each visitor's signed session cookie (`line_sim.py`): unit outcomes come from a hash of (seed, cycle), so any server instance returns the same counters, historian and verdicts, and one visitor's Emergency Stop never affects another. The 3D scene syncs its clock through `/api/data` (`sim_time`, `status_cycle`, `current_defect`).
- A stack-light tower mirrors the machine state (green running, amber paused, flashing red e-stop); orbit the camera to inspect the line.
- If WebGL or the CDN is unavailable, the page falls back to the original OpenCV 2D camera feed.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Feature engineering | Python, NumPy, scikit-image (HOG, Gabor) |
| Modeling | scikit-learn (Random Forest, Logistic Regression) |
| Dashboard | Flask, deterministic per-visitor line simulation (signed session cookie), Excel export, Three.js 3D line twin (OpenCV 2D feed as fallback) |
| Visualization | Matplotlib, Three.js (lazy-loaded), Canvas 2D |
| Retrieval & RAG | sentence-transformers (BGE), Chroma, UMAP, Qwen2.5 via transformers, NLI cross-encoder |
| CI | GitHub Actions |

---

## Architecture

```
NEU-CLS Dataset (1,800 images)
        |
        v
Feature Extraction (HOG + Gabor + grid stats)
        |
        v
Model Benchmark (dummy / logistic / random forest)
        |
        v
Review Queue Design (entropy-ranked routing)
        |
        v
Flask Dashboard (3D line twin + operator UI + historian + export)

Wikipedia knowledge base (43 articles, 728 passages)
        |
        v
BGE-base embeddings (768-d) -> Chroma vector DB
        |
        v
Top-4 retrieval -> local LLM answer -> NLI faithfulness check
        |
        v
/rag page (UMAP 3D/2D map + precomputed example questions)
```

---

## Local Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements-dev.txt   # app + analysis/chart dependencies

# Run full analysis benchmark (downloads NEU-CLS dataset on first run)
python analysis/run_neu_case_study.py

# Optional: rebuild the RAG layer (GPU recommended; downloads models on first run)
pip install -r requirements-rag.txt
python analysis/run_rag_case_study.py

# Start the dashboard
python main.py
```

App: `http://127.0.0.1:8080` | Case-study route: `/case-study` | RAG map: `/rag`

---

## Tests

```bash
python -m unittest discover -s tests -v
python -m py_compile main.py case_study.py rag_demo.py analysis/run_neu_case_study.py analysis/run_rag_case_study.py
```

---

## Proof Surfaces

| Document | Contents |
|----------|----------|
| [`docs/case-study.md`](docs/case-study.md) | Full methodology, results, limitations |
| [`docs/hiring-summary.md`](docs/hiring-summary.md) | Recruiter-facing one-page summary |
| [`docs/data/neu-cls-case-study/`](docs/data/neu-cls-case-study/) | JSON artifacts — metrics, confusion matrix, Pareto, review queue |
| [`docs/rag-case-study.md`](docs/rag-case-study.md) | RAG layer: method, retrieval and faithfulness results, projection honesty, limitations |
| [`docs/data/rag-knowledge-assistant/`](docs/data/rag-knowledge-assistant/) | JSON artifacts — corpus manifest, eval set, benchmarks, sample answers, embedding map |

---

## What This Proves

- Multi-class image classification with imbalanced error costs — recall is not enough, per-class cost weighting matters
- Pareto analysis identifies which defect classes to prioritise in inspection budget
- Entropy-based review queue captures 82% of errors by reviewing only 20% of samples
- Business framing: connects model output to quality decision routing, not just label prediction
- Embedding retrieval chosen by benchmark against a keyword baseline, served from a vector database
- LLM output measured, not eyeballed: sentence-level faithfulness with a no-retrieval control, and the model and prompt picked by that metric

---

## Limitations

- HOG + Gabor descriptors are handcrafted — deep learning (ResNet, EfficientNet) would likely improve accuracy further
- NEU-CLS is a research benchmark — results are not directly transferable to a live production line
- The Flask dashboard runs a simulated line (rendered as a 3D digital twin), not a real camera stream; defect classes on rejected plates are drawn procedurally, not taken from NEU-CLS images
- The historian and Excel export are derived from the visitor's simulated line, not persisted to a database
- The RAG knowledge base is encyclopaedic (Wikipedia), not plant SOPs; generated eval questions make retrieval easier than real queries, and NLI faithfulness is not answer correctness

---

## License

MIT
