<p align="center">
  <img src="docs/assets/nexus-official-midnight.svg" alt="NEXUS — personal brand identity" width="660">
</p>

# Visual QC Project

**SEO and social cards:** [Documented canonical URLs, screenshot sources and preview assets](docs/SEO_AND_SOCIAL_PREVIEW_GUIDE.md).

![Visual QC production HMI: simulated line running, digital twin, quality KPIs and defect event (real browser capture)](docs/assets/visual-qc-dashboard.png)

> **Real production screenshot (14 October 2026):** recorded during a successful Chromium E2E test of the Vercel operator HMI. It shows a **simulated** production line and observed simulated defect, not actual factory production. The independent measured CV model charts below remain unchanged. [Capture provenance](docs/assets/production_screenshot_provenance_20261014.json).

**Public replacement notebook:** [NEU-CLS robustness reproduction](notebooks/neu_cls_robustness_reproduction.ipynb). This is a *new* classical-model experiment; the formerly linked 100% CNN Kaggle notebook is **retired as an unavailable historical reference**, not restored or independently rerun. The actual image perturbation evidence and source hashes are documented below.

**Reproduced image robustness (15 October 2026):** [NEU-CLS perturbation study](docs/NEU_CLS_ROBUSTNESS_REPRODUCTION_2026-10-15.md) — same 360-image holdout; clean Random Forest accuracy 93.89%, dropping to 77.50% under contrast reduction. New source-backed benchmark; not the unrecovered Kaggle CNN notebook.

**Quality checks:** [CircleCI — Python/PostgreSQL validation](https://circleci.com/gh/senanurcetin/visual-qc-project/10) · GitHub Actions are not the current automatic CI provider.
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Computer vision analytics case study** — steel surface defect classification on the NEU-CLS dataset, combined with an operator-facing Flask QA dashboard. Covers EDA, feature engineering (HOG), model benchmarking, per-class metrics, Pareto error analysis, and confidence-based review queue design.

**AI engineering layer** — an embedding-based RAG assistant over a steel quality-engineering knowledge base, with a retriever benchmark, a Chroma vector index, NLI-scored answer faithfulness, and a 3D map of the embedding space that shows which passages each answer came from ([details](#ai-engineering-layer--steel-qc-knowledge-assistant-rag)).

**Live demo:** [visual-qc-project-pearl.vercel.app](https://visual-qc-project-pearl.vercel.app) — HMI with 3D line twin · [RAG map](https://visual-qc-project-pearl.vercel.app/rag) · [CV case study](https://visual-qc-project-pearl.vercel.app/case-study)

Portfolio entry: [senanur-cetin.vercel.app/projects/visual-qc-project](https://senanur-cetin.vercel.app/projects/visual-qc-project)

Short video: [`docs/assets/visual-qc-dashboard.webm`](docs/assets/visual-qc-dashboard.webm)

![Visual QC production operator dashboard on 375px mobile viewport](docs/assets/visual-qc-mobile-375.png)

---

## Functional XLSX report — source-backed, not a mockup

The Flask HMI serves **[Download session report (XLSX)](https://visual-qc-project-pearl.vercel.app/api/export_report)** after the visitor completes at least one simulated production unit. The real report generator is [`report.py`](report.py); [automated workbook checks](tests/test_report_workbook.py) inspect its formulas, chart source and embedded provenance statement.

- **Dashboard sheet:** total units, accepted/rejected counts, yield and mean OEE, each calculated by a genuine Excel formula linked to the detailed log. Cached results are included for static viewers.
- **Production Log:** timestamp, simulated unit identifier, status, defect and per-unit OEE, with filters and conditional formatting.
- **Defect Analysis:** six simulator defect classes, source-linked `COUNTIFS` counts, percentages and a real bar chart; invalid rejection labels fail export instead of producing a misleading category total.
- **Chart:** accepted versus rejected distribution, with stable NEXUS blue/lavender series labels even when failures outnumber passes.
- **Evidence scope:** the yellow header states **SIMULATED DEMO / NOT REAL PLANT DATA**. These measurements describe a visitor's deterministic factory-line simulation, **not** an actual plant or independently benchmarked NEU-CLS classifier performance.
- **Empty session:** export returns 404 until simulated units are completed; it does not invent historical plant records.

**Verified release (14 October 2026):** CircleCI Python/PostgreSQL checks passed on the release PR, the Vercel production deployment was confirmed, and the HMI passed six production Chromium E2E tests. README screenshots are genuine production browser captures; the WebM is historical. Workbook formulas, data provenance and empty-session behavior are validated by dedicated tests—not by screenshots.

---

## Problem

Steel surface inspection needs more than a classifier. A useful system must route uncertain cases, preserve a production log, and export reviewable evidence for quality teams. The model output is a decision surface — not just a label.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [NEU-CLS Steel Surface Defect Dataset — official Northeastern University page](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/) |
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

### Deep-learning comparison (October 2026)

| Model | Split | Accuracy | Macro F1 |
|---|---|---|---|
| Random Forest on HOG + Gabor + grid (above) | random 80/20, seed 42 | 0.939 | 0.939 |
| ResNet-18, ImageNet-pretrained, fine-tuned ([cnn/train_cnn.py](cnn/train_cnn.py)) | random 80/20, seed 42 | **1.000** (360/360) | 1.000 |
| same ResNet-18 | blocked: images 241-300 of each class held out | **1.000** (360/360) | 1.000 |

The blocked split guards against near-duplicate neighbouring crops of the same strip; the score did not
move. A perfect score here means NEU-CLS is saturated as a benchmark, not that the model is ready for a
production line, so the confidence-based review queue below still applies. The model is on Hugging Face as
[senanurcetin/neu-steel-defect-resnet18](https://huggingface.co/senanurcetin/neu-steel-defect-resnet18)
(ONNX) with an [in-browser demo](https://huggingface.co/spaces/senanurcetin/neu-steel-defect-demo).

**Where the 100% ends.** In a separately reported robustness exercise, a ResNet-18 trained the same way was tested on the same 360 images, each degraded at test time only. The previously linked Kaggle notebook is **not currently publicly accessible (HTTP 404, checked 14 October 2026)**, so the detailed robustness figures below are historical reported results rather than newly verified public notebook output:

| test-time condition | accuracy (two GPU runs) |
|---|---|
| clean | 1.000 |
| darker lighting (x0.5) | 1.000 |
| low contrast (x0.5) | 0.947–0.967 |
| brighter lighting (x1.6) | 0.822–0.828 |
| sensor noise, sd 0.05 | 0.692–0.808 |
| JPEG quality 15 | 0.678–0.767 |
| half resolution | 0.642–0.661 |
| slight blur (radius 1.5) | 0.556–0.631 |
| quarter resolution | 0.514–0.594 |
| sensor noise, sd 0.12 | 0.344–0.386 |
| strong blur (radius 3) | 0.356–0.375 |

GPU training is not bit-for-bit reproducible, so the figures move by a few points between runs; the
large losses repeat.

Focus, resolution and sensor noise are the conditions to control on the line (or to add to training as
augmentation); the clean-image score does not show them.

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

![Actual production Visual QC RAG interface (Chromium screenshot)](docs/assets/visual-qc-rag.png)

*Captured from the deployed /rag interface; retrieval and model quality are evaluated separately below.*

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

## Operator Workflow, SPC and Persistence

- **Review queue** (HMI panel): every rejected unit can be *confirmed* or *corrected* to the right defect class. Decisions are stored per visitor in Postgres when `DATABASE_URL` is set (Neon in the hosted demo, Postgres in Docker), otherwise in a local SQLite file. Schema changes are versioned migrations (`store.py`, table `schema_migrations`).
- **Background reading**: the `i` button on a rejected unit shows the most relevant passages from the knowledge base for its defect class (`/api/defect-knowledge`, a small TF-IDF over the published passage previews, so it needs no model at request time). It is encyclopaedic Wikipedia background, not plant procedures; the *Crazing* article, for instance, is about polymers.
- **Abuse limits**: decisions are capped per visitor and per table, expire after a retention period, and the write endpoints are rate limited per client (429 with `Retry-After`). Limits are tunable through `REVIEW_*` / `CLASSIFY_RATE_PER_MIN` (see `.env.example`); the rate limit is per server instance, the row quotas are the hard bound.
- **Corrections export**: `/api/review/export.csv` returns the visitor's corrected labels. Quality engineers get every operator's corrections from `/api/admin/corrections.csv`, which is disabled unless `REVIEW_ADMIN_TOKEN` is set and then needs `Authorization: Bearer <token>`. These are the retraining candidates.
- **SPC** (`/spc`): p-chart of the reject rate per batch of 10 units with 3-sigma control limits; out-of-control batches are flagged.
- **Classify an image** (`POST /api/classify`, multipart field `image`): serves the ONNX model exported by `analysis/run_dl_case_study.py --export-onnx` and returns label, calibrated confidence, entropy, a `needs_review` flag and a ranking. Returns 503 with a reason when no model is installed (as on the hosted demo).
- **Frame sources** (`camera.py`): the 2D feed reads from a `FrameSource` — the simulator, or a looped video with `FRAME_SOURCE=video:/path/clip.mp4`. `python analysis/bench_frames.py` reports p50/p95 latency and fps (simulated source: ~6 ms p50, ~7.7 ms p95, ~165 fps on the dev container).

---

## CNN Baseline

`analysis/run_dl_case_study.py` fine-tunes ResNet18 / EfficientNet-B0 on the **same 360-row holdout** as the Random Forest (a test pins the identity), fits a calibration temperature on a validation slice, and reports accuracy / macro-F1 with bootstrap 95% intervals, ECE before and after calibration, the calibrated review-queue budgets, optional 5-fold CV, Grad-CAM overlays and the ONNX export. `--smoke-test` runs the whole pipeline on synthetic textures in seconds without the dataset (it runs in CI); a real run needs the NEU-CLS download and, preferably, a GPU. See [`analysis/README.md`](analysis/README.md).

<!-- DL-RESULTS:START -->
**No real-data CNN run has been recorded yet** for this calibrated pipeline. The separate fine-tuned ResNet-18 baseline ([`cnn/train_cnn.py`](cnn/train_cnn.py)) has been run on NEU-CLS: see [Deep-learning comparison](#deep-learning-comparison-october-2026) above. After a real run of this pipeline, `python analysis/update_docs_from_dl.py` fills this block from `docs/data/neu-cls-dl/summary.json`, and the docs-consistency test keeps it in sync.
<!-- DL-RESULTS:END -->

---

## Docker

```bash
cp .env.example .env        # set SECRET_KEY and POSTGRES_PASSWORD
docker compose up --build   # app on :8080 + Postgres; mount ./models to enable /api/classify
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Feature engineering | Python, NumPy, scikit-image (HOG, Gabor) |
| Modeling | scikit-learn (Random Forest, Logistic Regression) |
| Dashboard | Flask, deterministic per-visitor line simulation (signed session cookie), Excel export, Three.js 3D line twin (OpenCV 2D feed as fallback) |
| Visualization | Matplotlib, Three.js (lazy-loaded), Canvas 2D |
| Retrieval & RAG | sentence-transformers (BGE), Chroma, UMAP, Qwen2.5 via transformers, NLI cross-encoder |
| Persistence | Postgres (Neon / Docker) or SQLite, psycopg 3, versioned migrations |
| Serving | ONNX Runtime (`/api/classify`), gunicorn in Docker |
| CI | CircleCI `python-postgres-quality`: Python 3.12, Ruff, mypy, unittest + PostgreSQL 16 and coverage; Chromium HMI E2E is additionally verified locally and on production. GitHub Actions are manual-only. |

---

## Architecture

```mermaid
flowchart LR
    subgraph Offline["Offline analysis (GPU optional)"]
        D[NEU-CLS<br/>1,800 images] --> F[HOG + Gabor + grid features]
        F --> RF[Random Forest<br/>benchmark]
        D --> CNN[ResNet18 / EfficientNet-B0<br/>calibration, bootstrap CIs, Grad-CAM]
        CNN --> ONNX[(model.onnx + meta.json)]
        W[Wikipedia KB<br/>43 articles, 728 passages] --> E[BGE-base embeddings] --> C[(Chroma)]
        C --> R[Top-4 retrieval + local LLM + NLI check]
    end
    subgraph App["Flask app (Vercel / Docker)"]
        SIM[Deterministic line simulation<br/>signed session cookie] --> HMI[HMI + 3D line twin]
        HMI --> RQ[Review queue<br/>confirm / correct]
        RQ --> DB[(Postgres or SQLite<br/>versioned migrations)]
        HMI --> SPC["/spc p-chart"]
        HMI --> KB[Defect background reading<br/>TF-IDF over KB previews]
        UP["POST /api/classify"] --> ORT[ONNX Runtime]
        DB --> ADM["/api/admin/corrections.csv<br/>retraining candidates"]
    end
    ONNX -. mounted at QC_MODEL_DIR .-> ORT
    R -. precomputed artifacts .-> RAGP["/rag map"]
    RF -. JSON artifacts .-> CS["/case-study"]
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
pip install -r requirements-dev.txt
ruff check . && mypy
coverage run -m unittest discover -s tests && coverage report      # 80% floor on the app modules

# Optional: run the store tests against a real Postgres, and the CNN pipeline tests
TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/db python -m unittest tests.test_review -v
pip install torch torchvision onnx onnxscript -r requirements-serve.txt && python -m unittest tests.test_dl_torch -v
```

---

### CI-free HMI browser acceptance

The opt-in [real-browser smoke suite](tests/e2e/test_smoke.py) exercises actual simulated line commands, the latched emergency stop, the 3D twin, analytics charts, the event historian, Excel export and the 375px mobile layout. It can save **real Chromium screenshots**, not illustrations, from a locally running Flask server.

After starting `python main.py` and installing Playwright/Chromium locally:

```bash
python -m pip install playwright
python -m playwright install chromium
E2E_BASE_URL=http://127.0.0.1:8080 E2E_SCREENSHOT_DIR=artifacts/hmi-browser python -m unittest discover -s tests/e2e -p 'test_smoke.py' -v
```

The optional capture files are `visual-qc-desktop-1440.png` (line running after a simulated rejection) and `visual-qc-mobile-375.png` (reachable twin, charts and historian). This procedure does not enable or trigger hosted CI. Real production Chromium screenshots and a 6/6 production E2E run were recorded on 14 October 2026. The screenshot-update PR was merged and the corresponding Vercel production deployment was verified; subsequent source changes must independently repeat CI and production checks.

---

## Proof Surfaces

| Document | Contents |
|----------|----------|
| [`docs/case-study.md`](docs/case-study.md) | Full methodology, results, limitations |
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
- The Flask dashboard runs a simulated line (rendered as a 3D digital twin) or a looped video, not a live camera; defect classes on rejected plates are drawn procedurally, not taken from NEU-CLS images
- The historian and Excel export are derived from the visitor's simulated line; only operator review decisions are persisted, and the hosted demo needs `DATABASE_URL` set to keep them
- The calibrated CNN pipeline (`analysis/run_dl_case_study.py`) is verified on synthetic data only; the separate fine-tuned ResNet-18 baseline does have a real-data result (1.000 on the random and the blocked split), which reads as a saturated benchmark, not production readiness
- The RAG knowledge base is encyclopaedic (Wikipedia), not plant SOPs; generated eval questions make retrieval easier than real queries, and NLI faithfulness is not answer correctness

---

## License

MIT
