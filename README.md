# Visual QC Project

<img align="right" src="docs/assets/nexus-official-midnight.svg" alt="NEXUS — personal brand identity" width="180">

Steel surface defect classification on the NEU-CLS dataset, an uncertainty-based review queue that routes low-confidence predictions to a human, and a grounded RAG assistant over a steel quality-engineering knowledge base — wrapped in an operator-facing Flask HMI running a **simulated** production line. Built for quality-engineering teams and technical reviewers who want to see how model output turns into reviewable quality decisions.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Quality checks:** [CircleCI — Python/PostgreSQL validation](https://circleci.com/gh/senanurcetin/visual-qc-project/10) · GitHub Actions are not the current automatic CI provider.

## At a glance

### Key results

- **Classifier:** Random Forest on HOG + Gabor + grid features — accuracy 0.939, Macro F1 0.939 on the 360-image NEU-CLS holdout ([benchmark](#model-benchmark)).
- **Review queue:** reviewing the top 20% highest-entropy samples captures 82% of the classifier's misclassifications (18 of 22) at 4.1× the yield of random review ([details](#review-queue--confidence-based-routing)).
- **Robustness:** same 360-image holdout; clean Random Forest accuracy 93.89%, dropping to 77.50% under contrast reduction ([reproduction](docs/NEU_CLS_ROBUSTNESS_REPRODUCTION_2026-10-15.md)).
- **Deep-learning comparison:** fine-tuned ResNet-18 scores 1.000 on the random and the blocked split — read as a saturated benchmark, not production readiness ([comparison](#deep-learning-comparison-october-2026)).

### What it is

**Computer vision analytics case study** — steel surface defect classification on the NEU-CLS dataset, combined with an operator-facing Flask QA dashboard. Covers EDA, feature engineering (HOG), model benchmarking, per-class metrics, Pareto error analysis, and confidence-based review queue design.

**AI engineering layer** — an embedding-based RAG assistant over a steel quality-engineering knowledge base, with a retriever benchmark, a Chroma vector index, NLI-scored answer faithfulness, and a 3D map of the embedding space that shows which passages each answer came from ([details](#ai-engineering-layer--steel-qc-knowledge-assistant-rag)).

### What it is not

- Not a working factory integration: the HMI runs a simulated line (3D digital twin) or a looped video, not a live camera.
- Not production-ready model evidence: NEU-CLS is a research benchmark, and the separate CV, robustness and RAG experiments must not be combined into one score. See [Limitations](#limitations).

## Live demo

**Live demo:** [visual-qc-project-pearl.vercel.app](https://visual-qc-project-pearl.vercel.app) — HMI with 3D line twin · [RAG map](https://visual-qc-project-pearl.vercel.app/rag) · [CV case study](https://visual-qc-project-pearl.vercel.app/case-study)

Portfolio entry: [senanur-cetin.vercel.app/projects/visual-qc-project](https://senanur-cetin.vercel.app/projects/visual-qc-project)

![Visual QC production HMI: simulated line running, digital twin, quality KPIs and defect event (real browser capture)](docs/assets/visual-qc-dashboard.png)

Short video: [`docs/assets/visual-qc-dashboard.webm`](docs/assets/visual-qc-dashboard.webm)

![Visual QC production operator dashboard on 375px mobile viewport](docs/assets/visual-qc-mobile-375.png)

## Quick start

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements-dev.txt   # app + analysis/chart dependencies
python main.py                        # dashboard at http://127.0.0.1:8080
```

Full analysis and RAG rebuild steps: [Local setup](#local-setup) · quality checks: [Tests](#tests).

## Limitations

- **Simulated line.** The HMI runs a simulated line (3D digital twin) or a looped video, not a live camera or factory integration. Screenshots are real browser captures of that simulation. Defect textures on rejected plates are procedural, not NEU-CLS images.
- **Simulated session data.** The historian and XLSX export are derived from the visitor's simulated session and are labelled as such. Only operator review decisions are persisted, and the hosted demo needs `DATABASE_URL` to keep them.
- **Benchmark, not plant data.** NEU-CLS is a research benchmark; results do not transfer directly to a production line. The ResNet-18 score of 1.000 on both splits shows the benchmark is saturated, not that the model is production-ready.
- **Separate experiments.** CV classification, the robustness study and the RAG evaluation measure different things and are not combined into one score.
- **Handcrafted features.** The selected Random Forest uses HOG + Gabor descriptors; the ResNet-18 comparison shows learned features score higher on this benchmark.
- **ResNet-18 robustness table is not reproducible.** It comes from a Kaggle notebook that is no longer public. The reproducible robustness evidence is the classical-model [perturbation study](docs/NEU_CLS_ROBUSTNESS_REPRODUCTION_2026-10-15.md).
- **Calibrated CNN pipeline.** `analysis/run_dl_case_study.py` has only been run on synthetic data (smoke test).
- **RAG scope.** The knowledge base is encyclopaedic (Wikipedia), not plant SOPs. Generated eval questions make retrieval easier than real queries, and NLI faithfulness is not answer correctness.
- **TensorRT numbers.** Single-machine laptop measurements (CPU figures vary with thermals and load). Backend accuracy is measured over all 1,800 images, including training images, so it shows parity between backends, not a held-out estimate. INT8 is not covered.

## Problem

Steel surface inspection needs more than a classifier. A useful system must route uncertain cases, preserve a production log, and export reviewable evidence for quality teams. The model output is a decision surface, not just a label.

## Dataset

[NEU-CLS Steel Surface Defect Dataset](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/) (official Northeastern University page): 1,800 grayscale images, 6 defect types, perfectly balanced (300 per class). Evaluation uses a deterministic 80/20 stratified holdout (seed=42). Features: 64×64 HOG descriptors + Gabor filter bank + grid statistics.

## Model Benchmark

![Model comparison](docs/assets/model-comparison.png)

| Model | Split | Accuracy | Macro F1 | Macro Precision |
|-------|-------|----------|----------|-----------------|
| Dummy baseline | random 80/20, seed 42 | 0.167 | 0.048 | 0.028 |
| Logistic regression | random 80/20, seed 42 | 0.869 | 0.868 | 0.872 |
| **Random Forest (selected)** | random 80/20, seed 42 | **0.939** | **0.939** | **0.942** |
| ResNet-18, ImageNet-pretrained, fine-tuned ([cnn/train_cnn.py](cnn/train_cnn.py)) | random 80/20, seed 42 | **1.000** (360/360) | 1.000 | — |
| same ResNet-18 | blocked: images 241-300 of each class held out | **1.000** (360/360) | 1.000 | — |

Random Forest on HOG + Gabor + grid features was selected among the classical models for highest accuracy and balanced per-class performance.

### Deep-learning comparison (October 2026)

The blocked split guards against near-duplicate neighbouring crops of the same strip; the score did not move. The model is on Hugging Face as [senanurcetin/neu-steel-defect-resnet18](https://huggingface.co/senanurcetin/neu-steel-defect-resnet18) (ONNX) with an [in-browser demo](https://huggingface.co/spaces/senanurcetin/neu-steel-defect-demo).

**Where the 100% ends.** A ResNet-18 trained the same way, tested on the same 360 images degraded at test time only (accuracy range over two GPU runs):

| condition | accuracy | condition | accuracy |
|---|---|---|---|
| clean | 1.000 | half resolution | 0.642–0.661 |
| darker lighting (x0.5) | 1.000 | slight blur (radius 1.5) | 0.556–0.631 |
| low contrast (x0.5) | 0.947–0.967 | quarter resolution | 0.514–0.594 |
| brighter lighting (x1.6) | 0.822–0.828 | sensor noise, sd 0.12 | 0.344–0.386 |
| sensor noise, sd 0.05 | 0.692–0.808 | strong blur (radius 3) | 0.356–0.375 |
| JPEG quality 15 | 0.678–0.767 | | |

GPU training is not bit-for-bit reproducible, so figures move by a few points between runs; the large losses repeat. Focus, resolution and sensor noise are the conditions to control on the line (or to add to training as augmentation); the clean-image score does not show them.

**Classical-model reproduction.** [`notebooks/neu_cls_robustness_reproduction.ipynb`](notebooks/neu_cls_robustness_reproduction.ipynb) runs a perturbation study on the Random Forest with the same holdout (clean 93.89%, 77.50% under contrast reduction); results and source hashes are in the [reproduction report](docs/NEU_CLS_ROBUSTNESS_REPRODUCTION_2026-10-15.md). The notebook is committed without outputs; an offline check verifies its syntax, empty outputs and the documented dataset layout.

## Error Analysis

![Class metrics](docs/assets/class-metrics.png) ![Confusion matrix](docs/assets/confusion-matrix.png) ![Pareto errors](docs/assets/pareto-errors.png)

| Class | Description | Precision | Recall | F1 | FN Cost |
|-------|-------------|-----------|--------|----|---------|
| Crazing | Micro-crack patterns — surface fatigue and coating stress | 0.967 | 0.983 | 0.975 | $100 |
| Inclusion | Foreign material trapped in the strip surface | 0.836 | 0.933 | 0.882 | $500 ← highest |
| Patches | Irregular patches — often require operator review before shipment | 0.966 | 0.950 | 0.958 | $200 |
| Pitted surface | Localized pitting — corrosion risk | 0.900 | 0.900 | 0.900 | $350 |
| Rolled-in scale | Scale rolled into strip — persistent texture distortion | 0.984 | 1.000 | 0.992 | $400 |
| Scratches | Linear scoring defects — commonly drive rework or rejection | 1.000 | 0.867 | 0.929 | $150 |

Inclusion has the lowest precision and the highest false-negative cost, so it is the primary target for the review queue.

22 total errors out of 360 holdout samples (6.1% error rate). Main confusion: **scratches → inclusion** (8 cases), then **inclusion → pitted_surface** (4). By true class (Pareto): scratches 8 (36.4%), pitted surface 6 (27.3%, cumulative 63.6%), inclusion 4 (18.2%, 81.8%), patches 3 (13.6%, 95.5%), crazing 1 (4.5%). Two classes account for 64% of all errors.

### Review queue — confidence-based routing

![Review queue curve](docs/assets/review-queue-curve.png)

Samples are ranked by prediction entropy (high entropy = low confidence = route to human review):

| Budget | Samples reviewed | Errors captured | Yield lift vs random |
|--------|-----------------|-----------------|---------------------|
| 10% | 36 | 45.5% | **4.5×** |
| 15% | 54 | 50.0% | 3.3× |
| 20% | 72 | 81.8% | **4.1×** |

## AI Engineering Layer — Steel QC Knowledge Assistant (RAG)

![Actual production Visual QC RAG interface (Chromium screenshot)](docs/assets/visual-qc-rag.png) ![Embedding map](docs/assets/rag-embedding-map.png)

A retrieval-augmented assistant for process questions a quality team asks (*"When is a process out of statistical control?"*, *"How is mill scale removed before painting?"*). Retrieval, scoring and the local Qwen generators run on one laptop GPU; a hosted Nemotron model on NVIDIA NIM (free API tier) is compared as a third generator. Answers on the page are precomputed, so no key ships with it. Full method: [`docs/rag-case-study.md`](docs/rag-case-study.md). Live page: `/rag`.

| Stage | Implementation |
|-------|----------------|
| Knowledge base | 43 Wikipedia articles (CC BY-SA, revision ids recorded) in 6 topics → 728 section-aware passages |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d), chosen from a 4-way benchmark |
| Vector DB | Chroma (HNSW, cosine) — top-4 identical to exact search on 100% of queries |
| Generation | Local Qwen2.5-1.5B / 3B and hosted `nvidia/nemotron-3.5-bolt-e4b` (NVIDIA NIM); generator and prompt chosen by measured faithfulness |
| Faithfulness | NLI cross-encoder checks every answer sentence against the retrieved passages |
| Visual | UMAP 3D map (Three.js, lazy-loaded) with a 2D fallback for phones and reduced motion |

**Retrieval benchmark** — 76 generated questions with a known source passage; the exact passage must come back.

![Retrieval benchmark](docs/assets/rag-retrieval-benchmark.png)

| Retriever | Hit@1 | Hit@4 | MRR@10 |
|-----------|-------|-------|--------|
| TF-IDF keyword baseline | 0.684 | 0.895 | 0.770 |
| MiniLM-L6 (384-d) | 0.605 | 0.855 | 0.736 |
| **BGE-small (384-d), selected** | **0.763** | 0.934 | **0.843** |
| BGE-base (768-d) | 0.724 | **0.947** | 0.831 |

The keyword baseline beats the small MiniLM model because generated questions reuse the passage's terms; without a baseline that would have gone unnoticed.

**Faithfulness** — an NLI cross-encoder checks each answer sentence against 1–3 sentence windows of the four retrieved passages (98 questions). The control is the same model answering without retrieval. Values are the share of answer sentences entailed by the retrieved passages.

![Faithfulness](docs/assets/rag-faithfulness.png)

| Generator | No retrieval | RAG, basic prompt | RAG, strict prompt |
|-----------|-------------|-------------------|--------------------|
| Qwen2.5-1.5B (fp16, local) | 13.0% | 45.1% | 48.8% |
| Qwen2.5-3B (4-bit NF4, local) | 20.3% | 60.2% | 77.3% |
| Nemotron 3.5 Bolt E4B (NVIDIA NIM, hosted) | 19.1% | 93.2% | **96.4%** (used on `/rag`) |

Without retrieval all three models land near 13–20%, so the gap is grounding, not model knowledge: with the same four passages Nemotron keeps 96.4% of its sentences inside the sources against 77.3% for the best local model, and 92.1% of its answers are fully supported (Qwen2.5-3B: 71.1%). It also declines more often (9.2% vs 1.0%) instead of answering beyond its sources. The configuration on the page was picked by the metric.

**3D projection** — highlighted passages on the map always come from the full 384-dimensional Chroma search. The projection keeps passage neighbourhoods (UMAP trustworthiness 0.99 vs 0.89 for PCA) but not question-to-passage distances: on average only 1.6 of the 4 points nearest a question in 3D are real results. The page prints that overlap under the map for every example question.

## Dashboard and Operator Workflow

The operator dashboard renders the simulated inspection line in 3D (Three.js): steel plates leave a feed hood, pass under a QC gantry with a camera and laser scan sheet, and are either stacked on an OK pallet or pushed into a reject bin by a pneumatic pusher. Rejected plates carry a procedural texture for one of the six NEU-CLS defect classes, and the event historian logs that class.

- **Deterministic simulation**: the line is a pure function of a small control state in each visitor's signed session cookie (`line_sim.py`). Unit outcomes come from a hash of (seed, cycle), so any server instance returns the same counters, historian and verdicts, and one visitor's Emergency Stop never affects another. The 3D scene syncs its clock through `/api/data` (`sim_time`, `status_cycle`, `current_defect`). A stack-light tower mirrors machine state (green running, amber paused, flashing red e-stop). Without WebGL or the CDN, the page falls back to the OpenCV 2D camera feed.
- **Review queue**: every rejected unit can be *confirmed* or *corrected* to the right defect class. Decisions are stored per visitor in Postgres when `DATABASE_URL` is set (Neon in the hosted demo, Postgres in Docker), otherwise in a local SQLite file. Schema changes are versioned migrations (`store.py`, table `schema_migrations`).
- **Background reading**: the `i` button on a rejected unit shows the most relevant knowledge-base passages for its defect class (`/api/defect-knowledge`, a small TF-IDF over the published passage previews, so no model is needed at request time). This is Wikipedia background, not plant procedures; the *Crazing* article, for instance, is about polymers.
- **Abuse limits**: decisions are capped per visitor and per table and expire after a retention period; write endpoints are rate limited per client (429 with `Retry-After`). Tunable through `REVIEW_*` / `CLASSIFY_RATE_PER_MIN` (see `.env.example`); the rate limit is per server instance, the row quotas are the hard bound.
- **Corrections export**: `/api/review/export.csv` returns the visitor's corrected labels. `/api/admin/corrections.csv` returns every operator's corrections (the retraining candidates); it is disabled unless `REVIEW_ADMIN_TOKEN` is set and then needs `Authorization: Bearer <token>`.
- **SPC** (`/spc`): p-chart of the reject rate per batch of 10 units with 3-sigma control limits; out-of-control batches are flagged.
- **Classify an image** (`POST /api/classify`, multipart field `image`): serves the ONNX model exported by `analysis/run_dl_case_study.py --export-onnx` and returns label, calibrated confidence, entropy, a `needs_review` flag and a ranking. Returns 503 with a reason when no model is installed (as on the hosted demo).
- **Frame sources** (`camera.py`): the 2D feed reads from a `FrameSource` — the simulator, or a looped video with `FRAME_SOURCE=video:/path/clip.mp4`. `python analysis/bench_frames.py` reports p50/p95 latency and fps (simulated source: ~6 ms p50, ~7.7 ms p95, ~165 fps on the dev container).
- **XLSX session report**: *Export Report* (`/api/export_report`, generator [`report.py`](report.py)) returns 404 until the visitor completes a simulated unit, then a workbook with a Dashboard sheet (units, accepted/rejected, yield and mean OEE as real Excel formulas over the log, with cached values for static viewers), a filtered and conditionally formatted Production Log, a Defect Analysis sheet (six classes, source-linked `COUNTIFS`, percentages and a bar chart; invalid rejection labels fail the export), an accepted-vs-rejected chart with stable NEXUS blue/lavender series, and a **SIMULATED DEMO / NOT REAL PLANT DATA** header. [Workbook tests](tests/test_report_workbook.py) check formulas, chart source, provenance statement and empty-session behaviour.

## CNN Baseline

`analysis/run_dl_case_study.py` fine-tunes ResNet18 / EfficientNet-B0 on the **same 360-row holdout** as the Random Forest (a test pins the identity), fits a calibration temperature on a validation slice, and reports accuracy / macro-F1 with bootstrap 95% intervals, ECE before and after calibration, the calibrated review-queue budgets, optional 5-fold CV, Grad-CAM overlays and the ONNX export. `--smoke-test` runs the whole pipeline on synthetic textures in seconds without the dataset (it runs in CI); a real run needs the NEU-CLS download and, preferably, a GPU. See [`analysis/README.md`](analysis/README.md).

<!-- DL-RESULTS:START -->
**No real-data CNN run has been recorded yet** for this calibrated pipeline. The separate fine-tuned ResNet-18 baseline ([`cnn/train_cnn.py`](cnn/train_cnn.py)) has been run on NEU-CLS: see [Deep-learning comparison](#deep-learning-comparison-october-2026) above. After a real run of this pipeline, `python analysis/update_docs_from_dl.py` fills this block from `docs/data/neu-cls-dl/summary.json`, and the docs-consistency test keeps it in sync.
<!-- DL-RESULTS:END -->

### Inference acceleration with TensorRT

The released ResNet-18 ONNX model ([Hugging Face](https://huggingface.co/senanurcetin/neu-steel-defect-resnet18)) runs unchanged on four ONNX Runtime backends. [`cnn/benchmark_inference.py`](cnn/benchmark_inference.py) measures each backend and compares its predictions with the CPU FP32 reference on every image. Raw output: [`docs/data/tensorrt-benchmark/benchmark.json`](docs/data/tensorrt-benchmark/benchmark.json).

| Backend | Latency, batch 1 (p50 / p95) | Throughput, batch 32 | Same top-1 as CPU | Max \|logit diff\| |
| --- | --- | --- | --- | --- |
| CPU FP32 | 47.779 / 77.988 ms | 34.1 img/s | reference | 0 |
| CUDA FP32 | 7.704 / 11.885 ms | 739.8 img/s | 100% | 0.0033 |
| TensorRT FP32 | 6.19 / 8.866 ms | 876.5 img/s | 100% | 0.00281 |
| **TensorRT FP16** | **3.286 / 5.021 ms** | **1684.1 img/s** | **100%** | 0.01657 |

TensorRT FP16 is about 14× faster than CPU at batch 1 and about 2.3× faster than CUDA FP32 at batch 32. All four backends agree on the predicted class for all 1800 images. Building the TensorRT engine takes about a minute the first time (`session_init_s` in the JSON); the engine is then cached. Setup: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), Windows 11, Python 3.11, ONNX Runtime 1.31.1, TensorRT 10.16, all 1800 NEU-CLS images, 300 timed runs after warm-up.

```bash
pip install "onnxruntime-gpu[cuda,cudnn]" "tensorrt-cu13>=10,<11" numpy pillow
python cnn/benchmark_inference.py --onnx neu_resnet18.onnx --data <NEU-CLS root> --out results/trt
```

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

| Layer | Technology |
|-------|-----------|
| Features / modeling | Python, NumPy, scikit-image (HOG, Gabor), scikit-learn (Random Forest, Logistic Regression) |
| Dashboard | Flask, deterministic per-visitor line simulation (signed session cookie), Excel export, Three.js 3D line twin (OpenCV 2D fallback), Matplotlib, Canvas 2D |
| Retrieval & RAG | sentence-transformers (BGE), Chroma, UMAP, Qwen2.5 via transformers, NLI cross-encoder |
| Persistence / serving | Postgres (Neon / Docker) or SQLite, psycopg 3, versioned migrations; ONNX Runtime (`/api/classify`), gunicorn in Docker |
| CI | CircleCI `python-postgres-quality`: Python 3.12, Ruff, mypy, unittest + PostgreSQL 16 and coverage. The GitHub Actions workflow is manual-only. |

## Local Setup

After the [Quick start](#quick-start) install:

```bash
python analysis/run_neu_case_study.py     # full analysis benchmark (downloads NEU-CLS on first run)

pip install -r requirements-rag.txt       # optional: rebuild the RAG layer (GPU recommended)
python analysis/run_rag_case_study.py

python main.py                            # app :8080 · /case-study · /rag

cp .env.example .env                      # Docker: set SECRET_KEY and POSTGRES_PASSWORD
docker compose up --build                 # app on :8080 + Postgres; mount ./models to enable /api/classify
```

## Tests

```bash
pip install -r requirements-dev.txt
ruff check . && mypy
coverage run -m unittest discover -s tests && coverage report      # 80% floor on the app modules

# Optional: store tests against a real Postgres, and the CNN pipeline tests
TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/db python -m unittest tests.test_review -v
pip install torch torchvision onnx onnxscript -r requirements-serve.txt && python -m unittest tests.test_dl_torch -v
```

**Browser acceptance (opt-in, not in hosted CI).** The [real-browser smoke suite](tests/e2e/test_smoke.py) exercises simulated line commands, the latched emergency stop, the 3D twin, analytics charts, the event historian, Excel export and the 375px mobile layout. With `python main.py` running:

```bash
python -m pip install playwright && python -m playwright install chromium
E2E_BASE_URL=http://127.0.0.1:8080 E2E_SCREENSHOT_DIR=artifacts/hmi-browser python -m unittest discover -s tests/e2e -p 'test_smoke.py' -v
```

It writes `visual-qc-desktop-1440.png` (line running after a simulated rejection) and `visual-qc-mobile-375.png` (twin, charts and historian). The README screenshots were captured this way against the Vercel deployment ([capture record](docs/assets/production_screenshot_provenance_20261014.json)); the WebM video is an older recording.

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/case-study.md`](docs/case-study.md) | Full CV methodology, results, limitations |
| [`docs/data/neu-cls-case-study/`](docs/data/neu-cls-case-study/) | JSON artifacts — metrics, confusion matrix, Pareto, review queue |
| [`docs/rag-case-study.md`](docs/rag-case-study.md) | RAG method, retrieval and faithfulness results, projection accuracy, limitations |
| [`docs/data/rag-knowledge-assistant/`](docs/data/rag-knowledge-assistant/) | JSON artifacts — corpus manifest, eval set, benchmarks, sample answers, embedding map |
| [SEO and social preview guide](docs/SEO_AND_SOCIAL_PREVIEW_GUIDE.md) | Canonical URLs, screenshot sources and social preview assets |

## License

MIT
