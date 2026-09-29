# Analysis

This folder upgrades `visual-qc-project` from a simulated QC workflow into a measurable industrial computer-vision case study.

## Dataset

- Dataset: `NEU-CLS`
- DOI: `10.6084/m9.figshare.28903550.v1`
- License: `CC BY 4.0`
- Scope: six steel-surface defect classes with 1,800 grayscale images

## What the pipeline does

`run_neu_case_study.py` downloads the public dataset into `analysis/.cache/`, extracts it, builds HOG descriptors, benchmarks classical models, and writes recruiter-facing JSON artifacts into `docs/data/neu-cls-case-study/`.

## Run it

```bash
python analysis/run_neu_case_study.py
```

## Why this matters

The app already proves operator workflow, reporting, and traceability. This pipeline adds a real CV benchmark, measurable evaluation, and a confidence-based review-queue story that is easier to defend in data-science interviews.

## RAG knowledge assistant

`run_rag_case_study.py` builds the AI engineering layer: it fetches a pinned Wikipedia knowledge base (CC BY-SA 4.0), chunks it (`rag/chunking.py`), benchmarks TF-IDF against three sentence-embedding models, indexes the winner in Chroma, generates answers with local Qwen2.5 models, scores faithfulness with an NLI cross-encoder, and fits the UMAP projections for the `/rag` page. Results go to `docs/data/rag-knowledge-assistant/`; caches (Wikipedia text, embeddings, model outputs, the Chroma index) go to `analysis/.cache/rag/`.

```bash
pip install -r requirements-rag.txt
python analysis/run_rag_case_study.py
python analysis/generate_rag_visuals.py
```

`rag/metrics.py` and `rag/chunking.py` depend only on NumPy and the standard library so the unit tests run in CI without the model stack. Method and results: [`docs/rag-case-study.md`](../docs/rag-case-study.md).

## CNN baseline (`run_dl_case_study.py`)

Fine-tunes ResNet18 / EfficientNet-B0 on the same 80/20 holdout as the Random Forest, then fits a
calibration temperature on a validation slice of the training part (never on the test rows) and
reports accuracy / macro-F1 with bootstrap 95% intervals, ECE before and after calibration, and
the calibrated review-queue budgets.

```bash
pip install -r requirements.txt -r requirements-dl.txt   # CUDA torch first for GPU, see requirements-dl.txt
python analysis/run_dl_case_study.py --preflight          # GPU, disk, dataset, weights: exits 1 if the run cannot work
python analysis/run_dl_case_study.py                      # single run
python analysis/run_dl_case_study.py --cv-folds 5         # plus 5-fold CV over all 1800 images
```

If the dataset host is unreachable from your machine, download NEU-CLS manually and put the zip at `analysis/.cache/NEU-CLS.zip`; `--preflight` prints the exact path. Writes `docs/data/neu-cls-dl/{summary,reliability,review-queue}.json`. The calibration and bootstrap
helpers (`analysis/dl/`) and the holdout-identity guarantee are covered by `tests/test_calibration.py`
and `tests/test_dl_split.py`; the training loop itself needs a GPU machine to exercise.

### After the real run

```bash
python analysis/update_docs_from_dl.py     # fills the CNN block in README.md from docs/data/neu-cls-dl/summary.json
git add README.md docs/data/neu-cls-dl && git commit -m "docs: record CNN results"
```

The script refuses a `--smoke-test` summary, and `tests/test_docs_consistency.py` fails if the README block ever drifts from the JSON.
