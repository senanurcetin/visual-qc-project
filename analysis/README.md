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
