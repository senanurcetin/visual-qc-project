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
