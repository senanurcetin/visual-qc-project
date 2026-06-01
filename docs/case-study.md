# Visual QC Project Case Study

## Positioning

`visual-qc-project` is an industrial computer-vision support case study. It combines two layers that matter in manufacturing:

- a lightweight operator-facing QC workflow in Flask
- a measurable steel defect classification benchmark on a real public dataset

## Problem

Surface inspection projects often stop at a model demo. Real quality teams still need a workflow for traceability, reporting, and operator review when the model is uncertain.

## Dataset and signal

- Dataset: `NEU-CLS`
- Domain: steel strip surface defect classification
- Size: `1,800` grayscale images
- Classes: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`
- License: `CC BY 4.0`

## Modeling approach

- Feature pipeline: `64x64` grayscale HOG descriptors
- Benchmarks:
  - dummy baseline
  - logistic regression
  - random forest
- Evaluation:
  - deterministic `80/20` stratified holdout
  - accuracy, macro precision, macro recall, macro F1
  - confidence-based review queue analysis

## Why the final model was selected

The random-forest baseline is the published model because it produced the strongest macro-F1 and accuracy without adding GPU or paid-infrastructure assumptions. That keeps the repo honest, reproducible, and free-tier friendly.

## Current results

- Accuracy: `0.8167`
- Macro precision: `0.8163`
- Macro recall: `0.8167`
- Macro F1: `0.8093`
- Review queue: the `15%` lowest-confidence predictions capture `33.3%` of routing errors with `2.22x` better error yield than random review

## Operational framing

The useful deployment story is not full autonomy. It is defect triage:

- classify likely defect type automatically
- surface low-confidence predictions to a human reviewer
- preserve the historian/export workflow already present in the app

## What this proves

- industrial quality problems can be framed as measured CV systems, not only dashboards
- model outputs can be connected to an operator review queue
- the repo can support both workflow design and DS-style evaluation

## Limitations

- the dataset is balanced and public, so it does not capture plant-specific drift or class imbalance
- the benchmark is classical CV, not a production deep-learning stack
- metrics should be treated as portfolio evidence, not a production claim
