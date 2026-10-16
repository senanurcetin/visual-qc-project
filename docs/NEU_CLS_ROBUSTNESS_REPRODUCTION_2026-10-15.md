# NEU-CLS image perturbation robustness — reproduced 15 October 2026

## Evidence identity

This is a **new, independently reproducible classical CV experiment**; it does not restore the unavailable Kaggle notebook *NEU Steel Defects: 100% Accuracy and Where It Ends*, nor does it re-run an absent CNN model checkpoint. No claimed 100% CNN result is attached to this study.

- Public dataset: **NEU-CLS**, Figshare DOI `10.6084/m9.figshare.28903550.v1`, CC BY 4.0
- Archive SHA-256: `df3f85c2826ac2f3b9fa1a9bc750781a80775c9fba00f7b8bc86138bfb4b1e94`
- Corpus: **1,800 images / 6 classes**
- Stratified train/holdout: **1,440 / 360**, random seed **42**
- Holdout filenames SHA-256: `26f63a336a40856b4af09a6ec8bf0daa27e6801ab79cb0cd8a29d253ab9480f5`
- Model: **250-tree Random Forest** using HOG (1,764) + Gabor (16) + grid statistics (32), fit only on original training images
- Perturbations: *only on held-out test images*, with deterministic RNG per sample; same trained model throughout

## Actual measured results

| Evaluation condition | Accuracy | Macro-F1 | Accuracy delta (percentage points) |
| --- | ---: | ---: | ---: |
| Clean baseline | 0.938889 | 0.939231 | 0.0000 |
| Gaussian noise, σ=12 | 0.888889 | 0.888087 | -5.0000 |
| Gaussian blur 5×5 | 0.933333 | 0.933577 | -0.5556 |
| Brightness +25 | 0.833333 | 0.839553 | -10.5556 |
| Contrast ×0.65 around gray=127.5 | 0.775000 | 0.773130 | -16.3889 |
| JPEG quality 45 | 0.933333 | 0.933486 | -0.5556 |

The largest measured weakness is **contrast reduction**. These values apply to the **balanced public benchmark only**, not production-line sensor drift or the unavailable Kaggle CNN experiment. No model weights are committed; the seed, preprocessing, split and training code reproduce the classical Random Forest model.

## Reproduce

From the repository root, create a compatible Python environment with OpenCV **4.x** and scikit-learn:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m analysis.run_perturbation_robustness
python -m pytest tests/test_robustness_transforms.py -q
```

The script downloads the pinned public NEU-CLS archive to ignored `analysis/.cache/` when absent and writes `docs/data/neu-cls-robustness/summary.json`. It does not ship or publish original images. It must not be interpreted as a re-run of proprietary data, live plant telemetry, or missing neural-network weights. Data and model artefacts outside the JSON summary remain local.

## Source and tests

- [Reproduction script](../analysis/run_perturbation_robustness.py)
- [Machine-readable results](data/neu-cls-robustness/summary.json)
- [Data-free deterministic transform tests](../tests/test_robustness_transforms.py)
- [Classical baseline method](../analysis/run_neu_case_study.py)

**Outstanding issue #50:** The original Kaggle notebook's ownership/version/public URL remains unverified. This study supplies genuine new robustness evidence but does not claim to recover that original notebook.

## Notebook publication and retired Kaggle link

The previous unavailable Kaggle notebook is explicitly **retired**, not recovered. A [public GitHub Jupyter notebook](../notebooks/neu_cls_robustness_reproduction.ipynb) supplies the exact classical CV perturbations, deterministic split and environment-level output code without API credentials. Its full-data cells have not been rerun in this patch; measured values cited above remain those from the independently verified original October 15 reproduction. Publishing a new Kaggle version requires the owner's authenticated Kaggle access and a separate public URL validation. The old historical CNN perturbation metrics are not reproduced by the Random Forest notebook.
