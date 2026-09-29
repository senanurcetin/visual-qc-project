"""Uncertainty estimates for holdout metrics (NumPy only)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    targets: np.ndarray,
    predictions: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap interval of `metric(targets, predictions)` over resampled test rows."""
    rng = np.random.default_rng(seed)
    n = len(targets)
    scores = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        scores[i] = metric(targets[idx], predictions[idx])
    tail = (1.0 - confidence) / 2.0 * 100.0
    return float(np.percentile(scores, tail)), float(np.percentile(scores, 100.0 - tail))
