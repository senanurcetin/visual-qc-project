"""Confidence calibration helpers (NumPy/SciPy only, no torch)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, targets: np.ndarray, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the true class."""
    p = softmax(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(targets)), targets], 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Single scalar T minimising NLL on held-out logits (temperature scaling)."""
    result = minimize_scalar(
        lambda log_t: nll(logits, targets, float(np.exp(log_t))),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )
    return float(np.exp(result.x))


def reliability_bins(probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Per confidence bin: sample count, mean confidence and accuracy."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lo) & (confidence <= hi)
        count = int(in_bin.sum())
        bins.append({
            "lower": float(lo), "upper": float(hi), "count": count,
            "confidence": float(confidence[in_bin].mean()) if count else None,
            "accuracy": float(correct[in_bin].mean()) if count else None,
        })
    return bins


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Sample-weighted mean |accuracy - confidence| over confidence bins."""
    total = len(targets)
    return float(sum(
        b["count"] / total * abs(b["accuracy"] - b["confidence"])
        for b in reliability_bins(probabilities, targets, n_bins) if b["count"]
    ))
