"""Retrieval, grounding and projection-fidelity metrics (NumPy only)."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

_CITATION = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
_STOPWORDS = set(
    "a an and are as at be by can for from has have how in is it its of on or that the this to was "
    "what when where which who why will with does do did during between into than then they their "
    "there these those used use using about also been being more most such".split()
)


def hit_at_k(ranked: Sequence[Sequence], gold: Sequence, k: int) -> float:
    """Share of queries whose gold item appears in the top-k ranked results."""
    if not gold:
        return 0.0
    return float(np.mean([g in list(r)[:k] for r, g in zip(ranked, gold, strict=False)]))


def mean_reciprocal_rank(ranked: Sequence[Sequence], gold: Sequence, k: int = 10) -> float:
    scores = []
    for r, g in zip(ranked, gold, strict=False):
        top = list(r)[:k]
        scores.append(1.0 / (top.index(g) + 1) if g in top else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text).strip()


def answer_sentences(answer: str) -> list[str]:
    """Split a generated answer into checkable claims (citations removed, fragments dropped)."""
    cleaned = strip_citations(answer).replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if len(p.split()) >= 4]


def faithfulness(entailment_probs: Sequence[float], threshold: float = 0.5) -> float | None:
    """Share of answer sentences entailed by at least one retrieved passage."""
    if len(entailment_probs) == 0:
        return None
    return float(np.mean(np.asarray(entailment_probs) >= threshold))


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z\-]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def lexical_overlap(question: str, passage: str) -> float:
    """Share of the question's content words that literally occur in the passage."""
    q = content_words(question)
    return len(q & content_words(passage)) / len(q) if q else 0.0


def neighbor_overlap(a: Sequence, b: Sequence) -> int:
    return len(set(a) & set(b))


def nearest_indices(points: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    """Euclidean k-nearest rows of `points` to `query` (used for the 3D/2D honesty check)."""
    d = np.linalg.norm(points - query[None, :], axis=1)
    return [int(i) for i in np.argsort(d)[:k]]


def premise_windows(passage: str, sizes: Sequence[int] = (1, 2, 3)) -> list[str]:
    """Sentence windows of a passage, used as NLI premises.

    MNLI-trained cross-encoders under-score entailment when the premise is a whole
    150-200 word passage; scoring against short windows and taking the max (SummaC-style)
    keeps a supported claim from being marked unsupported just because of premise length.
    """
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", passage.strip()) if s]
    windows: list[str] = []
    for size in sizes:
        for start in range(max(1, len(sentences) - size + 1)):
            window = " ".join(sentences[start:start + size])
            if window and window not in windows:
                windows.append(window)
    return windows
