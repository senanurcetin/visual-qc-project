"""Background reading for a rejected unit: passages from the steel-QC knowledge base for its defect class.

This links the vision side to the RAG side without a model at request time. It ranks the 728 passage
previews already published in docs/data/rag-knowledge-assistant/ with a small TF-IDF (the same kind
of keyword baseline the RAG benchmark compares against), so it runs on the serverless demo.
It shows encyclopaedic background from Wikipedia, not plant procedures.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from flask import Blueprint, jsonify, request

import line_sim

ARTIFACT_DIR = Path(__file__).resolve().parent / "docs" / "data" / "rag-knowledge-assistant"
knowledge_bp = Blueprint("knowledge", __name__)

# What an inspector would look up for each NEU-CLS class.
DEFECT_QUERIES = {
    "crazing": "crazing network of fine surface cracks fatigue stress coating",
    "inclusion": "non-metallic inclusions steel impurities slag oxides sulfides cleanliness",
    "patches": "surface patches uneven oxide scale rust discoloration surface finish",
    "pitted_surface": "pitting corrosion localized pits surface attack",
    "rolled-in_scale": "mill scale hot rolling scale rolled into surface pickling descaling",
    "scratches": "scratches abrasion linear surface damage roughness surface finish visual inspection",
}
# Sections that match the keywords but say nothing about the defect (history, art, ...).
SKIP_SECTIONS = frozenset({
    "in art", "in popular culture", "history", "historical background", "timeline of research history", "etymology",
})
STOPWORDS = frozenset(
    "a an and are as at be been by for from has have in is it its of on or that the their this to was were which with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


class TfIdfIndex:
    """Sublinear-tf, smoothed-idf, cosine-normalised TF-IDF over a fixed list of documents."""

    def __init__(self, documents: list[str]):
        tokenized = [tokenize(d) for d in documents]
        df = Counter(t for tokens in tokenized for t in set(tokens))
        n = len(documents)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vectors = [self._weigh(tokens) for tokens in tokenized]

    def _weigh(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(t for t in tokens if t in self.idf)
        vec = {t: (1 + math.log(c)) * self.idf[t] for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self._weigh(tokenize(query))
        scored = [(i, sum(w * vec.get(t, 0.0) for t, w in q.items())) for i, vec in enumerate(self.vectors)]
        return sorted((s for s in scored if s[1] > 0), key=lambda s: (-s[1], s[0]))[:k]


@lru_cache(maxsize=1)
def _load() -> tuple[list[dict], TfIdfIndex]:
    emb = json.loads((ARTIFACT_DIR / "embedding-map.json").read_text(encoding="utf-8"))
    corpus = json.loads((ARTIFACT_DIR / "corpus.json").read_text(encoding="utf-8"))
    urls = {a["title"]: a["url"] for a in corpus["articles"]}
    idx = {name: i for i, name in enumerate(emb["fields"])}
    passages = []
    for pid, section, preview, point in zip(emb["ids"], emb["sections"], emb["previews"], emb["points"], strict=True):
        article = emb["articles"][int(point[idx["article"]])]
        if section.split(" / ")[-1].lower() in SKIP_SECTIONS or section.split(" / ")[0].lower() in SKIP_SECTIONS:
            continue
        passages.append({
            "id": pid, "article": article, "topic": emb["topics"][int(point[idx["topic"]])],
            "section": section, "preview": preview, "url": urls.get(article),
        })
    # The article title counts twice: a passage from "Crazing" should beat one that merely mentions it.
    index = TfIdfIndex([f"{p['article']} {p['article']} {p['section']} {p['preview']}" for p in passages])
    return passages, index


@lru_cache(maxsize=64)
def passages_for_defect(defect: str, k: int = 3) -> tuple[dict, ...]:
    passages, index = _load()
    return tuple({**passages[i], "score": round(score, 4)} for i, score in index.search(DEFECT_QUERIES[defect], k))


@knowledge_bp.route("/api/defect-knowledge")
def defect_knowledge():
    defect = request.args.get("defect", "")
    if defect not in DEFECT_QUERIES:
        return jsonify({"error": "defect must be one of the NEU-CLS classes", "classes": line_sim.DEFECT_CLASSES}), 400
    try:
        k = max(1, min(int(request.args.get("k", "3")), 6))
    except ValueError:
        return jsonify({"error": "k must be an integer"}), 400
    return jsonify({
        "defect": defect,
        "source": "steel-QC knowledge base (Wikipedia, pinned revisions); background reading, not plant procedures",
        "passages": list(passages_for_defect(defect, k)),
    })
