from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, abort, jsonify, send_from_directory

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "docs" / "data" / "rag-knowledge-assistant"
WEB_DIR = ROOT / "web" / "rag"

ARTIFACTS = {
    "summary": "summary.json",
    "corpus": "corpus.json",
    "map": "embedding-map.json",
    "samples": "sample-queries.json",
    "retrieval": "retrieval-benchmark.json",
    "generation": "generation-eval.json",
    "fidelity": "projection-fidelity.json",
}

rag_bp = Blueprint("rag", __name__, static_folder=str(WEB_DIR), static_url_path="/rag/static")


def load_rag_artifacts() -> dict:
    missing = [name for name in ARTIFACTS.values() if not (ARTIFACT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"RAG artifacts are missing ({', '.join(missing)}). Run python analysis/run_rag_case_study.py.")
    return {key: json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8")) for key, name in ARTIFACTS.items()}


@rag_bp.route("/api/rag")
def rag_api():
    try:
        return jsonify(load_rag_artifacts())
    except FileNotFoundError as exc:
        abort(503, description=str(exc))


@rag_bp.route("/rag")
def rag_page():
    return send_from_directory(WEB_DIR, "index.html")
