from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, abort, jsonify, send_from_directory

ARTIFACT_DIR = Path(__file__).resolve().parent / "docs" / "data" / "neu-cls-case-study"

WEB_DIR = Path(__file__).resolve().parent / "web" / "case"

case_study_bp = Blueprint("case_study", __name__, static_folder=str(WEB_DIR), static_url_path="/case/static")


def _load_json(filename: str):
    path = ARTIFACT_DIR / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_case_study_artifacts() -> dict:
    summary = _load_json("summary.json")
    if summary is None:
        raise FileNotFoundError("Case-study artifacts are missing. Run python analysis/run_neu_case_study.py.")
    return {
        "summary": summary,
        "benchmarks": _load_json("benchmark-comparison.json") or [],
        "taxonomy": _load_json("defect-taxonomy.json") or [],
        "confusion": _load_json("confusion-matrix.json") or {"hotspots": [], "labels": [], "matrix": []},
        "review_queue": _load_json("review-queue.json") or {"review_budgets": [], "misclassified_examples": []},
        "class_metrics": _load_json("class-metrics.json") or [],
        "model_selection": _load_json("model-selection.json") or {},
        "dataset_profile": _load_json("dataset-profile.json") or {},
    }


@case_study_bp.route("/api/case-study")
def case_study_api():
    try:
        return jsonify(load_case_study_artifacts())
    except FileNotFoundError as exc:
        abort(503, description=str(exc))


@case_study_bp.route("/case-study")
def case_study_page():
    return send_from_directory(WEB_DIR, "index.html")
