"""Operator review of rejected units: confirm the predicted defect or correct it."""
from __future__ import annotations

import csv
import io
import os
import secrets
import time

from flask import Blueprint, Response, current_app, jsonify, request, session

import line_sim
from line_session import line_state
from ratelimit import rate_limited
from store import LimitReached, ReviewStore

review_bp = Blueprint("review", __name__)
REVIEW_WINDOW = 20  # most recent completed units offered for review


def get_store() -> ReviewStore:
    if "review_store" not in current_app.extensions:
        current_app.extensions["review_store"] = ReviewStore(current_app.config.get("DATABASE_URL"))
    return current_app.extensions["review_store"]


def visitor_id() -> str | None:
    """This visitor's id, or None if they have never written a decision."""
    vid = session.get("vid")
    return vid if isinstance(vid, str) else None


def ensure_visitor_id() -> str:
    vid = visitor_id()
    if vid is None:
        vid = session["vid"] = secrets.token_hex(8)
        session.permanent = True
    return vid


def rejected_units(line_state: dict) -> dict[str, dict]:
    rows = line_sim.history(line_state, time.time(), REVIEW_WINDOW)
    return {r["unit_id"]: r for r in rows if r["status"] != "OK"}


@review_bp.route("/api/review-queue")
def review_queue():
    vid = visitor_id()
    decisions = get_store().for_visitor(vid) if vid else {}
    items = []
    for unit_id, row in reversed(rejected_units(line_state()).items()):
        decision = decisions.get(unit_id)
        items.append({
            "unit_id": unit_id, "predicted_defect": row["defect"], "time": row["timestamp"].strftime("%H:%M:%S"),
            "decision": decision["decision"] if decision else None,
            "operator_label": decision["operator_label"] if decision else None,
        })
    return jsonify({"items": items, "classes": line_sim.DEFECT_CLASSES})


@review_bp.route("/api/review", methods=["POST"])
@rate_limited("review", calls=int(os.environ.get("REVIEW_RATE_PER_MIN", "30")))
def submit_review():
    body = request.get_json(silent=True) or {}
    unit_id, label = body.get("unit_id"), body.get("label")
    if not isinstance(unit_id, str) or (unit := rejected_units(line_state()).get(unit_id)) is None:
        return jsonify({"error": "unit is not a recent rejected unit on this line"}), 404
    if not isinstance(label, str) or label not in line_sim.DEFECT_CLASSES:
        return jsonify({"error": "label must be one of the defect classes"}), 400
    decision = "confirm" if label == unit["defect"] else "correct"
    try:
        saved = get_store().record(ensure_visitor_id(), unit_id, unit["defect"], decision, label)
    except LimitReached as exc:
        if exc.scope == "visitor":
            return jsonify({"error": "you have reached the review limit for this demo"}), 429
        return jsonify({"error": "review storage is full, try again later"}), 503
    return jsonify({k: saved[k] for k in ("unit_id", "predicted_defect", "decision", "operator_label", "updated_at")})


@review_bp.route("/api/review/export.csv")
def export_corrections():
    """Corrected labels of this visitor: candidates for the next retraining set."""
    vid = visitor_id()
    rows = [d for d in (get_store().for_visitor(vid).values() if vid else []) if d["decision"] == "correct"]
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["unit_id", "predicted_defect", "operator_label", "updated_at"])
    writer.writerows([[d["unit_id"], d["predicted_defect"], d["operator_label"], d["updated_at"]] for d in rows])
    return Response(out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=corrections.csv"})


@review_bp.route("/api/admin/corrections.csv")
def admin_corrections():
    """Quality-engineer view: corrected labels from every operator, as a retraining candidate set.

    Disabled (404) unless REVIEW_ADMIN_TOKEN is set; then it requires `Authorization: Bearer <token>`.
    """
    token = os.environ.get("REVIEW_ADMIN_TOKEN")
    if not token:
        return jsonify({"error": "not found"}), 404
    supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if not secrets.compare_digest(supplied.encode(), token.encode()):
        return jsonify({"error": "unauthorized"}), 401
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["visitor_id", "unit_id", "predicted_defect", "operator_label", "updated_at"])
    writer.writerows([[d["visitor_id"], d["unit_id"], d["predicted_defect"], d["operator_label"], d["updated_at"]]
                      for d in get_store().all_corrections()])
    return Response(out.getvalue(), mimetype="text/csv")
