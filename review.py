"""Operator review of rejected units: confirm the predicted defect or correct it."""
from __future__ import annotations

import csv
import io
import secrets
import time

from flask import Blueprint, Response, current_app, jsonify, request, session

import line_sim
from line_session import line_state
from store import ReviewStore

review_bp = Blueprint("review", __name__)
REVIEW_WINDOW = 20  # most recent completed units offered for review


def get_store() -> ReviewStore:
    if "review_store" not in current_app.extensions:
        current_app.extensions["review_store"] = ReviewStore(current_app.config.get("DATABASE_URL"))
    return current_app.extensions["review_store"]


def visitor_id(create: bool = False) -> str | None:
    vid = session.get("vid")
    if vid is None and create:
        vid = session["vid"] = secrets.token_hex(8)
        session.permanent = True
    return vid


def rejected_units(line_state: dict) -> dict[str, dict]:
    rows = line_sim.history(line_state, time.time(), REVIEW_WINDOW)
    return {r["unit_id"]: r for r in rows if r["status"] != "OK"}


@review_bp.route("/api/review-queue")
def review_queue():
    decisions = get_store().for_visitor(visitor_id()) if visitor_id() else {}
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
def submit_review():
    body = request.get_json(silent=True) or {}
    unit_id, label = body.get("unit_id"), body.get("label")
    unit = rejected_units(line_state()).get(unit_id)
    if unit is None:
        return jsonify({"error": "unit is not a recent rejected unit on this line"}), 404
    if label not in line_sim.DEFECT_CLASSES:
        return jsonify({"error": "label must be one of the defect classes"}), 400
    decision = "confirm" if label == unit["defect"] else "correct"
    saved = get_store().record(visitor_id(create=True), unit_id, unit["defect"], decision, label)
    return jsonify({k: saved[k] for k in ("unit_id", "predicted_defect", "decision", "operator_label", "updated_at")})


@review_bp.route("/api/review/export.csv")
def export_corrections():
    """Corrected labels of this visitor: candidates for the next retraining set."""
    rows = [d for d in (get_store().for_visitor(visitor_id()).values() if visitor_id() else []) if d["decision"] == "correct"]
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["unit_id", "predicted_defect", "operator_label", "updated_at"])
    writer.writerows([[d["unit_id"], d["predicted_defect"], d["operator_label"], d["updated_at"]] for d in rows])
    return Response(out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=corrections.csv"})
