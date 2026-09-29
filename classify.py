"""Image classification endpoint backed by the ONNX model exported by analysis/run_dl_case_study.py.

    POST /api/classify   multipart form, field `image` (any format OpenCV can decode, <= 5 MB)

The model directory (`QC_MODEL_DIR`, default `models/`) holds `model.onnx` and `meta.json`
(class names, input size, calibration temperature). Without the model or `onnxruntime` the
endpoint answers 503 with the reason, so the hosted demo (which ships neither) degrades cleanly.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request

from ratelimit import rate_limited

classify_bp = Blueprint("classify", __name__)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
REVIEW_ENTROPY_BITS = 1.0  # entropy above this routes the image to human review


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    # Deliberately not imported from analysis/: that directory is excluded from the Vercel bundle.
    z = np.asarray(logits, dtype=np.float64) / temperature
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class ModelUnavailable(RuntimeError):
    pass


def model_dir() -> Path:
    return Path(os.environ.get("QC_MODEL_DIR", Path(__file__).resolve().parent / "models"))


def load_model() -> dict:
    """Load (and cache on the app) the ONNX session and its metadata."""
    cached = current_app.extensions.get("classifier")
    if cached is not None:
        return cached
    directory = model_dir()
    if not (directory / "model.onnx").exists() or not (directory / "meta.json").exists():
        raise ModelUnavailable(f"no model found in {directory}; export one with analysis/run_dl_case_study.py --export-onnx")
    try:
        import onnxruntime
    except ImportError as exc:
        raise ModelUnavailable("onnxruntime is not installed (pip install -r requirements-serve.txt)") from exc
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    session = onnxruntime.InferenceSession(str(directory / "model.onnx"), providers=["CPUExecutionProvider"])
    current_app.extensions["classifier"] = {"session": session, "meta": meta}
    return current_app.extensions["classifier"]


def preprocess(data: bytes, size: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("could not decode the upload as an image")
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[None, None]


@classify_bp.route("/api/classify", methods=["POST"])
@rate_limited("classify", calls=int(os.environ.get("CLASSIFY_RATE_PER_MIN", "20")))
def classify():
    upload = request.files.get("image")
    if upload is None:
        return jsonify({"error": "send the image as multipart field 'image'"}), 400
    data = upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return jsonify({"error": f"image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"}), 413
    try:
        model = load_model()
    except ModelUnavailable as exc:
        return jsonify({"error": "classifier unavailable", "reason": str(exc)}), 503
    try:
        batch = preprocess(data, int(model["meta"]["image_size"]))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    logits = model["session"].run(["logits"], {"image": batch})[0]
    probs = softmax(logits, float(model["meta"].get("temperature", 1.0)))[0]
    names = model["meta"]["class_names"]
    entropy = float(-(probs * np.log2(np.clip(probs, 1e-12, 1.0))).sum())
    order = np.argsort(probs)[::-1]
    return jsonify({
        "label": names[int(order[0])],
        "confidence": round(float(probs[order[0]]), 4),
        "entropy_bits": round(entropy, 4),
        "max_entropy_bits": round(math.log2(len(names)), 4),
        "needs_review": entropy > REVIEW_ENTROPY_BITS,
        "ranking": [{"label": names[int(i)], "probability": round(float(probs[i]), 4)} for i in order],  # most likely first
        "probabilities": {names[int(i)]: round(float(probs[i]), 4) for i in order},  # keyed lookup (JSON key order is not meaningful)
    })
