from __future__ import annotations

import json
import math
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import cv2
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "analysis" / ".cache"
OUTPUT_DIR = ROOT / "docs" / "data" / "neu-cls-case-study"
DATASET_URL = "https://ndownloader.figshare.com/files/54094775"
DATASET_DOI = "10.6084/m9.figshare.28903550.v1"
DATASET_NAME = "NEU-CLS"
ZIP_PATH = CACHE_DIR / f"{DATASET_NAME}.zip"
EXTRACT_DIR = CACHE_DIR / DATASET_NAME
IMAGE_DIRS = (
    EXTRACT_DIR / "train" / "train" / "images",
    EXTRACT_DIR / "valid" / "valid" / "images",
)
CLASS_DESCRIPTIONS = {
    "crazing": "Micro-crack patterns that indicate surface fatigue and coating stress.",
    "inclusion": "Foreign material trapped in the strip surface that can propagate into downstream defects.",
    "patches": "Irregular surface patches that often require operator review before shipment.",
    "pitted_surface": "Localized pitting that degrades finish quality and can lead to corrosion risk.",
    "rolled-in_scale": "Scale rolled into the steel strip, causing persistent texture distortion.",
    "scratches": "Linear scoring defects that commonly drive rework or rejection decisions.",
}
HOG_DESCRIPTOR = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
RANDOM_SEED = 42


def ensure_dataset() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading {DATASET_NAME} dataset...")
        urlretrieve(DATASET_URL, ZIP_PATH)
    if not EXTRACT_DIR.exists():
        print("Extracting dataset archive...")
        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(EXTRACT_DIR)


def infer_label(image_path: Path) -> str:
    return image_path.stem.rsplit("_", 1)[0]


def iter_image_paths() -> Iterable[Path]:
    for image_dir in IMAGE_DIRS:
        yield from sorted(image_dir.glob("*.jpg"))


def compute_features(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    return HOG_DESCRIPTOR.compute(resized).reshape(-1)


def build_dataset() -> tuple[np.ndarray, np.ndarray, list[Path]]:
    image_paths = list(iter_image_paths())
    labels = np.array([infer_label(path) for path in image_paths])
    features = np.vstack([compute_features(path) for path in image_paths])
    return features, labels, image_paths


def make_models() -> dict[str, object]:
    return {
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }


def to_float(value: float) -> float:
    return round(float(value), 4)


def summarize_model(name: str, model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    started_at = time.time()
    model.fit(x_train, y_train)
    training_seconds = time.time() - started_at
    predictions = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    return {
        "model": name,
        "accuracy": to_float(accuracy_score(y_test, predictions)),
        "macro_precision": to_float(precision),
        "macro_recall": to_float(recall),
        "macro_f1": to_float(f1),
        "training_seconds": to_float(training_seconds),
    }


def build_review_queue(probabilities: np.ndarray, predictions: np.ndarray, targets: np.ndarray) -> list[dict]:
    confidence = probabilities.max(axis=1)
    is_error = predictions != targets
    error_count = int(is_error.sum())
    total_samples = len(targets)
    baseline_error_rate = error_count / total_samples if total_samples else 0.0
    order = np.argsort(confidence)
    budgets = []

    for fraction in (0.10, 0.15, 0.20):
        reviewed = max(1, math.ceil(total_samples * fraction))
        selected = order[:reviewed]
        captured_errors = int(is_error[selected].sum())
        review_yield = captured_errors / reviewed if reviewed else 0.0
        budgets.append(
            {
                "review_fraction": fraction,
                "reviewed_samples": reviewed,
                "captured_errors": captured_errors,
                "error_capture_rate": to_float(captured_errors / error_count) if error_count else 0.0,
                "review_yield": to_float(review_yield),
                "random_review_yield": to_float(baseline_error_rate),
                "yield_lift_vs_random": to_float(review_yield / baseline_error_rate) if baseline_error_rate else 0.0,
            }
        )

    return budgets


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dataset()
    x, y, image_paths = build_dataset()
    class_names = sorted(set(y))
    label_counts = Counter(y.tolist())

    (
        x_train,
        x_test,
        y_train,
        y_test,
        train_paths,
        test_paths,
    ) = train_test_split(
        x,
        y,
        image_paths,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    benchmarks = []
    models = make_models()
    for name, model in models.items():
        benchmarks.append(summarize_model(name, model, x_train, y_train, x_test, y_test))

    final_model_name = "random_forest"
    final_model = models[final_model_name]
    final_model.fit(x_train, y_train)
    final_predictions = final_model.predict(x_test)
    final_probabilities = final_model.predict_proba(x_test)
    report = classification_report(y_test, final_predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, final_predictions, labels=class_names)

    confidence = final_probabilities.max(axis=1)
    review_queue = build_review_queue(final_probabilities, final_predictions, y_test)
    primary_review_bucket = next(bucket for bucket in review_queue if bucket["review_fraction"] == 0.15)

    misclassified_rows = []
    for image_path, target, predicted, score in zip(test_paths, y_test, final_predictions, confidence):
        if predicted != target:
            misclassified_rows.append(
                {
                    "file": image_path.name,
                    "actual": target,
                    "predicted": predicted,
                    "confidence": to_float(score),
                }
            )
    misclassified_rows.sort(key=lambda row: row["confidence"])

    hotspots = []
    for actual_index, actual in enumerate(class_names):
        for predicted_index, predicted in enumerate(class_names):
            if actual == predicted:
                continue
            count = int(matrix[actual_index, predicted_index])
            if count:
                hotspots.append({"actual": actual, "predicted": predicted, "count": count})
    hotspots.sort(key=lambda row: row["count"], reverse=True)

    summary = {
        "project": "Industrial surface defect triage benchmark",
        "portfolio_role": "support case study",
        "dataset": {
            "name": DATASET_NAME,
            "doi": DATASET_DOI,
            "source_url": f"https://doi.org/{DATASET_DOI}",
            "license": "CC BY 4.0",
            "total_images": int(len(y)),
            "classes": class_names,
            "class_count": len(class_names),
            "class_distribution": dict(sorted(label_counts.items())),
            "evaluation_split": {
                "train_samples": int(len(y_train)),
                "holdout_samples": int(len(y_test)),
                "strategy": "Deterministic 80/20 stratified holdout over the full public corpus",
            },
        },
        "feature_pipeline": {
            "representation": "64x64 grayscale HOG descriptors",
            "feature_count": int(x.shape[1]),
            "reason": "Fast classical baseline that keeps the pipeline free-tier friendly and reproducible.",
        },
        "final_model": {
            "name": "Random forest on HOG descriptors",
            "accuracy": to_float(accuracy_score(y_test, final_predictions)),
            "macro_precision": to_float(report["macro avg"]["precision"]),
            "macro_recall": to_float(report["macro avg"]["recall"]),
            "macro_f1": to_float(report["macro avg"]["f1-score"]),
        },
        "review_queue": {
            "selected_budget_fraction": primary_review_bucket["review_fraction"],
            "selected_budget_label": "15% lowest-confidence predictions",
            "reviewed_samples": primary_review_bucket["reviewed_samples"],
            "captured_errors": primary_review_bucket["captured_errors"],
            "error_capture_rate": primary_review_bucket["error_capture_rate"],
            "review_yield": primary_review_bucket["review_yield"],
            "random_review_yield": primary_review_bucket["random_review_yield"],
            "yield_lift_vs_random": primary_review_bucket["yield_lift_vs_random"],
        },
        "operational_takeaway": "The model is strong enough to triage defect classes automatically, while a low-confidence review queue concentrates a meaningful share of routing errors into a small analyst workload.",
        "limitations": [
            "This is a public balanced dataset, so real plant imbalance and line drift are not represented.",
            "The model is a classical CV baseline rather than a production deep-learning stack.",
            "Holdout metrics should be treated as portfolio-grade evidence, not as a production claim.",
        ],
    }

    dataset_profile = {
        "name": DATASET_NAME,
        "class_count": len(class_names),
        "total_images": int(len(y)),
        "class_distribution": dict(sorted(label_counts.items())),
        "class_descriptions": [
            {"class_name": class_name, "description": CLASS_DESCRIPTIONS[class_name]}
            for class_name in class_names
        ],
    }

    defect_taxonomy = [
        {"class_name": class_name, "description": CLASS_DESCRIPTIONS[class_name]}
        for class_name in class_names
    ]

    class_metrics = []
    for class_name in class_names:
        class_report = report[class_name]
        class_metrics.append(
            {
                "class_name": class_name,
                "precision": to_float(class_report["precision"]),
                "recall": to_float(class_report["recall"]),
                "f1": to_float(class_report["f1-score"]),
                "support": int(class_report["support"]),
            }
        )

    model_selection = {
        "selected_model": final_model_name,
        "selection_reason": "Random forest produced the best macro-F1 and accuracy without introducing GPU or paid-infrastructure assumptions.",
        "benchmarks_considered": benchmarks,
        "next_step": "Upgrade from HOG features to a compact transfer-learning model when the repo needs a stronger accuracy ceiling.",
    }

    confusion_payload = {
        "labels": class_names,
        "matrix": matrix.astype(int).tolist(),
        "hotspots": hotspots[:8],
    }

    review_payload = {
        "review_budgets": review_queue,
        "misclassified_examples": misclassified_rows[:12],
    }

    write_json(OUTPUT_DIR / "summary.json", summary)
    write_json(OUTPUT_DIR / "dataset-profile.json", dataset_profile)
    write_json(OUTPUT_DIR / "benchmark-comparison.json", benchmarks)
    write_json(OUTPUT_DIR / "model-selection.json", model_selection)
    write_json(OUTPUT_DIR / "class-metrics.json", class_metrics)
    write_json(OUTPUT_DIR / "confusion-matrix.json", confusion_payload)
    write_json(OUTPUT_DIR / "review-queue.json", review_payload)
    write_json(OUTPUT_DIR / "defect-taxonomy.json", defect_taxonomy)

    print(f"Wrote case-study artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
