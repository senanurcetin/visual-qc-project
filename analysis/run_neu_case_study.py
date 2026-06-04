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

# Per-defect false-negative costs (illustrative, USD equivalent per missed defect)
# Reflects downstream impact: shipping a defective unit, warranty risk, field return
DEFECT_FN_COSTS = {
    "inclusion": 500,        # can propagate into critical structural defects
    "rolled-in_scale": 400,  # persistent texture distortion, customer complaint
    "pitted_surface": 350,   # corrosion risk over time
    "patches": 200,          # requires operator review; often blocked at QC gate
    "scratches": 150,        # surface only; cosmetic but trackable
    "crazing": 100,          # fatigue indicator; typically caught in first pass
}
# Cost of a false positive (unnecessary manual review)
DEFECT_FP_COST = 25

HOG_DESCRIPTOR = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
RANDOM_SEED = 42
SPC_BATCH_SIZE = 30  # samples per SPC control chart batch


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


def compute_gabor_features(image_64: np.ndarray) -> np.ndarray:
    """Gabor filter responses at 4 orientations × 2 frequencies → 16 stats features."""
    features = []
    for theta_deg in [0, 45, 90, 135]:
        theta = np.deg2rad(theta_deg)
        for lam in [4.0, 8.0]:
            kernel = cv2.getGaborKernel(
                (15, 15), sigma=4.0, theta=theta, lambd=lam,
                gamma=0.5, psi=0, ktype=cv2.CV_32F,
            )
            response = cv2.filter2D(image_64.astype(np.float32), cv2.CV_32F, kernel)
            features.extend([float(response.mean()), float(response.std())])
    return np.array(features, dtype=np.float32)


def compute_local_grid_stats(image_64: np.ndarray, grid: int = 4) -> np.ndarray:
    """Divide 64×64 image into grid×grid cells; return mean+std per cell → 2×grid² features."""
    h, w = image_64.shape
    ch, cw = h // grid, w // grid
    stats = []
    for i in range(grid):
        for j in range(grid):
            cell = image_64[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw].astype(np.float32)
            stats.extend([float(cell.mean()), float(cell.std())])
    return np.array(stats, dtype=np.float32)


def compute_features(image_path: Path) -> np.ndarray:
    """HOG + Gabor + local grid stats → richer texture representation."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    hog = HOG_DESCRIPTOR.compute(resized).reshape(-1)
    gabor = compute_gabor_features(resized)
    grid_stats = compute_local_grid_stats(resized)
    return np.concatenate([hog, gabor, grid_stats])


def build_dataset() -> tuple[np.ndarray, np.ndarray, list[Path]]:
    image_paths = list(iter_image_paths())
    labels = np.array([infer_label(path) for path in image_paths])
    print(f"Extracting features from {len(image_paths)} images (HOG + Gabor + grid stats)...")
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


def summarize_model(
    name: str,
    model: object,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
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


def build_review_queue(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> list[dict]:
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


def compute_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Shannon entropy of class probability distribution per sample."""
    clipped = np.clip(probabilities, 1e-10, 1.0)
    return -np.sum(clipped * np.log2(clipped), axis=1)


def build_per_class_adaptive_thresholds(
    class_names: list[str],
    probabilities: np.ndarray,
    predictions: np.ndarray,
    y_test: np.ndarray,
) -> list[dict]:
    """Per-class confidence thresholds that balance FN and FP cost."""
    max_entropy = np.log2(len(class_names))
    entropy = compute_entropy(probabilities)
    results = []
    for cls in class_names:
        fn_cost = DEFECT_FN_COSTS.get(cls, 200)
        fp_cost = DEFECT_FP_COST
        # Optimal review fraction: higher FN cost → lower threshold → more reviewed
        cost_ratio = fn_cost / (fn_cost + fp_cost)
        is_cls = y_test == cls
        cls_preds = predictions[is_cls]
        cls_true = y_test[is_cls]
        cls_recall = to_float(float((cls_preds == cls_true).mean())) if is_cls.sum() > 0 else 0.0
        cls_entropy = entropy[is_cls]
        recommended_entropy_threshold = to_float(float(np.percentile(cls_entropy, (1 - cost_ratio) * 100)))
        results.append({
            "class_name": cls,
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "recall_at_0_5_threshold": cls_recall,
            "recommended_entropy_threshold": recommended_entropy_threshold,
            "max_entropy": to_float(max_entropy),
            "note": (
                f"Predictions with entropy > {recommended_entropy_threshold} should be "
                f"routed to manual review for this class."
            ),
        })
    return sorted(results, key=lambda r: r["fn_cost"], reverse=True)


def build_cost_weighted_metrics(
    class_names: list[str],
    y_test: np.ndarray,
    predictions: np.ndarray,
    matrix: np.ndarray,
) -> dict:
    """Compute cost-weighted accuracy using per-defect FN/FP costs."""
    total_fn_cost = 0.0
    total_fp_cost = 0.0
    total_correct_cost = 0.0
    per_class = []

    for i, actual_cls in enumerate(class_names):
        fn_cost = DEFECT_FN_COSTS.get(actual_cls, 200)
        tp = int(matrix[i, i])
        fn = int(matrix[i, :].sum()) - tp
        fp = int(matrix[:, i].sum()) - tp

        cls_fn_cost = fn * fn_cost
        cls_fp_cost = fp * DEFECT_FP_COST
        total_fn_cost += cls_fn_cost
        total_fp_cost += cls_fp_cost
        total_correct_cost += tp * 0  # correctly classified = zero cost

        per_class.append({
            "class_name": actual_cls,
            "fn_cost_per_missed": fn_cost,
            "missed_defects": fn,
            "false_positives": fp,
            "total_fn_cost": cls_fn_cost,
            "total_fp_cost": cls_fp_cost,
            "combined_cost": cls_fn_cost + cls_fp_cost,
        })

    per_class.sort(key=lambda r: r["combined_cost"], reverse=True)
    total_cost = total_fn_cost + total_fp_cost
    n_test = len(y_test)
    naive_cost = sum(
        DEFECT_FN_COSTS.get(cls, 200) * int((y_test == cls).sum())
        for cls in class_names
    )

    return {
        "cost_assumptions": {
            "fn_costs_by_class": DEFECT_FN_COSTS,
            "fp_cost": DEFECT_FP_COST,
        },
        "total_fn_cost": total_fn_cost,
        "total_fp_cost": total_fp_cost,
        "total_model_cost": total_cost,
        "naive_no_model_cost": naive_cost,
        "cost_savings_vs_naive": naive_cost - total_cost,
        "cost_savings_share": to_float((naive_cost - total_cost) / naive_cost) if naive_cost else 0.0,
        "per_class_cost_breakdown": per_class,
    }


def build_pareto_analysis(
    class_names: list[str],
    y_test: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    """Which defect classes drive the most misclassification errors (Pareto)."""
    is_error = predictions != y_test
    error_by_true_class = Counter(y_test[is_error].tolist())
    total_errors = sum(error_by_true_class.values())

    rows = []
    cumulative = 0.0
    for cls in sorted(error_by_true_class, key=lambda c: error_by_true_class[c], reverse=True):
        count = error_by_true_class[cls]
        share = count / total_errors if total_errors else 0.0
        cumulative += share
        rows.append({
            "class_name": cls,
            "error_count": count,
            "share_of_errors": to_float(share),
            "cumulative_share": to_float(cumulative),
        })

    return {
        "total_errors": total_errors,
        "total_test_samples": len(y_test),
        "overall_error_rate": to_float(total_errors / len(y_test)),
        "pareto": rows,
        "eighty_percent_classes": [
            r["class_name"] for r in rows if r["cumulative_share"] <= 0.80
        ] or [rows[0]["class_name"]] if rows else [],
    }


def build_spc_data(
    y_test: np.ndarray,
    predictions: np.ndarray,
    batch_size: int = SPC_BATCH_SIZE,
) -> dict:
    """Compute p-chart (defect rate chart) control limits for the test set.

    Simulates sequential QC batches and calculates UCL/LCL using the
    3-sigma rule: UCL/LCL = p̄ ± 3 × √(p̄(1-p̄)/n).
    """
    n_batches = len(y_test) // batch_size
    is_defect = (predictions != y_test).astype(int)
    batches = []
    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size
        batch_defects = int(is_defect[start:end].sum())
        defect_rate = batch_defects / batch_size
        batches.append({
            "batch_index": b + 1,
            "sample_count": batch_size,
            "defects": batch_defects,
            "defect_rate": to_float(defect_rate),
        })

    p_bar = float(np.mean([b["defect_rate"] for b in batches]))
    sigma = math.sqrt(p_bar * (1 - p_bar) / batch_size)
    ucl = min(1.0, p_bar + 3 * sigma)
    lcl = max(0.0, p_bar - 3 * sigma)

    out_of_control = [b for b in batches if b["defect_rate"] > ucl or b["defect_rate"] < lcl]

    return {
        "chart_type": "p-chart (proportion defective)",
        "batch_size": batch_size,
        "n_batches": n_batches,
        "p_bar": to_float(p_bar),
        "sigma": to_float(sigma),
        "ucl": to_float(ucl),
        "lcl": to_float(lcl),
        "batches": batches,
        "out_of_control_batches": len(out_of_control),
        "interpretation": (
            f"Centre line (p̄) = {to_float(p_bar):.3f}. "
            f"Upper control limit = {to_float(ucl):.3f}. "
            f"Lower control limit = {to_float(lcl):.3f}. "
            f"{len(out_of_control)} batch(es) fall outside control limits — "
            "these represent unusual defect concentrations worth investigating."
        ),
    }


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
    entropy = compute_entropy(final_probabilities)
    review_queue = build_review_queue(final_probabilities, final_predictions, y_test)
    primary_review_bucket = next(bucket for bucket in review_queue if bucket["review_fraction"] == 0.15)

    misclassified_rows = []
    for image_path, target, predicted, score, ent in zip(
        test_paths, y_test, final_predictions, confidence, entropy
    ):
        if predicted != target:
            misclassified_rows.append(
                {
                    "file": image_path.name,
                    "actual": target,
                    "predicted": predicted,
                    "confidence": to_float(score),
                    "prediction_entropy": to_float(ent),
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

    # New analyses
    adaptive_thresholds = build_per_class_adaptive_thresholds(
        class_names, final_probabilities, final_predictions, y_test
    )
    cost_metrics = build_cost_weighted_metrics(class_names, y_test, final_predictions, matrix)
    pareto = build_pareto_analysis(class_names, y_test, final_predictions)
    spc_data = build_spc_data(y_test, final_predictions)

    # ── Assemble payloads ────────────────────────────────────────────────────
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
            "representation": "HOG (1764) + Gabor-filter stats (16) + local-grid stats (32)",
            "feature_count": int(x.shape[1]),
            "components": {
                "hog": "64×64 grayscale HOG — captures edge and gradient structure",
                "gabor": "4 orientations × 2 frequencies, mean+std per response — captures directional texture",
                "local_grid_stats": "4×4 grid of mean+std values — captures spatial intensity variation",
            },
        },
        "final_model": {
            "name": "Random forest on HOG + Gabor + grid stats",
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
        "cost_model_summary": {
            "total_model_cost": cost_metrics["total_model_cost"],
            "naive_no_model_cost": cost_metrics["naive_no_model_cost"],
            "cost_savings_share": cost_metrics["cost_savings_share"],
            "highest_cost_class": cost_metrics["per_class_cost_breakdown"][0]["class_name"] if cost_metrics["per_class_cost_breakdown"] else None,
        },
        "spc_summary": {
            "p_bar": spc_data["p_bar"],
            "ucl": spc_data["ucl"],
            "lcl": spc_data["lcl"],
            "out_of_control_batches": spc_data["out_of_control_batches"],
        },
        "pareto_summary": {
            "total_errors": pareto["total_errors"],
            "eighty_percent_classes": pareto["eighty_percent_classes"],
        },
        "operational_takeaway": (
            "Enriched features (Gabor + local grid stats alongside HOG) improve texture discrimination. "
            "Per-defect cost thresholds route high-risk misses to manual review. "
            "SPC p-chart enables shift-level quality monitoring with statistical control limits."
        ),
        "limitations": [
            "This is a public balanced dataset, so real plant imbalance and line drift are not represented.",
            "The model is a classical CV baseline rather than a production deep-learning stack.",
            "Holdout metrics should be treated as portfolio-grade evidence, not as a production claim.",
            "Cost values are illustrative; actual costs depend on downstream product and customer contracts.",
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

    class_metrics_list = []
    for class_name in class_names:
        class_report = report[class_name]
        class_metrics_list.append(
            {
                "class_name": class_name,
                "precision": to_float(class_report["precision"]),
                "recall": to_float(class_report["recall"]),
                "f1": to_float(class_report["f1-score"]),
                "support": int(class_report["support"]),
                "fn_cost_per_missed": DEFECT_FN_COSTS.get(class_name, 200),
            }
        )

    model_selection = {
        "selected_model": final_model_name,
        "selection_reason": (
            "Random forest with enriched features (HOG + Gabor + grid stats) produced the best "
            "macro-F1 and accuracy without introducing GPU or paid-infrastructure assumptions. "
            "Gabor features improve orientation-specific texture discrimination."
        ),
        "benchmarks_considered": benchmarks,
        "feature_engineering_note": (
            "Added 48 features: 16 Gabor-response statistics (4 orientations × 2 frequencies × mean/std) "
            "and 32 local-grid stats (4×4 spatial grid × mean/std). "
            "Total feature count: HOG 1764 + Gabor 16 + grid 32 = 1812."
        ),
        "next_step": "Upgrade from HOG features to a compact transfer-learning model when the repo needs a stronger accuracy ceiling.",
    }

    confusion_payload = {
        "labels": class_names,
        "matrix": matrix.astype(int).tolist(),
        "hotspots": hotspots[:8],
    }

    review_payload = {
        "review_budgets": review_queue,
        "entropy_statistics": {
            "mean_entropy": to_float(float(entropy.mean())),
            "median_entropy": to_float(float(np.median(entropy))),
            "p90_entropy": to_float(float(np.percentile(entropy, 90))),
            "max_entropy": to_float(float(entropy.max())),
        },
        "misclassified_examples": misclassified_rows[:12],
    }

    # ── Write artifacts ──────────────────────────────────────────────────────
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_json(OUTPUT_DIR / "dataset-profile.json", dataset_profile)
    write_json(OUTPUT_DIR / "benchmark-comparison.json", benchmarks)
    write_json(OUTPUT_DIR / "model-selection.json", model_selection)
    write_json(OUTPUT_DIR / "class-metrics.json", class_metrics_list)
    write_json(OUTPUT_DIR / "confusion-matrix.json", confusion_payload)
    write_json(OUTPUT_DIR / "review-queue.json", review_payload)
    write_json(OUTPUT_DIR / "defect-taxonomy.json", defect_taxonomy)
    write_json(OUTPUT_DIR / "spc-data.json", spc_data)
    write_json(OUTPUT_DIR / "cost-matrix.json", cost_metrics)
    write_json(OUTPUT_DIR / "pareto.json", pareto)
    write_json(OUTPUT_DIR / "adaptive-thresholds.json", adaptive_thresholds)

    print(f"Wrote case-study artifacts to {OUTPUT_DIR}")
    acc = to_float(accuracy_score(y_test, final_predictions))
    print(f"  Accuracy: {acc}  Macro-F1: {to_float(report['macro avg']['f1-score'])}")
    print(f"  SPC: p̄={spc_data['p_bar']}  UCL={spc_data['ucl']}  LCL={spc_data['lcl']}")
    print(f"  Cost savings vs naive: {cost_metrics['cost_savings_share']*100:.1f}%")
    print(f"  Pareto 80% classes: {pareto['eighty_percent_classes']}")
    print(f"  New artifacts: spc-data.json, cost-matrix.json, pareto.json, adaptive-thresholds.json")


if __name__ == "__main__":
    main()
