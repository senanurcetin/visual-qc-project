"""Deterministic NEU-CLS image-perturbation robustness evaluation.

Uses the same 80/20 seed/stratification and HOG + Gabor + grid features as
run_neu_case_study.py. Re-fits the documented 250-tree Random Forest using
TRAIN ONLY; test images are perturbed without altering training samples.
No claim is made about any unrecovered Kaggle neural-network notebook.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from analysis.run_neu_case_study import (
    HOG_DESCRIPTOR,
    RANDOM_SEED,
    ZIP_PATH,
    compute_gabor_features,
    compute_local_grid_stats,
    ensure_dataset,
    infer_label,
    iter_image_paths,
    make_models,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "neu-cls-robustness"


def features_from_gray(image: np.ndarray) -> np.ndarray:
    image64 = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    return np.concatenate([
        HOG_DESCRIPTOR.compute(image64).reshape(-1),
        compute_gabor_features(image64),
        compute_local_grid_stats(image64),
    ])


def perturb(image: np.ndarray, mode: str, seed: int) -> np.ndarray:
    """Fixed image-space interventions; RNG reset independently per sample."""
    image = np.asarray(image, dtype=np.uint8)
    if mode == "clean":
        return image.copy()
    if mode == "gaussian_noise_sigma_12":
        noise = np.random.default_rng(seed).normal(0, 12, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if mode == "gaussian_blur_5":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if mode == "brightness_plus_25":
        return np.clip(image.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    if mode == "contrast_065":
        return np.clip((image.astype(np.float32) - 127.5) * 0.65 + 127.5, 0, 255).astype(np.uint8)
    if mode == "jpeg_quality_45":
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 45])
        if not ok:
            raise RuntimeError("JPEG compression failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if decoded is None or decoded.shape != image.shape:
            raise RuntimeError("JPEG decoding failed")
        return decoded
    raise ValueError(f"Unknown perturbation {mode}")


MODES = (
    "clean", "gaussian_noise_sigma_12", "gaussian_blur_5",
    "brightness_plus_25", "contrast_065", "jpeg_quality_45",
)


def run(output: Path = OUT) -> dict:
    ensure_dataset()
    paths = list(iter_image_paths())
    labels = np.array([infer_label(p) for p in paths])
    if len(paths) != 1800 or len(set(labels)) != 6:
        raise RuntimeError(f"Unexpected NEU-CLS corpus: {len(paths)} rows, {len(set(labels))} classes")
    train_idx, test_idx = train_test_split(
        np.arange(len(paths)), test_size=0.20, random_state=RANDOM_SEED, stratify=labels
    )
    print(f"Train: {len(train_idx)}; held-out test: {len(test_idx)}", flush=True)
    print("Extracting original train features...", flush=True)
    x_train = []
    for i in train_idx:
        img = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Unreadable dataset image {paths[i]}")
        x_train.append(features_from_gray(img))
    model = make_models()["random_forest"]
    print("Fitting documented Random Forest on training samples only...", flush=True)
    model.fit(np.vstack(x_train), labels[train_idx])
    test_images = []
    for i in test_idx:
        img = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Unreadable dataset image {paths[i]}")
        test_images.append(img)
    y = labels[test_idx]
    results = {}
    clean_prediction = None
    for mode in MODES:
        print("Evaluating", mode, flush=True)
        features = np.vstack([
            features_from_gray(perturb(img, mode, RANDOM_SEED + int(idx)))
            for img, idx in zip(test_images, test_idx, strict=True)
        ])
        predicted = model.predict(features)
        if mode == "clean":
            clean_prediction = predicted.copy()
        results[mode] = {
            "samples": len(y),
            "accuracy": round(float(accuracy_score(y, predicted)), 6),
            "macro_f1": round(float(f1_score(y, predicted, average="macro", zero_division=0)), 6),
            "flip_fraction_vs_clean": round(float(np.mean(predicted != clean_prediction)), 6),
        }
    clean_acc = results["clean"]["accuracy"]
    for mode in MODES:
        results[mode]["accuracy_delta_percentage_points"] = round(
            100 * (results[mode]["accuracy"] - clean_acc), 4
        )
    manifest = {
        "evaluation_kind": "reproduced_neu_cls_random_forest_image_perturbations",
        "notebook_identity": "New reproducible study; not a recovery of an unavailable Kaggle notebook",
        "dataset_doi": "10.6084/m9.figshare.28903550.v1",
        "dataset_archive_sha256": hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest(),
        "holdout_image_list_sha256": hashlib.sha256(
            "\n".join(paths[i].name for i in test_idx).encode("utf-8")
        ).hexdigest(),
        "model": "250-tree random forest, HOG+Gabor+grid features",
        "random_state": RANDOM_SEED,
        "split": {"train": len(train_idx), "test": len(test_idx), "stratified": True},
        "experimental_scope": "Public balanced corpus, offline test; not real production robustness or 100 percent CNN replication",
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("OUTPUT", output / "summary.json", flush=True)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    run(args.output)
