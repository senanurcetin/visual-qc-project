"""Fine-tune a CNN on NEU-CLS and compare it with the Random Forest on the SAME holdout.

Needs a GPU machine (CPU works but is slow):

    pip install -r requirements-dl.txt
    python analysis/run_dl_case_study.py                 # single 80/20 run
    python analysis/run_dl_case_study.py --cv-folds 5    # plus 5-fold cross-validation

The 80/20 split reproduces the one in run_neu_case_study.py (same seed, same stratification), so
accuracy / macro-F1 / review-queue numbers are directly comparable. A further 10% of the training
part is held out to fit the calibration temperature and pick the best epoch; the test rows are
never used for either.

Outputs: docs/data/neu-cls-dl/*.json and analysis/.cache/neu_dl_logits.npz (test logits, for
Grad-CAM and later analysis).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.dl.calibration import (  # noqa: E402
    expected_calibration_error,
    fit_temperature,
    nll,
    reliability_bins,
    softmax,
)
from analysis.dl.stats import bootstrap_ci  # noqa: E402
from analysis.run_neu_case_study import (  # noqa: E402
    CACHE_DIR,
    RANDOM_SEED,
    build_review_queue,
    ensure_dataset,
    infer_label,
    iter_image_paths,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "data" / "neu-cls-dl"
BASELINE_FILE = ROOT / "docs" / "data" / "neu-cls-case-study" / "benchmark-comparison.json"
IMAGE_SIZE = 224
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_images(paths) -> np.ndarray:
    """Grayscale uint8 array of shape (N, IMAGE_SIZE, IMAGE_SIZE)."""
    out = np.empty((len(paths), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        out[i] = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return out


def build_model(arch: str, num_classes: int, pretrained: bool):
    import torch.nn as nn
    from torchvision import models

    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


def to_input(batch_u8, device):
    """uint8 (B,H,W) -> normalised float (B,3,H,W) on device."""
    import torch

    x = torch.from_numpy(batch_u8).to(device).float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def predict_logits(model, images, device, batch_size) -> np.ndarray:
    import torch

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            chunks.append(model(to_input(images[start:start + batch_size], device)).float().cpu().numpy())
    return np.concatenate(chunks)


def train_model(arch, x_train, y_train, x_val, y_val, num_classes, args, device):
    """Fine-tune and return the weights of the epoch with the best validation accuracy."""
    import copy

    import torch
    import torch.nn.functional as F

    torch.manual_seed(RANDOM_SEED)
    model = build_model(arch, num_classes, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(x_train) / args.batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    rng = np.random.default_rng(RANDOM_SEED)
    best_acc, best_state = -1.0, None

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(x_train))
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            batch = x_train[idx].copy()
            flip_h, flip_v = rng.random(len(idx)) < 0.5, rng.random(len(idx)) < 0.5
            batch[flip_h] = batch[flip_h][:, :, ::-1]
            batch[flip_v] = batch[flip_v][:, ::-1, :]
            inputs = to_input(np.ascontiguousarray(batch), device)
            targets = torch.from_numpy(y_train[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss) * len(idx)
        val_acc = accuracy_score(y_val, predict_logits(model, x_val, device, args.batch_size).argmax(1))
        print(f"  epoch {epoch + 1:>2}/{args.epochs}  train_loss={total_loss / len(order):.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_acc


def split_train_val(indices, labels):
    return train_test_split(indices, test_size=0.10, random_state=RANDOM_SEED, stratify=labels[indices])


def write_json(name, payload):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-pretrained", action="store_true", help="random init instead of ImageNet weights")
    parser.add_argument("--cv-folds", type=int, default=0, help="also run K-fold CV over all 1800 images (slow)")
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + ("" if device.type == "cuda" else "  (no GPU found - this will be slow)"))

    ensure_dataset()
    paths = list(iter_image_paths())
    labels_text = np.array([infer_label(p) for p in paths])
    class_names = sorted(set(labels_text.tolist()))
    labels = np.array([class_names.index(name) for name in labels_text])
    images = load_images(paths)

    # Same outer split as run_neu_case_study.py (indices depend only on the labels and the seed).
    all_idx = np.arange(len(paths))
    train_idx, test_idx = train_test_split(all_idx, test_size=0.20, random_state=RANDOM_SEED, stratify=labels)
    fit_idx, val_idx = split_train_val(train_idx, labels)

    print(f"Training {args.arch}: {len(fit_idx)} fit / {len(val_idx)} val / {len(test_idx)} test")
    started = time.time()
    model, best_val_acc = train_model(
        args.arch, images[fit_idx], labels[fit_idx], images[val_idx], labels[val_idx], len(class_names), args, device
    )
    train_seconds = time.time() - started

    val_logits = predict_logits(model, images[val_idx], device, args.batch_size)
    test_logits = predict_logits(model, images[test_idx], device, args.batch_size)
    y_test = labels[test_idx]
    temperature = fit_temperature(val_logits, labels[val_idx])
    raw_p, cal_p = softmax(test_logits), softmax(test_logits, temperature)
    preds = cal_p.argmax(1)

    def macro_f1(t, p):
        return float(f1_score(t, p, average="macro", zero_division=0))

    acc, f1 = float(accuracy_score(y_test, preds)), macro_f1(y_test, preds)
    acc_ci = bootstrap_ci(y_test, preds, lambda t, p: float((t == p).mean()))
    f1_ci = bootstrap_ci(y_test, preds, macro_f1)
    baseline = next(m for m in json.loads(BASELINE_FILE.read_text()) if m["model"] == "random_forest")

    summary = {
        "architecture": args.arch, "pretrained": not args.no_pretrained, "epochs": args.epochs,
        "device": device.type, "train_seconds": round(train_seconds, 1), "seed": RANDOM_SEED,
        "split": {"fit": len(fit_idx), "val": len(val_idx), "test": len(test_idx)},
        "best_val_accuracy": round(best_val_acc, 4),
        "test": {
            "accuracy": round(acc, 4), "accuracy_ci95": [round(v, 4) for v in acc_ci],
            "macro_f1": round(f1, 4), "macro_f1_ci95": [round(v, 4) for v in f1_ci],
        },
        "random_forest_baseline": {"accuracy": baseline["accuracy"], "macro_f1": baseline["macro_f1"]},
        "calibration": {
            "temperature": round(temperature, 4),
            "nll_before": round(nll(test_logits, y_test), 4), "nll_after": round(nll(test_logits, y_test, temperature), 4),
            "ece_before": round(expected_calibration_error(raw_p, y_test), 4),
            "ece_after": round(expected_calibration_error(cal_p, y_test), 4),
        },
    }

    if args.cv_folds > 1:
        fold_acc = []
        for k, (tr, te) in enumerate(StratifiedKFold(args.cv_folds, shuffle=True, random_state=RANDOM_SEED).split(all_idx, labels)):
            print(f"CV fold {k + 1}/{args.cv_folds}")
            f_idx, v_idx = split_train_val(tr, labels)
            fold_model, _ = train_model(
                args.arch, images[f_idx], labels[f_idx], images[v_idx], labels[v_idx], len(class_names), args, device
            )
            fold_acc.append(float(accuracy_score(labels[te], predict_logits(fold_model, images[te], device, args.batch_size).argmax(1))))
        summary["cross_validation"] = {
            "folds": args.cv_folds, "accuracy_per_fold": [round(a, 4) for a in fold_acc],
            "accuracy_mean": round(float(np.mean(fold_acc)), 4), "accuracy_std": round(float(np.std(fold_acc)), 4),
        }

    write_json("summary.json", summary)
    write_json("reliability.json", {
        "before": reliability_bins(raw_p, y_test), "after": reliability_bins(cal_p, y_test),
    })
    write_json("review-queue.json", {"review_budgets": build_review_queue(cal_p, preds, y_test)})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_DIR / "neu_dl_logits.npz", test_logits=test_logits, test_idx=test_idx, targets=y_test,
             temperature=temperature, class_names=np.array(class_names))
    torch.save(model.state_dict(), CACHE_DIR / f"neu_dl_{args.arch}.pt")

    print(json.dumps(summary, indent=2))
    print(f"\nRandom Forest on the same holdout: acc={baseline['accuracy']}  macro_f1={baseline['macro_f1']}")
    print(f"{args.arch}: acc={acc:.4f} {tuple(round(v, 4) for v in acc_ci)}  macro_f1={f1:.4f}")


if __name__ == "__main__":
    main()
