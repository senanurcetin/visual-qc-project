"""Fine-tune a CNN on NEU-CLS and compare it with the Random Forest on the SAME holdout.

Needs a GPU machine for a real run (CPU works but is slow):

    pip install -r requirements-dl.txt
    python analysis/run_dl_case_study.py                       # single 80/20 run
    python analysis/run_dl_case_study.py --cv-folds 5          # plus 5-fold cross-validation
    python analysis/run_dl_case_study.py --export-onnx models/ # also write the serving model
    python analysis/run_dl_case_study.py --smoke-test          # 1-2 min pipeline check, no dataset

The 80/20 split reproduces the one in run_neu_case_study.py (same seed, same stratification), so
accuracy / macro-F1 / review-queue numbers are directly comparable. A further 10% of the training
part is held out to fit the calibration temperature and pick the best epoch; the test rows are
never used for either.

Outputs: docs/data/neu-cls-dl/*.json, analysis/.cache/neu_dl_logits.npz (test logits) and, with
--gradcam-samples N, Grad-CAM overlays under analysis/.cache/gradcam/. --smoke-test trains on
synthetic images and writes everything to a temporary directory; its numbers mean nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
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
SMOKE_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def load_images(paths, size: int = IMAGE_SIZE) -> np.ndarray:
    """Grayscale uint8 array of shape (N, size, size)."""
    out = np.empty((len(paths), size, size), dtype=np.uint8)
    for i, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        out[i] = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return out


def synthetic_dataset(per_class: int, size: int, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Learnable stand-in for NEU-CLS: each class is a differently oriented / spaced stripe texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    images, labels = [], []
    for cls in range(len(SMOKE_CLASSES)):
        angle, freq = cls * np.pi / len(SMOKE_CLASSES), 4 + 3 * cls
        pattern = np.sin(2 * np.pi * freq * (xx * np.cos(angle) + yy * np.sin(angle)))
        for _ in range(per_class):
            phase = rng.uniform(0, 2 * np.pi)
            img = 0.5 + 0.25 * np.roll(pattern, int(phase * size / 6), axis=1) + rng.normal(0, 0.08, (size, size))
            images.append(np.clip(img * 255, 0, 255).astype(np.uint8))
            labels.append(cls)
    return np.stack(images), np.array(labels)


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


def cam_layer(model, arch: str):
    """Last convolutional block, the usual Grad-CAM target."""
    return model.layer4[-1] if arch == "resnet18" else model.features[-1]


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


def export_onnx(model, out_dir: Path, class_names, temperature: float, arch: str, size: int) -> Path:
    """Serving model: float32 (B,1,size,size) in [0,1] -> logits; normalisation is baked in."""
    import torch
    import torch.nn as nn

    class Serving(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        def forward(self, x):
            return self.inner((x.repeat(1, 3, 1, 1) - self.mean) / self.std)

    out_dir.mkdir(parents=True, exist_ok=True)
    wrapped = Serving(model.cpu().eval()).eval()
    path = out_dir / "model.onnx"
    torch.onnx.export(
        wrapped, torch.zeros(1, 1, size, size), str(path), input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False,
    )
    (out_dir / "meta.json").write_text(json.dumps({
        "architecture": arch, "image_size": size, "class_names": list(class_names), "temperature": temperature,
    }, indent=2) + "\n", encoding="utf-8")
    return path


def grad_cam(model, arch: str, images_u8: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM heat maps in [0,1] for the predicted class. Returns (maps (N,H,W), predictions (N,))."""
    import torch.nn.functional as F

    model.eval()
    store: dict = {}
    layer = cam_layer(model, arch)

    def forward_hook(_, __, output):
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))

    handle = layer.register_forward_hook(forward_hook)
    maps, preds = [], []
    try:
        for image in images_u8:
            model.zero_grad(set_to_none=True)
            logits = model(to_input(image[None], device))
            cls = int(logits.argmax(1))
            logits[0, cls].backward()
            weights = store["grad"].mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=image.shape, mode="bilinear", align_corners=False)[0, 0]
            cam = cam - cam.min()
            maps.append((cam / cam.max().clamp_min(1e-8)).detach().cpu().numpy())
            preds.append(cls)
    finally:
        handle.remove()
    return np.stack(maps), np.array(preds)


def overlay(image_u8: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR), 0.55, heat, 0.45, 0)


def split_train_val(indices, labels):
    return train_test_split(indices, test_size=0.10, random_state=RANDOM_SEED, stratify=labels[indices])


def write_json(out_dir: Path, name: str, payload) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--no-pretrained", action="store_true", help="random init instead of ImageNet weights")
    parser.add_argument("--cv-folds", type=int, default=0, help="also run K-fold CV over all 1800 images (slow)")
    parser.add_argument("--export-onnx", type=Path, metavar="DIR", help="write model.onnx + meta.json for /api/classify")
    parser.add_argument("--gradcam-samples", type=int, default=0, help="save Grad-CAM overlays for N test images")
    parser.add_argument("--smoke-test", action="store_true", help="tiny synthetic run to check the pipeline; no dataset")
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + ("" if device.type == "cuda" else "  (no GPU found - this will be slow)"))

    if args.smoke_test:
        args.no_pretrained, args.epochs, args.image_size = True, min(args.epochs, 3), min(args.image_size, 64)
        images, labels = synthetic_dataset(per_class=40, size=args.image_size)
        class_names = list(SMOKE_CLASSES)
        scratch = Path(tempfile.mkdtemp(prefix="neu_dl_smoke_"))
        out_dir, cache_dir = scratch / "data", scratch / "cache"
        baseline = {"accuracy": None, "macro_f1": None}
        print(f"Smoke test on synthetic images; outputs in {scratch}")
    else:
        ensure_dataset()
        paths = list(iter_image_paths())
        labels_text = np.array([infer_label(p) for p in paths])
        class_names = sorted(set(labels_text.tolist()))
        labels = np.array([class_names.index(name) for name in labels_text])
        images = load_images(paths, args.image_size)
        out_dir, cache_dir = OUTPUT_DIR, CACHE_DIR
        baseline = next(m for m in json.loads(BASELINE_FILE.read_text()) if m["model"] == "random_forest")

    # Same outer split as run_neu_case_study.py (indices depend only on the labels and the seed).
    all_idx = np.arange(len(labels))
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

    summary = {
        "smoke_test": args.smoke_test,
        "architecture": args.arch, "pretrained": not args.no_pretrained, "epochs": args.epochs,
        "image_size": args.image_size, "device": device.type, "train_seconds": round(train_seconds, 1), "seed": RANDOM_SEED,
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

    write_json(out_dir, "summary.json", summary)
    write_json(out_dir, "reliability.json", {"before": reliability_bins(raw_p, y_test), "after": reliability_bins(cal_p, y_test)})
    write_json(out_dir, "review-queue.json", {"review_budgets": build_review_queue(cal_p, preds, y_test)})
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_dir / "neu_dl_logits.npz", test_logits=test_logits, test_idx=test_idx, targets=y_test,
             temperature=temperature, class_names=np.array(class_names))
    torch.save(model.state_dict(), cache_dir / f"neu_dl_{args.arch}.pt")

    if args.gradcam_samples > 0:
        # Prefer misclassified rows: that is where the heat map explains the most.
        order = np.argsort(preds == y_test, kind="stable")[: args.gradcam_samples]
        maps, cam_preds = grad_cam(model.to(device), args.arch, images[test_idx][order], device)
        cam_dir = cache_dir / "gradcam"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for rank, (row, cam, pred) in enumerate(zip(order, maps, cam_preds, strict=True)):
            name = f"{rank:02d}_true-{class_names[y_test[row]]}_pred-{class_names[pred]}.png"
            cv2.imwrite(str(cam_dir / name), overlay(images[test_idx][row], cam))
        print(f"Grad-CAM overlays: {cam_dir}")

    if args.export_onnx or args.smoke_test:
        target = args.export_onnx or (out_dir.parent / "onnx")
        print(f"Exported serving model: {export_onnx(model, target, class_names, temperature, args.arch, args.image_size)}")

    print(json.dumps(summary, indent=2))
    if baseline["accuracy"] is not None:
        print(f"\nRandom Forest on the same holdout: acc={baseline['accuracy']}  macro_f1={baseline['macro_f1']}")
    print(f"{args.arch}: acc={acc:.4f} {tuple(round(v, 4) for v in acc_ci)}  macro_f1={f1:.4f}")


if __name__ == "__main__":
    main()
