"""NEU-CLS steel defects: fine-tuned ResNet-18 vs the project's handcrafted-feature Random Forest.

Two splits, same model and budget:
  random  deterministic 80/20 stratified holdout (seed 42) — the project's protocol (RF: 0.939)
  blocked each class's images 241-300 held out — NEU images are crops of the same strips, and
          neighbouring indices can be near-duplicates, which a random split puts on both sides
Exports the random-split model to ONNX (+ labels.json) for the Hugging Face release.

  python train_cnn.py --data <dir with class folders> --out <dir>
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torchvision import models, transforms

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SEED, SIZE = 42, 224


def load(root):
    paths, labels, idx = [], [], []
    for p in sorted(Path(root).rglob("*.jpg")):
        c = p.parent.name
        if c in CLASSES:
            paths.append(p); labels.append(CLASSES.index(c)); idx.append(int(re.findall(r"(\d+)", p.stem)[-1]))
    return np.array(paths), np.array(labels), np.array(idx)


class DS(torch.utils.data.Dataset):
    def __init__(self, paths, labels, train):
        self.p, self.y = paths, labels
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(10)] if train else []
        self.t = transforms.Compose([transforms.Grayscale(3), transforms.Resize((SIZE, SIZE)), *aug, transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.t(Image.open(self.p[i])), self.y[i]


def fit(tr_p, tr_y, te_p, te_y, device, epochs=12):
    torch.manual_seed(SEED)
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m = m.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dl = torch.utils.data.DataLoader(DS(tr_p, tr_y, True), batch_size=32, shuffle=True, num_workers=0)
    for _ep in range(epochs):
        m.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); nn.functional.cross_entropy(m(x), y, label_smoothing=0.05).backward(); opt.step()
        sched.step()
    m.eval()
    preds, probs = [], []
    with torch.no_grad():
        for x, _ in torch.utils.data.DataLoader(DS(te_p, te_y, False), batch_size=64):
            p = torch.softmax(m(x.to(device)), 1).cpu().numpy()
            probs.append(p); preds.append(p.argmax(1))
    return m, np.concatenate(preds), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths, y, idx = load(a.data)
    print(f"{len(paths)} images, {np.bincount(y).tolist()} per class")
    results = {}
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    blocked_te = idx > 240
    splits = {"random 80/20 (seed 42)": (tr, te), "blocked (indices 241-300 held out)": (np.flatnonzero(~blocked_te), np.flatnonzero(blocked_te))}
    for name, (tr_i, te_i) in splits.items():
        t0 = time.time()
        m, pred, prob = fit(paths[tr_i], y[tr_i], paths[te_i], y[te_i], device)
        acc, f1 = accuracy_score(y[te_i], pred), f1_score(y[te_i], pred, average="macro")
        results[name] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4), "n_test": int(len(te_i)),
                         "report": classification_report(y[te_i], pred, target_names=CLASSES, output_dict=True, digits=4)}
        print(f"{name}: accuracy {acc:.4f} | macro F1 {f1:.4f} | {len(te_i)} test images ({time.time() - t0:.0f}s)", flush=True)
        if name.startswith("random"):
            dummy = torch.randn(1, 3, SIZE, SIZE, device=device)
            torch.onnx.export(m, dummy, out / "neu_resnet18.onnx", input_names=["image"], output_names=["logits"],
                              dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
            torch.save(m.state_dict(), out / "neu_resnet18.pt")
    (out / "labels.json").write_text(json.dumps({"classes": CLASSES, "input_size": SIZE, "grayscale_to_rgb": True,
                                                 "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, indent=1))
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
