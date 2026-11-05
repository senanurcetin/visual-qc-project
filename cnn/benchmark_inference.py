"""Benchmark the released NEU-CLS ResNet-18 ONNX model on CPU, CUDA and TensorRT (FP32 / FP16).

Uses ONNX Runtime execution providers, so every backend runs the same ONNX file:

    pip install "onnxruntime-gpu[cuda,cudnn]" "tensorrt-cu13>=10,<11" numpy pillow
    python cnn/benchmark_inference.py --onnx neu_resnet18.onnx --data <NEU-CLS root> --out results/trt

For each backend it reports
  * latency at batch 1 (p50 / p95, ms) and throughput at --batch (images/s), after warm-up;
  * agreement with the CPU FP32 reference on every image: same top-1 class, max |logit difference|;
  * top-1 accuracy over all images found under --data.

Accuracy here is over the whole folder (training images included), so it is a parity check
between backends, not a held-out generalisation estimate. `--max-images` limits the image count.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_images(root, max_images=None):
    """Return (N, 3, 224, 224) float32 inputs and labels, preprocessed exactly as in train_cnn.py."""
    paths = sorted(p for p in Path(root).rglob("*.jpg") if p.parent.name in CLASSES)
    if max_images:
        paths = paths[:max_images]
    if not paths:
        sys.exit(f"no NEU-CLS images found under {root}")
    xs, ys = [], []
    for p in paths:
        # Grayscale(3) -> Resize((224, 224)) -> ToTensor -> Normalize, matching the training transforms.
        img = Image.open(p).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
        xs.append((np.repeat(a[None], 3, axis=0) - MEAN) / STD)
        ys.append(CLASSES.index(p.parent.name))
    return np.stack(xs).astype(np.float32), np.array(ys)


def add_tensorrt_libs_to_path():
    """The tensorrt wheel ships its DLLs/.so files in the tensorrt_libs package; ORT looks them up on PATH."""
    try:
        import tensorrt_libs
    except ImportError:
        return
    libs = str(Path(tensorrt_libs.__file__).parent)
    os.environ["PATH"] = libs + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(libs)


def backends(cache_dir, batch):
    shapes = f"image:1x3x{SIZE}x{SIZE}"
    trt_common = {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache_dir),
        "trt_profile_min_shapes": shapes,
        "trt_profile_opt_shapes": f"image:{batch}x3x{SIZE}x{SIZE}",
        "trt_profile_max_shapes": f"image:{batch}x3x{SIZE}x{SIZE}",
    }
    return {
        "cpu_fp32": [("CPUExecutionProvider", {})],
        "cuda_fp32": [("CUDAExecutionProvider", {}), "CPUExecutionProvider"],
        "tensorrt_fp32": [("TensorrtExecutionProvider", {**trt_common, "trt_fp16_enable": False}),
                          "CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt_fp16": [("TensorrtExecutionProvider", {**trt_common, "trt_fp16_enable": True}),
                          "CUDAExecutionProvider", "CPUExecutionProvider"],
    }


def run_all(sess, x, batch):
    return np.concatenate([sess.run(None, {"image": x[i:i + batch]})[0] for i in range(0, len(x), batch)])


def time_it(sess, x, runs):
    for _ in range(20):
        sess.run(None, {"image": x})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"image": x})
        t.append(time.perf_counter() - t0)
    return np.array(t) * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--data", required=True, help="folder containing <class>/*.jpg (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--only", nargs="*", help="subset of backends to run")
    a = ap.parse_args()

    add_tensorrt_libs_to_path()
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()  # load CUDA / cuDNN from the nvidia-* pip wheels, if installed

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    x, y = load_images(a.data, a.max_images)
    print(f"{len(x)} images, batch {a.batch}, onnxruntime {ort.__version__}, providers {ort.get_available_providers()}")

    results, ref = {}, None
    for name, providers in backends(out / "trt_cache", a.batch).items():
        if a.only and name not in a.only:
            continue
        t_init = time.perf_counter()
        sess = ort.InferenceSession(a.onnx, providers=providers)  # TensorRT builds its engine here
        init_s = time.perf_counter() - t_init
        used = sess.get_providers()[0]
        wanted = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        if used != wanted:
            print(f"{name}: skipped, {wanted} not available (session fell back to {used})")
            continue
        t0 = time.perf_counter()
        logits = run_all(sess, x, a.batch)
        first_pass_s = time.perf_counter() - t0
        lat1 = time_it(sess, x[:1], a.runs)
        latb = time_it(sess, x[:a.batch], max(20, a.runs // 4))
        pred = logits.argmax(1)
        if ref is None:
            ref = (name, logits, pred)
        r = {
            "provider": used,
            "latency_batch1_ms_p50": round(float(np.percentile(lat1, 50)), 3),
            "latency_batch1_ms_p95": round(float(np.percentile(lat1, 95)), 3),
            f"throughput_batch{a.batch}_img_per_s": round(a.batch / (float(np.median(latb)) / 1000.0), 1),
            "session_init_s": round(init_s, 2),  # includes the TensorRT engine build unless cached
            "first_pass_all_images_s": round(first_pass_s, 2),
            "accuracy_all_images": round(float((pred == y).mean()), 4),
            "reference": ref[0],
            "top1_agreement_with_reference": round(float((pred == ref[2]).mean()), 4),
            "max_abs_logit_diff_vs_reference": round(float(np.abs(logits - ref[1]).max()), 5),
        }
        results[name] = r
        print(f"{name}: {json.dumps(r)}", flush=True)

    meta = {"n_images": int(len(x)), "batch": a.batch, "runs": a.runs, "onnx": Path(a.onnx).name,
            "onnxruntime": ort.__version__, "python": platform.python_version(), "platform": platform.platform()}
    try:
        import tensorrt
        meta["tensorrt"] = tensorrt.__version__
    except ImportError:
        pass
    (out / "benchmark.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print("wrote", out / "benchmark.json")


if __name__ == "__main__":
    main()
