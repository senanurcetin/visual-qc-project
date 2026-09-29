"""End-to-end check of the CNN pipeline on synthetic images. Skipped when torch/onnxruntime are missing.

Runs `run_dl_case_study.py --smoke-test` (training, calibration, bootstrap, Grad-CAM, ONNX export)
and then serves the exported model through /api/classify with the real onnxruntime.
"""
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HAVE_STACK = all(importlib.util.find_spec(m) for m in ("torch", "torchvision", "onnxruntime", "onnx"))


@unittest.skipUnless(HAVE_STACK, "needs torch, torchvision, onnx and onnxruntime")
class SmokePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run = subprocess.run(
            [sys.executable, str(ROOT / "analysis" / "run_dl_case_study.py"), "--smoke-test", "--gradcam-samples", "3"],
            capture_output=True, text=True, timeout=600, cwd=ROOT,
        )
        cls.run_result = run
        found = re.search(r"outputs in (\S+)", run.stdout)
        cls.scratch = Path(found.group(1)) if found else None

    def test_script_succeeds_and_writes_expected_outputs(self):
        self.assertEqual(self.run_result.returncode, 0, self.run_result.stderr[-2000:])
        data = self.scratch / "data"
        summary = json.loads((data / "summary.json").read_text())
        self.assertTrue(summary["smoke_test"])
        lo, hi = summary["test"]["accuracy_ci95"]
        self.assertLessEqual(lo, summary["test"]["accuracy"])
        self.assertGreaterEqual(hi, summary["test"]["accuracy"])
        self.assertGreater(summary["calibration"]["temperature"], 0)
        self.assertEqual(len(json.loads((data / "reliability.json").read_text())["after"]), 10)
        self.assertEqual(len(json.loads((data / "review-queue.json").read_text())["review_budgets"]), 3)
        self.assertEqual(len(list((self.scratch / "cache" / "gradcam").glob("*.png"))), 3)

    def test_training_learns_the_synthetic_classes(self):
        summary = json.loads((self.scratch / "data" / "summary.json").read_text())
        self.assertGreater(summary["test"]["accuracy"], 0.5)  # chance is 1/6

    def test_onnx_matches_pytorch(self):
        import onnxruntime
        import torch

        sys.path.insert(0, str(ROOT))
        from analysis.run_dl_case_study import build_model, synthetic_dataset, to_input

        onnx_dir = self.scratch / "onnx"
        meta = json.loads((onnx_dir / "meta.json").read_text())
        model = build_model("resnet18", len(meta["class_names"]), pretrained=False)
        model.load_state_dict(torch.load(self.scratch / "cache" / "neu_dl_resnet18.pt"))
        model.eval()
        images, _ = synthetic_dataset(1, meta["image_size"])
        with torch.no_grad():
            expected = model(to_input(images, torch.device("cpu"))).numpy()
        session = onnxruntime.InferenceSession(str(onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        got = session.run(["logits"], {"image": (images.astype(np.float32) / 255.0)[:, None]})[0]
        np.testing.assert_allclose(got, expected, atol=1e-4)

    def test_exported_model_is_served_by_the_api(self):
        import main

        old = os.environ.get("QC_MODEL_DIR")
        os.environ["QC_MODEL_DIR"] = str(self.scratch / "onnx")
        main.app.extensions.pop("classifier", None)
        try:
            ok, png = cv2.imencode(".png", np.random.default_rng(0).integers(0, 255, (100, 100), dtype=np.uint8))
            response = main.app.test_client().post(
                "/api/classify", data={"image": (io.BytesIO(png.tobytes()), "x.png")}, content_type="multipart/form-data"
            )
            self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
            body = response.get_json()
            self.assertIn(body["label"], json.loads((self.scratch / "onnx" / "meta.json").read_text())["class_names"])
            self.assertAlmostEqual(sum(body["probabilities"].values()), 1.0, places=2)
        finally:
            main.app.extensions.pop("classifier", None)
            if old is None:
                os.environ.pop("QC_MODEL_DIR", None)
            else:
                os.environ["QC_MODEL_DIR"] = old


if __name__ == "__main__":
    unittest.main()
