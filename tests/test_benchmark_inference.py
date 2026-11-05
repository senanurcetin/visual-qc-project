"""Checks for cnn/benchmark_inference.py that need neither a GPU nor the NEU-CLS download."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cnn"))
import benchmark_inference as bench  # noqa: E402

HAVE_ORT = all(importlib.util.find_spec(m) for m in ("onnx", "onnxruntime"))


def write_images(root, value=128, per_class=2, size=(200, 200)):
    for c in bench.CLASSES:
        d = Path(root) / "images" / c
        d.mkdir(parents=True)
        for i in range(per_class):
            Image.fromarray(np.full(size[::-1], value, dtype=np.uint8)).save(d / f"{c}_{i + 1}.jpg")


class PreprocessingTest(unittest.TestCase):
    def test_matches_training_transforms(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_images(tmp, value=128)
            x, y = bench.load_images(tmp)
        self.assertEqual(x.shape, (12, 3, 224, 224))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(sorted(set(y.tolist())), list(range(6)))
        # A uniform grey image becomes (v / 255 - mean) / std per channel, as in train_cnn.py.
        expected = (128 / 255.0 - bench.MEAN.ravel()) / bench.STD.ravel()
        np.testing.assert_allclose(x[0, :, 100, 100], expected, rtol=1e-5)

    def test_ignores_unknown_folders_and_limits_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_images(tmp, per_class=3)
            other = Path(tmp) / "images" / "not_a_class"
            other.mkdir()
            Image.new("L", (10, 10)).save(other / "x_1.jpg")
            x, _ = bench.load_images(tmp, max_images=5)
        self.assertEqual(len(x), 5)


@unittest.skipUnless(HAVE_ORT, "onnx / onnxruntime not installed")
class CpuEndToEndTest(unittest.TestCase):
    def test_cpu_backend_writes_report(self):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        # Tiny stand-in classifier: global average pool -> 3x6 linear layer, same I/O names as the release.
        w = numpy_helper.from_array(np.eye(3, 6, dtype=np.float32), "w")
        graph = helper.make_graph(
            [helper.make_node("GlobalAveragePool", ["image"], ["p"]),
             helper.make_node("Flatten", ["p"], ["f"]),
             helper.make_node("MatMul", ["f", "w"], ["logits"])],
            "tiny",
            [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 224, 224])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 6])],
            initializer=[w],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            onnx.save(model, tmp / "tiny.onnx")
            write_images(tmp / "data")
            subprocess.run([sys.executable, str(ROOT / "cnn" / "benchmark_inference.py"), "--onnx", str(tmp / "tiny.onnx"),
                            "--data", str(tmp / "data"), "--out", str(tmp / "out"), "--batch", "4", "--runs", "5",
                            "--only", "cpu_fp32"], check=True, capture_output=True, text=True)
            report = json.loads((tmp / "out" / "benchmark.json").read_text())
        self.assertEqual(report["meta"]["n_images"], 12)
        cpu = report["results"]["cpu_fp32"]
        self.assertEqual(cpu["provider"], "CPUExecutionProvider")
        self.assertEqual(cpu["top1_agreement_with_reference"], 1.0)
        self.assertEqual(cpu["max_abs_logit_diff_vs_reference"], 0.0)
        for key in ("latency_batch1_ms_p50", "throughput_batch4_img_per_s", "session_init_s", "accuracy_all_images"):
            self.assertIn(key, cpu)


class ReadmeMatchesBenchmarkJsonTest(unittest.TestCase):
    """The TensorRT table in README.md must quote docs/data/tensorrt-benchmark/benchmark.json."""

    def test_table_numbers_come_from_the_json(self):
        results = json.loads((ROOT / "docs/data/tensorrt-benchmark/benchmark.json").read_text())["results"]
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        section = readme.split("### Inference acceleration with TensorRT", 1)[1].split("\n## ", 1)[0]
        for name, r in results.items():
            self.assertIn(f"{r['latency_batch1_ms_p50']} / {r['latency_batch1_ms_p95']} ms", section, name)
            self.assertIn(f"{r['throughput_batch32_img_per_s']} img/s", section, name)
            self.assertEqual(r["top1_agreement_with_reference"], 1.0, name)


if __name__ == "__main__":
    unittest.main()
