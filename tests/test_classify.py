import io
import os
import tempfile
import unittest

import cv2
import numpy as np

import classify
import main

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


class FakeSession:
    """Stands in for onnxruntime.InferenceSession."""

    def __init__(self, logits):
        self.logits, self.seen = np.array([logits], dtype=np.float32), None

    def run(self, _outputs, feeds):
        self.seen = feeds["image"]
        return [self.logits]


def png_bytes(size=40, value=128):
    ok, buf = cv2.imencode(".png", np.full((size, size, 3), value, dtype=np.uint8))
    assert ok
    return buf.tobytes()


class ClassifyEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = main.app.test_client()
        main.app.extensions.pop("classifier", None)
        self._old = os.environ.get("QC_MODEL_DIR")
        self.empty = tempfile.TemporaryDirectory()
        os.environ["QC_MODEL_DIR"] = self.empty.name

    def tearDown(self):
        main.app.extensions.pop("classifier", None)
        if self._old is None:
            os.environ.pop("QC_MODEL_DIR", None)
        else:
            os.environ["QC_MODEL_DIR"] = self._old
        self.empty.cleanup()

    def install(self, logits, temperature=1.0, size=32):
        session = FakeSession(logits)
        main.app.extensions["classifier"] = {
            "session": session, "meta": {"class_names": CLASSES, "image_size": size, "temperature": temperature},
        }
        return session

    def post(self, data, name="a.png"):
        return self.client.post("/api/classify", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")

    def test_503_with_a_reason_when_no_model_is_installed(self):
        response = self.post(png_bytes())
        self.assertEqual(response.status_code, 503)
        self.assertIn("no model found", response.get_json()["reason"])

    def test_requires_an_image_field(self):
        self.install([0, 0, 0, 0, 0, 0])
        self.assertEqual(self.client.post("/api/classify").status_code, 400)

    def test_rejects_data_that_is_not_an_image(self):
        self.install([0, 0, 0, 0, 0, 0])
        self.assertEqual(self.post(b"definitely not an image").status_code, 400)

    def test_rejects_oversized_uploads(self):
        self.install([0, 0, 0, 0, 0, 0])
        self.assertEqual(self.post(b"\0" * (classify.MAX_UPLOAD_BYTES + 10)).status_code, 413)

    def test_confident_prediction(self):
        session = self.install([0, 0, 0, 0, 9, 0], size=32)
        body = self.post(png_bytes()).get_json()
        self.assertEqual(body["label"], "rolled-in_scale")
        self.assertGreater(body["confidence"], 0.99)
        self.assertFalse(body["needs_review"])
        self.assertEqual(body["ranking"][0]["label"], "rolled-in_scale")
        self.assertEqual([r["probability"] for r in body["ranking"]], sorted((r["probability"] for r in body["ranking"]), reverse=True))
        self.assertAlmostEqual(sum(body["probabilities"].values()), 1.0, places=2)
        self.assertEqual(session.seen.shape, (1, 1, 32, 32))  # resized to the model's input size
        self.assertEqual(session.seen.dtype, np.float32)
        self.assertTrue(0.0 <= session.seen.min() and session.seen.max() <= 1.0)

    def test_uncertain_prediction_is_routed_to_review(self):
        self.install([0.1, 0.0, 0.05, 0.0, 0.1, 0.0])
        body = self.post(png_bytes()).get_json()
        self.assertTrue(body["needs_review"])
        self.assertGreater(body["entropy_bits"], 2.4)

    def test_temperature_softens_the_probabilities(self):
        self.install([0, 0, 0, 0, 3, 0], temperature=1.0)
        sharp = self.post(png_bytes()).get_json()["confidence"]
        self.install([0, 0, 0, 0, 3, 0], temperature=3.0)
        self.assertLess(self.post(png_bytes()).get_json()["confidence"], sharp)


if __name__ == "__main__":
    unittest.main()
