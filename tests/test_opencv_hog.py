import unittest

import cv2
import numpy as np


class OpenCvHogTests(unittest.TestCase):
    """The HOG feature pipeline needs cv2.HOGDescriptor; OpenCV 5.0 removed it.

    This fails with an explanation instead of an obscure ImportError deep in the analysis code
    when a dependency bump pulls in a release without it.
    """

    def test_hog_descriptor_is_available_and_has_the_expected_length(self):
        self.assertTrue(
            hasattr(cv2, "HOGDescriptor"),
            f"OpenCV {cv2.__version__} has no cv2.HOGDescriptor. Pin opencv-python-headless<5: the "
            "features behind every reported Random Forest number are computed with it.",
        )
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        descriptor = hog.compute(np.zeros((64, 64), dtype=np.uint8))
        self.assertEqual(descriptor.size, 1764)  # matches the "HOG (1764)" documented in summary.json

    def test_requirements_keep_opencv_below_five(self):
        from pathlib import Path

        pins = [line for line in Path(__file__).resolve().parents[1].joinpath("requirements.txt").read_text().splitlines()
                if line.startswith("opencv-python-headless")]
        self.assertEqual(len(pins), 1)
        self.assertTrue(pins[0].split("==")[1].startswith("4."), pins[0])


if __name__ == "__main__":
    unittest.main()
