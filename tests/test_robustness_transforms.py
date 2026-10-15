"""Data-free contracts for deterministic image perturbations."""
import unittest

import numpy as np

from analysis.run_perturbation_robustness import MODES, perturb


class RobustnessTransformTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(256, dtype=np.uint8).reshape(16, 16)

    def test_noise_repeatable_per_image(self):
        a = perturb(self.image, "gaussian_noise_sigma_12", 44)
        b = perturb(self.image, "gaussian_noise_sigma_12", 44)
        c = perturb(self.image, "gaussian_noise_sigma_12", 45)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_all_transforms_preserve_shape_type_and_valid_range(self):
        for name in MODES:
            with self.subTest(name=name):
                out = perturb(self.image, name, 42)
                self.assertEqual(out.shape, self.image.shape)
                self.assertEqual(out.dtype, np.uint8)
                self.assertGreaterEqual(int(out.min()), 0)
                self.assertLessEqual(int(out.max()), 255)

    def test_clean_returns_copy(self):
        out = perturb(self.image, "clean", 42)
        np.testing.assert_array_equal(out, self.image)
        self.assertIsNot(out, self.image)

    def test_unknown_perturbation_fails_closed(self):
        with self.assertRaises(ValueError):
            perturb(self.image, "adversarial_magic", 42)


if __name__ == "__main__":
    unittest.main()
