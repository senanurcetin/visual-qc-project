import unittest

import numpy as np

from analysis.dl.calibration import (
    expected_calibration_error,
    fit_temperature,
    nll,
    reliability_bins,
    softmax,
)
from analysis.dl.stats import bootstrap_ci


def overconfident_logits(n=4000, classes=6, sharpen=3.0, seed=0):
    """Labels drawn from p; logits = sharpen * log p, so the model is systematically overconfident."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(classes), size=n)
    targets = np.array([rng.choice(classes, p=row) for row in p])
    return sharpen * np.log(p), targets


class CalibrationTests(unittest.TestCase):
    def test_softmax_rows_sum_to_one_and_temperature_softens(self):
        logits = np.array([[4.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        p = softmax(logits)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertLess(softmax(logits, 3.0)[0].max(), p[0].max())

    def test_temperature_scaling_recovers_the_sharpening_factor(self):
        logits, targets = overconfident_logits(sharpen=3.0)
        t = fit_temperature(logits, targets)
        self.assertAlmostEqual(t, 3.0, delta=0.4)
        self.assertLess(nll(logits, targets, t), nll(logits, targets, 1.0))

    def test_temperature_scaling_reduces_ece(self):
        logits, targets = overconfident_logits()
        before = expected_calibration_error(softmax(logits), targets)
        after = expected_calibration_error(softmax(logits, fit_temperature(logits, targets)), targets)
        self.assertLess(after, before / 2)

    def test_well_calibrated_model_keeps_temperature_near_one(self):
        logits, targets = overconfident_logits(sharpen=1.0)
        self.assertAlmostEqual(fit_temperature(logits, targets), 1.0, delta=0.15)

    def test_reliability_bins_cover_every_sample(self):
        logits, targets = overconfident_logits(n=500)
        bins = reliability_bins(softmax(logits), targets, n_bins=10)
        self.assertEqual(sum(b["count"] for b in bins), 500)


class BootstrapTests(unittest.TestCase):
    @staticmethod
    def accuracy(t, p):
        return float((t == p).mean())

    def test_interval_brackets_the_point_estimate_and_shrinks_with_n(self):
        rng = np.random.default_rng(1)

        def make(n):
            t = rng.integers(0, 6, n)
            p = np.where(rng.random(n) < 0.9, t, (t + 1) % 6)
            return t, p

        widths = []
        for n in (100, 1600):
            t, p = make(n)
            lo, hi = bootstrap_ci(t, p, self.accuracy, n_resamples=500)
            self.assertLessEqual(lo, self.accuracy(t, p))
            self.assertGreaterEqual(hi, self.accuracy(t, p))
            widths.append(hi - lo)
        self.assertLess(widths[1], widths[0])

    def test_is_deterministic_for_a_seed(self):
        t = np.arange(60) % 6
        p = t.copy()
        p[:6] = (p[:6] + 1) % 6
        a = bootstrap_ci(t, p, self.accuracy, n_resamples=200, seed=7)
        self.assertEqual(a, bootstrap_ci(t, p, self.accuracy, n_resamples=200, seed=7))


if __name__ == "__main__":
    unittest.main()
