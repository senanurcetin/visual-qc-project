import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from analysis import update_docs_from_dl as ud

RF = {"model": "random_forest", "accuracy": 0.9389, "macro_f1": 0.9392}


def summary(**overrides):
    base = {
        "smoke_test": False, "architecture": "resnet18", "pretrained": True, "epochs": 15, "device": "cuda",
        "train_seconds": 312.4, "seed": 42, "split": {"fit": 1296, "val": 144, "test": 360},
        "test": {"accuracy": 0.9611, "accuracy_ci95": [0.9361, 0.9806], "macro_f1": 0.9608, "macro_f1_ci95": [0.9352, 0.9801]},
        "calibration": {"temperature": 1.412, "ece_before": 0.031, "ece_after": 0.012},
    }
    base.update(overrides)
    return base


class RenderTests(unittest.TestCase):
    def test_table_has_both_models_with_intervals_and_calibration(self):
        block = ud.render_block(summary(), RF)
        self.assertIn("| Random Forest (HOG + Gabor + grid) | 0.939 | 0.939 | n/a |", block)
        self.assertIn("| resnet18 | 0.961 (0.936–0.981) | 0.961 (0.935–0.980) | 0.031 → 0.012 |", block)
        self.assertIn("360-row holdout", block)
        self.assertIn("15 epochs on cuda in 312 s", block)
        self.assertIn("temperature 1.412", block)

    def test_says_when_the_random_forest_is_inside_the_interval(self):
        self.assertIn("not clearly separated", ud.render_block(summary(), RF))

    def test_says_when_the_random_forest_is_outside_the_interval(self):
        s = summary()
        s["test"]["accuracy_ci95"] = [0.95, 0.99]
        self.assertIn("lies outside the CNN's 95% interval (below it)", ud.render_block(s, RF))

    def test_cross_validation_and_no_pretraining_are_reported(self):
        s = summary(pretrained=False, cross_validation={"folds": 5, "accuracy_mean": 0.951, "accuracy_std": 0.012})
        block = ud.render_block(s, RF)
        self.assertIn("resnet18 (no pretraining)", block)
        self.assertIn("5-fold cross-validation accuracy: 0.951 ± 0.012", block)

    def test_refuses_a_smoke_test_summary(self):
        with self.assertRaises(ud.SmokeSummary):
            ud.render_block(summary(smoke_test=True), RF)


class ApplyTests(unittest.TestCase):
    TEMPLATE = f"intro\n\n{ud.START}\nplaceholder\n{ud.END}\n\noutro\n"

    def test_replaces_only_the_marked_block_and_is_idempotent(self):
        once = ud.apply(self.TEMPLATE, "NEW")
        self.assertEqual(once, f"intro\n\n{ud.START}\nNEW\n{ud.END}\n\noutro\n")
        self.assertEqual(ud.apply(once, "NEW"), once)

    def test_requires_exactly_one_marker_pair(self):
        with self.assertRaises(ValueError):
            ud.apply("no markers here", "x")
        with self.assertRaises(ValueError):
            ud.apply(self.TEMPLATE * 2, "x")


class CommandTests(unittest.TestCase):
    def run_main(self, summary_dict):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = {"README": tmp / "README.md", "SUMMARY": tmp / "summary.json", "BASELINE": tmp / "base.json"}
            paths["README"].write_text(ApplyTests.TEMPLATE, encoding="utf-8")
            paths["BASELINE"].write_text(json.dumps([RF]), encoding="utf-8")
            if summary_dict is not None:
                paths["SUMMARY"].write_text(json.dumps(summary_dict), encoding="utf-8")
            saved = {k: getattr(ud, k) for k in paths} | {"ROOT": ud.ROOT}
            for k, v in paths.items():
                setattr(ud, k, v)
            ud.ROOT = tmp
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    code = ud.main()
                return code, paths["README"].read_text(encoding="utf-8")
            finally:
                for k, v in saved.items():
                    setattr(ud, k, v)

    def test_updates_the_readme_from_a_real_summary(self):
        code, readme = self.run_main(copy.deepcopy(summary()))
        self.assertEqual(code, 0)
        self.assertIn("| resnet18 |", readme)
        self.assertNotIn("placeholder", readme)

    def test_refuses_smoke_results_and_leaves_the_readme_alone(self):
        code, readme = self.run_main(summary(smoke_test=True))
        self.assertEqual(code, 2)
        self.assertIn("placeholder", readme)

    def test_fails_clearly_without_a_summary(self):
        code, readme = self.run_main(None)
        self.assertEqual(code, 1)
        self.assertIn("placeholder", readme)


if __name__ == "__main__":
    unittest.main()
