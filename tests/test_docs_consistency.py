"""Guards the headline numbers in README.md / docs/hiring-summary.md against the JSON artifacts."""

import json
import unittest
from pathlib import Path

from analysis import update_docs_from_dl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "neu-cls-case-study"


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


class DocsMatchArtifactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.hiring = (ROOT / "docs" / "hiring-summary.md").read_text(encoding="utf-8")
        cls.rf = next(m for m in load("benchmark-comparison.json") if m["model"] == "random_forest")
        cls.budgets = {b["review_fraction"]: b for b in load("review-queue.json")["review_budgets"]}

    def test_random_forest_row_in_readme(self):
        row = f"| **{self.rf['accuracy']:.3f}** | **{self.rf['macro_f1']:.3f}** | **{self.rf['macro_precision']:.3f}** |"
        self.assertIn(row, self.readme)

    def test_random_forest_headline_in_hiring_summary(self):
        for key in ("accuracy", "macro_f1", "macro_precision"):
            self.assertIn(f"**{self.rf[key]:.4f}**", self.hiring, key)

    def test_review_queue_20_percent(self):
        b = self.budgets[0.2]
        self.assertIn(f"**{b['error_capture_rate'] * 100:.1f}%**", self.hiring)
        self.assertIn(f"{b['error_capture_rate'] * 100:.1f}%", self.readme)
        self.assertIn(f"{b['yield_lift_vs_random']:.1f}×", self.readme)

    def test_error_count_in_hiring_summary(self):
        holdout = load("summary.json")["dataset"]["evaluation_split"]["holdout_samples"]
        errors = sum(sum(row) for row in load("confusion-matrix.json")["matrix"]) - sum(
            load("confusion-matrix.json")["matrix"][i][i] for i in range(len(load("confusion-matrix.json")["matrix"]))
        )
        self.assertIn(f"{errors / holdout * 100:.1f}% ({errors} / {holdout} holdout samples)", self.hiring)


class CnnResultsBlockTests(unittest.TestCase):
    """The CNN block in README.md may only ever reflect a real (non-smoke) summary.json."""

    def setUp(self):
        self.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.summary_file = ROOT / "docs" / "data" / "neu-cls-dl" / "summary.json"

    def test_readme_has_exactly_one_results_block(self):
        self.assertEqual(self.readme.count(update_docs_from_dl.START), 1)
        self.assertEqual(self.readme.count(update_docs_from_dl.END), 1)

    def test_committed_results_are_real_and_match_the_readme(self):
        if not self.summary_file.exists():
            self.skipTest("no CNN run recorded yet")
        summary = json.loads(self.summary_file.read_text(encoding="utf-8"))
        self.assertFalse(summary.get("smoke_test"), "docs/data/neu-cls-dl/summary.json is from a smoke test")
        rf = next(m for m in load("benchmark-comparison.json") if m["model"] == "random_forest")
        block = update_docs_from_dl.render_block(summary, rf)
        self.assertIn(block, self.readme, "README CNN block is stale: run python analysis/update_docs_from_dl.py")

    def test_placeholder_is_only_present_while_no_run_is_committed(self):
        has_placeholder = "No real-data CNN run has been recorded yet" in self.readme
        self.assertEqual(has_placeholder, not self.summary_file.exists())


if __name__ == "__main__":
    unittest.main()
