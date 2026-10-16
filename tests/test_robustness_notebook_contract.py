"""The public NEU-CLS notebook is explicit about evidence and never stores keys."""
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RobustnessNotebookContract(unittest.TestCase):
    def test_sources_and_provenance(self):
        file = ROOT / "notebooks" / "neu_cls_robustness_reproduction.ipynb"
        data = json.loads(file.read_text(encoding="utf-8"))
        assert data["nbformat"] == 4
        text = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])
        self.assertIn("26f63a336a40856b4af09a6ec8bf0daa27e6801ab79cb0cd8a29d253ab9480f5", text)
        self.assertIn("RandomForestClassifier", text)
        self.assertIn("n_estimators=250", text)
        self.assertIn("gaussian_noise_sigma_12", text)
        self.assertIn("test_idx", text)
        self.assertIn("not original Kaggle CNN notebook", text)
        for forbidden in ["KGAT_", "KAGGLE_API_TOKEN", "private_key", "service_account"]:
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
