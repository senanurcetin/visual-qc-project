"""Offline provenance contract for the published NEU-CLS notebook."""
import ast
import json
import unittest
from pathlib import Path


class RobustnessNotebookTests(unittest.TestCase):
    def test_public_notebook_is_unexecuted_and_names_its_data_gate(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/neu_cls_robustness_reproduction.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(code)
        for index, cell in enumerate(code):
            with self.subTest(cell=index):
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        opening = "".join(notebook["cells"][0]["source"])
        self.assertIn("10.6084/m9.figshare.28903550.v1", opening)
        self.assertIn("train/train/images", opening)
        self.assertIn("valid/valid/images", opening)
