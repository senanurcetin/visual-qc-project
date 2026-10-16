"""Test preflight safety rules without downloading datasets or touching CI services."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_preflight.py"
spec = importlib.util.spec_from_file_location("nexus_ci_preflight", SCRIPT)
assert spec and spec.loader
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PreflightTests(unittest.TestCase):
    def test_rejects_unescaped_circleci_heredoc(self):
        with self.assertRaisesRegex(ValueError, "unescaped heredoc"):
            checker.check_circleci("command: docker exec -i app python - <<'PY'")
        checker.check_circleci("command: docker exec -i app python - \\<<'PY'")

    def test_rejects_notebook_code_syntax_errors_and_outputs(self):
        sample = {
            "nbformat": 4,
            "cells": [
                {"cell_type": "code", "source": ["x = 5\n"], "execution_count": None, "outputs": []}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.ipynb"
            path.write_text(json.dumps(sample), encoding="utf-8")
            checker.check_notebook(path)
            sample["cells"][0]["source"] = ["if True print('bad')"]
            path.write_text(json.dumps(sample), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "syntax error"):
                checker.check_notebook(path)
            sample["cells"][0]["source"] = ["x = 5\n"]
            sample["cells"][0]["outputs"] = [{"output_type": "stream", "text": "unreviewed"}]
            path.write_text(json.dumps(sample), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "saved output"):
                checker.check_notebook(path)

    def test_repository_files_pass(self):
        self.assertEqual(checker.main(), 0)


if __name__ == "__main__":
    unittest.main()
