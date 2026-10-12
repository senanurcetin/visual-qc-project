"""Regression tests for the genuine exported QC workbook, without Excel itself."""
import io
import unittest
import zipfile
from xml.etree import ElementTree as ET

import pandas as pd

from report import build_report

XML = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def sample(statuses=("OK", "FAIL", "OK")):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-10-13", periods=len(statuses), freq="min"),
        "unit_id": [f"UNIT-{i}" for i in range(len(statuses))],
        "status": list(statuses),
        "defect": ["NONE" if s == "OK" else "SCRATCH" for s in statuses],
        "oee_score": [0.7 + i * 0.05 for i in range(len(statuses))],
    })


class WorkbookReportTests(unittest.TestCase):
    def test_spreadsheet_contains_formula_kpis_and_historical_source_label(self):
        result = build_report(sample())
        with zipfile.ZipFile(io.BytesIO(result.getvalue())) as package:
            self.assertIn("xl/worksheets/sheet1.xml", package.namelist())
            self.assertIn("xl/worksheets/sheet2.xml", package.namelist())
            self.assertIn("xl/charts/chart1.xml", package.namelist())
            xml = ET.fromstring(package.read("xl/worksheets/sheet1.xml"))
            formulas = [cell.text or "" for cell in xml.iter(XML + "f")]
            self.assertTrue(any("COUNTA" in f for f in formulas))
            self.assertTrue(any("COUNTIF" in f for f in formulas))
            self.assertTrue(any("AVERAGE" in f for f in formulas))
            shared = package.read("xl/sharedStrings.xml").decode("utf-8")
            self.assertIn("SIMULATED DEMO / NOT REAL PLANT DATA", shared)
            self.assertIn("NOT REAL PLANT DATA", shared)
            charts = package.read("xl/charts/chart1.xml").decode("utf-8")
            self.assertIn("4EA8FF", charts)
            self.assertIn("8F9BFF", charts)
            # The underlying chart data lives in hidden AA:AB. XlsxWriter
            # defaults to plotVisOnly=1, which suppresses those data points.
            # With show_hidden_data(), plotVisOnly must be absent/false.
            self.assertNotIn("plotVisOnly", charts)

    def test_all_rejected_still_uses_stable_status_order(self):
        with zipfile.ZipFile(build_report(sample(("FAIL", "FAIL")))) as package:
            strings = package.read("xl/sharedStrings.xml").decode("utf-8")
            self.assertIn(">OK<", strings)
            self.assertIn(">FAIL<", strings)

    def test_no_fabricated_empty_session_or_invalid_oee(self):
        with self.assertRaises(ValueError):
            build_report(sample(()))
        invalid = sample()
        invalid.loc[0, "oee_score"] = 3
        with self.assertRaises(ValueError):
            build_report(invalid)

    def test_malformed_status_rejected(self):
        wrong = sample()
        wrong.loc[0, "status"] = "UNKNOWN"
        with self.assertRaises(ValueError):
            build_report(wrong)


if __name__ == "__main__":
    unittest.main()
