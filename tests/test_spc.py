import math
import time
import unittest

import line_sim
import main
from spc import p_chart


def rows(pattern, batch_size=10):
    """Units U_0000.. where pattern[i] is the number of rejects in batch i."""
    out = []
    for b, rejects in enumerate(pattern):
        for i in range(batch_size):
            out.append({"unit_id": f"U_{b * batch_size + i:04}", "status": "FAIL" if i < rejects else "OK"})
    return out


class PChartTests(unittest.TestCase):
    def test_centre_and_three_sigma_limits(self):
        chart = p_chart(rows([1, 2, 1, 2]))
        self.assertAlmostEqual(chart["center"], 0.15)
        sigma = math.sqrt(0.15 * 0.85 / 10)
        self.assertAlmostEqual(chart["ucl"], 0.15 + 3 * sigma)
        self.assertEqual(chart["lcl"], 0.0)  # 0.15 - 3 sigma is negative -> clamped
        self.assertEqual([b["rejects"] for b in chart["batches"]], [1, 2, 1, 2])
        self.assertFalse(any(b["out_of_control"] for b in chart["batches"]))

    def test_flags_a_batch_above_the_upper_limit(self):
        chart = p_chart(rows([0, 0, 0, 0, 0, 0, 0, 0, 0, 8]))
        self.assertEqual([b["batch"] for b in chart["batches"] if b["out_of_control"]], [9])

    def test_incomplete_batches_are_dropped(self):
        partial = rows([1, 1, 1])[:-4]  # last batch has only 6 units
        self.assertEqual([b["batch"] for b in p_chart(partial)["batches"]], [0, 1])

    def test_keeps_only_the_most_recent_batches(self):
        chart = p_chart(rows([1] * 50), max_batches=30)
        self.assertEqual(len(chart["batches"]), 30)
        self.assertEqual(chart["batches"][-1]["batch"], 49)

    def test_empty_history(self):
        self.assertEqual(p_chart([])["batches"], [])
        self.assertIsNone(p_chart([])["center"])

    def test_upper_limit_never_exceeds_one(self):
        self.assertLessEqual(p_chart(rows([10, 10, 10]))["ucl"], 1.0)


class SpcRouteTests(unittest.TestCase):
    def test_route_reports_batches_for_a_running_line(self):
        client = main.app.test_client()
        with client.session_transaction() as sess:
            sess["line"] = line_sim.apply_command(line_sim.new_state(0.0), "START", time.time() - 200)
        data = client.get("/api/spc").get_json()
        self.assertGreaterEqual(len(data["batches"]), 3)
        self.assertLessEqual(data["lcl"], data["center"])
        self.assertLessEqual(data["center"], data["ucl"])

    def test_page_is_served_and_hmi_links_to_it(self):
        client = main.app.test_client()
        self.assertIn(b"Reject rate p-chart", client.get("/spc").data)
        self.assertIn(b'href="/spc"', client.get("/").data)


if __name__ == "__main__":
    unittest.main()
