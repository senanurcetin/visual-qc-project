import unittest

import line_sim
import main


class LineSimulationTests(unittest.TestCase):
    def setUp(self):
        self.state = line_sim.new_state(now=1000.0)

    def run_for(self, seconds, start=1000.0):
        self.state = line_sim.apply_command(self.state, "START", start)
        return start + seconds

    def test_forced_failure_gets_a_defect_class(self):
        self.state = line_sim.apply_command(self.state, "SIMULATE_FAIL", 1000.0)  # before start: first unit
        now = self.run_for(3.8)
        snap = line_sim.snapshot(self.state, now)
        self.assertEqual(snap["total_units"], 1)
        self.assertEqual(snap["nok_units"], 1)
        self.assertEqual(snap["recent_logs"][0]["status"], "FAIL")
        self.assertIn(snap["recent_logs"][0]["defect"], line_sim.DEFECT_CLASSES)

    def test_simulate_fail_while_running_targets_the_next_unit(self):
        now = self.run_for(1.0)
        self.state = line_sim.apply_command(self.state, "SIMULATE_FAIL", now)
        self.assertIn(1, self.state["forced"])
        self.assertEqual(line_sim.unit(self.state, 1)[0], "FAIL")

    def test_counts_are_a_pure_function_of_time(self):
        now = self.run_for(3 * line_sim.CYCLE_SECONDS + 0.5)
        a, b = line_sim.snapshot(self.state, now), line_sim.snapshot(dict(self.state), now)
        self.assertEqual(a, b)
        self.assertEqual(a["total_units"], 3)
        self.assertEqual(a["status_cycle"], 3)
        self.assertEqual(a["ok_units"] + a["nok_units"], 3)

    def test_pause_and_estop_freeze_the_clock(self):
        now = self.run_for(5.0)
        self.state = line_sim.apply_command(self.state, "PAUSE", now)
        self.assertEqual(line_sim.sim_time(self.state, now + 60), 5.0)
        self.state = line_sim.apply_command(self.state, "START", now + 60)
        self.state = line_sim.apply_command(self.state, "ESTOP", now + 62)
        self.assertEqual(line_sim.sim_time(self.state, now + 999), 7.0)
        self.assertEqual(line_sim.snapshot(self.state, now + 999)["system_mode"], "ESTOP")

    def test_reset_starts_a_fresh_line(self):
        now = self.run_for(9.0)
        self.state = line_sim.apply_command(self.state, "RESET", now)
        snap = line_sim.snapshot(self.state, now + 5)
        self.assertEqual((snap["system_mode"], snap["total_units"], snap["current_unit_status"]), ("PAUSED", 0, "PENDING"))

    def test_history_matches_snapshot(self):
        now = self.run_for(10 * line_sim.CYCLE_SECONDS)
        rows = line_sim.history(self.state, now)
        snap = line_sim.snapshot(self.state, now)
        self.assertEqual(len(rows), snap["total_units"])
        self.assertEqual(sum(r["status"] == "OK" for r in rows), snap["ok_units"])


class DashboardRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = main.app.test_client()

    def test_api_data_exposes_twin_sync_fields(self):
        payload = self.client.get("/api/data").get_json()
        for key in ("sim_time", "cycle_seconds", "current_unit_id", "status_cycle", "current_defect", "system_mode"):
            self.assertIn(key, payload)
        self.assertEqual(payload["cycle_seconds"], main.ANIMATION_CYCLE)

    def test_visitors_have_independent_lines(self):
        alice, bob = main.app.test_client(), main.app.test_client()
        alice.post("/api/control", json={"command": "START"})
        bob.post("/api/control", json={"command": "ESTOP"})
        self.assertEqual(alice.get("/api/data").get_json()["system_mode"], "RUNNING")
        self.assertEqual(bob.get("/api/data").get_json()["system_mode"], "ESTOP")

    def test_state_survives_without_server_memory(self):
        # The line lives in the signed cookie: a fresh app instance with the same cookie sees the same line.
        self.client.post("/api/control", json={"command": "START"})
        cookie = self.client.get_cookie("session")
        other = main.app.test_client()
        other.set_cookie("session", cookie.value)
        self.assertEqual(other.get("/api/data").get_json()["system_mode"], "RUNNING")

    def test_control_without_json_body_is_ignored(self):
        self.assertEqual(self.client.post("/api/control", data="x").status_code, 200)

    def test_export_report_is_an_excel_file(self):
        self.assertEqual(self.client.get("/api/export_report").status_code, 404)  # nothing produced yet
        with self.client.session_transaction() as sess:
            state = line_sim.new_state(0.0)
            sess["line"] = line_sim.apply_command(state, "START", __import__("time").time() - 20)
        response = self.client.get("/api/export_report")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data[:2], b"PK")

    def test_index_mounts_the_3d_twin_with_fallback(self):
        html = self.client.get("/").data
        self.assertIn(b'id="line-twin"', html)
        self.assertIn(b'id="twin-fallback"', html)
        self.assertIn(b"/line/static/line3d.js", html)

    def test_twin_assets_are_served(self):
        for asset in ("line3d.js", "line3d.css"):
            response = self.client.get(f"/line/static/{asset}")
            self.assertEqual(response.status_code, 200, asset)
            response.close()


if __name__ == "__main__":
    unittest.main()
