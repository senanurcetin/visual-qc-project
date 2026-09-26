import time
import unittest
from unittest import mock

import main


class SimulationTests(unittest.TestCase):
    def setUp(self):
        with main.data_lock:
            main.factory_state.clear()
            main.factory_state.update(main.get_initial_state())

    def _run_at(self, sim_time):
        """Put the line in RUNNING mode at a given simulation time and advance one tick."""
        with main.data_lock:
            main.factory_state["system_mode"] = "RUNNING"
            main.factory_state["sim_accumulated_time"] = sim_time
            main.factory_state["sim_start_time"] = time.time()
        main.advance_simulation()

    @mock.patch("main.save_log_to_db")
    def test_forced_failure_gets_a_defect_class_and_cycle(self, save):
        main.factory_state["force_fail_next"] = True
        self._run_at(2 * main.ANIMATION_CYCLE + 0.1)  # start of cycle 2
        state = main.factory_state
        self.assertEqual(state["current_unit_status"], "FAIL")
        self.assertIn(state["current_defect"], main.DEFECT_CLASSES)
        self.assertEqual(state["status_cycle"], 2)
        self.assertFalse(state["force_fail_next"])

        self._run_at(2 * main.ANIMATION_CYCLE + 3.8)  # end of the same cycle
        self.assertEqual(state["total_units"], 1)
        self.assertEqual(state["nok_units"], 1)
        self.assertEqual(state["recent_logs"][0]["defect"], state["current_defect"])
        save.assert_called_once()

    @mock.patch("main.save_log_to_db")
    def test_ok_unit_has_no_defect(self, save):
        with mock.patch("main.random.random", return_value=0.99):
            self._run_at(0.1)
        self.assertEqual(main.factory_state["current_unit_status"], "OK")
        self.assertIsNone(main.factory_state["current_defect"])

    def test_paused_line_does_not_advance(self):
        main.advance_simulation()
        self.assertEqual(main.factory_state["total_units"], 0)
        self.assertEqual(main.factory_state["current_unit_status"], "PENDING")


class DashboardRouteTests(unittest.TestCase):
    def setUp(self):
        # Keep the background simulation thread out of the tests so SimulationTests stay deterministic.
        patcher = mock.patch.object(main, "start_simulation")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = main.app.test_client()

    def test_api_data_exposes_twin_sync_fields(self):
        payload = self.client.get("/api/data").get_json()
        for key in ("sim_time", "cycle_seconds", "current_unit_id", "status_cycle", "current_defect", "system_mode"):
            self.assertIn(key, payload)
        self.assertEqual(payload["cycle_seconds"], main.ANIMATION_CYCLE)

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
