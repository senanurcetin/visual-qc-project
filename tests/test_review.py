import tempfile
import time
import unittest
from pathlib import Path

import line_sim
import main
from store import ReviewStore


def start_with_failed_first_unit(client, seconds=40):
    """A line that started `seconds` ago with unit U_0000 forced to fail."""
    with client.session_transaction() as sess:
        state = line_sim.apply_command(line_sim.new_state(0.0), "SIMULATE_FAIL", 0.0)
        sess["line"] = line_sim.apply_command(state, "START", time.time() - seconds)


class ReviewApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        main.app.config["DATABASE_URL"] = f"sqlite:///{Path(self.tmp.name) / 'reviews.db'}"
        main.app.extensions.pop("review_store", None)
        self.client = main.app.test_client()
        start_with_failed_first_unit(self.client)

    def tearDown(self):
        main.app.config.pop("DATABASE_URL", None)
        main.app.extensions.pop("review_store", None)
        self.tmp.cleanup()

    def queue(self, client=None):
        return (client or self.client).get("/api/review-queue").get_json()

    def test_queue_lists_rejected_units_with_their_predicted_defect(self):
        item = next(i for i in self.queue()["items"] if i["unit_id"] == "U_0000")
        self.assertIn(item["predicted_defect"], line_sim.DEFECT_CLASSES)
        self.assertIsNone(item["decision"])
        self.assertEqual(self.queue()["classes"], line_sim.DEFECT_CLASSES)

    def test_confirm_and_correct_are_told_apart_and_persisted(self):
        predicted = next(i for i in self.queue()["items"] if i["unit_id"] == "U_0000")["predicted_defect"]
        other = next(c for c in line_sim.DEFECT_CLASSES if c != predicted)

        confirm = self.client.post("/api/review", json={"unit_id": "U_0000", "label": predicted}).get_json()
        self.assertEqual(confirm["decision"], "confirm")
        correct = self.client.post("/api/review", json={"unit_id": "U_0000", "label": other}).get_json()
        self.assertEqual((correct["decision"], correct["operator_label"]), ("correct", other))

        item = next(i for i in self.queue()["items"] if i["unit_id"] == "U_0000")
        self.assertEqual((item["decision"], item["operator_label"]), ("correct", other))  # last write wins

    def test_rejects_unknown_units_and_labels(self):
        self.assertEqual(self.client.post("/api/review", json={"unit_id": "U_9999", "label": "crazing"}).status_code, 404)
        self.assertEqual(self.client.post("/api/review", json={"unit_id": "U_0000", "label": "nope"}).status_code, 400)
        self.assertEqual(self.client.post("/api/review", data="x").status_code, 404)

    def test_ok_units_cannot_be_reviewed(self):
        with self.client.session_transaction() as sess:
            state = dict(sess["line"])
        ok_ids = [r["unit_id"] for r in line_sim.history(state, time.time()) if r["status"] == "OK"]
        self.assertTrue(ok_ids, "the fixture line should have produced at least one OK unit")
        response = self.client.post("/api/review", json={"unit_id": ok_ids[0], "label": "crazing"})
        self.assertEqual(response.status_code, 404)

    def test_decisions_are_private_to_each_visitor(self):
        self.client.post("/api/review", json={"unit_id": "U_0000", "label": "scratches"})
        other = main.app.test_client()
        start_with_failed_first_unit(other)
        item = next(i for i in self.queue(other)["items"] if i["unit_id"] == "U_0000")
        self.assertIsNone(item["decision"])

    def test_export_contains_only_corrections(self):
        predicted = next(i for i in self.queue()["items"] if i["unit_id"] == "U_0000")["predicted_defect"]
        self.client.post("/api/review", json={"unit_id": "U_0000", "label": predicted})
        self.assertEqual(self.client.get("/api/review/export.csv").get_data(as_text=True).count("\n"), 1)  # header only
        other = next(c for c in line_sim.DEFECT_CLASSES if c != predicted)
        self.client.post("/api/review", json={"unit_id": "U_0000", "label": other})
        rows = self.client.get("/api/review/export.csv").get_data(as_text=True).strip().splitlines()
        self.assertEqual(len(rows), 2)
        self.assertIn(f"U_0000,{predicted},{other},", rows[1])

    def test_polling_the_queue_does_not_rewrite_the_cookie(self):
        self.queue()
        self.assertNotIn("Set-Cookie", self.client.get("/api/review-queue").headers)


class ReviewStoreTests(unittest.TestCase):
    def test_upsert_keeps_one_row_per_visitor_and_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ReviewStore(f"sqlite:///{Path(tmp) / 's.db'}")
            store.record("v1", "U_0001", "crazing", "confirm", "crazing")
            store.record("v1", "U_0001", "crazing", "correct", "patches")
            store.record("v2", "U_0001", "crazing", "confirm", "crazing")
            self.assertEqual(store.for_visitor("v1")["U_0001"]["operator_label"], "patches")
            self.assertEqual(len(store.for_visitor("v1")), 1)
            self.assertEqual(store.for_visitor("v2")["U_0001"]["decision"], "confirm")
            self.assertEqual(store.for_visitor("nobody"), {})

    def test_backend_is_chosen_from_the_url(self):
        self.assertTrue(ReviewStore("postgresql://u:p@host/db").postgres)
        self.assertFalse(ReviewStore("sqlite:///x.db").postgres)


if __name__ == "__main__":
    unittest.main()
