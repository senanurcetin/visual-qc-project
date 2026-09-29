import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path

import line_sim
import main
from store import MIGRATIONS, LimitReached, ReviewStore


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


class AdminRoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        main.app.config["DATABASE_URL"] = f"sqlite:///{Path(self.tmp.name) / 'a.db'}"
        main.app.extensions.pop("review_store", None)
        self.client = main.app.test_client()
        start_with_failed_first_unit(self.client)
        predicted = next(i for i in self.client.get("/api/review-queue").get_json()["items"] if i["unit_id"] == "U_0000")["predicted_defect"]
        self.other = next(c for c in line_sim.DEFECT_CLASSES if c != predicted)
        self.client.post("/api/review", json={"unit_id": "U_0000", "label": self.other})

    def tearDown(self):
        main.app.config.pop("DATABASE_URL", None)
        main.app.extensions.pop("review_store", None)
        os.environ.pop("REVIEW_ADMIN_TOKEN", None)
        self.tmp.cleanup()

    def test_disabled_without_a_configured_token(self):
        os.environ.pop("REVIEW_ADMIN_TOKEN", None)
        self.assertEqual(main.app.test_client().get("/api/admin/corrections.csv").status_code, 404)

    def test_requires_the_bearer_token(self):
        os.environ["REVIEW_ADMIN_TOKEN"] = "s3cret"
        anon = main.app.test_client()  # a different visitor than the one who corrected the label
        self.assertEqual(anon.get("/api/admin/corrections.csv").status_code, 401)
        self.assertEqual(anon.get("/api/admin/corrections.csv", headers={"Authorization": "Bearer nope"}).status_code, 401)

    def test_returns_every_visitors_corrections_with_the_token(self):
        os.environ["REVIEW_ADMIN_TOKEN"] = "s3cret"
        response = main.app.test_client().get("/api/admin/corrections.csv", headers={"Authorization": "Bearer s3cret"})
        rows = response.get_data(as_text=True).strip().splitlines()
        self.assertEqual(rows[0], "visitor_id,unit_id,predicted_defect,operator_label,updated_at")
        self.assertEqual(len(rows), 2)
        self.assertIn(",U_0000,", rows[1])
        self.assertIn(self.other, rows[1])


class StoreContract:
    """Behaviour every backend must share. Subclasses provide `make_store()`."""

    def make_store(self, **kwargs) -> ReviewStore:
        raise NotImplementedError

    def setUp(self):
        self.uid = uuid.uuid4().hex[:8]  # isolates rows when the database is shared

    def v(self, name):
        return f"{name}-{self.uid}"

    def test_upsert_keeps_one_row_per_visitor_and_unit(self):
        store = self.make_store()
        store.record(self.v("v1"), "U_0001", "crazing", "confirm", "crazing")
        store.record(self.v("v1"), "U_0001", "crazing", "correct", "patches")
        store.record(self.v("v2"), "U_0001", "crazing", "confirm", "crazing")
        self.assertEqual(store.for_visitor(self.v("v1"))["U_0001"]["operator_label"], "patches")
        self.assertEqual(len(store.for_visitor(self.v("v1"))), 1)
        self.assertEqual(store.for_visitor(self.v("v2"))["U_0001"]["decision"], "confirm")
        self.assertEqual(store.for_visitor(self.v("nobody")), {})

    def test_all_corrections_spans_visitors_and_skips_confirmations(self):
        store = self.make_store()
        store.record(self.v("a"), "U_0001", "crazing", "correct", "patches")
        store.record(self.v("b"), "U_0002", "scratches", "correct", "inclusion")
        store.record(self.v("c"), "U_0003", "patches", "confirm", "patches")
        mine = [c for c in store.all_corrections() if c["visitor_id"].endswith(self.uid)]
        self.assertEqual(sorted(c["visitor_id"][0] for c in mine), ["a", "b"])

    def test_migrations_apply_once_in_order_and_new_ones_are_picked_up(self):
        store = self.make_store()
        store.init_schema()  # a fresh database applies everything; a shared one may already have it
        self.assertEqual(store.init_schema(), [])
        extra = 1000 + int(self.uid, 16) % 100000
        later = self.make_store(migrations=[*MIGRATIONS, (extra, ["CREATE INDEX IF NOT EXISTS ix_" + self.uid + " ON review_decisions (unit_id)"])])
        self.assertEqual(later.init_schema(), [extra])
        self.assertEqual(later.init_schema(), [])


class QuotaContract:
    """Per-visitor quota, identical on every backend (provided by test_review.StoreContract subclasses)."""

    def test_per_visitor_quota_blocks_new_units_but_not_edits_or_other_visitors(self):
        store = self.make_store(max_per_visitor=2)
        me, other = self.v("me"), self.v("other")
        store.record(me, "U_0001", "crazing", "confirm", "crazing")
        store.record(me, "U_0002", "crazing", "confirm", "crazing")
        with self.assertRaises(LimitReached) as ctx:
            store.record(me, "U_0003", "crazing", "confirm", "crazing")
        self.assertEqual(ctx.exception.scope, "visitor")
        store.record(me, "U_0001", "crazing", "correct", "patches")  # editing an existing decision is fine
        self.assertEqual(store.for_visitor(me)["U_0001"]["operator_label"], "patches")
        store.record(other, "U_0003", "crazing", "confirm", "crazing")  # quotas are per visitor


class SqliteStoreTests(StoreContract, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "s.db"
        self.addCleanup(self.tmp.cleanup)

    def make_store(self, **kwargs):
        return ReviewStore(f"sqlite:///{self.path}", **kwargs)

    def test_fresh_database_applies_every_migration(self):
        self.assertEqual(self.make_store().init_schema(), [v for v, _ in MIGRATIONS])

    def test_backend_is_chosen_from_the_url(self):
        self.assertTrue(ReviewStore("postgresql://u:p@host/db").postgres)
        self.assertFalse(ReviewStore("sqlite:///x.db").postgres)


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "set TEST_DATABASE_URL to run against a real Postgres")
class PostgresStoreTests(StoreContract, QuotaContract, unittest.TestCase):
    def make_store(self, **kwargs):
        store = ReviewStore(os.environ["TEST_DATABASE_URL"], **kwargs)
        self.assertTrue(store.postgres)
        return store


if __name__ == "__main__":
    unittest.main()
