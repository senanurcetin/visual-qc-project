import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import main
import ratelimit
from ratelimit import RateLimiter
from store import LimitReached, ReviewStore
from tests.test_review import QuotaContract, start_with_failed_first_unit


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class RateLimiterTests(unittest.TestCase):
    def test_allows_up_to_the_limit_then_reports_when_to_retry(self):
        clock = FakeClock()
        limiter = RateLimiter(3, 60, clock)
        self.assertEqual([limiter.check("a") for _ in range(3)], [None, None, None])
        clock.now += 10
        self.assertAlmostEqual(limiter.check("a"), 50.0)

    def test_window_slides(self):
        clock = FakeClock()
        limiter = RateLimiter(2, 60, clock)
        limiter.check("a")
        clock.now += 30
        limiter.check("a")
        self.assertIsNotNone(limiter.check("a"))
        clock.now += 31  # the first call has left the window, the second has not
        self.assertIsNone(limiter.check("a"))
        self.assertIsNotNone(limiter.check("a"))

    def test_clients_are_independent(self):
        limiter = RateLimiter(1, 60, FakeClock())
        self.assertIsNone(limiter.check("a"))
        self.assertIsNone(limiter.check("b"))
        self.assertIsNotNone(limiter.check("a"))

    def test_memory_is_bounded_by_forgetting_idle_clients(self):
        clock = FakeClock()
        limiter = RateLimiter(1, 60, clock, max_keys=50)
        for i in range(50):
            limiter.check(f"c{i}")
        clock.now += 120
        for i in range(50, 60):
            limiter.check(f"c{i}")
        self.assertLess(len(limiter.hits), 30)


class ClientKeyTests(unittest.TestCase):
    def key(self, headers, vercel):
        old = os.environ.pop("VERCEL", None)
        if vercel:
            os.environ["VERCEL"] = "1"
        try:
            with main.app.test_request_context("/", headers=headers, environ_base={"REMOTE_ADDR": "10.0.0.9"}):
                return ratelimit.client_key()
        finally:
            os.environ.pop("VERCEL", None)
            if old is not None:
                os.environ["VERCEL"] = old

    def test_uses_the_connecting_address_by_default_and_ignores_spoofable_headers(self):
        headers = {"X-Forwarded-For": "1.2.3.4", "X-Vercel-Forwarded-For": "5.6.7.8"}
        self.assertEqual(self.key(headers, vercel=False), "10.0.0.9")

    def test_on_vercel_uses_the_edge_header_but_never_client_x_forwarded_for(self):
        self.assertEqual(self.key({"X-Vercel-Forwarded-For": "5.6.7.8, 9.9.9.9", "X-Forwarded-For": "1.2.3.4"}, vercel=True), "5.6.7.8")
        self.assertEqual(self.key({"X-Forwarded-For": "1.2.3.4"}, vercel=True), "10.0.0.9")


class EndpointLimitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        main.app.config["DATABASE_URL"] = f"sqlite:///{Path(self.tmp.name) / 'l.db'}"
        main.app.extensions.pop("review_store", None)
        ratelimit.reset_limiters()
        self.client = main.app.test_client()
        start_with_failed_first_unit(self.client)

    def tearDown(self):
        main.app.config.pop("DATABASE_URL", None)
        main.app.extensions.pop("review_store", None)
        for name in ("REVIEW_MAX_PER_VISITOR", "REVIEW_MAX_ROWS"):
            os.environ.pop(name, None)
        ratelimit.reset_limiters()
        self.tmp.cleanup()

    def submit(self):
        return self.client.post("/api/review", json={"unit_id": "U_0000", "label": "crazing"})

    def test_review_is_rate_limited_with_retry_after(self):
        ratelimit._limiters["review"] = RateLimiter(2, 60)
        self.assertEqual([self.submit().status_code for _ in range(2)], [200, 200])
        response = self.submit()
        self.assertEqual(response.status_code, 429)
        self.assertGreaterEqual(int(response.headers["Retry-After"]), 1)

    def test_classify_is_rate_limited_before_touching_the_model(self):
        ratelimit._limiters["classify"] = RateLimiter(1, 60)
        first = self.client.post("/api/classify")  # 400: no image, but it counts against the limit
        self.assertEqual(first.status_code, 400)
        self.assertEqual(self.client.post("/api/classify").status_code, 429)

    def test_visitor_quota_returns_429(self):
        os.environ["REVIEW_MAX_PER_VISITOR"] = "0"
        main.app.extensions.pop("review_store", None)
        response = self.submit()
        self.assertEqual(response.status_code, 429)
        self.assertIn("limit", response.get_json()["error"])

    def test_full_storage_returns_503(self):
        os.environ["REVIEW_MAX_ROWS"] = "0"
        main.app.extensions.pop("review_store", None)
        self.assertEqual(self.submit().status_code, 503)


class SqliteQuotaTests(QuotaContract, unittest.TestCase):
    def setUp(self):
        self.uid = "q"
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "q.db"
        self.addCleanup(self.tmp.cleanup)

    def v(self, name):
        return name

    def make_store(self, **kwargs):
        return ReviewStore(f"sqlite:///{self.path}", **kwargs)

    def test_total_capacity_refuses_new_rows_when_nothing_is_old_enough_to_prune(self):
        store = self.make_store(max_total=2)
        store.record("a", "U_0001", "crazing", "confirm", "crazing")
        store.record("b", "U_0001", "crazing", "confirm", "crazing")
        with self.assertRaises(LimitReached) as ctx:
            store.record("c", "U_0001", "crazing", "confirm", "crazing")
        self.assertEqual(ctx.exception.scope, "storage")

    def test_expired_rows_are_pruned_to_make_room(self):
        store = self.make_store(max_total=2, ttl_days=30)
        store.record("a", "U_0001", "crazing", "confirm", "crazing")
        store.record("b", "U_0001", "crazing", "confirm", "crazing")
        future = datetime.now(UTC) + timedelta(days=31)
        self.assertEqual(store.prune(now=future), 2)
        self.assertEqual(store.for_visitor("a"), {})
        store.record("c", "U_0001", "crazing", "confirm", "crazing")  # room again

    def test_prune_keeps_recent_rows(self):
        store = self.make_store(ttl_days=30)
        store.record("a", "U_0001", "crazing", "confirm", "crazing")
        self.assertEqual(store.prune(now=datetime.now(UTC) + timedelta(days=10)), 0)
        self.assertEqual(len(store.for_visitor("a")), 1)


if __name__ == "__main__":
    unittest.main()
