"""Persistence for operator review decisions.

`DATABASE_URL` selects the backend: a `postgres://` / `postgresql://` URL uses psycopg (Neon in
production), anything else falls back to a local SQLite file. Only a tiny, portable SQL subset is
used so both backends share one code path.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_decisions (
    visitor_id      TEXT NOT NULL,
    unit_id         TEXT NOT NULL,
    predicted_defect TEXT NOT NULL,
    decision        TEXT NOT NULL,
    operator_label  TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (visitor_id, unit_id)
)
"""
COLUMNS = ("visitor_id", "unit_id", "predicted_defect", "decision", "operator_label", "updated_at")


def default_url() -> str:
    return os.environ.get("DATABASE_URL") or f"sqlite:///{Path(tempfile.gettempdir()) / 'visual_qc_reviews.db'}"


class ReviewStore:
    def __init__(self, url: str | None = None):
        self.url = url or default_url()
        self.postgres = self.url.startswith(("postgres://", "postgresql://"))
        self.ph = "%s" if self.postgres else "?"
        self._ready = False

    @contextmanager
    def _connect(self):
        if self.postgres:
            import psycopg  # imported lazily so the SQLite path needs nothing extra

            conn = psycopg.connect(self.url, connect_timeout=10)
        else:
            path = self.url.removeprefix("sqlite:///")
            conn = sqlite3.connect(path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA)
        self._ready = True

    def _ensure(self) -> None:
        if not self._ready:
            self.init_schema()

    def record(self, visitor_id: str, unit_id: str, predicted_defect: str, decision: str, operator_label: str) -> dict:
        """Insert or replace this visitor's decision for a unit."""
        self._ensure()
        now = datetime.now(UTC).isoformat(timespec="seconds")
        p = self.ph
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO review_decisions ({', '.join(COLUMNS)}) VALUES ({', '.join([p] * 6)}) "
                "ON CONFLICT (visitor_id, unit_id) DO UPDATE SET predicted_defect = excluded.predicted_defect, "
                "decision = excluded.decision, operator_label = excluded.operator_label, updated_at = excluded.updated_at",
                (visitor_id, unit_id, predicted_defect, decision, operator_label, now),
            )
        return dict(zip(COLUMNS, (visitor_id, unit_id, predicted_defect, decision, operator_label, now), strict=True))

    def for_visitor(self, visitor_id: str) -> dict[str, dict]:
        """Decisions keyed by unit id."""
        self._ensure()
        with self._connect() as conn:
            cur = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM review_decisions WHERE visitor_id = {self.ph}", (visitor_id,))
            rows = cur.fetchall()
        return {r[1]: dict(zip(COLUMNS, r, strict=True)) for r in rows}
