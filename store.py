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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

MIGRATIONS: list[tuple[int, list[str]]] = [
    (1, ["""
        CREATE TABLE IF NOT EXISTS review_decisions (
            visitor_id       TEXT NOT NULL,
            unit_id          TEXT NOT NULL,
            predicted_defect TEXT NOT NULL,
            decision         TEXT NOT NULL,
            operator_label   TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            PRIMARY KEY (visitor_id, unit_id)
        )"""]),
    (2, ["CREATE INDEX IF NOT EXISTS review_decisions_decision_idx ON review_decisions (decision, updated_at)"]),
]
COLUMNS = ("visitor_id", "unit_id", "predicted_defect", "decision", "operator_label", "updated_at")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


class LimitReached(Exception):
    """A write was refused: `scope` is "visitor" (their own quota) or "storage" (whole table)."""

    def __init__(self, scope: str):
        super().__init__(f"{scope} limit reached")
        self.scope = scope


def default_url() -> str:
    return os.environ.get("DATABASE_URL") or f"sqlite:///{Path(tempfile.gettempdir()) / 'visual_qc_reviews.db'}"


class ReviewStore:
    def __init__(
        self,
        url: str | None = None,
        migrations: list[tuple[int, list[str]]] | None = None,
        max_per_visitor: int | None = None,
        max_total: int | None = None,
        ttl_days: int | None = None,
    ):
        # Soft limits (checked before insert, not transactional): they bound abuse of the public demo.
        self.max_per_visitor = max_per_visitor if max_per_visitor is not None else _env_int("REVIEW_MAX_PER_VISITOR", 200)
        self.max_total = max_total if max_total is not None else _env_int("REVIEW_MAX_ROWS", 50_000)
        self.ttl_days = ttl_days if ttl_days is not None else _env_int("REVIEW_TTL_DAYS", 90)
        self.url = url or default_url()
        self.migrations = MIGRATIONS if migrations is None else migrations
        self.postgres = self.url.startswith(("postgres://", "postgresql://"))
        self.ph = "%s" if self.postgres else "?"
        self._ready = False

    @contextmanager
    def _connect(self):
        conn: Any
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

    def init_schema(self) -> list[int]:
        """Apply pending migrations in version order; return the versions applied by this call.

        Every statement is idempotent (IF NOT EXISTS) and versions are recorded with ON CONFLICT DO
        NOTHING, so two serverless instances migrating at the same time cannot break each other.
        """
        applied_now = []
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
            done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()}
            for version, statements in sorted(self.migrations):
                if version in done:
                    continue
                for statement in statements:
                    conn.execute(statement)
                conn.execute(
                    f"INSERT INTO schema_migrations (version, applied_at) VALUES ({self.ph}, {self.ph}) ON CONFLICT (version) DO NOTHING",
                    (version, datetime.now(UTC).isoformat(timespec="seconds")),
                )
                applied_now.append(version)
        self._ready = True
        return applied_now

    def _ensure(self) -> None:
        if not self._ready:
            self.init_schema()
            self.prune()  # once per process: drop decisions past their retention

    def prune(self, now: datetime | None = None) -> int:
        """Delete decisions last updated more than `ttl_days` ago; returns how many."""
        cutoff = ((now or datetime.now(UTC)) - timedelta(days=self.ttl_days)).isoformat(timespec="seconds")
        with self._connect() as conn:
            return conn.execute(f"DELETE FROM review_decisions WHERE updated_at < {self.ph}", (cutoff,)).rowcount

    def _count(self, conn: Any, where: str = "", params: tuple = ()) -> int:
        return conn.execute(f"SELECT COUNT(*) FROM review_decisions {where}", params).fetchone()[0]

    def record(self, visitor_id: str, unit_id: str, predicted_defect: str, decision: str, operator_label: str) -> dict:
        """Insert or replace this visitor's decision for a unit.

        Raises LimitReached when a *new* row would exceed the visitor's quota or the table's capacity;
        changing a decision that already exists is always allowed.
        """
        self._ensure()
        now = datetime.now(UTC).isoformat(timespec="seconds")
        p = self.ph
        with self._connect() as conn:
            exists = self._count(conn, f"WHERE visitor_id = {p} AND unit_id = {p}", (visitor_id, unit_id))
            if not exists:
                if self._count(conn, f"WHERE visitor_id = {p}", (visitor_id,)) >= self.max_per_visitor:
                    raise LimitReached("visitor")
                if self._count(conn) >= self.max_total:
                    self.prune()
                    if self._count(conn) >= self.max_total:
                        raise LimitReached("storage")
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

    def all_corrections(self) -> list[dict]:
        """Corrected labels from every visitor, oldest first (quality-engineer view)."""
        self._ensure()
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT {', '.join(COLUMNS)} FROM review_decisions WHERE decision = 'correct' ORDER BY updated_at, visitor_id, unit_id"
            )
            return [dict(zip(COLUMNS, r, strict=True)) for r in cur.fetchall()]
