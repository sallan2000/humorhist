"""SQLite schema layer for the humorhist content pipeline.

Only stdlib sqlite3 is used. All SQL is parameterised; the single place a
table name is interpolated (set_status) validates it against a whitelist
first.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone

# Allowed statuses per table. Used to validate set_status inputs and to
# whitelist the table name before interpolation.
_VALID_STATUSES: dict[str, set[str]] = {
    "pool": {"new", "drafted", "rejected", "used"},
    "drafts": {"pending", "approved", "rejected"},
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pool (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL, year INT, date_hint TEXT,
  summary TEXT, source_url TEXT, source_name TEXT,
  funny_score REAL,
  status TEXT DEFAULT 'new',
  harvested_at TEXT
);
CREATE TABLE IF NOT EXISTS drafts (
  id TEXT PRIMARY KEY, pool_id TEXT REFERENCES pool(id),
  brief_json TEXT, angles_json TEXT,
  status TEXT DEFAULT 'pending',
  editor_line TEXT, editor_notes TEXT,
  created_at TEXT, reviewed_at TEXT
);
CREATE TABLE IF NOT EXISTS queue (
  id INTEGER PRIMARY KEY AUTOINCREMENT, draft_id TEXT REFERENCES drafts(id),
  scheduled_for TEXT, published INT DEFAULT 0
);
CREATE TABLE IF NOT EXISTS posts (
  draft_id TEXT PRIMARY KEY, slug TEXT, url TEXT,
  tweet_id TEXT, published_at TEXT
);
"""

_ID_LEN = 16


def connect(path: str) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys on and Row factory set."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    """Create all tables if they do not exist. Idempotent."""
    conn.executescript(_SCHEMA)
    # Phase 4 (B+): editable post copy lives on the queue row. Added after the
    # original schema shipped, so guard with PRAGMA checks to make the migration
    # safe to re-run on already-migrated databases.
    _ensure_queue_copy_columns(conn)
    conn.commit()


def _ensure_queue_copy_columns(conn: sqlite3.Connection) -> None:
    """Add post_copy / post_copy_at to queue if absent (no-op if present)."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(queue)")}
    if "post_copy" not in existing:
        conn.execute("ALTER TABLE queue ADD COLUMN post_copy TEXT")
    if "post_copy_at" not in existing:
        conn.execute("ALTER TABLE queue ADD COLUMN post_copy_at TEXT")


def make_id(*parts: str) -> str:
    """Stable 16-char id from parts joined by '|' (sha1 hex digest)."""
    joined = "|".join(parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    return digest[:_ID_LEN]


def upsert_pool_item(
    conn: sqlite3.Connection,
    *,
    id: str,
    title: str,
    year: int | None,
    date_hint: str | None,
    summary: str | None,
    source_url: str | None,
    source_name: str | None,
) -> bool:
    """Insert a pool item if absent (INSERT OR IGNORE).

    Returns True if newly inserted, False if it already existed. Never
    overwrites an existing row's funny_score or status. Sets harvested_at to
    the current UTC ISO8601 timestamp on insert.
    """
    harvested_at = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO pool
          (id, title, year, date_hint, summary, source_url, source_name, harvested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (id, title, year, date_hint, summary, source_url, source_name, harvested_at),
    )
    conn.commit()
    return cur.rowcount == 1


def set_status(conn: sqlite3.Connection, table: str, row_id: str, status: str) -> None:
    """Update the status column of a pool or drafts row.

    Table name is validated against a whitelist; status must be legal for the
    given table. Raises ValueError otherwise.
    """
    if table not in _VALID_STATUSES:
        raise ValueError(f"invalid table name: {table!r}")
    if status not in _VALID_STATUSES[table]:
        raise ValueError(f"invalid status {status!r} for table {table!r}")

    conn.execute(
        f"UPDATE {table} SET status = ? WHERE id = ?",
        (status, row_id),
    )
    conn.commit()


def set_funny_score(conn: sqlite3.Connection, pool_id: str, score: float) -> None:
    """Set the funny_score for a pool item."""
    conn.execute(
        "UPDATE pool SET funny_score = ? WHERE id = ?",
        (score, pool_id),
    )
    conn.commit()


def get_pool_item(conn: sqlite3.Connection, pool_id: str) -> sqlite3.Row | None:
    """Return the pool row for the given id, or None if absent."""
    cur = conn.execute("SELECT * FROM pool WHERE id = ?", (pool_id,))
    return cur.fetchone()


def draft_exists_for_pool(conn: sqlite3.Connection, pool_id: str) -> bool:
    """True if a drafts row already exists for this pool item.

    Used to make the drafting step non-destructive: a re-run (e.g. on a weekly
    schedule) must not overwrite an existing draft — including one the editor
    has already reviewed, edited, or approved.
    """
    cur = conn.execute("SELECT 1 FROM drafts WHERE pool_id = ? LIMIT 1", (pool_id,))
    return cur.fetchone() is not None


def counts(conn: sqlite3.Connection) -> dict:
    """Return a dict of row counts for pool, drafts, queue, posts."""
    result: dict[str, int] = {}
    for table in ("pool", "drafts", "queue", "posts"):
        cur = conn.execute(f"SELECT COUNT(*) AS n FROM {table}")
        result[table] = int(cur.fetchone()["n"])
    return result
