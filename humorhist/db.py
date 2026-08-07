"""SQLite schema layer for the humorhist content pipeline.

Only stdlib sqlite3 is used. All SQL is parameterised; the single place a
table name is interpolated (set_status) validates it against a whitelist
first.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone

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
  harvested_at TEXT,
  note TEXT
);
CREATE TABLE IF NOT EXISTS drafts (
  id TEXT PRIMARY KEY, pool_id TEXT REFERENCES pool(id),
  brief_json TEXT, angles_json TEXT,
  status TEXT DEFAULT 'pending',
  editor_line TEXT, editor_notes TEXT,
  created_at TEXT, reviewed_at TEXT,
  defer_until TEXT
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
    _ensure_defer_column(conn)
    _ensure_pool_note_column(conn)
    conn.commit()


def _ensure_defer_column(conn: sqlite3.Connection) -> None:
    """Add drafts.defer_until if absent (no-op if present)."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(drafts)")}
    if "defer_until" not in existing:
        conn.execute("ALTER TABLE drafts ADD COLUMN defer_until TEXT")


def _ensure_pool_note_column(conn: sqlite3.Connection) -> None:
    """Add pool.note if absent (no-op if present)."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(pool)")}
    if "note" not in existing:
        conn.execute("ALTER TABLE pool ADD COLUMN note TEXT")


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


def add_suggested_pool_item(
    conn: sqlite3.Connection,
    *,
    title: str,
    note: str | None = None,
    source_url: str | None = None,
    year: int | None = None,
) -> str:
    """Insert an editor-suggested pool candidate (via Telegram /suggest).

    Suggested items enter with status ``'new'`` and a NULL score so they flow
    through the normal draft pipeline. The human's note (if any) is stored on
    the pool row for the fact-check/angle step to see. Idempotent on the stable
    id (sha1 of the title) — re-suggesting the same topic updates the note.

    Returns the new/updated pool id.
    """
    pool_id = make_id("suggest", title.strip().lower())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO pool (id, title, year, summary, source_url, source_name,
                          status, harvested_at, note)
        VALUES (?, ?, ?, ?, ?, 'editor-suggestion', 'new', ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          note = excluded.note,
          status = 'new'
        """,
        (pool_id, title.strip(), year, note or "", source_url, now, note or ""),
    )
    conn.commit()
    return pool_id


def defer_draft(conn: sqlite3.Connection, draft_id: str, days: int = 30) -> None:
    """Push a pending draft down the queue for ``days`` (the /later command).

    Sets ``drafts.defer_until`` to now+days. ``pending_drafts`` sorts deferred
    drafts after their window, so they stay out of the review surface until
    then. Raises ValueError if the draft is unknown or already reviewed.
    """
    row = conn.execute("SELECT status FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    if row is None:
        raise ValueError(f"no draft with id {draft_id!r}")
    if row["status"] != "pending":
        raise ValueError(f"draft {draft_id!r} has status {row['status']!r}; cannot defer")
    when = datetime.now(timezone.utc) + timedelta(days=days)
    conn.execute(
        "UPDATE drafts SET defer_until = ? WHERE id = ?",
        (when.isoformat(), draft_id),
    )
    conn.commit()


def clear_defer(conn: sqlite3.Connection, draft_id: str) -> None:
    """Clear a draft's defer_until (used when it is reviewed/acted on)."""
    conn.execute("UPDATE drafts SET defer_until = NULL WHERE id = ?", (draft_id,))
    conn.commit()


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
