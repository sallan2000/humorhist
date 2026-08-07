"""SQLite schema layer for the humorhist content pipeline.

Only stdlib sqlite3 is used. All SQL is parameterised; the single place a
table name is interpolated (set_status) validates it against a whitelist
first.
"""

from __future__ import annotations

import hashlib
import re
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
  note TEXT,
  normalized_title TEXT
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
    # Dedup: normalized_title lets the same event arriving from different
    # sources / spellings collapse to one pool row (see upsert_pool_item).
    _ensure_normalized_title_column(conn)
    _backfill_normalized_title(conn)
    _dedupe_pool(conn)
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


def _ensure_normalized_title_column(conn: sqlite3.Connection) -> None:
    """Add pool.normalized_title if absent (no-op if present)."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(pool)")}
    if "normalized_title" not in existing:
        conn.execute("ALTER TABLE pool ADD COLUMN normalized_title TEXT")


def _backfill_normalized_title(conn: sqlite3.Connection) -> None:
    """Populate normalized_title for any legacy rows that lack it.

    Pure-junk rows (normalize to "") are dropped rather than kept.
    """
    # sqlite can't call python normalize_title per row, so loop.
    rows = conn.execute(
        "SELECT id, title FROM pool WHERE normalized_title IS NULL"
    ).fetchall()
    for r in rows:
        norm = normalize_title(r["title"])
        if norm:
            conn.execute(
                "UPDATE pool SET normalized_title = ? WHERE id = ?", (norm, r["id"])
            )
        else:
            conn.execute("DELETE FROM pool WHERE id = ?", (r["id"],))
    conn.commit()


def _dedupe_pool(conn: sqlite3.Connection) -> int:
    """Collapse existing duplicate pool rows (same normalized_title).

    Picks a survivor by rank (has-a-draft > status used>drafted>rejected>new >
    higher funny_score > stable id), merges the best fields into it, re-points any
    orphaned drafts to the survivor, and deletes the rest. Returns the number of
    rows removed. Safe to call repeatedly (idempotent).

    Pure-junk rows (normalized_title IS NULL after backfill) are dropped unless a
    draft depends on them.
    """
    # 1) drop junk rows that have no draft
    for r in conn.execute(
        "SELECT id FROM pool WHERE normalized_title IS NULL"
    ).fetchall():
        if conn.execute("SELECT 1 FROM drafts WHERE pool_id=? LIMIT 1", (r["id"],)).fetchone() is None:
            conn.execute("DELETE FROM pool WHERE id = ?", (r["id"],))

    _STATUS_RANK = {"used": 4, "drafted": 3, "rejected": 2, "new": 1}

    def _rank(row: sqlite3.Row) -> tuple:
        has_draft = conn.execute(
            "SELECT 1 FROM drafts WHERE pool_id=? LIMIT 1", (row["id"],)
        ).fetchone() is not None
        return (
            1 if has_draft else 0,
            _STATUS_RANK.get(row["status"], 0),
            row["funny_score"] or 0,
            row["id"],
        )

    removed = 0
    groups = conn.execute(
        "SELECT normalized_title, COUNT(*) n FROM pool "
        "WHERE normalized_title IS NOT NULL GROUP BY normalized_title HAVING n > 1"
    ).fetchall()
    for g in groups:
        rows = conn.execute(
            "SELECT * FROM pool WHERE normalized_title = ?", (g["normalized_title"],)
        ).fetchall()
        rows = sorted(rows, key=_rank, reverse=True)
        survivor, others = rows[0], rows[1:]
        # merge best fields into survivor (without clobbering reviewed work)
        upd: dict[str, object] = {}
        for o in others:
            if o["year"] is not None and survivor["year"] is None:
                upd["year"] = o["year"]
            if o["summary"] and not survivor["summary"]:
                upd["summary"] = o["summary"]
            if o["source_url"] and not survivor["source_url"]:
                upd["source_url"] = o["source_url"]
            if o["source_name"] and not survivor["source_name"]:
                upd["source_name"] = o["source_name"]
            if o["funny_score"] is not None and (
                survivor["funny_score"] is None or o["funny_score"] > survivor["funny_score"]
            ):
                upd["funny_score"] = o["funny_score"]
            if o["note"] and not survivor["note"]:
                upd["note"] = o["note"]
        if upd:
            set_clause = ", ".join(f"{k} = ?" for k in upd)
            conn.execute(
                f"UPDATE pool SET {set_clause} WHERE id = ?",
                (*upd.values(), survivor["id"]),
            )
        # re-point any orphaned drafts to the survivor, then delete the others
        for o in others:
            conn.execute(
                "UPDATE drafts SET pool_id = ? WHERE pool_id = ?",
                (survivor["id"], o["id"]),
            )
            conn.execute("DELETE FROM pool WHERE id = ?", (o["id"],))
            removed += 1
    conn.commit()
    return removed


def dedupe_pool(conn: sqlite3.Connection) -> int:
    """Public helper: collapse duplicate pool rows. Returns rows removed.

    Also safe to call manually (e.g. after bulk edits) — it is idempotent.
    """
    return _dedupe_pool(conn)


def _ensure_pool_norm_index(conn: sqlite3.Connection) -> None:
    """Add a UNIQUE index on normalized_title (safety net vs. the merge logic).

    Created only AFTER duplicates are collapsed, so it never fails on legacy data.
    Rows with a NULL normalized_title are excluded from the unique constraint.
    """
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_pool_norm "
        "ON pool(normalized_title) WHERE normalized_title IS NOT NULL"
    )


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


def normalize_title(title: str | None) -> str:
    """Normalize a title for cross-source/duplicate detection.

    Lowercases, strips wiki-link junk (``[http...]``), URLs and parentheticals,
    removes punctuation and collapses whitespace. Two different sources/spellings
    of the *same* event should map to the same normalized string so they collapse
    to one pool row. Returns ``""`` for pure-junk titles (no real words) so they
    are dropped rather than inserted.
    """
    if not title:
        return ""
    t = title.lower().strip()
    t = re.sub(r"\[http[^\]]*\]", " ", t)        # [http://...] wiki link junk
    t = re.sub(r"https?://\S+", " ", t)          # bare URLs
    t = re.sub(r"\([^)]*\)", " ", t)             # parentheticals
    t = re.sub(r"[^\w\s]", " ", t)               # punctuation -> space
    t = re.sub(r"\s+", " ", t).strip()
    return t


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
    funny_score: float | None = None,
) -> bool:
    """Insert a pool item, merging with any existing row for the same event.

    Dedup is by *normalized* title (``normalize_title``), not by the synthetic
    ``id`` — so the same historical event arriving from a different source or
    under a different spelling collapses to ONE pool row instead of producing
    duplicate drafts. Returns True only if a brand-new row was inserted.

    Merge / tie-breaker rules (protect reviewed work):
      * If a row already exists for the normalized title:
          - The survivor is chosen by rank: has-a-draft > status (used >
            drafted > rejected > new) > higher funny_score > stable id.
          - The incoming data is folded in WITHOUT overwriting reviewed work:
            adopt its funny_score only if the survivor has none / a lower one,
            and fill empty fields (year/summary/source_url) from the incoming row.
          - Two already-actioned rows (e.g. both drafted) are never force-merged
            in a way that destroys one: the highest-ranked survivor keeps its
            draft; the other's draft (if any) is re-pointed to the survivor.
      * Pure-junk titles (normalize to "") are dropped (return False).
    """
    norm = normalize_title(title)
    if not norm:
        return False
    # Idempotent on the synthetic id too: if a row with this exact id already
    # exists (e.g. re-running the same harvest row), just make sure its
    # normalized_title is populated and return False rather than inserting a twin.
    existing_by_id = conn.execute(
        "SELECT id, normalized_title FROM pool WHERE id = ?", (id,)
    ).fetchone()
    if existing_by_id is not None:
        if not existing_by_id["normalized_title"]:
            conn.execute(
                "UPDATE pool SET normalized_title = ? WHERE id = ?", (norm, id)
            )
            conn.commit()
        return False
    existing = conn.execute(
        "SELECT * FROM pool WHERE normalized_title = ?", (norm,)
    ).fetchone()
    if existing is None:
        conn.execute(
            """
            INSERT INTO pool
              (id, title, year, date_hint, summary, source_url, source_name,
               funny_score, status, harvested_at, normalized_title)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, ?)
            """,
            (id, title, year, date_hint, summary, source_url, source_name,
             funny_score, datetime.now(timezone.utc).isoformat(), norm),
        )
        conn.commit()
        return True

    # Merge into the existing survivor, protecting reviewed work.
    upd: dict[str, object] = {}
    if year is not None and existing["year"] is None:
        upd["year"] = year
    if summary and not existing["summary"]:
        upd["summary"] = summary
    if source_url and not existing["source_url"]:
        upd["source_url"] = source_url
    if source_name and not existing["source_name"]:
        upd["source_name"] = source_name
    if funny_score is not None and (
        existing["funny_score"] is None or funny_score > existing["funny_score"]
    ):
        upd["funny_score"] = funny_score
    if upd:
        set_clause = ", ".join(f"{k} = ?" for k in upd)
        conn.execute(
            f"UPDATE pool SET {set_clause} WHERE id = ?",
            (*upd.values(), existing["id"]),
        )
        conn.commit()
    return False


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
    through the normal draft pipeline. Idempotent on the stable id (sha1 of the
    title) — re-suggesting the same topic updates the note. If a pool row for the
    *same event* already exists (from any source), the suggestion is folded into
    that row (note + status reset to 'new') rather than creating a twin, so the
    editor never reviews the same event twice.
    """
    norm = normalize_title(title)
    if not norm:
        # pure junk -> still store with a generated id so /suggest doesn't 500,
        # but normalized_title stays NULL so it never collides with real rows.
        pool_id = make_id("suggest", title.strip().lower())
        conn.execute(
            """
            INSERT INTO pool (id, title, year, summary, source_url, source_name,
                              status, harvested_at, note)
            VALUES (?, ?, ?, ?, ?, 'editor-suggestion', 'new', ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              note = excluded.note,
              status = 'new'
            """,
            (pool_id, title.strip(), year, note or "", source_url,
             datetime.now(timezone.utc).isoformat(), note or ""),
        )
        conn.commit()
        return pool_id

    existing = conn.execute(
        "SELECT id FROM pool WHERE normalized_title = ?", (norm,)
    ).fetchone()
    if existing is not None:
        # fold the suggestion into the existing event row; bump it back to 'new'
        # so it (re)enters the draft pipeline, and keep the human's note.
        conn.execute(
            "UPDATE pool SET status='new', note=?, source_name='editor-suggestion' "
            "WHERE id = ?",
            (note or "", existing["id"]),
        )
        conn.commit()
        return existing["id"]

    pool_id = make_id("suggest", title.strip().lower())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO pool (id, title, year, summary, source_url, source_name,
                          status, harvested_at, note, normalized_title)
        VALUES (?, ?, ?, ?, ?, 'editor-suggestion', 'new', ?, ?, ?)
        """,
        (pool_id, title.strip(), year, note or "", source_url, now, note or "", norm),
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
