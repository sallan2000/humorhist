"""Phase 3 review gate: capture a human approve/reject/annotate decision.

This module is deliberately transport-agnostic. The CLI review loop
(``humorhist.cli.cmd_review``) and any future Telegram transport both call
``apply_review()`` -- all the durable state transitions live here so they can
be unit-tested without a network or a tty.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from functools import wraps

import humorhist.db as db

# Decisions the review gate understands. ``skip`` is handled by the caller
# (it means "leave pending, move on") and never reaches apply_review.
APPROVE = "approve"
REJECT = "reject"
_VALID_DECISIONS = {APPROVE, REJECT}

# Statuses that are still within the review gate. Once a draft has left this
# set (e.g. Phase 4 marked it "used") it is no longer reviewable here.
_REVIEWABLE_STATUSES = {"pending", "approved", "rejected"}


@contextmanager
def _transaction(conn: sqlite3.Connection):
    """Keep caller-owned transactions open while making our work atomic."""
    conn.execute("SAVEPOINT review_operation")
    try:
        yield
        conn.execute("RELEASE SAVEPOINT review_operation")
    except BaseException:
        # SQLite can already have rolled back the entire transaction (e.g. a
        # RAISE(ROLLBACK) trigger); do not mask the original error in cleanup.
        with suppress(sqlite3.Error):
            conn.execute("ROLLBACK TO SAVEPOINT review_operation")
            conn.execute("RELEASE SAVEPOINT review_operation")
        raise


def _atomic(func):
    @wraps(func)
    def wrapper(conn, *args, **kwargs):
        with _transaction(conn):
            return func(conn, *args, **kwargs)
    return wrapper


def pending_drafts(conn: sqlite3.Connection) -> list[dict]:
    """Return pending drafts ready for review, oldest first.

    Future-deferred drafts remain pending in the database but are excluded
    until their defer window opens. Expired deferrals sort as normal pending.
    """
    now = datetime.now(UTC).isoformat()
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT * FROM drafts
            WHERE status = 'pending'
              AND (defer_until IS NULL OR defer_until <= ?)
            ORDER BY created_at ASC, id ASC
            """,
            (now,),
        )
    ]


def reviewed_summary(conn: sqlite3.Connection) -> dict:
    """Return a review-progress snapshot keyed by status.

    For each of pending/approved/rejected, gives a count and the list of topic
    titles (pool.title) so a reviewer can see what's been decided. Approved and
    rejected rows are the "reviewed" topics; pending are still open.
    """
    rows = conn.execute(
        """
        SELECT d.status AS status, p.title AS title
        FROM drafts d
        LEFT JOIN pool p ON p.id = d.pool_id
        ORDER BY d.status, p.title
        """
    ).fetchall()

    summary: dict[str, dict] = {
        "pending": {"count": 0, "titles": []},
        "approved": {"count": 0, "titles": []},
        "rejected": {"count": 0, "titles": []},
    }
    for r in rows:
        status = r["status"]
        if status not in summary:
            continue
        summary[status]["count"] += 1
        summary[status]["titles"].append(r["title"] or "(unknown)")
    return summary


@_atomic
def apply_review(
    conn: sqlite3.Connection,
    draft_id: str,
    *,
    decision: str,
    editor_line: str | None = None,
    notes: str | None = None,
    merge: bool = False,
) -> None:
    """Record a human review decision for one draft.

    Sets ``drafts.status`` to approved/rejected, stamps ``reviewed_at``, and
    stores an optional ``editor_line`` (a one-line steer for the writing pass)
    and ``editor_notes``. Idempotent on the same decision; re-reviewing lets the
    editor flip approve<->reject and update notes.

    When ``merge=True`` (used when only *some* fields are supplied by a later
    interaction, e.g. the Telegram "Add notes" button), any field left as
    ``None`` keeps its existing value instead of being overwritten. This stops a
    notes-only re-apply from clobbering a previously-entered ``editor_line``.

    Raises ``ValueError`` on a bad decision, an unknown draft id, or a draft
    whose status is outside the review gate.
    """
    decision = (decision or "").strip().lower()
    if decision not in _VALID_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(_VALID_DECISIONS)}, got {decision!r}")

    row = conn.execute("SELECT status FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    if row is None:
        raise ValueError(f"no draft with id {draft_id!r}")
    if row["status"] not in _REVIEWABLE_STATUSES:
        raise ValueError(f"draft {draft_id!r} has status {row['status']!r}; not reviewable")

    new_status = "approved" if decision == APPROVE else "rejected"
    reviewed_at = datetime.now(UTC).isoformat()

    # Resolve the field values to write. A plain apply (merge=False) overwrites
    # both fields with whatever was passed (None => cleared). A merge applies
    # only the fields the caller actually supplied, keeping the others.
    if merge:
        cur = conn.execute(
            "SELECT editor_line, editor_notes FROM drafts WHERE id = ?",
            (draft_id,),
        ).fetchone()
        cur_line = (cur["editor_line"] if cur else None) or ""
        cur_notes = (cur["editor_notes"] if cur else None) or ""
        editor_line, editor_notes = (
            (editor_line.strip() if editor_line else cur_line.strip()) or None,
            (notes.strip() if notes else cur_notes.strip()) or None,
        )
    else:
        editor_line, editor_notes = (
            editor_line.strip() if editor_line else None,
            notes.strip() if notes else None,
        )

    conn.execute(
        """
        UPDATE drafts
        SET status = ?, reviewed_at = ?, editor_line = ?, editor_notes = ?,
            defer_until = NULL
        WHERE id = ?
        """,
        (
            new_status,
            reviewed_at,
            editor_line,
            editor_notes,
            draft_id,
        ),
    )
    # Keep the Phase 4 queue consistent with the decision: an approval is
    # immediately ready to publish; a rejection (or an approve->reject flip)
    # must not linger in the queue.
    if decision == APPROVE:
        enqueue_approved(conn)
    else:
        remove_from_queue(conn, draft_id)


@_atomic
def remove_from_queue(conn: sqlite3.Connection, draft_id: str) -> int:
    """Delete any queue row for `draft_id` (e.g. after a reject/flip). Returns count removed."""
    if conn.execute("SELECT 1 FROM queue WHERE draft_id = ? AND published = 1", (draft_id,)).fetchone():
        raise ValueError("undo publication before removing this draft from the queue")
    cur = conn.execute("DELETE FROM queue WHERE draft_id = ?", (draft_id,))
    return cur.rowcount


def _publication_row(conn: sqlite3.Connection, draft_id: str) -> sqlite3.Row:
    draft = conn.execute("SELECT status FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    if draft is None:
        raise ValueError(f"no draft with id {draft_id!r}")
    if draft["status"] != "approved":
        raise ValueError(f"draft {draft_id!r} is not approved")
    rows = conn.execute("SELECT id, published, published_at FROM queue WHERE draft_id = ?", (draft_id,)).fetchall()
    if len(rows) != 1:
        raise ValueError(f"draft {draft_id!r} must have exactly one queue row")
    if rows[0]["published"] not in (0, 1):
        raise ValueError(f"draft {draft_id!r} has an invalid publication state")
    return rows[0]


@_atomic
def mark_posted(conn: sqlite3.Connection, draft_id: str) -> str:
    """Record a manual post and return its UTC ISO timestamp; never post remotely.

    Repeating the action returns the original timestamp. Legacy published rows
    with no timestamp are not claimed as manual records. No copy is required.
    """
    row = _publication_row(conn, draft_id)
    if row["published"] == 1:
        if row["published_at"] is None:
            raise ValueError("already published without a manual publication timestamp")
        return row["published_at"]
    if row["published_at"] is not None:
        raise ValueError("unpublished row has an inconsistent publication timestamp")
    now = datetime.now(UTC).isoformat()
    cur = conn.execute("UPDATE queue SET published = 1, published_at = ? WHERE id = ?", (now, row["id"]))
    if cur.rowcount != 1:
        raise ValueError("publication was not recorded")
    return now


@_atomic
def undo_posted(conn: sqlite3.Connection, draft_id: str, *, expected_published_at: str | None = None) -> bool:
    """Undo a manual record; return True if changed, False if already undone.

    Pass the timestamp shown on a confirmation button to prevent a stale undo
    from reversing a later publication. Legacy published records cannot be undone.
    """
    row = _publication_row(conn, draft_id)
    if row["published"] == 0:
        if row["published_at"] is not None:
            raise ValueError("unpublished row has an inconsistent publication timestamp")
        return False
    if row["published_at"] is None:
        raise ValueError("cannot undo a legacy publication without a timestamp")
    if expected_published_at is not None and expected_published_at != row["published_at"]:
        raise ValueError("publication has changed since confirmation")
    cur = conn.execute("UPDATE queue SET published = 0, published_at = NULL WHERE id = ?", (row["id"],))
    if cur.rowcount != 1:
        raise ValueError("publication undo was not recorded")
    return True


def enqueue_approved(
    conn: sqlite3.Connection,
    scheduled_for: str | None = None,
    *,
    image_dir: str | None = None,
) -> int:
    """Move every `approved` draft into `queue` (Phase 4 handoff).

    Idempotent: drafts already in `queue` are skipped, so this is safe to run
    repeatedly. Returns the number of *new* rows inserted into `queue`.

    `scheduled_for` is an optional ISO timestamp; when omitted the row is left
    unscheduled (published = 0) for the publisher to pick up in arrival order.

    Source links are populated best-effort without network access. `image_dir`
    is accepted for compatibility but never triggers image generation; use an
    explicit image command to avoid bulk charges and automatic retries.
    """
    inserted_ids = []
    with _transaction(conn):
        approved = conn.execute("SELECT id FROM drafts WHERE status = 'approved'").fetchall()
        already = {r["draft_id"] for r in conn.execute("SELECT draft_id FROM queue")}
        for r in approved:
            draft_id = r["id"]
            if draft_id in already:
                continue
            conn.execute(
                "INSERT INTO queue (draft_id, scheduled_for, published) VALUES (?, ?, 0)",
                (draft_id, scheduled_for),
            )
            inserted_ids.append(draft_id)
    # Source links are local; image generation is always an explicit operation.
    for draft_id in inserted_ids:
        _populate_publish_artifacts(conn, draft_id)
    # Preserve the old link-backfill behavior without generating paid images.
    if image_dir:
        for r in conn.execute("SELECT draft_id FROM queue WHERE source_link IS NULL AND published = 0").fetchall():
            _populate_publish_artifacts(conn, r["draft_id"])
    return len(inserted_ids)


def _populate_publish_artifacts(conn: sqlite3.Connection, draft_id: str) -> None:
    """Best-effort: persist a local 'learn more' source link if one exists."""
    # 1) Shortened 'learn more' link (no network; derived from pool.source_url).
    try:
        import humorhist.db as db

        url, name = db.pool_source_url(conn, draft_id)
        if url:
            with _transaction(conn):
                conn.execute(
                    "UPDATE queue SET source_link = ? WHERE draft_id = ?",
                    (db.shorten_url(url, shorten=False), draft_id),
                )
    except Exception as exc:  # noqa: BLE001
        print(f"[queue] source link not set for {draft_id}: {exc}")

def queued_drafts(conn: sqlite3.Connection) -> list[dict]:
    """Return queued (unpublished) drafts, oldest first, joined to pool title.

    Includes ``source_link`` and ``image_path`` so the read-only ``publish``
    export can show the "learn more" link and story image when present.
    """
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT q.id AS queue_id, q.draft_id, q.scheduled_for, q.published,
                   q.post_copy, q.source_link, q.image_path, p.title AS title,
                   d.short_code AS short_code
            FROM queue q
            LEFT JOIN drafts d ON d.id = q.draft_id
            LEFT JOIN pool p ON p.id = d.pool_id
            WHERE q.published = 0
            ORDER BY q.id ASC
            """
        )
    ]


def format_publish_export(conn: sqlite3.Connection, rows: list[dict], *, include_image_path: bool = True) -> str:
    """Render the read-only 'publish' export for queued, copy-bearing drafts.

    Pure formatting — sends nothing, writes nothing. Each row shows the post
    copy, the 'learn more' source link and story image path (when present), and
    a readiness checklist (``copy OK | joke OK | image ...``) so the editor can
    see at a glance which queued posts are paste-ready. Shared by the CLI
    (``cmd_publish``) and the Telegram ``/publish`` command so the two can't
    drift; callers filter to copy-bearing rows first. When image paths are
    hidden, report only that a path is recorded, not that the file is usable.
    """
    joke_by_draft: dict[str, str | None] = {}
    for r in rows:
        d = conn.execute("SELECT editor_line FROM drafts WHERE id = ?", (r["draft_id"],)).fetchone()
        joke_by_draft[r["draft_id"]] = (d["editor_line"] if d else None)
    parts: list[str] = [f"{len(rows)} post(s) ready to paste manually:", ""]
    for r in rows:
        short = r.get("short_code") or r["draft_id"]
        parts.append(f"── {short} — {r['title']}")
        parts.append(r["post_copy"])
        link = r.get("source_link")
        if link:
            parts.append(f"learn more: {link}")
        img = r.get("image_path")
        if img and include_image_path:
            parts.append(f"image: {img}")
        has_joke = bool(joke_by_draft.get(r["draft_id"]))
        has_img = bool(img)
        image_status = "image none (optional)"
        if has_img:
            image_status = "image OK" if include_image_path else "image path recorded (availability not checked)"
        flags = [
            "copy OK",
            "joke OK" if has_joke else "joke MISSING",
            image_status,
        ]
        parts.append("  readiness: " + " | ".join(flags))
        parts.append("")
    parts.append("Post these by hand on your platforms. Nothing was sent automatically.")
    return "\n".join(parts)


def deferred_drafts(conn: sqlite3.Connection) -> list[dict]:
    """Return pending drafts currently deferred (``defer_until`` in the future).

    Used by the Telegram ``/listlater`` command so a deferred draft can be
    brought forward with ``/reviewnow`` (or the ``reviewnow:<id>`` button) —
    the missing "review now" shortcut for a ``/later`` defer (GAP 4). Ordered
    by how soon each defer window opens (soonest first).
    """
    now = datetime.now(UTC).isoformat()
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT d.id AS draft_id, p.title AS title, d.defer_until,
                   d.short_code AS short_code
            FROM drafts d
            LEFT JOIN pool p ON p.id = d.pool_id
            WHERE d.status = 'pending'
              AND d.defer_until IS NOT NULL
              AND d.defer_until > ?
            ORDER BY d.defer_until ASC, d.id ASC
            """,
            (now,),
        )
    ]


def bring_forward(conn: sqlite3.Connection, draft_id: str | None = None) -> int:
    """Clear the defer window on deferred draft(s) so they re-enter review.

    When ``draft_id`` is given, clears just that draft (raising ValueError if
    it's unknown or not deferred). When ``None``, clears every currently
    deferred draft in a single UPDATE. Returns the number of drafts brought
    forward.
    """
    if draft_id is not None:
        row = conn.execute("SELECT status, defer_until FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        if row is None:
            raise ValueError(f"no draft with id {draft_id!r}")
        if not row["defer_until"]:
            raise ValueError(f"draft {draft_id!r} is not deferred")
        db.clear_defer(conn, draft_id)
        return 1
    return db.clear_deferred(conn)


def approved_drafts(conn: sqlite3.Connection) -> list[dict]:
    """Return approved drafts (greenlit), oldest first, joined to pool title.

    Used by the Telegram /listapproved command so the editor can browse what
    they've greenlit and add more notes to any of them.
    """
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT d.id AS draft_id, d.editor_line, d.editor_notes, d.reviewed_at,
                   p.title AS title, d.short_code AS short_code
            FROM drafts d
            LEFT JOIN pool p ON p.id = d.pool_id
            WHERE d.status = 'approved'
            ORDER BY d.reviewed_at ASC, d.id ASC
            """
        )
    ]


def rejected_drafts(conn: sqlite3.Connection) -> list[dict]:
    """Return rejected drafts, newest-reviewed first, joined to pool title.

    Used by the Telegram /listrejected command so a mistaken reject can be
    sent back for re-review (the editor's undo for a bad tap).
    """
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT d.id AS draft_id, d.editor_line, d.editor_notes, d.reviewed_at,
                   p.title AS title, d.short_code AS short_code
            FROM drafts d
            LEFT JOIN pool p ON p.id = d.pool_id
            WHERE d.status = 'rejected'
            ORDER BY d.reviewed_at DESC, d.id ASC
            """
        )
    ]


@_atomic
def reopen_draft(conn: sqlite3.Connection, draft_id: str) -> None:
    """Send a rejected (or approved) draft back to 'pending' for re-review.

    Clears the decision (reviewed_at, editor_line, editor_notes) and any defer
    window, and removes it from the publish queue so it re-enters the review
    surface cleanly. This is the editor's undo for a mistaken reject/approve.

    Raises ValueError on an unknown id, or if the draft is already pending
    (nothing to reopen).
    """
    row = conn.execute("SELECT status FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    if row is None:
        raise ValueError(f"no draft with id {draft_id!r}")
    if row["status"] == "pending":
        raise ValueError(f"draft {draft_id!r} is already pending")
    conn.execute(
        """
        UPDATE drafts
        SET status = 'pending', reviewed_at = NULL, editor_line = NULL,
            editor_notes = NULL, defer_until = NULL
        WHERE id = ?
        """,
        (draft_id,),
    )
    # Keep the queue consistent: a reopened draft is not approved, so it must
    # not linger in the publish queue.
    remove_from_queue(conn, draft_id)


def defer_draft(conn: sqlite3.Connection, draft_id: str, days: int = 30) -> None:
    """Defer a pending draft for ``days`` (the /later command). See ``db.defer_draft``."""
    db.defer_draft(conn, draft_id, days=days)


def clear_defer(conn: sqlite3.Connection, draft_id: str) -> None:
    """Clear a draft's defer_until (convenience over ``db.clear_defer``)."""
    db.clear_defer(conn, draft_id)


def stuck_captures(conn: sqlite3.Connection) -> list[dict]:
    """Return approved+queued drafts that still lack a human ``editor_line``.

    These were committed (the editor tapped Approve) but they walked away
    before sending their one-line joke, so the draft sits in the publish queue
    with a blank human voice. Surfaced by the status surfaces (CLI ``status`` /
    Telegram ``/status``) as a nudge, and fixed with ``set_editor_line`` (CLI
    ``setjoke`` / Telegram ``/setjoke``).
    """
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT d.id AS draft_id, p.title AS title, d.short_code AS short_code
            FROM drafts d
            LEFT JOIN pool p ON p.id = d.pool_id
            WHERE d.status = 'approved'
              AND d.editor_line IS NULL
              AND EXISTS (SELECT 1 FROM queue q WHERE q.draft_id = d.id AND q.published = 0)
            ORDER BY d.reviewed_at ASC, d.id ASC
            """
        )
    ]


def set_editor_line(conn: sqlite3.Connection, draft_id: str, editor_line: str | None) -> None:
    """Set (or clear, with empty/None) the human ``editor_line`` on a draft.

    Does NOT change the draft's status or touch the queue — it's purely the
    fix for a "stuck capture" (an approved draft whose joke was never filled).
    Raises ValueError on an unknown id.
    """
    row = conn.execute("SELECT id FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    if row is None:
        raise ValueError(f"no draft with id {draft_id!r}")
    conn.execute(
        "UPDATE drafts SET editor_line = ? WHERE id = ?",
        (editor_line.strip() if editor_line else None, draft_id),
    )
    conn.commit()
