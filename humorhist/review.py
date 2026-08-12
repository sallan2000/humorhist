"""Phase 3 review gate: capture a human approve/reject/annotate decision.

This module is deliberately transport-agnostic. The CLI review loop
(``humorhist.cli.cmd_review``) and any future Telegram transport both call
``apply_review()`` -- all the durable state transitions live here so they can
be unit-tested without a network or a tty.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import humorhist.db as db

# Decisions the review gate understands. ``skip`` is handled by the caller
# (it means "leave pending, move on") and never reaches apply_review.
APPROVE = "approve"
REJECT = "reject"
_VALID_DECISIONS = {APPROVE, REJECT}

# Statuses that are still within the review gate. Once a draft has left this
# set (e.g. Phase 4 marked it "used") it is no longer reviewable here.
_REVIEWABLE_STATUSES = {"pending", "approved", "rejected"}


def pending_drafts(conn: sqlite3.Connection) -> list[dict]:
    """Return all drafts with status 'pending', review-order friendly.

    Deferred drafts (``defer_until`` set in the future) sort *after* the
    non-deferred ones, so they stay out of the review surface until their
    window opens. Drafts whose defer window has already passed sort as normal
    pending (the /later window expired).
    """
    now = datetime.now(timezone.utc).isoformat()
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT * FROM drafts
            WHERE status = 'pending'
            ORDER BY
              CASE WHEN defer_until IS NOT NULL AND defer_until > ? THEN 1 ELSE 0 END,
              created_at ASC,
              id ASC
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
        raise ValueError(
            f"decision must be one of {sorted(_VALID_DECISIONS)}, got {decision!r}"
        )

    row = conn.execute("SELECT status FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    if row is None:
        raise ValueError(f"no draft with id {draft_id!r}")
    if row["status"] not in _REVIEWABLE_STATUSES:
        raise ValueError(
            f"draft {draft_id!r} has status {row['status']!r}; not reviewable"
        )

    new_status = "approved" if decision == APPROVE else "rejected"
    reviewed_at = datetime.now(timezone.utc).isoformat()

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
    conn.commit()
    # Keep the Phase 4 queue consistent with the decision: an approval is
    # immediately ready to publish; a rejection (or an approve->reject flip)
    # must not linger in the queue.
    if decision == APPROVE:
        enqueue_approved(conn)
    else:
        remove_from_queue(conn, draft_id)


def remove_from_queue(conn: sqlite3.Connection, draft_id: str) -> int:
    """Delete any queue row for `draft_id` (e.g. after a reject/flip). Returns count removed."""
    cur = conn.execute("DELETE FROM queue WHERE draft_id = ?", (draft_id,))
    if cur.rowcount:
        conn.commit()
    return cur.rowcount


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

    When `image_dir` is given, each newly-queued draft also gets a best-effort
    story image (and a persisted 'learn more' source link) — this is the
    *publish-time* generation step (image moved off the approve flow so it aligns
    with "about to be published"). Both are best-effort: a missing image
    credential or generation failure is logged and skipped, never blocking the
    enqueue.
    """
    approved = conn.execute(
        "SELECT id FROM drafts WHERE status = 'approved'"
    ).fetchall()
    already = {
        r["draft_id"]
        for r in conn.execute("SELECT draft_id FROM queue")
    }
    inserted = 0
    for r in approved:
        draft_id = r["id"]
        if draft_id in already:
            continue
        conn.execute(
            "INSERT INTO queue (draft_id, scheduled_for, published) VALUES (?, ?, 0)",
            (draft_id, scheduled_for),
        )
        inserted += 1
        # Publish-time artifacts (best-effort): shortened 'learn more' link +
        # story image. None of this blocks the enqueue.
        _populate_publish_artifacts(conn, draft_id, image_dir=image_dir)
    if inserted:
        conn.commit()
    # When an image dir is supplied (the explicit publish/enqueue step), also
    # backfill artifacts for drafts already sitting in the queue without them —
    # e.g. rows auto-enqueued at approve time (where image_dir was not yet known,
    # or an image/link was skipped). This keeps the publish step idempotent and
    # makes it the single place images are generated.
    if image_dir:
        for r in conn.execute(
            "SELECT draft_id FROM queue WHERE image_path IS NULL"
        ).fetchall():
            _populate_publish_artifacts(conn, r["draft_id"], image_dir=image_dir)
    return inserted


def _populate_publish_artifacts(
    conn: sqlite3.Connection, draft_id: str, *, image_dir: str | None = None
) -> None:
    """Best-effort: persist a 'learn more' source link (always, if a source
    exists) and a story image (only if ``image_dir`` is given and image is
    available).

    A reader-facing convenience: the link points at the original article
    (Wikipedia, etc.) behind the story; the image is generated via FAL FLUX.
    Either failing is logged and swallowed — the draft still enters the queue.
    """
    # 1) Shortened 'learn more' link (no network; derived from pool.source_url).
    try:
        import humorhist.db as db

        url, name = db.pool_source_url(conn, draft_id)
        if url:
            db.set_source_link(conn, draft_id, db.shorten_url(url, shorten=False))
    except Exception as exc:  # noqa: BLE001
        print(f"[queue] source link not set for {draft_id}: {exc}")

    # 2) Story image (best-effort; only when a target dir is supplied).
    if image_dir:
        try:
            from humorhist import imagegen as ig

            ig.generate_image_for_queue(conn, draft_id, out_dir=image_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[queue] image not generated for {draft_id}: {exc}")


def queued_drafts(conn: sqlite3.Connection) -> list[dict]:
    """Return queued (unpublished) drafts, oldest first, joined to pool title."""
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT q.id AS queue_id, q.draft_id, q.scheduled_for, q.published,
                   q.post_copy, p.title AS title
            FROM queue q
            LEFT JOIN drafts d ON d.id = q.draft_id
            LEFT JOIN pool p ON p.id = d.pool_id
            ORDER BY q.id ASC
            """
        )
    ]


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
                   p.title AS title
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
                   p.title AS title
            FROM drafts d
            LEFT JOIN pool p ON p.id = d.pool_id
            WHERE d.status = 'rejected'
            ORDER BY d.reviewed_at DESC, d.id ASC
            """
        )
    ]


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
    conn.commit()
    # Keep the queue consistent: a reopened draft is not approved, so it must
    # not linger in the publish queue.
    remove_from_queue(conn, draft_id)


def defer_draft(conn: sqlite3.Connection, draft_id: str, days: int = 30) -> None:
    """Defer a pending draft for ``days`` (the /later command). See ``db.defer_draft``."""
    db.defer_draft(conn, draft_id, days=days)


def clear_defer(conn: sqlite3.Connection, draft_id: str) -> None:
    """Clear a draft's defer_until (convenience over ``db.clear_defer``)."""
    db.clear_defer(conn, draft_id)
