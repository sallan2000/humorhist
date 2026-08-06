"""Phase 3 review gate: capture a human approve/reject/annotate decision.

This module is deliberately transport-agnostic. The CLI review loop
(``humorhist.cli.cmd_review``) and any future Telegram transport both call
``apply_review()`` -- all the durable state transitions live here so they can
be unit-tested without a network or a tty.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

# Decisions the review gate understands. ``skip`` is handled by the caller
# (it means "leave pending, move on") and never reaches apply_review.
APPROVE = "approve"
REJECT = "reject"
_VALID_DECISIONS = {APPROVE, REJECT}

# Statuses that are still within the review gate. Once a draft has left this
# set (e.g. Phase 4 marked it "used") it is no longer reviewable here.
_REVIEWABLE_STATUSES = {"pending", "approved", "rejected"}


def pending_drafts(conn: sqlite3.Connection) -> list[dict]:
    """Return all drafts with status 'pending', oldest first."""
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM drafts WHERE status = 'pending' ORDER BY created_at ASC, id ASC"
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
) -> None:
    """Record a human review decision for one draft.

    Sets ``drafts.status`` to approved/rejected, stamps ``reviewed_at``, and
    stores an optional ``editor_line`` (a one-line steer for the writing pass)
    and ``editor_notes``. Idempotent on the same decision; re-reviewing lets the
    editor flip approve<->reject and update notes.

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

    conn.execute(
        """
        UPDATE drafts
        SET status = ?, reviewed_at = ?, editor_line = ?, editor_notes = ?
        WHERE id = ?
        """,
        (
            new_status,
            reviewed_at,
            editor_line.strip() if editor_line else None,
            notes.strip() if notes else None,
            draft_id,
        ),
    )
    conn.commit()
