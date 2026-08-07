"""Phase 3.4 — buffer-health monitor for the humorhist pipeline.

The whole product's resilience rests on never running out of approved,
reviewed content. This module answers one question: *how many days of posts
do we have in the tank, and what should happen at this level?*

"Depth in days" = the count of unpublished queued drafts (each is one day of
drip publishing). The thresholds come straight from the plan:

    >= 7   silent   (healthy — a week+ of buffer)
    < 7    nudge    (Telegram: you have N days left, review soon)
    < 3    escalate (Telegram: buffer critically low)
    pending < 5 -> auto-draft more candidates so there's always something
                   to review (best-effort; needs an LLM key)
"""

from __future__ import annotations

import sqlite3

# Depth thresholds (in unpublished queued drafts == days of buffer).
SILENT_THRESHOLD = 7
NUDGE_THRESHOLD = 7
ESCALATE_THRESHOLD = 3
# Auto-draft more candidates when fewer than this many pending drafts remain.
AUTO_DRAFT_PENDING_FLOOR = 5
AUTO_DRAFT_COUNT = 10


def buffer_depth(conn: sqlite3.Connection) -> int:
    """Number of unpublished queued drafts — i.e. days of publishing buffer."""
    row = conn.execute(
        "SELECT count(*) AS n FROM queue WHERE published = 0"
    ).fetchone()
    return int(row["n"])


def pending_count(conn: sqlite3.Connection) -> int:
    """Number of drafts still awaiting a review decision."""
    row = conn.execute(
        "SELECT count(*) AS n FROM drafts WHERE status = 'pending'"
    ).fetchone()
    return int(row["n"])


def buffer_health(conn: sqlite3.Connection) -> dict:
    """Compute the buffer health snapshot.

    Returns a dict with ``depth`` (days of buffer), ``pending`` (undecided
    drafts), and ``level`` — one of ``"silent"``, ``"nudge"``, ``"escalate"``.
    """
    depth = buffer_depth(conn)
    pending = pending_count(conn)
    if depth < ESCALATE_THRESHOLD:
        level = "escalate"
    elif depth < NUDGE_THRESHOLD:
        level = "nudge"
    else:
        level = "silent"
    return {
        "depth": depth,
        "pending": pending,
        "level": level,
        "auto_draft": pending < AUTO_DRAFT_PENDING_FLOOR,
    }


def health_message(health: dict, will_draft: bool | None = None) -> str:
    """Human-readable Telegram/cron message for a buffer-health report.

    ``will_draft`` overrides whether the "auto-drafting" line is shown (the
    actual decision this run, which may be False even when the buffer says
    auto-draft is *needed* — e.g. no --auto-draft flag, or no LLM key).
    """
    depth = health["depth"]
    pending = health["pending"]
    if health["level"] == "escalate":
        lead = "🚨 BUFFER CRITICALLY LOW"
    elif health["level"] == "nudge":
        lead = "⚠️ Buffer running low"
    else:
        lead = "✅ Buffer healthy"
    lines = [
        lead,
        f"  • Approved + queued (unpublished): {depth} day(s) of buffer",
        f"  • Pending review: {pending} draft(s)",
    ]
    if (will_draft if will_draft is not None else health["auto_draft"]):
        lines.append(
            f"  • Auto-drafting {AUTO_DRAFT_COUNT} more candidates (pending < "
            f"{AUTO_DRAFT_PENDING_FLOOR})."
        )
    return "\n".join(lines)


def run_buffer_check(
    conn: sqlite3.Connection,
    *,
    client=None,
    auto_draft: bool = True,
    chat_id: str | None = None,
    telegram=None,
) -> dict:
    """Compute health, optionally auto-draft, and optionally notify via Telegram.

    ``client`` is an optional LLM client for auto-drafting. ``telegram`` is an
    optional Telegram transport and ``chat_id`` its target; a message is sent
    only when the level is ``nudge``/``escalate`` (silent => no nag). Best-effort:
    auto-draft and Telegram failures are reported in the returned dict, never raised.

    Returns the health dict plus ``drafted`` (count auto-drafted) and
    ``notified`` (bool).
    """
    health = buffer_health(conn)
    result: dict = dict(health)

    drafted = 0
    draft_error = None
    if auto_draft and health["auto_draft"] and client is not None:
        try:
            from humorhist.drafting import draft_candidates

            res = draft_candidates(conn, client, count=AUTO_DRAFT_COUNT)
            drafted = res["drafted"]
        except Exception as exc:  # noqa: BLE001 - monitor must never crash the cron
            draft_error = str(exc)
    result["drafted"] = drafted
    result["draft_error"] = draft_error

    notified = False
    if telegram is not None and chat_id and health["level"] != "silent":
        try:
            will_draft = bool(auto_draft and client is not None)
            telegram.send_message(chat_id, health_message(health, will_draft=will_draft))
            notified = True
        except Exception as exc:  # noqa: BLE001
            result["notify_error"] = str(exc)
    result["notified"] = notified
    return result
