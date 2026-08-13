"""Phase 4 (B+) post-copy generation for approved drafts.

An approved draft has a rich ``brief`` (verified facts, flagged
misconceptions) and a set of comic ``angles``. This module turns that into a
short, paste-ready (or later, auto-postable) post and stores it on the queue
row as ``post_copy``. The copy is *editable in place* — both the CLI and the
Telegram transport open the queue row and let the editor rewrite it.

Generation needs an LLM, so it is deliberately kept OUT of ``review.apply_review``
(which is pure and transport-agnostic). Both approve paths (CLI review loop and
the Telegram review bot) call ``generate_post_copy`` / ``fill_post_copy`` right
after ``apply_review`` returns.

The character budget is a config knob (``HUMORHIST_CHAR_LIMIT``, default 280)
so the cap can be raised later (e.g. when an X account is paid for) without a
code change. Generation always targets the limit; if the model overshoots, the
copy is hard-trimmed to the last word boundary within the limit and flagged.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime

import humorhist.db as db
from humorhist.llm import LLMClient

# Default budget. X's classic free-post limit is 280; raise via env when the
# account is paid (or for Mastodon's 500). Read at call time, not import time,
# so the env can be set between import and use.
DEFAULT_CHAR_LIMIT = 280


def char_limit() -> int:
    """Return the active character budget for generated post copy."""
    try:
        return int(os.environ.get("HUMORHIST_CHAR_LIMIT", DEFAULT_CHAR_LIMIT))
    except (TypeError, ValueError):
        return DEFAULT_CHAR_LIMIT


POST_COPY_SYSTEM_PROMPT = (
    "You are the social copywriter for a history-humor account. You turn a "
    "fact-checked brief and a chosen comic angle into ONE short post.\n"
    "Rules:\n"
    "  - Use only the VERIFIED FACTS. Never repeat anything listed under "
    "MISCONCEPTIONS (those are popular myths the account must not spread).\n"
    "  - If an EDITOR LINE is given, treat it as a hard steer for tone/lead.\n"
    "  - Be funny but factual. No invented quotes, dates, or deaths.\n"
    "  - Output ONLY the post text. No preamble, no hashtags unless natural.\n"
    "  - Stay within {limit} characters. Be ruthless about it.\n"
    'Return a JSON object: {{"post": "<the post text>"}}.\n'
)


def _build_user_prompt(draft: dict, pool: dict | None, limit: int) -> str:
    brief = json.loads(draft["brief_json"] or "{}")
    angles = json.loads(draft["angles_json"] or "{}")

    facts = brief.get("verified_facts", [])
    misconceptions = brief.get("misconceptions", [])
    angle_name = (angles.get("angles") or [{}])[0].get("angle_name", "")
    hook = angles.get("suggested_hook", "")
    title = pool["title"] if pool else "(unknown)"
    year = pool["year"] if pool else ""

    editor_line = draft.get("editor_line") or "(none)"

    parts = [
        f"TOPIC: {title} ({year})",
        f"EDITOR LINE (tone steer): {editor_line}",
        f"LEAD ANGLE: {angle_name}",
        f"SUGGESTED HOOK: {hook}",
        "",
        "VERIFIED FACTS:",
    ]
    parts += [f"  - {f}" for f in facts] or ["  - (none provided)"]
    if misconceptions:
        parts.append("")
        parts.append("MISCONCEPTIONS TO AVOID (do NOT repeat these):")
        parts += [f"  - {m}" for m in misconceptions]
    parts.append(f"\nCHAR LIMIT: {limit}")
    return "\n".join(parts)


def _trim_to_limit(text: str, limit: int) -> tuple[str, bool]:
    """Hard-trim to the last word boundary within ``limit``.

    Returns (trimmed_text, was_trimmed). Whitespace-collapsed first so we never
    return a trailing space.
    """
    text = " ".join(text.split())
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    # back up to the last space so we don't split a word
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut, True


def generate_post_copy(
    client: LLMClient,
    draft: dict,
    pool: dict | None = None,
    limit: int | None = None,
) -> str:
    """Generate short post copy for one draft. Returns the copy string.

    ``limit`` overrides the env-driven default. Raises ``LLMError`` on a bad
    model response, or any transport error from ``client``.
    """
    limit = limit if limit is not None else char_limit()
    user = _build_user_prompt(draft, pool, limit)
    system = POST_COPY_SYSTEM_PROMPT.format(limit=limit)

    # hy3:free spends tokens "reasoning" before it emits content, so a small
    # budget truncates (finish_reason:"length", content:null) and the fallback
    # can surface a degenerate stub. Disable reasoning for clean direct output,
    # and retry a few times if the model still returns something unusable.
    last_err: Exception | None = None
    for _ in range(3):
        try:
            result = client.complete_json(system, user, max_tokens=2048, reasoning_off=True)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
        if isinstance(result, dict) and result.get("post"):
            text = result["post"].strip()
        elif isinstance(result, str) and result.strip():
            text = result.strip()
        else:
            last_err = ValueError(f"model returned no post copy: {result!r}")
            continue
        # reject degenerate output (truncated stubs, ellipsis-only, too short)
        if len(text) < 12 or set(text) <= {"."}:
            last_err = ValueError(f"degenerate copy rejected: {text!r}")
            continue
        return _trim_to_limit(text, limit)[0]

    from humorhist.llm import LLMError

    raise LLMError(f"post copy generation failed: {last_err}")


def fill_post_copy(
    conn: sqlite3.Connection,
    client: LLMClient | None,
    limit: int | None = None,
    draft_id: str | None = None,
    force: bool = False,
) -> int:
    """Generate + store ``post_copy`` for approved, queued rows lacking it.

    Idempotent: rows that already have copy are skipped (unless ``force`` is
    set, which regenerates and overwrites even existing copy — used by the CLI
    ``copy regen`` command). ``draft_id`` scopes the fill to a single draft
    (used right after an approve). When ``client`` is None (no LLM key
    available) the function is a no-op returning 0, so callers can always call
    it safely without crashing the approve path.

    Returns the number of rows newly filled.
    """
    limit = limit if limit is not None else char_limit()
    if client is None:
        return 0

    if draft_id is not None:
        sql = """
            SELECT q.draft_id AS draft_id
            FROM queue q
            JOIN drafts d ON d.id = q.draft_id
            WHERE d.status = 'approved' AND q.draft_id = ?
        """
        if not force:
            sql += " AND q.post_copy IS NULL"
        rows = conn.execute(sql, (draft_id,)).fetchall()
    else:
        sql = """
            SELECT q.draft_id AS draft_id
            FROM queue q
            JOIN drafts d ON d.id = q.draft_id
            WHERE d.status = 'approved'
        """
        if not force:
            sql += " AND q.post_copy IS NULL"
        rows = conn.execute(sql).fetchall()

    filled = 0
    for r in rows:
        did = r["draft_id"]
        draft = conn.execute("SELECT * FROM drafts WHERE id = ?", (did,)).fetchone()
        if draft is None:
            continue
        pool = db.get_pool_item(conn, draft["pool_id"])
        try:
            copy = generate_post_copy(client, dict(draft), dict(pool) if pool else None, limit)
        except Exception as exc:  # noqa: BLE001 - one bad draft must not abort the fill
            print(f"[copywriter] failed for {did}: {exc}")
            continue
        now = datetime.now(UTC).isoformat()
        conn.execute(
            "UPDATE queue SET post_copy = ?, post_copy_at = ? WHERE draft_id = ?",
            (copy, now, did),
        )
        filled += 1
    if filled:
        conn.commit()
    return filled


def set_post_copy(conn: sqlite3.Connection, draft_id: str, copy: str) -> None:
    """Store an editor's manually edited copy for a queued draft."""
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "UPDATE queue SET post_copy = ?, post_copy_at = ? WHERE draft_id = ?",
        (copy, now, draft_id),
    )
    conn.commit()
