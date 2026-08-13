"""LLM funny pre-screen for humorhist pool candidates.

This is a FILTER, not a comedy writer. It asks an LLM to rate how genuinely
funny or absurd each harvested historical event is, so the drafting stage can
later pick the best candidates. A human still writes the actual jokes.

All LLM-touching code goes through the ``LLMClient`` protocol (see
``humorhist.llm``) so tests can inject ``StubClient`` and never hit the
network.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import humorhist.db as db
from humorhist.llm import LLMClient, LLMError

# Maximum number of complete_json attempts per batch (initial attempt + one
# retry). A failed batch is retried once before being counted as failed.
_MAX_ATTEMPTS = 2


SCREEN_SYSTEM_PROMPT = """\
You are a pre-screening judge for a historical-humour content pipeline. \
Your job is to RATE events, not to write jokes.

For every historical event listed below, rate how genuinely funny or absurd \
it is to a general modern audience.

Scoring scale (return a single number per item):
- 0 = completely mundane, nothing funny about it
- 10 = laugh-out-loud absurd

Rules:
- REWARD: absurd bureaucracy, the gap between intention and outcome, a
ridiculous specific detail, cosmic irony, and dignified institutions behaving
stupidly.
- Be honest and unsentimental. Rate purely on how absurd or funny the event
is to a modern audience.

Output STRICT JSON only: an array of objects, one per numbered item, in the \
same order as listed:
[{"n": <the item number>, "score": <0-10 number>, "reason": "<max 12 words>"}]

Return one entry for EVERY numbered item given, in order. Do not add any \
commentary outside the JSON.
"""


def build_batch_prompt(items: list[dict]) -> str:
    """Render candidate events as a numbered list for the LLM.

    Each item is expected to have ``title`` and (optionally) ``year`` and
    ``summary`` keys. Items are numbered 1..N.
    """
    lines: list[str] = []
    for i, item in enumerate(items, start=1):
        year = item.get("year")
        year_str = f" ({year})" if year is not None else ""
        title = item.get("title", "")
        summary = (item.get("summary") or "").strip()
        if summary:
            lines.append(f"{i}.{year_str} {title} — {summary}")
        else:
            lines.append(f"{i}.{year_str} {title}")
    return "\n".join(lines)


def _clamp_score(value: Any) -> float | None:
    """Return ``value`` clamped to 0-10, or ``None`` if non-numeric."""
    if isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num < 0:
        num = 0.0
    elif num > 10:
        num = 10.0
    return num


def _item_number(entry: Any) -> int | None:
    """Extract a 1-based item number from a response entry, or ``None``."""
    if not isinstance(entry, dict):
        return None
    n = entry.get("n")
    if isinstance(n, bool):
        return None
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if not isinstance(n, int):
        return None
    return n


def score_batch(client: LLMClient, items: list[dict]) -> dict[str, float]:
    """Score one batch of candidate items with a single LLM call.

    Returns a mapping of pool id -> clamped score (0-10). Robust to:
    - the model returning fewer entries than sent (missing ones are simply
      absent from the result),
    - unknown item numbers (ignored),
    - non-numeric scores (skipped without crashing).

    Raises ``LLMError`` only when the response is unusable as a whole (not a
    JSON array, or the underlying call failed). Callers should treat that as a
    failed batch and may retry.
    """
    if not items:
        return {}

    prompt = build_batch_prompt(items)
    result = client.complete_json(SCREEN_SYSTEM_PROMPT, prompt)

    if not isinstance(result, list):
        raise LLMError(f"expected a JSON array of scores, got {type(result).__name__}")

    scores: dict[str, float] = {}
    for entry in result:
        n = _item_number(entry)
        if n is None or not (1 <= n <= len(items)):
            continue
        score = _clamp_score(entry.get("score"))
        if score is None:
            continue
        pool_id = items[n - 1].get("id")
        if pool_id is None:
            continue
        scores[str(pool_id)] = score

    return scores


def _select_unscored(conn: sqlite3.Connection, limit: int | None) -> list[dict]:
    """Return rows with a NULL funny_score, ordered deterministically."""
    if limit is None:
        cur = conn.execute("SELECT * FROM pool WHERE funny_score IS NULL ORDER BY rowid")
    else:
        cur = conn.execute(
            "SELECT * FROM pool WHERE funny_score IS NULL ORDER BY rowid LIMIT ?",
            (limit,),
        )
    return [dict(row) for row in cur.fetchall()]


def screen_pool(
    conn: sqlite3.Connection,
    client: LLMClient,
    batch_size: int = 20,
    limit: int | None = None,
) -> dict:
    """Score every unscored pool row and persist the funny_score.

    Only rows where ``funny_score IS NULL`` are considered (already-scored
    rows are never re-scored). Rows are processed in batches of ``batch_size``;
    each batch is a single LLM call. A failed batch (LLMError, malformed JSON,
    etc.) is retried once; if it still fails it is counted in ``failed_batches``
    and the run continues with the next batch.

    Returns a summary dict with keys: ``scored``, ``batches``,
    ``failed_batches``, ``skipped``.
    """
    rows = _select_unscored(conn, limit)

    scored = 0
    batches = 0
    failed_batches = 0
    # We only select NULL rows, so no already-scored rows are encountered.
    skipped = 0

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        batches += 1

        batch_scores: dict[str, float] | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                batch_scores = score_batch(client, batch)
                break
            except LLMError:
                # Retry once; on the final attempt give up on this batch.
                if attempt < _MAX_ATTEMPTS - 1:
                    continue
                batch_scores = None

        if batch_scores is None:
            failed_batches += 1
            continue

        for pool_id, score in batch_scores.items():
            db.set_funny_score(conn, pool_id, score)
            scored += 1

    return {
        "scored": scored,
        "batches": batches,
        "failed_batches": failed_batches,
        "skipped": skipped,
    }
