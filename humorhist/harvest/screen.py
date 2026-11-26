"""LLM funny pre-screen for humorhist pool candidates.

This is a FILTER, not a comedy writer. It asks an LLM to rate how genuinely
funny or absurd each harvested historical event is, so the drafting stage can
later pick the best candidates. A human still writes the actual jokes.

All screening calls go through the
``SingleAttemptClient`` protocol so tests can inject ``StubClient`` and
never hit the network.
"""

from __future__ import annotations

import contextlib
import math
import sqlite3
import uuid
from collections import Counter
from typing import Any, Protocol

from humorhist.llm import LLMCallError, LLMError


class SingleAttemptClient(Protocol):
    """A screening client must never retry a billable request internally."""

    def complete_json_once(self, system: str, user: str, **kwargs: Any) -> Any: ...


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
    if not math.isfinite(num):
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


def score_batch(client: SingleAttemptClient, items: list[dict]) -> dict[str, float]:
    """Score one batch of candidate items with a single LLM call.

    Returns a mapping of pool id -> clamped score (0-10). Robust to:
    - the model returning fewer entries than sent (missing ones are simply
      absent from the result),
    - unknown item numbers (ignored),
    - non-numeric scores (skipped without crashing).

    Raises ``LLMError`` only when the response is unusable as a whole (not a
    JSON array, or the underlying call failed). Callers should treat that as a
    failed batch without an automatic retry.
    """
    if not items:
        return {}

    prompt = build_batch_prompt(items)
    call = getattr(client, "complete_json_once", None)
    if not callable(call):
        raise LLMCallError("invalid_response")
    result = call(SCREEN_SYSTEM_PROMPT, prompt, reasoning_off=True)

    if not isinstance(result, list):
        raise LLMCallError("invalid_response")

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


def _select_unscored(conn: sqlite3.Connection, limit: int | None, retry_failed: bool = False) -> list[dict]:
    """Apply the limit only after excluding blocked attempts."""
    cur = conn.execute(
        "SELECT p.*, a.generation AS attempt_generation FROM pool p LEFT JOIN screen_attempts a ON a.pool_id=p.id "
        "WHERE p.funny_score IS NULL AND (a.pool_id IS NULL OR (? AND a.state='failed')) "
        "ORDER BY p.rowid LIMIT ?", (retry_failed, -1 if limit is None else limit))
    return [dict(row) for row in cur.fetchall()]


def screen_attempt_status(conn: sqlite3.Connection) -> list[dict]:
    """Inspect reservations without requiring credentials or making requests."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='screen_attempts'").fetchone():
        return []
    return [dict(r) for r in conn.execute(
        "SELECT a.pool_id, a.state, p.title, p.funny_score FROM screen_attempts a "
        "LEFT JOIN pool p ON p.id=a.pool_id ORDER BY a.pool_id")]


def format_screen_failures(failures) -> str:
    """Render only vetted categories and HTTP status codes, never raw errors."""
    allowed = {"http_error", "timeout", "network_error", "invalid_response", "unknown"}
    counts = Counter()
    for failure in failures if isinstance(failures, list) else []:
        if not isinstance(failure, dict):
            continue
        category = failure.get("category")
        if not isinstance(category, str) or category not in allowed:
            category = "unknown"
        status = failure.get("status")
        if category != "http_error" or type(status) is not int or not 100 <= status <= 599:
            status = None
        label = f"{category}_http_{status}" if status else category
        counts[label] += 1
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def screen_pool(
    conn: sqlite3.Connection,
    client: SingleAttemptClient,
    batch_size: int = 20,
    limit: int | None = None,
    retry_failed: bool = False,
) -> dict:
    """Score unscored rows once; persist reservations before billable calls.

    Failed or interrupted attempts stay blocked across runs. Only failed attempts
    may be retried explicitly; interrupted reservations require operator review.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if conn.in_transaction:
        raise RuntimeError("screening requires a connection without an active transaction")
    # Serialize schema upgrades with claims, retaining all legacy reservations.
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("CREATE TABLE IF NOT EXISTS screen_attempts "
                     "(pool_id TEXT PRIMARY KEY, state TEXT NOT NULL, generation TEXT)")
        columns = {r[1] for r in conn.execute("PRAGMA table_info(screen_attempts)")}
        if "generation" not in columns:
            conn.execute("ALTER TABLE screen_attempts ADD COLUMN generation TEXT")
        conn.execute("UPDATE screen_attempts SET generation=lower(hex(randomblob(16))) WHERE generation IS NULL")
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    rows = _select_unscored(conn, limit, retry_failed)
    blocked = conn.execute(
        "SELECT COUNT(*) FROM pool p JOIN screen_attempts a ON a.pool_id=p.id "
        "WHERE p.funny_score IS NULL AND (a.state != 'failed' OR NOT ?)",
        (retry_failed,)).fetchone()[0]
    scored = batches = failed_batches = 0
    skipped = blocked
    failures = []
    for start in range(0, len(rows), batch_size):
        candidate_batch = rows[start:start + batch_size]
        batch = []
        # Claim only this batch atomically. Never release other failed attempts.
        try:
            conn.execute("BEGIN IMMEDIATE")
            for row in candidate_batch:
                generation = uuid.uuid4().hex
                if row["attempt_generation"] is None:
                    cursor = conn.execute(
                        "INSERT INTO screen_attempts(pool_id,state,generation) "
                        "SELECT id,'pending',? FROM pool WHERE id=? AND funny_score IS NULL "
                        "ON CONFLICT(pool_id) DO NOTHING", (generation, row["id"]))
                else:
                    cursor = conn.execute(
                        "UPDATE screen_attempts SET state='pending', generation=? "
                        "WHERE pool_id=? AND state='failed' AND generation=? AND ? "
                        "AND EXISTS (SELECT 1 FROM pool WHERE id=? AND funny_score IS NULL)",
                        (generation, row["id"], row["attempt_generation"], retry_failed, row["id"]))
                if cursor.rowcount:
                    batch.append(row)
                else:
                    skipped += 1
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                with contextlib.suppress(sqlite3.Error):
                    conn.rollback()
            raise
        if not batch:
            continue
        batches += 1
        try:
            batch_scores = score_batch(client, batch)
        except LLMError as exc:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.executemany("UPDATE screen_attempts SET state = 'failed' WHERE pool_id = ? AND state = 'pending'",
                                 [(row["id"],) for row in batch])
                conn.commit()
            except BaseException:
                if conn.in_transaction:
                    with contextlib.suppress(sqlite3.Error):
                        conn.rollback()
                raise
            failed_batches += 1
            category = exc.category if isinstance(exc, LLMCallError) else "unknown"
            if not isinstance(category, str) or category not in {
                "http_error", "timeout", "network_error", "invalid_response"
            }:
                category = "unknown"
            status = exc.status if isinstance(exc, LLMCallError) else None
            if category != "http_error" or type(status) is not int or not 100 <= status <= 599:
                status = None
            failures.append({"category": category, "status": status})
            # Do not traverse the rest of the pool after a provider failure.
            skipped += len(rows) - start - len(candidate_batch)
            break
        # The provider call is outside the transaction; persist the whole batch
        # together. A failed transaction leaves every reservation pending.
        try:
            conn.execute("BEGIN IMMEDIATE")
            for pool_id, score in batch_scores.items():
                conn.execute("UPDATE pool SET funny_score=? WHERE id=? AND funny_score IS NULL", (score, pool_id))
                conn.execute("DELETE FROM screen_attempts WHERE pool_id=? AND state='pending'", (pool_id,))
            conn.executemany("UPDATE screen_attempts SET state='failed' WHERE pool_id=? AND state='pending'",
                             [(row["id"],) for row in batch if str(row["id"]) not in batch_scores])
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                with contextlib.suppress(sqlite3.Error):
                    conn.rollback()
            raise
        scored += len(batch_scores)
        if len(batch_scores) != len(batch):
            failed_batches += 1
            failures.append({"category": "invalid_response", "status": None})
            skipped += len(rows) - start - len(candidate_batch)
            break
    blocked_total = conn.execute(
        "SELECT COUNT(*) FROM pool p JOIN screen_attempts a ON a.pool_id=p.id "
        "WHERE p.funny_score IS NULL").fetchone()[0]
    unscored = len(rows) - scored + (blocked if limit is None else 0)
    return {"scored": scored, "batches": batches, "failed_batches": failed_batches,
            "skipped": skipped, "unscored": unscored, "blocked": blocked_total,
            "selected": len(rows) + (blocked if limit is None else 0), "failures": failures}
