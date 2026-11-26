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

from humorhist.llm import LLMCallError, LLMError, ScreeningClient, StubClient


class SingleAttemptClient(Protocol):
    """A screening client must never retry a billable request internally."""

    def preflight(self) -> str: ...

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


def _preflight_client(client: SingleAttemptClient) -> str:
    """Authorize only the concrete live client or the canned offline stub.

    Unknown adapters and subclasses fail closed; tests needing custom behavior
    must explicitly inject a test-only verifier instead of implying safety.
    """
    if type(client) is StubClient:
        return StubClient.preflight(client)
    if type(client) is ScreeningClient:
        return ScreeningClient.preflight(client)
    return "unavailable"


def score_batch(client: SingleAttemptClient, items: list[dict]) -> dict[str, float]:
    """Screen a standalone batch only after the mandatory cost preflight."""
    if not items:
        return {}
    category = _preflight_client(client)
    if category != "free":
        raise LLMCallError("preflight_" + (category if category in _PREFLIGHT_CATEGORIES else "unavailable"))
    return _score_preflighted_batch(client, items)


def _score_preflighted_batch(client: SingleAttemptClient, items: list[dict]) -> dict[str, float]:
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
    if type(client) is StubClient:
        # Do not dispatch through instance attributes: a grafted live method is
        # not an offline stub. Call only the class-owned canned-response path.
        result = StubClient.complete_json(client, SCREEN_SYSTEM_PROMPT, prompt, reasoning_off=True)
    elif type(client) is ScreeningClient:
        result = ScreeningClient.complete_json(client, SCREEN_SYSTEM_PROMPT, prompt, reasoning_off=True)
    else:
        # Only an explicitly injected test verifier can admit custom doubles.
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


_DIAGNOSTIC_COLUMNS = ("category", "status", "started_at", "completed_at", "run_id", "batch_id")
_PREFLIGHT_CATEGORIES = {"invalid_model", "unavailable", "invalid_catalog", "model_missing", "unsafe_pricing"}
_ALLOWED_CATEGORIES = {"http_error", "timeout", "network_error", "invalid_response", "unknown"} | {
    "preflight_" + category for category in _PREFLIGHT_CATEGORIES
}


def _safe_failure(category: Any, status: Any) -> tuple[str, int | None]:
    if not isinstance(category, str) or category not in _ALLOWED_CATEGORIES:
        category = "unknown"
    if category != "http_error" or type(status) is not int or not 100 <= status <= 599:
        status = None
    return category, status


def _check_claims(conn: sqlite3.Connection, batch: list[dict]) -> None:
    for row in batch:
        found = conn.execute(
            "SELECT 1 FROM screen_attempts WHERE pool_id=? AND state='pending' AND generation=?",
            (row["id"], row["claim_generation"]),
        ).fetchone()
        if not found:
            raise RuntimeError("screening reservation changed during request")


def _complete_batch(
    conn: sqlite3.Connection, batch_id: str, state: str, scored: int, failed: int,
    category: str | None = None, status: int | None = None,
) -> None:
    """Append a terminal event in the same transaction as reservation outcomes."""
    cursor = conn.execute(
        "INSERT INTO screen_batches "
        "(batch_id,run_id,state,started_at,completed_at,selected_count,scored_count,failed_count,category,status) "
        "SELECT batch_id,run_id,?,started_at,strftime('%Y-%m-%dT%H:%M:%fZ','now'),"
        "selected_count,?,?,?,? FROM screen_batches WHERE batch_id=? AND state='pending' "
        "AND NOT EXISTS (SELECT 1 FROM screen_batches WHERE batch_id=? AND state!='pending')",
        (state, scored, failed, category, status, batch_id, batch_id),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("screening batch changed during request")


def screen_attempt_status(conn: sqlite3.Connection) -> list[dict]:
    """Inspect reservations without requiring credentials or making requests."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='screen_attempts'").fetchone():
        return []
    columns = {r[1] for r in conn.execute("PRAGMA table_info(screen_attempts)")}
    fields = ", ".join(f"a.{name}" if name in columns else f"NULL AS {name}" for name in _DIAGNOSTIC_COLUMNS)
    return [dict(r) for r in conn.execute(
        f"SELECT a.pool_id, a.state, p.title, p.funny_score, {fields} FROM screen_attempts a "
        "LEFT JOIN pool p ON p.id=a.pool_id ORDER BY a.pool_id")]


def format_screen_failures(failures) -> str:
    """Render only vetted categories and HTTP status codes, never raw errors."""
    counts = Counter()
    for failure in failures if isinstance(failures, list) else []:
        if not isinstance(failure, dict):
            continue
        category, status = _safe_failure(failure.get("category"), failure.get("status"))
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
        for name in _DIAGNOSTIC_COLUMNS:
            if name not in columns:
                kind = "INTEGER" if name == "status" else "TEXT"
                conn.execute(f"ALTER TABLE screen_attempts ADD COLUMN {name} {kind}")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS screen_batches ("
            "batch_id TEXT NOT NULL, run_id TEXT NOT NULL, "
            "state TEXT NOT NULL CHECK(state IN ('pending','success','partial','failure')), "
            "started_at TEXT NOT NULL, completed_at TEXT, selected_count INTEGER NOT NULL, "
            "scored_count INTEGER, failed_count INTEGER, category TEXT, status INTEGER, "
            "PRIMARY KEY(batch_id,state))"
        )
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS screen_batches_terminal "
                     "ON screen_batches(batch_id) WHERE state != 'pending'")
        conn.execute("UPDATE screen_attempts SET generation=lower(hex(randomblob(16))) WHERE generation IS NULL")
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    rows = _select_unscored(conn, limit, retry_failed)
    # Reservations can appear after selection. Count the union without counting
    # selected IDs twice; a lost claim is accounted for separately below.
    selected_ids = {row["id"] for row in rows}
    reservations = conn.execute(
        "SELECT p.id,a.state FROM pool p JOIN screen_attempts a ON a.pool_id=p.id "
        "WHERE p.funny_score IS NULL").fetchall()
    blocked_ids = {row[0] for row in reservations}
    blocked = sum(row[0] not in selected_ids and (row[1] != 'failed' or not retry_failed)
                  for row in reservations)
    if rows:
        category = _preflight_client(client)
        if category != "free":
            if category not in _PREFLIGHT_CATEGORIES:
                category = "unavailable"
            return {"scored": 0, "batches": 0, "failed_batches": 0,
                    "skipped": len(rows) + blocked, "unscored": len(rows) + (blocked if limit is None else 0),
                    "blocked": len(blocked_ids), "selected": len(rows) + (blocked if limit is None else 0),
                    "eligible_selected": len(rows), "preflight_failed": True, "preflight_status": category,
                    "preflight_post_blocked": 0,
                    "failures": [{"category": "preflight_" + category, "status": None}]}
    scored = batches = failed_batches = 0
    preflight_failed = False
    preflight_status = "free" if rows else "not_checked"
    preflight_post_blocked = 0
    skipped = blocked
    failures = []
    run_id = uuid.uuid4().hex
    batch_ids = []
    release_uncertain = False
    for start in range(0, len(rows), batch_size):
        candidate_batch = rows[start:start + batch_size]
        batch = []
        batch_id = uuid.uuid4().hex
        # Claim only this batch atomically. Never release other failed attempts.
        try:
            conn.execute("BEGIN IMMEDIATE")
            for row in candidate_batch:
                generation = uuid.uuid4().hex
                previous = None
                if row["attempt_generation"] is None:
                    cursor = conn.execute(
                        "INSERT INTO screen_attempts(pool_id,state,generation,started_at,run_id,batch_id) "
                        "SELECT id,'pending',?,strftime('%Y-%m-%dT%H:%M:%fZ','now'),?,? "
                        "FROM pool WHERE id=? AND funny_score IS NULL "
                        "ON CONFLICT(pool_id) DO NOTHING", (generation, run_id, batch_id, row["id"]))
                else:
                    previous = conn.execute("SELECT * FROM screen_attempts WHERE pool_id=?", (row["id"],)).fetchone()
                    cursor = conn.execute(
                        "UPDATE screen_attempts SET state='pending', generation=?, category=NULL, status=NULL, "
                        "started_at=strftime('%Y-%m-%dT%H:%M:%fZ','now'), completed_at=NULL, run_id=?, batch_id=? "
                        "WHERE pool_id=? AND state='failed' AND generation=? AND ? "
                        "AND EXISTS (SELECT 1 FROM pool WHERE id=? AND funny_score IS NULL)",
                        (generation, run_id, batch_id, row["id"], row["attempt_generation"], retry_failed, row["id"]))
                if cursor.rowcount:
                    row["claim_generation"] = generation
                    row["previous_attempt"] = dict(previous) if previous is not None else None
                    batch.append(row)
                else:
                    skipped += 1
            if batch:
                conn.execute(
                    "INSERT INTO screen_batches(batch_id,run_id,state,started_at,selected_count) "
                    "VALUES (?,?,'pending',strftime('%Y-%m-%dT%H:%M:%fZ','now'),?)",
                    (batch_id, run_id, len(batch)),
                )
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                with contextlib.suppress(sqlite3.Error):
                    conn.rollback()
            raise
        if not batch:
            continue
        batch_ids.append(batch_id)
        batches += 1
        try:
            batch_scores = _score_preflighted_batch(client, batch)
        except LLMError as exc:
            category, status = _safe_failure(
                exc.category if isinstance(exc, LLMCallError) else None,
                exc.status if isinstance(exc, LLMCallError) else None,
            )
            # Only the concrete client's own gate proves no POST occurred.
            no_post = type(client) is ScreeningClient and category.startswith("preflight_")
            if no_post:
                preflight_failed = True
                preflight_status = category.removeprefix("preflight_")
                preflight_post_blocked += 1
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    _check_claims(conn, batch)
                    for row in batch:
                        previous = row["previous_attempt"]
                        if previous is None:
                            conn.execute("DELETE FROM screen_attempts WHERE pool_id=? AND state='pending' "
                                         "AND generation=?",
                                         (row["id"], row["claim_generation"]))
                        else:
                            fields = [key for key in previous if key != "pool_id"]
                            conn.execute(
                                "UPDATE screen_attempts SET " + ", ".join(f"{key}=?" for key in fields)
                                + " WHERE pool_id=? AND state='pending' AND generation=?",
                                [previous[key] for key in fields] + [row["id"], row["claim_generation"]],
                            )
                    _complete_batch(conn, batch_id, "failure", 0, 0, category, status)
                    conn.commit()
                except (sqlite3.Error, RuntimeError):
                    if conn.in_transaction:
                        with contextlib.suppress(sqlite3.Error):
                            conn.rollback()
                    release_uncertain = True
                failures.append({"category": category, "status": status})
                skipped += len(rows) - start - len(candidate_batch)
                break
            try:
                conn.execute("BEGIN IMMEDIATE")
                _check_claims(conn, batch)
                conn.executemany(
                    "UPDATE screen_attempts SET state='failed', category=?, status=?, "
                    "completed_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                    "WHERE pool_id=? AND state='pending' AND generation=?",
                    [(category, status, row["id"], row["claim_generation"]) for row in batch],
                )
                _complete_batch(conn, batch_id, "failure", 0, len(batch), category, status)
                conn.commit()
            except BaseException:
                if conn.in_transaction:
                    with contextlib.suppress(sqlite3.Error):
                        conn.rollback()
                raise
            failed_batches += 1
            failures.append({"category": category, "status": status})
            # Do not traverse the rest of the pool after a provider failure.
            skipped += len(rows) - start - len(candidate_batch)
            break
        # The provider call is outside the transaction; persist the whole batch
        # together. A failed transaction leaves every reservation pending.
        try:
            conn.execute("BEGIN IMMEDIATE")
            _check_claims(conn, batch)
            generations = {str(row["id"]): row["claim_generation"] for row in batch}
            for pool_id, score in batch_scores.items():
                conn.execute("UPDATE pool SET funny_score=? WHERE id=? AND funny_score IS NULL", (score, pool_id))
                conn.execute("DELETE FROM screen_attempts WHERE pool_id=? AND state='pending' AND generation=?",
                             (pool_id, generations[pool_id]))
            conn.executemany(
                "UPDATE screen_attempts SET state='failed', category='invalid_response', status=NULL, "
                "completed_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                "WHERE pool_id=? AND state='pending' AND generation=?",
                [(row["id"], row["claim_generation"]) for row in batch if str(row["id"]) not in batch_scores],
            )
            missing = len(batch) - len(batch_scores)
            outcome = "failure" if missing == len(batch) else "partial" if missing else "success"
            _complete_batch(conn, batch_id, outcome, len(batch_scores), missing,
                            "invalid_response" if missing else None)
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
            "selected": len(rows) + (blocked if limit is None else 0), "eligible_selected": len(rows),
            "preflight_failed": preflight_failed, "preflight_status": preflight_status,
            "preflight_post_blocked": preflight_post_blocked, "release_uncertain": release_uncertain,
            "run_id": run_id if batch_ids else None, "batch_ids": batch_ids, "failures": failures}
