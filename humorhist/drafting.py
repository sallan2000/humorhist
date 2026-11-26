"""Draft assembly: pick top pool candidates, fact-check them, generate angles.

This is the orchestration layer between the harvest/screen stage and the
human review loop. For each selected candidate it:

1. fetches the Wikipedia extract for the source article,
2. runs the fact-check pass to build a verified brief,
3. generates comic angles from that brief,
4. writes a ``drafts`` row with status ``pending`` and marks the pool item
   ``drafted``.

A failure on one candidate never aborts the run.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from datetime import UTC, datetime

import humorhist.db as db
from humorhist.brief import AnglesError, generate_angles
from humorhist.factcheck import FactCheckError, factcheck, fetch_wikipedia_extract
from humorhist.llm import LLMClient, LLMError, LLMUnavailable

# Candidates are drawn from a window this many times larger than the number
# requested, then sampled, so repeated runs do not always pick the same
# top-scoring items.
SELECTION_WINDOW_MULTIPLIER = 4

STAGES = frozenset({"prepare", "extract", "factcheck", "angles", "persist", "unknown"})
CATEGORIES = frozenset({"unavailable", "provider", "source", "factcheck", "validation", "database", "unexpected"})


class DraftStageError(Exception):
    """A deliberately redacted failure at a known drafting stage."""

    def __init__(self, stage: str, category: str):
        self.stage = stage
        self.category = category
        super().__init__(f"{stage}:{category}")


def _category(stage: str, exc: Exception) -> str:
    if isinstance(exc, LLMUnavailable):
        return "unavailable"
    if isinstance(exc, LLMError):
        return "provider"
    if isinstance(exc, FactCheckError):
        # FactCheckError conflates transport, response and validation failures;
        # do not infer a cause from its unsafe free-form message.
        return "source" if stage == "extract" else "factcheck"
    if isinstance(exc, AnglesError):
        return "validation"
    if isinstance(exc, sqlite3.Error):
        return "database"
    return "unexpected"


def _stage(stage, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        # Exception messages, chained tracebacks and dynamic class names can
        # contain provider URLs or credentials; expose only fixed vocabulary.
        raise DraftStageError(stage, _category(stage, exc)) from None


def format_failure_counts(failures) -> str:
    """Aggregate only allowlisted stage/category values; never render input."""
    counts = Counter()
    for failure in failures if isinstance(failures, list) else []:
        if not isinstance(failure, dict):
            continue
        stage = failure.get("stage")
        category = failure.get("category")
        if not isinstance(stage, str) or stage not in STAGES:
            stage = "unknown"
        if not isinstance(category, str) or category not in CATEGORIES:
            category = "unexpected"
        counts[(stage, category)] += 1
    return ", ".join(f"{stage}:{category}={count}" for (stage, category), count in sorted(counts.items()))


def describe_failure_counts(failures) -> str:
    """Describe stages without rendering untrusted exception text."""
    labels = {
        "prepare": "preparing candidates", "extract": "fetching Wikipedia extracts",
        "factcheck": "fact-checking", "angles": "generating angles", "persist": "saving drafts",
        "unknown": "an unknown stage",
    }
    counts = Counter()
    for failure in failures if isinstance(failures, list) else []:
        stage = failure.get("stage") if isinstance(failure, dict) else None
        if not isinstance(stage, str) or stage not in STAGES:
            stage = "unknown"
        counts[stage] += 1
    return ", ".join(f"{count} during {labels[stage]}" for stage, count in sorted(counts.items()))


def select_candidates(
    conn: sqlite3.Connection,
    count: int = 3,
    min_score: float = 7.0,
    rng: random.Random | None = None,
) -> list[sqlite3.Row]:
    """Pick up to ``count`` unused pool items scoring at least ``min_score``.

    Selection samples from a larger window of top-scoring candidates so the
    same handful of items are not drafted every time.
    """
    rng = rng or random.Random()  # nosec B311 - selection sampling, not crypto
    window = max(count * SELECTION_WINDOW_MULTIPLIER, count)
    rows = conn.execute(
        """
        SELECT * FROM pool
        WHERE status = 'new'
          AND funny_score IS NOT NULL
          AND funny_score >= ?
        ORDER BY funny_score DESC
        LIMIT ?
        """,
        (min_score, window),
    ).fetchall()

    if len(rows) <= count:
        return list(rows)
    return rng.sample(list(rows), count)


def _row_to_item(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "year": row["year"],
        "summary": row["summary"],
        "source_url": row["source_url"],
        # build_factcheck_prompt() looks for "url"; without this alias the
        # source link never reaches the fact-check prompt.
        "url": row["source_url"],
        "source_name": row["source_name"],
        "funny_score": row["funny_score"],
    }


def draft_one(
    conn: sqlite3.Connection,
    client: LLMClient,
    row: sqlite3.Row,
    *,
    http_client=None,
) -> str:
    """Fact-check and generate angles for one pool row; write a drafts row.

    Returns the new draft id. Raises on failure (caller decides whether to
    continue).
    """
    item = _stage("prepare", _row_to_item, row)

    # Non-destructive: never overwrite an existing draft for this pool item
    # (the editor may have reviewed/edited/approved it). Callers select only
    # status='new' items, but a re-run or scheduled pass must still be safe.
    if _stage("prepare", db.draft_exists_for_pool, conn, item["id"]):
        return _stage("prepare", db.make_id, "draft", item["id"])

    extract = _stage("extract", fetch_wikipedia_extract, item["source_url"] or item["title"], client=http_client)
    brief = _stage("factcheck", factcheck, client, item, extract)
    angles = _stage("angles", generate_angles, client, item, brief)

    return _stage("persist", _persist_draft, conn, item, brief, angles)


def _persist_draft(conn, item, brief, angles):

    draft_id = db.make_id("draft", item["id"])
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO drafts
          (id, pool_id, brief_json, angles_json, status, created_at, short_code)
        VALUES (?, ?, ?, ?, 'pending', ?, ?)
        """,
        (
            draft_id,
            item["id"],
            json.dumps(brief, ensure_ascii=False),
            json.dumps(angles, ensure_ascii=False),
            now,
            db.short_code_for_table(conn, "drafts", draft_id),
        ),
    )
    db.set_status(conn, "pool", item["id"], "drafted")
    conn.commit()
    return draft_id


def draft_candidates(
    conn: sqlite3.Connection,
    client: LLMClient,
    count: int = 3,
    min_score: float = 7.0,
    rng: random.Random | None = None,
    http_client=None,
) -> dict:
    """Draft up to ``count`` candidates, tolerating individual failures."""
    rows = select_candidates(conn, count=count, min_score=min_score, rng=rng)

    drafted: list[str] = []
    failures: list[dict] = []

    for row in rows:
        try:
            drafted.append(draft_one(conn, client, row, http_client=http_client))
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not abort the run
            stage = exc.stage if isinstance(exc, DraftStageError) and exc.stage in STAGES else "unknown"
            category = exc.category if isinstance(exc, DraftStageError) and exc.category in CATEGORIES else "unexpected"
            failures.append({"pool_id": row["id"], "stage": stage, "category": category})

    return {
        "selected": len(rows),
        "drafted": len(drafted),
        "failed": len(failures),
        "draft_ids": drafted,
        "failures": failures,
    }
