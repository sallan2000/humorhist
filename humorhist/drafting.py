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
from datetime import datetime, timezone

import humorhist.db as db
from humorhist.brief import generate_angles
from humorhist.factcheck import fetch_wikipedia_extract, factcheck
from humorhist.llm import LLMClient

# Candidates are drawn from a window this many times larger than the number
# requested, then sampled, so repeated runs do not always pick the same
# top-scoring items.
SELECTION_WINDOW_MULTIPLIER = 4


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
    rng = rng or random.Random()
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
    item = _row_to_item(row)

    extract = fetch_wikipedia_extract(
        item["source_url"] or item["title"], client=http_client
    )
    brief = factcheck(client, item, extract)
    angles = generate_angles(client, item, brief)

    draft_id = db.make_id("draft", item["id"])
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO drafts
          (id, pool_id, brief_json, angles_json, status, created_at)
        VALUES (?, ?, ?, ?, 'pending', ?)
        """,
        (
            draft_id,
            item["id"],
            json.dumps(brief, ensure_ascii=False),
            json.dumps(angles, ensure_ascii=False),
            now,
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
            failures.append({"pool_id": row["id"], "title": row["title"], "error": str(exc)})

    return {
        "selected": len(rows),
        "drafted": len(drafted),
        "failed": len(failures),
        "draft_ids": drafted,
        "failures": failures,
    }
