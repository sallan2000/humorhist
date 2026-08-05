"""Tests for humorhist.harvest.screen -- the LLM funny pre-screen.

TDD with ``StubClient``: no network calls are ever made. Real file-backed
DBs via ``tmp_path`` are used to verify actual persistence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import humorhist.db as db
from humorhist.harvest import screen
from humorhist.llm import LLMError, StubClient


def _fresh_db(tmp_path: Path):
    path = tmp_path / "test.sqlite"
    conn = db.connect(str(path))
    db.migrate(conn)
    return conn


def _insert_rows(conn, n: int, *, scored: list[int] | None = None) -> list[str]:
    """Insert ``n`` pool rows; set scores for the (0-indexed) ``scored`` rows.

    Returns the list of inserted pool ids in insertion order.
    """
    scored = scored or []
    ids: list[str] = []
    for i in range(n):
        pid = f"pool{i:03d}"
        db.upsert_pool_item(
            conn,
            id=pid,
            title=f"Event {i}",
            year=1900 + i,
            date_hint=None,
            summary=f"Summary for event {i}",
            source_url=f"https://example.com/{i}",
            source_name="test",
        )
        ids.append(pid)
    for idx in scored:
        db.set_funny_score(conn, ids[idx], 7.0)
    return ids


def _batch_response(n_items: int, score: float = 5.0) -> list[dict]:
    """A well-formed LLM response array for ``n_items`` numbered items."""
    return [
        {"n": i + 1, "score": score, "reason": "ok"}
        for i in range(n_items)
    ]


# --- build_batch_prompt -----------------------------------------------------


def test_build_batch_prompt_numbers_items():
    items = [
        {"id": "a", "title": "The Emu War", "year": 1932, "summary": "Australia lost."},
        {"id": "b", "title": "Cadaver Synod", "year": 897, "summary": "Pope tried a corpse."},
        {"id": "c", "title": "No Year Event", "year": None, "summary": "No year."},
    ]
    prompt = screen.build_batch_prompt(items)
    # numbered 1..N
    assert "1." in prompt
    assert "2." in prompt
    assert "3." in prompt
    # titles and years are rendered
    assert "The Emu War" in prompt and "1932" in prompt
    assert "Cadaver Synod" in prompt and "897" in prompt
    assert "No Year Event" in prompt


# --- score_batch ------------------------------------------------------------


def test_score_batch_maps_ids():
    items = [
        {"id": "a", "title": "One", "year": 1, "summary": "s"},
        {"id": "b", "title": "Two", "year": 2, "summary": "s"},
        {"id": "c", "title": "Three", "year": 3, "summary": "s"},
    ]
    stub = StubClient(
        [
            [
                {"n": 1, "score": 3, "reason": "r"},
                {"n": 2, "score": 7, "reason": "r"},
                {"n": 3, "score": 9, "reason": "r"},
            ]
        ]
    )
    result = screen.score_batch(stub, items)
    assert result == {"a": 3.0, "b": 7.0, "c": 9.0}


def test_score_batch_handles_partial_response():
    items = [
        {"id": "a", "title": "One", "year": 1, "summary": "s"},
        {"id": "b", "title": "Two", "year": 2, "summary": "s"},
        {"id": "c", "title": "Three", "year": 3, "summary": "s"},
    ]
    stub = StubClient(
        [
            [
                {"n": 1, "score": 4, "reason": "r"},
                {"n": 2, "score": 6, "reason": "r"},
            ]
        ]
    )
    result = screen.score_batch(stub, items)
    assert result == {"a": 4.0, "b": 6.0}
    assert "c" not in result


def test_score_batch_ignores_unknown_n():
    items = [
        {"id": "a", "title": "One", "year": 1, "summary": "s"},
        {"id": "b", "title": "Two", "year": 2, "summary": "s"},
    ]
    stub = StubClient(
        [
            [
                {"n": 1, "score": 5, "reason": "r"},
                {"n": 99, "score": 8, "reason": "r"},
            ]
        ]
    )
    result = screen.score_batch(stub, items)
    assert result == {"a": 5.0}
    assert "b" not in result


def test_score_batch_clamps():
    items = [
        {"id": "a", "title": "Low", "year": 1, "summary": "s"},
        {"id": "b", "title": "High", "year": 2, "summary": "s"},
    ]
    stub = StubClient(
        [
            [
                {"n": 1, "score": -3, "reason": "r"},
                {"n": 2, "score": 15, "reason": "r"},
            ]
        ]
    )
    result = screen.score_batch(stub, items)
    assert result == {"a": 0.0, "b": 10.0}


def test_score_batch_skips_non_numeric():
    items = [
        {"id": "a", "title": "Num", "year": 1, "summary": "s"},
        {"id": "b", "title": "Word", "year": 2, "summary": "s"},
    ]
    stub = StubClient(
        [
            [
                {"n": 1, "score": 6, "reason": "r"},
                {"n": 2, "score": "banana", "reason": "r"},
            ]
        ]
    )
    result = screen.score_batch(stub, items)
    assert result == {"a": 6.0}
    assert "b" not in result


# --- screen_pool ------------------------------------------------------------


def test_screen_pool_only_scores_null(tmp_path):
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 3, scored=[1])  # row 1 pre-scored at 7.0
    stub = StubClient([_batch_response(2)])
    result = screen.screen_pool(conn, stub, batch_size=20)
    assert result["scored"] == 2
    # the pre-scored row is left untouched
    row = db.get_pool_item(conn, ids[1])
    assert row["funny_score"] == 7.0
    # the two unscored rows are now scored
    assert db.get_pool_item(conn, ids[0])["funny_score"] is not None
    assert db.get_pool_item(conn, ids[2])["funny_score"] is not None


def test_screen_pool_batches(tmp_path):
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 25)
    stub = StubClient(
        [_batch_response(10), _batch_response(10), _batch_response(5)]
    )
    result = screen.screen_pool(conn, stub, batch_size=10)
    assert result["batches"] == 3
    assert len(stub.calls) == 3
    assert result["scored"] == 25
    assert result["failed_batches"] == 0


def test_screen_pool_continues_on_batch_failure(tmp_path):
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 15)
    # batch 1 (10 rows) fails twice (exhausting the retry); batch 2 (5) succeeds
    stub = StubClient(
        [LLMError("boom"), LLMError("boom"), _batch_response(5)]
    )
    result = screen.screen_pool(conn, stub, batch_size=10)
    assert result["failed_batches"] == 1
    assert result["scored"] == 5
    assert len(stub.calls) == 3
    # batch 1's 10 rows remain unscored; batch 2's 5 are scored
    null_count = conn.execute(
        "SELECT COUNT(*) AS n FROM pool WHERE funny_score IS NULL"
    ).fetchone()["n"]
    assert null_count == 10


def test_screen_pool_retries_once(tmp_path):
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 4)
    # first attempt raises, the single retry succeeds
    stub = StubClient([LLMError("boom"), _batch_response(4)])
    result = screen.screen_pool(conn, stub, batch_size=10)
    assert result["failed_batches"] == 0
    assert result["scored"] == 4
    assert len(stub.calls) == 2
