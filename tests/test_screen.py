"""Tests for humorhist.harvest.screen -- the LLM funny pre-screen.

TDD with ``StubClient``: no network calls are ever made. Real file-backed
DBs via ``tmp_path`` are used to verify actual persistence.
"""

from __future__ import annotations

from pathlib import Path

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
    return [{"n": i + 1, "score": score, "reason": "ok"} for i in range(n_items)]


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
    stub = StubClient([_batch_response(10), _batch_response(10), _batch_response(5)])
    result = screen.screen_pool(conn, stub, batch_size=10)
    assert result["batches"] == 3
    assert len(stub.calls) == 3
    assert result["scored"] == 25
    assert result["failed_batches"] == 0


def test_screen_pool_stops_and_persists_failures(tmp_path):
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 15)
    stub = StubClient([LLMError("secret-token"), _batch_response(5)])
    result = screen.screen_pool(conn, stub, batch_size=10)
    assert (result["failed_batches"], result["scored"], result["skipped"], result["unscored"]) == (1, 0, 5, 15)
    assert len(stub.calls) == 1
    assert screen.format_screen_failures(result["failures"]) == "unknown=1"
    conn.close()
    conn = db.connect(str(tmp_path / "test.sqlite"))
    retry = StubClient([_batch_response(5)])
    result = screen.screen_pool(conn, retry, batch_size=10)
    assert result["skipped"] == 10 and result["scored"] == 5
    assert len(retry.calls) == 1
    explicit = StubClient([_batch_response(10)])
    assert screen.screen_pool(conn, explicit, retry_failed=True)["scored"] == 10


def test_pending_attempt_is_not_retried_even_with_explicit_failed_retry(tmp_path):
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 1)
    conn.execute("CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL)")
    conn.execute("INSERT INTO screen_attempts VALUES (?, 'pending')", (ids[0],))
    conn.commit()
    stub = StubClient([])
    result = screen.screen_pool(conn, stub, retry_failed=True)
    assert result["skipped"] == 1 and result["unscored"] == 1 and not stub.calls


def test_screen_failure_format_rejects_untrusted_values():
    assert screen.format_screen_failures([
        {"category": "secret", "status": 401},
        {"category": "http_error", "status": 429},
        {"category": "http_error", "status": "secret"},
    ]) == "http_error=1, http_error_http_429=1, unknown=1"


def test_screen_prompt_no_death_taste_penalty():
    """The pre-screen must NOT down-rank death/suffering topics.

    Regression guard for the editorial decision to let grim-but-absurd events
    score on absurdity alone. If a "PENALISE ... death ... tragedy" rule is
    ever reintroduced, this fails.
    """
    prompt = screen.SCREEN_SYSTEM_PROMPT
    forbidden = [
        "PENALISE",
        "penalise",
        "penalize",
        "human suffering",
        "death played",
        "tragedy",
        "not funny merely because",
    ]
    prompt_lower = prompt.lower()
    for term in forbidden:
        assert term.lower() not in prompt_lower, f"screen prompt still contains taste-filter wording: {term!r}"
    # Sanity: the absurdity-reward guidance is still present.
    assert "reward" in prompt_lower


def test_real_client_http_error_is_safe_and_single_request(tmp_path):
    import httpx
    import respx

    from humorhist.llm import NousClient

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 25)
    with respx.mock:
        route = respx.post("https://provider.invalid/chat/completions").mock(
            return_value=httpx.Response(429, text="secret-provider-body")
        )
        result = screen.screen_pool(conn, NousClient(api_key="secret", base_url="https://provider.invalid"), batch_size=10)
        assert route.call_count == 1
    assert result["unscored"] == 25 and result["skipped"] == 15
    assert screen.format_screen_failures(result["failures"]) == "http_error_http_429=1"
    assert "secret" not in repr(result)


def test_real_client_timeout_is_not_retried_or_resubmitted(tmp_path):
    import httpx
    import respx

    from humorhist.llm import NousClient

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 2)
    with respx.mock:
        route = respx.post("https://provider.invalid/chat/completions").mock(
            side_effect=httpx.ReadTimeout("secret timeout")
        )
        client = NousClient(api_key="secret", base_url="https://provider.invalid")
        result = screen.screen_pool(conn, client)
        assert route.call_count == 1
        assert screen.format_screen_failures(result["failures"]) == "timeout=1"
        assert screen.screen_pool(conn, client)["skipped"] == 2
        assert route.call_count == 1


def test_retry_crash_only_claims_current_batch(tmp_path):
    import pytest
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 3)
    conn.execute("CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL)")
    conn.executemany("INSERT INTO screen_attempts VALUES (?, 'failed')", [(i,) for i in ids])
    conn.commit()
    class Crash:
        def complete_json_once(self, *a, **kw):
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        screen.screen_pool(conn, Crash(), batch_size=1, retry_failed=True)
    assert [r['state'] for r in screen.screen_attempt_status(conn)] == ['pending', 'failed', 'failed']
    assert screen.screen_pool(conn, StubClient([]), limit=0)['scored'] == 0


def test_limit_excludes_blocked_rows_before_select(tmp_path):
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 3)
    conn.execute("CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL)")
    conn.execute("INSERT INTO screen_attempts VALUES (?, 'pending')", (ids[0],))
    conn.commit()
    result = screen.screen_pool(conn, StubClient([_batch_response(1)]), limit=1)
    assert result['scored'] == 1
    assert db.get_pool_item(conn, ids[1])['funny_score'] == 5


def test_batch_failure_rolls_back_all_scores_preserves_original_error(tmp_path):
    import sqlite3

    import pytest
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 2)
    conn.execute("CREATE TRIGGER fail_score BEFORE UPDATE OF funny_score ON pool WHEN NEW.id='pool001' BEGIN SELECT RAISE(ROLLBACK, 'score failure'); END")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match='score failure'):
        screen.screen_pool(conn, StubClient([_batch_response(2)]))
    assert all(db.get_pool_item(conn, i)['funny_score'] is None for i in ids)
    assert [r['state'] for r in screen.screen_attempt_status(conn)] == ['pending', 'pending']


def test_success_clears_reservation_and_rejects_nonfinite(tmp_path):
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 3)
    stub = StubClient([[{'n': 1, 'score': 3}, {'n': 2, 'score': float('nan')}, {'n': 3, 'score': float('inf')}]])
    result = screen.screen_pool(conn, stub)
    assert result['scored'] == 1 and result['unscored'] == 2
    assert [(r['pool_id'], r['state']) for r in screen.screen_attempt_status(conn)] == [(ids[1], 'failed'), (ids[2], 'failed')]


def test_concurrent_claim_prevents_duplicate_requests(tmp_path):
    import threading
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 1)
    entered, release = threading.Event(), threading.Event()
    calls = []
    class Blocking:
        def complete_json_once(self, *a, **kw):
            calls.append(1)
            entered.set()
            assert release.wait(5)
            return _batch_response(1)
    def run():
        c = db.connect(str(tmp_path / 'test.sqlite'))
        try:
            screen.screen_pool(c, Blocking(), retry_failed=True)
        finally:
            c.close()
    t = threading.Thread(target=run)
    t.start()
    assert entered.wait(5)
    try:
        result = screen.screen_pool(conn, StubClient([]), retry_failed=True)
        assert result['scored'] == 0 and len(calls) == 1
    finally:
        release.set()
        t.join(5)
    assert not t.is_alive()
    assert not screen.screen_attempt_status(conn)


def test_legacy_rescreen_disabled_without_db_or_network(monkeypatch, capsys):
    from scripts import rescreen_60
    assert rescreen_60.main() != 0
    assert 'disabled' in capsys.readouterr().out


def test_cli_screen_status_and_incomplete_exit(tmp_path, monkeypatch, capsys):
    from humorhist import cli, llm
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 1)
    conn.close()
    monkeypatch.setattr(llm, 'default_client', lambda: StubClient([[]]))
    path = str(tmp_path / 'test.sqlite')
    assert cli.main(['--db', path, 'screen']) == 1
    assert cli.main(['--db', path, 'screen-status']) == 0
    assert 'failed  pool000' in capsys.readouterr().out


def test_stale_retry_selection_cannot_claim_new_failure(tmp_path, monkeypatch):
    import threading

    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 1)
    conn.execute("CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL)")
    conn.execute("INSERT INTO screen_attempts VALUES (?, 'failed')", (ids[0],))
    conn.commit()
    selected = threading.Event()
    release = threading.Event()
    original = screen._select_unscored
    calls = []
    errors = []

    def select(c, limit, retry_failed=False):
        rows = original(c, limit, retry_failed)
        if threading.current_thread().name == "stale":
            selected.set()
            assert release.wait(5)
        return rows

    monkeypatch.setattr(screen, "_select_unscored", select)
    def run():
        c = db.connect(str(tmp_path / "test.sqlite"))
        try:
            stub = StubClient([LLMError("failure")])
            screen.screen_pool(c, stub, retry_failed=True)
            calls.extend(stub.calls)
        except BaseException as exc:
            errors.append(exc)
        finally:
            c.close()
    thread = threading.Thread(target=run, name="stale")
    thread.start()
    assert selected.wait(5)
    first = StubClient([LLMError("failure")])
    assert screen.screen_pool(conn, first, retry_failed=True)["failed_batches"] == 1
    release.set()
    thread.join(5)
    assert not thread.is_alive() and not errors
    assert len(first.calls) == 1 and not calls
    assert screen.screen_attempt_status(conn)[0]["state"] == "failed"
    assert conn.execute("SELECT generation FROM screen_attempts").fetchone()[0]


def test_invalid_and_partial_batches_stop_with_reservations(tmp_path):
    for index, response in enumerate(([], "not json", [{"n": 1, "score": 7}])):
        path = tmp_path / str(index)
        path.mkdir()
        conn = _fresh_db(path)
        ids = _insert_rows(conn, 4)
        stub = StubClient([response, _batch_response(2)])
        result = screen.screen_pool(conn, stub, batch_size=2)
        assert len(stub.calls) == 1 and result["failed_batches"] == 1
        assert result["scored"] == (1 if index == 2 else 0)
        assert result["blocked"] == (1 if index == 2 else 2)
        assert db.get_pool_item(conn, ids[2])["funny_score"] is None
        assert result["failures"][0]["category"] == "invalid_response"
        conn.close()


def test_screen_requires_single_attempt_interface(tmp_path):
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 1)
    class RetryingClient:
        def complete_json(self, *args, **kwargs):
            raise AssertionError("unsafe fallback called")
    result = screen.screen_pool(conn, RetryingClient())
    assert result["failed_batches"] == 1 and result["blocked"] == 1


def test_limited_cli_reports_blocked_but_succeeds(tmp_path, monkeypatch, capsys):
    from humorhist import cli, llm
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 2)
    conn.execute("CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL)")
    conn.execute("INSERT INTO screen_attempts VALUES (?, 'failed')", (ids[0],))
    conn.commit()
    conn.close()
    monkeypatch.setattr(llm, "default_client", lambda: StubClient([_batch_response(1)]))
    assert cli.main(["--db", str(tmp_path / "test.sqlite"), "screen", "--limit", "1"]) == 0
    output = capsys.readouterr().out
    assert "'blocked': 1" in output and "'unscored': 0" in output


def test_screen_all_limit_with_blocked_exits_nonzero(tmp_path, monkeypatch, capsys):
    import sys

    from humorhist import llm
    from scripts import screen_all
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 2)
    conn.execute("CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL)")
    conn.execute("INSERT INTO screen_attempts VALUES (?, 'failed')", (ids[0],))
    conn.commit()
    conn.close()
    monkeypatch.setattr(llm, "resilient_client", lambda **kw: StubClient([_batch_response(1)]))
    monkeypatch.setattr(screen_all, "LOG", tmp_path / "screen.log")
    monkeypatch.setattr(sys, "argv", ["screen_all", "--db", str(tmp_path / "test.sqlite"), "--limit", "1"])
    assert screen_all.main() == 1
    assert "blocked=1" in capsys.readouterr().out


def test_legacy_schema_migration_preserves_reservations_and_tokens(tmp_path):
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 2)
    conn.execute("CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL)")
    conn.executemany("INSERT INTO screen_attempts VALUES (?, ?)", [(ids[0], "pending"), (ids[1], "failed")])
    conn.commit()
    screen.screen_pool(conn, StubClient([]), limit=0)
    before = [tuple(row) for row in conn.execute("SELECT pool_id,state,generation FROM screen_attempts ORDER BY pool_id")]
    assert [(r[0], r[1]) for r in before] == [(ids[0], "pending"), (ids[1], "failed")]
    assert all(row[2] for row in before) and before[0][2] != before[1][2]
    screen.screen_pool(conn, StubClient([]), limit=0)
    assert before == [tuple(row) for row in conn.execute("SELECT pool_id,state,generation FROM screen_attempts ORDER BY pool_id")]
