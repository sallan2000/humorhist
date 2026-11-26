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


class _OfflineInjectedClient(StubClient):
    """Marker for local behavioral test doubles, admitted only by the fixture."""


@pytest.fixture(autouse=True)
def _inject_offline_behavioral_clients(monkeypatch):
    production_preflight = screen._preflight_client

    def check(client):
        if isinstance(client, _OfflineInjectedClient):
            return "free"
        return production_preflight(client)

    monkeypatch.setattr(screen, "_preflight_client", check)


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


def test_preflight_blocks_before_reservations_and_is_anonymous(tmp_path):
    import httpx
    import respx

    from humorhist.llm import ScreeningClient

    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 2)
    screen.screen_pool(conn, StubClient([]), limit=0)
    conn.execute("INSERT INTO screen_attempts(pool_id,state,generation) VALUES (?, 'failed', 'old')", (ids[0],))
    conn.commit()
    before = [tuple(r) for r in conn.execute("SELECT * FROM screen_attempts")]
    for response, expected in [
        (httpx.Response(200, json={"data": [{"id": "model", "pricing": {"prompt": "1", "completion": "0"}}]}), "unsafe_pricing"),
        (httpx.Response(200, json={"data": [{"id": "model", "pricing": {"prompt": "0"}}]}), "unsafe_pricing"),
        (httpx.Response(200, json={"data": []}), "model_missing"),
        (httpx.Response(500, text="secret-provider"), "unavailable"),
    ]:
        with respx.mock(assert_all_called=False) as mock:
            get = mock.get("https://inference-api.nousresearch.com/v1/models").mock(return_value=response)
            post = mock.post("https://inference-api.nousresearch.com/v1/chat/completions").mock()
            result = screen.screen_pool(conn, ScreeningClient(api_key="secret-key", model="model"), retry_failed=True)
            assert get.call_count == 1 and post.call_count == 0
            assert "authorization" not in get.calls[0].request.headers
        assert result["preflight_status"] == expected and result["preflight_failed"]
        assert result["failed_batches"] == result["batches"] == 0 and result["eligible_selected"] == 2
        assert screen.format_screen_failures(result["failures"]) == "preflight_" + expected + "=1"
        assert "secret" not in repr(result)
        assert [tuple(r) for r in conn.execute("SELECT * FROM screen_attempts")] == before
        assert conn.execute("SELECT COUNT(*) FROM screen_batches").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM pool WHERE funny_score IS NOT NULL").fetchone()[0] == 0


def test_preflight_success_checks_exact_model_and_no_eligible_rows_skip(tmp_path):
    import json

    import httpx
    import pytest
    import respx

    from humorhist.llm import ScreeningClient

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 1)
    client = ScreeningClient(api_key="secret-key", model="exact-model")
    with pytest.raises(AttributeError):
        client.model = "different-model"
    with pytest.raises(AttributeError):
        client._model = "different-model"
    with respx.mock(assert_all_called=False) as mock:
        checks = []
        def catalog(request):
            assert not conn.in_transaction
            checks.append(conn.execute("SELECT COUNT(*) FROM screen_attempts").fetchone()[0])
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"data": [{"id": "exact-model", "pricing": {"prompt": "0", "completion": "0"}}]})
        get = mock.get("https://inference-api.nousresearch.com/v1/models").mock(side_effect=catalog)
        post = mock.post("https://inference-api.nousresearch.com/v1/chat/completions").mock(return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": '[{"n":1,"score":5}]'}}]}))
        assert screen.screen_pool(conn, client)["scored"] == 1
        assert json.loads(post.calls[0].request.content)["model"] == "exact-model"
        assert get.call_count == 2 and post.call_count == 1 and checks == [0, 1]
        assert screen.screen_pool(conn, client)["scored"] == 0
        assert get.call_count == 2 and post.call_count == 1


def test_preflight_invalid_model_and_generic_nous_fail_closed(tmp_path):
    import respx

    from humorhist.llm import NousClient, ScreeningClient

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 1)
    with respx.mock(assert_all_called=False) as mock:
        route = mock.route().mock()
        assert screen.screen_pool(conn, ScreeningClient(api_key="key", model="bad model"))["preflight_status"] == "invalid_model"
        assert screen.screen_pool(conn, NousClient(api_key="key"))["preflight_status"] == "unavailable"
        assert route.call_count == 0
        assert conn.execute("SELECT COUNT(*) FROM screen_attempts").fetchone()[0] == 0


def test_each_pool_batch_checks_catalog_before_post(tmp_path):
    import httpx
    import respx

    from humorhist.llm import ScreeningClient

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 2)
    events = []
    with respx.mock(assert_all_called=False) as mock:
        def catalog(request):
            events.append("get")
            return httpx.Response(200, json={"data": [{"id": "exact", "pricing": {"prompt": "0", "completion": "0"}}]})

        def completion(request):
            events.append("post")
            return httpx.Response(200, json={"choices": [{"message": {"content": '[{"n":1,"score":5}]'}}]})

        mock.get("https://inference-api.nousresearch.com/v1/models").mock(side_effect=catalog)
        mock.post("https://inference-api.nousresearch.com/v1/chat/completions").mock(side_effect=completion)
        assert screen.screen_pool(conn, ScreeningClient(api_key="secret", model="exact"), batch_size=1)["scored"] == 2
    assert events == ["get", "get", "post", "get", "post"]


def test_second_catalog_failure_blocks_post_after_claim(tmp_path):
    import httpx
    import respx

    from humorhist.llm import ScreeningClient

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 2)
    with respx.mock(assert_all_called=False) as mock:
        get = mock.get("https://inference-api.nousresearch.com/v1/models").mock(side_effect=[
            httpx.Response(200, json={"data": [{"id": "exact", "pricing": {"prompt": "0", "completion": "0"}}]}),
            httpx.Response(503, text="private-error"),
        ])
        post = mock.post("https://inference-api.nousresearch.com/v1/chat/completions").mock()
        result = screen.screen_pool(conn, ScreeningClient(api_key="secret", model="exact"), batch_size=1)
        assert get.call_count == 2 and post.call_count == 0
    assert result["preflight_failed"] and result["preflight_status"] == "unavailable"
    assert result["preflight_post_blocked"] == result["batches"] == 1 and result["failed_batches"] == 0
    assert result["scored"] == 0 and result["skipped"] == 1
    assert screen.screen_attempt_status(conn) == [] and result["blocked"] == 0
    assert "private" not in repr(result) and "secret" not in repr(result)


def test_postclaim_refusal_restores_previous_failed_generation(tmp_path):
    import httpx
    import respx

    from humorhist.llm import LLMCallError, ScreeningClient

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 1)
    screen.screen_pool(conn, StubClient([LLMCallError('timeout')]))
    before = dict(conn.execute('SELECT * FROM screen_attempts').fetchone())
    with respx.mock(assert_all_called=False) as mock:
        mock.get('https://inference-api.nousresearch.com/v1/models').mock(side_effect=[
            httpx.Response(200, json={'data': [{'id': 'exact', 'pricing': {'prompt': '0', 'completion': '0'}}]}),
            httpx.Response(503),
        ])
        post = mock.post('https://inference-api.nousresearch.com/v1/chat/completions').mock()
        result = screen.screen_pool(conn, ScreeningClient(api_key='secret', model='exact'), retry_failed=True)
    assert post.call_count == 0 and result['failed_batches'] == 0 and result['blocked'] == 1
    assert dict(conn.execute('SELECT * FROM screen_attempts').fetchone()) == before
    assert not result['release_uncertain']


def test_postclaim_refusal_cas_mismatch_retains_other_generation(tmp_path):
    import httpx
    import respx

    from humorhist.llm import ScreeningClient

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 2)
    count = 0

    def catalog(request):
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(200, json={'data': [{'id': 'exact', 'pricing': {'prompt': '0', 'completion': '0'}}]})
        other = db.connect(str(tmp_path / 'test.sqlite'))
        other.execute("UPDATE screen_attempts SET generation='new-owner' WHERE pool_id='pool000'")
        other.commit()
        other.close()
        return httpx.Response(503)

    with respx.mock(assert_all_called=False) as mock:
        mock.get('https://inference-api.nousresearch.com/v1/models').mock(side_effect=catalog)
        post = mock.post('https://inference-api.nousresearch.com/v1/chat/completions').mock()
        result = screen.screen_pool(conn, ScreeningClient(api_key='secret', model='exact'))
    assert post.call_count == 0 and result['release_uncertain'] and result['failed_batches'] == 0
    assert result['blocked'] == 2
    assert conn.execute("SELECT generation FROM screen_attempts WHERE pool_id='pool000'").fetchone()[0] == 'new-owner'


def test_postclaim_release_write_failure_rolls_back_whole_batch(tmp_path):
    import httpx
    import respx

    from humorhist.llm import ScreeningClient

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 2)
    screen.screen_pool(conn, StubClient([]), limit=0)
    conn.execute("CREATE TRIGGER no_release BEFORE DELETE ON screen_attempts WHEN OLD.pool_id='pool001' "
                 "BEGIN SELECT RAISE(ABORT, 'release failed'); END")
    conn.commit()
    with respx.mock(assert_all_called=False) as mock:
        mock.get('https://inference-api.nousresearch.com/v1/models').mock(side_effect=[
            httpx.Response(200, json={'data': [{'id': 'exact', 'pricing': {'prompt': '0', 'completion': '0'}}]}),
            httpx.Response(503),
        ])
        post = mock.post('https://inference-api.nousresearch.com/v1/chat/completions').mock()
        result = screen.screen_pool(conn, ScreeningClient(api_key='secret', model='exact'))
    assert post.call_count == 0 and result['release_uncertain'] and result['blocked'] == 2
    assert result['failed_batches'] == 0
    assert [r['state'] for r in screen.screen_attempt_status(conn)] == ['pending', 'pending']
    assert [r['state'] for r in conn.execute('SELECT state FROM screen_batches')] == ['pending']


def test_forged_approval_and_grafted_stub_methods_cannot_send(tmp_path):
    import httpx
    import respx

    from humorhist.llm import LLMCallError, NousClient, ScreeningClient

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 1)
    live = ScreeningClient(api_key="secret", model="exact")
    live._approved_model = "exact"  # Legacy approval-shaped value must be ignored.
    provider = NousClient(api_key="secret", base_url="https://provider.invalid")
    stub = StubClient([_batch_response(1), _batch_response(1)])
    stub.complete_json_once = provider.complete_json_once
    stub.complete_json = provider.complete_json
    with respx.mock(assert_all_called=False) as mock:
        get = mock.get("https://inference-api.nousresearch.com/v1/models").mock(return_value=httpx.Response(503))
        post = mock.route(method="POST").mock()
        with pytest.raises(LLMCallError, match="preflight_unavailable"):
            live.complete_json_once("s", "u")
        assert get.call_count == 1 and post.call_count == 0
        assert screen.score_batch(stub, [{"id": "a", "title": "a"}]) == {"a": 5.0}
        assert screen.screen_pool(conn, stub)["scored"] == 1
        assert get.call_count == 1 and post.call_count == 0


def test_direct_completion_checks_catalog_for_every_post():
    import httpx
    import respx

    from humorhist.llm import LLMCallError, ScreeningClient

    client = ScreeningClient(api_key="secret", model="exact")
    with respx.mock(assert_all_called=False) as mock:
        get = mock.get("https://inference-api.nousresearch.com/v1/models").mock(side_effect=[
            httpx.Response(200, json={"data": [{"id": "exact", "pricing": {"prompt": "0", "completion": "0"}}]}),
            httpx.Response(200, json={"data": [{"id": "exact", "pricing": {"prompt": "1", "completion": "0"}}]}),
        ])
        post = mock.post("https://inference-api.nousresearch.com/v1/chat/completions").mock(return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": '[]'}}]}))
        assert client.complete_json_once("s", "u") == []
        with pytest.raises(LLMCallError, match="preflight_unsafe_pricing"):
            client.complete_json_once("s", "u")
        assert get.call_count == 2 and post.call_count == 1


def test_wrapped_or_raw_provider_cannot_bypass_preflight(tmp_path):
    import respx

    from humorhist.llm import LLMCallError, NousClient

    class Wrapper:
        def __init__(self):
            self.inner = NousClient(api_key="secret", base_url="https://provider.invalid")

        def complete_json_once(self, *args, **kwargs):
            return self.inner.complete_json_once(*args, **kwargs)

    class ForgedWrapper(Wrapper):
        def preflight(self):
            return "free"

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 1)
    with respx.mock(assert_all_called=False) as mock:
        route = mock.route().mock()
        for client in (Wrapper(), ForgedWrapper(), NousClient(api_key="secret", base_url="https://provider.invalid")):
            result = screen.screen_pool(conn, client)
            assert result["preflight_failed"] and result["eligible_selected"] == 1
            assert result["batches"] == result["failed_batches"] == 0
            with pytest.raises(LLMCallError, match="preflight_unavailable"):
                screen.score_batch(client, [{"id": "1", "title": "test"}])
        assert route.call_count == 0
        assert conn.execute("SELECT COUNT(*) FROM screen_attempts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM screen_batches").fetchone()[0] == 0


def test_screen_all_logs_preflight_failure_before_break(tmp_path, monkeypatch, capsys):
    import sys

    import httpx
    import respx

    from humorhist import llm
    from scripts import screen_all

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 1)
    conn.close()
    monkeypatch.setattr(llm, "screening_client", lambda **kwargs: llm.ScreeningClient(api_key="secret", model="exact"))
    monkeypatch.setattr(screen_all, "LOG", tmp_path / "screen.log")
    monkeypatch.setattr(sys, "argv", ["screen_all", "--db", str(tmp_path / "test.sqlite")])
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://inference-api.nousresearch.com/v1/models").mock(return_value=httpx.Response(500, text="secret-body"))
        post = mock.post("https://inference-api.nousresearch.com/v1/chat/completions").mock()
        assert screen_all.main() == 1
        assert post.call_count == 0
    output = capsys.readouterr().out
    assert "preflight_unavailable=1" in output
    assert "secret" not in output
    assert "preflight_unavailable=1" in (tmp_path / "screen.log").read_text()


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

    from humorhist.llm import ScreeningClient

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 25)
    with respx.mock:
        respx.get("https://inference-api.nousresearch.com/v1/models").mock(return_value=httpx.Response(
            200, json={"data": [{"id": "test-model", "pricing": {"prompt": "0", "completion": "0"}}]}))
        route = respx.post("https://inference-api.nousresearch.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, text="secret-provider-body")
        )
        result = screen.screen_pool(conn, ScreeningClient(api_key="secret", model="test-model"), batch_size=10)
        assert route.call_count == 1
    assert result["unscored"] == 25 and result["skipped"] == 15
    assert screen.format_screen_failures(result["failures"]) == "http_error_http_429=1"
    assert "secret" not in repr(result)


def test_real_client_timeout_is_not_retried_or_resubmitted(tmp_path):
    import httpx
    import respx

    from humorhist.llm import ScreeningClient

    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 2)
    with respx.mock:
        respx.get("https://inference-api.nousresearch.com/v1/models").mock(return_value=httpx.Response(
            200, json={"data": [{"id": "test-model", "pricing": {"prompt": "0", "completion": "0"}}]}))
        route = respx.post("https://inference-api.nousresearch.com/v1/chat/completions").mock(
            side_effect=httpx.ReadTimeout("secret timeout")
        )
        client = ScreeningClient(api_key="secret", model="test-model")
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
    class Crash(_OfflineInjectedClient):
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
    assert result['scored'] == 1 and result['eligible_selected'] == 1
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
    class Blocking(_OfflineInjectedClient):
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
    monkeypatch.setattr(llm, 'screening_client', lambda: StubClient([[]]))
    path = str(tmp_path / 'test.sqlite')
    assert cli.main(['--db', path, 'screen']) == 1
    assert cli.main(['--db', path, 'screen-status']) == 0
    assert 'failed  id_json="pool000"' in capsys.readouterr().out


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
    assert result["preflight_failed"] and result["failed_batches"] == 0 and result["blocked"] == 0


def test_limited_cli_reports_blocked_but_succeeds(tmp_path, monkeypatch, capsys):
    from humorhist import cli, llm
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 2)
    conn.execute("CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL)")
    conn.execute("INSERT INTO screen_attempts VALUES (?, 'failed')", (ids[0],))
    conn.commit()
    conn.close()
    monkeypatch.setattr(llm, "screening_client", lambda: StubClient([_batch_response(1)]))
    assert cli.main(["--db", str(tmp_path / "test.sqlite"), "screen", "--limit", "1"]) == 0
    output = capsys.readouterr().out
    assert "'blocked': 1" in output and "'unscored': 0" in output


def test_screen_all_limit_with_blocked_exits_zero(tmp_path, monkeypatch, capsys):
    import sys

    from humorhist import llm
    from scripts import screen_all
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 2)
    conn.execute("CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL)")
    conn.execute("INSERT INTO screen_attempts VALUES (?, 'failed')", (ids[0],))
    conn.commit()
    conn.close()
    monkeypatch.setattr(llm, "screening_client", lambda **kw: StubClient([_batch_response(1)]))
    monkeypatch.setattr(screen_all, "LOG", tmp_path / "screen.log")
    monkeypatch.setattr(sys, "argv", ["screen_all", "--db", str(tmp_path / "test.sqlite"), "--limit", "1"])
    assert screen_all.main() == 0
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


def test_diagnostics_migrate_legacy_without_mutating_status(tmp_path):
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 9)
    conn.execute("CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL, generation TEXT)")
    conn.executemany("INSERT INTO screen_attempts VALUES (?, 'failed', ?)", [(i, 'old-' + i) for i in ids])
    conn.commit()
    before = conn.execute("SELECT sql FROM sqlite_master WHERE name='screen_attempts'").fetchone()[0]
    status = screen.screen_attempt_status(conn)
    assert len(status) == 9 and all(r['category'] is None and r['started_at'] is None for r in status)
    assert conn.execute("SELECT sql FROM sqlite_master WHERE name='screen_attempts'").fetchone()[0] == before
    for _ in range(2):
        screen.screen_pool(conn, StubClient([]), limit=0)
    rows = conn.execute("SELECT pool_id, generation, category, status, started_at, completed_at, run_id, batch_id FROM screen_attempts ORDER BY pool_id").fetchall()
    assert len(rows) == 9 and all(r[1] == 'old-' + r[0] and all(x is None for x in r[2:]) for r in rows)


def test_persist_safe_failure_diagnostics_and_retry(tmp_path):
    from humorhist.llm import LLMCallError
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 2)
    screen.screen_pool(conn, StubClient([LLMCallError('http_error', 429)]))
    rows = screen.screen_attempt_status(conn)
    assert all(r['category'] == 'http_error' and r['status'] == 429 and r['started_at'] and r['completed_at'] for r in rows)
    assert rows[0]['run_id'] == rows[1]['run_id'] and rows[0]['batch_id'] == rows[1]['batch_id']
    old_run = rows[0]['run_id']
    screen.screen_pool(conn, StubClient([LLMCallError('secret', 401)]), retry_failed=True)
    rows = screen.screen_attempt_status(conn)
    assert all(r['category'] == 'unknown' and r['status'] is None and r['run_id'] != old_run for r in rows)
    assert 'secret' not in repr(rows)


def test_partial_diagnostics_atomic_and_pending_during_call(tmp_path):
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 2)
    class Partial(_OfflineInjectedClient):
        def complete_json_once(self, *args, **kwargs):
            assert not conn.in_transaction
            rows = screen.screen_attempt_status(conn)
            assert all(r['state'] == 'pending' and r['started_at'] and r['completed_at'] is None for r in rows)
            return [{'n': 1, 'score': 4}]
    screen.screen_pool(conn, Partial())
    row = screen.screen_attempt_status(conn)[0]
    assert row['state'] == 'failed' and row['category'] == 'invalid_response' and row['completed_at']


def test_stale_outcome_cannot_change_new_generation(tmp_path):
    import pytest
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 1)
    class Replaced(_OfflineInjectedClient):
        def complete_json_once(self, *args, **kwargs):
            other = db.connect(str(tmp_path / 'test.sqlite'))
            other.execute("UPDATE screen_attempts SET generation='new-generation' WHERE pool_id=?", (ids[0],))
            other.commit()
            other.close()
            return _batch_response(1)
    with pytest.raises(RuntimeError):
        screen.screen_pool(conn, Replaced())
    assert db.get_pool_item(conn, ids[0])['funny_score'] is None
    assert conn.execute('SELECT generation, state FROM screen_attempts').fetchone()[:] == ('new-generation', 'pending')


def test_diagnostic_write_failure_rolls_back_entire_batch(tmp_path):
    import sqlite3

    import pytest
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 2)
    screen.screen_pool(conn, StubClient([]), limit=0)
    conn.execute("CREATE TRIGGER fail_diagnostic BEFORE UPDATE OF category ON screen_attempts "
                 "WHEN NEW.pool_id='pool001' BEGIN SELECT RAISE(ROLLBACK, 'diagnostic failure'); END")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match='diagnostic failure'):
        screen.screen_pool(conn, StubClient([[{'n': 1, 'score': 6}]]))
    assert all(db.get_pool_item(conn, i)['funny_score'] is None for i in ids)
    rows = screen.screen_attempt_status(conn)
    assert all(r['state'] == 'pending' and r['category'] is None and r['completed_at'] is None for r in rows)


def test_stale_failure_does_not_overwrite_new_generation(tmp_path):
    import pytest
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 1)
    class Replaced(_OfflineInjectedClient):
        def complete_json_once(self, *args, **kwargs):
            other = db.connect(str(tmp_path / 'test.sqlite'))
            other.execute("UPDATE screen_attempts SET generation='new', category='timeout' WHERE pool_id=?", (ids[0],))
            other.commit()
            other.close()
            raise LLMError('secret body')
    with pytest.raises(RuntimeError, match='reservation changed'):
        screen.screen_pool(conn, Replaced())
    assert conn.execute('SELECT state, generation, category FROM screen_attempts').fetchone()[:] == ('pending', 'new', 'timeout')


def test_retry_claim_clears_previous_failure_details(tmp_path):
    from humorhist.llm import LLMCallError
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 1)
    screen.screen_pool(conn, StubClient([LLMCallError('http_error', 500)]))
    previous = screen.screen_attempt_status(conn)[0]
    class Crash(_OfflineInjectedClient):
        def complete_json_once(self, *args, **kwargs):
            row = screen.screen_attempt_status(conn)[0]
            assert row['state'] == 'pending' and row['started_at']
            assert row['category'] is None and row['status'] is None and row['completed_at'] is None
            assert row['run_id'] != previous['run_id'] and row['batch_id'] != previous['batch_id']
            raise KeyboardInterrupt()
    import pytest
    with pytest.raises(KeyboardInterrupt):
        screen.screen_pool(conn, Crash(), retry_failed=True)
    assert screen.screen_attempt_status(conn)[0]['state'] == 'pending'


def test_batch_log_success_is_durable_and_contains_only_safe_metadata(tmp_path):
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 2)
    result = screen.screen_pool(conn, StubClient([_batch_response(2)]))
    assert result['scored'] == 2 and screen.screen_attempt_status(conn) == []
    row = dict(conn.execute("SELECT * FROM screen_batches WHERE state='success'").fetchone())
    assert row['state'] == 'success' and row['started_at'] and row['completed_at']
    assert row['selected_count'] == 2 and row['scored_count'] == 2 and row['failed_count'] == 0
    assert row['category'] is None and row['status'] is None and row['run_id'] and row['batch_id']
    assert 'pool000' not in repr(row) and 'Summary' not in repr(row)
    assert [r[0] for r in conn.execute('SELECT state FROM screen_batches ORDER BY rowid')] == ['pending', 'success']


def test_batch_log_pending_survives_crash_and_legacy_migration(tmp_path):
    import pytest
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 2)
    conn.execute('CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL, generation TEXT)')
    conn.execute("INSERT INTO screen_attempts VALUES (?, 'failed', 'legacy')", (ids[0],))
    conn.commit()
    class Crash(_OfflineInjectedClient):
        def complete_json_once(self, *args, **kwargs):
            assert not conn.in_transaction
            row = dict(conn.execute('SELECT * FROM screen_batches').fetchone())
            assert row['state'] == 'pending' and row['completed_at'] is None and row['selected_count'] == 1
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        screen.screen_pool(conn, Crash())
    screen.screen_pool(conn, StubClient([]), limit=0)
    assert conn.execute('SELECT state,generation FROM screen_attempts WHERE pool_id=?', (ids[0],)).fetchone()[:] == ('failed', 'legacy')
    row = dict(conn.execute('SELECT * FROM screen_batches').fetchone())
    assert row['state'] == 'pending' and row['started_at'] and row['completed_at'] is None
    assert row['scored_count'] is None and row['failed_count'] is None


def test_batch_log_partial_and_failure_are_sanitized(tmp_path):
    from humorhist.llm import LLMCallError
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 2)
    screen.screen_pool(conn, StubClient([[{'n': 1, 'score': 3}]]))
    screen.screen_pool(conn, StubClient([LLMCallError('http_error', 429)]), retry_failed=True)
    rows = [dict(r) for r in conn.execute("SELECT * FROM screen_batches WHERE state != 'pending' ORDER BY rowid")]
    assert [(r['state'], r['selected_count'], r['scored_count'], r['failed_count'], r['category'], r['status']) for r in rows] == [
        ('partial', 2, 1, 1, 'invalid_response', None), ('failure', 1, 0, 1, 'http_error', 429)]
    assert all(r['completed_at'] for r in rows) and rows[0]['run_id'] != rows[1]['run_id']


def test_batch_log_outcome_rolls_back_with_failed_score_write(tmp_path):
    import sqlite3

    import pytest
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 2)
    conn.execute("CREATE TRIGGER broken BEFORE UPDATE OF funny_score ON pool WHEN NEW.id='pool001' BEGIN SELECT RAISE(ROLLBACK, 'broken'); END")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match='broken'):
        screen.screen_pool(conn, StubClient([_batch_response(2)]))
    row = dict(conn.execute('SELECT * FROM screen_batches').fetchone())
    assert row['state'] == 'pending' and row['completed_at'] is None and row['scored_count'] is None
    assert all(r['state'] == 'pending' for r in screen.screen_attempt_status(conn))


def test_batch_log_insert_failure_rolls_back_reservation_outcomes(tmp_path):
    import sqlite3

    import pytest
    conn = _fresh_db(tmp_path)
    ids = _insert_rows(conn, 2)
    screen.screen_pool(conn, StubClient([]), limit=0)
    conn.execute("CREATE TRIGGER broken_log BEFORE INSERT ON screen_batches WHEN NEW.state='partial' "
                 "BEGIN SELECT RAISE(ABORT, 'log failure'); END")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match='log failure'):
        screen.screen_pool(conn, StubClient([[{'n': 1, 'score': 5}]]))
    assert all(db.get_pool_item(conn, i)['funny_score'] is None for i in ids)
    assert all(r['state'] == 'pending' and r['category'] is None for r in screen.screen_attempt_status(conn))
    assert [r[0] for r in conn.execute('SELECT state FROM screen_batches')] == ['pending']


def test_batch_log_multiple_batches_share_run_and_do_not_reveal_errors(tmp_path):
    from humorhist.llm import LLMCallError
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 2)
    result = screen.screen_pool(conn, StubClient([_batch_response(1), LLMCallError('secret-body', 401)]), batch_size=1)
    rows = [dict(r) for r in conn.execute("SELECT * FROM screen_batches WHERE state != 'pending' ORDER BY rowid")]
    assert len(rows) == 2 and rows[0]['run_id'] == rows[1]['run_id']
    assert rows[0]['batch_id'] != rows[1]['batch_id']
    assert result['run_id'] == rows[0]['run_id']
    assert result['batch_ids'] == [row['batch_id'] for row in rows]
    assert rows[1]['category'] == 'unknown' and rows[1]['status'] is None
    assert 'secret-body' not in repr(rows)


def test_empty_response_is_logged_as_failure(tmp_path):
    conn = _fresh_db(tmp_path)
    _insert_rows(conn, 1)
    screen.screen_pool(conn, StubClient([[]]))
    row = conn.execute("SELECT state, scored_count, failed_count, category FROM screen_batches WHERE state != 'pending'").fetchone()
    assert row[:] == ('failure', 0, 1, 'invalid_response')
