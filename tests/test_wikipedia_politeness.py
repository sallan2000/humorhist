"""Offline tests of identification, pacing and access-denial handling."""
from datetime import UTC, datetime

import httpx
import pytest

from humorhist import db, drafting
from humorhist.factcheck import FactCheckError, fetch_wikipedia_extract
from humorhist.harvest import wikipedia_lists as wiki
from humorhist.wikipedia import USER_AGENT, parse_retry_after


@pytest.mark.parametrize('value,expected', [
    ('30', 30), ('9999999999999999999999999999999', 3600),
    ('-1', None), ('1.5', None), ('junk', None), ('9' * 129, None),
    ('Wed, 21 Oct 2015 07:28:00 GMT', 60),
    ('Wed, 21 Oct 2015 07:26:00 GMT', 0),
])
def test_retry_after_bounded(value, expected):
    now = datetime(2015, 10, 21, 7, 27, tzinfo=UTC)
    assert parse_retry_after(value, now=now) == expected


@pytest.mark.parametrize('status', [403, 429])
def test_harvest_stops_and_counts_unattempted(respx_mock, tmp_path, status, caplog):
    route = respx_mock.get(wiki.API_URL).mock(return_value=httpx.Response(status, text='SECRET', headers={'Retry-After': '99999'}))
    conn = db.connect(str(tmp_path / 'db'))
    db.migrate(conn)
    try:
        result = wiki.harvest_wikipedia_lists(conn, pages=['a', 'b', 'c'], sleep_seconds=0)
    finally:
        conn.close()
    assert route.call_count == 1
    assert result['pages_failed'] == 1 and result['pages_unattempted'] == 2
    assert result[f'http_{status}'] == 1
    assert 'SECRET' not in caplog.text


def test_paces_redirect_and_after_error(respx_mock, tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(wiki.time, 'sleep', sleeps.append)
    responses = [httpx.Response(500), httpx.Response(200, json={'parse': {'wikitext': '#REDIRECT [[target]]'}}), httpx.Response(200, json={'parse': {'wikitext': 'text'}})]
    route = respx_mock.get(wiki.API_URL).mock(side_effect=responses)
    conn = db.connect(str(tmp_path / 'db'))
    db.migrate(conn)
    try:
        result = wiki.harvest_wikipedia_lists(conn, pages=['a', 'b'], sleep_seconds=0.25)
    finally:
        conn.close()
    assert result['pages_failed'] == 1 and result['pages_fetched'] == 1
    assert sleeps == [0.25, 0.25]
    assert all(call.request.headers['User-Agent'] == USER_AGENT for call in route.calls)


def test_factcheck_rejects_automatic_redirect_even_with_injected_client(respx_mock):
    route = respx_mock.get('https://en.wikipedia.org/api/rest_v1/page/summary/Foo').mock(return_value=httpx.Response(302, headers={'Location': 'https://attacker.example/other'}))
    with httpx.Client(follow_redirects=True, headers={'User-Agent': 'bad'}) as client:
        with pytest.raises(FactCheckError, match='unsupported origin'):
            fetch_wikipedia_extract('Foo', client)
    assert route.call_count == 1
    assert route.calls[0].request.headers['User-Agent'] == USER_AGENT


@pytest.mark.parametrize('status', [403, 429])
def test_drafting_stops_extracts_on_access_denial(monkeypatch, status):
    rows = [{'id': str(i)} for i in range(3)]
    monkeypatch.setattr(drafting, 'select_candidates', lambda *a, **kw: rows)
    calls = []
    def attempt(*args, **kwargs):
        calls.append(1)
        def fail():
            exc = FactCheckError('SECRET')
            exc.status_code = status
            raise exc
        drafting._stage('extract', fail)
    monkeypatch.setattr(drafting, 'draft_one', attempt)
    result = drafting.draft_candidates(None, None)
    assert len(calls) == 1 and result['failed'] == 1 and result['unattempted'] == 2 and result['drafted'] == 0
    assert 'SECRET' not in str(result)


def test_factcheck_redirect_is_paced_and_identified(respx_mock, monkeypatch):
    from humorhist import factcheck
    sleeps = []
    monkeypatch.setattr(factcheck.time, 'sleep', sleeps.append)
    first = respx_mock.get('https://en.wikipedia.org/api/rest_v1/page/summary/Foo').mock(return_value=httpx.Response(302, headers={'Location': '/other'}))
    second = respx_mock.get('https://en.wikipedia.org/other').mock(return_value=httpx.Response(200, json={'extract': 'ok'}))
    with httpx.Client(follow_redirects=True) as client:
        assert fetch_wikipedia_extract('Foo', client) == 'ok'
    assert sleeps == [0.5]
    assert first.calls[0].request.headers['User-Agent'] == second.calls[0].request.headers['User-Agent'] == USER_AGENT


def test_retry_after_exposed_only_as_bounded_hint(respx_mock):
    respx_mock.get(wiki.API_URL).mock(return_value=httpx.Response(429, headers={'Retry-After': '99999'}, text='SECRET'))
    with pytest.raises(wiki.HarvestError) as err:
        wiki.fetch_page_wikitext('x', sleep_seconds=0)
    assert err.value.retry_after == 3600
    assert 'SECRET' not in str(err.value)


def test_redirect_failure_stops_and_preserves_counts(respx_mock, tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(wiki.time, 'sleep', sleeps.append)
    route = respx_mock.get(wiki.API_URL).mock(side_effect=[httpx.Response(200, json={'parse': {'wikitext': '#REDIRECT [[target]]'}}), httpx.Response(429)])
    conn = db.connect(str(tmp_path / 'db'))
    db.migrate(conn)
    try:
        result = wiki.harvest_wikipedia_lists(conn, pages=['a', 'b'], sleep_seconds=0.25)
    finally:
        conn.close()
    assert route.call_count == 2 and sleeps == [0.25]
    assert result['http_429'] == result['pages_failed'] == result['pages_unattempted'] == 1
