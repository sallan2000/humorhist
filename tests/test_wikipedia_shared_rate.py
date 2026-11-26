"""Cross-entry-point and cross-process Wikimedia transport tests, offline."""
import argparse
import multiprocessing
import time

import httpx
import pytest

from humorhist import cli, db, drafting
from humorhist.factcheck import FactCheckError, fetch_wikipedia_extract
from humorhist.harvest import wikipedia_lists as wiki
from humorhist.wikipedia import WikipediaCooldown, request


def _attempt_from_process(queue):
    try:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))) as client:
            request(client, 'https://en.wikipedia.org/w/api.php')
    except WikipediaCooldown:
        queue.put('blocked')
    else:
        queue.put('sent')


def _interrupted_request(entered):
    def wait_forever(req):
        entered.set()
        time.sleep(30)
        return httpx.Response(200)
    with httpx.Client(transport=httpx.MockTransport(wait_forever)) as client:
        request(client, 'https://en.wikipedia.org/w/api.php')


def test_process_death_releases_lock_but_fails_closed():
    ctx = multiprocessing.get_context('fork')
    entered = ctx.Event()
    proc = ctx.Process(target=_interrupted_request, args=(entered,))
    proc.start()
    try:
        assert entered.wait(5)
    finally:
        proc.terminate()
        proc.join(5)
    queue = ctx.Queue()
    next_proc = ctx.Process(target=_attempt_from_process, args=(queue,))
    next_proc.start()
    next_proc.join(5)
    assert next_proc.exitcode == 0 and queue.get(timeout=1) == 'blocked'


def test_429_shared_across_endpoints_and_process(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(429, headers={'Retry-After': '7200'}, text='SECRET')
    )) as client:
        with pytest.raises(wiki.HarvestError) as exc:
            wiki.fetch_page_wikitext('page', client, sleep_seconds=0)
    assert exc.value.status_code == 429
    state = tmp_path / '.cache/humorhist/wikipedia-rate.state'
    assert float(state.read_text().split()[1]) > time.time() + 7100
    def no_request(req):
        pytest.fail('cooldown was bypassed')
    with httpx.Client(transport=httpx.MockTransport(no_request)) as client:
        with pytest.raises(FactCheckError) as err:
            fetch_wikipedia_extract('page', client)
    assert err.value.local_cooldown and 'SECRET' not in str(err.value)
    ctx = multiprocessing.get_context('fork')
    queue = ctx.Queue()
    proc = ctx.Process(target=_attempt_from_process, args=(queue,))
    proc.start()
    proc.join(5)
    assert proc.exitcode == 0 and queue.get(timeout=1) == 'blocked'


def test_redirect_retry_after_prevents_second_hop():
    calls = []
    def reply(req):
        calls.append(req)
        return httpx.Response(302, headers={'Location': '/other', 'Retry-After': '60'}, text='SECRET')
    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        with pytest.raises(FactCheckError) as err:
            fetch_wikipedia_extract('page', client)
    assert len(calls) == 1 and err.value.local_cooldown
    assert 'SECRET' not in str(err.value)


def test_cli_harvest_partial_failure_nonzero(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(wiki, 'harvest_wikipedia_lists', lambda conn: {
        'inserted': 1, 'pages_failed': 1, 'pages_unattempted': 3,
    })
    args = argparse.Namespace(db=str(tmp_path / 'db'), wikipedia_only=True, seed_only=False)
    assert cli.cmd_harvest(args) == 1
    assert "'pages_unattempted': 3" in capsys.readouterr().out


def test_harvest_then_draft_cooldown_counts_selected_and_unattempted(tmp_path, monkeypatch):
    with httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(429, headers={'Retry-After': '60'}, text='SECRET')
    )) as client:
        conn = db.connect(str(tmp_path / 'db'))
        db.migrate(conn)
        result = wiki.harvest_wikipedia_lists(conn, pages=['a', 'b'], client=client, sleep_seconds=0)
        assert result['pages_failed'] == 1 and result['pages_unattempted'] == 1
        second = wiki.harvest_wikipedia_lists(conn, pages=['a', 'b'], client=client, sleep_seconds=0)
        assert second['pages_failed'] == 0 and second['pages_unattempted'] == 2
        rows = [{'id': str(i)} for i in range(3)]
        monkeypatch.setattr(drafting, 'select_candidates', lambda *a, **kw: rows)
        def draft_one(*args, **kwargs):
            return drafting._stage('extract', fetch_wikipedia_extract, 'page', client=client)
        monkeypatch.setattr(drafting, 'draft_one', draft_one)
        drafted = drafting.draft_candidates(conn, None)
        conn.close()
    assert drafted['selected'] == 3 and drafted['unattempted'] == 3 and drafted['failed'] == 0
    assert 'SECRET' not in str(drafted)


def test_lock_wait_is_bounded(monkeypatch):
    from humorhist import wikipedia
    entered = multiprocessing.get_context('fork').Event()
    proc = multiprocessing.get_context('fork').Process(target=_interrupted_request, args=(entered,))
    proc.start()
    try:
        assert entered.wait(5)
        monkeypatch.setattr(wikipedia, 'LOCK_WAIT', 0.1)
        with httpx.Client(transport=httpx.MockTransport(lambda r: pytest.fail('sent'))) as client:
            with pytest.raises(WikipediaCooldown, match='lock wait deadline'):
                request(client, 'https://en.wikipedia.org/')
    finally:
        proc.terminate()
        proc.join(5)


def test_transport_error_sets_cooldown(tmp_path):
    def fail(req):
        raise httpx.ReadError('network', request=req)
    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(httpx.ReadError):
            request(client, 'https://en.wikipedia.org/')
        with pytest.raises(WikipediaCooldown, match='deadline.*error'):
            request(client, 'https://en.wikipedia.org/')
    assert float((tmp_path / '.cache/humorhist/wikipedia-rate.state').read_text().split()[1]) > time.time() + 55


def test_403_persists_and_reset_refuses_known_deadline(monkeypatch, capsys):
    import sys

    from humorhist import wikipedia
    with httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(403, headers={'Retry-After': '7200'})
    )) as client:
        request(client, 'https://en.wikipedia.org/')
        with pytest.raises(WikipediaCooldown, match='403 blocked'):
            request(client, 'https://en.wikipedia.org/')
    monkeypatch.setattr(sys, 'argv', ['limiter', 'inspect'])
    wikipedia.main()
    assert 'reason=403' in capsys.readouterr().out
    monkeypatch.setattr(sys, 'argv', ['limiter', 'reset', '--confirm-producers-stopped-and-permission-reviewed'])
    with pytest.raises(SystemExit):
        wikipedia.main()
    assert 'deadline has not expired' in capsys.readouterr().err


def test_malformed_header_inspect_and_offline_reset(monkeypatch, capsys):
    import sys

    from humorhist import wikipedia
    with httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(429, headers={'Retry-After': 'bad'})
    )) as client:
        request(client, 'https://en.wikipedia.org/')
        with pytest.raises(WikipediaCooldown, match='indefinitely.*malformed'):
            request(client, 'https://en.wikipedia.org/')
    monkeypatch.setattr(sys, 'argv', ['limiter', 'reset'])
    with pytest.raises(SystemExit):
        wikipedia.main()
    monkeypatch.setattr(sys, 'argv', ['limiter', 'reset', '--confirm-producers-stopped-and-permission-reviewed'])
    wikipedia.main()
    assert 'State reset' in capsys.readouterr().out
    assert wikipedia._load(wikipedia._directory() / 'wikipedia-rate.state')[1] == 0


def test_harvest_redirect_retry_after_and_same_origin():
    calls = []
    def reply(req):
        calls.append(req)
        return httpx.Response(302, headers={'Location': '/other', 'Retry-After': '60'})
    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        with pytest.raises(wiki.HarvestError) as err:
            wiki.fetch_page_wikitext('x', client, sleep_seconds=0)
    assert len(calls) == 1 and err.value.local_cooldown and err.value.source_attempted


def test_harvest_follows_same_origin_redirect():
    calls = []
    def reply(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(302, headers={'Location': '/other'})
        return httpx.Response(200, json={'parse': {'wikitext': 'ok'}})
    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        assert wiki.fetch_page_wikitext('x', client, sleep_seconds=0) == 'ok'
    assert str(calls[1].url) == 'https://en.wikipedia.org/other'


def test_state_replace_fsyncs_directory(monkeypatch, tmp_path):
    import os
    import stat

    from humorhist import wikipedia
    synced = []
    original = os.fsync
    def record(fd):
        synced.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        original(fd)
    monkeypatch.setattr(wikipedia.os, 'fsync', record)
    wikipedia._save(tmp_path, 1, 2, 'deadline')
    assert synced == [False, True]


def test_reset_refuses_unexpired_429(monkeypatch):
    import sys

    from humorhist import wikipedia
    with httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(429, headers={'Retry-After': '7200'})
    )) as client:
        request(client, 'https://en.wikipedia.org/')
    monkeypatch.setattr(sys, 'argv', ['limiter', 'reset', '--confirm-producers-stopped-and-permission-reviewed'])
    with pytest.raises(SystemExit):
        wikipedia.main()
    assert wikipedia._load(wikipedia._directory() / 'wikipedia-rate.state')[1] > time.time() + 7100


def test_legacy_state_requires_review_and_preserves_deadline(tmp_path):
    state = tmp_path / '.cache/humorhist/wikipedia-rate.state'
    state.parent.mkdir(parents=True)
    deadline = time.time() + 7200
    state.write_text(f'0 {deadline}')
    from humorhist import wikipedia

    assert wikipedia._load(state)[1:] == (float('inf'), 'legacy', deadline)
    with httpx.Client(transport=httpx.MockTransport(lambda r: pytest.fail('legacy state bypassed'))) as client:
        with pytest.raises(WikipediaCooldown, match='indefinitely.*legacy'):
            request(client, 'https://en.wikipedia.org/')


def test_403_malformed_retry_after_has_unknown_reset_deadline(monkeypatch):
    import sys

    from humorhist import wikipedia
    with httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(403, headers={'Retry-After': 'malformed'})
    )) as client:
        request(client, 'https://en.wikipedia.org/')
    monkeypatch.setattr(sys, 'argv', ['limiter', 'reset', '--confirm-producers-stopped-and-permission-reviewed'])
    with pytest.raises(SystemExit):
        wikipedia.main()


def test_harvest_rejects_cross_origin_redirect():
    calls = []
    def reply(req):
        calls.append(req)
        return httpx.Response(302, headers={'Location': 'https://example.com/'})
    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        with pytest.raises(wiki.HarvestError, match='unsupported origin'):
            wiki.fetch_page_wikitext('x', client, sleep_seconds=0)
    assert len(calls) == 1
