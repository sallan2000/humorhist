"""Offline end-to-end checks for screening reports and reservation scope."""
import logging
import sys

import httpx
import pytest
import respx

import humorhist.db as db
import humorhist.drafting as drafting
import humorhist.harvest.screen as screen
import humorhist.llm as llm
import humorhist.telegram as telegram
from scripts import screen_all, weekly_pipeline


def setup_db(tmp_path):
    path = str(tmp_path / 'isolated.sqlite')
    conn = db.connect(path)
    db.migrate(conn)
    conn.executemany("INSERT INTO pool (id,title,status) VALUES (?,?,'new')", [('a', 'a'), ('b', 'b')])
    conn.commit()
    screen.screen_pool(conn, llm.StubClient([]), limit=0)
    return path, conn


@pytest.mark.parametrize('limit,fail,expected', [(1, False, 0), (None, False, 1), (1, True, 1)])
def test_screen_all_selection_scope(tmp_path, monkeypatch, limit, fail, expected):
    path, conn = setup_db(tmp_path)
    conn.execute("INSERT INTO screen_attempts (pool_id,state,generation) VALUES ('a','failed','old')")
    conn.commit()
    conn.close()
    fake = llm.StubClient([llm.LLMCallError('secret') if fail else [{'n': 1, 'score': 5}]])
    monkeypatch.setattr(llm, 'screening_client', lambda **kw: fake)
    monkeypatch.setattr(screen_all, 'LOG', tmp_path / 'log')
    monkeypatch.setattr(sys, 'argv', ['screen_all', '--db', path] + ([] if limit is None else ['--limit', str(limit)]))
    assert screen_all.main() == expected
    log = (tmp_path / 'log').read_text()
    assert 'blocked=' in log and 'secret' not in log
    conn = db.connect(path)
    row = conn.execute("SELECT run_id,batch_id FROM screen_batches WHERE state='pending'").fetchone()
    assert 'run_id=' + row['run_id'] in log and 'batch_id=' + row['batch_id'] in log
    conn.close()


def test_screen_all_backfills_after_lost_claim_without_resubmission(tmp_path, monkeypatch):
    path, conn = setup_db(tmp_path)
    conn.executemany("INSERT INTO pool (id,title,status) VALUES (?,?,'new')", [('c', 'c'), ('d', 'd')])
    conn.execute("INSERT INTO screen_attempts (pool_id,state,generation) VALUES ('d','failed','old')")
    conn.commit()
    conn.close()
    original = screen._select_unscored
    selections = []

    def select(conn, limit, retry_failed=False):
        rows = original(conn, limit, retry_failed)
        selections.append(([row['id'] for row in rows], retry_failed))
        if len(selections) == 1:
            other = db.connect(path)
            other.execute("INSERT INTO screen_attempts (pool_id,state,generation) VALUES ('a','pending','race')")
            other.commit()
            other.close()
        return rows

    client = llm.StubClient([[{'n': 1, 'score': 5}], [{'n': 1, 'score': 6}]])
    monkeypatch.setattr(screen, '_select_unscored', select)
    monkeypatch.setattr(llm, 'screening_client', lambda **kw: client)
    monkeypatch.setattr(screen_all, 'LOG', tmp_path / 'log')
    monkeypatch.setattr(sys, 'argv', ['screen_all', '--db', path, '--limit', '2', '--batch-size', '1'])
    assert screen_all.main() == 1
    assert selections == [(['a', 'b'], False), (['c'], False)]
    assert len(client.calls) == 2
    assert ' b' in client.calls[0]['user'] and ' c' in client.calls[1]['user']
    output = (tmp_path / 'log').read_text()
    assert 'selection_incomplete=True eligible_selected=2 scored=1' in output
    assert 'scored_total=2' in output
    conn = db.connect(path)
    assert [(row['id'], row['funny_score']) for row in conn.execute(
        'SELECT id, funny_score FROM pool ORDER BY id')] == [('a', None), ('b', 5), ('c', 6), ('d', None)]
    assert [(row['pool_id'], row['state']) for row in conn.execute(
        'SELECT pool_id, state FROM screen_attempts ORDER BY pool_id')] == [('a', 'pending'), ('d', 'failed')]
    conn.close()


@pytest.mark.parametrize('entry', ['screen_all', 'weekly', 'telegram'])
def test_uncertain_release_warning_and_durable_ids(tmp_path, monkeypatch, caplog, entry):
    path, conn = setup_db(tmp_path)
    conn.execute("CREATE TRIGGER no_release BEFORE DELETE ON screen_attempts "
                 "BEGIN SELECT RAISE(ABORT, 'secret-trigger'); END")
    conn.commit()
    client = llm.ScreeningClient(api_key='secret-key', model='exact')
    monkeypatch.setattr(llm, 'screening_client', lambda **kw: client)
    monkeypatch.setattr(drafting, 'draft_candidates', lambda *a, **k: {'selected': 0, 'drafted': 0, 'failed': 0})
    monkeypatch.setattr(weekly_pipeline, '_llm', lambda: object())
    monkeypatch.setattr(telegram, '_telegram_job', lambda c, t, chat, kind, progress, work: work(c))
    transport = telegram.StubTelegram()
    lines = []
    monkeypatch.setattr(weekly_pipeline, 'log', lines.append)
    monkeypatch.setattr(screen_all, 'LOG', tmp_path / 'log')
    with caplog.at_level(logging.INFO, logger='humorhist.telegram'), respx.mock(assert_all_called=False) as mock:
        mock.get('https://inference-api.nousresearch.com/v1/models').mock(side_effect=[
            httpx.Response(200, json={'data': [{'id': 'exact', 'pricing': {'prompt': '0', 'completion': '0'}}]}),
            httpx.Response(503),
        ])
        post = mock.post('https://inference-api.nousresearch.com/v1/chat/completions').mock()
        if entry == 'screen_all':
            monkeypatch.setattr(sys, 'argv', ['screen_all', '--db', path, '--limit', '1'])
            assert screen_all.main() == 1
            output = (tmp_path / 'log').read_text()
        elif entry == 'weekly':
            monkeypatch.setattr(sys, 'argv', ['weekly', '--db', path, '--no-harvest'])
            assert weekly_pipeline.main() == 1
            output = '\n'.join(lines)
        else:
            telegram.telegram_screen(conn, transport, 'chat', limit=1)
            output = caplog.text
            assert 'reservations may remain pending' in transport.sent[-1]['text']
            assert 'Screening incomplete' in transport.sent[-1]['text']
        assert post.call_count == 0
    row = conn.execute("SELECT run_id,batch_id FROM screen_batches WHERE state='pending'").fetchone()
    assert row is not None
    assert 'run_id=' + row['run_id'] in output and 'batch_id=' + row['batch_id'] in output
    assert 'release_uncertain=True' in output
    if entry != 'telegram':
        assert 'reservations may remain pending' in output
    assert 'secret' not in output
    conn.close()


@pytest.mark.parametrize('entry', [screen_all, weekly_pipeline])
def test_scheduled_ids_reject_untrusted_text(entry):
    text = entry._screen_ids({'run_id': 'secret\nother', 'batch_ids': ['a' * 32, 'secret', 1]})
    assert text == 'batch_id=' + 'a' * 32
