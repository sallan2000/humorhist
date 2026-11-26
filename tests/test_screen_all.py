"""Offline tests for scheduled screening observability."""
import sys

import pytest

import humorhist.db as db
import humorhist.harvest.screen as screen
import humorhist.llm as llm
from scripts import screen_all


@pytest.mark.parametrize('category,status,label', [
    ('http_error', 503, 'http_error_http_503'),
    ('timeout', None, 'timeout'),
    ('invalid_response', None, 'invalid_response'),
])
def test_scheduled_failure_logs_reason_and_stops(tmp_path, monkeypatch, capsys, category, status, label):
    path = str(tmp_path / 'test.sqlite')
    conn = db.connect(path)
    db.migrate(conn)
    conn.execute("INSERT INTO pool (id,title,status) VALUES ('pool000','x','new')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(screen_all, 'LOG', tmp_path / 'screen.log')
    monkeypatch.setattr(sys, 'argv', ['screen_all', '--db', path])
    monkeypatch.setattr(llm, 'screening_client', lambda **kw: object())
    calls = []
    def fake_screen(*args, **kw):
        calls.append(1)
        return {'scored': 0, 'failed_batches': 1,
                'failures': [{'category': category, 'status': status, 'error': 'secret-provider'}]}
    monkeypatch.setattr(screen, 'screen_pool', fake_screen)
    assert screen_all.main() == 1
    out = capsys.readouterr().out
    assert len(calls) == 1 and label in out and 'failed_batches=1' in out
    assert 'secret-provider' not in out


def test_scheduled_unavailable_redacts_exception(tmp_path, monkeypatch, capsys):
    path = str(tmp_path / 'test.sqlite')
    conn = db.connect(path)
    db.migrate(conn)
    conn.execute("INSERT INTO pool (id,title,status) VALUES ('pool000','x','new')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(screen_all, 'LOG', tmp_path / 'screen.log')
    monkeypatch.setattr(sys, 'argv', ['screen_all', '--db', path])
    def unavailable(**kw):
        raise llm.LLMUnavailable('secret-provider')
    monkeypatch.setattr(llm, 'resilient_client', unavailable)
    assert screen_all.main() == 1
    assert 'secret-provider' not in capsys.readouterr().out


@pytest.mark.parametrize('available,expected', [(False, 1), (True, 0)])
def test_zero_limit_validates_credentials_without_requests(tmp_path, monkeypatch, available, expected):
    path = str(tmp_path / 'test.sqlite')
    monkeypatch.setattr(screen_all, 'LOG', tmp_path / 'screen.log')
    monkeypatch.setattr(sys, 'argv', ['screen_all', '--db', path, '--limit', '0'])
    calls = []
    def client(**kwargs):
        calls.append(1)
        if not available:
            raise llm.LLMUnavailable('no credential')
        return object()
    def no_screen(*args, **kwargs):
        pytest.fail('zero limit must not screen or make requests')
    monkeypatch.setattr(llm, 'screening_client', client)
    monkeypatch.setattr(screen, 'screen_pool', no_screen)
    assert screen_all.main() == expected
    assert calls == [1]


def test_zero_limit_without_actual_credentials_fails(tmp_path, monkeypatch):
    monkeypatch.delenv('HUMORHIST_LLM_API_KEY', raising=False)
    monkeypatch.delenv('HUMORHIST_LLM_API_KEY_FILE', raising=False)
    monkeypatch.setattr(screen_all, 'LOG', tmp_path / 'screen.log')
    monkeypatch.setattr(sys, 'argv', ['screen_all', '--db', str(tmp_path / 'test.sqlite'), '--limit', '0'])
    monkeypatch.setattr(screen, 'screen_pool', lambda *a, **kw: pytest.fail('unexpected screening'))
    assert screen_all.main() == 1
