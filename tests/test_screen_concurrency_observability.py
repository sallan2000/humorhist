"""Screening counts stay disjoint across competing reservations."""
import pytest

from humorhist import db
from humorhist.harvest import screen
from humorhist.llm import StubClient


@pytest.mark.parametrize('interleave', ['after_selection', 'before_claim'])
@pytest.mark.parametrize('limit', [None, 1])
def test_concurrent_reservation_counted_once(tmp_path, monkeypatch, interleave, limit):
    path = str(tmp_path / 'test.sqlite')
    conn = db.connect(path)
    db.migrate(conn)
    conn.executemany("INSERT INTO pool (id,title,status) VALUES (?,?,'new')",
                     [('a', 'a'), ('b', 'b'), ('c', 'c')])
    conn.execute('CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL, generation TEXT)')
    conn.execute("INSERT INTO screen_attempts VALUES ('c','pending','existing')")
    conn.commit()

    def reserve():
        other = db.connect(path)
        try:
            other.execute("INSERT INTO screen_attempts (pool_id,state,generation) VALUES ('a','pending','racing')")
            other.commit()
        finally:
            other.close()

    if interleave == 'after_selection':
        original = screen._select_unscored
        def select(*args, **kwargs):
            result = original(*args, **kwargs)
            reserve()
            return result
        monkeypatch.setattr(screen, '_select_unscored', select)
    else:
        original_uuid = screen.uuid.uuid4
        called = False
        def uuid4():
            nonlocal called
            if not called:
                called = True
                reserve()
            return original_uuid()
        monkeypatch.setattr(screen.uuid, 'uuid4', uuid4)

    client = StubClient([[{'n': 1, 'score': 5}]])
    result = screen.screen_pool(conn, client, limit=limit)
    assert result['blocked'] == 2
    assert result['skipped'] == 2
    if limit is None:
        assert (result['selected'], result['eligible_selected'], result['scored'], result['unscored']) == (3, 2, 1, 2)
        assert len(client.calls) == 1
    else:
        assert (result['selected'], result['eligible_selected'], result['scored'], result['unscored']) == (1, 1, 0, 1)
        assert not client.calls
    assert not conn.in_transaction
    conn.close()


@pytest.mark.parametrize('limit', [None, 1])
def test_preclaim_refusal_counts_racing_selected_reservation(tmp_path, monkeypatch, limit):
    path = str(tmp_path / 'test.sqlite')
    conn = db.connect(path)
    db.migrate(conn)
    conn.executemany("INSERT INTO pool (id,title,status) VALUES (?,?,'new')", [('a', 'a'), ('b', 'b'), ('c', 'c')])
    conn.execute('CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL, generation TEXT)')
    conn.execute("INSERT INTO screen_attempts VALUES ('c','pending','old')")
    conn.commit()
    original = screen._select_unscored

    def select(*args, **kwargs):
        rows = original(*args, **kwargs)
        other = db.connect(path)
        other.execute("INSERT INTO screen_attempts (pool_id,state,generation) VALUES ('a','pending','race')")
        other.commit()
        other.close()
        return rows

    monkeypatch.setattr(screen, '_select_unscored', select)
    monkeypatch.setattr(screen, '_preflight_client', lambda client: 'unavailable')
    result = screen.screen_pool(conn, StubClient([]), limit=limit)
    assert result['blocked'] == 2
    assert result['selected'] == (3 if limit is None else 1)
    assert result['unscored'] == (3 if limit is None else 1)
    assert result['skipped'] == (3 if limit is None else 2)
    conn.close()
