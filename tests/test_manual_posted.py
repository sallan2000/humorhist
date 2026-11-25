"""Manual publication is a local, reversible queue state, not a publisher."""
import sqlite3
from datetime import UTC, datetime

import pytest

from humorhist import db, review


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "test.sqlite"))
    db.migrate(c)
    c.execute("INSERT INTO pool (id, title, status) VALUES ('p', 'Example', 'drafted')")
    c.execute("INSERT INTO drafts (id, pool_id, status) VALUES ('d', 'p', 'approved')")
    c.execute("INSERT INTO queue (draft_id, post_copy, image_path, source_link) VALUES ('d', 'copy', 'image', 'link')")
    c.commit()
    yield c
    c.close()


def test_mark_and_undo_preserve_artifacts_and_export(conn):
    before = dict(conn.execute("SELECT * FROM queue").fetchone())
    timestamp = review.mark_posted(conn, 'd')
    assert datetime.fromisoformat(timestamp).utcoffset() == UTC.utcoffset(None)
    assert review.mark_posted(conn, 'd') == timestamp
    assert review.queued_drafts(conn) == []
    assert review.stuck_captures(conn) == []
    assert review.enqueue_approved(conn) == 0
    after = dict(conn.execute("SELECT * FROM queue").fetchone())
    assert after == {**before, 'published': 1, 'published_at': timestamp}
    assert review.undo_posted(conn, 'd', expected_published_at=timestamp) is True
    assert review.undo_posted(conn, 'd', expected_published_at=timestamp) is False
    assert dict(conn.execute("SELECT * FROM queue").fetchone()) == before
    assert review.queued_drafts(conn)[0]['post_copy'] == 'copy'


@pytest.mark.parametrize('operation', [review.mark_posted, review.undo_posted])
@pytest.mark.parametrize('state', ['missing', 'not-approved', 'not-queued', 'duplicate', 'invalid'])
def test_reject_bad_state(conn, operation, state):
    draft_id = 'd'
    if state == 'missing':
        draft_id = 'missing'
    elif state == 'not-approved':
        conn.execute("UPDATE drafts SET status='pending'")
    elif state == 'not-queued':
        conn.execute("DELETE FROM queue")
    elif state == 'duplicate':
        conn.execute("INSERT INTO queue (draft_id) VALUES ('d')")
    else:
        conn.execute("UPDATE queue SET published=2")
    conn.commit()
    before = [tuple(r) for r in conn.execute('SELECT * FROM queue')]
    with pytest.raises(ValueError):
        operation(conn, draft_id)
    assert [tuple(r) for r in conn.execute('SELECT * FROM queue')] == before


def test_legacy_publication_is_not_claimed_or_undone(conn):
    conn.execute('UPDATE queue SET published=1')
    conn.commit()
    for operation in (review.mark_posted, review.undo_posted):
        with pytest.raises(ValueError, match='timestamp'):
            operation(conn, 'd')
    assert conn.execute('SELECT published FROM queue').fetchone()[0] == 1


def test_stale_undo_rejected(conn):
    timestamp = review.mark_posted(conn, 'd')
    with pytest.raises(ValueError, match='changed'):
        review.undo_posted(conn, 'd', expected_published_at=timestamp + 'stale')
    assert conn.execute('SELECT published FROM queue').fetchone()[0] == 1


@pytest.mark.parametrize('operation', ['remove', 'reject', 'reopen'])
def test_published_row_cannot_be_removed_by_review_actions(conn, operation):
    timestamp = review.mark_posted(conn, 'd')
    with pytest.raises(ValueError, match='undo publication'):
        if operation == 'remove':
            review.remove_from_queue(conn, 'd')
        elif operation == 'reject':
            review.apply_review(conn, 'd', decision='reject')
        else:
            review.reopen_draft(conn, 'd')
    assert conn.execute('SELECT status FROM drafts').fetchone()[0] == 'approved'
    assert conn.execute('SELECT published_at FROM queue').fetchone()[0] == timestamp


@pytest.mark.parametrize('undo', [False, True])
def test_ignored_update_does_not_report_success(conn, undo):
    if undo:
        review.mark_posted(conn, 'd')
    conn.execute("CREATE TRIGGER ignore_publication BEFORE UPDATE OF published ON queue "
                 "BEGIN SELECT RAISE(IGNORE); END")
    with pytest.raises(ValueError, match='not recorded'):
        (review.undo_posted if undo else review.mark_posted)(conn, 'd')
    assert conn.execute('SELECT published FROM queue').fetchone()[0] == int(undo)


@pytest.mark.parametrize('undo', [False, True])
def test_trigger_failure_rolls_back_and_caller_transaction_survives(conn, undo):
    if undo:
        review.mark_posted(conn, 'd')
    before = tuple(conn.execute('SELECT * FROM queue').fetchone())
    conn.execute("CREATE TRIGGER fail_publication BEFORE UPDATE OF published ON queue BEGIN SELECT RAISE(ABORT, 'failed'); END")
    conn.execute("UPDATE pool SET title='uncommitted'")
    with pytest.raises(sqlite3.IntegrityError, match='failed'):
        (review.undo_posted if undo else review.mark_posted)(conn, 'd')
    assert conn.in_transaction
    assert tuple(conn.execute('SELECT * FROM queue').fetchone()) == before
    assert conn.execute('SELECT title FROM pool').fetchone()[0] == 'uncommitted'
    conn.rollback()
    assert conn.execute('SELECT title FROM pool').fetchone()[0] == 'Example'


@pytest.mark.parametrize('undo', [False, True])
def test_success_respects_caller_transaction(conn, undo):
    if undo:
        review.mark_posted(conn, 'd')
    before = tuple(conn.execute('SELECT * FROM queue').fetchone())
    conn.execute('BEGIN')
    (review.undo_posted if undo else review.mark_posted)(conn, 'd')
    assert conn.in_transaction
    conn.rollback()
    assert tuple(conn.execute('SELECT * FROM queue').fetchone()) == before


def test_migration_adds_nullable_timestamp_without_changing_legacy_rows(tmp_path):
    c = db.connect(str(tmp_path / 'legacy.sqlite'))
    c.executescript(db._SCHEMA)
    c.execute("INSERT INTO pool (id,title) VALUES ('p','Example')")
    c.execute("INSERT INTO drafts (id,pool_id,status) VALUES ('d','p','approved')")
    c.execute("INSERT INTO queue (draft_id,published) VALUES ('d',1)")
    c.commit()
    db.migrate(c)
    db.migrate(c)
    row = c.execute('SELECT draft_id,published,published_at FROM queue').fetchone()
    assert tuple(row) == ('d', 1, None)
    c.close()
