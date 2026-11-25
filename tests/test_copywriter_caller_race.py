"""Ensure input rechecks preserve a caller-owned transaction."""

import humorhist.copywriter as cw
import humorhist.db as db
from humorhist.llm import StubClient


def test_changed_direction_in_caller_transaction_is_not_overwritten(tmp_path, monkeypatch):
    conn = db.connect(str(tmp_path / 'test.sqlite'))
    db.migrate(conn)
    conn.execute("INSERT INTO pool (id, title) VALUES ('p', 'Title')")
    conn.execute("INSERT INTO drafts (id, pool_id, brief_json, angles_json, status) "
                 "VALUES ('d', 'p', '{}', '{}', 'approved')")
    conn.execute("INSERT INTO queue (draft_id, published, post_copy) VALUES ('d', 0, 'Original')")
    conn.commit()
    conn.execute("UPDATE pool SET note = 'caller work' WHERE id = 'p'")
    changed = []

    def generate(*args):
        conn.execute("UPDATE drafts SET editor_notes = 'new direction' WHERE id = 'd'")
        changed.append(True)
        return 'Stale copy'

    monkeypatch.setattr(cw, 'generate_post_copy', generate)
    assert cw.fill_post_copy(conn, StubClient([]), force=True) == 0
    assert changed == [True]
    assert conn.in_transaction
    assert conn.execute('SELECT post_copy FROM queue').fetchone()[0] == 'Original'
    conn.rollback()
    assert conn.execute('SELECT note FROM pool').fetchone()[0] is None
    assert conn.execute('SELECT editor_notes FROM drafts').fetchone()[0] is None
    conn.close()
