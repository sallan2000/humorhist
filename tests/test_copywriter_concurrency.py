"""Offline optimistic-concurrency checks for copy generation."""

import pytest

import humorhist.copywriter as cw
import humorhist.db as db
from humorhist.llm import StubClient


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("mutation", [
    "UPDATE drafts SET brief_json = '{}', reviewed_at = 'new' WHERE id = 'd1'",
    "UPDATE drafts SET angles_json = '{}' WHERE id = 'd1'",
    "UPDATE drafts SET editor_line = 'new direction' WHERE id = 'd1'",
    "UPDATE drafts SET editor_notes = 'new notes' WHERE id = 'd1'",
    "UPDATE drafts SET status = 'rejected' WHERE id = 'd1'",
    "UPDATE drafts SET pool_id = 'p2' WHERE id = 'd1'",
    "UPDATE pool SET title = 'New title' WHERE id = 'p1'",
    "UPDATE pool SET year = 1900 WHERE id = 'p1'",
    "DELETE FROM pool WHERE id = 'p1'",
    "UPDATE queue SET draft_id = 'd2' WHERE draft_id = 'd1'",
])
def test_generation_skips_changed_inputs(tmp_path, monkeypatch, force, mutation):
    path = str(tmp_path / 'test.sqlite')
    conn = db.connect(path)
    db.migrate(conn)
    conn.execute("INSERT INTO pool (id, title, year) VALUES ('p1', 'Old title', 1800)")
    conn.execute("INSERT INTO pool (id, title, year) VALUES ('p2', 'Other title', 1801)")
    conn.execute("INSERT INTO drafts (id, pool_id, brief_json, angles_json, status) "
                 "VALUES ('d1', 'p1', '{\"verified_facts\": []}', '{\"angles\": []}', 'approved')")
    conn.execute("INSERT INTO drafts (id, pool_id, brief_json, angles_json, status) "
                 "VALUES ('d2', 'p2', '{}', '{}', 'approved')")
    original = 'Original copy' if force else None
    conn.execute("INSERT INTO queue (draft_id, published, post_copy, post_copy_at) VALUES ('d1', 0, ?, 'old')", (original,))
    conn.commit()
    other = db.connect(path)
    if mutation.startswith("DELETE FROM pool"):
        other.execute("PRAGMA foreign_keys = OFF")
    changed = []

    def generate(*args):
        assert not conn.in_transaction
        other.execute(mutation)
        other.commit()
        changed.append(True)
        return 'Stale generated copy'

    monkeypatch.setattr(cw, 'generate_post_copy', generate)
    assert cw.fill_post_copy(conn, StubClient([]), draft_id='d1', force=force) == 0
    assert changed == [True]
    assert tuple(conn.execute('SELECT post_copy, post_copy_at FROM queue').fetchone()) == (original, 'old')
    other.close()
    conn.close()


def test_generation_skips_when_missing_pool_appears(tmp_path, monkeypatch):
    path = str(tmp_path / 'test.sqlite')
    conn = db.connect(path)
    db.migrate(conn)
    conn.execute("PRAGMA foreign_keys = OFF")  # Model a legacy orphaned reference.
    conn.execute("INSERT INTO drafts (id, pool_id, brief_json, angles_json, status) "
                 "VALUES ('d1', 'missing', '{}', '{}', 'approved')")
    conn.execute("INSERT INTO queue (draft_id, published) VALUES ('d1', 0)")
    conn.commit()
    other = db.connect(path)

    def generate(*args):
        other.execute("INSERT INTO pool (id, title) VALUES ('missing', 'Now exists')")
        other.commit()
        return 'Stale generated copy'

    monkeypatch.setattr(cw, 'generate_post_copy', generate)
    assert cw.fill_post_copy(conn, StubClient([])) == 0
    assert conn.execute('SELECT post_copy FROM queue').fetchone()[0] is None
    other.close()
    conn.close()
