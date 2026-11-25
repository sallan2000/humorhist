"""Offline coverage for the historical regen script's non-destructive behavior."""

import argparse

import pytest

from humorhist import db, drafting, llm
from scripts import regen_drafts


@pytest.mark.parametrize("has_existing", [False, True])
def test_existing_drafts_are_skipped_and_fresh_phase_runs(tmp_path, monkeypatch, has_existing):
    conn = db.connect(str(tmp_path / "test.sqlite"))
    try:
        db.migrate(conn)
        if has_existing:
            conn.execute("INSERT INTO pool (id, title, status) VALUES ('p', 'Title', 'drafted')")
            conn.execute(
                "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, editor_notes) "
                "VALUES ('d', 'p', ?, ?, 'approved', 'editor work')",
                ('{"facts": ["original"]}', '{"angles": ["original"]}'),
            )
            conn.commit()
        before = [tuple(row) for row in conn.execute("SELECT * FROM drafts")]
        logs = []
        calls = []
        monkeypatch.setattr(regen_drafts, "log", logs.append)
        monkeypatch.setattr(llm, "resilient_client", lambda **kwargs: calls.append("client") or object())
        monkeypatch.setattr(
            drafting, "draft_one", lambda *a, **k: pytest.fail("existing draft was regenerated")
        )
        monkeypatch.setattr(
            drafting, "draft_candidates",
            lambda worker, client, **kwargs: calls.append((worker, kwargs)) or {"failed": 0},
        )
        args = argparse.Namespace(fresh_count=2, min_score=8.0)

        assert regen_drafts._run(conn, args) == 0
        assert logs[0] == (
            f"REGEN: skipped {int(has_existing)} existing draft pool item(s); "
            "no regeneration occurred (replacement is not supported)"
        )
        assert calls == ["client", (conn, {"count": 2, "min_score": 8.0})]
        assert [tuple(row) for row in conn.execute("SELECT * FROM drafts")] == before
    finally:
        conn.close()
