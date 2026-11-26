"""Drafting exclusion uses the actual SQLite database, without live services."""
import sqlite3
import sys

import pytest

import humorhist.db as db
import humorhist.drafting as drafting
import humorhist.env as env
import humorhist.harvest.screen as screen
import humorhist.harvest.seed as seed
import humorhist.harvest.wikipedia_lists as wiki
import humorhist.job_control as control
import humorhist.llm as llm
import humorhist.telegram as telegram


@pytest.fixture
def jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "load_env", lambda: None)
    from scripts import buffer_health, weekly_pipeline
    monkeypatch.setattr(llm, "screening_client", lambda **kwargs: object())

    logs = []
    for job in (weekly_pipeline, buffer_health):
        monkeypatch.setattr(job, "log", logs.append)
        monkeypatch.setattr(job, "_llm", lambda: object())
    monkeypatch.setattr(seed, "load_seed", lambda conn: {"inserted": 0})
    monkeypatch.setattr(wiki, "harvest_wikipedia_lists", lambda conn: {"inserted": 0})
    monkeypatch.setattr(screen, "screen_pool", lambda *a, **kw: {"scored": 0, "batches": 0, "failed_batches": 0})
    monkeypatch.setenv("HUMORHIST_TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setenv("HUMORHIST_TELEGRAM_BOT_TOKEN", "token")
    sent = []

    class Transport:
        def send_message(self, chat, message):
            sent.append(message)

    monkeypatch.setattr(telegram, "TelegramClient", Transport)
    return weekly_pipeline, buffer_health, logs, sent


def test_lock_path_uses_actual_database_and_rejects_memory(tmp_path):
    path = tmp_path / "main.sqlite"
    conn = sqlite3.connect(path)
    alias = tmp_path / "alias.sqlite"
    alias.symlink_to(path)
    via_alias = sqlite3.connect(alias)
    via_uri = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)
    memory = sqlite3.connect(":memory:")
    try:
        expected = path.with_name("main.sqlite.draft.lock")
        assert {control.lock_path_for_db(c) for c in (conn, via_alias, via_uri)} == {expected}
        with pytest.raises(ValueError, match="file-backed"):
            control.lock_path_for_db(memory)
    finally:
        for c in (conn, via_alias, via_uri, memory):
            c.close()


@pytest.mark.parametrize("outer_index", [0, 1])
def test_both_jobs_exclude_each_other_and_continue_other_phases(jobs, monkeypatch, tmp_path, outer_index):
    weekly, buffer, logs, sent = jobs
    path = tmp_path / "db.sqlite"
    conn = db.connect(str(path))
    db.migrate(conn)
    conn.close()
    alias = tmp_path / "alias.sqlite"
    alias.symlink_to(path)
    outer, inner = (weekly, buffer) if outer_index == 0 else (buffer, weekly)
    calls = []

    def draft(conn, *args, **kwargs):
        calls.append("outer")
        assert control.try_acquire(control.lock_path_for_db(conn)) is None
        monkeypatch.setattr(sys, "argv", ["job", "--db", str(alias)])
        assert inner.main() == 1
        return {"selected": 0, "drafted": 0, "failed": 0}

    monkeypatch.setattr(drafting, "draft_candidates", draft)
    monkeypatch.setattr(sys, "argv", ["job", "--db", str(path)])
    assert outer.main() == 0
    assert calls == ["outer"]
    assert any("draft:busy" in line for line in logs)
    assert any("HARVEST seed" in line for line in logs)
    assert any("SCREEN:" in line for line in logs)
    assert any("HEALTH:" in line for line in logs)
    assert sent and ("draft:busy" in sent[0] if inner is buffer else True)
    lock = control.try_acquire(path.with_name(path.name + ".draft.lock"))
    assert lock is not None
    lock.close()


@pytest.mark.parametrize("job_index", [0, 1])
def test_drafting_error_releases_lock_and_lock_error_is_reported(jobs, monkeypatch, tmp_path, job_index):
    job, logs = jobs[job_index], jobs[2]
    path = tmp_path / "db.sqlite"
    monkeypatch.setattr(sys, "argv", ["job", "--db", str(path)])

    def fail(*args, **kwargs):
        raise RuntimeError("sensitive")

    monkeypatch.setattr(drafting, "draft_candidates", fail)
    assert job.main() == 1
    lock = control.try_acquire(path.with_name(path.name + ".draft.lock"))
    assert lock is not None
    lock.close()
    assert "draft:RuntimeError" in logs[-1]
    assert "sensitive" not in " ".join(logs)
    logs.clear()
    monkeypatch.setattr(control, "try_acquire", fail)
    assert job.main() == 1
    assert "draft:RuntimeError" in logs[-1]
    assert any("HEALTH:" in line for line in logs) if job_index else any("NOTIFY:" in line for line in logs)
