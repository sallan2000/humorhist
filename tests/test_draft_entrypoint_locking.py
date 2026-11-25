"""Offline contention and cleanup tests for every synchronous draft entry point."""
import argparse
import importlib
import sys

import pytest

from humorhist import buffer, cli, db, drafting, job_control, llm


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "main.sqlite"
    conn = db.connect(str(path))
    db.migrate(conn)
    yield path, conn
    conn.close()


def test_cli_busy_through_symlink_and_release(database, tmp_path, monkeypatch, capsys):
    path, conn = database
    alias = tmp_path / "alias.sqlite"
    alias.symlink_to(path)
    args = argparse.Namespace(db=str(alias), count=2, min_score=7)
    monkeypatch.setattr(llm, "default_client", lambda: object())
    called = []
    monkeypatch.setattr(drafting, "draft_candidates", lambda *a, **k: called.append(True) or {"failed": 0})
    with job_control.try_acquire(job_control.lock_path_for_db(conn)):
        assert cli.cmd_draft(args) == 1
        assert "busy" in capsys.readouterr().out.lower()
        assert not called
    assert cli.cmd_draft(args) == 0
    assert called == [True]


def test_buffer_busy_and_exception_release(database, monkeypatch):
    _, conn = database
    called = []
    monkeypatch.setattr(drafting, "draft_candidates", lambda *a, **k: called.append(True))
    with job_control.try_acquire(job_control.lock_path_for_db(conn)):
        result = buffer.run_buffer_check(conn, client=object())
        assert "busy" in result["draft_error"]
        assert not result["draft_started"]
        assert not called
    def fail(*args, **kwargs):
        assert job_control.try_acquire(job_control.lock_path_for_db(conn)) is None
        raise RuntimeError("draft failed")
    monkeypatch.setattr(drafting, "draft_candidates", fail)
    assert buffer.run_buffer_check(conn, client=object())["draft_error"] == "draft failed"
    with job_control.try_acquire(job_control.lock_path_for_db(conn)) as lock:
        assert lock is not None


@pytest.mark.parametrize("name", ["scripts.run_drafts", "scripts.regen_drafts"])
def test_script_busy_success_and_exception_cleanup(name, database, monkeypatch):
    path, conn = database
    module = importlib.import_module(name)
    monkeypatch.setattr(sys, "argv", [name, "--db", str(path)])
    logs = []
    monkeypatch.setattr(module, "log", logs.append)
    calls = []
    def run(worker, args):
        calls.append(True)
        assert job_control.try_acquire(job_control.lock_path_for_db(worker)) is None
        return 0
    monkeypatch.setattr(module, "_run", run)
    with job_control.try_acquire(job_control.lock_path_for_db(conn)):
        assert module.main() == 1
    assert "BUSY" in logs[-1]
    assert not calls
    assert module.main() == 0
    def fail(worker, args):
        raise RuntimeError("failure")
    monkeypatch.setattr(module, "_run", fail)
    with pytest.raises(RuntimeError, match="failure"):
        module.main()
    with job_control.try_acquire(job_control.lock_path_for_db(conn)) as lock:
        assert lock is not None


def test_cli_buffer_busy_returns_nonzero(database, monkeypatch, capsys):
    path, conn = database
    monkeypatch.setattr(llm, "default_client", lambda: object())
    args = argparse.Namespace(db=str(path), auto_draft=True, notify=False, chat_id=None)
    with job_control.try_acquire(job_control.lock_path_for_db(conn)):
        assert cli.cmd_buffer(args) == 1
    output = capsys.readouterr().out.lower()
    assert "busy" in output
    assert "auto-drafting" not in output


@pytest.mark.parametrize("name", ["scripts.run_drafts", "scripts.regen_drafts"])
def test_script_real_work_holds_lock_and_reports_failures(name, database, monkeypatch):
    path, conn = database
    module = importlib.import_module(name)
    monkeypatch.setattr(sys, "argv", [name, "--db", str(path)])
    monkeypatch.setattr(module, "log", lambda message: None)
    monkeypatch.setattr(llm, "resilient_client", lambda **kwargs: object())

    def check_lock(worker):
        assert job_control.try_acquire(job_control.lock_path_for_db(worker)) is None

    if name.endswith("run_drafts"):
        def select(worker, **kwargs):
            check_lock(worker)
            return [{"title": "one"}, {"title": "two"}]
        calls = []
        def draft_one(worker, client, row):
            check_lock(worker)
            calls.append(row)
            if len(calls) == 2:
                raise RuntimeError("one failed")
            return "id"
        monkeypatch.setattr(drafting, "select_candidates", select)
        monkeypatch.setattr(drafting, "draft_one", draft_one)
    else:
        def candidates(worker, client, **kwargs):
            check_lock(worker)
            return {"drafted": 1, "failed": 1}
        monkeypatch.setattr(drafting, "draft_candidates", candidates)
    assert module.main() == 1
    lock = job_control.try_acquire(job_control.lock_path_for_db(conn))
    assert lock is not None
    lock.close()
