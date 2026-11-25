"""Background jobs use isolated connections and the scheduled drafting lock."""
import sqlite3
import threading

import pytest

from humorhist import db, job_control


def test_draft_job_nonblocking_connection_and_shared_lock(tmp_path):
    path = tmp_path / "app.sqlite"
    conn = db.connect(str(path))
    conn.execute("create table items (value text)")
    conn.commit()
    started, release = threading.Event(), threading.Event()

    def work(worker):
        assert worker is not conn
        assert worker.execute("pragma foreign_keys").fetchone()[0] == 1
        started.set()
        assert release.wait(5)
        worker.execute("insert into items values ('done')")
        worker.commit()
        return 7

    future = job_control.start_db_job(conn, work, drafting=True)
    try:
        assert started.wait(5)
        assert job_control.start_db_job(conn, work, drafting=True) is None
        assert job_control.try_acquire(job_control.lock_path_for_db(conn)) is None
        # Non-drafting jobs are independent of the drafting lock.
        assert job_control.start_db_job(conn, lambda c: c.execute("select 1").fetchone()[0]).result(5) == 1
    finally:
        release.set()
    assert future.result(5) == 7
    assert conn.execute("select value from items").fetchone()[0] == "done"
    retry = job_control.start_db_job(conn, lambda c: 'retried', drafting=True)
    assert retry.result(5) == 'retried'
    conn.close()


def test_failure_rolls_back_reports_safely_and_can_retry(tmp_path):
    conn = db.connect(str(tmp_path / "app.sqlite"))
    conn.execute("create table items (value text)")
    conn.commit()
    messages = []

    def fail(worker):
        worker.execute("insert into items values ('unfinished')")
        raise ValueError("private credential")

    def report(message):
        messages.append(message)
        raise RuntimeError("transport failure")

    future = job_control.start_db_job(conn, fail, drafting=True, on_error=report)
    with pytest.raises(ValueError, match="private credential"):
        future.result(5)
    assert len(messages) == 1 and "private credential" not in messages[0]
    assert conn.execute("select count(*) from items").fetchone()[0] == 0
    assert job_control.start_db_job(conn, lambda c: 2, drafting=True).result(5) == 2
    conn.close()


def test_memory_database_rejected():
    conn = sqlite3.connect(":memory:")
    with pytest.raises(ValueError, match="file-backed"):
        job_control.start_db_job(conn, lambda c: None)
    conn.close()


def test_connection_open_failure_reported_and_lock_released(tmp_path, monkeypatch):
    conn = db.connect(str(tmp_path / 'app.sqlite'))
    messages = []

    def fail(_):
        raise RuntimeError('sensitive')

    monkeypatch.setattr(db, 'connect', fail)
    future = job_control.start_db_job(conn, lambda c: None, drafting=True, on_error=messages.append)
    with pytest.raises(RuntimeError, match='sensitive'):
        future.result(5)
    assert messages and 'sensitive' not in messages[0]
    lock = job_control.try_acquire(job_control.lock_path_for_db(conn))
    assert lock is not None
    lock.close()
    conn.close()
