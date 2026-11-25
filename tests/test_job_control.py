import os
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from humorhist.job_control import start_job, try_acquire


def test_busy_then_release_and_idempotent_close(tmp_path):
    path = tmp_path / "draft.lock"
    lock = try_acquire(path)
    assert lock is not None
    assert try_acquire(path) is None
    lock.close()
    lock.close()
    with try_acquire(path) as next_lock:
        assert next_lock is not None
    assert path.exists()  # Never remove the inode on release.
    assert path.read_bytes() == b""


def test_racing_threads_have_one_winner(tmp_path):
    path = tmp_path / "draft.lock"
    barrier = threading.Barrier(12)
    acquired = []
    guard = threading.Lock()

    def contender():
        barrier.wait(timeout=5)
        lock = try_acquire(path)
        with guard:
            acquired.append(lock)
        # Retain all locks until all attempts have finished.

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda _: contender(), range(12)))
    winners = [lock for lock in acquired if lock is not None]
    assert len(winners) == 1
    winners[0].close()


def test_job_is_nonblocking_and_busy_and_result(tmp_path):
    path = tmp_path / "draft.lock"
    started, finish = threading.Event(), threading.Event()

    def work():
        started.set()
        assert finish.wait(5)
        return 42

    future = start_job(path, work)
    assert future is not None and started.wait(5)
    try:
        assert not future.cancel()
        assert start_job(path, lambda: 1) is None
        assert try_acquire(path) is None
    finally:
        finish.set()
    assert future.result(timeout=5) == 42
    lock = try_acquire(path)
    assert lock is not None
    lock.close()


def test_worker_exception_and_start_failure_release(tmp_path, monkeypatch):
    path = tmp_path / "draft.lock"

    def fail():
        raise ValueError("job failed")

    future = start_job(path, fail)
    with pytest.raises(ValueError, match="job failed"):
        future.result(timeout=5)
    lock = try_acquire(path)
    assert lock is not None
    lock.close()

    def fail_start(_thread):
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread start failed"):
        start_job(path, lambda: None)
    lock = try_acquire(path)
    assert lock is not None
    lock.close()


def test_process_death_releases_lock(tmp_path):
    path = tmp_path / "draft.lock"
    code = (
        "import sys, time; from humorhist.job_control import try_acquire; "
        "lock=try_acquire(sys.argv[1]); assert lock; print('locked', flush=True); time.sleep(30)"
    )
    child = subprocess.Popen([sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        assert try_acquire(path) is None
        child.kill()
        child.wait(timeout=5)
        lock = try_acquire(path)
        assert lock is not None
        lock.close()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        child.stdout.close()


def test_separate_sqlite_connection_and_failure_preserve_state(tmp_path):
    path, database = tmp_path / "draft.lock", tmp_path / "app.sqlite"
    conn = sqlite3.connect(database)
    conn.execute("create table entries (value text)")
    conn.execute("insert into entries values ('original')")
    conn.commit()

    def work():
        # A worker owns its connection; failure rolls its transaction back.
        worker = sqlite3.connect(database)
        try:
            with worker:
                worker.execute("insert into entries values ('unfinished')")
                raise RuntimeError("abort")
        finally:
            worker.close()

    future = start_job(path, work)
    with pytest.raises(RuntimeError, match="abort"):
        future.result(timeout=5)
    assert conn.execute("select value from entries").fetchall() == [("original",)]
    conn.close()
    assert os.stat(path).st_size == 0


def test_missing_parent_is_an_error_not_busy(tmp_path):
    with pytest.raises(FileNotFoundError):
        try_acquire(tmp_path / "missing" / "draft.lock")
