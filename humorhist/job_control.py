"""Non-queuing, cross-process exclusion for expensive drafting jobs.

All participants must use the *same* lock path on a local filesystem that
supports flock. Keep the empty lock file permanently: unlinking or replacing it
can split the lock into two independent inodes. This module never touches the
application database or stores job arguments, results, or credentials on disk.

The lock is held for the lifetime of the job, not a time-based lease: a slow
but live job cannot accidentally overlap a replacement. Process death closes
its descriptor and releases the lock. Do not fork while holding the lock: an
inherited descriptor can keep it alive in the child. These locks are advisory;
every competing entry point must participate.
"""
from __future__ import annotations

import errno
import fcntl
import os
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import suppress
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def database_path(conn: sqlite3.Connection) -> Path:
    """Resolve the actual main database before handing work to another thread."""
    filename = next(row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main")
    if not filename:
        raise ValueError("background jobs require a file-backed main database")
    return Path(filename).resolve(strict=True)


def lock_path_for_db(conn: sqlite3.Connection) -> Path:
    """Return the shared drafting lock for the connection's actual main database.

    SQLite resolves URI names and aliases when opening the connection; use its
    reported filename, then resolve symlinks. Memory/temporary databases have no
    stable cross-process identity and are deliberately rejected. Hard-link
    aliases and replacing a database while jobs run are not supported.
    """
    path = database_path(conn)
    return path.with_name(path.name + ".draft.lock")


class JobLock:
    """A held flock. Close exactly once, preferably via a with statement."""

    def __init__(self, fd: int):
        self._fd = fd

    def close(self) -> None:
        if self._fd != -1:
            fd, self._fd = self._fd, -1
            os.close(fd)

    def __enter__(self) -> JobLock:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def try_acquire(path: str | os.PathLike[str]) -> JobLock | None:
    """Acquire immediately, or return None if busy; propagate other errors.

    The parent directory must already exist. Use one stable, trusted path for
    all processes working on the same database, including after a restart.
    """
    fd = os.open(Path(path), os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return None
        raise
    return JobLock(fd)


def start_job(path: str | os.PathLike[str], work: Callable[[], T]) -> Future[T] | None:
    """Start one background thread, or return None immediately when busy.

    The returned Future carries the result or exception. Cancellation is not
    supported once accepted. The caller must open and close any SQLite
    connection inside work, not pass a connection created by another thread.
    The daemon worker is not durable across process exit; persisted application
    data remains the responsibility of the work's transaction boundaries.
    """
    return _start_job(work, try_acquire(path), require_lock=True)


def _start_job(work: Callable[[], T], lock: JobLock | None, *, require_lock: bool) -> Future[T] | None:
    if require_lock and lock is None:
        return None
    future: Future[T] = Future()
    future.set_running_or_notify_cancel()

    def run() -> None:
        try:
            result = work()
        except BaseException as exc:
            if lock is not None:
                lock.close()
            future.set_exception(exc)
        else:
            if lock is not None:
                lock.close()
            future.set_result(result)

    try:
        threading.Thread(target=run, name="humorhist-job", daemon=True).start()
    except BaseException:
        if lock is not None:
            lock.close()
        raise
    return future


def start_db_job(
    conn: sqlite3.Connection,
    work: Callable[[sqlite3.Connection], T],
    *,
    drafting: bool = False,
    on_error: Callable[[str], None] | None = None,
) -> Future[T] | None:
    """Run work asynchronously on its own connection to the same file.

    With drafting=True, return None when another drafting job holds the shared
    cross-process lock. Errors remain on the Future; on_error receives a fixed,
    non-sensitive message in the worker thread (and must be thread-safe).
    Reporter failure never obscures the original error. A successful return is
    only in-memory acceptance: jobs are *not durable* and do not resume after
    restart. Check persisted state before manually retrying interrupted work;
    provider calls and earlier commits may already have completed.

    Work owns transaction boundaries; uncommitted changes are rolled back on
    failure and on close. Callers must not pass an in-memory or temporary DB,
    nor share their connection with the worker. Work may commit incrementally.
    """
    from humorhist import db

    path = database_path(conn)

    def run() -> T:
        worker = None
        try:
            worker = db.connect(str(path))
            return work(worker)
        except BaseException:
            if on_error is not None:
                # Reporting must not replace the work error or leak secrets.
                with suppress(Exception):
                    on_error("Background job failed. Check application state before retrying.")
            raise
        finally:
            if worker is not None:
                worker.close()

    if drafting:
        return start_job(path.with_name(path.name + ".draft.lock"), run)
    return _start_job(run, None, require_lock=False)
