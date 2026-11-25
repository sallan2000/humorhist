"""Offline integration tests for responsive Telegram jobs."""

import threading

import pytest

from humorhist import db, job_control
from humorhist import telegram as tg


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(str(tmp_path / "test.sqlite"))
    db.migrate(connection)
    yield connection
    connection.close()


def update(text, uid):
    return {"update_id": uid, "message": {"text": text, "chat": {"id": 123, "type": "private"}, "from": {"id": 123}}}


def test_polling_responds_while_harvest_waits(conn, monkeypatch):
    started, release = threading.Event(), threading.Event()
    futures = []
    original = tg._telegram_job

    def capture(*args, **kwargs):
        future = original(*args, **kwargs)
        futures.append(future)
        return future

    def slow(worker):
        started.set()
        assert release.wait(5)
        return 0

    monkeypatch.setattr(tg, "_telegram_job", capture)
    monkeypatch.setattr("humorhist.harvest.seed.load_seed", slow)
    monkeypatch.setattr("humorhist.harvest.wikipedia_lists.harvest_wikipedia_lists", lambda c: 0)

    class Transport(tg.StubTelegram):
        calls = 0

        def get_updates(self, offset=0, timeout=0):
            self.calls += 1
            if self.calls == 1:
                return [update("/harvest", 1)]
            if self.calls == 2:
                assert started.wait(5)
                return [update("/status", 2)]
            assert any("Reviewed" in m["text"] or "pending" in m["text"].lower() for m in self.sent)
            assert not futures[0].done()
            release.set()
            return []

    stub = Transport()
    try:
        tg.run_review_bot(conn, stub, "123", max_iterations=3)
    finally:
        release.set()
    futures[0].result(5)
    assert sum("Harvest done" in m["text"] for m in stub.sent) == 1


def test_failure_is_sanitized_connection_closed_and_job_retriable(conn, monkeypatch):
    stub = tg.StubTelegram()
    workers = []
    original_connect = db.connect

    class TrackedConnection:
        closed = False

        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def close(self):
            self.inner.close()
            self.closed = True

    def connect(*args, **kwargs):
        worker = TrackedConnection(original_connect(*args, **kwargs))
        workers.append(worker)
        return worker

    monkeypatch.setattr(db, "connect", connect)

    def fail(worker):
        raise ValueError("SECRET-bot-token")

    monkeypatch.setattr("humorhist.harvest.seed.load_seed", fail)
    future = tg.telegram_harvest(conn, stub, "123", wikipedia=False)
    with pytest.raises(ValueError):
        future.result(5)
    assert sum("job failed" in m["text"] for m in stub.sent) == 1
    assert "SECRET-bot-token" not in str(stub.sent)
    assert workers[0].closed
    monkeypatch.setattr("humorhist.harvest.seed.load_seed", lambda worker: 1)
    tg.telegram_harvest(conn, stub, "123", wikipedia=False).result(5)


def test_busy_scheduled_lock_blocks_draft_and_buffer(conn, monkeypatch):
    monkeypatch.setattr(tg, "_get_llm", lambda *a, **kw: object())
    stub = tg.StubTelegram()
    lock = job_control.try_acquire(job_control.lock_path_for_db(conn))
    assert lock is not None
    try:
        assert tg.telegram_draft(conn, stub, "123") is None
        assert tg.telegram_buffer(conn, stub, "123") is None
        assert sum("already running" in m["text"] for m in stub.sent) == 2
    finally:
        lock.close()


def test_harvest_duplicate_busy(conn, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow(worker):
        started.set()
        assert release.wait(5)
        return 0

    monkeypatch.setattr("humorhist.harvest.seed.load_seed", slow)
    stub = tg.StubTelegram()
    future = tg.telegram_harvest(conn, stub, "123", wikipedia=False)
    try:
        assert started.wait(5)
        assert tg.telegram_harvest(conn, stub, "123", wikipedia=False) is None
    finally:
        release.set()
    future.result(5)
    assert sum("already running" in m["text"] for m in stub.sent) == 1
    assert sum("Harvest done" in m["text"] for m in stub.sent) == 1


def test_external_draft_still_triggers_one_nudge(conn):
    class Transport(tg.StubTelegram):
        calls = 0

        def get_updates(self, offset=0, timeout=0):
            self.calls += 1
            if self.calls == 1:
                conn.execute("INSERT INTO pool (id,title,status) VALUES ('p','External','drafted')")
                conn.execute("INSERT INTO drafts (id,pool_id,status) VALUES ('d','p','pending')")
                conn.commit()
            return []

    stub = Transport()
    tg.run_review_bot(conn, stub, "123", max_iterations=3)
    assert sum("🆕 1 new draft(s)" in m["text"] for m in stub.sent) == 1
