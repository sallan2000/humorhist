"""Offline regression coverage for background-job reporting and admission."""
import threading

import pytest

from humorhist import db, job_control
from humorhist import telegram as tg


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(str(tmp_path / "jobs.sqlite"))
    db.migrate(connection)
    yield connection
    connection.close()


@pytest.mark.parametrize("failed,scored", [(1, 1), (0, 1), (0, 2)])
def test_screen_reports_batch_and_incomplete_counts(conn, monkeypatch, failed, scored):
    conn.executemany("INSERT INTO pool (id,title,status) VALUES (?, ?, 'new')", [(str(n), str(n)) for n in range(3)])
    conn.commit()
    monkeypatch.setattr(tg, "_get_llm", lambda *a: object())
    monkeypatch.setattr("humorhist.harvest.screen.screen_pool", lambda *a, **kw: {
        "scored": scored, "batches": 2, "failed_batches": failed,
    })
    stub = tg.StubTelegram()
    tg.telegram_screen(conn, stub, "123", limit=2).result(5)
    text = stub.sent[-1]["text"]
    assert f"scored={scored}/2" in text
    assert f"failed_batches={failed}" in text and "batches=2" in text
    assert f"unscored={2 - scored}" in text
    assert ("incomplete" in text) == bool(failed or scored < 2)


def test_failed_ack_never_starts_work_and_releases_admission(conn, monkeypatch):
    futures = []
    original = job_control.start_db_job

    def capture(*args, **kwargs):
        future = original(*args, **kwargs)
        futures.append(future)
        return future

    class BadAck(tg.StubTelegram):
        def send_message(self, *args, **kwargs):
            raise RuntimeError("ack failed")

    monkeypatch.setattr(job_control, "start_db_job", capture)
    ran = threading.Event()
    with pytest.raises(RuntimeError, match="ack failed"):
        tg._telegram_job(conn, BadAck(), "123", "draft", "progress", lambda w: ran.set(), drafting=True)
    finished = threading.Event()
    futures[0].add_done_callback(lambda _: finished.set())
    assert finished.wait(5)
    assert not ran.is_set()
    with job_control.try_acquire(job_control.lock_path_for_db(conn)) as lock:
        assert lock is not None
    tg._telegram_job(conn, tg.StubTelegram(), "123", "draft", "progress", lambda w: ran.set(), drafting=True).result(5)
    assert ran.is_set()


@pytest.mark.parametrize("drafted,failed", [(1, 2), (0, 2)])
def test_buffer_partial_counts_and_announcements(conn, monkeypatch, drafted, failed):
    monkeypatch.setattr(tg, "_get_llm", lambda *a: object())
    monkeypatch.setattr("humorhist.drafting.draft_candidates", lambda *a, **kw: {
        "drafted": drafted, "failed": failed, "draft_ids": ["d1"] if drafted else [],
    })
    stub = tg.StubTelegram()
    tg.telegram_buffer(conn, stub, "123").result(5)
    text = stub.sent[-1]["text"]
    assert f"drafted={drafted}" in text and f"failed={failed}" in text
    assert "no candidates drafted" not in text
    path = str(job_control.database_path(conn))
    with tg._jobs_lock:
        announced = tg._jobs_announced.pop(path, set())
    assert announced == ({"d1"} if drafted else set())


def test_busy_response_never_holds_jobs_mutex(conn):
    key = (str(job_control.database_path(conn)), "harvest")
    with tg._jobs_lock:
        tg._jobs_active.add(key)

    class CheckLock(tg.StubTelegram):
        def send_message(self, *args, **kwargs):
            assert tg._jobs_lock.acquire(blocking=False)
            tg._jobs_lock.release()
            return super().send_message(*args, **kwargs)

    try:
        assert tg._telegram_job(conn, CheckLock(), "123", "harvest", "progress", lambda w: None) is None
    finally:
        with tg._jobs_lock:
            tg._jobs_active.discard(key)


def test_expired_cancel_never_holds_intent_mutex(conn):
    class CheckLock(tg.StubTelegram):
        def answer_callback_query(self, *args, **kwargs):
            acquired = threading.Event()

            def check():
                with tg._intent_lock:
                    acquired.set()

            thread = threading.Thread(target=check)
            thread.start()
            try:
                assert acquired.wait(2)
            finally:
                thread.join(2)
            return super().answer_callback_query(*args, **kwargs)

    tg.handle_callback(conn, CheckLock(), "123", {"callback_query": {"id": "x", "data": "cancel:d1:missing"}})


def test_screen_message_redacts_and_counts_reasons(conn, monkeypatch):
    conn.execute("INSERT INTO pool (id,title,status) VALUES ('p','p','new')")
    conn.commit()
    monkeypatch.setattr(tg, "_get_llm", lambda *a: object())
    monkeypatch.setattr("humorhist.harvest.screen.screen_pool", lambda *a, **kw: {
        "scored": 0, "batches": 1, "failed_batches": 1, "skipped": 0,
        "failures": [{"category": "http_error", "status": 403, "error": "secret"},
                     {"category": "secret-token", "status": "secret"}],
    })
    stub = tg.StubTelegram()
    tg.telegram_screen(conn, stub, "123").result(5)
    text = stub.sent[-1]["text"]
    assert "http_error_http_403=1" in text and "unknown=1" in text
    assert "secret" not in text


def test_limited_screen_shows_blocked_without_marking_incomplete(conn, monkeypatch):
    from humorhist.llm import StubClient

    conn.executemany("INSERT INTO pool (id,title,status) VALUES (?, ?, 'new')", [(str(n), str(n)) for n in range(2)])
    conn.execute("CREATE TABLE screen_attempts (pool_id TEXT PRIMARY KEY, state TEXT NOT NULL)")
    conn.execute("INSERT INTO screen_attempts VALUES ('0', 'failed')")
    conn.commit()
    monkeypatch.setattr(tg, "_get_llm", lambda *a: StubClient([[{"n": 1, "score": 5}]]))
    stub = tg.StubTelegram()
    tg.telegram_screen(conn, stub, "123", limit=1).result(5)
    text = stub.sent[-1]["text"]
    assert "scored=1/1" in text and "blocked=1" in text and "unscored=0" in text
    assert "incomplete" not in text
