"""Regression tests for Telegram confirmation and snapshot safety (offline)."""
import sqlite3
import threading

import pytest

from humorhist import db, job_control
from humorhist import telegram as tg


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "audit.sqlite"))
    db.migrate(c)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("INSERT INTO pool(id,title) VALUES ('p','Original')")
    c.execute("INSERT INTO drafts(id,pool_id,status) VALUES ('d','p','approved')")
    c.execute("INSERT INTO queue(draft_id,post_copy) VALUES ('d','old copy')")
    c.commit()
    yield c
    c.close()


def tap(conn, stub, data):
    return tg.handle_callback(conn, stub, "123", {"callback_query": {"id": "x", "data": data}})


def buttons(stub):
    return stub.sent[-1]["reply_markup"]["inline_keyboard"][0]


@pytest.mark.parametrize("action", ["reject", "posted"])
def test_cancel_revokes_intent_and_restart_fails_closed(conn, action):
    stub = tg.StubTelegram()
    tap(conn, stub, f"{action}:d:{tg._decision_token(conn, 'd')}")
    confirm, cancel = [b["callback_data"] for b in buttons(stub)]
    tap(conn, stub, cancel)
    tap(conn, stub, confirm)
    assert conn.execute("SELECT status FROM drafts").fetchone()[0] == "approved"
    assert conn.execute("SELECT published FROM queue").fetchone()[0] == 0
    tap(conn, stub, f"{action}:d:{tg._decision_token(conn, 'd')}")
    confirm = buttons(stub)[0]["callback_data"]
    tg._intents.clear()  # equivalent to a process restart
    tap(conn, stub, confirm)
    assert conn.execute("SELECT status FROM drafts").fetchone()[0] == "approved"
    assert conn.execute("SELECT published FROM queue").fetchone()[0] == 0


def test_confirmation_is_one_use(conn):
    stub = tg.StubTelegram()
    tap(conn, stub, f"posted:d:{tg._decision_token(conn, 'd')}")
    confirm = buttons(stub)[0]["callback_data"]
    tap(conn, stub, confirm)
    assert conn.execute("SELECT published FROM queue").fetchone()[0] == 1
    tap(conn, stub, confirm)
    assert conn.execute("SELECT published FROM queue").fetchone()[0] == 1
    assert "expired" in stub.sent[-1]["text"].lower()


def test_queue_snapshot_then_stale_button_rejected(conn, monkeypatch):
    other = db.connect(str(job_control.database_path(conn)))
    original = tg.review.queued_drafts

    def concurrent(c):
        rows = original(c)
        other.execute("UPDATE queue SET post_copy='new copy' WHERE draft_id='d'")
        other.execute("UPDATE pool SET title='New title' WHERE id='p'")
        other.commit()
        return rows

    monkeypatch.setattr(tg.review, "queued_drafts", concurrent)
    stub = tg.StubTelegram()
    tg.send_queue_list(conn, stub, "123")
    assert "old copy" in stub.sent[-1]["text"]
    assert "Original" in stub.sent[-1]["text"]
    button = stub.sent[-1]["reply_markup"]["inline_keyboard"][2][0]["callback_data"]
    tap(conn, stub, button)
    assert "expired" in stub.sent[-1]["text"].lower()
    assert conn.execute("SELECT published FROM queue").fetchone()[0] == 0
    other.close()


def test_version_schema_survives_rollback_and_cross_connection(conn):
    old = tg._decision_token(conn, "d")
    conn.execute("UPDATE queue SET post_copy='temporary'")
    conn.rollback()
    other = db.connect(str(job_control.database_path(conn)))
    other.execute("UPDATE queue SET post_copy='changed'")
    other.commit()
    assert tg._decision_token(conn, "d") != old
    assert other.execute("SELECT count(*) FROM sqlite_master WHERE type='trigger' AND name LIKE 'hh_tg_%'").fetchone()[0] == 7
    other.close()


def test_real_background_buffer_topup_and_busy(conn, monkeypatch):
    monkeypatch.setattr(tg, "_get_llm", lambda *a: object())
    started, release = threading.Event(), threading.Event()
    def draft(worker, llm, count):
        started.set()
        assert release.wait(5)
        return {"drafted": 1, "draft_ids": ["new"], "failed": 0}
    monkeypatch.setattr("humorhist.drafting.draft_candidates", draft)
    stub = tg.StubTelegram()
    future = tg.telegram_buffer(conn, stub, "123")
    try:
        assert started.wait(5)
        assert tg.telegram_buffer(conn, stub, "123") is None
        assert any("already running" in m["text"] for m in stub.sent)
    finally:
        release.set()
    future.result(5)
    assert any("Auto-drafted 1" in m["text"] for m in stub.sent)


@pytest.mark.parametrize("action", ["reject", "posted"])
def test_rolled_back_confirmation_can_retry_once(conn, monkeypatch, action):
    stub = tg.StubTelegram()
    tap(conn, stub, f"{action}:d:{tg._decision_token(conn, 'd')}")
    confirm = buttons(stub)[0]["callback_data"]
    name = "apply_review" if action == "reject" else "mark_posted"
    original = getattr(tg.review, name)

    def transient(c, *args, **kwargs):
        c.execute("UPDATE drafts SET editor_notes='rolled back' WHERE id='d'")
        raise sqlite3.OperationalError("injected failure")

    monkeypatch.setattr(tg.review, name, transient)
    with pytest.raises(sqlite3.OperationalError):
        tap(conn, stub, confirm)
    assert conn.execute("SELECT editor_notes FROM drafts WHERE id='d'").fetchone()[0] is None
    monkeypatch.setattr(tg.review, name, original)
    tap(conn, stub, confirm)
    if action == "reject":
        assert conn.execute("SELECT status FROM drafts").fetchone()[0] == "rejected"
    else:
        assert conn.execute("SELECT published FROM queue").fetchone()[0] == 1
    tap(conn, stub, confirm)
    assert "expired" in stub.sent[-1]["text"].lower()


def test_draft_snapshot_is_consistent_during_cross_connection_write(conn, monkeypatch):
    other = db.connect(str(job_control.database_path(conn)))
    old_row = dict(conn.execute("SELECT * FROM drafts WHERE id='d'").fetchone())
    old_get = tg.db.get_pool_item

    def interleaved(c, pool_id):
        pool = old_get(c, pool_id)
        other.execute("UPDATE drafts SET editor_notes='unseen' WHERE id='d'")
        other.execute("UPDATE pool SET title='Unseen title' WHERE id='p'")
        other.commit()
        return pool

    monkeypatch.setattr(tg.db, "get_pool_item", interleaved)
    stub = tg.StubTelegram()
    tg._send_one(conn, stub, "123", old_row)
    assert "Original" in "".join(m["text"] for m in stub.sent)
    reject = next(b["callback_data"] for row in stub.sent[-1]["reply_markup"]["inline_keyboard"]
                  for b in row if b["callback_data"].startswith("reject:"))
    tap(conn, stub, reject)
    assert "expired" in stub.sent[-1]["text"].lower()
    other.close()


@pytest.mark.parametrize("view", ["draft", "queue", "copy", "posted", "approved"])
def test_uncommitted_versions_never_escape_as_buttons(conn, view):
    row = dict(conn.execute("SELECT * FROM drafts WHERE id='d'").fetchone())
    conn.execute("UPDATE drafts SET editor_notes='temporary' WHERE id='d'")
    stub = tg.StubTelegram()
    if view == "draft":
        tg._send_one(conn, stub, "123", row)
    elif view == "queue":
        tg.send_queue_list(conn, stub, "123")
    elif view == "copy":
        tg.send_copy_content(conn, stub, "123", "d")
    elif view == "posted":
        tg.send_posted_list(conn, stub, "123")
    else:
        tg.send_approved_list(conn, stub, "123")
    assert all("reply_markup" not in m for m in stub.sent)
    assert conn.in_transaction
    conn.rollback()
    conn.execute("UPDATE drafts SET editor_notes='different' WHERE id='d'")
    conn.commit()
    assert all("reply_markup" not in m for m in stub.sent)


def test_noop_updates_preserve_version_but_real_changes_invalidate(conn):
    token = tg._decision_token(conn, "d")
    conn.execute("UPDATE pool SET title=title WHERE id='p'")
    conn.execute("UPDATE drafts SET status=status WHERE id='d'")
    conn.execute("UPDATE queue SET post_copy=post_copy WHERE draft_id='d'")
    conn.commit()
    assert tg._decision_token(conn, "d") == token
    conn.execute("UPDATE queue SET post_copy='new' WHERE draft_id='d'")
    conn.commit()
    assert tg._decision_token(conn, "d") != token
