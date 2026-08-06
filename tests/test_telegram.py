"""Tests for humorhist.telegram -- Phase 3.3/3.4 Telegram review transport.

TDD with a real file-backed DB via tmp_path and a StubTelegram (no network).
The stub mimics the Bot API response shapes closely enough to exercise the
real review logic: callbacks must call review.apply_review, text replies must
be stored as notes, and the bot must send one message per pending draft.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import humorhist.db as db
import humorhist.review as review
import humorhist.telegram as tg


def _fresh_db(tmp_path: Path):
    path = tmp_path / "test.sqlite"
    conn = db.connect(str(path))
    db.migrate(conn)
    return conn


def _seed_pending(conn, draft_id: str = "d1") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO pool (id, title, status) VALUES ('pool-x', 'Pool X', 'drafted')"
    )
    conn.execute(
        """INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at)
           VALUES (?, 'pool-x', '{}', '{}', 'pending', '2026-01-01T00:00:00+00:00')""",
        (draft_id,),
    )
    conn.commit()


def _callback_update(update_id, cb_id, decision, draft_id, msg_id=10):
    return {
        "update_id": update_id,
        "callback_query": {
            "id": cb_id,
            "data": f"{decision}:{draft_id}",
            "message": {"message_id": msg_id},
        },
    }


def _text_update(update_id, text, reply_to, msg_id=20):
    return {
        "update_id": update_id,
        "message": {
            "message_id": msg_id,
            "text": text,
            "reply_to_message": {"message_id": reply_to},
        },
    }


# --- send_pending_drafts -----------------------------------------------------


def test_send_pending_drafts_sends_one_message_per_pending(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    _seed_pending(conn, "d2")

    stub = tg.StubTelegram()
    sent = tg.send_pending_drafts(conn, stub, "chat")

    assert len(sent) == 2
    # each sent message carried the inline Approve/Reject keyboard
    for s in stub.sent:
        kb = s["reply_markup"]["inline_keyboard"][0]
        assert {b["callback_data"] for b in kb} == {"approve:d1", "reject:d1"} or \
               {b["callback_data"] for b in kb} == {"approve:d2", "reject:d2"}


def test_send_pending_drafts_empty_when_none_pending(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram()
    sent = tg.send_pending_drafts(conn, stub, "chat")
    assert sent == []
    assert stub.sent == []


# --- handle_callback ---------------------------------------------------------


def test_handle_callback_approve_calls_apply_review(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()

    res = tg.handle_callback(conn, stub, "chat", _callback_update(1, "cb1", "approve", "d1"))

    assert res is not None
    assert res["draft_id"] == "d1"
    assert res["decision"] == "approve"
    row = conn.execute("SELECT status, reviewed_at FROM drafts WHERE id='d1'").fetchone()
    assert row["status"] == "approved"
    assert row["reviewed_at"] is not None
    # callback was answered and a notes prompt was sent
    assert "cb1" in stub.answered
    assert any("optional notes" in m["text"] for m in stub.sent)


def test_handle_callback_reject_calls_apply_review(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()

    res = tg.handle_callback(conn, stub, "chat", _callback_update(1, "cb1", "reject", "d1"))

    assert res["decision"] == "reject"
    row = conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()
    assert row["status"] == "rejected"


def test_handle_callback_ignores_unknown_data(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()
    upd = {"update_id": 1, "callback_query": {"id": "cb1", "data": "bogus", "message": {"message_id": 10}}}

    res = tg.handle_callback(conn, stub, "chat", upd)

    assert res is None
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "pending"


# --- handle_text (notes) -----------------------------------------------------


def test_handle_text_stores_notes(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()
    # first approve -> draft approved, notes prompt sent as message id 1
    cb_res = tg.handle_callback(conn, stub, "chat", _callback_update(1, "cb1", "approve", "d1"))
    note_msg_id = cb_res["note_message_id"]

    res = tg.handle_text(conn, stub, "chat", {note_msg_id: "d1"}, _text_update(2, "tighten the third angle", note_msg_id))

    assert res is not None and res.get("noted") == "d1"
    row = conn.execute("SELECT editor_notes FROM drafts WHERE id='d1'").fetchone()
    assert row["editor_notes"] == "tighten the third angle"


def test_handle_text_skip_clears_awaiting(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()
    cb_res = tg.handle_callback(conn, stub, "chat", _callback_update(1, "cb1", "approve", "d1"))
    note_msg_id = cb_res["note_message_id"]

    awaiting = {note_msg_id: "d1"}
    res = tg.handle_text(conn, stub, "chat", awaiting, _text_update(2, "/skip", note_msg_id))

    assert res.get("skipped") == "d1"
    assert note_msg_id not in awaiting
    # notes left empty
    assert conn.execute("SELECT editor_notes FROM drafts WHERE id='d1'").fetchone()["editor_notes"] is None


def test_handle_text_ignores_unrelated_message(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram()
    res = tg.handle_text(conn, stub, "chat", {}, _text_update(2, "hello", 999))
    assert res is None


# --- run_review_bot ----------------------------------------------------------


def test_run_review_bot_once_pumps_callback(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram(updates=[_callback_update(1, "cb1", "approve", "d1")])

    decided = tg.run_review_bot(conn, stub, "chat", once=True)

    assert decided == 1
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "approved"
    # no pending left
    assert review.pending_drafts(conn) == []


def test_run_review_bot_no_pending_sends_notice(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram()

    decided = tg.run_review_bot(conn, stub, "chat", once=True)

    # no pending drafts, no updates -> nothing decided; but the loop now opens
    # with a (silent-when-empty) reviewed-summary message.
    assert decided == 0
    assert len(stub.sent) == 1
    assert "Review progress" in stub.sent[0]["text"]


def test_notify_new_drafts_used_for_nudge(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()

    n = tg.run_review_bot(conn, stub, "chat", once=True)
    # run_review_bot sends the pending draft (with buttons) but NOT the
    # "N drafts awaiting" nudge text -- that is notify_new_drafts's job.
    assert n == 0
    assert any("approve:d1" in str(m) for m in stub.sent)
    assert not any("awaiting review" in m["text"] for m in stub.sent)
    # the dedicated nudge reports the pending count
    assert tg.notify_new_drafts(conn, stub, "chat") == 1
    assert any("1 draft(s) awaiting" in m["text"] for m in stub.sent)


# --- notify_new_drafts -------------------------------------------------------


def test_notify_new_drafts_reports_count(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    _seed_pending(conn, "d2")
    stub = tg.StubTelegram()

    n = tg.notify_new_drafts(conn, stub, "chat")

    assert n == 2
    assert any("2 draft(s) awaiting" in m["text"] for m in stub.sent)


def test_notify_new_drafts_silent_when_empty(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram()
    n = tg.notify_new_drafts(conn, stub, "chat")
    assert n == 0
    assert stub.sent == []


# --- reviewed_summary --------------------------------------------------------


def _seed_with_statuses(tmp_path):
    conn = _fresh_db(tmp_path)
    conn.execute("INSERT OR IGNORE INTO pool (id, title, status) VALUES ('p1', 'Emu War', 'drafted')")
    conn.execute("INSERT OR IGNORE INTO pool (id, title, status) VALUES ('p2', 'Acoustic Kitty', 'drafted')")
    conn.execute("INSERT OR IGNORE INTO pool (id, title, status) VALUES ('p3', 'Napoleons Rabbits', 'drafted')")
    conn.execute(
        "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) VALUES ('d1','p1','{}','{}','approved','2026-01-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) VALUES ('d2','p2','{}','{}','rejected','2026-01-02T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) VALUES ('d3','p3','{}','{}','pending','2026-01-03T00:00:00+00:00')"
    )
    conn.commit()
    return conn


def test_reviewed_summary_counts_and_titles(tmp_path):
    conn = _seed_with_statuses(tmp_path)
    s = review.reviewed_summary(conn)
    assert s["approved"]["count"] == 1 and s["approved"]["titles"] == ["Emu War"]
    assert s["rejected"]["count"] == 1 and s["rejected"]["titles"] == ["Acoustic Kitty"]
    assert s["pending"]["count"] == 1 and s["pending"]["titles"] == ["Napoleons Rabbits"]


def test_format_reviewed_summary_includes_all_sections(tmp_path):
    conn = _seed_with_statuses(tmp_path)
    text = tg.format_reviewed_summary(review.reviewed_summary(conn))
    assert "Emu War" in text
    assert "Acoustic Kitty" in text
    assert "Approved" in text and "Rejected" in text and "Pending: 1" in text


def test_send_reviewed_summary_dms_breakdown(tmp_path):
    conn = _seed_with_statuses(tmp_path)
    stub = tg.StubTelegram()
    text = tg.send_reviewed_summary(conn, stub, "chat")
    assert len(stub.sent) == 1
    assert "Emu War" in text and "Acoustic Kitty" in text


def test_telegram_review_sends_summary_first(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram(updates=[])
    tg.run_review_bot(conn, stub, "chat", once=True)
    # first message is the summary, second is the pending draft
    assert len(stub.sent) == 2
    assert "Review progress" in stub.sent[0]["text"]
    assert "approve:d1" in str(stub.sent[1])


class _RaisingOnDraftStub(tg.StubTelegram):
    """Stub that fails to send one specific draft (simulates a Telegram 400)."""

    def send_message(self, chat_id, text, reply_markup=None):
        if "d1" in text:
            raise RuntimeError("simulated 400")
        return super().send_message(chat_id, text, reply_markup=reply_markup)


def test_send_pending_drafts_skips_failing_draft(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    _seed_pending(conn, "d2")
    stub = _RaisingOnDraftStub(updates=[])
    sent = tg.send_pending_drafts(conn, stub, "chat")
    # d1 failed, d2 still sent; loop must not crash
    assert len(sent) == 1
    assert "approve:d2" in str(sent[0])


def test_handle_text_fallback_when_single_awaiting(tmp_path):
    conn = _fresh_db(tmp_path)
    conn.execute("INSERT OR IGNORE INTO pool (id, title) VALUES ('p1','X')")
    conn.execute(
        "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) "
        "VALUES ('d1','p1','{}','{}','approved','2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    stub = tg.StubTelegram()
    awaiting = {"note1": "d1"}
    upd = {"update_id": 5, "message": {"message_id": 10, "chat": {"id": "chat"},
                                       "text": "tighten the third angle"}}
    res = tg.handle_text(conn, stub, "chat", awaiting, upd)
    assert res == {"noted": "d1"}
    row = conn.execute("SELECT editor_notes FROM drafts WHERE id='d1'").fetchone()
    assert row["editor_notes"] == "tighten the third angle"


def test_chunk_text_respects_limit():
    text = "\n".join(f"line {i}" for i in range(5000))
    chunks = tg._chunk_text(text, limit=1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert "\n".join(chunks) == text  # no data lost across chunk boundaries
    assert len(chunks) > 1


def test_send_long_puts_buttons_on_last_chunk():
    stub = tg.StubTelegram()
    sent = tg._send_long(stub, "chat", "a\n" * 9000, reply_markup={"k": 1})
    assert len(sent) > 1
    assert all("reply_markup" not in m for m in sent[:-1])
    assert sent[-1]["reply_markup"] == {"k": 1}
