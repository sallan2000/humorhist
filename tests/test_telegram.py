"""Tests for humorhist.telegram -- Phase 3.3/3.4 Telegram review transport.

TDD with a real file-backed DB via tmp_path and a StubTelegram (no network).
The stub mimics the Bot API response shapes closely enough to exercise the
real review logic: callbacks must call review.apply_review, text replies must
be stored as notes, and the bot must send one message per pending draft.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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


def _confirm_update(update_id, cb_id, decision, draft_id, msg_id=10):
    """A *confirmed* approve/reject tap (the second step of the GAP 3 gate)."""
    return {
        "update_id": update_id,
        "callback_query": {
            "id": cb_id,
            "data": f"confirm:{decision}:{draft_id}",
            "message": {"message_id": msg_id},
        },
    }


def _cancel_update(update_id, cb_id, draft_id, msg_id=10):
    return {
        "update_id": update_id,
        "callback_query": {
            "id": cb_id,
            "data": f"cancel:{draft_id}",
            "message": {"message_id": msg_id},
        },
    }


def _approve_and_confirm(conn, stub, draft_id="d1"):
    """Drive a pending draft through the full approve + confirm gate.

    Returns the ``handle_callback`` result of the *confirm* step (which carries
    ``note_message_id`` + ``stage`` so the editor_line can be captured).
    """
    tg.handle_callback(
        conn, stub, "chat",
        _callback_update(1, "cb1", "approve", draft_id),
    )
    # the confirm callback returns the editor_line prompt metadata
    return tg.handle_callback(
        conn, stub, "chat",
        _confirm_update(2, "cb2", "approve", draft_id),
    )


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
    # each sent message carried the inline Approve/Reject/Later keyboard
    for s in stub.sent:
        kb = s["reply_markup"]["inline_keyboard"][0]
        cbs = {b["callback_data"] for b in kb}
        assert cbs == {"approve:d1", "reject:d1", "later:d1"} or \
               cbs == {"approve:d2", "reject:d2", "later:d2"}


def test_send_pending_drafts_empty_when_none_pending(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram()
    sent = tg.send_pending_drafts(conn, stub, "chat")
    assert sent == []
    assert stub.sent == []


# --- handle_callback ---------------------------------------------------------


def test_handle_callback_approve_opens_confirm_gate(tmp_path):
    """GAP 3: an initial approve tap does NOT commit — it opens a confirm gate."""
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()

    res = tg.handle_callback(conn, stub, "chat", _callback_update(1, "cb1", "approve", "d1"))

    # returns a 'confirming' marker, not the committed editor_line prompt
    assert res is not None
    assert res["draft_id"] == "d1"
    assert res["decision"] == "approve"
    assert res.get("confirming") is True
    # NOTHING committed yet
    row = conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()
    assert row["status"] == "pending"
    # a confirm/cancel message was sent
    assert any("Confirm" in m["text"] for m in stub.sent)


def test_handle_callback_approve_confirm_commits(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()

    # first tap -> gate; second tap -> commit
    tg.handle_callback(conn, stub, "chat", _callback_update(1, "cb1", "approve", "d1"))
    res = tg.handle_callback(conn, stub, "chat", _confirm_update(2, "cb2", "approve", "d1"))

    assert res["decision"] == "approve"
    row = conn.execute("SELECT status, reviewed_at FROM drafts WHERE id='d1'").fetchone()
    assert row["status"] == "approved"
    assert row["reviewed_at"] is not None
    # callback answered + editor_line prompt sent
    assert "cb2" in stub.answered
    assert any("one-line joke" in m["text"] for m in stub.sent)
    assert res["stage"] == "editor_line"


def test_handle_callback_approve_cancel_is_noop(tmp_path):
    """GAP 3: cancelling the gate leaves the draft untouched."""
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()

    tg.handle_callback(conn, stub, "chat", _callback_update(1, "cb1", "approve", "d1"))
    res = tg.handle_callback(conn, stub, "chat", _cancel_update(2, "cb2", "d1"))

    assert res.get("cancelled") is True
    row = conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()
    assert row["status"] == "pending"
    assert review.pending_drafts(conn) == [{"id": "d1", "pool_id": "pool-x", "brief_json": "{}", "angles_json": "{}", "status": "pending", "created_at": "2026-01-01T00:00:00+00:00", "editor_line": None, "editor_notes": None, "reviewed_at": None, "defer_until": None}]


def test_handle_callback_reject_calls_apply_review(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()

    # reject also goes through the confirm gate
    tg.handle_callback(conn, stub, "chat", _callback_update(1, "cb1", "reject", "d1"))
    res = tg.handle_callback(conn, stub, "chat", _confirm_update(2, "cb2", "reject", "d1"))

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


def test_handle_text_captures_editor_line_then_optional_notes(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()
    cb_res = _approve_and_confirm(conn, stub, "d1")
    editor_line_msg_id = cb_res["note_message_id"]
    awaiting = {
        "d1": {
            "note_message_id": editor_line_msg_id,
            "stage": "editor_line",
            "decision": "approve",
        }
    }

    # first reply -> the human joke (editor_line)
    res1 = tg.handle_text(
        conn, stub, "chat", awaiting, _text_update(2, "The bear was a tax dodge.", editor_line_msg_id)
    )
    assert res1.get("editor_line_set") == "d1"
    row = conn.execute("SELECT editor_line, editor_notes FROM drafts WHERE id='d1'").fetchone()
    assert row["editor_line"] == "The bear was a tax dodge."
    # a secondary notes prompt should now be open
    assert awaiting["d1"]["stage"] == "notes"

    # second reply -> optional notes, must NOT clobber the editor_line
    notes_msg_id = awaiting["d1"]["note_message_id"]
    res2 = tg.handle_text(
        conn, stub, "chat", awaiting, _text_update(3, "tighten the third angle", notes_msg_id)
    )
    assert res2.get("noted") == "d1"
    row = conn.execute("SELECT editor_line, editor_notes FROM drafts WHERE id='d1'").fetchone()
    assert row["editor_line"] == "The bear was a tax dodge."
    assert row["editor_notes"] == "tighten the third angle"


def test_handle_text_editor_line_skip_then_notes(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()
    cb_res = _approve_and_confirm(conn, stub, "d1")
    awaiting = {"d1": {"note_message_id": cb_res["note_message_id"], "stage": "editor_line", "decision": "approve"}}

    res1 = tg.handle_text(conn, stub, "chat", awaiting, _text_update(2, "/skip", cb_res["note_message_id"]))
    assert res1 is not None
    row = conn.execute("SELECT editor_line, editor_notes FROM drafts WHERE id='d1'").fetchone()
    assert row["editor_line"] is None
    assert awaiting["d1"]["stage"] == "notes"


def test_handle_text_notes_merge_keeps_existing_editor_line(tmp_path):
    """A /listapproved 'Add notes' re-apply must not wipe a prior editor_line."""
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    # pre-set an editor_line via a normal apply_review
    review.apply_review(conn, "d1", decision="approve", editor_line="Pre-existing joke")
    stub = tg.StubTelegram()
    # simulate the /listapproved notes button -> awaiting has stage 'notes'
    cb_res = tg.handle_callback(conn, stub, "chat", _callback_update(1, "cb1", "notes", "d1"))
    note_msg_id = cb_res["note_message_id"]
    awaiting = {"d1": {"note_message_id": note_msg_id, "stage": "notes", "decision": "approve"}}

    res = tg.handle_text(conn, stub, "chat", awaiting, _text_update(2, "add a citation", note_msg_id))
    assert res.get("noted") == "d1"
    row = conn.execute("SELECT editor_line, editor_notes FROM drafts WHERE id='d1'").fetchone()
    assert row["editor_line"] == "Pre-existing joke"  # preserved by merge
    assert row["editor_notes"] == "add a citation"


def test_handle_text_skip_clears_awaiting(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()
    cb_res = _approve_and_confirm(conn, stub, "d1")
    awaiting = {"d1": {"note_message_id": cb_res["note_message_id"], "stage": "editor_line", "decision": "approve"}}

    res = tg.handle_text(conn, stub, "chat", awaiting, _text_update(2, "/skip", cb_res["note_message_id"]))
    # skipping the editor_line moves to the notes stage, not out
    assert awaiting["d1"]["stage"] == "notes"
    # now skip notes too -> cleared
    notes_msg_id = awaiting["d1"]["note_message_id"]
    res2 = tg.handle_text(conn, stub, "chat", awaiting, _text_update(3, "/skip", notes_msg_id))
    assert res2.get("skipped") == "d1"
    assert "d1" not in awaiting
    # nothing stored
    row = conn.execute("SELECT editor_line, editor_notes FROM drafts WHERE id='d1'").fetchone()
    assert row["editor_line"] is None and row["editor_notes"] is None


def test_handle_text_ignores_unrelated_message(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram()
    res = tg.handle_text(conn, stub, "chat", {}, _text_update(2, "hello", 999))
    assert res is None


# --- run_review_bot ----------------------------------------------------------


def test_run_review_bot_once_pumps_callback(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    # GAP 3: an approve now needs a separate confirm tap
    stub = tg.StubTelegram(updates=[
        _callback_update(1, "cb1", "approve", "d1"),
        _confirm_update(2, "cb2", "approve", "d1"),
    ])

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
    awaiting = {"d1": {"note_message_id": "note1"}}
    upd = {"update_id": 5, "message": {"message_id": 10, "chat": {"id": "chat"},
                                       "text": "tighten the third angle"}}
    res = tg.handle_text(conn, stub, "chat", awaiting, upd)
    assert res == {"noted": "d1"}
    row = conn.execute("SELECT editor_notes FROM drafts WHERE id='d1'").fetchone()
    assert row["editor_notes"] == "tighten the third angle"


class _SeqStub(tg.StubTelegram):
    """Returns a scripted sequence of update batches, then empty."""

    def __init__(self, batches):
        super().__init__(updates=[])
        self._batches = list(batches)

    def get_updates(self, offset=0, timeout=0):
        return self._batches.pop(0) if self._batches else []


def _callback_batch(draft_id, update_id=1):
    return [{
        "update_id": update_id,
        "callback_query": {
            "id": f"cb{update_id}",
            "data": f"approve:{draft_id}",
            "message": {"message_id": 100 + update_id},
        },
    }]


def _reject_batch(draft_id, update_id=1):
    return [{
        "update_id": update_id,
        "callback_query": {
            "id": f"cb{update_id}",
            "data": f"reject:{draft_id}",
            "message": {"message_id": 100 + update_id},
        },
    }]


def _confirm_batch(draft_id, update_id=1):
    """The second step of the GAP 3 confirm gate (confirm:<decision>:<id>)."""
    return [{
        "update_id": update_id,
        "callback_query": {
            "id": f"cb{update_id}",
            "data": f"confirm:approve:{draft_id}",
            "message": {"message_id": 100 + update_id},
        },
    }]


def _confirm_reject_batch(draft_id, update_id=1):
    return [{
        "update_id": update_id,
        "callback_query": {
            "id": f"cb{update_id}",
            "data": f"confirm:reject:{draft_id}",
            "message": {"message_id": 100 + update_id},
        },
    }]


def _message_batch(text, update_id=1):
    return [{
        "update_id": update_id,
        "message": {"message_id": 100 + update_id, "chat": {"id": "chat"}, "text": text},
    }]


def test_run_review_bot_one_by_one(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    _seed_pending(conn, "d2")
    # realistic one-at-a-time flow: /reviewdraft, then for EACH draft a tap plus
    # its joke + /skip notes capture, before the next draft is shown.
    stub = _SeqStub([
        _message_batch("/reviewdraft", 1),
        _callback_batch("d1", 2),
        _confirm_batch("d1", 2),
        _message_batch("joke for d1", 3),
        _message_batch("/skip", 4),
        _callback_batch("d2", 5),
        _confirm_batch("d2", 5),
        _message_batch("joke for d2", 6),
        _message_batch("/skip", 7),
    ])
    decided = tg.run_review_bot(conn, stub, "chat", max_iterations=50)
    assert decided == 2
    # both drafts persisted as approved, in decision order
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "approved"
    assert conn.execute("SELECT status FROM drafts WHERE id='d2'").fetchone()["status"] == "approved"
    # jokes captured
    assert conn.execute("SELECT editor_line FROM drafts WHERE id='d1'").fetchone()["editor_line"] == "joke for d1"
    assert conn.execute("SELECT editor_line FROM drafts WHERE id='d2'").fetchone()["editor_line"] == "joke for d2"
    # one draft shown at a time (not dumped all at once up front). With the
    # GAP 3 confirm gate each draft yields TWO button messages (the Approve
    # tap, then the Confirm/Cancel gate) = 4 total for two drafts.
    button_msgs = [m for m in stub.sent if "reply_markup" in m]
    assert len(button_msgs) == 4  # 2 drafts x (approve + confirm), not 8 simultaneously
    # and the confirm gate was actually shown for each
    assert sum("confirm:approve:d1" in str(m) for m in button_msgs) == 1
    assert sum("confirm:approve:d2" in str(m) for m in button_msgs) == 1


def test_review_session_waits_for_joke_before_next_draft(tmp_path):
    """Approve must NOT race the next draft: d2's buttons appear only after d1's
    editor_line (joke) + notes capture is complete."""
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    _seed_pending(conn, "d2")

    class _ScriptStub(tg.StubTelegram):
        def __init__(self, batches):
            super().__init__(updates=[])
            self._batches = list(batches)
        def get_updates(self, offset=0, timeout=0):
            return self._batches.pop(0) if self._batches else []

    # /reviewdraft -> approve d1 -> CONFIRM d1 -> joke -> /skip notes ->
    # approve d2 -> CONFIRM d2 -> joke -> /skip
    stub = _ScriptStub([
        _message_batch("/reviewdraft", 1),
        _callback_batch("d1", 2),
        _confirm_batch("d1", 2),
        _message_batch("the emus did nothing wrong", 3),  # editor_line reply
        _message_batch("/skip", 4),                        # end notes capture
        _callback_batch("d2", 5),
        _confirm_batch("d2", 5),
        _message_batch("and the humans lost", 6),
        _message_batch("/skip", 7),
    ])
    decided = tg.run_review_bot(conn, stub, "chat", max_iterations=60)

    # d1's joke was captured before any d2 button could be shown
    assert conn.execute("SELECT editor_line FROM drafts WHERE id='d1'").fetchone()["editor_line"] == "the emus did nothing wrong"
    # find button messages per draft
    d1_btn = next(i for i, m in enumerate(stub.sent) if "reply_markup" in m and "approve:d1" in str(m))
    d2_btn = next(i for i, m in enumerate(stub.sent) if "reply_markup" in m and "approve:d2" in str(m))
    joke_prompt = next(i for i, m in enumerate(stub.sent) if "one-line joke" in m["text"])
    # the joke prompt (and thus the whole d1 capture) precedes d2's button
    assert joke_prompt < d2_btn, "next draft appeared before the joke prompt"
    assert d1_btn < d2_btn, "d2 button raced ahead of d1"
    # both drafts fully decided, in order, with their jokes captured
    assert decided == 2
    assert conn.execute("SELECT editor_line FROM drafts WHERE id='d2'").fetchone()["editor_line"] == "and the humans lost"


def test_listapproved_lists_greenlit_drafts_with_buttons(tmp_path):
    conn = _seed_with_statuses(tmp_path)  # d1 approved, d2 rejected, d3 pending
    stub = tg.StubTelegram()
    n = tg.send_approved_list(conn, stub, "chat")
    assert n == 1
    msg = stub.sent[0]
    assert "Emu War" in msg["text"]
    # the approved draft has an inline 'view' button (opens content + add notes)
    btns = msg["reply_markup"]["inline_keyboard"]
    assert btns[0][0]["callback_data"] == "view:d1"


def test_listapproved_add_notes_via_button(tmp_path):
    conn = _fresh_db(tmp_path)
    conn.execute("INSERT OR IGNORE INTO pool (id, title) VALUES ('p1','Emu War')")
    conn.execute(
        "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) "
        "VALUES ('d1','p1','{}','{}','approved','2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    stub = tg.StubTelegram()
    # 1) user taps the 'add notes' button on the /listapproved list
    cb_upd = {
        "update_id": 1,
        "callback_query": {
            "id": "cb1",
            "data": "notes:d1",
            "message": {"message_id": 100},
        },
    }
    res = tg.handle_callback(conn, stub, "chat", cb_upd)
    assert res and res["draft_id"] == "d1"
    awaiting = {"d1": {"note_message_id": res["note_message_id"]}}
    # 2) user replies with the note
    note_upd = {"update_id": 2, "message": {"message_id": 101, "chat": {"id": "chat"}, "text": "lead with the emus"}}
    r = tg.handle_text(conn, stub, "chat", awaiting, note_upd)
    assert r == {"noted": "d1"}
    row = conn.execute("SELECT editor_notes FROM drafts WHERE id='d1'").fetchone()
    assert row["editor_notes"] == "lead with the emus"
    # still approved + still in queue (idempotent re-approve)
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "approved"
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone() is not None


def test_unknown_command_is_rejected(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram()
    stub.get_updates = lambda offset=0, timeout=0: [
        {"update_id": 1, "message": {"message_id": 9, "chat": {"id": "chat"}, "text": "/bogus"}}
    ]
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    # the bot replied with the unknown-command message
    assert any("Unknown command" in m["text"] for m in stub.sent)


def test_listapproved_opens_draft_content_with_add_notes_button(tmp_path):
    conn = _seed_with_statuses(tmp_path)  # d1 approved
    stub = tg.StubTelegram()
    # 1) tap the 'view' button from /listapproved
    view_upd = {
        "update_id": 1,
        "callback_query": {
            "id": "cb1", "data": "view:d1", "message": {"message_id": 100},
        },
    }
    res = tg.handle_callback(conn, stub, "chat", view_upd)
    assert res is None  # view just opens content, no note prompt yet
    # the draft content was sent (rendered), chunked, last chunk has Add notes btn
    content_msgs = [m for m in stub.sent if "reply_markup" in m]
    assert content_msgs, "draft content should have been sent"
    last = content_msgs[-1]
    assert last["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "notes:d1"
    # the rendered body should actually contain the topic
    full = "\n".join(m["text"] for m in stub.sent)
    assert "Emu War" in full


def test_send_draft_content_shows_post_copy_and_copy_button(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn, draft_id="d1", with_copy=True)
    stub = tg.StubTelegram()
    tg.send_draft_content(conn, stub, "chat", "d1")
    full = "\n".join(m["text"] for m in stub.sent)
    # post copy + char count visible when opening an approved draft
    assert "POST COPY" in full
    assert "France invaded" in full  # the seeded copy text
    assert "/280" in full  # char count against the limit
    # a button to open the edit/regenerate copy view
    kb = stub.sent[-1]["reply_markup"]["inline_keyboard"][0]
    cbs = {b["callback_data"] for b in kb}
    assert "copy:d1" in cbs  # opens send_copy_content (edit/regen)
    assert "notes:d1" in cbs  # existing add-notes button preserved


def test_send_draft_content_copy_absent_shows_zero_count(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn, draft_id="d1", with_copy=False)
    stub = tg.StubTelegram()
    tg.send_draft_content(conn, stub, "chat", "d1")
    full = "\n".join(m["text"] for m in stub.sent)
    assert "POST COPY" in full
    assert "0/280" in full  # no copy yet, counts as zero against the limit
    kb = stub.sent[-1]["reply_markup"]["inline_keyboard"][0]
    assert any(b["callback_data"] == "copy:d1" for b in kb)


def test_send_draft_content_missing_draft_is_safe(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram()
    sent = tg.send_draft_content(conn, stub, "chat", "does-not-exist")
    assert sent == []
    assert any("No such draft" in m["text"] for m in stub.sent)


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


# --- B+ post-copy editing (Telegram) ---------------------------------------


def _seed_approved_queued(conn, draft_id="d1", with_copy=True):
    conn.execute(
        "INSERT OR IGNORE INTO pool (id, title, status) VALUES ('pool-x', 'Pastry War', 'drafted')"
    )
    conn.execute(
        "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) "
        "VALUES (?, 'pool-x', '{}', '{}', 'approved', '2026-01-01T00:00:00+00:00')",
        (draft_id,),
    )
    copy = "'France invaded Mexico over a pastry shop.'" if with_copy else "NULL"
    conn.execute(
        f"INSERT INTO queue (draft_id, scheduled_for, published, post_copy) "
        f"VALUES ('{draft_id}', NULL, 0, {copy})"
    )
    conn.commit()


def test_send_queue_list_shows_copy_and_char_count(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)
    stub = tg.StubTelegram()
    n = tg.send_queue_list(conn, stub, "chat")
    assert n == 1
    text = stub.sent[-1]["text"]
    assert "Pastry War" in text
    assert "France invaded" in text
    kb = stub.sent[-1]["reply_markup"]["inline_keyboard"][0][0]
    assert kb["callback_data"] == "copy:d1"


def test_send_queue_list_empty_when_no_queue(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram()
    n = tg.send_queue_list(conn, stub, "chat")
    assert n == 0
    assert "empty" in stub.sent[-1]["text"].lower()


def test_listapproved_shows_reject_button(tmp_path):
    """An approved draft listed in /listapproved carries a confirm-gated
    Reject button so it can be un-approved from Telegram (closes the
    approve-then-change-mind gap)."""
    conn = _seed_with_statuses(tmp_path)  # d1 approved
    stub = tg.StubTelegram()
    n = tg.send_approved_list(conn, stub, "chat")
    assert n == 1
    kb = stub.sent[-1]["reply_markup"]["inline_keyboard"][0]
    cbs = {b["callback_data"] for b in kb}
    assert "view:d1" in cbs
    assert "reject:d1" in cbs  # the new un-approve button
    assert any(b["text"].startswith("❌ Reject") for b in kb)


def test_listqueue_shows_reject_and_reopen_buttons(tmp_path):
    """/listqueue rows now expose Reject (confirm-gated) and Reopen
    (back-to-pending) buttons so an approved+queued draft can be fully
    un-approved from Telegram without the CLI."""
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)  # d1 approved + queued
    stub = tg.StubTelegram()
    n = tg.send_queue_list(conn, stub, "chat")
    assert n == 1
    kb = stub.sent[-1]["reply_markup"]["inline_keyboard"]
    cbs = {b["callback_data"] for row in kb for b in row}
    assert "copy:d1" in cbs
    assert "remove:d1" in cbs
    assert "reject:d1" in cbs   # new
    assert "reopen:d1" in cbs   # new


def test_reject_button_from_list_unapproves_after_confirm(tmp_path):
    """End-to-end: the reject:d1 tap opens the confirm gate (GAP 3); the
    confirm:reject:d1 tap flips the approved draft to rejected and pulls it
    out of the publish queue. A reject must NOT prompt for the one-line joke
    (that's an approve-only capture) and must NOT store an editor_line."""
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)  # d1 approved + queued
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone() is not None
    stub = tg.StubTelegram()
    # 1) tap the Reject button -> confirm gate (no commit yet)
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "reject:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res == {"draft_id": "d1", "decision": "reject", "confirming": True}
    # still approved + queued until confirm
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "approved"
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone() is not None
    # 2) confirm -> rejected + dequeued; no joke prompt, no editor_line stored
    messages_before = len(stub.sent)
    res2 = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 2, "callback_query": {"id": "cb2", "data": "confirm:reject:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res2 == {"draft_id": "d1", "decision": "reject"}  # no note_message_id
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "rejected"
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone() is None
    assert conn.execute("SELECT editor_line FROM drafts WHERE id='d1'").fetchone()["editor_line"] is None
    assert any("reject" in m["text"].lower() for m in stub.sent)
    # the only new message after confirm is the rejection confirmation (no joke prompt)
    new_msgs = stub.sent[messages_before:]
    assert not any("one-line joke" in m["text"].lower() for m in new_msgs)


def test_reject_in_review_session_counts_decision_without_joke_prompt(tmp_path):
    """In the one-by-one /reviewdraft flow, confirming a Reject increments the
    decided count but does NOT open a joke/notes prompt (approve-only capture),
    so the bot moves straight on to the next pending draft."""
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    _seed_pending(conn, "d2")
    stub = _SeqStub([
        _message_batch("/reviewdraft", 1),
        _reject_batch("d1", 2),
        _confirm_reject_batch("d1", 2),   # confirm REJECT d1
        _callback_batch("d2", 3),
        _confirm_batch("d2", 3),           # confirm APPROVE d2
        _message_batch("joke for d2", 4),
        _message_batch("/skip", 5),
    ])
    decided = tg.run_review_bot(conn, stub, "chat", max_iterations=50)
    assert decided == 2  # both decisions counted, including the reject
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "rejected"
    assert conn.execute("SELECT status FROM drafts WHERE id='d2'").fetchone()["status"] == "approved"
    # d1 rejected -> no editor_line; d2 approved -> joke captured
    assert conn.execute("SELECT editor_line FROM drafts WHERE id='d1'").fetchone()["editor_line"] is None
    assert conn.execute("SELECT editor_line FROM drafts WHERE id='d2'").fetchone()["editor_line"] == "joke for d2"
    # d1's reject produced no "one-line joke" prompt
    assert not any("one-line joke" in m["text"].lower() and "d1" in m["text"] for m in stub.sent)



def test_reject_button_cancel_keeps_approval(tmp_path):
    """Cancelling the reject confirm gate leaves the draft approved + queued."""
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)  # d1 approved + queued
    stub = tg.StubTelegram()
    tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "reject:d1",
                                            "message": {"message_id": 1}}},
    )
    tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 2, "callback_query": {"id": "cb2", "data": "cancel:d1",
                                            "message": {"message_id": 1}}},
    )
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "approved"
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone() is not None
    assert any("cancelled" in m["text"] for m in stub.sent)


def test_view_command_re_reads_any_draft_from_phone(tmp_path):
    """`/view <id>` must render a draft's full content from Telegram regardless
    of status (pending/approved/rejected) -- the phone has no other way to
    re-read a pending draft outside a /reviewdraft session."""
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")  # a pending draft, not in queue/listapproved
    stub = tg.StubTelegram(updates=[
        {"update_id": 1, "message": {"message_id": 9, "chat": {"id": "chat"},
                                     "text": "/view d1"}},
    ])
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    # the draft body was sent (rendered), incl. the topic + sources
    full = "\n".join(m["text"] for m in stub.sent)
    assert "Pool X" in full  # the seeded pending draft's pool title
    assert "VERIFIED FACTS" in full  # the full draft renderer ran


def test_view_command_missing_id_shows_usage(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram(updates=[
        {"update_id": 1, "message": {"message_id": 9, "chat": {"id": "chat"},
                                     "text": "/view"}},
    ])
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    assert any("Usage: /view <draft_id>" in m["text"] for m in stub.sent)


def test_view_command_unknown_id_is_safe(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram(updates=[
        {"update_id": 1, "message": {"message_id": 9, "chat": {"id": "chat"},
                                     "text": "/view nope"}},
    ])
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    assert any("No such draft" in m["text"] for m in stub.sent)


def test_send_queue_list_shows_draft_id_and_remove_button(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)  # d1 approved + queued
    stub = tg.StubTelegram()
    n = tg.send_queue_list(conn, stub, "chat")
    assert n == 1
    text = stub.sent[-1]["text"]
    # the numeric draft id is shown so /viewcopy <id> / /queue remove <id> are usable
    assert "#d1" in text
    kb = stub.sent[-1]["reply_markup"]["inline_keyboard"][0]
    cbs = {b["callback_data"] for b in kb}
    assert "copy:d1" in cbs
    # Remove button pulls the draft out of the queue without typing the id
    assert "remove:d1" in cbs


def test_remove_callback_pulls_draft_from_queue(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)  # d1 approved + queued
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone() is not None
    stub = tg.StubTelegram()
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "remove:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res is None
    # removed from the queue but kept approved
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone() is None
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "approved"
    assert any("removed from the queue" in m["text"] for m in stub.sent)


def test_remove_callback_when_not_in_queue_is_safe(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram()
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "remove:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res is None
    assert any("was not in the queue" in m["text"] for m in stub.sent)


def _seed_rejected(conn, draft_id="d1"):
    conn.execute(
        "INSERT OR IGNORE INTO pool (id, title, status) VALUES ('pool-x', 'Pastry War', 'drafted')"
    )
    conn.execute(
        "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, reviewed_at) "
        "VALUES (?, 'pool-x', '{}', '{}', 'rejected', '2026-01-01T00:00:00+00:00')",
        (draft_id,),
    )
    conn.commit()


def test_send_rejected_list_shows_reopen_button(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_rejected(conn, "d1")
    stub = tg.StubTelegram()
    n = tg.send_rejected_list(conn, stub, "chat")
    assert n == 1
    text = stub.sent[-1]["text"]
    assert "#d1" in text
    assert "Pastry War" in text
    kb = stub.sent[-1]["reply_markup"]["inline_keyboard"][0][0]
    assert kb["callback_data"] == "reopen:d1"
    assert kb["text"].startswith("↩️ Reopen")


def test_send_rejected_list_empty_when_none(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram()
    n = tg.send_rejected_list(conn, stub, "chat")
    assert n == 0
    assert "No rejected" in stub.sent[-1]["text"]


def _seed_deferred(conn, draft_id="d1", days=30):
    conn.execute(
        "INSERT OR IGNORE INTO pool (id, title, status) VALUES ('pool-x', 'Pastry War', 'drafted')"
    )
    future = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
    conn.execute(
        "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at, defer_until) "
        "VALUES (?, 'pool-x', '{}', '{}', 'pending', '2026-01-01T00:00:00+00:00', ?)",
        (draft_id, future),
    )
    conn.commit()


def test_send_deferred_list_shows_reviewnow_button(tmp_path):
    """GAP 4: /listlater lists deferred drafts each with a bring-forward button."""
    conn = _fresh_db(tmp_path)
    _seed_deferred(conn, "d1")
    stub = tg.StubTelegram()
    n = tg.send_deferred_list(conn, stub, "chat")
    assert n == 1
    text = stub.sent[-1]["text"]
    assert "#d1" in text
    assert "Pastry War" in text
    kb = stub.sent[-1]["reply_markup"]["inline_keyboard"][0][0]
    assert kb["callback_data"] == "reviewnow:d1"
    assert kb["text"].startswith("⏩ Review now")


def test_send_deferred_list_empty_when_none(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram()
    n = tg.send_deferred_list(conn, stub, "chat")
    assert n == 0
    assert "No deferred" in stub.sent[-1]["text"]


def test_reviewnow_callback_brings_draft_forward(tmp_path):
    """GAP 4: tapping the reviewnow:<id> button clears the defer + returns to review."""
    conn = _fresh_db(tmp_path)
    _seed_deferred(conn, "d1")
    stub = tg.StubTelegram()
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "reviewnow:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res is None
    # defer cleared -> back in the review surface
    assert [d["id"] for d in review.pending_drafts(conn)] == ["d1"]
    assert [r["draft_id"] for r in review.deferred_drafts(conn)] == []


def test_reviewnow_callback_when_not_deferred_is_safe(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")  # not deferred
    stub = tg.StubTelegram()
    tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "reviewnow:d1",
                                            "message": {"message_id": 1}}},
    )
    # a clean "Cannot bring forward" message, no crash
    assert any("Cannot bring forward" in m["text"] for m in stub.sent)


def test_setjoke_callback_opens_editor_line_prompt(tmp_path):
    """GAP 4b: setjoke:<id> re-opens the joke prompt on an approved+queued draft."""
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    review.apply_review(conn, "d1", decision="approve")  # approved + queued, no joke
    stub = tg.StubTelegram()
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "setjoke:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res["stage"] == "editor_line"
    assert res.get("setjoke") is True
    # reply with the joke -> set via set_editor_line (no status/queue change)
    note_id = res["note_message_id"]
    tg.handle_text(
        conn, stub, "chat",
        {"d1": {"note_message_id": note_id, "stage": "editor_line", "decision": "approve", "setjoke": True}},
        _text_update(2, "the bear was a tax dodge", note_id),
    )
    row = conn.execute("SELECT editor_line, status FROM drafts WHERE id='d1'").fetchone()
    assert row["editor_line"] == "the bear was a tax dodge"
    assert row["status"] == "approved"  # unchanged
    assert review.stuck_captures(conn) == []  # no longer stuck


def test_reopen_callback_sends_draft_back_to_pending(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_rejected(conn, "d1")
    stub = tg.StubTelegram()
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "reopen:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res is None
    # status flipped back to pending
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "pending"
    # decision fields cleared
    row = conn.execute(
        "SELECT reviewed_at, editor_line, editor_notes FROM drafts WHERE id='d1'"
    ).fetchone()
    assert row["reviewed_at"] is None and row["editor_line"] is None
    assert any("back to pending" in m["text"] for m in stub.sent)


def test_reopen_callback_when_not_rejected_is_safe(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")  # already pending -> nothing to reopen
    stub = tg.StubTelegram()
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "reopen:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res is None
    assert any("Cannot reopen" in m["text"] for m in stub.sent)


def test_reopen_draft_removes_from_queue(tmp_path):
    """A reopened approved+queued draft must leave the publish queue."""
    import humorhist.review as review

    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)  # d1 approved + queued
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone() is not None
    review.reopen_draft(conn, "d1")
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "pending"
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone() is None


def test_listrejected_command_dispatches(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_rejected(conn, "d1")
    stub = tg.StubTelegram(updates=[
        {"update_id": 1, "message": {"message_id": 9, "chat": {"id": "chat"},
                                     "text": "/listrejected"}},
    ])
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    assert any("Pastry War" in m["text"] for m in stub.sent)


def test_send_copy_content_carries_edit_regen_buttons(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)
    stub = tg.StubTelegram()
    tg.send_copy_content(conn, stub, "chat", "d1")
    kb = stub.sent[-1]["reply_markup"]["inline_keyboard"][0]
    cbs = {b["callback_data"] for b in kb}
    assert "editcopy:d1" in cbs
    assert "regencopy:d1" in cbs
    assert "France invaded" in stub.sent[-1]["text"]


def test_regencopy_without_llm_sends_clean_unavailable(tmp_path, monkeypatch):
    """A Telegram-only user with no LLM key must get a friendly message, not a
    raw traceback, when they tap Regenerate copy."""
    from humorhist.llm import LLMUnavailable

    def _boom():
        raise LLMUnavailable("no key")

    monkeypatch.setattr("humorhist.llm.resilient_client", _boom)
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)
    stub = tg.StubTelegram()
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "regencopy:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res is None  # no crash; action skipped gracefully
    assert any("LLM unavailable" in m["text"] for m in stub.sent)
    # the stored copy must be unchanged
    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    assert row["post_copy"] == "France invaded Mexico over a pastry shop."


def test_editor_line_approve_without_llm_skips_copy_gracefully(tmp_path, monkeypatch):
    """Approve must still succeed (and enqueue) even when no LLM key exists for
    post-copy generation -- it just skips the copy step silently. Goes through
    the confirm gate first (GAP 3)."""
    from humorhist.llm import LLMUnavailable

    def _boom():
        raise LLMUnavailable("no key")

    monkeypatch.setattr("humorhist.llm.resilient_client", _boom)
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()
    # initial tap opens the confirm gate (no commit)
    tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "approve:d1",
                                            "message": {"message_id": 1}}},
    )
    # confirm tap commits
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 2, "callback_query": {"id": "cb2", "data": "confirm:approve:d1",
                                            "message": {"message_id": 1}}},
    )
    # draft approved + queued despite no LLM for the copy step
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "approved"
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone() is not None
    # the joke-prompt was sent (so the review flow continues normally)
    assert any("one-line joke" in m["text"] for m in stub.sent)



def test_enqueue_generates_story_image_and_persists(tmp_path, monkeypatch):
    """On enqueue (the publish step), story image gen must run when an image
    client is available, persist prompt + path on the queue row, and the
    generated PNG must exist on disk. Image generation is OFF the approve flow.
    """
    import humorhist.imagegen as ig
    from humorhist.llm import StubClient

    monkeypatch.setattr(
        "humorhist.llm.default_client",
        lambda: StubClient([{"prompt": "a wry period scene of the tax-dodge bear"}, {"post": "The bear was a tax dodge."}]),
    )
    img_client = ig.StubImageClient([b"\x89PNG\r\n\x1a\n fake-png-bytes"])
    monkeypatch.setattr("humorhist.imagegen.resilient_image_client", lambda: img_client)

    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    img_dir = tmp_path / "images"

    # approve first (apply_review auto-enqueues WITHOUT an image dir, so no image
    # is generated yet — it is deferred to the explicit publish/enqueue step)
    tg.handle_callback(conn, tg.StubTelegram(), "chat", _callback_update(1, "cb1", "approve", "d1"))
    cb_res = tg.handle_callback(conn, tg.StubTelegram(), "chat", _confirm_update(2, "cb2", "approve", "d1"))
    assert db.get_image(conn, "d1")["image_path"] is None

    # now the publish/enqueue step generates the image via the backfill path
    import humorhist.review as review

    review.enqueue_approved(conn, image_dir=str(img_dir))

    info = db.get_image(conn, "d1")
    assert info is not None
    assert info["image_prompt"] == "a wry period scene of the tax-dodge bear"
    assert info["image_path"] and info["image_path"].endswith("d1.png")
    assert Path(info["image_path"]).is_file()


def test_enqueue_sets_learn_more_source_link(tmp_path, monkeypatch):
    """Enqueue must persist a 'learn more' shortened link from the pool source_url."""
    from humorhist.imagegen import ImageUnavailable
    from humorhist.llm import StubClient

    monkeypatch.setattr("humorhist.llm.default_client", lambda: StubClient([{"post": "x"}]))
    # No image client available: enqueue must still set the link, just skip image.
    def _no_img():
        raise ImageUnavailable("no key")
    monkeypatch.setattr("humorhist.imagegen.resilient_image_client", _no_img)

    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    conn.execute("UPDATE drafts SET status='approved' WHERE id='d1'")
    conn.execute("UPDATE pool SET source_url='https://en.wikipedia.org/wiki/The_Great_Emu_War', source_name='Wikipedia'")
    conn.commit()
    import humorhist.review as review

    review.enqueue_approved(conn, image_dir=None)
    link = db.get_source_link(conn, "d1")
    assert link == "https://en.wikipedia.org/wiki/The_Great_Emu_War"



def test_approve_skips_image_when_no_image_client(tmp_path, monkeypatch):
    """When no image credential is available, approve must still succeed and NOT
    send a photo -- image generation now happens at enqueue (publish), so an
    approve with no image key simply defers the (skipped) image to that step."""
    from humorhist.imagegen import ImageUnavailable
    from humorhist.llm import StubClient

    def _no_img():
        raise ImageUnavailable("no key")

    monkeypatch.setattr("humorhist.llm.default_client", lambda: StubClient([{"post": "x"}]))
    monkeypatch.setattr("humorhist.imagegen.resilient_image_client", _no_img)

    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()
    # initial tap opens the confirm gate; confirm commits + prompts the joke
    tg.handle_callback(conn, stub, "chat", _callback_update(1, "cb1", "approve", "d1"))
    cb_res = tg.handle_callback(conn, stub, "chat", _confirm_update(2, "cb2", "approve", "d1"))
    awaiting = {"d1": {"note_message_id": cb_res["note_message_id"], "stage": "editor_line", "decision": "approve"}}

    tg.handle_text(
        conn, stub, "chat", awaiting, _text_update(3, "The bear was a tax dodge.", cb_res["note_message_id"]),
        image_dir=str(tmp_path / "images"),
    )
    # no photo sent
    assert not any("photo" in m for m in stub.sent)
    # queue row exists but no image persisted
    info = db.get_image(conn, "d1")
    assert info is not None
    assert info["image_path"] is None
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "approved"


def test_callback_copy_opens_copy_content(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)
    stub = tg.StubTelegram()
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "copy:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res is None
    assert any("POST COPY" in m["text"] for m in stub.sent)


def test_callback_editcopy_prompts_and_handle_text_saves(tmp_path):
    from humorhist.copywriter import set_post_copy

    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)
    stub = tg.StubTelegram()
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "editcopy:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res["editcopy_message_id"] is not None
    prompt_id = res["editcopy_message_id"]
    awaiting = {"d1": {"editcopy_message_id": prompt_id}}

    out = tg.handle_text(
        conn, stub, "chat", awaiting,
        _text_update(2, "My hand-edited version.", prompt_id),
    )
    assert out.get("editcopy_saved") == "d1"
    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    assert row["post_copy"] == "My hand-edited version."
    assert "d1" not in awaiting


def test_callback_editcopy_cancel_keeps_copy(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)
    stub = tg.StubTelegram()
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "editcopy:d1",
                                            "message": {"message_id": 1}}},
    )
    awaiting = {"d1": {"editcopy_message_id": res["editcopy_message_id"]}}
    out = tg.handle_text(
        conn, stub, "chat", awaiting,
        _text_update(2, "/cancel", res["editcopy_message_id"]),
    )
    assert out.get("editcopy_cancelled") == "d1"
    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    assert row["post_copy"] == "France invaded Mexico over a pastry shop."


def test_callback_regencopy_regenerates(tmp_path, monkeypatch):
    from humorhist.copywriter import fill_post_copy
    from humorhist.llm import StubClient

    real_fill = fill_post_copy

    def fake_fill(conn, client, draft_id=None, limit=None, **kwargs):
        return real_fill(
            conn, StubClient([{"post": "A regenerated pastry-war quip."}]),
            draft_id=draft_id, **kwargs,
        )

    monkeypatch.setattr("humorhist.copywriter.fill_post_copy", fake_fill)
    import humorhist.llm as llm

    monkeypatch.setattr(llm, "default_client", lambda: StubClient([{"post": "x"}]))

    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)
    stub = tg.StubTelegram()
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "regencopy:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res is None
    row = conn.execute("SELECT post_copy FROM queue WHERE draft_id='d1'").fetchone()
    assert row["post_copy"] == "A regenerated pastry-war quip."


def test_dispatch_listqueue_and_viewcopy(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)
    stub = tg.StubTelegram()
    tg.send_queue_list(conn, stub, "chat")
    assert any("Pastry War" in m["text"] for m in stub.sent)
    stub.sent.clear()
    tg.send_copy_content(conn, stub, "chat", "d1")
    assert any("POST COPY" in m["text"] for m in stub.sent)


# --- /later (defer a pending draft) ----------------------------------------


def test_handle_callback_later_defers_draft(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram()
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "later:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res and res.get("deferred") is True
    row = conn.execute("SELECT defer_until, status FROM drafts WHERE id='d1'").fetchone()
    assert row["defer_until"] is not None
    assert row["status"] == "pending"  # still reviewable later
    assert any("deferred" in m["text"].lower() for m in stub.sent)


def test_run_review_bot_later_command(tmp_path):
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    stub = tg.StubTelegram(updates=[
        {"update_id": 1, "message": {"message_id": 9, "chat": {"id": "chat"}, "text": "/later d1"}},
    ])
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    row = conn.execute("SELECT defer_until FROM drafts WHERE id='d1'").fetchone()
    assert row["defer_until"] is not None
    assert any("deferred 30 days" in m["text"] for m in stub.sent)


# --- /suggest (editor-submitted pool candidate) ----------------------------


def test_run_review_bot_suggest_command(tmp_path):
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram(updates=[
        {"update_id": 1, "message": {"message_id": 9, "chat": {"id": "chat"},
                                      "text": "/suggest The Dancing Plague of 1518"}},
    ])
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    row = conn.execute("SELECT title, status, source_name FROM pool WHERE title='The Dancing Plague of 1518'").fetchone()
    assert row is not None
    assert row["status"] == "new"
    assert row["source_name"] == "editor-suggestion"
    assert any("Suggested" in m["text"] for m in stub.sent)


# --- notes -> angle regeneration (pending draft) ---------------------------


def test_notes_on_pending_draft_regenerates_angles(tmp_path, monkeypatch):
    import json
    import humorhist.llm as llm
    from humorhist.llm import StubClient

    # A minimal valid angles payload (3 angles) for the stub to return.
    def _angles():
        return {
            "angles": [
                {"angle_name": "A", "setup": "s", "why_it_lands": "w",
                 "pitfalls": "p", "raw_material": ["r"]},
                {"angle_name": "B", "setup": "s", "why_it_lands": "w",
                 "pitfalls": "p", "raw_material": ["r"]},
                {"angle_name": "C", "setup": "s", "why_it_lands": "w",
                 "pitfalls": "p", "raw_material": ["r"]},
            ],
            "strongest_single_detail": "d",
            "suggested_hook": "h",
        }

    conn = _fresh_db(tmp_path)
    # seed a PENDING draft with a brief + angles so regeneration has something
    conn.execute("INSERT OR IGNORE INTO pool (id, title, status) VALUES ('pool-x','Emu War','drafted')")
    conn.execute(
        "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) "
        "VALUES ('d1','pool-x',?,?,'pending','2026-01-01T00:00:00+00:00')",
        (json.dumps({"verified_facts": ["x"]}), json.dumps(_angles())),
    )
    conn.commit()

    monkeypatch.setattr(llm, "resilient_client", lambda: StubClient([_angles()]))
    stub = tg.StubTelegram()
    res = tg.handle_callback(
        conn, stub, "chat",
        {"update_id": 1, "callback_query": {"id": "cb1", "data": "notes:d1",
                                            "message": {"message_id": 1}}},
    )
    assert res and res.get("regenerate_angles") is True
    note_msg_id = res["note_message_id"]
    awaiting = {"d1": {"note_message_id": note_msg_id, "stage": "notes",
                        "decision": "approve", "regenerate_angles": True}}
    out = tg.handle_text(
        conn, stub, "chat", awaiting,
        _text_update(2, "lean into bureaucracy", note_msg_id),
    )
    assert out.get("angles_regenerated") == "d1"
    row = conn.execute("SELECT editor_notes FROM drafts WHERE id='d1'").fetchone()
    assert row["editor_notes"] == "lean into bureaucracy"
    # draft stays pending (no review decision made)
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "pending"


# --- discovery/draft commands (/harvest, /screen, /draft) ------------------


def test_harvest_command_runs_and_reports(tmp_path, monkeypatch):
    conn = _fresh_db(tmp_path)
    monkeypatch.setattr(
        "humorhist.harvest.seed.load_seed", lambda c: 3, raising=False
    )
    monkeypatch.setattr(
        "humorhist.harvest.wikipedia_lists.harvest_wikipedia_lists",
        lambda c: 5,
        raising=False,
    )
    stub = tg.StubTelegram(updates=_message_batch("/harvest", 1))
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    assert any("Harvesting new events" in m["text"] for m in stub.sent)
    assert any("Harvest done" in m["text"] and "Pool now" in m["text"] for m in stub.sent)


def test_screen_command_skips_without_llm(tmp_path, monkeypatch):
    from humorhist.llm import LLMUnavailable

    def _boom():
        raise LLMUnavailable("no key")

    monkeypatch.setattr("humorhist.llm.resilient_client", _boom)
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram(updates=_message_batch("/screen", 1))
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    assert any("LLM unavailable" in m["text"] for m in stub.sent)


def test_draft_command_runs_and_reports(tmp_path, monkeypatch):
    import json
    import humorhist.llm as llm
    from humorhist.llm import StubClient

    # a draft_candidates stub: we don't call the LLM in this test, but the
    # command path fetches a client first, so give it one.
    monkeypatch.setattr(llm, "resilient_client", lambda: StubClient([{}]))

    def fake_draft(conn2, client, count=3, min_score=7.0, **kw):
        conn2.execute(
            "INSERT OR IGNORE INTO pool (id, title, status) VALUES ('p1','X','drafted')"
        )
        conn2.execute(
            "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) "
            "VALUES ('nd1','p1','{}','{}','pending','2026-01-01T00:00:00+00:00')"
        )
        conn2.commit()
        return {"selected": 1, "drafted": 1, "failed": 0, "draft_ids": ["nd1"], "failures": []}

    monkeypatch.setattr("humorhist.drafting.draft_candidates", fake_draft)
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram(updates=_message_batch("/draft 2", 1))
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    assert any("Drafting 2 candidate(s)" in m["text"] for m in stub.sent)
    assert any("Drafted 1 new draft(s)" in m["text"] for m in stub.sent)
    assert conn.execute("SELECT status FROM drafts WHERE id='nd1'").fetchone()["status"] == "pending"


# --- /buffer command + proactive nudge ------------------------------------


def test_buffer_command_reports_health(tmp_path, monkeypatch):
    import humorhist.llm as llm
    from humorhist.llm import StubClient

    monkeypatch.setattr(llm, "resilient_client", lambda: StubClient([{}]))
    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram(updates=_message_batch("/buffer", 1))
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    assert any("BUFFER" in m["text"] for m in stub.sent)


def test_buffer_enqueue_command_sweeps_approved(tmp_path, monkeypatch):
    import humorhist.llm as llm
    from humorhist.llm import StubClient

    monkeypatch.setattr(llm, "resilient_client", lambda: StubClient([{}]))
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    # approve it so it becomes enqueue-able
    conn.execute("UPDATE drafts SET status='approved' WHERE id='d1'")
    conn.commit()
    stub = tg.StubTelegram(updates=_message_batch("/buffer enqueue", 1))
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    assert any("Enqueued 1 approved draft(s)" in m["text"] for m in stub.sent)
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone() is not None


def test_proactive_nudge_on_new_pending_draft(tmp_path, monkeypatch):
    """When a new draft appears between poll cycles, the bot nudges once."""
    import humorhist.llm as llm
    from humorhist.llm import StubClient

    monkeypatch.setattr(llm, "resilient_client", lambda: StubClient([{}]))

    def fake_draft(conn2, client, count=3, min_score=7.0, **kw):
        conn2.execute(
            "INSERT OR IGNORE INTO pool (id, title, status) VALUES ('p1','X','drafted')"
        )
        conn2.execute(
            "INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) "
            "VALUES ('nd1','p1','{}','{}','pending','2026-01-01T00:00:00+00:00')"
        )
        conn2.commit()
        return {"selected": 1, "drafted": 1, "failed": 0, "draft_ids": ["nd1"], "failures": []}

    monkeypatch.setattr("humorhist.drafting.draft_candidates", fake_draft)

    conn = _fresh_db(tmp_path)
    stub = tg.StubTelegram(
        updates=[
            *_message_batch("/draft", 1),
            {"update_id": 2, "message": {"message_id": 102, "chat": {"id": "chat"}, "text": "/status"}},
            {"update_id": 3, "message": {"message_id": 103, "chat": {"id": "chat"}, "text": "/noop"}},
        ]
    )
    tg.run_review_bot(conn, stub, "chat", max_iterations=5, poll_timeout=1)
    assert any("🆕 1 new draft(s) awaiting review" in m["text"] for m in stub.sent)


# --- /queue command ---------------------------------------------------------


def test_queue_command_lists_queue(tmp_path, monkeypatch):
    import humorhist.llm as llm
    from humorhist.llm import StubClient

    monkeypatch.setattr(llm, "resilient_client", lambda: StubClient([{}]))
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)
    stub = tg.StubTelegram(updates=_message_batch("/queue", 1))
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    assert any("Queued drafts" in m["text"] for m in stub.sent)


def test_queue_enqueue_sweeps_approved(tmp_path, monkeypatch):
    import humorhist.llm as llm
    from humorhist.llm import StubClient

    monkeypatch.setattr(llm, "resilient_client", lambda: StubClient([{}]))
    conn = _fresh_db(tmp_path)
    _seed_pending(conn, "d1")
    conn.execute("UPDATE drafts SET status='approved' WHERE id='d1'")
    conn.commit()
    stub = tg.StubTelegram(updates=_message_batch("/queue enqueue", 1))
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    assert any("Enqueued 1 approved draft(s)" in m["text"] for m in stub.sent)
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone() is not None


def test_queue_remove_pulls_draft_back(tmp_path, monkeypatch):
    import humorhist.llm as llm
    from humorhist.llm import StubClient

    monkeypatch.setattr(llm, "resilient_client", lambda: StubClient([{}]))
    conn = _fresh_db(tmp_path)
    _seed_approved_queued(conn)
    stub = tg.StubTelegram(updates=_message_batch("/queue remove d1", 1))
    tg.run_review_bot(conn, stub, "chat", max_iterations=5)
    assert any("Removed `d1` from the queue" in m["text"] for m in stub.sent)
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone() is None
    # draft stays approved (not deleted)
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()["status"] == "approved"


