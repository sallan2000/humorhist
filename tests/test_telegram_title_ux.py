"""Offline Telegram title-first interaction and message-boundary regressions."""
import socket

import pytest

import humorhist.db as db
import humorhist.review as review
import humorhist.telegram as tg


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access forbidden")
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "ux.sqlite"))
    db.migrate(c)
    return c


def seed(conn, did, code, title="Duplicate title", status="pending", created="2026-01-01"):
    conn.execute("INSERT INTO pool (id,title,status) VALUES (?,?,?)", ("pool-" + did, title, "drafted"))
    conn.execute("INSERT INTO drafts (id,pool_id,short_code,brief_json,angles_json,status,created_at) VALUES (?,?,?,'{}','{}',?,?)", (did, "pool-" + did, code, status, created))
    conn.commit()


def message(i, text, sender=123):
    return {"update_id": i, "message": {"text": text, "chat": {"id": 123, "type": "private"}, "from": {"id": sender}}}


def callback(i, data, sender=123):
    return {"update_id": i, "callback_query": {"id": str(i), "data": data, "from": {"id": sender}, "message": {"chat": {"id": 123, "type": "private"}}}}


def run(conn, updates):
    stub = tg.StubTelegram(updates)
    tg.run_review_bot(conn, stub, "123", once=True)
    return stub


def test_short_full_and_invalid_commands_are_resolved_safely(conn):
    seed(conn, "full-id-1", "ABCDE")
    stub = run(conn, [message(1, "/view #ABCDE"), message(2, "/later ABCDE"), message(3, "/reviewnow full-id-1"), message(4, "/later missing"), message(5, "/later ABCDE extra")])
    text = "\n".join(m.get("text", "") for m in stub.sent)
    assert "Duplicate title (#ABCDE)" in text
    assert "No such draft" in text and "exactly one" in text
    assert conn.execute("SELECT defer_until FROM drafts WHERE id='full-id-1'").fetchone()[0] is None


def test_collision_uses_bare_id_and_hash_short_code(conn):
    seed(conn, "collision", "FIRST")
    seed(conn, "other", "collision")
    assert tg._resolve_draft(conn, "collision") == "collision"
    assert tg._resolve_draft(conn, "#collision") == "other"
    stub = run(conn, [message(1, "/later FIRST", sender=456), callback(2, "confirm:reject:collision", sender=456)])
    assert all(r[0] == "pending" and r[1] is None for r in conn.execute("SELECT status,defer_until FROM drafts"))
    assert "2" not in stub.answered


def test_next_wraps_without_mutation_and_excludes_deferred(conn):
    for n in range(3):
        seed(conn, f"id{n}", f"CODE{n}", created=f"2026-01-0{n+1}")
    review.defer_draft(conn, "id1")
    stub = run(conn, [callback(1, "next:id0"), callback(2, "next:id2")])
    buttons = [m["reply_markup"]["inline_keyboard"][0][0]["callback_data"] for m in stub.sent if m.get("reply_markup") and m["reply_markup"]["inline_keyboard"][0][0]["callback_data"].startswith("approve:")]
    assert [b.split(":")[1] for b in buttons[-2:]] == ["id2", "id0"]
    assert [r[0] for r in conn.execute("SELECT status FROM drafts ORDER BY id")] == ["pending"] * 3
    assert {r["id"] for r in review.pending_drafts(conn)} == {"id0", "id2"}
    assert {"1", "2"} <= stub.answered


def test_long_unicode_and_context_on_actionable_final_message(conn):
    seed(conn, "full-id", "SHORT", title="😀" * 1000)
    conn.execute("UPDATE drafts SET brief_json=?", ('{"verified_facts":["' + '😀' * 6000 + '"]}',))
    conn.commit()
    stub = tg.StubTelegram()
    tg.send_pending_drafts(conn, stub, "123")
    assert len(stub.sent) > 2
    assert all(tg._units(m["text"]) <= 4096 and m["text"] for m in stub.sent)
    assert "#SHORT" in stub.sent[-1]["text"]
    assert stub.sent[-1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]  .startswith("approve:full-id:")
    raw = "😀" * 10000
    assert "".join(tg._chunk_text(raw)) == raw


def test_paginated_lists_preserve_all_buttons_and_duplicate_titles(conn):
    for n in range(42):
        seed(conn, f"draft-{n:03}", f"S{n:03}", title="Same " + "😀" * 150, status="approved")
    stub = tg.StubTelegram()
    assert tg.send_approved_list(conn, stub, "123") == 42
    assert len(stub.sent) > 1
    callbacks = []
    for m in stub.sent:
        assert tg._units(m["text"]) <= 4096
        keys = m["reply_markup"]["inline_keyboard"]
        assert len(keys) <= 16
        for buttons in keys:
            callbacks.extend(b["callback_data"] for b in buttons)
    assert set(x for x in callbacks if x.startswith("view:")) == {f"view:draft-{n:03}" for n in range(42)}
    assert all(len(x.encode()) <= 64 for x in callbacks)


def test_confirm_uses_full_id_and_correct_title(conn):
    seed(conn, "long-full-uuid", "ABCDE", title="The emu war")
    stub = tg.StubTelegram()
    token = tg._decision_token(conn, "long-full-uuid")
    tg.handle_callback(conn, stub, "123", callback(1, f"reject:long-full-uuid:{token}"))
    assert "The emu war (#ABCDE)" in stub.sent[-1]["text"]
    confirm = stub.sent[-1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    assert confirm.startswith("confirm:reject:long-full-uuid:")
    assert conn.execute("SELECT status FROM drafts").fetchone()[0] == "pending"
    tg.handle_callback(conn, stub, "123", callback(2, confirm))
    assert conn.execute("SELECT status FROM drafts").fetchone()[0] == "rejected"


def test_stale_prompt_cannot_reapprove_rejected_draft(conn):
    seed(conn, "draft", "SHORT", status="approved")
    stub = tg.StubTelegram()
    entry = tg._open_notes_prompt(conn, stub, "123", "draft")
    review.apply_review(conn, "draft", decision="reject")
    awaiting = {"draft": entry}
    result = tg.handle_text(conn, stub, "123", awaiting, message(2, "accidental note"))
    assert result == {"expired": "draft"}
    assert conn.execute("SELECT status FROM drafts").fetchone()[0] == "rejected"
    assert not awaiting


def test_fast_buttons_do_not_disable_normal_confirmation(conn):
    seed(conn, "draft", "SHORT")
    stub = tg.StubTelegram()
    token = tg._decision_token(conn, "draft")
    assert tg._keyboard("draft", fast=True, token=token)["inline_keyboard"][0][0]["callback_data"] == f"fastapprove:draft:{token}"
    result = tg.handle_callback(conn, stub, "123", callback(1, f"approve:draft:{token}"))
    assert result["confirming"]
    assert conn.execute("SELECT status FROM drafts").fetchone()[0] == "pending"
    tg.handle_callback(conn, stub, "123", callback(2, f"fastreject:draft:{token}"))
    assert conn.execute("SELECT status FROM drafts").fetchone()[0] == "rejected"


def test_queue_pages_bind_correct_actions(conn):
    for n in range(21):
        did = f"d{n:02}"
        seed(conn, did, f"S{n:02}", status="approved")
        conn.execute("INSERT INTO queue (draft_id,published,post_copy) VALUES (?,0,?)", (did, "😀" * 300))
    conn.commit()
    stub = tg.StubTelegram()
    assert tg.send_queue_list(conn, stub, "123") == 21
    assert len(stub.sent) >= 3
    actions = []
    for m in stub.sent:
        assert tg._units(m["text"]) <= 4096
        for row in m["reply_markup"]["inline_keyboard"]:
            actions.extend(b["callback_data"] for b in row)
    for n in range(21):
        for action in ("copy", "remove", "reject", "reopen"):
            assert any(a == f"{action}:d{n:02}" or a.startswith(f"{action}:d{n:02}:") for a in actions)


def test_stale_fast_and_confirmation_do_not_reverse_or_erase(conn):
    seed(conn, "d1", "ONE")
    stub = tg.StubTelegram()
    token = tg._decision_token(conn, "d1")
    tg.handle_callback(conn, stub, "123", callback(1, f"approve:d1:{token}"))
    confirm = stub.sent[-1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    review.apply_review(conn, "d1", decision="approve", editor_line="joke", notes="notes")
    for data in (confirm, f"fastreject:d1:{token}", f"reject:d1:{token}", "confirm:reject:d1"):
        tg.handle_callback(conn, stub, "123", callback(2, data))
    row = conn.execute("SELECT status,editor_line,editor_notes FROM drafts WHERE id='d1'").fetchone()
    assert tuple(row) == ("approved", "joke", "notes")


def test_deferred_draft_reject_is_blocked_at_both_steps(conn):
    seed(conn, "d1", "ONE")
    stub = tg.StubTelegram()
    token = tg._decision_token(conn, "d1")
    tg.handle_callback(conn, stub, "123", callback(1, f"reject:d1:{token}"))
    confirm = stub.sent[-1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    review.defer_draft(conn, "d1")
    for data in (f"fastreject:d1:{token}", f"reject:d1:{token}", confirm):
        tg.handle_callback(conn, stub, "123", callback(2, data))
    row = conn.execute("SELECT status,defer_until FROM drafts WHERE id='d1'").fetchone()
    assert row["status"] == "pending" and row["defer_until"]
    stub = tg.StubTelegram()
    tg.run_review_session(conn, stub, "123", {}, [0])
    assert any("/listlater" in m.get("text", "") for m in stub.sent)


def test_explicit_unmatched_reply_never_writes_to_other_prompt(conn):
    seed(conn, "d1", "ONE", status="approved")
    conn.execute("INSERT INTO queue (draft_id,published,post_copy) VALUES ('d1',0,'original')")
    stub = tg.StubTelegram()
    awaiting = {"d1": {"editcopy_message_id": 42}}
    update = message(1, "wrong copy")
    update["message"]["reply_to_message"] = {"message_id": 123456}
    assert tg.handle_text(conn, stub, "123", awaiting, update) is None
    assert conn.execute("SELECT post_copy FROM queue").fetchone()[0] == "original"
    assert awaiting["d1"]["editcopy_message_id"] == 42


def test_title_chunk_boundaries_url_and_publish_totals(conn):
    seed(conn, "d1", "ONE", title="😀" * 3000, status="approved")
    seed(conn, "d2", "TWO", status="approved")
    for did in ("d1", "d2"):
        conn.execute("INSERT INTO queue (draft_id,published,post_copy) VALUES (?,0,?)", (did, "copy"))
    conn.execute("UPDATE queue SET source_link=? WHERE draft_id='d1'", ("https://example.org/" + "x" * 600,))
    conn.commit()
    stub = tg.StubTelegram()
    tg.send_draft_content(conn, stub, "123", "d1")
    assert "#ONE" in stub.sent[0]["text"]
    stub = tg.StubTelegram()
    tg.send_queue_list(conn, stub, "123")
    assert "link omitted" in "\n".join(m["text"] for m in stub.sent)
    stub = tg.StubTelegram()
    tg.telegram_publish(conn, stub, "123")
    assert "Post 1 of 2" in stub.sent[0]["text"] and "Post 2 of 2" in stub.sent[1]["text"]
    text = ("word " * 1000) + ("😀" * 5000)
    chunks = tg._chunk_text(text, 100)
    assert "".join(chunks) == text
    assert chunks[0].endswith(" ") and all(tg._units(c) <= 100 for c in chunks)


def test_stale_skipnotes_does_not_clear_new_edit_prompt(conn):
    seed(conn, "d1", "ONE", status="approved")
    stub = tg.StubTelegram()
    awaiting = {"d1": {"editcopy_message_id": 22}}
    upd = callback(1, "skipnotes:d1")
    upd["callback_query"]["message"]["message_id"] = 10
    assert tg.handle_callback(conn, stub, "123", upd, awaiting=awaiting) is None
    assert awaiting == {"d1": {"editcopy_message_id": 22}}
    assert "expired" in stub.sent[-1]["text"]
    awaiting["d1"] = {"stage": "notes", "note_message_id": 10}
    assert tg.handle_callback(conn, stub, "123", upd, awaiting=awaiting) == {"skipped": "d1"}


def test_notes_prompt_is_not_a_decision(conn):
    seed(conn, "d1", "ONE", status="approved")
    stub = tg.StubTelegram([callback(1, "notes:d1")])
    assert tg.run_review_bot(conn, stub, "123", once=True) == 0
    assert any("Notes for already-approved" in m.get("text", "") for m in stub.sent)


def test_callback_byte_limits_for_uuid_ids(conn):
    did = "12345678-1234-1234-1234-123456789012"
    seed(conn, did, "ONE")
    stub = tg.StubTelegram()
    tg.send_pending_drafts(conn, stub, "123")
    buttons = stub.sent[-1]["reply_markup"]["inline_keyboard"]
    for row in buttons:
        assert all(len(b["callback_data"].encode("utf-8")) <= 64 for b in row)
    token = tg._decision_token(conn, did)
    tg.handle_callback(conn, stub, "123", callback(1, f"approve:{did}:{token}"))
    confirm = stub.sent[-1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    assert len(confirm.encode("utf-8")) <= 64


def test_old_fast_button_expires_after_approve_and_reopen(conn):
    seed(conn, "d1", "ONE")
    old = tg._decision_token(conn, "d1")
    review.apply_review(conn, "d1", decision="approve")
    review.reopen_draft(conn, "d1")
    stub = tg.StubTelegram()
    tg.handle_callback(conn, stub, "123", callback(1, f"fastreject:d1:{old}"))
    assert conn.execute("SELECT status FROM drafts").fetchone()[0] == "pending"
    assert "expired" in stub.sent[-1]["text"]


def test_old_fast_button_expires_after_other_connection_aba(conn):
    seed(conn, "d1", "ONE")
    old = tg._decision_token(conn, "d1")
    path = conn.execute("PRAGMA database_list").fetchone()[2]
    other = db.connect(path)
    review.apply_review(other, "d1", decision="approve")
    review.reopen_draft(other, "d1")
    other.close()
    stub = tg.StubTelegram()
    tg.handle_callback(conn, stub, "123", callback(1, f"fastreject:d1:{old}"))
    assert conn.execute("SELECT status FROM drafts").fetchone()[0] == "pending"
    assert "expired" in stub.sent[-1]["text"]


def test_oversized_list_entry_keeps_keyboard_and_context():
    stub = tg.StubTelegram()
    item = "  • Long topic (#CODE)\n" + "😀" * 5000
    tg._send_list_pages(stub, "123", ["Header", item], [[{"text": "Open", "callback_data": "view:d1"}]])
    assert len(stub.sent) > 1
    assert all(tg._units(m["text"]) <= 4096 for m in stub.sent)
    assert "Long topic (#CODE)" in stub.sent[-1]["text"]
    assert stub.sent[-1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "view:d1"


@pytest.mark.parametrize("action", ["remove", "reopen", "later", "reviewnow", "regencopy", "editcopy"])
def test_unversioned_destructive_callbacks_are_rejected(conn, action):
    seed(conn, "d1", "ONE", status="approved" if action != "later" else "pending")
    before = tuple(conn.execute("SELECT status,editor_line,editor_notes,defer_until FROM drafts").fetchone())
    stub = tg.StubTelegram()
    assert tg.handle_callback(conn, stub, "123", callback(1, f"{action}:d1")) is None
    assert tuple(conn.execute("SELECT status,editor_line,editor_notes,defer_until FROM drafts").fetchone()) == before
    assert "open it again" in stub.sent[-1]["text"]


def test_stale_notes_after_approved_reopened_approved_preserves_new_work(conn):
    seed(conn, "d1", "ONE", status="approved")
    stub = tg.StubTelegram()
    entry = tg._open_notes_prompt(conn, stub, "123", "d1")
    review.reopen_draft(conn, "d1")
    review.apply_review(conn, "d1", decision="approve", editor_line="new joke", notes="new note")
    result = tg.handle_text(conn, stub, "123", {"d1": entry}, message(2, "old note"))
    assert result == {"expired": "d1"}
    assert tuple(conn.execute("SELECT status,editor_line,editor_notes FROM drafts").fetchone()) == (
        "approved", "new joke", "new note")


@pytest.mark.parametrize("action", ["remove", "reopen"])
def test_old_queue_button_does_not_touch_new_approval(conn, action):
    seed(conn, "d1", "ONE", status="approved")
    conn.execute("INSERT INTO queue (draft_id,published,post_copy) VALUES ('d1',0,'old')")
    conn.commit()
    old = tg._decision_token(conn, "d1")
    review.reopen_draft(conn, "d1")
    review.apply_review(conn, "d1", decision="approve", editor_line="new joke", notes="new notes")
    stub = tg.StubTelegram()
    tg.handle_callback(conn, stub, "123", callback(1, f"{action}:d1:{old}"))
    assert tuple(conn.execute("SELECT status,editor_line,editor_notes FROM drafts").fetchone()) == (
        "approved", "new joke", "new notes")
    assert conn.execute("SELECT 1 FROM queue WHERE draft_id='d1'").fetchone()


def test_old_later_button_cannot_defer_new_pending_cycle(conn):
    seed(conn, "d1", "ONE")
    old = tg._decision_token(conn, "d1")
    review.apply_review(conn, "d1", decision="reject")
    review.reopen_draft(conn, "d1")
    stub = tg.StubTelegram()
    tg.handle_callback(conn, stub, "123", callback(1, f"later:d1:{old}"))
    assert conn.execute("SELECT defer_until FROM drafts").fetchone()[0] is None


def test_unrelated_changes_do_not_expire_card_or_prompt(conn):
    seed(conn, "d1", "ONE")
    seed(conn, "d2", "TWO")
    token = tg._decision_token(conn, "d1")
    review.defer_draft(conn, "d2")
    assert tg._decision_token(conn, "d1") == token
    stub = tg.StubTelegram()
    tg.handle_callback(conn, stub, "123", callback(1, f"fastreject:d1:{token}"))
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()[0] == "rejected"


def test_version_persists_across_bot_restart_and_external_writes(conn):
    seed(conn, "d1", "ONE")
    token = tg._decision_token(conn, "d1")
    path = conn.execute("PRAGMA database_list").fetchone()[2]
    other = db.connect(path)
    assert tg._decision_token(other, "d1") == token
    review.apply_review(other, "d1", decision="reject")
    review.reopen_draft(other, "d1")
    other.close()
    assert tg._decision_token(conn, "d1") != token


def test_version_schema_preserves_caller_transaction(conn):
    seed(conn, "d1", "ONE")
    conn.execute("UPDATE pool SET title='uncommitted' WHERE id='pool-d1'")
    assert conn.in_transaction
    tg._decision_token(conn, "d1")
    assert conn.in_transaction
    conn.rollback()
    assert conn.execute("SELECT title FROM pool").fetchone()[0] == "Duplicate title"
    assert tg._decision_token(conn, "d1")


def test_very_long_source_url_is_omitted_not_split(conn):
    seed(conn, "d1", "ONE", status="approved")
    url = "https://example.org/" + "x" * 5000
    conn.execute("INSERT INTO queue (draft_id,published,post_copy,source_link) VALUES ('d1',0,'copy',?)", (url,))
    conn.commit()
    stub = tg.StubTelegram()
    tg.send_draft_content(conn, stub, "123", "d1")
    text = "\n".join(m.get("text", "") for m in stub.sent)
    assert "URL omitted: too long" in text
    assert "https://example.org/" not in text
    stub = tg.StubTelegram()
    tg.telegram_publish(conn, stub, "123")
    assert "URL omitted: too long" in "\n".join(m.get("text", "") for m in stub.sent)


def test_unrelated_write_does_not_expire_editorial_prompt(conn):
    seed(conn, "d1", "ONE", status="approved")
    seed(conn, "d2", "TWO")
    stub = tg.StubTelegram()
    entry = tg._open_notes_prompt(conn, stub, "123", "d1")
    review.defer_draft(conn, "d2")
    result = tg.handle_text(conn, stub, "123", {"d1": entry}, message(1, "fresh note"))
    assert result == {"noted": "d1"}
    assert conn.execute("SELECT editor_notes FROM drafts WHERE id='d1'").fetchone()[0] == "fresh note"


def test_angle_regeneration_discards_result_if_draft_changes(conn, monkeypatch):
    import humorhist.brief as brief
    seed(conn, "d1", "ONE")
    stub = tg.StubTelegram()
    entry = tg._open_notes_prompt(conn, stub, "123", "d1")
    monkeypatch.setattr(tg, "_get_llm", lambda *a, **k: object())
    def concurrent_edit(*args, **kwargs):
        conn.execute("UPDATE drafts SET editor_notes='newer' WHERE id='d1'")
        conn.commit()
        return {"angles": [{"setup": "stale"}]}
    monkeypatch.setattr(brief, "generate_angles", concurrent_edit)
    result = tg.handle_text(conn, stub, "123", {"d1": entry}, message(1, "older steer"))
    assert result == {"expired": "d1"}
    assert tuple(conn.execute("SELECT angles_json,editor_notes FROM drafts WHERE id='d1'").fetchone()) == ("{}", "newer")


def test_versioned_later_respects_caller_owned_transaction(conn):
    seed(conn, "d1", "ONE")
    conn.execute("UPDATE pool SET title='uncommitted' WHERE id='pool-d1'")
    token = tg._decision_token(conn, "d1")
    stub = tg.StubTelegram()
    result = tg.handle_callback(conn, stub, "123", callback(1, f"later:d1:{token}"))
    assert result is None and conn.in_transaction
    assert "uncommitted" in stub.sent[-1]["text"]
    conn.rollback()
    assert conn.execute("SELECT defer_until FROM drafts").fetchone()[0] is None
    assert conn.execute("SELECT title FROM pool").fetchone()[0] == "Duplicate title"
