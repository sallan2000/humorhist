"""Offline coverage for publication buttons and persistent draft clocks."""
import pytest

from humorhist import db, review
from humorhist import telegram as tg


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "db.sqlite"))
    db.migrate(c)
    for did in ("one", "two"):
        c.execute("INSERT INTO pool (id,title,status) VALUES (?,?,?)", (did, f"Title {did}", "drafted"))
        c.execute("INSERT INTO drafts (id,pool_id,short_code,brief_json,angles_json,status) "
                  "VALUES (?,?,?,'{}','{}','approved')", (did, did, did.upper()))
        c.execute("INSERT INTO queue (draft_id,published) VALUES (?,0)", (did,))
    c.commit()
    yield c
    c.close()


def tap(c, stub, data):
    tg.handle_callback(c, stub, "123", {"callback_query": {"id": "x", "data": data}})


def button(stub, prefix):
    return next(b["callback_data"] for m in stub.sent for row in m.get("reply_markup", {}).get("inline_keyboard", [])
                for b in row if b["callback_data"].startswith(prefix))


def test_confirmation_and_safe_undo_after_republication(conn):
    stub = tg.StubTelegram()
    tg.send_queue_list(conn, stub, "123")
    posted = button(stub, "posted:one:")
    tap(conn, stub, posted)
    assert "Title one (#ONE)" in stub.sent[-1]["text"]
    confirm = button(stub, "postyes:one:")
    assert conn.execute("SELECT published FROM queue WHERE draft_id='one'").fetchone()[0] == 0
    tap(conn, stub, confirm)
    timestamp = conn.execute("SELECT published_at FROM queue WHERE draft_id='one'").fetchone()[0]
    assert timestamp in stub.sent[-1]["text"]
    undo = button(stub, "unpost:one:")
    listing = tg.StubTelegram()
    tg.send_posted_list(conn, listing, "123")
    assert timestamp in listing.sent[0]["text"] and "Title one (#ONE)" in listing.sent[0]["text"]
    tap(conn, stub, undo)
    review.mark_posted(conn, "one")
    tap(conn, stub, undo)
    assert conn.execute("SELECT published FROM queue WHERE draft_id='one'").fetchone()[0] == 1
    tap(conn, stub, confirm)
    assert conn.execute("SELECT published FROM queue WHERE draft_id='one'").fetchone()[0] == 1
    assert "expired" in stub.sent[-1]["text"].lower()


def test_stale_confirmation_and_legacy_publication(conn):
    stub = tg.StubTelegram()
    token = tg._decision_token(conn, "one")
    tap(conn, stub, f"posted:one:{token}")
    confirm = button(stub, "postyes:one:")
    conn.execute("UPDATE queue SET post_copy='new' WHERE draft_id='one'")
    conn.commit()
    tap(conn, stub, confirm)
    assert conn.execute("SELECT published FROM queue WHERE draft_id='one'").fetchone()[0] == 0
    conn.execute("UPDATE queue SET published=1 WHERE draft_id='one'")
    conn.commit()
    listing = tg.StubTelegram()
    tg.send_posted_list(conn, listing, "123")
    assert "Legacy publication" in listing.sent[0]["text"]
    assert not any("unpost:" in str(m) for m in listing.sent)


def test_published_navigation_command(conn):
    stamp = review.mark_posted(conn, "one")
    update = {"update_id": 1, "message": {"text": "/listposted", "chat": {"id": 123, "type": "private"},
                                         "from": {"id": 123}}}
    stub = tg.StubTelegram([update])
    tg.run_review_bot(conn, stub, "123", once=True)
    assert any(stamp in m.get("text", "") and "Title one (#ONE)" in m["text"] for m in stub.sent)
    assert button(stub, "unpost:one:")


def test_draft_clock_survives_restart_and_ignores_other_records(conn, tmp_path):
    token = tg._decision_token(conn, "one")
    conn.commit()
    conn.execute("UPDATE queue SET post_copy='unrelated' WHERE draft_id='two'")
    conn.commit()
    other = db.connect(str(tmp_path / "db.sqlite"))
    try:
        assert tg._decision_token(other, "one") == token
        other.execute("UPDATE queue SET post_copy='changed' WHERE draft_id='one'")
        other.commit()
        assert tg._decision_token(conn, "one") != token
    finally:
        other.close()


@pytest.mark.parametrize("action,change", [
    ("remove", "UPDATE queue SET post_copy='new' WHERE draft_id='one'"),
    ("reopen", "UPDATE drafts SET editor_notes='new' WHERE id='one'"),
    ("later", "UPDATE drafts SET editor_notes='new' WHERE id='one'"),
])
def test_old_mutation_buttons_expire(conn, action, change):
    if action == "later":
        conn.execute("UPDATE drafts SET status='pending' WHERE id='one'")
    token = tg._decision_token(conn, "one")
    conn.execute(change)
    conn.commit()
    before = tuple(conn.execute("SELECT status,defer_until FROM drafts WHERE id='one'").fetchone())
    stub = tg.StubTelegram()
    tap(conn, stub, f"{action}:one:{token}")
    assert "expired" in stub.sent[-1]["text"].lower()
    assert tuple(conn.execute("SELECT status,defer_until FROM drafts WHERE id='one'").fetchone()) == before
    assert conn.execute("SELECT count(*) FROM queue WHERE draft_id='one'").fetchone()[0] == 1
