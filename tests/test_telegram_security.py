"""Authorization and credential-redaction regressions for Telegram polling."""

import traceback

import httpx
import pytest
import respx

import humorhist.db as db
import humorhist.telegram as tg

CHAT = 12345


@pytest.mark.parametrize("once", [False, True])
@pytest.mark.parametrize("chat_id", ["-10012345", "-123", "@channel", "0", "", " 123", "+123", "0123", "１２３", None, 123, True])
def test_invalid_chat_id_fails_before_transport_or_database(chat_id, once):
    class NoSideEffects:
        def __getattr__(self, name):
            pytest.fail(f"unexpected side effect: {name}")

    with pytest.raises(ValueError, match="positive numeric private"):
        tg.run_review_bot(NoSideEffects(), NoSideEffects(), chat_id, once=once)


def message(uid, text, chat=CHAT, sender=CHAT, kind="private"):
    return {"update_id": uid, "message": {"text": text, "chat": {"id": chat, "type": kind}, "from": {"id": sender}}}


def callback(uid, data, chat=CHAT, sender=CHAT, kind="private"):
    return {"update_id": uid, "callback_query": {"id": str(uid), "data": data, "from": {"id": sender},
            "message": {"message_id": 1, "chat": {"id": chat, "type": kind}}}}


def test_once_acknowledges_processed_updates_without_losing_new_ones(tmp_path):
    conn = db.connect(str(tmp_path / "db.sqlite"))
    db.migrate(conn)

    class Queue(tg.StubTelegram):
        def __init__(self):
            super().__init__()
            self.queue = [message(1, "/suggest old")]
            self.offsets = []

        def get_updates(self, offset=0, timeout=0):
            self.offsets.append(offset)
            self.queue = [u for u in self.queue if u["update_id"] >= offset]
            if len(self.offsets) == 2:
                self.queue.append(message(2, "/suggest new"))
            return list(self.queue)

    stub = Queue()
    tg.run_review_bot(conn, stub, str(CHAT), once=True)
    assert stub.offsets == [0, 2]
    assert [u["update_id"] for u in stub.queue] == [2]
    tg.run_review_bot(conn, stub, str(CHAT), once=True)
    assert stub.offsets == [0, 2, 0, 3]
    assert not stub.queue
    assert [row[0] for row in conn.execute("SELECT title FROM pool ORDER BY title")] == ["new", "old"]


@pytest.mark.parametrize("update", [
    message(1, "/help", chat=777),
    message(1, "/help", sender=777),
    message(1, "/help", kind="group"),
    {"message": {"text": "/help"}},
    callback(1, "next:d1", chat=777),
    callback(1, "next:d1", sender=777),
    callback(1, "next:d1", kind="group"),
    {"callback_query": {"from": {"id": CHAT}, "inline_message_id": "x"}},
])
def test_rejects_foreign_or_incomplete_updates(update):
    assert not tg._authorized_update(update, str(CHAT))


def test_authorized_callback_commits_review(tmp_path):
    conn = db.connect(str(tmp_path / "db.sqlite"))
    db.migrate(conn)
    conn.execute("INSERT INTO pool (id, title, status) VALUES ('p', 'Title', 'drafted')")
    conn.execute("INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) "
                 "VALUES ('d1', 'p', '{}', '{}', 'pending', '2026-01-01')")
    conn.commit()
    token = tg._decision_token(conn, "d1")
    stub = tg.StubTelegram()
    tg.handle_callback(conn, stub, str(CHAT), callback(0, f"reject:d1:{token}"))
    confirm = stub.sent[-1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    stub._updates = [callback(1, confirm)]
    assert tg.run_review_bot(conn, stub, str(CHAT), once=True) == 1
    assert conn.execute("SELECT status FROM drafts WHERE id='d1'").fetchone()[0] == "rejected"
    assert "1" in stub.answered


def test_polling_rejects_commands_callbacks_and_prompt_replies(tmp_path, monkeypatch):
    sessions = []
    monkeypatch.setattr(tg, "run_review_session", lambda *args, **kwargs: sessions.append(1) or 0)
    conn = db.connect(str(tmp_path / "db.sqlite"))
    db.migrate(conn)
    conn.execute("INSERT INTO pool (id, title, status) VALUES ('p', 'Title', 'drafted')")
    conn.execute("INSERT INTO drafts (id, pool_id, brief_json, angles_json, status, created_at) "
                 "VALUES ('d1', 'p', '{}', '{}', 'pending', '2026-01-01')")
    conn.commit()
    batches = [
        [message(1, "/suggest injected", sender=777), callback(2, "confirm:reject:d1", sender=777),
         callback(3, "next:d1", chat=777), message(4, "/notes d1")],
        [message(5, "injected note", sender=777), message(6, "/skip")],
    ]

    class Poller(tg.StubTelegram):
        def __init__(self):
            super().__init__()
            self.offsets = []

        def get_updates(self, offset=0, timeout=0):
            self.offsets.append(offset)
            return batches.pop(0) if batches else []

    stub = Poller()
    tg.run_review_bot(conn, stub, str(CHAT), max_iterations=3)
    assert stub.offsets == [0, 5, 7]
    row = conn.execute("SELECT status, editor_notes FROM drafts WHERE id='d1'").fetchone()
    assert row["status"] == "pending"
    assert row["editor_notes"] is None
    assert conn.execute("SELECT COUNT(*) FROM pool").fetchone()[0] == 1
    assert not stub.answered
    assert not sessions
    assert any("Notes left blank" in item["text"] for item in stub.sent)
    assert not any("injected" in item["text"] for item in stub.sent)


@pytest.mark.parametrize("method", ["message", "photo"])
@pytest.mark.parametrize("failure", ["http", "body", "network"])
def test_telegram_errors_redact_token_from_traceback(method, failure, caplog):
    token = "123456:FAKE_SECRET_TOKEN"
    caplog.set_level("INFO", logger="httpx")
    endpoint = "sendMessage" if method == "message" else "sendPhoto"
    with respx.mock(assert_all_called=True) as router:
        route = router.post(f"https://api.telegram.org/bot{token}/{endpoint}")
        if failure == "http":
            route.mock(return_value=httpx.Response(401))
        elif failure == "body":
            route.mock(return_value=httpx.Response(200, json={"ok": False, "description": f"bad {token}"}))
        else:
            route.mock(side_effect=httpx.ConnectError(f"failed for {token}"))
        client = tg.TelegramClient(token=token, max_retries=0)
        with pytest.raises(tg.TelegramError) as caught:
            if method == "message":
                client.send_message(str(CHAT), "test")
            else:
                client.send_photo(str(CHAT), b"image")
    assert token not in str(caught.value)
    assert token not in "".join(traceback.format_exception(caught.value))
    assert token not in caplog.text
    assert "[REDACTED]" in str(caught.value)
