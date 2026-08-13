"""Tests for the real Telegram Bot API client (humorhist.telegram.TelegramClient).

These exercise the production long-poll transport with `respx` mocking the
Bot API HTTP calls, so no network is touched. The review *logic* is covered
elsewhere via StubTelegram; here we guard the transport layer that actually
runs in the systemd unit.
"""

from __future__ import annotations

import httpx
import respx

import humorhist.telegram as tg

API_BASE = "https://api.telegram.org"


def _client(token: str = "TESTTOKEN") -> tg.TelegramClient:
    return tg.TelegramClient(token=token, timeout=5.0, max_retries=0)


def test_get_updates_returns_results():
    with respx.mock(assert_all_called=False) as rt:
        rt.post(f"{API_BASE}/botTESTTOKEN/getUpdates").mock(
            return_value=httpx.Response(
                200, json={"ok": True, "result": [{"update_id": 1}]}
            )
        )
        client = _client()
        updates = client.get_updates(offset=0, timeout=0)
    assert updates == [{"update_id": 1}]


def test_send_message_posts_and_returns_result():
    with respx.mock(assert_all_called=False) as rt:
        route = rt.post(f"{API_BASE}/botTESTTOKEN/sendMessage").mock(
            return_value=httpx.Response(
                200, json={"ok": True, "result": {"message_id": 42}}
            )
        )
        client = _client()
        res = client.send_message("chat", "hello", reply_markup={"k": 1})
    assert res["message_id"] == 42
    # the reply_markup was forwarded as JSON
    body = route.calls.last.request.content.decode()
    assert '"k":1' in body


def test_send_photo_bytes_uses_multipart():
    png = b"\x89PNG\r\n fake-bytes"
    with respx.mock(assert_all_called=False) as rt:
        route = rt.post(f"{API_BASE}/botTESTTOKEN/sendPhoto").mock(
            return_value=httpx.Response(
                200, json={"ok": True, "result": {"message_id": 7}}
            )
        )
        client = _client()
        res = client.send_photo("chat", png, caption="a duck")
    assert res["message_id"] == 7
    req = route.calls.last.request
    assert req.headers["content-type"].startswith("multipart/form-data")
    assert b"image.png" in req.content
    assert b"a duck" in req.content


def test_send_photo_string_is_json_param():
    with respx.mock(assert_all_called=False) as rt:
        rt.post(f"{API_BASE}/botTESTTOKEN/sendPhoto").mock(
            return_value=httpx.Response(
                200, json={"ok": True, "result": {"message_id": 8}}
            )
        )
        client = _client()
        res = client.send_photo("chat", "FILEID123")
    assert res["message_id"] == 8


def test_answer_callback_query():
    with respx.mock(assert_all_called=False) as rt:
        rt.post(f"{API_BASE}/botTESTTOKEN/answerCallbackQuery").mock(
            return_value=httpx.Response(200, json={"ok": True, "result": True})
        )
        client = _client()
        res = client.answer_callback_query("cbid", text="done")
    assert res is True


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("HUMORHIST_TELEGRAM_BOT_TOKEN", raising=False)
    client = tg.TelegramClient(token="", timeout=1.0)
    try:
        client.get_updates()
    except tg.TelegramError as exc:
        assert "no bot token" in str(exc)
    else:
        raise AssertionError("expected TelegramError for missing token")


def test_non_ok_response_raises():
    with respx.mock(assert_all_called=False) as rt:
        rt.post(f"{API_BASE}/botTESTTOKEN/sendMessage").mock(
            return_value=httpx.Response(200, json={"ok": False, "error_code": 400})
        )
        client = _client()
        try:
            client.send_message("chat", "hi")
        except tg.TelegramError as exc:
            assert "Telegram API error" in str(exc)
        else:
            raise AssertionError("expected TelegramError on non-ok body")


def test_retry_then_succeed():
    # max_retries=2 -> up to 3 attempts. First two 500, third succeeds.
    calls = {"n": 0}

    def _flaky(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, json={"ok": False})
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    with respx.mock(assert_all_called=False) as rt:
        rt.post(f"{API_BASE}/botTESTTOKEN/sendMessage").mock(side_effect=_flaky)
        client = tg.TelegramClient(token="TESTTOKEN", timeout=1.0, max_retries=2)
        res = client.send_message("chat", "retry me")
    assert res["message_id"] == 1
    assert calls["n"] == 3


def test_retry_exhausted_raises():
    with respx.mock(assert_all_called=False) as rt:
        rt.post(f"{API_BASE}/botTESTTOKEN/sendMessage").mock(
            return_value=httpx.Response(500, json={"ok": False})
        )
        client = tg.TelegramClient(token="TESTTOKEN", timeout=1.0, max_retries=1)
        try:
            client.send_message("chat", "boom")
        except tg.TelegramError:
            pass
        else:
            raise AssertionError("expected TelegramError after retries exhausted")
