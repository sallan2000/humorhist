"""Phase 3.3/3.4 — Telegram review transport for humorhist.

The review *decisions* live in ``humorhist.review.apply_review``; this module is
purely the transport: it shows each pending draft (via the shared
``humorhist.render.render_draft``) with inline Approve/Reject buttons, turns a
button tap into a call to ``apply_review``, and captures an optional follow-up
text reply as editor notes. It also exposes ``notify_new_drafts`` for nudges.

Network access is isolated behind the ``TelegramTransport`` protocol so tests
inject ``StubTelegram`` and never hit the API. The real ``TelegramClient`` speaks
the Bot API over httpx using long-polling (getUpdates) — NO webhook, because the
host is behind Cloudflare/NAT and does not expose ports.

Config (env, never in the repo):
    HUMORHIST_TELEGRAM_BOT_TOKEN   required for real calls
    HUMORHIST_TELEGRAM_CHAT_ID      your chat id to DM (the reviewer)
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Protocol

import httpx

import humorhist.db as db
import humorhist.render as render
import humorhist.review as review

API_BASE = "https://api.telegram.org"


# --------------------------------------------------------------------------- #
# Transport protocol + stub                                                   #
# --------------------------------------------------------------------------- #


class TelegramTransport(Protocol):
    """Minimal Bot API surface the review loop needs."""

    def get_updates(self, offset: int, timeout: int) -> list[dict]: ...

    def send_message(
        self, chat_id: str, text: str, reply_markup: dict | None = None
    ) -> dict: ...

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> dict: ...


class StubTelegram:
    """Deterministic, network-free Telegram for tests.

    Feed queued updates via ``updates=[...]``; ``get_updates`` returns them.
    ``send_message`` records each message (including reply_markup) in ``.sent``
    and returns a message dict with a synthetic message_id; answered callback
    ids land in ``.answered``.
    """

    def __init__(self, updates: list[dict] | None = None) -> None:
        self.sent: list[dict] = []
        self.answered: set[str] = set()
        self._updates = list(updates or [])
        self._mid = 0

    def get_updates(self, offset: int = 0, timeout: int = 0) -> list[dict]:
        return self._updates

    def send_message(
        self, chat_id: str, text: str, reply_markup: dict | None = None
    ) -> dict:
        self._mid += 1
        msg: dict[str, Any] = {
            "message_id": self._mid,
            "chat_id": chat_id,
            "text": text,
        }
        if reply_markup is not None:
            msg["reply_markup"] = reply_markup
        self.sent.append(msg)
        return msg

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> dict:
        self.answered.add(callback_query_id)
        return {}


class TelegramError(RuntimeError):
    """Raised when a Bot API call fails."""


class TelegramClient:
    """Real Bot API client (long-poll). Token from env or constructor."""

    def __init__(
        self,
        token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.token = token or os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN", "")
        self.timeout = timeout
        self.max_retries = max_retries

    def _call(self, method: str, params: dict) -> dict:
        if not self.token:
            raise TelegramError(
                "no bot token: set HUMORHIST_TELEGRAM_BOT_TOKEN or pass token="
            )
        url = f"{API_BASE}/bot{self.token}/{method}"
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=params)
                    resp.raise_for_status()
                    body = resp.json()
                if not body.get("ok"):
                    raise TelegramError(f"Telegram API error: {body}")
                return body["result"]
            except Exception as exc:  # noqa: BLE001 - retry transient failures
                last = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise TelegramError(f"Telegram call {method} failed: {last}")

    def get_updates(self, offset: int = 0, timeout: int = 0) -> list[dict]:
        return self._call(
            "getUpdates",
            {
                "offset": offset,
                "timeout": timeout,
                "allowed_updates": ["message", "callback_query"],
            },
        )

    def send_message(
        self, chat_id: str, text: str, reply_markup: dict | None = None
    ) -> dict:
        params: dict[str, Any] = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        if reply_markup is not None:
            params["reply_markup"] = reply_markup
        return self._call("sendMessage", params)

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> dict:
        return self._call(
            "answerCallbackQuery", {"callback_query_id": callback_query_id, "text": text}
        )


def _keyboard(draft_id: str) -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "✅ Approve", "callback_data": f"approve:{draft_id}"},
                {"text": "❌ Reject", "callback_data": f"reject:{draft_id}"},
            ]
        ]
    }


# --------------------------------------------------------------------------- #
# Review transport logic (transport-agnostic)                                 #
# --------------------------------------------------------------------------- #


def send_pending_drafts(
    conn: sqlite3.Connection, client: TelegramTransport, chat_id: str
) -> list[dict]:
    """Send one Telegram message per pending draft with Approve/Reject buttons.

    Returns the list of sent message dicts (each carrying its reply_markup).
    """
    sent: list[dict] = []
    for row in review.pending_drafts(conn):
        pool = db.get_pool_item(conn, row["pool_id"])
        text = render.render_draft(row, pool)
        sent.append(client.send_message(chat_id, text, reply_markup=_keyboard(row["id"])))
    return sent


def handle_callback(
    conn: sqlite3.Connection, client: TelegramTransport, chat_id: str, update: dict
) -> dict | None:
    """Process a callback_query (button tap). Returns a result dict or None.

    On a valid approve/reject the decision is persisted via review.apply_review
    and a follow-up message is sent inviting optional editor notes; the returned
    dict carries ``note_message_id`` so the loop can map a later text reply.
    """
    cq = update.get("callback_query")
    if not cq:
        return None
    data = (cq.get("data") or "").strip()
    if ":" not in data:
        return None
    decision, _, draft_id = data.partition(":")
    if decision not in ("approve", "reject"):
        return None

    try:
        review.apply_review(conn, draft_id, decision=decision)
    except ValueError:
        client.answer_callback_query(cq["id"], text="already handled")
        return None

    client.answer_callback_query(cq["id"], text=f"{decision}d")
    note = client.send_message(
        chat_id,
        f"Draft `{draft_id}` {decision}d. Reply here with optional notes "
        f"(or send /skip to leave blank):",
    )
    return {"draft_id": draft_id, "decision": decision, "note_message_id": note["message_id"]}


def handle_text(
    conn: sqlite3.Connection,
    client: TelegramTransport,
    chat_id: str,
    awaiting: dict,
    update: dict,
) -> dict | None:
    """Process a text message as optional editor notes.

    ``awaiting`` maps note_message_id -> draft_id (populated by handle_callback).
    A reply whose reply_to_message_id is a tracked note prompt stores the text as
    editor notes on the already-approved draft. ``/skip`` clears the prompt
    without storing notes.
    """
    msg = update.get("message")
    if not msg or "text" not in msg:
        return None
    reply_to = (msg.get("reply_to_message") or {}).get("message_id")
    if reply_to not in awaiting:
        return None
    draft_id = awaiting.pop(reply_to)
    text = msg["text"].strip()
    if text == "/skip":
        return {"skipped": draft_id}
    # re-apply with the same (approve) decision so notes persist idempotently
    review.apply_review(conn, draft_id, decision="approve", notes=text)
    client.send_message(chat_id, f"Notes saved for `{draft_id}`.")
    return {"noted": draft_id}


def run_review_bot(
    conn: db.Connection,
    client: TelegramTransport,
    chat_id: str,
    *,
    once: bool = False,
    poll_timeout: int = 30,
) -> int:
    """Run the review loop. Returns the number of decisions processed.

    ``once=True`` processes the currently-queued updates once and returns (used
    by tests and one-shot CLI runs). ``once=False`` long-polls forever for the
    durable systemd runner (interrupt the process to stop).
    """
    send_pending_drafts(conn, client, chat_id)
    awaiting: dict[int, str] = {}
    decided = 0
    offset = 0

    def _process(updates: list[dict]) -> None:
        nonlocal offset, decided
        for upd in updates:
            offset = max(offset, upd.get("update_id", 0) + 1)
            if "callback_query" in upd:
                res = handle_callback(conn, client, chat_id, upd)
                if res:
                    decided += 1
                    awaiting[res["note_message_id"]] = res["draft_id"]
            elif "message" in upd:
                handle_text(conn, client, chat_id, awaiting, upd)

    if once:
        _process(client.get_updates(offset=offset, timeout=0))
        return decided

    while True:
        try:
            _process(client.get_updates(offset=offset, timeout=poll_timeout))
        except TelegramError as exc:
            print(f"[telegram] {exc}; retrying in 5s")
            time.sleep(5)


def notify_new_drafts(conn: db.Connection, client: TelegramTransport, chat_id: str) -> int:
    """DM the reviewer how many drafts are awaiting review. Returns that count.

    Silent (no message) when there is nothing pending, so we don't nag.
    """
    n = len(review.pending_drafts(conn))
    if n == 0:
        return 0
    client.send_message(
        chat_id,
        f"📝 {n} draft(s) awaiting review. Run `review` or check Telegram to decide.",
    )
    return n
