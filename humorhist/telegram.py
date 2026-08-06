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
        params: dict[str, Any] = {"chat_id": chat_id, "text": text}
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


def _chunk_text(text: str, limit: int = 4000) -> list[str]:
    """Split long text into <=limit-char chunks on line boundaries.

    Telegram caps a message at 4096 chars; drafts can be far longer, so we send
    the draft as several messages. Buttons go only on the final chunk.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > limit:
            chunks.append(current)
            current = line
        else:
            current = (current + "\n" + line) if current else line
    if current:
        chunks.append(current)
    return chunks


def _send_long(
    client: TelegramTransport, chat_id: str, text: str, reply_markup: dict | None = None
) -> list[dict]:
    """Send `text` as one or more messages; attach `reply_markup` to the last."""
    chunks = _chunk_text(text)
    sent: list[dict] = []
    for i, chunk in enumerate(chunks):
        markup = reply_markup if (i == len(chunks) - 1) else None
        sent.append(client.send_message(chat_id, chunk, reply_markup=markup))
    return sent


# --------------------------------------------------------------------------- #
# Review transport logic (transport-agnostic)                                 #
# --------------------------------------------------------------------------- #


def _pending_ids(conn: sqlite3.Connection) -> set[str]:
    return {d["id"] for d in review.pending_drafts(conn)}


def _send_one(
    conn: sqlite3.Connection, client: TelegramTransport, chat_id: str, row: dict
) -> list[dict]:
    """Send a single draft (chunked) with Approve/Reject buttons on the last chunk."""
    pool = db.get_pool_item(conn, row["pool_id"])
    text = render.render_draft(row, pool)
    try:
        return _send_long(client, chat_id, text, reply_markup=_keyboard(row["id"]))
    except Exception as exc:  # noqa: BLE001 - one bad draft must not kill the loop
        print(f"[telegram] failed to send draft {row['id']}: {exc}")
        return []


def send_pending_drafts(
    conn: sqlite3.Connection, client: TelegramTransport, chat_id: str
) -> list[dict]:
    """Send every pending draft (one message series each) with Approve/Reject buttons.

    Used by the ``--once`` dump mode. The default review loop sends drafts
    one-at-a-time instead (see ``run_review_bot``). Returns all sent messages.
    """
    sent: list[dict] = []
    for row in review.pending_drafts(conn):
        sent.extend(_send_one(conn, client, chat_id, row))
    return sent


def handle_callback(
    conn: sqlite3.Connection, client: TelegramTransport, chat_id: str, update: dict
) -> dict | None:
    """Process an inline-button tap.

    - ``approve:<id>`` / ``reject:<id>``: record the decision, then prompt for
      optional notes (reply or /skip).
    - ``notes:<id>``: from /listapproved -- prompt for notes on an already
      approved draft (re-applying approve is idempotent and keeps its queue row).

    Returns a dict describing the action so the caller can track note state.
    """
    cq = update.get("callback_query")
    if not cq:
        return None
    data = (cq.get("data") or "").strip()
    if data.startswith("approve:") or data.startswith("reject:"):
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
    if data.startswith("notes:"):
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="add notes")
        note = client.send_message(
            chat_id,
            f"Notes for already-approved draft `{draft_id}`. Reply here "
            f"(or send /skip to leave the existing notes untouched):",
        )
        return {"draft_id": draft_id, "note_message_id": note["message_id"]}
    return None


def handle_text(
    conn: sqlite3.Connection,
    client: TelegramTransport,
    chat_id: str,
    awaiting: dict,
    update: dict,
) -> dict | None:
    """Process a text message as optional editor notes.

    ``awaiting`` maps draft_id -> {"note_message_id": <id>}. A reply whose
    reply_to_message_id matches a tracked note prompt stores the text as editor
    notes on that draft (re-applying approve, which is idempotent). ``/skip``
    clears the prompt without storing notes. Falls back to the single open note
    prompt when the user just types instead of replying to the prompt.
    """
    msg = update.get("message")
    if not msg or "text" not in msg:
        return None
    reply_to = (msg.get("reply_to_message") or {}).get("message_id")
    text = msg["text"].strip()

    draft_id = None
    for did, st in awaiting.items():
        if st.get("note_message_id") == reply_to:
            draft_id = did
            break
    if draft_id is None and len(awaiting) == 1 and text != "/skip":
        draft_id = next(iter(awaiting))
    if draft_id is None:
        return None
    if text == "/skip":
        awaiting.pop(draft_id, None)
        client.send_message(chat_id, f"Notes left blank for `{draft_id}`.")
        return {"skipped": draft_id}
    # re-apply with the same (approve) decision so notes persist idempotently;
    # an already-approved draft stays approved and keeps its queue row.
    review.apply_review(conn, draft_id, decision="approve", notes=text)
    awaiting.pop(draft_id, None)
    client.send_message(chat_id, f"Notes saved for `{draft_id}`.")
    return {"noted": draft_id}


HELP_TEXT = (
    "HumorHist review bot\n\n"
    "/reviewdraft - review pending drafts one by one (Approve/Reject + notes)\n"
    "/listapproved - list drafts you've greenlit; tap one to add notes\n"
    "/status - approved / rejected / pending breakdown\n"
    "/help - this message"
)


def send_approved_list(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str) -> int:
    """DM a list of approved drafts, each with an inline 'add notes' button.

    Returns the number of approved drafts listed.
    """
    rows = review.approved_drafts(conn)
    if not rows:
        client.send_message(chat_id, "No approved drafts yet.")
        return 0
    lines = ["✅ Approved drafts (tap to add notes):"]
    keyboard = []
    for r in rows:
        title = r["title"] or "(unknown)"
        lines.append(f"  • {title}")
        keyboard.append(
            [{"text": f"📝 {title[:32]}", "callback_data": f"notes:{r['draft_id']}"}]
        )
    client.send_message(
        chat_id, "\n".join(lines), reply_markup={"inline_keyboard": keyboard}
    )
    return len(rows)


def run_review_session(
    conn: sqlite3.Connection,
    client: TelegramTransport,
    chat_id: str,
    awaiting: dict,
    offset_ref: list[int],
    *,
    poll_timeout: int = 30,
    max_iterations: int = 1_000_000,
) -> int:
    """The one-draft-at-a-time review flow, triggered by /reviewdraft.

    Sends the 📊 progress block, then one pending draft with Approve/Reject
    buttons, waits for the tap (and optional note), then the next. When none are
    left it idles, polling for late notes/taps and newly-generated drafts.
    """
    decided = 0

    def _handle(upd: dict) -> str | None:
        offset_ref[0] = max(offset_ref[0], upd.get("update_id", 0) + 1)
        if "callback_query" in upd:
            res = handle_callback(conn, client, chat_id, upd)
            if res and "decision" in res:
                nonlocal decided
                decided += 1
                awaiting[res["draft_id"]] = {"note_message_id": res["note_message_id"]}
                return res["draft_id"]
        elif "message" in upd:
            handle_text(conn, client, chat_id, awaiting, upd)
        return None

    send_reviewed_summary(conn, client, chat_id)
    sent: set[str] = set()
    caught_up = False
    iters = 0
    while True:
        iters += 1
        if iters > max_iterations:
            return decided
        pending = review.pending_drafts(conn)
        draft = next((d for d in pending if d["id"] not in sent), None)
        if draft is None:
            if not caught_up:
                client.send_message(chat_id, "✅ All caught up — no drafts pending.")
                caught_up = True
            try:
                for upd in client.get_updates(offset=offset_ref[0], timeout=poll_timeout):
                    _handle(upd)
            except TelegramError as exc:
                print(f"[telegram] {exc}; retrying in 5s")
                time.sleep(5)
            continue
        _send_one(conn, client, chat_id, draft)
        sent.add(draft["id"])
        caught_up = False
        while draft["id"] in _pending_ids(conn):
            try:
                upds = client.get_updates(offset=offset_ref[0], timeout=poll_timeout)
            except TelegramError as exc:
                print(f"[telegram] {exc}; retrying in 5s")
                time.sleep(5)
                continue
            for upd in upds:
                rid = _handle(upd)
                if rid == draft["id"]:
                    break
    return decided


def run_review_bot(
    conn: sqlite3.Connection,
    client: TelegramTransport,
    chat_id: str,
    *,
    once: bool = False,
    poll_timeout: int = 30,
    max_iterations: int = 1_000_000,
) -> int:
    """Command-driven Telegram review bot (long-poll).

    Idles and reacts to ``/commands`` instead of pushing drafts on startup:

      /reviewdraft  -> start the one-by-one review flow (run_review_session)
      /listapproved -> list approved drafts with 'add notes' buttons
      /status       -> reviewed/pending breakdown
      /help, /start -> this message

    ``once=True`` keeps the legacy dump behaviour (send summary + all pending,
    process queued updates once) for one-shot CLI runs and tests.
    """
    offset_ref = [0]
    awaiting: dict[str, dict] = {}
    decided = 0

    def _handle(upd: dict) -> None:
        offset_ref[0] = max(offset_ref[0], upd.get("update_id", 0) + 1)
        if "callback_query" in upd:
            res = handle_callback(conn, client, chat_id, upd)
            if res and "decision" in res:
                nonlocal decided
                decided += 1
                awaiting[res["draft_id"]] = {"note_message_id": res["note_message_id"]}
            elif res and "note_message_id" in res:
                # notes: button from /listapproved
                awaiting[res["draft_id"]] = {"note_message_id": res["note_message_id"]}
            return
        msg = upd.get("message")
        if not msg:
            return
        text = (msg.get("text") or "").strip()
        if text.startswith("/"):
            _dispatch(text)
            return
        handle_text(conn, client, chat_id, awaiting, upd)

    def _dispatch(text: str) -> None:
        nonlocal decided
        cmd = text.split()[0].lower()
        if cmd == "/reviewdraft":
            decided += run_review_session(
                conn, client, chat_id, awaiting, offset_ref,
                poll_timeout=poll_timeout, max_iterations=max_iterations,
            )
        elif cmd == "/listapproved":
            send_approved_list(conn, client, chat_id)
        elif cmd == "/status":
            send_reviewed_summary(conn, client, chat_id)
        elif cmd in ("/help", "/start"):
            client.send_message(chat_id, HELP_TEXT)
        else:
            client.send_message(chat_id, "Unknown command. Send /help.")

    if once:
        send_reviewed_summary(conn, client, chat_id)
        for row in review.pending_drafts(conn):
            _send_one(conn, client, chat_id, row)
        for upd in client.get_updates(offset=offset_ref[0], timeout=0):
            _handle(upd)
        return decided

    client.send_message(
        chat_id,
        "HumorHist review bot ready. Send /reviewdraft to review, "
        "/listapproved to browse greenlit drafts, /help for commands.",
    )
    iters = 0
    while True:
        iters += 1
        if iters > max_iterations:
            return decided
        try:
            for upd in client.get_updates(offset=offset_ref[0], timeout=poll_timeout):
                _handle(upd)
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
        f"\U0001F4DD {n} draft(s) awaiting review. Run `review` or check Telegram to decide.",
    )
    return n


def format_reviewed_summary(summary: dict) -> str:
    """Render the reviewed/pending breakdown as a Telegram-friendly text block.

    Lists approved and rejected topics (the "reviewed" ones) plus the pending
    count, so the reviewer can see what's already been decided.
    """
    lines = ["📊 Review progress"]

    approved = summary.get("approved", {})
    rejected = summary.get("rejected", {})
    pending = summary.get("pending", {})

    if approved["titles"]:
        lines.append(f"\n✅ Approved ({approved['count']}):")
        lines.extend(f"  • {t}" for t in approved["titles"])
    else:
        lines.append(f"\n✅ Approved: 0")

    if rejected["titles"]:
        lines.append(f"\n❌ Rejected ({rejected['count']}):")
        lines.extend(f"  • {t}" for t in rejected["titles"])
    else:
        lines.append(f"\n❌ Rejected: 0")

    lines.append(f"\n⏳ Pending: {pending['count']}")
    return "\n".join(lines)


def send_reviewed_summary(
    conn: db.Connection, client: TelegramTransport, chat_id: str
) -> str:
    """DM the reviewer the approved/rejected/pending breakdown. Returns the text."""
    text = format_reviewed_summary(review.reviewed_summary(conn))
    client.send_message(chat_id, text)
    return text
