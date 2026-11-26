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
    HUMORHIST_TELEGRAM_CHAT_ID      reviewer's private numeric chat/user ID
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx

import humorhist.db as db
import humorhist.job_control as job_control
import humorhist.render as render
import humorhist.review as review

logger = logging.getLogger("humorhist.telegram")

API_BASE = "https://api.telegram.org"


class _TelegramURLLogFilter(logging.Filter):
    """Keep httpx's ordinary INFO request log from exposing Bot API tokens."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = re.sub(r"(https://api\.telegram\.org/bot)[^/\s]+", r"\1[REDACTED]", message)
        if redacted != message:
            record.msg, record.args = redacted, ()
        return True


_httpx_logger = logging.getLogger("httpx")
if not any(isinstance(item, _TelegramURLLogFilter) for item in _httpx_logger.filters):
    _httpx_logger.addFilter(_TelegramURLLogFilter())


def _resolve_image_dir(image_dir: str | None) -> str | None:
    """Resolve where generated story images are written.

    Priority: explicit ``image_dir`` argument, then the ``HUMORHIST_IMAGE_DIR``
    env var, then ``<repo>/data/images``. Returns ``None`` only if even the
    fallback can't be located (shouldn't happen); callers treat a missing dir as
    "skip image generation" rather than crashing the approve path.
    """
    if image_dir:
        return str(image_dir)
    env_dir = os.environ.get("HUMORHIST_IMAGE_DIR")
    if env_dir:
        return env_dir
    try:
        repo = Path(__file__).resolve().parent.parent
        return str(repo / "data" / "images")
    except Exception:  # noqa: BLE001
        return None


def _get_llm(chat_id: str, client: TelegramTransport, *, silent: bool = False):
    """Return a resilient LLM client, or ``None`` if none is available.

    On unavailability it DMs a clean "LLM unavailable" message (unless
    ``silent``) instead of letting a raw traceback reach the user's phone. This
    is the key to Telegram-primary operation surviving logout: every LLM call
    goes through here and degrades gracefully when the Nous OAuth token has
    expired and no static ``HUMORHIST_LLM_API_KEY`` is set.
    """
    from humorhist.llm import LLMUnavailable, resilient_client

    try:
        return resilient_client()
    except LLMUnavailable as exc:
        if not silent:
            client.send_message(
                chat_id,
                f"⚠️ LLM unavailable right now: {exc}. "
                f"Set HUMORHIST_LLM_API_KEY for unattended use, or keep a "
                f"Hermes session open. (Action skipped.)",
            )
        return None


def _get_image_client(chat_id: str, client: TelegramTransport, *, silent: bool = False):
    """Return a resilient image client, or ``None`` if none is available.

    Mirrors ``_get_llm``: missing ``HUMORHIST_IMAGE_API_KEY`` yields ``None``
    (callers skip image generation, post copy + pipeline unaffected) instead of
    surfacing a traceback to the user's phone.
    """
    from humorhist.imagegen import ImageUnavailable, resilient_image_client

    try:
        return resilient_image_client()
    except ImageUnavailable as exc:
        if not silent:
            client.send_message(
                chat_id,
                f"🖼️ Story image skipped (no image credential: {exc}). "
                f"Set HUMORHIST_IMAGE_API_KEY to enable. (Post copy unaffected.)",
            )
        return None


# --------------------------------------------------------------------------- #
# Transport protocol + stub                                                   #
# --------------------------------------------------------------------------- #


class TelegramTransport(Protocol):
    """Minimal Bot API surface the review loop needs."""

    def get_updates(self, offset: int, timeout: int) -> list[dict]: ...

    def send_message(self, chat_id: str, text: str, reply_markup: dict | None = None) -> dict: ...

    def send_photo(self, chat_id: str, photo: bytes | str, caption: str | None = None) -> dict: ...

    def send_document(self, chat_id: str, document: bytes, caption: str) -> dict: ...

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
        return [update for update in self._updates if update.get("update_id", 0) >= offset]

    def send_message(self, chat_id: str, text: str, reply_markup: dict | None = None) -> dict:
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

    def send_photo(self, chat_id: str, photo: bytes | str, caption: str | None = None) -> dict:
        self._mid += 1
        msg: dict[str, Any] = {
            "message_id": self._mid,
            "chat_id": chat_id,
            "photo": photo,
        }
        if caption is not None:
            msg["caption"] = caption
        self.sent.append(msg)
        return msg

    def send_document(self, chat_id: str, document: bytes, caption: str) -> dict:
        self._mid += 1
        msg = {"message_id": self._mid, "chat_id": chat_id, "document": document, "caption": caption}
        self.sent.append(msg)
        return msg


class TelegramError(RuntimeError):
    """Raised when a Bot API call fails."""


def _authorized_update(update: dict, chat_id: str) -> bool:
    """Accept only the configured user's private messages and button taps.

    Inline callbacks have no private message and are intentionally rejected.
    Telegram private chat IDs are the same as the corresponding user IDs.
    """
    if not isinstance(update, dict):
        return False
    callback = update.get("callback_query")
    if callback is not None:
        if not isinstance(callback, dict):
            return False
        message = callback.get("message")
        sender = callback.get("from")
    else:
        message = update.get("message")
        sender = message.get("from") if isinstance(message, dict) else None
    if not isinstance(message, dict) or not isinstance(sender, dict):
        return False
    chat = message.get("chat")
    if not isinstance(chat, dict) or chat.get("type") != "private":
        return False
    expected = str(chat_id)
    return (
        chat.get("id") is not None
        and sender.get("id") is not None
        and str(chat["id"]) == expected
        and str(sender["id"]) == expected
    )


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
            raise TelegramError("no bot token: set HUMORHIST_TELEGRAM_BOT_TOKEN or pass token=")
        url = f"{API_BASE}/bot{self.token}/{method}"
        timeout: float | httpx.Timeout = self.timeout
        if method == "getUpdates" and params.get("timeout", 0) > 0:
            # The server can hold the response for the full poll interval.
            # Give the read phase extra time without relaxing connect/write/pool.
            timeout = httpx.Timeout(self.timeout, read=max(self.timeout, params["timeout"] + 10.0))
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, json=params)
                    resp.raise_for_status()
                    body = resp.json()
                if not body.get("ok"):
                    raise TelegramError(f"Telegram API error: {body}")
                return body["result"]
            except Exception as exc:  # noqa: BLE001 - retry transient failures
                last = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
        detail = str(last).replace(self.token, "[REDACTED]")
        raise TelegramError(f"Telegram call {method} failed: {detail}") from None

    def get_updates(self, offset: int = 0, timeout: int = 0) -> list[dict]:
        return self._call(
            "getUpdates",
            {
                "offset": offset,
                "timeout": timeout,
                "allowed_updates": ["message", "callback_query"],
            },
        )

    def send_message(self, chat_id: str, text: str, reply_markup: dict | None = None) -> dict:
        params: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_markup is not None:
            params["reply_markup"] = reply_markup
        return self._call("sendMessage", params)

    def send_document(self, chat_id: str, document: bytes, caption: str) -> dict:
        """Upload bytes, never a local path; do not expose API errors or retry uploads."""
        if not self.token:
            raise TelegramError("no bot token: set HUMORHIST_TELEGRAM_BOT_TOKEN or pass token=")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{API_BASE}/bot{self.token}/sendDocument",
                    data={"chat_id": chat_id, "caption": caption},
                    files={"document": ("queue-export.txt", document, "text/plain; charset=utf-8")},
                )
                resp.raise_for_status()
                body = resp.json()
            if not body.get("ok"):
                raise TelegramError("Telegram rejected document")
            return body["result"]
        except Exception:
            raise TelegramError("Telegram document upload failed") from None

    def send_photo(self, chat_id: str, photo: bytes | str, caption: str | None = None) -> dict:
        params: dict[str, Any] = {"chat_id": chat_id}
        # ``photo`` may be raw bytes (we send multipart) or a file_id / URL
        # (we send as a string param). Telegram's sendPhoto takes either.
        if isinstance(photo, (bytes, bytearray)):
            files = {"photo": ("image.png", bytes(photo), "image/png")}
            if caption is not None:
                params["caption"] = caption
            # _call only does JSON; do a small multipart POST here.
            if not self.token:
                raise TelegramError("no bot token: set HUMORHIST_TELEGRAM_BOT_TOKEN or pass token=")
            url = f"{API_BASE}/bot{self.token}/sendPhoto"
            last: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    with httpx.Client(timeout=self.timeout) as client:
                        resp = client.post(url, data=params, files=files)
                        resp.raise_for_status()
                        body = resp.json()
                    if not body.get("ok"):
                        raise TelegramError(f"Telegram API error: {body}")
                    return body["result"]
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    if attempt < self.max_retries:
                        time.sleep(2**attempt)
            detail = str(last).replace(self.token, "[REDACTED]")
            raise TelegramError(f"Telegram sendPhoto failed: {detail}") from None
        params["photo"] = photo
        if caption is not None:
            params["caption"] = caption
        return self._call("sendPhoto", params)

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> dict:
        return self._call("answerCallbackQuery", {"callback_query_id": callback_query_id, "text": text})


def _keyboard(draft_id: str, *, fast: bool = False, token: str = "") -> dict:
    # The Next button advances the one-by-one review WITHOUT a parked modal
    # loop: after deciding this draft the user taps Next (or sends /reviewdraft
    # again) to pull the next pending draft from the command loop. This keeps
    # /reviewdraft non-modal — the bot stays responsive to other commands the
    # whole time instead of swallowing the user until every draft is done.
    return {
        "inline_keyboard": [
            [
                {
                    "text": "✅ Approve now" if fast else "✅ Approve",
                    "callback_data": f"{'fastapprove' if fast else 'approve'}:{draft_id}:{token}",
                },
                {
                    "text": "❌ Reject now" if fast else "❌ Reject",
                    "callback_data": f"{'fastreject' if fast else 'reject'}:{draft_id}:{token}",
                },
                {"text": "⏸ Later", "callback_data": f"{'fastlater' if fast else 'later'}:{draft_id}:{token}"},
            ],
            [
                {"text": "⏭ Next draft ▶️", "callback_data": f"{'nextfast' if fast else 'next'}:{draft_id}"},
            ],
        ]
    }


def _next_keyboard(draft_id: str, *, fast: bool = False) -> dict:
    """Keyboard for the 'no more pending' state shown after the last draft."""
    return {
        "inline_keyboard": [
            [
                {"text": "⏭ Review more ▶️", "callback_data": f"{'nextfast' if fast else 'next'}:{draft_id}"},
            ],
            [{"text": "⚡ Fast review (no confirmation)", "callback_data": f"nextfast:{draft_id}"}],
            [{"text": "🏠 Menu", "callback_data": "menu:home"}],
        ]
    }


def _main_menu(client, chat_id):
    client.send_message(chat_id, "What would you like to do? /help lists all commands.", reply_markup={
        "inline_keyboard": [
            [{"text": "📖 Review", "callback_data": "menu:review"}],
            [{"text": "📋 Queue", "callback_data": "menu:queue"}],
            [{"text": "🔎 Find drafts", "callback_data": "menu:find"}],
        ]
    })


def _find_menu(client, chat_id):
    with _intent_lock:
        # Opening a new menu revokes unused buttons from older menus in this chat.
        for nonce, entry in list(_intents.items()):
            if entry[:2] == (str(chat_id), "") and entry[2] in ("screen", "draft"):
                del _intents[nonce]
        screen = _new_intent(chat_id, "", "screen", "")
        draft = _new_intent(chat_id, "", "draft", "")
    client.send_message(
        chat_id,
        "Find drafts in three separate steps: 1. Harvest candidates; 2. Screen them; "
        "3. Draft the best candidates. Each button starts only that step. "
        "Harvest uses network sources; screening and drafting use an LLM and may incur charges.",
        reply_markup={"inline_keyboard": [
            [{"text": "1️⃣ Harvest candidates", "callback_data": "find:harvest"}],
            [{"text": "2️⃣ Screen candidates (LLM)", "callback_data": f"find:screen:{screen}"}],
            [{"text": "3️⃣ Create drafts (LLM)", "callback_data": f"find:draft:{draft}"}],
            [{"text": "🏠 Menu", "callback_data": "menu:home"}],
        ]},
    )


# Confirmation intents deliberately disappear on restart. Consume after commit;
# cancellation and consumption are serialized, including concurrent callbacks.
_intent_lock = threading.RLock()
_intents: dict[str, tuple[str, str, str, str]] = {}


def _new_intent(chat_id: str, draft_id: str, action: str, version: str) -> str:
    with _intent_lock:
        nonce = secrets.token_urlsafe(8)
        _intents[nonce] = (str(chat_id), draft_id, action, version)
        return nonce


@contextlib.contextmanager
def _claim_intent(nonce: str, chat_id: str, draft_id: str, action: str):
    """Serialize cancel/confirm and consume only after the database commits."""
    with _intent_lock:
        entry = _intents.get(nonce)
        if not entry or entry[:3] != (str(chat_id), draft_id, action):
            yield None
            return
        yield entry[3]
        _intents.pop(nonce, None)


def _can_issue(conn, client, chat_id):
    if conn.in_transaction:
        client.send_message(chat_id, "⚠️ An uncommitted transaction is open; commit or roll it back before retrying.")
        return False
    return True


def _confirm_keyboard(draft_id: str, decision: str, token: str) -> dict:
    """Confirm / cancel buttons shown after an initial approve/reject tap.

    ``decision`` is ``approve`` or ``reject``. Tapping confirm commits the
    decision (the original commit); cancel backs out with no state change.
    """
    verb = "Approve" if decision == "approve" else "Reject"
    return {
        "inline_keyboard": [
            [
                {
                    "text": f"✅ Yes, {verb}",
                    "callback_data": f"confirm:{decision}:{draft_id}:{token}",
                },
                {"text": "↩️ Cancel", "callback_data": f"cancel:{draft_id}:{token}"},
            ]
        ]
    }


def _units(text: str) -> int:
    """Conservatively count Telegram UTF-16 code units."""
    return len(text.encode("utf-16-le")) // 2


def _chunk_text(text: str, limit: int = 4000) -> list[str]:
    """Split on line/word boundaries when possible, hard-splitting long words."""
    if limit < 2:
        raise ValueError("limit must be at least 2")
    chunks, current, size = [], [], 0
    for ch in text:
        n = _units(ch)
        if size + n > limit:
            boundary = next((i + 1 for i in range(len(current) - 1, -1, -1) if current[i] == "\n"), 0)
            if not boundary:
                boundary = next((i + 1 for i in range(len(current) - 1, -1, -1) if current[i].isspace()), 0)
            if not boundary:
                boundary = len(current)
            chunks.append("".join(current[:boundary]))
            current = current[boundary:]
            size = _units("".join(current))
        current.append(ch)
        size += n
    if current or not chunks:
        chunks.append("".join(current))
    return chunks


def _send_long(
    client: TelegramTransport, chat_id: str, text: str, reply_markup: dict | None = None, *, context: str | None = None
) -> list[dict]:
    """Attach selection context and keyboard to the final message."""
    # A URL longer than one Telegram message cannot be clickable intact.
    text = re.sub(
        r"https?://[^\s]+",
        lambda m: (
            "[URL omitted: too long for Telegram; retrieve the full URL from the original record or export]"
            if _units(m.group()) > 3700
            else m.group()
        ),
        text,
    )
    chunks = _chunk_text(text)
    if len(chunks) > 1 and context:
        # Reserve room for context on the actionable final message.
        chunks = _chunk_text(text, 4000 - _units(context) - 2)
        chunks[-1] += "\n\n" + context
    return [
        client.send_message(chat_id, chunk, reply_markup=reply_markup if i == len(chunks) - 1 else None)
        for i, chunk in enumerate(chunks)
    ]


def _label(conn: sqlite3.Connection, draft_id: str) -> str:
    row = conn.execute(
        "SELECT p.title, d.short_code FROM drafts d LEFT JOIN pool p ON p.id=d.pool_id WHERE d.id=?",
        (draft_id,),
    ).fetchone()
    if not row:
        return f"unknown draft (#{draft_id})"
    title = " ".join((row["title"] or "(unknown)").split())
    if len(title) > 100:
        title = title[:99] + "…"
    return f"{title} (#{row['short_code'] or draft_id})"


def _resolve_draft(conn: sqlite3.Connection, token: str) -> str:
    """A bare exact ID wins; # always addresses a short code only."""
    short_only = token.startswith("#")
    value = token[1:] if short_only else token
    if not value or any(ch.isspace() for ch in value):
        raise ValueError("Invalid draft identifier.")
    if not short_only:
        row = conn.execute("SELECT id FROM drafts WHERE id = ?", (value,)).fetchone()
        if row:
            return row["id"]
    rows = conn.execute("SELECT id FROM drafts WHERE short_code = ?", (value,)).fetchall()
    if not rows:
        raise ValueError(f"No such draft: {token}.")
    if len(rows) != 1:
        raise ValueError(f"Ambiguous draft identifier: {token}.")
    return rows[0]["id"]


def _decision_token(conn: sqlite3.Connection, draft_id: str) -> str | None:
    row = conn.execute(
        "SELECT d.id, COALESCE(v.version, 0) FROM drafts d "
        "LEFT JOIN telegram_draft_versions v ON v.draft_id=d.id WHERE d.id=?",
        (draft_id,),
    ).fetchone()
    if row is None:
        return None
    # Use the actual monotonic clock (not a truncated hash that can collide).
    # The namespace rejects pre-upgrade callbacks; cap at ten ASCII bytes so
    # UUID-bearing confirmation callbacks stay within Telegram's 64-byte limit.
    token = f"v{row[1]:x}"
    if len(token) > 10:
        raise TelegramError("draft version exceeds Telegram callback capacity")
    return token


def _decision_allowed(conn: sqlite3.Connection, draft_id: str, decision: str, token: str) -> bool:
    if not token or token != _decision_token(conn, draft_id):
        return False
    row = conn.execute("SELECT status FROM drafts WHERE id=?", (draft_id,)).fetchone()
    if not row:
        return False
    if row["status"] == "pending":
        return any(r["id"] == draft_id for r in review.pending_drafts(conn))
    return decision == "reject" and row["status"] == "approved"


def _expired_decision(conn, client, chat_id, cq, draft_id):
    client.answer_callback_query(cq["id"], text="Draft changed; refresh")
    client.send_message(chat_id, f"⚠️ Decision expired for {_label(conn, draft_id)}. No change made; open it again.")


def _send_list_pages(client, chat_id, lines, keyboard, rows_per_item=1):
    """Keep every list entry and its buttons on the same bounded page."""
    header, entries = [], []
    for line in lines:
        if line.startswith("  • "):
            entries.append([line])
        elif entries:
            entries[-1].append(line)
        else:
            header.append(line)
    page, keys = list(header), []
    for i, entry in enumerate(entries):
        item_keys = keyboard[i * rows_per_item : (i + 1) * rows_per_item]
        if keys and (_units("\n".join(page + entry)) > 3800 or len(keys) + len(item_keys) > 16):
            client.send_message(chat_id, "\n".join(page), reply_markup={"inline_keyboard": keys})
            page, keys = list(header), []
        if not keys and _units("\n".join(page + entry)) > 3800:
            _send_long(
                client,
                chat_id,
                "\n".join(page + entry),
                reply_markup={"inline_keyboard": item_keys},
                context=entry[0].splitlines()[0][:200],
            )
            page, keys = list(header), []
            continue
        page.extend(entry)
        keys.extend(item_keys)
    if keys:
        client.send_message(chat_id, "\n".join(page), reply_markup={"inline_keyboard": keys})


# --------------------------------------------------------------------------- #
# Review transport logic (transport-agnostic)                                 #
# --------------------------------------------------------------------------- #


def _send_one(
    conn: sqlite3.Connection, client: TelegramTransport, chat_id: str, row: dict, *, fast: bool = False
) -> list[dict]:
    """Send a single draft (chunked) with Approve/Reject buttons on the last chunk."""
    if not _can_issue(conn, client, chat_id):
        return []
    draft_id = row["id"]
    with review._transaction(conn):
        current = conn.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
        if current is None:
            return []
        pool = db.get_pool_item(conn, current["pool_id"])
        text = render.render_draft(current, pool)
        token = _decision_token(conn, draft_id)
        label = _label(conn, draft_id)
    try:
        return _send_long(
            client, chat_id, text,
            reply_markup=_keyboard(draft_id, fast=fast, token=token or ""),
            context=label,
        )
    except Exception as exc:  # noqa: BLE001 - one bad draft must not kill the loop
        logger.error("[telegram] failed to send draft %s: %s", draft_id, exc)
        return []


def send_pending_drafts(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str) -> list[dict]:
    """Send every pending draft (one message series each) with Approve/Reject buttons.

    Used by the ``--once`` dump mode. The default review loop sends drafts
    one-at-a-time instead (see ``run_review_bot``). Returns all sent messages.
    """
    sent: list[dict] = []
    for row in review.pending_drafts(conn):
        sent.extend(_send_one(conn, client, chat_id, row))
    return sent


def _open_notes_prompt(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str, draft_id: str) -> dict:
    """Open the notes / angle-steer prompt for a draft; return the awaiting entry.

    For a *pending* draft the reply regenerates the angles using the note as
    steering (``regenerate_angles=True``); for an already-approved draft the
    reply is stored as a free note with ``merge=True`` so an existing
    ``editor_line`` is never clobbered. Shared by the ``notes:<id>`` callback
    and the ``/notes <id>`` typed command so the two can't drift.
    """
    if not _can_issue(conn, client, chat_id):
        return {}
    row = conn.execute("SELECT status FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    if row is None:
        client.send_message(chat_id, f"⚠️ No such draft: {_label(conn, draft_id)}.")
        return {"error": True}
    if row["status"] not in ("pending", "approved"):
        client.send_message(chat_id, f"⚠️ Notes unavailable for {_label(conn, draft_id)}: draft is {row['status']}.")
        return {"error": True}
    is_pending = row["status"] == "pending"
    prompt = (
        f"Reply with a note to STEER the angles for pending draft {_label(conn, draft_id)} "
        f"(or /skip to keep the current angles):"
        if is_pending
        else f"Notes for already-approved draft {_label(conn, draft_id)}. Reply here "
        f"(or send /skip to leave the existing notes untouched):"
    )
    version = _decision_token(conn, draft_id)
    note = client.send_message(chat_id, prompt, reply_markup={
        "inline_keyboard": [[{"text": "⏭ Skip notes", "callback_data": f"skipnotes:{draft_id}"}]]
    })
    return {
        "draft_id": draft_id,
        "note_message_id": note["message_id"],
        "stage": "notes",
        "decision": "approve",
        "regenerate_angles": bool(is_pending),
        "version": version,
    }


def handle_callback(
    conn: sqlite3.Connection,
    client: TelegramTransport,
    chat_id: str,
    update: dict,
    *,
    fast: bool = False,
    awaiting: dict | None = None,
) -> dict | None:
    """Process an inline-button tap.

    - ``approve:<id>`` / ``reject:<id>``: record the decision, then prompt for
      optional notes (reply or /skip).
    - ``notes:<id>``: from /listapproved -- prompt for notes on an already
      approved draft (re-applying approve is idempotent and keeps its queue row).
    - ``view:<id>``: from /listapproved -- open the draft's full content (with an
      inline 'Add notes' button on the last chunk).
    - ``remove:<id>``: from /listqueue -- pull a draft back out of the queue
      (kept approved), the button equivalent of ``/queue remove <id>``.

    Returns a dict describing the action so the caller can track note state.
    """
    cq = update.get("callback_query")
    if not cq:
        return None
    if not _can_issue(conn, client, chat_id):
        client.answer_callback_query(cq["id"], text="Transaction open; retry")
        return None
    data = (cq.get("data") or "").strip()
    if data.startswith("export:"):
        draft_id, sep, token = data[len("export:"):].rpartition(":")
        with review._transaction(conn):
            if not sep or not token or token != _decision_token(conn, draft_id):
                _expired_decision(conn, client, chat_id, cq, draft_id or data[len("export:"):])
                return None
            row = next((r for r in review.queued_drafts(conn) if r["draft_id"] == draft_id), None)
            if row is None:
                _expired_decision(conn, client, chat_id, cq, draft_id)
                return None
            label = _label(conn, draft_id)
            export = (review.format_publish_export(conn, [row], include_image_path=False)
                      if row.get("post_copy") else None)
            if export:
                short = row.get("short_code") or draft_id
                export = export.replace(f"── {short} — {row['title']}", f"── {label}", 1)
        client.answer_callback_query(cq["id"], text="")
        if export is None:
            client.send_message(chat_id, f"⚠️ {label} has no post copy to export. Edit or regenerate copy first.")
        else:
            chunks = _chunk_text(export)
            if len(chunks) > 1:
                chunks = _chunk_text(export, 4000 - _units(label) - 2)
            urls = [m.group() for m in re.finditer(r"https?://[^\s]+", export)]
            split_urls = {url for url in urls if _units(url) > 3700 or not any(url in chunk for chunk in chunks)}
            if split_urls:
                try:
                    client.send_document(chat_id, export.encode("utf-8"), "Full queue export (UTF-8 text)")
                except Exception:  # noqa: BLE001 - upload outcome can be ambiguous
                    client.send_message(
                        chat_id, "⚠️ Could not confirm the export document was sent. "
                        "Check this chat before trying the export again; no post was made."
                    )
                    return None
                export = re.sub(
                    r"https?://[^\s]+",
                    lambda m: "[Full URL in attached text document]",
                    export,
                )
            _send_long(client, chat_id, export, context=label, reply_markup={"inline_keyboard": [[
                {"text": "✅ Mark manually posted", "callback_data": f"posted:{draft_id}:{token}"}
            ]]})
        return None
    if data.startswith(("posted:", "postyes:", "unpost:")):
        action, _, rest = data.partition(":")
        draft_id, sep, token = rest.rpartition(":")
        timestamp = ""
        new_version = ""
        try:
            claim = (
                _claim_intent(token, chat_id, draft_id, "posted")
                if action == "postyes" else contextlib.nullcontext(token)
            )
            with claim as confirmed_token, review._transaction(conn):
                token = confirmed_token
                if token is None:
                    _expired_decision(conn, client, chat_id, cq, draft_id or rest)
                    return None
                if not sep or token != _decision_token(conn, draft_id):
                    _expired_decision(conn, client, chat_id, cq, draft_id or rest)
                    return None
                row = review._publication_row(conn, draft_id)
                if action in ("posted", "postyes"):
                    if row["published"] or row["published_at"] is not None:
                        raise ValueError("publication state changed; refresh the queue")
                    if action == "postyes":
                        timestamp = review.mark_posted(conn, draft_id)
                        new_version = _decision_token(conn, draft_id)
                else:
                    if not row["published"] or row["published_at"] is None:
                        raise ValueError("no timestamped manual publication to undo")
                    timestamp = row["published_at"]
                    review.undo_posted(conn, draft_id, expected_published_at=timestamp)
        except ValueError as exc:
            client.answer_callback_query(cq["id"], text="Cannot change publication")
            client.send_message(chat_id, f"⚠️ {_label(conn, draft_id)}: {exc}. No change made.")
            return None
        client.answer_callback_query(cq["id"], text="")
        if action == "posted":
            token = _new_intent(chat_id, draft_id, "posted", token)
            client.send_message(
                chat_id,
                f"⚠️ Mark {_label(conn, draft_id)} as manually posted? This does not post remotely.",
                reply_markup={
                    "inline_keyboard": [
                        [
                            {"text": "✅ Confirm posted", "callback_data": f"postyes:{draft_id}:{token}"},
                            {"text": "↩️ Cancel", "callback_data": f"cancel:{draft_id}:{token}"},
                        ]
                    ]
                },
            )
        elif action == "postyes":
            client.send_message(
                chat_id,
                f"✅ {_label(conn, draft_id)} marked manually posted at {timestamp} (UTC). "
                "This records your action; no remote post was made. /listposted",
                reply_markup={
                    "inline_keyboard": [
                        [{"text": "↩️ Undo posted", "callback_data": f"unpost:{draft_id}:{new_version}"}]
                    ]
                },
            )
        else:
            client.send_message(
                chat_id,
                f"↩️ Undid manual publication of {_label(conn, draft_id)} "
                f"recorded at {timestamp}. It is back in /listqueue.",
            )
        return None
    if data.startswith(("fastapprove:", "fastreject:", "approve:", "reject:")):
        action, _, rest = data.partition(":")
        draft_id, sep, token = rest.rpartition(":")
        decision = action.removeprefix("fast")
        if not sep or not _decision_allowed(conn, draft_id, decision, token):
            _expired_decision(conn, client, chat_id, cq, draft_id or rest)
            return None
        if action.startswith("fast") or fast:
            return _commit_decision(
                conn, client, chat_id, cq, draft_id, decision, token, fast=action.startswith("fast") or fast
            )
        token = _new_intent(chat_id, draft_id, decision, token)
        client.answer_callback_query(cq["id"], text="")
        client.send_message(
            chat_id,
            f"⚠️ Confirm: {decision} draft {_label(conn, draft_id)}?",
            reply_markup=_confirm_keyboard(draft_id, decision, token),
        )
        return {"draft_id": draft_id, "decision": decision, "confirming": True}
    if data.startswith("confirm:"):
        parts = data.split(":")
        if len(parts) != 4 or parts[1] not in ("approve", "reject"):
            _expired_decision(conn, client, chat_id, cq, parts[2] if len(parts) > 2 else "")
            return None
        _, decision, draft_id, nonce = parts
        return _commit_decision(conn, client, chat_id, cq, draft_id, decision, None, nonce=nonce)
    if data.startswith("cancel:"):
        rest = data.partition(":")[2]
        draft_id, sep, nonce = rest.rpartition(":")
        with _intent_lock:
            entry = _intents.get(nonce) if sep else None
            valid = bool(entry and entry[:2] == (str(chat_id), draft_id))
            if valid:
                del _intents[nonce]
        if not valid:
            _expired_decision(conn, client, chat_id, cq, draft_id or rest)
            return None
        client.answer_callback_query(cq["id"], text="cancelled")
        client.send_message(
            chat_id,
            f"↩️ Decision for {_label(conn, draft_id)} cancelled — no change made. "
            f"Tap ✅/❌ again when you're sure, or /reviewdraft to move on.",
        )
        return {"draft_id": draft_id, "cancelled": True}
    if data.startswith("notes:"):
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="add notes")
        return _open_notes_prompt(conn, client, chat_id, draft_id)
    if data.startswith("skipjoke:"):
        draft_id = data.partition(":")[2]
        entry = (awaiting or {}).get(draft_id, {})
        message_id = (cq.get("message") or {}).get("message_id")
        if (awaiting is None or entry.get("stage") != "editor_line" or message_id is None
                or entry.get("note_message_id") != message_id):
            client.answer_callback_query(cq["id"], text="Prompt expired")
            return None
        client.answer_callback_query(cq["id"], text="skipped")
        return handle_text(conn, client, chat_id, awaiting, {
            "message": {"text": "/skip", "reply_to_message": {"message_id": message_id}}
        })
    if data.startswith("skipnotes:"):
        # ⏭ Skip (no notes): dismiss the optional notes prompt. Equivalent to a
        # /skip reply, but a one-tap button so the reviewer never has to type it.
        _, _, draft_id = data.partition(":")
        entry = (awaiting or {}).get(draft_id, {})
        message_id = (cq.get("message") or {}).get("message_id")
        if entry.get("stage") != "notes" or message_id is None or entry.get("note_message_id") != message_id:
            client.answer_callback_query(cq["id"], text="Prompt expired")
            client.send_message(chat_id, f"⚠️ Notes prompt expired for {_label(conn, draft_id)}; no change made.")
            return None
        client.answer_callback_query(cq["id"], text="skipped")
        client.send_message(
            chat_id, f"⏭️ No notes saved for {_label(conn, draft_id)}.",
            reply_markup=_next_keyboard(draft_id, fast=entry.get("fast", False))
        )
        return {"skipped": draft_id}

    if data.startswith(("later:", "fastlater:")):
        rest = data.partition(":")[2]
        draft_id, sep, token = rest.rpartition(":")
        try:
            with review._transaction(conn):
                if not sep or not _decision_allowed(conn, draft_id, "approve", token):
                    _expired_decision(conn, client, chat_id, cq, draft_id or rest)
                    return None
                until = (datetime.now(UTC) + timedelta(days=30)).isoformat()
                conn.execute("UPDATE drafts SET defer_until=? WHERE id=?", (until, draft_id))
        except ValueError:
            client.answer_callback_query(cq["id"], text="cannot defer")
            return None
        client.answer_callback_query(cq["id"], text="deferred 30d")
        client.send_message(
            chat_id,
            f"⏸ Draft {_label(conn, draft_id)} deferred for 30 days (it'll resurface in the review queue then).",
            reply_markup=_next_keyboard(draft_id, fast=data.startswith("fastlater:")),
        )
        return {"draft_id": draft_id, "deferred": True}
    if data.startswith("view:"):
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="opened")
        send_draft_content(conn, client, chat_id, draft_id)
        return None
    if data.startswith("copy:"):
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="opened copy")
        send_copy_content(conn, client, chat_id, draft_id)
        return None
    if data.startswith("remove:"):
        rest = data.partition(":")[2]
        draft_id, sep, token = rest.rpartition(":")
        with review._transaction(conn):
            if not sep or token != _decision_token(conn, draft_id):
                _expired_decision(conn, client, chat_id, cq, draft_id or rest)
                return None
            removed = review.remove_from_queue(conn, draft_id)
        client.answer_callback_query(cq["id"], text="removed from queue")
        if removed:
            client.send_message(
                chat_id,
                f"↩️ {_label(conn, draft_id)} removed from the queue (kept approved). List it again with /listqueue.",
            )
        else:
            client.send_message(chat_id, f"⚠️ {_label(conn, draft_id)} was not in the queue.")
        return None
    if data.startswith("reopen:"):
        rest = data.partition(":")[2]
        draft_id, sep, token = rest.rpartition(":")
        try:
            with review._transaction(conn):
                if not sep or token != _decision_token(conn, draft_id):
                    _expired_decision(conn, client, chat_id, cq, draft_id or rest)
                    return None
                review.reopen_draft(conn, draft_id)
        except ValueError as exc:
            client.send_message(chat_id, f"Cannot reopen {_label(conn, draft_id)}: {exc}")
            return None
        client.answer_callback_query(cq["id"], text="reopened for re-review")
        client.send_message(
            chat_id,
            f"↩️ {_label(conn, draft_id)} sent back to pending for re-review. List it with /reviewdraft.",
        )
        return None
    if data.startswith("reviewnow:"):
        # GAP 4: bring a deferred draft forward for immediate review.
        rest = data.partition(":")[2]
        draft_id, sep, token = rest.rpartition(":")
        try:
            with review._transaction(conn):
                if not sep or token != _decision_token(conn, draft_id):
                    _expired_decision(conn, client, chat_id, cq, draft_id or rest)
                    return None
                row = conn.execute("SELECT defer_until FROM drafts WHERE id=?", (draft_id,)).fetchone()
                if not row or not row["defer_until"]:
                    raise ValueError("draft is not deferred")
                conn.execute("UPDATE drafts SET defer_until=NULL WHERE id=?", (draft_id,))
        except ValueError as exc:
            client.send_message(chat_id, f"Cannot bring forward {_label(conn, draft_id)}: {exc}")
            return None
        client.answer_callback_query(cq["id"], text="brought forward")
        client.send_message(
            chat_id,
            f"⏩ {_label(conn, draft_id)} brought forward — it's back in the review queue. "
            f"Send /reviewdraft to see it (or tap ✅/❌ next time one is shown).",
        )
        return None
    if data.startswith("setjoke:"):
        # A stuck-capture nudge (from /status) or the 📝 Add joke button: the
        # draft is approved+queued but its one-line joke (editor_line) is blank.
        # Prompt for the joke directly. The reply is captured by handle_text's
        # editor_line stage with setjoke=True, which calls review.set_editor_line
        # — a targeted UPDATE that fills editor_line WITHOUT touching notes/queue.
        #
        # The OLD code opened a copy-editing menu (regencopy/editcopy) here, which
        # only wrote queue.post_copy and never touched editor_line — so the
        # stuck-capture nudge never cleared. Copy editing stays reachable via the
        # 📝 Copy / editmenu buttons on an approved draft instead.
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="add joke")
        version = _decision_token(conn, draft_id)
        note = client.send_message(
            chat_id,
            f"Reply with the one-line joke (editor_line) for approved "
            f"draft {_label(conn, draft_id)} — the human voice that steers post copy "
            f"(or /skip to leave it blank):",
        )
        return {
            "draft_id": draft_id,
            "note_message_id": note["message_id"],
            "stage": "editor_line",
            "decision": "approve",
            "setjoke": True,
            "version": version,
        }
    if data.startswith("editmenu:"):
        # GAP fix: the ✏️ Edit button now opens a CHOICE menu (Regenerate vs type
        # manually) instead of dropping straight into a bare reply box. Regenerate
        # always works — notes are optional, so it's offered even when nothing has
        # been added as an editor note yet.
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="edit options")
        menu = {
            "inline_keyboard": [
                [
                    {
                        "text": "🔄 Regenerate copy",
                        "callback_data": f"regencopy:{draft_id}:{_decision_token(conn, draft_id)}",
                    },
                    {
                        "text": "✏️ Type copy myself",
                        "callback_data": f"editcopy:{draft_id}:{_decision_token(conn, draft_id)}",
                    },
                ]
            ]
        }
        client.send_message(
            chat_id,
            f"How do you want to change the post copy for {_label(conn, draft_id)}?\n"
            f"• 🔄 Regenerate — ask the LLM for fresh copy (works with or without notes)\n"
            f"• ✏️ Type myself — reply with your own copy",
            reply_markup=menu,
        )
        return None
    if data.startswith("editcopy:"):
        rest = data.partition(":")[2]
        draft_id, sep, token = rest.rpartition(":")
        if not sep or token != _decision_token(conn, draft_id):
            _expired_decision(conn, client, chat_id, cq, draft_id or rest)
            return None
        client.answer_callback_query(cq["id"], text="editing copy")
        version = token
        prompt = client.send_message(
            chat_id,
            f"Reply with the new post copy for {_label(conn, draft_id)} (or send /cancel to keep the current version):",
        )
        return {"draft_id": draft_id, "editcopy_message_id": prompt["message_id"], "version": version}
    if data.startswith("regencopy:"):
        rest = data.partition(":")[2]
        draft_id, sep, token = rest.rpartition(":")
        if not sep or token != _decision_token(conn, draft_id):
            _expired_decision(conn, client, chat_id, cq, draft_id or rest)
            return None
        client.answer_callback_query(cq["id"], text="regenerating")
        from humorhist.copywriter import fill_post_copy

        llm = _get_llm(chat_id, client)
        if llm is None:
            return None
        try:
            n = fill_post_copy(conn, llm, draft_id=draft_id, force=True)
        except Exception as exc:  # noqa: BLE001
            client.send_message(chat_id, f"Regeneration failed for {_label(conn, draft_id)}: {exc}")
            return None
        if n == 0:
            client.send_message(
                chat_id,
                f"Nothing generated for {_label(conn, draft_id)} (no LLM key available?). The copy is unchanged.",
            )
            return None
        send_copy_content(conn, client, chat_id, draft_id)
        return None
    return None


def _commit_decision(
    conn: sqlite3.Connection,
    client: TelegramTransport,
    chat_id: str,
    cq: dict,
    draft_id: str,
    decision: str,
    token: str | None,
    *,
    nonce: str | None = None,
    fast: bool = False,
) -> dict | None:
    """Commit an approve/reject and run the decision-specific follow-up.

    Shared by the confirm-gate path (``confirm:<decision>:<id>``) and the
    ``/reviewdraft fast`` path (first tap commits directly). Returns the result
    dict the caller uses to track note/joke capture state.
    """
    try:
        claim = _claim_intent(nonce, chat_id, draft_id, decision) if nonce else contextlib.nullcontext(token)
        with claim as confirmed_token, review._transaction(conn):
            if not confirmed_token or not _decision_allowed(conn, draft_id, decision, confirmed_token):
                _expired_decision(conn, client, chat_id, cq, draft_id)
                return None
            review.apply_review(conn, draft_id, decision=decision)
            version = _decision_token(conn, draft_id)
    except ValueError:
        client.answer_callback_query(cq["id"], text="already handled")
        return None
    client.answer_callback_query(cq["id"], text=f"{decision}d")
    if decision == "reject":
        # A reject needs no human-voice capture — the joke is the point of
        # an *approve*. Just confirm and finish (no editor_line/notes prompt,
        # no extra round-trips). The draft is already rejected + dequeued.
        client.send_message(
            chat_id,
            f"🚫 Draft {_label(conn, draft_id)} rejected. It's out of the queue; "
            f"re-list it any time with /listrejected (or /reviewdraft if "
            f"you change your mind later).",
            reply_markup=_next_keyboard(draft_id, fast=fast),
        )
        return {"draft_id": draft_id, "decision": "reject", "committed": True}
    # The joke is the whole point of the product: capture the human-written
    # editor_line first. A reply to this prompt becomes editor_line (which
    # also steers B+ post-copy generation); /skip leaves it blank.
    note = client.send_message(
        chat_id,
        f"Draft {_label(conn, draft_id)} approved. Reply here with your one-line "
        f"joke (the editor_line) — or send /skip to leave it blank. "
        "Replying or skipping automatically attempts LLM post-copy generation if an LLM is available; "
        "this may incur charges:",
        reply_markup={"inline_keyboard": [[{"text": "⏭ Skip joke", "callback_data": f"skipjoke:{draft_id}"}]]},
    )
    # Stash the decision so a follow-up reply knows whether to also capture
    # a (secondary) notes step after the editor_line.
    return {
        "draft_id": draft_id,
        "decision": decision,
        "note_message_id": note["message_id"],
        "stage": "editor_line",
        "fast": fast,
        "committed": True,
        "version": version,
    }


def handle_text(
    conn: sqlite3.Connection,
    client: TelegramTransport,
    chat_id: str,
    awaiting: dict,
    update: dict,
    *,
    image_dir: str | None = None,
) -> dict | None:
    """Process a reply to a review prompt.

    There are three prompt kinds tracked in ``awaiting``:

    * ``editcopy``  — a reply replaces the post copy (handled, unchanged).
    * ``editor_line`` (stage) — the first reply after an approve/reject tap.
      Stored as ``editor_line`` (the human joke; this steers B+ copy). A
      ``/skip`` leaves it blank. After capturing it we prompt for *optional*
      longer notes (the ``notes`` stage).
    * ``notes`` (stage) — a secondary reply (or the ``/listapproved`` "Add
      notes" button). Stored as ``editor_notes`` with ``merge=True`` so a
      notes-only save never clobbers an already-entered ``editor_line``.

    ``/skip`` at any note/editor_line prompt clears it without storing.
    Falls back to the single open prompt when the user types instead of
    replying to the prompt.
    """
    msg = update.get("message")
    if not msg or "text" not in msg:
        return None
    if not _can_issue(conn, client, chat_id):
        return None
    reply_to = (msg.get("reply_to_message") or {}).get("message_id")
    text = msg["text"].strip()

    # Resolve which draft (and which kind of prompt) this reply answers.
    draft_id = None
    st = None
    for did, s in awaiting.items():
        if reply_to is not None and s.get("editcopy_message_id") == reply_to:
            draft_id, st = did, s
            break
        if reply_to is not None and s.get("note_message_id") == reply_to:
            draft_id, st = did, s
            break
    # Fallback: a single open prompt the user typed at rather than replied to.
    # /skip and /cancel are valid replies to a prompt, so they resolve here too.
    if draft_id is None and reply_to is None and len(awaiting) == 1:
        did = next(iter(awaiting))
        draft_id, st = did, awaiting[did]
    if draft_id is None:
        return None

    # A prompt can outlive a decision in another message. Never let a stale
    # reply silently re-approve a rejected/reopened draft.
    row = conn.execute("SELECT status FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    expected = "pending" if (st or {}).get("regenerate_angles") else "approved"
    if (
        not row
        or not st
        or row["status"] != expected
        or not st.get("version")
        or st["version"] != _decision_token(conn, draft_id)
    ):
        awaiting.pop(draft_id, None)
        client.send_message(
            chat_id, f"⚠️ Prompt expired for {_label(conn, draft_id)}; draft status changed. No change made."
        )
        return {"expired": draft_id}

    # editcopy prompt: replace the post copy (unchanged behaviour).
    if st is not None and st.get("editcopy_message_id") is not None:
        if text == "/cancel":
            awaiting.pop(draft_id, None)
            client.send_message(chat_id, f"Kept the existing copy for {_label(conn, draft_id)}.")
            return {"editcopy_cancelled": draft_id}
        from humorhist.copywriter import set_post_copy

        with review._transaction(conn):
            if st["version"] != _decision_token(conn, draft_id):
                awaiting.pop(draft_id, None)
                client.send_message(chat_id, f"⚠️ Prompt expired for {_label(conn, draft_id)}. No change made.")
                return {"expired": draft_id}
            updated = set_post_copy(conn, draft_id, text)
        awaiting.pop(draft_id, None)
        if not updated:
            client.send_message(chat_id, f"Post copy not saved for {_label(conn, draft_id)}: no unpublished queue row.")
            return {"editcopy_failed": draft_id}
        client.send_message(chat_id, f"Post copy saved for {_label(conn, draft_id)}.")
        return {"editcopy_saved": draft_id}

    stage = (st or {}).get("stage", "notes")
    decision = (st or {}).get("decision", "approve")

    # --- editor_line stage: capture the human joke (the product's point) ---
    if stage == "editor_line":
        editor_line = None if text == "/skip" else text
        if (st or {}).get("setjoke"):
            # A "stuck capture" fix: the draft is already approved + queued.
            # Just set the joke (no re-commit, no re-enqueue, no copy regen).
            with review._transaction(conn):
                if st["version"] != _decision_token(conn, draft_id):
                    awaiting.pop(draft_id, None)
                    client.send_message(chat_id, f"⚠️ Prompt expired for {_label(conn, draft_id)}. No change made.")
                    return {"expired": draft_id}
                conn.execute(
                    "UPDATE drafts SET editor_line=? WHERE id=?", ((editor_line or "").strip() or None, draft_id)
                )
            awaiting.pop(draft_id, None)
            if editor_line is None:
                client.send_message(
                    chat_id, f"Joke left blank for {_label(conn, draft_id)}.", reply_markup=_next_keyboard(draft_id)
                )
            else:
                client.send_message(
                    chat_id, f"Joke saved for {_label(conn, draft_id)}.", reply_markup=_next_keyboard(draft_id)
                )
            return {"editor_line_set": draft_id}
        with review._transaction(conn):
            if st["version"] != _decision_token(conn, draft_id):
                awaiting.pop(draft_id, None)
                client.send_message(chat_id, f"⚠️ Prompt expired for {_label(conn, draft_id)}. No change made.")
                return {"expired": draft_id}
            review.apply_review(conn, draft_id, decision=decision, editor_line=editor_line)
        # B+ handoff: on approve, generate initial post copy now that we have
        # the editor_line to steer it. Best-effort (missing LLM key => skip).
        if decision == "approve":
            from humorhist.copywriter import fill_post_copy

            llm = _get_llm(chat_id, client, silent=True)
            if llm is not None:
                try:
                    fill_post_copy(conn, llm, draft_id=draft_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[telegram] post copy not generated for %s: %s", draft_id, exc)
        # secondary optional notes step
        version = _decision_token(conn, draft_id)
        note = client.send_message(
            chat_id,
            f"Got it. Optionally reply with longer notes for {_label(conn, draft_id)} (or /skip to finish):",
            reply_markup={
                "inline_keyboard": [
                    [{"text": "⏭ Skip (no notes)", "callback_data": f"skipnotes:{draft_id}"}],
                ]
            },
        )
        awaiting[draft_id] = {
            "note_message_id": note["message_id"],
            "stage": "notes",
            "fast": st.get("fast", False),
            "decision": decision,
            "version": version,
        }
        return {"editor_line_set": draft_id}

    # --- notes stage: optional free-form annotation (merge, don't clobber) ---
    if text == "/skip":
        awaiting.pop(draft_id, None)
        client.send_message(
            chat_id, f"Notes left blank for {_label(conn, draft_id)}.",
            reply_markup=_next_keyboard(draft_id, fast=st.get("fast", False))
        )
        return {"skipped": draft_id}
    # If this note came from a *pending* draft's "steer angles" prompt, regenerate
    # the angles with the note as steering (the /notes -> angles behaviour).
    if (st or {}).get("regenerate_angles"):
        from humorhist.brief import _row_to_item, generate_angles

        llm = _get_llm(chat_id, client)
        if llm is None:
            return {"angle_regen_failed": draft_id}
        try:
            draft = conn.execute("SELECT pool_id, brief_json FROM drafts WHERE id=?", (draft_id,)).fetchone()
            if draft is None:
                raise ValueError("draft no longer exists")
            item = _row_to_item(db.get_pool_item(conn, draft["pool_id"]))
            angles = generate_angles(llm, item, json.loads(draft["brief_json"] or "{}"), steering_note=text)
            with review._transaction(conn):
                if st["version"] != _decision_token(conn, draft_id):
                    awaiting.pop(draft_id, None)
                    client.send_message(chat_id, f"⚠️ Prompt expired for {_label(conn, draft_id)}. No change made.")
                    return {"expired": draft_id}
                conn.execute(
                    "UPDATE drafts SET angles_json=?, editor_notes=? WHERE id=?",
                    (json.dumps(angles, ensure_ascii=False), text.strip() or None, draft_id),
                )
        except Exception as exc:  # noqa: BLE001 - keep the loop alive on LLM failure
            client.send_message(chat_id, f"Angle regen failed for {_label(conn, draft_id)}: {exc}")
            return {"angle_regen_failed": draft_id}
        awaiting.pop(draft_id, None)
        client.send_message(
            chat_id,
            f"🔄 Angles regenerated for {_label(conn, draft_id)} using your note as steering. "
            "Review them with /reviewdraft.",
            reply_markup=_next_keyboard(draft_id),
        )
        return {"angles_regenerated": draft_id}
    # merge=True keeps the editor_line we just captured; only notes change.
    with review._transaction(conn):
        if st["version"] != _decision_token(conn, draft_id):
            awaiting.pop(draft_id, None)
            client.send_message(chat_id, f"⚠️ Prompt expired for {_label(conn, draft_id)}. No change made.")
            return {"expired": draft_id}
        review.apply_review(conn, draft_id, decision=decision, notes=text, merge=True)
    awaiting.pop(draft_id, None)
    client.send_message(
        chat_id, f"Notes saved for {_label(conn, draft_id)}.",
        reply_markup=_next_keyboard(draft_id, fast=st.get("fast", False))
    )
    return {"noted": draft_id}


HELP_TEXT = (
    "HumorHist review bot — the whole pipeline runs from here.\n\n"
    "QUICK START: /reviewdraft to decide on pending drafts · /listqueue to manage "
    "approved ones · /status for progress · /help for everything.\n\n"
    "Tip: /reviewdraft fast commits on the first tap (skips the Confirm gate) — "
    "undo from /listapproved or /listqueue.\n\n"
    "DISCOVERY & DRAFTING\n"
    "/harvest - top up the candidate pool from seed + Wikipedia lists\n"
    "/screen [limit] - LLM-score unscored pool candidates (funny_score)\n"
    "/screen status - inspect blocked screening attempts (no LLM request)\n"
    "/draft [count] [min_score] - fact-check + generate angles for top candidates\n"
    "/suggest <topic> - add an editor-suggested event to the pool\n\n"
    "REVIEW\n"
    "/reviewdraft - show the next pending draft (non-modal)\n"
    "    tap the ⏭ Next button (or send /reviewdraft again) to advance; each tap opens a Confirm/Cancel gate\n"
    "/reviewdraft fast - same, but commits on the first tap (no gate); undo via /listapproved or /listqueue\n"
    "    each draft has ✅ Approve / ❌ Reject / ⏸ Later buttons\n"
    "    on Approve confirm, the bot asks for your one-line joke, then notes\n"
    "    /later <id> defers a pending draft 30 days (#id shown in /listapproved)\n"
    "    /listlater - list deferred drafts; tap to bring one forward now\n"
    "    /reviewnow [<id>] - bring a deferred draft (or ALL) back for review\n"
    "/setjoke <id> - set the one-line joke on an approved draft (fixes a blank capture)\n"
    "/jokes - list approved drafts missing their one-line joke (tap to fill one)\n"
    "/notes <id> - steer angles (pending draft) or add notes (approved); reply to the prompt\n"
    "/listapproved - list approved drafts with their #id; tap one to open it\n"
    "    each row also has a ❌ Reject button (confirm-gated) to un-approve it\n"
    "/listrejected - list rejected drafts; tap one to reopen for re-review\n"
    "/listqueue - list queued drafts with their #id, copy + Remove/Reject/Reopen\n"
    "    Mark posted records a manual publication after confirmation; /listposted shows undo controls\n"
    "/listposted - list recorded publications and undo timestamped manual records\n"
    "    each row: ✏️ Edit copy, 🗑 Remove (keep approved),\n"
    "             ❌ Reject (with confirm), ↩️ Reopen (back to pending)\n"
    "    /queue remove <id> pulls a draft out of the queue (kept approved)\n"
    "/view <id> - re-read any draft's full content (pending/approved/rejected)\n"
    "    (use the #id shown by /listapproved or /listqueue)\n"
    "/viewcopy <id> - open a queued draft's post copy (✏️ Edit / 🔄 Regenerate)\n"
    "BUFFER & STATUS\n"
    "/buffer - buffer health report + on-demand top-up\n"
    "/buffer enqueue - also sweep approved drafts into the queue\n"
    "/queue - list the publish queue (approved+queued drafts)\n"
    "/queue enqueue - sweep approved drafts into the queue\n"
    "/queue remove <id> - pull a draft back out of the queue (kept approved)\n"
    "/image <id> - (re)generate the story image for an approved+queued draft\n"
    "/status - approved / rejected / pending breakdown (+ stuck-capture nudge)\n"
    "/publish - read-only export of queued posts (copy + link + image + checklist) for manual posting\n"
    "/menu or /start - open the guided menu\n"
    "/help - this message\n\n"
    "The bot also nudges you with 🆕 N new draft(s) when fresh drafts appear."
)


def send_approved_list(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str) -> int:
    """DM a list of approved drafts, each with an inline 'view' button.

    Tapping a draft opens its full content (see ``send_draft_content``), from
    which the reviewer can optionally add notes. Returns the count listed.
    """
    if not _can_issue(conn, client, chat_id):
        return 0
    rows = review.approved_drafts(conn)
    if not rows:
        client.send_message(
            chat_id,
            "No approved drafts yet. Run /reviewdraft to review pending ones, or /status for progress.",
        )
        return 0
    lines = [
        "✅ Approved drafts (tap to open; #id is the draft number):",
        "  👁 Open draft   ❌ Reject (confirm-gated, sends back to pending)",
    ]
    keyboard = []
    for r in rows:
        title = " ".join((r["title"] or "(unknown)").split())
        title = title if len(title) <= 100 else title[:99] + "…"
        did = r["draft_id"]
        short = r.get("short_code") or did
        lines.append(f"  • {title} (#{short})")
        keyboard.append(
            [
                {"text": f"👁 Open: {title[:18]} (#{short})", "callback_data": f"view:{did}"},
                {
                    "text": f"❌ Reject: {title[:16]} (#{short})",
                    "callback_data": f"reject:{did}:{_decision_token(conn, did)}",
                },
            ]
        )
    _send_list_pages(client, chat_id, lines, keyboard)
    return len(rows)


def send_rejected_list(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str) -> int:
    """DM a list of rejected drafts, each with an inline 'reopen' button.

    Tapping a draft sends it back to pending for re-review (the editor's undo
    for a mistaken reject). Returns the count listed.
    """
    if not _can_issue(conn, client, chat_id):
        return 0
    rows = review.rejected_drafts(conn)
    if not rows:
        client.send_message(
            chat_id,
            "No rejected drafts. Reviewed drafts land here only after you reject "
            "them from /reviewdraft or /listapproved.",
        )
        return 0
    lines = [
        "❌ Rejected drafts (tap to send one back to pending for re-review; #id is the draft number):",
        "  ↩️ Reopen to pending",
    ]
    keyboard = []
    for r in rows:
        title = " ".join((r["title"] or "(unknown)").split())
        title = title if len(title) <= 100 else title[:99] + "…"
        did = r["draft_id"]
        short = r.get("short_code") or did
        lines.append(f"  • {title} (#{short})")
        keyboard.append(
            [
                {
                    "text": f"↩️ Reopen: {title[:18]} (#{short})",
                    "callback_data": f"reopen:{did}:{_decision_token(conn, did)}",
                }
            ]
        )
    _send_list_pages(client, chat_id, lines, keyboard)
    return len(rows)


def send_deferred_list(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str) -> int:
    """DM a list of deferred drafts, each with an inline 'review now' button.

    Tapping a draft clears its defer window (``/later``) and returns it to the
    review surface immediately — the missing "review now" shortcut for a
    deferred draft (GAP 4). Returns the count listed.
    """
    if not _can_issue(conn, client, chat_id):
        return 0
    rows = review.deferred_drafts(conn)
    if not rows:
        client.send_message(
            chat_id,
            "No deferred drafts. (/later defers one 30 days from /reviewdraft or /listapproved.)",
        )
        return 0
    lines = [
        "⏸ Deferred drafts (tap to clear the 30-day wait and bring one back to review now; #id is the draft number):",
        "  ⏩ Review now",
    ]
    keyboard = []
    for r in rows:
        title = " ".join((r["title"] or "(unknown)").split())
        title = title if len(title) <= 100 else title[:99] + "…"
        did = r["draft_id"]
        short = r.get("short_code") or did
        lines.append(f"  • {title} (#{short})")
        keyboard.append(
            [
                {
                    "text": f"⏩ Review now: {title[:16]} (#{short})",
                    "callback_data": f"reviewnow:{did}:{_decision_token(conn, did)}",
                }
            ]
        )
    _send_list_pages(client, chat_id, lines, keyboard)
    return len(rows)


def send_draft_content(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str, draft_id: str) -> list[dict]:
    """Send a single draft's full content (chunked), with an 'Add notes' button.

    Used when the reviewer opens an approved draft from /listapproved. The last
    chunk carries an inline 'Add notes' button (callback ``notes:<id>``) so they
    can annotate it without leaving Telegram. It also appends the current post
    copy (with its ``N/limit`` char count) and a '📝 Copy' button that opens the
    edit/regenerate view — so opening an approved draft shows the caption too.
    """
    if not _can_issue(conn, client, chat_id):
        return []
    from humorhist.copywriter import char_limit

    row = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    if row is None:
        client.send_message(chat_id, f"No such draft: {draft_id}")
        return []
    pool = db.get_pool_item(conn, row["pool_id"])
    text = render.render_draft(row, pool)

    q = conn.execute("SELECT post_copy FROM queue WHERE draft_id = ?", (draft_id,)).fetchone()
    limit = char_limit()
    copy = (dict(q) if q else {}).get("post_copy") if q else None
    if copy:
        copy_block = f"\n\n📝 POST COPY ({len(copy)}/{limit} chars):\n{copy}\n(tap 📝 Copy to edit or regenerate)"
    else:
        copy_block = f"\n\n📝 POST COPY: (none yet — 0/{limit} chars)\n(tap 📝 Copy to generate / edit)"
    text = text + copy_block

    # 'Learn more' link: a shortened reader-facing pointer to the source article.
    try:
        from humorhist.db import get_source_link, shorten_url

        link = get_source_link(conn, draft_id)
        if link:
            text = text + f"\n\n🔗 Learn more: {shorten_url(link, shorten=False)}"
    except Exception as exc:  # noqa: BLE001 - link is a bonus; never break the view
        logger.warning("[telegram] failed to render source link for %s: %s", draft_id, exc)

    notes_btn = {
        "inline_keyboard": [
            [
                {"text": "✏️ Add notes", "callback_data": f"notes:{draft_id}"},
                {"text": "📝 Copy", "callback_data": f"copy:{draft_id}"},
            ]
        ]
    }
    try:
        sent = _send_long(client, chat_id, text, reply_markup=notes_btn, context=_label(conn, draft_id))
    except Exception as exc:  # noqa: BLE001
        logger.error("[telegram] failed to send draft content %s: %s", draft_id, exc)
        return []
    # A+B: show the generated story image alongside the content, if present.
    try:
        info = db.get_image(conn, draft_id)
        if info and info.get("image_path"):
            img_path = info["image_path"]
            if Path(img_path).is_file():
                with open(img_path, "rb") as fh:
                    client.send_photo(
                        chat_id,
                        fh.read(),
                        caption=f"🖼️ Story image for {_label(conn, draft_id)}",
                    )
    except Exception as exc:  # noqa: BLE001 - image is a bonus; never break the view
        logger.error("[telegram] failed to send story image %s: %s", draft_id, exc)
    return sent


# --------------------------------------------------------------------------- #
# B+ post-copy editing (edit/regenerate the eventual post before publishing)  #
# --------------------------------------------------------------------------- #


def send_queue_list(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str) -> int:
    """DM the reviewer the approved+queued drafts with their post-copy status.

    Each row shows the topic, the current copy (or 'no copy yet'), and a char
    count against the active limit. Tapping a row opens its copy with edit +
    regenerate buttons (see ``send_copy_content``). Returns the count listed.
    """
    if not _can_issue(conn, client, chat_id):
        return 0
    from humorhist.copywriter import char_limit

    limit = char_limit()
    with review._transaction(conn):
        rows = review.queued_drafts(conn)
        for row in rows:
            row["version"] = _decision_token(conn, row["draft_id"])
            row["link"] = db.get_source_link(conn, row["draft_id"])
    # only show rows that are still queued (unpublished)
    rows = [r for r in rows if not r["published"]]
    if not rows:
        client.send_message(
            chat_id,
            "Queue is empty (nothing approved + queued). Approve drafts with "
            "/reviewdraft, then they appear here ready to publish. See /listposted for recorded publications.",
        )
        return 0
    lines = [f"📋 Queued drafts ({len(rows)}) — edit copy before publishing (#id = draft number):"]
    lines.append("  Export for manual posting; mark posted only after you post it yourself.")
    lines.append("  ✏️ Edit copy   🗑 Remove from queue   ❌ Reject draft (confirm)   ↩️ Reopen to pending")
    keyboard: list[list[dict]] = []
    for r in rows:
        title = " ".join((r["title"] or "(unknown)").split())
        title = title if len(title) <= 100 else title[:99] + "…"
        did = r["draft_id"]
        short = r.get("short_code") or did
        copy = r.get("post_copy")
        if copy:
            snippet = copy if len(copy) <= 80 else copy[:77] + "..."
            status = f"{len(copy)}/{limit}"
        else:
            snippet = "(no copy yet)"
            status = f"0/{limit}"
        lines.append(f"  • {title} (#{short})\n    {snippet}  [{status}]")
        link = r["link"]
        if link:
            displayed_link = db.shorten_url(link, shorten=False)
            if _units(displayed_link) > 500:
                lines.append("    🔗 Source link omitted from list (too long); view it in the database or export.")
            else:
                lines.append(f"    🔗 {displayed_link}")
        # Row 1: edit copy + remove from queue (distinct icons; 'Remove' no longer
        # shares the ↩️ glyph with 'Reopen').
        keyboard.append(
            [
                {"text": f"✏️ Edit: {title[:16]} (#{short})", "callback_data": f"copy:{did}"},
                {
                    "text": f"🗑 Remove: {title[:16]} (#{short})",
                    "callback_data": f"remove:{did}:{r['version']}",
                },
            ]
        )
        # Row 2: reject (confirm-gated) + reopen to pending.
        keyboard.append(
            [
                {
                    "text": f"❌ Reject: {title[:16]} (#{short})",
                    "callback_data": f"reject:{did}:{r['version']}",
                },
                {
                    "text": f"↩️ Reopen: {title[:16]} (#{short})",
                    "callback_data": f"reopen:{did}:{r['version']}",
                },
            ]
        )
        keyboard.append(
            [
                {
                    "text": f"📤 Export: {title[:16]} (#{short})",
                    "callback_data": f"export:{did}:{r['version']}",
                },
                {
                    "text": f"✅ Mark manually posted: {title[:16]} (#{short})",
                    "callback_data": f"posted:{did}:{r['version']}",
                }
            ]
        )
    _send_list_pages(client, chat_id, lines, keyboard, rows_per_item=3)
    return len(rows)


def send_posted_list(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str) -> int:
    """Show publication records, allowing only timestamped records to be undone."""
    if not _can_issue(conn, client, chat_id):
        return 0
    rows = conn.execute(
        "SELECT q.draft_id, q.published_at, COALESCE(v.version, 0) AS version "
        "FROM queue q LEFT JOIN telegram_draft_versions v ON v.draft_id=q.draft_id "
        "WHERE q.published=1 ORDER BY q.id"
    ).fetchall()
    if not rows:
        client.send_message(chat_id, "No recorded publications. See /listqueue to mark a manual post.")
        return 0
    lines = ["📬 Recorded publications — /listqueue shows unpublished drafts:"]
    keyboard = []
    for row in rows:
        did, timestamp = row["draft_id"], row["published_at"]
        label = _label(conn, did)
        lines.append(f"  • {label}\n    {timestamp or 'Legacy publication (no timestamp; cannot undo)'}")
        if timestamp:
            button = {"text": f"↩️ Undo: {label[:35]}", "callback_data": f"unpost:{did}:v{row['version']:x}"}
        else:
            button = {"text": f"👁 View: {label[:35]}", "callback_data": f"view:{did}"}
        keyboard.append([button])
    _send_list_pages(client, chat_id, lines, keyboard)
    return len(rows)


def send_copy_content(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str, draft_id: str) -> list[dict]:
    """Send a queued draft's post copy with inline Edit + Regenerate buttons.

    Tapping 'Edit' (callback ``editmenu:<id>``) opens a choice menu — 'Regenerate'
    (callback ``regencopy:<id>``, works with or without editor notes) and 'Type
    myself' (callback ``editcopy:<id>``, a reply that replaces the copy). Tapping
    'Regenerate' asks the LLM for fresh copy and re-sends. The copy is read from
    ``queue.post_copy``; if absent, a hint to regenerate is shown.
    """
    if not _can_issue(conn, client, chat_id):
        return []
    from humorhist.copywriter import char_limit

    with review._transaction(conn):
        row = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        q = conn.execute("SELECT post_copy, post_copy_at FROM queue WHERE draft_id = ?", (draft_id,)).fetchone()
        label = _label(conn, draft_id)
        version = _decision_token(conn, draft_id)
    if row is None:
        client.send_message(chat_id, f"No such draft: {draft_id}")
        return []
    limit = char_limit()
    copy = (dict(q) if q else {}).get("post_copy") if q else None

    header = f"✏️ POST COPY — {label}\n({len(copy) if copy else 0}/{limit} chars)\n"
    body = copy or "(no copy yet — tap 🔄 Regenerate)"
    markup = {
        "inline_keyboard": [
            [
                {"text": "✏️ Edit", "callback_data": f"editmenu:{draft_id}"},
                {"text": "🔄 Regenerate", "callback_data": f"regencopy:{draft_id}:{version}"},
            ]
        ]
    }
    try:
        return _send_long(client, chat_id, header + "\n" + body, reply_markup=markup, context=label)
    except Exception as exc:  # noqa: BLE001
        logger.error("[telegram] failed to send copy content %s: %s", draft_id, exc)
        return []


# Protect per-database job admission across bot instances in this process. The
# shared flock additionally excludes drafting jobs in scheduled processes.
_jobs_lock = threading.Lock()
_jobs_active: set[tuple[str, str]] = set()
_jobs_announced: dict[str, set[str]] = {}


def _record_announced(worker, result):
    ids = result.get("draft_ids", [])
    if ids:
        path = str(job_control.database_path(worker))
        with _jobs_lock:
            _jobs_announced.setdefault(path, set()).update(ids)


def _telegram_job(conn, client, chat_id, kind, progress, work, *, drafting=False):
    """Admit one job without sharing the poller's SQLite connection."""
    try:
        key = (str(job_control.database_path(conn)), kind)
    except Exception:
        client.send_message(chat_id, "⚠️ Background job could not start. Check application state before retrying.")
        return None
    gate = threading.Event()
    cancelled = threading.Event()
    with _jobs_lock:
        busy = key in _jobs_active
        if not busy:
            _jobs_active.add(key)
    if busy:
        client.send_message(chat_id, f"⚠️ {kind.capitalize()} job already running.")
        return None

    def run(worker):
        gate.wait()
        if cancelled.is_set():
            return None
        return work(worker)

    def cleanup(_future):
        with _jobs_lock:
            _jobs_active.discard(key)

    def on_error(_message):
        client.send_message(chat_id, f"⚠️ {kind.capitalize()} job failed. Check application state before retrying.")

    try:
        future = job_control.start_db_job(conn, run, drafting=drafting, on_error=on_error)
    except Exception:
        with _jobs_lock:
            _jobs_active.discard(key)
        client.send_message(chat_id, "⚠️ Background job could not start. Check application state before retrying.")
        return None
    if future is None:
        with _jobs_lock:
            _jobs_active.discard(key)
        client.send_message(chat_id, "⚠️ Drafting is already running; try again when it finishes.")
        return None
    future.add_done_callback(cleanup)
    # Never start work unless the acknowledgement was delivered successfully.
    try:
        client.send_message(chat_id, progress)
    except BaseException:
        cancelled.set()
        raise
    finally:
        gate.set()
    return future


def telegram_harvest(conn, client, chat_id, *, seed: bool = True, wikipedia: bool = True):
    """Harvest asynchronously; notify progress and completion."""

    def work(worker):
        from humorhist.harvest.seed import load_seed
        from humorhist.harvest.wikipedia_lists import harvest_wikipedia_lists

        lines = []
        failures = 0
        sources = 0
        if seed:
            sources += 1
            try:
                result = load_seed(worker)
                lines.append(
                    f"Seed list: {result['inserted']} added, {result['skipped_duplicate']} already in pool, "
                    f"{result['skipped_invalid']} invalid."
                )
            except Exception:  # noqa: BLE001 - report the other source independently
                failures += 1
                lines.append("Seed list: failed; some candidates may have been added before the error.")
        if wikipedia:
            sources += 1
            try:
                result = harvest_wikipedia_lists(worker)
                failed = result['pages_failed']
                failures += bool(failed or result.get("pages_unattempted"))
                lines.append(
                    f"Wikipedia: {result['inserted']} added, {result['skipped_duplicate']} already in pool, "
                    f"{result['skipped_invalid']} invalid; {result['pages_fetched']} pages read, "
                    f"{failed} pages failed."
                )
                if result.get("pages_unattempted", 0):
                    lines.append(
                        f"Wikipedia: {result['pages_unattempted']} pages not attempted "
                        "after access denial or rate limiting."
                    )
                reasons = []
                for key, label in (
                    ("http_403", "access denied (HTTP 403)"),
                    ("http_429", "rate limited (HTTP 429)"),
                    ("http_other", "other HTTP errors"),
                    ("network_failed", "network errors"),
                    ("other_failed", "other errors"),
                ):
                    if result.get(key, 0):
                        reasons.append(f"{label}: {result[key]}")
                if reasons:
                    lines.append("Wikipedia failures: " + "; ".join(reasons) + ".")
            except Exception:  # noqa: BLE001 - never expose provider errors in chat
                failures += 1
                lines.append("Wikipedia: failed; some candidates may have been added before the error.")
        if not sources:
            heading = "🌾 Harvest skipped: no sources selected."
        elif failures:
            heading = "⚠️ Harvest incomplete: one or more sources failed."
        else:
            heading = "🌾 Harvest complete."
        pool = db.counts(worker)['pool']
        lines.extend([
            f"Candidate pool: {pool} total.",
            "Harvest collects candidates only; it does not create drafts.",
            "Next: /screen to score unscored candidates, then /draft to create drafts.",
        ])
        if failures:
            lines.append("Check the source errors before retrying the harvest.")
        client.send_message(chat_id, "\n".join([heading, *lines]))

    return _telegram_job(
        conn, client, chat_id, "harvest", "🌾 Harvesting candidates (this does not create drafts)…", work
    )


def telegram_screen(conn, client, chat_id, *, batch_size: int = 20, limit: int | None = None):
    """Score candidates on a background connection."""
    llm = _get_llm(chat_id, client)
    if llm is None:
        return None

    def work(worker):
        from humorhist.harvest.screen import format_screen_failures, screen_pool

        selected = worker.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM pool WHERE funny_score IS NULL LIMIT ?)",
            (-1 if limit is None else limit,),
        ).fetchone()[0]
        result = screen_pool(worker, llm, batch_size=batch_size, limit=limit)
        # Results may contain provider errors; report only safe counters.
        scored = result["scored"]
        selected = result.get("selected", selected)
        incomplete = result.get("unscored", max(0, selected - scored))
        status = "⚠️ Screening incomplete" if incomplete or result["failed_batches"] else "🔍 Screened"
        reasons = format_screen_failures(result.get("failures"))
        client.send_message(
            chat_id, f"{status}: scored={scored}/{selected}, batches={result['batches']}, "
            f"failed_batches={result['failed_batches']}, unscored={incomplete}, "
            f"skipped={result.get('skipped', 0)}, blocked={result.get('blocked', 0)}."
            + (f" Reasons: {reasons}." if reasons else "")
        )

    return _telegram_job(
        conn, client, chat_id, "screen", f"🔍 Scoring the pool with the LLM (batch {batch_size})…", work
    )


def telegram_image(conn, client, chat_id, draft_id: str | None) -> None:
    """Inbound /image: (re)generate the story image for an approved+queued draft.

    Thin wrapper over ``humorhist.imagegen.generate_image_for_queue`` — the same
    best-effort path the publish/enqueue step uses. Useful from the phone to
    retry a failed/billed-out FAL call, or to regenerate with a different
    ``HUMORHIST_IMAGE_STYLE``. No DB migration needed (the image columns already
    exist on the queue row).
    """
    if not draft_id:
        client.send_message(chat_id, "Usage: /image <draft_id>")
        return
    row = conn.execute("SELECT status FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    if row is None:
        client.send_message(chat_id, f"⚠️ No such draft: {_label(conn, draft_id)}.")
        return
    if row["status"] != "approved":
        client.send_message(
            chat_id,
            f"⚠️ {_label(conn, draft_id)} is '{row['status']}', not approved/queued. "
            "Images are generated for approved drafts only.",
        )
        return
    if not conn.execute("SELECT 1 FROM queue WHERE draft_id = ?", (draft_id,)).fetchone():
        client.send_message(
            chat_id,
            f"⚠️ {_label(conn, draft_id)} has no queue row yet — approve it (or /queue enqueue) first.",
        )
        return
    client.send_message(chat_id, f"🖼️ Generating story image for {_label(conn, draft_id)}…")
    try:
        from humorhist import imagegen as ig

        # _resolve_image_dir falls back to <repo>/data/images; the str() default
        # covers its documented "shouldn't happen" None so we never pass None.
        out_dir = _resolve_image_dir(None) or str(Path(__file__).resolve().parent.parent / "data" / "images")
        result = ig.generate_image_for_queue(conn, draft_id, out_dir=out_dir)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the loop
        client.send_message(chat_id, f"⚠️ Image generation failed for {_label(conn, draft_id)}: {exc}")
        return
    if result is None:
        client.send_message(
            chat_id,
            f"🖼️ Story image skipped for {_label(conn, draft_id)} — no HUMORHIST_IMAGE_API_KEY set. "
            "Set it to enable images (post copy unaffected).",
        )
        return
    path, prompt = result
    _send_long(
        client, chat_id, f"🖼️ Image ready for {_label(conn, draft_id)}:\n{prompt}", context=_label(conn, draft_id)
    )
    try:
        with open(path, "rb") as fh:
            client.send_photo(chat_id, fh.read(), caption=f"🖼️ Story image for {_label(conn, draft_id)}")
    except Exception as exc:  # noqa: BLE001 - photo is a bonus; the prompt + path are persisted
        logger.warning("[telegram] failed to send image for %s: %s", draft_id, exc)


def telegram_buffer(conn, client, chat_id, *, enqueue: bool = False, auto_draft: bool = True):
    """Report health promptly and run any top-up on an isolated connection."""
    import humorhist.buffer as buf

    llm = _get_llm(chat_id, client) if auto_draft else None
    health = buf.buffer_health(conn)
    if not enqueue and not (llm and health["auto_draft"]):
        client.send_message(chat_id, buf.health_message(health, will_draft=False))
        return None

    def work(worker):
        if enqueue:
            n = review.enqueue_approved(worker, image_dir=_resolve_image_dir(None))
            client.send_message(chat_id, f"📥 Enqueued {n} approved draft(s) into the queue.")
        result = {"drafted": 0}
        if llm and buf.buffer_health(worker)["auto_draft"]:
            # _telegram_job already owns the shared drafting lock; calling
            # run_buffer_check here would attempt to acquire it a second time.
            from humorhist.drafting import draft_candidates

            result = draft_candidates(worker, llm, count=buf.AUTO_DRAFT_COUNT)
        if result.get("drafted"):
            _record_announced(worker, result)
        if result.get("failed") or result.get("unattempted"):
            outcome = "partially completed" if result.get("drafted") else "failed"
            client.send_message(
                chat_id, f"⚠️ Buffer top-up {outcome}: drafted={result.get('drafted', 0)}, "
                f"failed={result.get('failed', 0)}, selected={result.get('selected', 0)}, "
                f"unattempted={result.get('unattempted', 0)}. Check application state before retrying."
            )
        elif result.get("drafted"):
            client.send_message(chat_id, f"✍️ Auto-drafted {result['drafted']} candidate(s). Send /reviewdraft.")
        else:
            client.send_message(chat_id, "✓ Buffer check complete; no candidates drafted.")

    return _telegram_job(
        conn,
        client,
        chat_id,
        "buffer",
        buf.health_message(health, will_draft=bool(llm and health["auto_draft"])),
        work,
        drafting=bool(llm),
    )


def telegram_queue(conn, client, chat_id, *, action: str = "list", draft_id: str | None = None) -> None:
    """Inbound /queue: manage the publish queue (Phase 4 handoff).

    `/queue`              -> list approved+queued drafts (same as /listqueue)
    `/queue enqueue`      -> sweep approved drafts into the queue
    `/queue remove <id>`  -> pull a draft back out of the queue (keeps it approved)
    """
    if action == "enqueue":
        n = review.enqueue_approved(conn, image_dir=_resolve_image_dir(None))
        client.send_message(chat_id, f"📥 Enqueued {n} approved draft(s) into the queue.")
        send_queue_list(conn, client, chat_id)
    elif action == "remove":
        if not draft_id:
            client.send_message(chat_id, "Usage: /queue remove <draft_id>")
            return
        removed = review.remove_from_queue(conn, draft_id)
        if removed:
            client.send_message(chat_id, f"↩️ Removed {_label(conn, draft_id)} from the queue (kept approved).")
        else:
            client.send_message(chat_id, f"⚠️ {_label(conn, draft_id)} was not in the queue.")
    else:
        send_queue_list(conn, client, chat_id)


def telegram_publish(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str) -> None:
    """Inbound /publish: read-only DM export of queued, copy-bearing posts.

    Messages each queued draft's post copy + 'learn more' link + story image
    path + a per-draft readiness checklist (copy / joke / image), exactly
    mirroring `humorhist publish` on the CLI. NOTHING is posted to any platform
    and NOTHING in the DB changes — the editor still pastes by hand. The text is
    built by ``review.format_publish_export`` so the CLI and Telegram outputs
    can't drift.
    """
    rows = review.queued_drafts(conn)
    rows = [r for r in rows if r.get("post_copy")]
    if not rows:
        client.send_message(
            chat_id,
            "Nothing queued with post copy yet. Run /queue enqueue and "
            "/viewcopy <id> + 🔄 Regenerate, or use `humorhist queue --enqueue` "
            "and `humorhist copy regen` on the CLI.",
        )
        return
    for index, row in enumerate(rows, 1):
        short = row.get("short_code") or row["draft_id"]
        export = review.format_publish_export(conn, [row])
        if len(rows) > 1:
            export = export.replace(
                "1 post(s) ready to paste manually:", f"Post {index} of {len(rows)} ready to paste manually:", 1
            )
        export = export.replace(f"── {short} — {row['title']}", f"── {_label(conn, row['draft_id'])}", 1)
        _send_long(client, chat_id, export, context=_label(conn, row["draft_id"]))


def telegram_draft(conn, client, chat_id, *, count: int = 3, min_score: float = 7.0):
    """Draft candidates in the background under the shared drafting lock."""
    llm = _get_llm(chat_id, client)
    if llm is None:
        return None

    def work(worker):
        from humorhist.drafting import draft_candidates

        result = draft_candidates(worker, llm, count=count, min_score=min_score)
        pending = len(review.pending_drafts(worker))
        drafted = result.get("drafted", 0)
        failed = result.get("failed", 0)
        from humorhist.drafting import describe_failure_counts

        details = describe_failure_counts(result.get("failures"))
        failure_note = f" Failures: {details}." if details else ""
        if result.get("unattempted", 0):
            failure_note += f" Not attempted after Wikipedia access denial or rate limiting: {result['unattempted']}."
        client.send_message(
            chat_id,
            f"✍️ Drafted {drafted} new draft(s). Selected: {result.get('selected', 0)}. {pending} pending total — "
            f"send /reviewdraft to review them. Failed: {failed}.{failure_note}",
        )
        _record_announced(worker, result)

    return _telegram_job(
        conn, client, chat_id, "draft", f"✍️ Drafting {count} candidate(s) (min score {min_score})…", work, drafting=True
    )


def _next_pending(conn, pending, after_id):
    if not pending:
        return None
    ids = [r["id"] for r in pending]
    if after_id in ids:
        return pending[(ids.index(after_id) + 1) % len(ids)]
    if after_id:
        previous = conn.execute("SELECT created_at, id FROM drafts WHERE id=?", (after_id,)).fetchone()
        if previous:
            key = (previous["created_at"], previous["id"])
            return next((r for r in pending if (r["created_at"], r["id"]) > key), pending[0])
    return pending[0]


def run_review_session(
    conn: sqlite3.Connection,
    client: TelegramTransport,
    chat_id: str,
    awaiting: dict,
    offset_ref: list[int],
    *,
    poll_timeout: int = 30,
    max_iterations: int = 1_000_000,
    image_dir: str | None = None,
    fast: bool = False,
    after_id: str | None = None,
) -> int:
    """Show the NEXT pending draft (one at a time) and return immediately.

    This is deliberately NON-modal: it sends a single pending draft with its
    Approve/Reject/Later + Next buttons and returns to the caller (the idle
    ``run_review_bot`` command loop). The user decides via the buttons, and
    advances with the ⏭ Next button (or by sending /reviewdraft again) — both
    re-enter this function to pull the next pending draft.

    Why one-shot: the old version parked in a ``while`` loop that swallowed the
    user until every draft was decided or a /command bailed them out. That made
    the bot feel dead mid-review. Now the bot is responsive the whole time;
    joke/notes capture is handled by the normal idle ``handle_text`` path via
    the ``awaiting`` map, so a half-finished capture simply waits for the reply
    instead of blocking the loop.

    Returns 0 — decisions are counted by the caller's idle ``_handle`` when the
    confirm/cancel gate actually commits, not here.
    """
    send_reviewed_summary(conn, client, chat_id)
    pending = review.pending_drafts(conn)
    draft = _next_pending(conn, pending, after_id)
    if draft is None:
        if review.deferred_drafts(conn):
            client.send_message(
                chat_id, "⏸ No drafts ready now. Deferred drafts remain; use /listlater to review them.",
                reply_markup={"inline_keyboard": [[{"text": "⏸ Deferred drafts", "callback_data": "menu:later"}],
                                                    [{"text": "🏠 Menu", "callback_data": "menu:home"}]]},
            )
            return 0
        client.send_message(
            chat_id,
            "✅ All caught up — no drafts pending. Send /reviewdraft any time "
            "to check again (new drafts from /draft or the daily timer will "
            "appear here).",
            reply_markup={"inline_keyboard": [[{"text": "🔎 Find drafts", "callback_data": "menu:find"}],
                                                [{"text": "🏠 Menu", "callback_data": "menu:home"}]]},
        )
        return 0
    if after_id and draft is pending[0]:
        previous = conn.execute("SELECT created_at, id FROM drafts WHERE id=?", (after_id,)).fetchone()
        if previous and (previous["created_at"], previous["id"]) >= (draft["created_at"], draft["id"]):
            client.send_message(chat_id, f"↩️ Back to the start of pending drafts: {_label(conn, draft['id'])}.")
    _send_one(conn, client, chat_id, draft, fast=fast)
    return 0


def run_review_bot(
    conn: sqlite3.Connection,
    client: TelegramTransport,
    chat_id: str,
    *,
    once: bool = False,
    poll_timeout: int = 30,
    max_iterations: int = 1_000_000,
    image_dir: str | None = None,
) -> int:
    """Command-driven Telegram review bot (long-poll).

    Idles and reacts to ``/commands`` instead of pushing drafts on startup:

      /reviewdraft  -> show the next pending draft (non-modal; tap ⏭ Next or send /reviewdraft again to advance)
      /listapproved -> list approved drafts; tap one to open its content
      /status       -> reviewed/pending breakdown
      /help, /start -> this message

    ``once=True`` keeps the legacy dump behaviour (send summary + all pending,
    process queued updates once) for one-shot CLI runs and tests.
    """
    # Reject group IDs, usernames and ambiguous/non-canonical numeric forms
    # before reading the database or contacting the transport.
    if not isinstance(chat_id, str) or re.fullmatch(r"[1-9][0-9]*", chat_id) is None:
        raise ValueError("chat_id must be a positive numeric private chat/user ID")
    offset_ref = [0]
    awaiting: dict[str, dict] = {}
    decided = 0
    current_id: str | None = None
    fast_mode = False

    def _handle(upd: dict) -> None:
        nonlocal current_id, fast_mode
        offset_ref[0] = max(offset_ref[0], upd.get("update_id", 0) + 1)
        # Acknowledge rejected updates in the polling offset, but never dispatch
        # them or mutate a pending review session's state.
        if not _authorized_update(upd, chat_id):
            return
        if not _can_issue(conn, client, chat_id):
            return
        # ⏭ Next button: re-enter the (non-modal) review session to pull the
        # next pending draft, without the user typing /reviewdraft again.
        if "callback_query" in upd:
            data = (upd.get("callback_query") or {}).get("data") or ""
            if data.startswith(("find:screen", "find:draft")):
                parts = data.split(":")
                valid = False
                if len(parts) == 3 and parts[1] in ("screen", "draft"):
                    with _intent_lock:
                        entry = _intents.get(parts[2])
                        if entry == (str(chat_id), "", parts[1], ""):
                            del _intents[parts[2]]
                            valid = True
                if not valid:
                    client.answer_callback_query(
                        upd["callback_query"]["id"], text="Button expired; open Find drafts again"
                    )
                    return
                # Consume before acknowledgement and dispatch; never hold the lock over I/O.
                client.answer_callback_query(upd["callback_query"]["id"], text="")
                _dispatch("/" + parts[1])
                return
            if data in ("menu:home", "menu:review", "menu:queue", "menu:find", "menu:later",
                        "find:harvest", "review:normal", "review:fast"):
                client.answer_callback_query(upd["callback_query"]["id"], text="")
                if data == "menu:home":
                    _main_menu(client, chat_id)
                elif data == "menu:review":
                    client.send_message(
                        chat_id, "Choose review mode. Normal asks for confirmation; fast commits on the first tap.",
                        reply_markup={"inline_keyboard": [
                            [{"text": "📖 Normal review (confirm)", "callback_data": "review:normal"}],
                            [{"text": "⚡ Fast review (no confirmation)", "callback_data": "review:fast"}],
                            [{"text": "🏠 Menu", "callback_data": "menu:home"}],
                        ]})
                elif data == "menu:queue":
                    send_queue_list(conn, client, chat_id)
                elif data == "menu:later":
                    send_deferred_list(conn, client, chat_id)
                elif data == "menu:find":
                    _find_menu(client, chat_id)
                elif data.startswith("find:"):
                    _dispatch("/" + data.partition(":")[2])
                else:
                    _dispatch("/reviewdraft" + (" fast" if data == "review:fast" else ""))
                return
            if data.startswith(("next:", "nextfast:")):
                client.answer_callback_query(upd["callback_query"]["id"], text="")
                fast_mode = data.startswith("nextfast:")
                after_id = data.partition(":")[2]
                pending = review.pending_drafts(conn)
                next_row = _next_pending(conn, pending, after_id)
                current_id = next_row["id"] if next_row else None
                run_review_session(
                    conn,
                    client,
                    chat_id,
                    awaiting,
                    offset_ref,
                    poll_timeout=poll_timeout,
                    max_iterations=max_iterations,
                    image_dir=image_dir,
                    after_id=after_id,
                    fast=fast_mode,
                )
                return
        if "callback_query" in upd:
            res = handle_callback(conn, client, chat_id, upd, awaiting=awaiting)
            # A committed decision (the confirm: step) carries "decision" but
            # not "confirming"; the initial tap carries "confirming": True and
            # must NOT count as a decision yet.
            if res and res.get("committed"):
                nonlocal decided
                decided += 1
                # A reject (no note_message_id) just records the decision and
                # finishes; only an approve registers a joke/notes capture prompt.
                if "note_message_id" in res:
                    awaiting[res["draft_id"]] = {
                        "note_message_id": res["note_message_id"],
                        "stage": res.get("stage", "editor_line"),
                        "fast": res.get("fast", False),
                        "decision": res.get("decision", "approve"),
                        # Propagate setjoke so handle_text routes the reply through
                        # review.set_editor_line (targeted UPDATE) instead of the
                        # full apply_review re-approve path, which would clobber
                        # any existing editor_notes with NULL.
                        "setjoke": res.get("setjoke", False),
                        "regenerate_angles": res.get("regenerate_angles", False),
                        "version": res.get("version"),
                    }
            elif res and "note_message_id" in res:
                # notes: button from /listapproved
                awaiting[res["draft_id"]] = dict(res)
            elif res and "editcopy_message_id" in res:
                # post-copy edit prompt: register so the next reply is captured
                awaiting[res["draft_id"]] = dict(res)
            elif res and "skipped" in res:
                # ⏭ Skip (no notes) button: clear the optional notes capture.
                awaiting.pop(res["skipped"], None)
            return
        msg = upd.get("message")
        if not msg:
            return
        text = (msg.get("text") or "").strip()
        if text.startswith("/"):
            # /skip and /cancel are replies to an in-flight joke/notes/edit-copy
            # prompt (tracked in `awaiting`), NOT bot commands. Route them to
            # handle_text so the prompt resolves instead of being swallowed by
            # _dispatch as an "Unknown command" (which would leak the awaiting
            # entry and wedge later prompts).
            if text in ("/skip", "/cancel") and awaiting:
                handle_text(conn, client, chat_id, awaiting, upd, image_dir=image_dir)
            else:
                _dispatch(text)
            return
        handle_text(conn, client, chat_id, awaiting, upd, image_dir=image_dir)

    def _dispatch(text: str) -> None:
        nonlocal decided, current_id, fast_mode
        cmd = text.split()[0].lower()
        parts = text.split()
        takes_id = cmd in {"/view", "/viewcopy", "/later", "/reviewnow", "/setjoke", "/notes", "/image"}
        queue_remove = cmd == "/queue" and len(parts) > 1 and parts[1].lower() == "remove"
        position = 2 if queue_remove else 1
        if (takes_id or queue_remove) and len(parts) > position:
            if len(parts) != position + 1:
                client.send_message(chat_id, "Invalid draft identifier; supply exactly one code or full ID.")
                return
            try:
                parts[position] = _resolve_draft(conn, parts[position])
            except ValueError as exc:
                client.send_message(chat_id, str(exc))
                return
            text = " ".join(parts)
        if cmd == "/reviewdraft":
            # /reviewdraft        -> normal flow (each tap opens a Confirm/Cancel gate)
            # /reviewdraft fast   -> commits on the first tap (no confirm gate);
            #                        recovery is the per-draft undo (reject/reopen
            #                        from /listapproved or /listqueue)
            fast = "fast" in text.split()
            fast_mode = fast
            pending = review.pending_drafts(conn)
            after_id = current_id
            next_row = _next_pending(conn, pending, after_id)
            current_id = next_row["id"] if next_row else None
            decided += run_review_session(
                conn,
                client,
                chat_id,
                awaiting,
                offset_ref,
                poll_timeout=poll_timeout,
                max_iterations=max_iterations,
                fast=fast,
                after_id=after_id,
            )
        elif cmd == "/listapproved":
            send_approved_list(conn, client, chat_id)
        elif cmd == "/listqueue":
            send_queue_list(conn, client, chat_id)
        elif cmd == "/listposted":
            send_posted_list(conn, client, chat_id)
        elif cmd == "/listrejected":
            send_rejected_list(conn, client, chat_id)
        elif cmd == "/viewcopy":
            target = text.split()
            if len(target) < 2:
                client.send_message(chat_id, "Usage: /viewcopy <draft_id>")
            else:
                send_copy_content(conn, client, chat_id, target[1])
        elif cmd == "/view":
            target = text.split()
            if len(target) < 2:
                client.send_message(chat_id, "Usage: /view <draft_id>")
            else:
                # Re-read any draft's full content from the phone, regardless of
                # status (pending/approved/rejected). /viewcopy is for queued
                # post-copy only; /view is the general "show me the draft" view.
                send_draft_content(conn, client, chat_id, target[1])
        elif cmd == "/status":
            send_reviewed_summary(conn, client, chat_id)
        elif cmd == "/jokes":
            send_stuck_jokes_list(conn, client, chat_id)
        elif cmd == "/later":
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                client.send_message(chat_id, "Usage: /later <draft_id>")
            else:
                did = parts[1].strip()
                try:
                    review.defer_draft(conn, did)
                except ValueError as exc:
                    client.send_message(chat_id, f"Cannot defer {_label(conn, did)}: {exc}")
                else:
                    client.send_message(chat_id, f"⏸ {_label(conn, did)} deferred 30 days.")
        elif cmd == "/listlater":
            send_deferred_list(conn, client, chat_id)
        elif cmd == "/reviewnow":
            # /reviewnow            -> bring ALL deferred drafts forward
            # /reviewnow <id>       -> bring one deferred draft forward
            rest = text.split(maxsplit=1)
            did = rest[1].strip() if len(rest) > 1 else None
            try:
                n = review.bring_forward(conn, did)
            except ValueError as exc:
                client.send_message(chat_id, f"Cannot bring forward: {exc}")
            else:
                if did is None:
                    client.send_message(
                        chat_id,
                        f"⏩ Brought forward {n} deferred draft(s) — they're back "
                        f"in the review queue. Send /reviewdraft to see them.",
                    )
                else:
                    client.send_message(
                        chat_id,
                        f"⏩ {_label(conn, did)} brought forward — it's back in the review queue. "
                        "Send /reviewdraft to see it.",
                    )
        elif cmd == "/setjoke":
            rest = text.split(maxsplit=1)
            if len(rest) < 2:
                client.send_message(chat_id, "Usage: /setjoke <draft_id> (then reply with the joke)")
            else:
                did = rest[1].strip()
                status = conn.execute("SELECT status FROM drafts WHERE id = ?", (did,)).fetchone()
                if status["status"] != "approved":
                    client.send_message(chat_id, f"⚠️ {_label(conn, did)} is not approved; no joke changed.")
                    return
                prompt = client.send_message(
                    chat_id,
                    f"Reply with the one-line joke (editor_line) for approved "
                    f"draft {_label(conn, did)} — the human voice for the post copy "
                    f"(or /skip to leave it blank):",
                )
                awaiting[did] = {
                    "note_message_id": prompt["message_id"],
                    "stage": "editor_line",
                    "decision": "approve",
                    "setjoke": True,
                    "version": _decision_token(conn, did),
                }
        elif cmd == "/notes":
            # /notes <id> opens the angle-steer / notes prompt — the typed shortcut
            # for the ✏️ Add notes button on a draft's content view. For a *pending
            # draft the reply regenerates the angles (steering); for an approved
            # draft it stores a note (merge=True so editor_line is preserved). The
            # prompt is built by _open_notes_prompt (shared with the notes:<id>
            # callback so the two entry points can't drift).
            rest = text.split(maxsplit=1)
            if len(rest) < 2:
                client.send_message(chat_id, "Usage: /notes <draft_id>")
            else:
                did = rest[1].strip()
                entry = _open_notes_prompt(conn, client, chat_id, did)
                if not entry.get("error"):
                    awaiting[did] = entry
        elif cmd == "/suggest":
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                client.send_message(chat_id, "Usage: /suggest <topic or event>")
            else:
                topic = parts[1].strip()
                pool_id = db.add_suggested_pool_item(conn, title=topic)
                client.send_message(
                    chat_id,
                    f"💡 Suggested `{topic}` added to the pool (id `{pool_id[:8]}…`). "
                    f"It'll be drafted in a future harvest/draft pass.",
                )
        elif cmd == "/harvest":
            telegram_harvest(conn, client, chat_id)
        elif cmd == "/screen":
            # optional /screen [limit]
            rest = text.split(maxsplit=1)
            if len(rest) > 1 and rest[1].strip().lower() == "status":
                from humorhist.harvest.screen import screen_attempt_status
                attempts = screen_attempt_status(conn)
                pending = sum(r["state"] == "pending" for r in attempts)
                failed = sum(r["state"] == "failed" for r in attempts)
                client.send_message(
                    chat_id, f"Screening reservations: pending={pending}, failed={failed}, total={len(attempts)}. "
                    "Use CLI screen-status for IDs and README recovery instructions; retries may incur charges."
                )
            elif len(rest) == 1 or rest[1].strip().isdigit():
                limit = int(rest[1]) if len(rest) > 1 else None
                telegram_screen(conn, client, chat_id, limit=limit)
            else:
                client.send_message(chat_id, "Usage: /screen [limit] or /screen status")
        elif cmd == "/draft":
            # optional /draft [count] [min_score]
            bits = text.split()[1:]
            count = 3
            min_score = 7.0
            if len(bits) >= 1 and bits[0].isdigit():
                count = int(bits[0])
            if len(bits) >= 2:
                with contextlib.suppress(ValueError):
                    min_score = float(bits[1])
            telegram_draft(conn, client, chat_id, count=count, min_score=min_score)
        elif cmd == "/buffer":
            # optional /buffer enqueue
            enqueue = "enqueue" in text.split()
            telegram_buffer(conn, client, chat_id, enqueue=enqueue)
        elif cmd == "/queue":
            # /queue | /queue enqueue | /queue remove <id>
            bits = text.split()[1:]
            action = bits[0].lower() if bits else "list"
            if action == "remove":
                telegram_queue(conn, client, chat_id, action="remove", draft_id=bits[1] if len(bits) > 1 else None)
            else:
                telegram_queue(conn, client, chat_id, action=action)
        elif cmd == "/publish":
            # Read-only export: DM the queued, copy-bearing posts (copy + link + image
            # + readiness checklist) so the editor can paste by hand. NOTHING is posted
            # and NOTHING in the DB changes. Mirrors `humorhist publish` on the CLI.
            telegram_publish(conn, client, chat_id)
        elif cmd == "/image":
            # /image <id>  -> (re)generate the story image for an approved+queued draft
            rest = text.split(maxsplit=1)
            telegram_image(conn, client, chat_id, rest[1].strip() if len(rest) > 1 else None)
        elif cmd in ("/start", "/menu"):
            _main_menu(client, chat_id)
        elif cmd == "/help":
            client.send_message(chat_id, HELP_TEXT)
        else:
            client.send_message(chat_id, "Unknown command. Send /help.")

    if once:
        send_reviewed_summary(conn, client, chat_id)
        for row in review.pending_drafts(conn):
            _send_one(conn, client, chat_id, row)
        for upd in client.get_updates(offset=offset_ref[0], timeout=0):
            _handle(upd)
        if offset_ref[0]:
            # This confirms only IDs below the offset. Any newer updates in
            # the response remain unconfirmed for the next polling session.
            client.get_updates(offset=offset_ref[0], timeout=0)
        return decided

    _main_menu(client, chat_id)
    iters = 0
    last_pending: set[str] | None = None
    try:
        job_db_path = str(job_control.database_path(conn))
    except Exception:
        job_db_path = None
    while True:
        iters += 1
        if iters > max_iterations:
            return decided
        # Defer the nudge while our own drafting job runs. Its completion
        # message already announces its new IDs; external new drafts still nudge.
        with _jobs_lock:
            drafting_active = any((job_db_path, kind) in _jobs_active for kind in ("draft", "buffer"))
            announced = _jobs_announced.pop(job_db_path, set()) if job_db_path and not drafting_active else set()
        if not drafting_active:
            try:
                pending_now = {row["id"] for row in review.pending_drafts(conn)}
            except Exception:  # noqa: BLE001 - DB read must never break the poll loop
                pending_now = last_pending or set()
            new_pending = pending_now - (last_pending or set()) - announced
            if last_pending is not None and new_pending:
                client.send_message(
                    chat_id,
                    f"🆕 {len(new_pending)} new draft(s) awaiting review "
                    f"({len(pending_now)} total). Send /reviewdraft.",
                )
            last_pending = pending_now
        try:
            for upd in client.get_updates(offset=offset_ref[0], timeout=poll_timeout):
                _handle(upd)
        except TelegramError as exc:
            logger.error("[telegram] %s; retrying in 5s", exc)
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
        f"\U0001f4dd {n} draft(s) awaiting review. Run `review` or check Telegram to decide.",
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
        lines.append("\n✅ Approved: 0")

    if rejected["titles"]:
        lines.append(f"\n❌ Rejected ({rejected['count']}):")
        lines.extend(f"  • {t}" for t in rejected["titles"])
    else:
        lines.append("\n❌ Rejected: 0")

    lines.append(f"\n⏳ Pending: {pending['count']}")
    return "\n".join(lines)


def send_reviewed_summary(conn: db.Connection, client: TelegramTransport, chat_id: str) -> str:
    """DM the reviewer the approved/rejected/pending breakdown. Returns the text.

    Also nudges on GAP 4b "stuck captures" — approved+queued drafts whose
    one-line joke (editor_line) was never filled — so a committed-but-blank
    human voice doesn't sit silently in the publish queue.
    """
    summary = review.reviewed_summary(conn)
    for status in ("approved", "rejected"):
        rows = conn.execute(
            "SELECT d.id FROM drafts d LEFT JOIN pool p ON p.id=d.pool_id WHERE d.status=? ORDER BY p.title, d.id",
            (status,),
        ).fetchall()
        summary[status]["titles"] = [_label(conn, r["id"]) for r in rows]
    text = format_reviewed_summary(summary)
    _send_long(client, chat_id, text)
    if review.stuck_captures(conn):
        send_stuck_jokes_list(conn, client, chat_id)
    return text


def send_stuck_jokes_list(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str) -> int:
    """DM approved+queued drafts that still lack a one-line joke (editor_line).

    Each row shows the topic title alongside the #short_code and carries an inline
    📝 Add joke button (setjoke:<id>) that prompts for the joke directly. Used by:
    - ``/jokes`` (dedicated command for filling missing jokes)
    - ``/status`` nudge (via ``send_reviewed_summary``)

    Returns the count listed (0 if none are stuck).
    """
    if not _can_issue(conn, client, chat_id):
        return 0
    stuck = review.stuck_captures(conn)
    if not stuck:
        client.send_message(
            chat_id,
            "No stuck jokes — every approved draft has its one-line joke. 🎭",
        )
        return 0
    lines = [
        f"⚠️ {len(stuck)} approved draft(s) are missing their one-line joke "
        f"(committed but the joke was never captured):",
        "  Tap a draft's 📝 Add joke button to fill it (or /setjoke <id>).",
    ]
    keyboard: list[list[dict]] = []
    for s in stuck:
        title = " ".join((s["title"] or "(unknown)").split())
        title = title if len(title) <= 100 else title[:99] + "…"
        short = s.get("short_code") or s["draft_id"]
        lines.append(f"  • {title} (#{short})")
        keyboard.append(
            [
                {
                    "text": f"📝 Add joke: {title[:16]} (#{short})",
                    "callback_data": f"setjoke:{s['draft_id']}",
                }
            ]
        )
    _send_list_pages(client, chat_id, lines, keyboard)
    return len(stuck)
