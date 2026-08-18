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

import contextlib
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Protocol

import httpx

import humorhist.db as db
import humorhist.render as render
import humorhist.review as review

logger = logging.getLogger("humorhist.telegram")

API_BASE = "https://api.telegram.org"


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
            raise TelegramError("no bot token: set HUMORHIST_TELEGRAM_BOT_TOKEN or pass token=")
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
                    time.sleep(2**attempt)
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

    def send_message(self, chat_id: str, text: str, reply_markup: dict | None = None) -> dict:
        params: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_markup is not None:
            params["reply_markup"] = reply_markup
        return self._call("sendMessage", params)

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
            raise TelegramError(f"Telegram sendPhoto failed: {last}")
        params["photo"] = photo
        if caption is not None:
            params["caption"] = caption
        return self._call("sendPhoto", params)

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> dict:
        return self._call("answerCallbackQuery", {"callback_query_id": callback_query_id, "text": text})


def _keyboard(draft_id: str) -> dict:
    # The Next button advances the one-by-one review WITHOUT a parked modal
    # loop: after deciding this draft the user taps Next (or sends /reviewdraft
    # again) to pull the next pending draft from the command loop. This keeps
    # /reviewdraft non-modal — the bot stays responsive to other commands the
    # whole time instead of swallowing the user until every draft is done.
    return {
        "inline_keyboard": [
            [
                {"text": "✅ Approve", "callback_data": f"approve:{draft_id}"},
                {"text": "❌ Reject", "callback_data": f"reject:{draft_id}"},
                {"text": "⏸ Later", "callback_data": f"later:{draft_id}"},
            ],
            [
                {"text": "⏭ Next draft ▶️", "callback_data": f"next:{draft_id}"},
            ],
        ]
    }


def _next_keyboard(draft_id: str) -> dict:
    """Keyboard for the 'no more pending' state shown after the last draft."""
    return {
        "inline_keyboard": [
            [
                {"text": "⏭ Review more ▶️", "callback_data": f"next:{draft_id}"},
            ]
        ]
    }


def _confirm_keyboard(draft_id: str, decision: str) -> dict:
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
                    "callback_data": f"confirm:{decision}:{draft_id}",
                },
                {"text": "↩️ Cancel", "callback_data": f"cancel:{draft_id}"},
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


def _send_long(client: TelegramTransport, chat_id: str, text: str, reply_markup: dict | None = None) -> list[dict]:
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


def _send_one(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str, row: dict) -> list[dict]:
    """Send a single draft (chunked) with Approve/Reject buttons on the last chunk."""
    pool = db.get_pool_item(conn, row["pool_id"])
    text = render.render_draft(row, pool)
    try:
        return _send_long(client, chat_id, text, reply_markup=_keyboard(row["id"]))
    except Exception as exc:  # noqa: BLE001 - one bad draft must not kill the loop
        logger.error("[telegram] failed to send draft %s: %s", row["id"], exc)
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


def handle_callback(
    conn: sqlite3.Connection,
    client: TelegramTransport,
    chat_id: str,
    update: dict,
    *,
    fast: bool = False,
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
    data = (cq.get("data") or "").strip()
    if data.startswith("approve:") or data.startswith("reject:"):
        decision, _, draft_id = data.partition(":")
        if decision not in ("approve", "reject"):
            return None
        if fast:
            # /reviewdraft fast: commit on first tap, no confirm gate. The bot
            # stays responsive via the per-draft undo (reopen/reject-from-list),
            # so a fat-finger is recoverable without a two-tap gate.
            client.answer_callback_query(cq["id"], text=f"fast {decision}")
            return _commit_decision(conn, client, chat_id, cq, draft_id, decision)
        # GAP 3: a tap is a *proposal*, not a commit. Show a confirm/cancel
        # gate so a fat-finger Approve can't enqueue + fire copy-gen before the
        # editor can reconsider. Nothing is written to the DB yet.
        # NOTE: keep the callback-answer text empty-ish — a non-empty toast
        # ("confirm approve?") reads like the decision already happened on some
        # clients, which is misleading. The real prompt carries the buttons.
        client.answer_callback_query(cq["id"], text="")
        verb = "approve" if decision == "approve" else "reject"
        client.send_message(
            chat_id,
            f"⚠️ Confirm: {verb} draft `{draft_id}`?",
            reply_markup=_confirm_keyboard(draft_id, decision),
        )
        return {"draft_id": draft_id, "decision": decision, "confirming": True}
    if data.startswith("confirm:"):
        # format: confirm:<decision>:<draft_id>
        _, rest = data.split(":", 1)
        decision, _, draft_id = rest.partition(":")
        if decision not in ("approve", "reject"):
            return None
        return _commit_decision(conn, client, chat_id, cq, draft_id, decision)
    if data.startswith("cancel:"):
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="cancelled")
        client.send_message(
            chat_id,
            f"↩️ Decision for `{draft_id}` cancelled — no change made. "
            f"Tap ✅/❌ again when you're sure, or /reviewdraft to move on.",
        )
        return {"draft_id": draft_id, "cancelled": True}
    if data.startswith("notes:"):
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="add notes")
        # For a *pending* draft, a note should regenerate the angles (steering),
        # not just store text. For an already-approved draft, store the note.
        row = conn.execute("SELECT status FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        is_pending = row and row["status"] == "pending"
        prompt = (
            f"Reply with a note to STEER the angles for pending draft `{draft_id}` "
            f"(or /skip to keep the current angles):"
            if is_pending
            else f"Notes for already-approved draft `{draft_id}`. Reply here "
            f"(or send /skip to leave the existing notes untouched):"
        )
        note = client.send_message(chat_id, prompt)
        return {
            "draft_id": draft_id,
            "note_message_id": note["message_id"],
            "stage": "notes",
            "decision": "approve",
            "regenerate_angles": bool(is_pending),
        }
    if data.startswith("later:"):
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="deferred 30d")
        try:
            review.defer_draft(conn, draft_id)
        except ValueError:
            client.answer_callback_query(cq["id"], text="cannot defer")
            return None
        client.send_message(
            chat_id,
            f"⏸ Draft `{draft_id}` deferred for 30 days (it'll resurface in the review queue then).",
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
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="removed from queue")
        removed = review.remove_from_queue(conn, draft_id)
        if removed:
            client.send_message(
                chat_id,
                f"↩️ `#{draft_id}` removed from the queue (kept approved). List it again with /listqueue.",
            )
        else:
            client.send_message(chat_id, f"⚠️ `#{draft_id}` was not in the queue.")
        return None
    if data.startswith("reopen:"):
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="reopened for re-review")
        try:
            review.reopen_draft(conn, draft_id)
        except ValueError as exc:
            client.send_message(chat_id, f"Cannot reopen: {exc}")
            return None
        client.send_message(
            chat_id,
            f"↩️ `#{draft_id}` sent back to pending for re-review. List it with /reviewdraft.",
        )
        return None
    if data.startswith("reviewnow:"):
        # GAP 4: bring a deferred draft forward for immediate review.
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="brought forward")
        try:
            review.bring_forward(conn, draft_id)
        except ValueError as exc:
            client.send_message(chat_id, f"Cannot bring forward: {exc}")
            return None
        client.send_message(
            chat_id,
            f"⏩ `#{draft_id}` brought forward — it's back in the review queue. "
            f"Send /reviewdraft to see it (or tap ✅/❌ next time one is shown).",
        )
        return None
    if data.startswith("setjoke:"):
        # GAP 4b fix (extended): an approved+queued draft whose joke (editor_line)
        # was never filled. Previously this dropped straight into a bare reply box
        # with no LLM option. Now it opens the same choice menu as the copy editor
        # so Generate/Regenerate is always one tap away (works with no note yet).
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="add joke")
        menu = {
            "inline_keyboard": [
                [
                    {"text": "🔄 Generate copy (LLM)", "callback_data": f"regencopy:{draft_id}"},
                    {"text": "✏️ Write joke myself", "callback_data": f"editcopy:{draft_id}"},
                ]
            ]
        }
        client.send_message(
            chat_id,
            f"How do you want to fill the post copy for `{draft_id}`?\n"
            f"• 🔄 Generate copy — ask the LLM for fresh copy (works with or without a joke/note)\n"
            f"• ✏️ Write joke myself — reply with your one-line joke, then copy is generated from it",
            reply_markup=menu,
        )
        return None
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
                    {"text": "🔄 Regenerate copy", "callback_data": f"regencopy:{draft_id}"},
                    {"text": "✏️ Type copy myself", "callback_data": f"editcopy:{draft_id}"},
                ]
            ]
        }
        client.send_message(
            chat_id,
            f"How do you want to change the post copy for `{draft_id}`?\n"
            f"• 🔄 Regenerate — ask the LLM for fresh copy (works with or without notes)\n"
            f"• ✏️ Type myself — reply with your own copy",
            reply_markup=menu,
        )
        return None
    if data.startswith("editcopy:"):
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="editing copy")
        prompt = client.send_message(
            chat_id,
            f"Reply with the new post copy for `{draft_id}` (or send /cancel to keep the current version):",
        )
        return {"draft_id": draft_id, "editcopy_message_id": prompt["message_id"]}
    if data.startswith("regencopy:"):
        _, _, draft_id = data.partition(":")
        client.answer_callback_query(cq["id"], text="regenerating")
        from humorhist.copywriter import fill_post_copy

        llm = _get_llm(chat_id, client)
        if llm is None:
            return None
        try:
            n = fill_post_copy(conn, llm, draft_id=draft_id, force=True)
        except Exception as exc:  # noqa: BLE001
            client.send_message(chat_id, f"Regeneration failed: {exc}")
            return None
        if n == 0:
            client.send_message(
                chat_id,
                "Nothing generated (no LLM key available?). The copy is unchanged.",
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
) -> dict | None:
    """Commit an approve/reject and run the decision-specific follow-up.

    Shared by the confirm-gate path (``confirm:<decision>:<id>``) and the
    ``/reviewdraft fast`` path (first tap commits directly). Returns the result
    dict the caller uses to track note/joke capture state.
    """
    try:
        review.apply_review(conn, draft_id, decision=decision)
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
            f"🚫 Draft `{draft_id}` rejected. It's out of the queue; "
            f"re-list it any time with /listrejected (or /reviewdraft if "
            f"you change your mind later).",
        )
        return {"draft_id": draft_id, "decision": "reject"}
    # The joke is the whole point of the product: capture the human-written
    # editor_line first. A reply to this prompt becomes editor_line (which
    # also steers B+ post-copy generation); /skip leaves it blank.
    note = client.send_message(
        chat_id,
        f"Draft `{draft_id}` approved. Reply here with your one-line "
        f"joke (the editor_line) — or send /skip to leave it blank:",
    )
    # Stash the decision so a follow-up reply knows whether to also capture
    # a (secondary) notes step after the editor_line.
    return {
        "draft_id": draft_id,
        "decision": decision,
        "note_message_id": note["message_id"],
        "stage": "editor_line",
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
    if draft_id is None and len(awaiting) == 1:
        did = next(iter(awaiting))
        draft_id, st = did, awaiting[did]
    if draft_id is None:
        return None

    # editcopy prompt: replace the post copy (unchanged behaviour).
    if st is not None and st.get("editcopy_message_id") is not None:
        if text == "/cancel":
            awaiting.pop(draft_id, None)
            client.send_message(chat_id, f"Kept the existing copy for `{draft_id}`.")
            return {"editcopy_cancelled": draft_id}
        from humorhist.copywriter import set_post_copy

        set_post_copy(conn, draft_id, text)
        awaiting.pop(draft_id, None)
        client.send_message(chat_id, f"Post copy saved for `{draft_id}`.")
        return {"editcopy_saved": draft_id}

    stage = (st or {}).get("stage", "notes")
    decision = (st or {}).get("decision", "approve")

    # --- editor_line stage: capture the human joke (the product's point) ---
    if stage == "editor_line":
        editor_line = None if text == "/skip" else text
        if (st or {}).get("setjoke"):
            # A "stuck capture" fix: the draft is already approved + queued.
            # Just set the joke (no re-commit, no re-enqueue, no copy regen).
            review.set_editor_line(conn, draft_id, editor_line)
            awaiting.pop(draft_id, None)
            if editor_line is None:
                client.send_message(chat_id, f"Joke left blank for `{draft_id}`.")
            else:
                client.send_message(chat_id, f"Joke saved for `{draft_id}`.")
            return {"editor_line_set": draft_id}
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
        note = client.send_message(
            chat_id,
            f"Got it. Optionally reply with longer notes for `{draft_id}` (or /skip to finish):",
        )
        awaiting[draft_id] = {
            "note_message_id": note["message_id"],
            "stage": "notes",
            "decision": decision,
        }
        return {"editor_line_set": draft_id}

    # --- notes stage: optional free-form annotation (merge, don't clobber) ---
    if text == "/skip":
        awaiting.pop(draft_id, None)
        client.send_message(chat_id, f"Notes left blank for `{draft_id}`.")
        return {"skipped": draft_id}
    # If this note came from a *pending* draft's "steer angles" prompt, regenerate
    # the angles with the note as steering (the /notes -> angles behaviour).
    if (st or {}).get("regenerate_angles"):
        from humorhist.brief import regenerate_angles

        llm = _get_llm(chat_id, client)
        if llm is None:
            return {"angle_regen_failed": draft_id}
        try:
            regenerate_angles(conn, llm, draft_id, steering_note=text)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive on LLM failure
            client.send_message(chat_id, f"Angle regen failed for `{draft_id}`: {exc}")
            return {"angle_regen_failed": draft_id}
        awaiting.pop(draft_id, None)
        client.send_message(
            chat_id,
            f"🔄 Angles regenerated for `{draft_id}` using your note as steering. Review them with /reviewdraft.",
        )
        return {"angles_regenerated": draft_id}
    # merge=True keeps the editor_line we just captured; only notes change.
    review.apply_review(conn, draft_id, decision=decision, notes=text, merge=True)
    awaiting.pop(draft_id, None)
    client.send_message(chat_id, f"Notes saved for `{draft_id}`.")
    return {"noted": draft_id}


HELP_TEXT = (
    "HumorHist review bot — the whole pipeline runs from here.\n\n"
    "QUICK START: /reviewdraft to decide on pending drafts · /listqueue to manage "
    "approved ones · /status for progress · /help for everything.\n\n"
    "DISCOVERY & DRAFTING\n"
    "/harvest - top up the candidate pool from seed + Wikipedia lists\n"
    "/screen [limit] - LLM-score unscored pool candidates (funny_score)\n"
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
    "    /setjoke <id> - set the one-line joke on an approved draft (blank-capture fix)\n"
    "/listapproved - list approved drafts with their #id; tap one to open it\n"
    "    each row also has a ❌ Reject button (confirm-gated) to un-approve it\n"
    "/listrejected - list rejected drafts; tap one to reopen for re-review\n"
    "/listqueue - list queued drafts with their #id, copy + Remove/Reject/Reopen\n"
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
    "/help - this message\n\n"
    "The bot also nudges you with 🆕 N new draft(s) when fresh drafts appear."
)


def send_approved_list(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str) -> int:
    """DM a list of approved drafts, each with an inline 'view' button.

    Tapping a draft opens its full content (see ``send_draft_content``), from
    which the reviewer can optionally add notes. Returns the count listed.
    """
    rows = review.approved_drafts(conn)
    if not rows:
        client.send_message(
            chat_id,
            "No approved drafts yet. Run /reviewdraft to review pending ones, "
            "or /status for progress.",
        )
        return 0
    lines = [
        "✅ Approved drafts (tap to open; #id is the draft number):",
        "  👁 Open draft   ❌ Reject (confirm-gated, sends back to pending)",
    ]
    keyboard = []
    for r in rows:
        title = r["title"] or "(unknown)"
        did = r["draft_id"]
        short = r.get("short_code") or did
        lines.append(f"  • #{short} {title}")
        keyboard.append(
            [
                {"text": f"👁 Open: {title[:18]} (#{short})", "callback_data": f"view:{did}"},
                {"text": f"❌ Reject #{short}", "callback_data": f"reject:{did}"},
            ]
        )
    client.send_message(chat_id, "\n".join(lines), reply_markup={"inline_keyboard": keyboard})
    return len(rows)


def send_rejected_list(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str) -> int:
    """DM a list of rejected drafts, each with an inline 'reopen' button.

    Tapping a draft sends it back to pending for re-review (the editor's undo
    for a mistaken reject). Returns the count listed.
    """
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
        title = r["title"] or "(unknown)"
        did = r["draft_id"]
        short = r.get("short_code") or did
        lines.append(f"  • #{short} {title}")
        keyboard.append([{"text": f"↩️ Reopen: {title[:18]} (#{short})", "callback_data": f"reopen:{did}"}])
    client.send_message(chat_id, "\n".join(lines), reply_markup={"inline_keyboard": keyboard})
    return len(rows)


def send_deferred_list(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str) -> int:
    """DM a list of deferred drafts, each with an inline 'review now' button.

    Tapping a draft clears its defer window (``/later``) and returns it to the
    review surface immediately — the missing "review now" shortcut for a
    deferred draft (GAP 4). Returns the count listed.
    """
    rows = review.deferred_drafts(conn)
    if not rows:
        client.send_message(
            chat_id,
            "No deferred drafts. (/later defers one 30 days from /reviewdraft "
            "or /listapproved.)",
        )
        return 0
    lines = [
        "⏸ Deferred drafts (tap to clear the 30-day wait and bring one back to review now; #id is the draft number):",
        "  ⏩ Review now",
    ]
    keyboard = []
    for r in rows:
        title = r["title"] or "(unknown)"
        did = r["draft_id"]
        short = r.get("short_code") or did
        lines.append(f"  • #{short} {title}")
        keyboard.append([{"text": f"⏩ Review now: {title[:16]} (#{short})", "callback_data": f"reviewnow:{did}"}])
    client.send_message(chat_id, "\n".join(lines), reply_markup={"inline_keyboard": keyboard})
    return len(rows)


def send_draft_content(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str, draft_id: str) -> list[dict]:
    """Send a single draft's full content (chunked), with an 'Add notes' button.

    Used when the reviewer opens an approved draft from /listapproved. The last
    chunk carries an inline 'Add notes' button (callback ``notes:<id>``) so they
    can annotate it without leaving Telegram. It also appends the current post
    copy (with its ``N/limit`` char count) and a '📝 Copy' button that opens the
    edit/regenerate view — so opening an approved draft shows the caption too.
    """
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
        sent = _send_long(client, chat_id, text, reply_markup=notes_btn)
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
                        caption=f"🖼️ Story image for `{draft_id}`"
                        + (f" — prompt: {info['image_prompt']}" if info.get("image_prompt") else ""),
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
    from humorhist.copywriter import char_limit

    limit = char_limit()
    rows = review.queued_drafts(conn)
    # only show rows that are still queued (unpublished)
    rows = [r for r in rows if not r["published"]]
    if not rows:
        client.send_message(
            chat_id,
            "Queue is empty (nothing approved + queued). Approve drafts with "
            "/reviewdraft, then they appear here ready to publish.",
        )
        return 0
    lines = [f"📋 Queued drafts ({len(rows)}) — edit copy before publishing (#id = draft number):"]
    lines.append("  ✏️ Edit copy   🗑 Remove from queue   ❌ Reject draft (confirm)   ↩️ Reopen to pending")
    keyboard: list[list[dict]] = []
    for r in rows:
        title = r["title"] or "(unknown)"
        did = r["draft_id"]
        short = r.get("short_code") or did
        copy = r.get("post_copy")
        if copy:
            snippet = copy if len(copy) <= 80 else copy[:77] + "..."
            status = f"{len(copy)}/{limit}"
        else:
            snippet = "(no copy yet)"
            status = f"0/{limit}"
        lines.append(f"  • #{short} {title}\n    {snippet}  [{status}]")
        link = db.get_source_link(conn, did)
        if link:
            lines.append(f"    🔗 {db.shorten_url(link, shorten=False)}")
        # Row 1: edit copy + remove from queue (distinct icons; 'Remove' no longer
        # shares the ↩️ glyph with 'Reopen').
        keyboard.append(
            [
                {"text": f"✏️ Edit copy #{short}", "callback_data": f"copy:{did}"},
                {"text": f"🗑 Remove #{short}", "callback_data": f"remove:{did}"},
            ]
        )
        # Row 2: reject (confirm-gated) + reopen to pending.
        keyboard.append(
            [
                {"text": f"❌ Reject #{short}", "callback_data": f"reject:{did}"},
                {"text": f"↩️ Reopen #{short}", "callback_data": f"reopen:{did}"},
            ]
        )
    client.send_message(chat_id, "\n".join(lines), reply_markup={"inline_keyboard": keyboard})
    return len(rows)


def send_copy_content(conn: sqlite3.Connection, client: TelegramTransport, chat_id: str, draft_id: str) -> list[dict]:
    """Send a queued draft's post copy with inline Edit + Regenerate buttons.

    Tapping 'Edit' (callback ``editmenu:<id>``) opens a choice menu — 'Regenerate'
    (callback ``regencopy:<id>``, works with or without editor notes) and 'Type
    myself' (callback ``editcopy:<id>``, a reply that replaces the copy). Tapping
    'Regenerate' asks the LLM for fresh copy and re-sends. The copy is read from
    ``queue.post_copy``; if absent, a hint to regenerate is shown.
    """
    from humorhist.copywriter import char_limit

    row = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    if row is None:
        client.send_message(chat_id, f"No such draft: {draft_id}")
        return []
    q = conn.execute("SELECT post_copy, post_copy_at FROM queue WHERE draft_id = ?", (draft_id,)).fetchone()
    limit = char_limit()
    copy = (dict(q) if q else {}).get("post_copy") if q else None

    pool = db.get_pool_item(conn, row["pool_id"])
    title = pool["title"] if pool else "(unknown)"
    header = f"✏️ POST COPY — {title}\n({len(copy) if copy else 0}/{limit} chars)\n"
    body = copy or "(no copy yet — tap 🔄 Regenerate)"
    markup = {
        "inline_keyboard": [
            [
                {"text": "✏️ Edit", "callback_data": f"editmenu:{draft_id}"},
                {"text": "🔄 Regenerate", "callback_data": f"regencopy:{draft_id}"},
            ]
        ]
    }
    try:
        return _send_long(client, chat_id, header + "\n" + body, reply_markup=markup)
    except Exception as exc:  # noqa: BLE001
        logger.error("[telegram] failed to send copy content %s: %s", draft_id, exc)
        return []


def telegram_harvest(conn, client, chat_id, *, seed: bool = True, wikipedia: bool = True) -> None:
    """Inbound /harvest: top up the candidate pool from seed + Wikipedia lists.

    No LLM needed. Sends a progress line, runs the harvest, then the new pool
    totals. A long harvest briefly blocks the single-threaded poll loop, which
    is acceptable — the user sees the "harvesting…" line immediately.
    """
    from humorhist.harvest.seed import load_seed
    from humorhist.harvest.wikipedia_lists import harvest_wikipedia_lists

    client.send_message(chat_id, "🌾 Harvesting new events…")
    added_seed = added_wiki = None
    try:
        if seed:
            added_seed = load_seed(conn)
        if wikipedia:
            added_wiki = harvest_wikipedia_lists(conn)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the loop
        client.send_message(chat_id, f"⚠️ Harvest error: {exc}")
        return
    totals = db.counts(conn)
    parts = []
    if seed:
        parts.append(f"seed +{added_seed}" if added_seed is not None else "seed skipped")
    if wikipedia:
        parts.append(f"wiki +{added_wiki}" if added_wiki is not None else "wiki skipped")
    client.send_message(
        chat_id,
        f"🌾 Harvest done ({'; '.join(parts)}). Pool now: {totals}",
    )


def telegram_screen(conn, client, chat_id, *, batch_size: int = 20, limit: int | None = None) -> None:
    """Inbound /screen: LLM-score unscored pool candidates."""
    from humorhist.harvest.screen import screen_pool

    llm = _get_llm(chat_id, client)
    if llm is None:
        return
    client.send_message(chat_id, f"🔍 Scoring the pool with the LLM (batch {batch_size})…")
    try:
        result = screen_pool(conn, llm, batch_size=batch_size, limit=limit)
    except Exception as exc:  # noqa: BLE001
        client.send_message(chat_id, f"⚠️ Screening error: {exc}")
        return
    client.send_message(chat_id, f"🔍 Screened. {result}")


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
        client.send_message(chat_id, f"⚠️ No such draft: `{draft_id}`.")
        return
    if row["status"] != "approved":
        client.send_message(
            chat_id,
            f"⚠️ `{draft_id}` is '{row['status']}', not approved/queued. Images are generated for approved drafts only.",
        )
        return
    if not conn.execute("SELECT 1 FROM queue WHERE draft_id = ?", (draft_id,)).fetchone():
        client.send_message(
            chat_id,
            f"⚠️ `{draft_id}` has no queue row yet — approve it (or /queue enqueue) first.",
        )
        return
    client.send_message(chat_id, f"🖼️ Generating story image for `{draft_id}`…")
    try:
        from humorhist import imagegen as ig

        # _resolve_image_dir falls back to <repo>/data/images; the str() default
        # covers its documented "shouldn't happen" None so we never pass None.
        out_dir = _resolve_image_dir(None) or str(Path(__file__).resolve().parent.parent / "data" / "images")
        result = ig.generate_image_for_queue(conn, draft_id, out_dir=out_dir)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the loop
        client.send_message(chat_id, f"⚠️ Image generation failed for `{draft_id}`: {exc}")
        return
    if result is None:
        client.send_message(
            chat_id,
            "🖼️ Story image skipped — no HUMORHIST_IMAGE_API_KEY set. Set it to enable images (post copy unaffected).",
        )
        return
    path, prompt = result
    client.send_message(chat_id, f"🖼️ Image ready for `{draft_id}`:\n{prompt}")
    try:
        with open(path, "rb") as fh:
            client.send_photo(chat_id, fh.read(), caption=f"🖼️ Story image for `{draft_id}`")
    except Exception as exc:  # noqa: BLE001 - photo is a bonus; the prompt + path are persisted
        logger.warning("[telegram] failed to send image for %s: %s", draft_id, exc)


def telegram_buffer(conn, client, chat_id, *, enqueue: bool = False, auto_draft: bool = True) -> None:
    """Inbound /buffer: report buffer health, optionally sweep + top up.

    `/buffer`           -> health report (+ auto-draft if the LLM is available)
    `/buffer enqueue`   -> also move approved drafts into the queue first
    """
    import humorhist.buffer as buf

    if enqueue:
        n = review.enqueue_approved(conn, image_dir=_resolve_image_dir(None))
        client.send_message(chat_id, f"📥 Enqueued {n} approved draft(s) into the queue.")

    llm = _get_llm(chat_id, client) if auto_draft else None
    result = buf.run_buffer_check(conn, client=llm, auto_draft=bool(llm), chat_id=None, telegram=None)
    client.send_message(chat_id, buf.health_message(result, will_draft=bool(llm)))
    if result.get("drafted"):
        client.send_message(chat_id, f"✍️ Auto-drafted {result['drafted']} candidate(s). Send /reviewdraft.")
    if result.get("draft_error"):
        client.send_message(chat_id, f"⚠️ Auto-draft error: {result['draft_error']}")


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
            client.send_message(chat_id, f"↩️ Removed `{draft_id}` from the queue (kept approved).")
        else:
            client.send_message(chat_id, f"⚠️ `{draft_id}` was not in the queue.")
    else:
        send_queue_list(conn, client, chat_id)


def telegram_draft(conn, client, chat_id, *, count: int = 3, min_score: float = 7.0) -> None:
    """Inbound /draft: fact-check + generate angles for top pool candidates.

    This is the upstream hole for a Telegram-only editor: it's the only way to
    turn pool candidates into reviewable pending drafts without the CLI.
    """
    from humorhist.drafting import draft_candidates

    llm = _get_llm(chat_id, client)
    if llm is None:
        return
    client.send_message(chat_id, f"✍️ Drafting {count} candidate(s) (min score {min_score})…")
    try:
        result = draft_candidates(conn, llm, count=count, min_score=min_score)
    except Exception as exc:  # noqa: BLE001
        client.send_message(chat_id, f"⚠️ Drafting error: {exc}")
        return
    pending = len(review.pending_drafts(conn))
    drafted = result.get("drafted", 0)
    client.send_message(
        chat_id,
        f"✍️ Drafted {drafted} new draft(s). {pending} pending total — send /reviewdraft to review them.",
    )


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
    draft = next(iter(pending), None)
    if draft is None:
        client.send_message(
            chat_id,
            "✅ All caught up — no drafts pending. Send /reviewdraft any time "
            "to check again (new drafts from /draft or the daily timer will "
            "appear here).",
        )
        return 0
    _send_one(conn, client, chat_id, draft)
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
    offset_ref = [0]
    awaiting: dict[str, dict] = {}
    decided = 0

    def _handle(upd: dict) -> None:
        offset_ref[0] = max(offset_ref[0], upd.get("update_id", 0) + 1)
        # ⏭ Next button: re-enter the (non-modal) review session to pull the
        # next pending draft, without the user typing /reviewdraft again.
        if "callback_query" in upd:
            data = (upd.get("callback_query") or {}).get("data") or ""
            if data.startswith("next:"):
                run_review_session(
                    conn,
                    client,
                    chat_id,
                    awaiting,
                    offset_ref,
                    poll_timeout=poll_timeout,
                    max_iterations=max_iterations,
                    image_dir=image_dir,
                )
                return
        if "callback_query" in upd:
            res = handle_callback(conn, client, chat_id, upd)
            # A committed decision (the confirm: step) carries "decision" but
            # not "confirming"; the initial tap carries "confirming": True and
            # must NOT count as a decision yet.
            if res and "decision" in res and "confirming" not in res:
                nonlocal decided
                decided += 1
                # A reject (no note_message_id) just records the decision and
                # finishes; only an approve registers a joke/notes capture prompt.
                if "note_message_id" in res:
                    awaiting[res["draft_id"]] = {
                        "note_message_id": res["note_message_id"],
                        "stage": res.get("stage", "editor_line"),
                        "decision": res.get("decision", "approve"),
                    }
            elif res and "note_message_id" in res:
                # notes: button from /listapproved
                awaiting[res["draft_id"]] = {"note_message_id": res["note_message_id"]}
            elif res and "editcopy_message_id" in res:
                # post-copy edit prompt: register so the next reply is captured
                awaiting[res["draft_id"]] = {"editcopy_message_id": res["editcopy_message_id"]}
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
        nonlocal decided
        cmd = text.split()[0].lower()
        if cmd == "/reviewdraft":
            # /reviewdraft        -> normal flow (each tap opens a Confirm/Cancel gate)
            # /reviewdraft fast   -> commits on the first tap (no confirm gate);
            #                        recovery is the per-draft undo (reject/reopen
            #                        from /listapproved or /listqueue)
            fast = "fast" in text.split()
            decided += run_review_session(
                conn,
                client,
                chat_id,
                awaiting,
                offset_ref,
                poll_timeout=poll_timeout,
                max_iterations=max_iterations,
                fast=fast,
            )
        elif cmd == "/listapproved":
            send_approved_list(conn, client, chat_id)
        elif cmd == "/listqueue":
            send_queue_list(conn, client, chat_id)
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
        elif cmd == "/later":
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                client.send_message(chat_id, "Usage: /later <draft_id>")
            else:
                did = parts[1].strip()
                try:
                    review.defer_draft(conn, did)
                except ValueError as exc:
                    client.send_message(chat_id, f"Cannot defer: {exc}")
                else:
                    client.send_message(chat_id, f"⏸ `{did}` deferred 30 days.")
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
                        f"⏩ `#{did}` brought forward — it's back in the review queue. Send /reviewdraft to see it.",
                    )
        elif cmd == "/setjoke":
            rest = text.split(maxsplit=1)
            if len(rest) < 2:
                client.send_message(chat_id, "Usage: /setjoke <draft_id> (then reply with the joke)")
            else:
                did = rest[1].strip()
                prompt = client.send_message(
                    chat_id,
                    f"Reply with the one-line joke (editor_line) for approved "
                    f"draft `{did}` — the human voice for the post copy "
                    f"(or /skip to leave it blank):",
                )
                awaiting[did] = {
                    "note_message_id": prompt["message_id"],
                    "stage": "editor_line",
                    "decision": "approve",
                    "setjoke": True,
                }
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
            limit = int(rest[1]) if len(rest) > 1 and rest[1].strip().isdigit() else None
            telegram_screen(conn, client, chat_id, limit=limit)
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
        elif cmd == "/image":
            # /image <id>  -> (re)generate the story image for an approved+queued draft
            rest = text.split(maxsplit=1)
            telegram_image(conn, client, chat_id, rest[1].strip() if len(rest) > 1 else None)
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
    last_pending: int | None = None
    while True:
        iters += 1
        if iters > max_iterations:
            return decided
        # Proactive nudge: if new pending drafts appeared since we last looked
        # (created by /draft, the daily timer, etc.), tell the reviewer once.
        try:
            pending_now = len(review.pending_drafts(conn))
        except Exception:  # noqa: BLE001 - DB read must never break the poll loop
            pending_now = last_pending or 0
        if last_pending is not None and pending_now > last_pending:
            client.send_message(
                chat_id,
                f"🆕 {pending_now - last_pending} new draft(s) awaiting review "
                f"({pending_now} total). Send /reviewdraft.",
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
    text = format_reviewed_summary(review.reviewed_summary(conn))
    client.send_message(chat_id, text)
    stuck = review.stuck_captures(conn)
    if stuck:
        lines = [
            f"⚠️ {len(stuck)} approved draft(s) are missing their one-line joke "
            f"(committed but the joke was never captured):"
        ]
        keyboard: list[list[dict]] = []
        for s in stuck:
            short = s.get("short_code") or s["draft_id"]
            lines.append(f"  • #{short} {s['title'] or '(unknown)'}")
            keyboard.append(
                [{"text": f"📝 Add joke: #{short}", "callback_data": f"setjoke:{s['draft_id']}"}]
            )
        lines.append("Tap a draft above to add its joke, or run /setjoke <id>.")
        client.send_message(
            chat_id,
            "\n".join(lines),
            reply_markup={"inline_keyboard": keyboard},
        )
    return text
