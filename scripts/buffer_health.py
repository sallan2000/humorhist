#!/usr/bin/env python3
"""Daily buffer-health monitor for humorhist (Phase 3.4).

Runs on a systemd --user timer (see humorhist-buffer.timer). Each run:

  1. Computes buffer depth (unpublished queued drafts == days of buffer).
  2. If pending drafts are scarce (< 5), auto-drafts more candidates
     (best-effort; needs the Nous LLM token from ~/.hermes/auth.json).
  3. If the buffer is low (< 7 days) or critical (< 3), DMs a Telegram alert.
     Healthy buffers stay silent — no nagging when there's a week+ of content.

The LLM token is re-read per network call so a long run survives the hourly
Nous token expiry (it only refreshes while Hermes runs). A missing token just
means auto-draft is skipped; the report + Telegram alert still work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AUTH = Path.home() / ".hermes" / "auth.json"
LOG = REPO / "data" / "buffer_health.log"

sys.path.insert(0, str(REPO))

import humorhist.env as env  # noqa: E402

env.load_env()

os.environ.setdefault("HUMORHIST_LLM_MODEL", "tencent/hy3:free")


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def current_token() -> str | None:
    try:
        return json.loads(AUTH.read_text())["providers"]["nous"]["access_token"]
    except Exception:  # noqa: BLE001 - token is optional for the report
        return None


def _llm():
    from humorhist.llm import NousClient

    token = current_token()
    if not token:
        import humorhist.llm as llm

        return llm.NousClient(api_key="", max_retries=1, timeout=120.0)
    return NousClient(api_key=token, max_retries=2, timeout=300.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "data" / "humorhist.sqlite"))
    ap.add_argument(
        "--no-auto-draft",
        action="store_true",
        help="report only; never draft new candidates",
    )
    ap.add_argument(
        "--no-notify",
        action="store_true",
        help="never send a Telegram alert (report to stdout/log only)",
    )
    args = ap.parse_args()

    import humorhist.db as db
    import humorhist.buffer as buf
    from humorhist import telegram as tg

    conn = db.connect(args.db)
    db.migrate(conn)

    client = None if args.no_auto_draft else _llm()
    telegram = None
    chat_id = os.environ.get("HUMORHIST_TELEGRAM_CHAT_ID")
    if not args.no_auto_draft and not os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN"):
        log("AUTO-DRAFT: no bot token -> LLM key path still attempted for drafting")
    if not args.no_notify and chat_id and os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN"):
        telegram = tg.TelegramClient()
    else:
        log("NOTIFY: skipped (--no-notify or Telegram not configured)")

    result = buf.run_buffer_check(
        conn,
        client=client,
        auto_draft=not args.no_auto_draft,
        chat_id=chat_id,
        telegram=telegram,
    )
    log(buf.health_message(result))
    if result.get("drafted"):
        log(f"AUTO-DRAFT: {result['drafted']} candidate(s) drafted")
    if result.get("draft_error"):
        log(f"AUTO-DRAFT error: {result['draft_error']}")
    if result.get("notified"):
        log("NOTIFY: low-buffer alert sent to Telegram")
    if result.get("notify_error"):
        log(f"NOTIFY error: {result['notify_error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
