#!/usr/bin/env python3
"""Daily buffer-health monitor for humorhist (Phase 3.4).

Runs on a systemd --user timer (see humorhist-buffer.timer). Each run:

  1. Computes buffer depth (unpublished queued drafts == days of buffer).
  2. If pending drafts are scarce (< 5), auto-drafts more candidates
     (best-effort; uses HUMORHIST_LLM_API_KEY or the live Nous OAuth token).
  3. If the buffer is low (< 7 days) or critical (< 3), DMs a Telegram alert.
     Healthy buffers stay silent — no nagging when there's a week+ of content.

  The LLM credential is re-read per network call via resilient_client() so a long
  run survives the hourly Nous token expiry. A missing credential just means
  auto-draft is skipped; the report + Telegram alert still work.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "data" / "buffer_health.log"

sys.path.insert(0, str(REPO))

import humorhist.env as env  # noqa: E402

env.load_env()

os.environ.setdefault("HUMORHIST_LLM_MODEL", "tencent/hy3:free")


def log(msg: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def _llm():
    from humorhist.llm import LLMUnavailable, resilient_client

    try:
        return resilient_client()
    except LLMUnavailable:
        # No static key and no live Nous token -> auto-draft simply can't run.
        return None


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

    import humorhist.buffer as buf
    import humorhist.db as db
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
