#!/usr/bin/env python3
"""Durable Telegram review-loop runner (Phase 3.3).

Long-polls Telegram for Approve/Reject taps on pending drafts and writes the
decision via humorhist.review.apply_review. Intended to run as a systemd --user
unit so it survives logout (Linger=yes). Stop with `systemctl --user stop` or
Ctrl-C.

Config (env, never in the repo):
    HUMORHIST_TELEGRAM_BOT_TOKEN   required
    HUMORHIST_TELEGRAM_CHAT_ID      the reviewer's chat id
    HUMORHIST_DB                   optional db path
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("HUMORHIST_LLM_MODEL", "tencent/hy3:free")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("HUMORHIST_DB", str(REPO / "data" / "humorhist.sqlite")))
    ap.add_argument("--once", action="store_true", help="process queued updates and exit")
    args = ap.parse_args()

    import humorhist.db as db
    import humorhist.telegram as tg

    token = os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("HUMORHIST_TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("error: set HUMORHIST_TELEGRAM_BOT_TOKEN and HUMORHIST_TELEGRAM_CHAT_ID")
        return 2

    conn = db.connect(args.db)
    db.migrate(conn)
    client = tg.TelegramClient()

    if args.once:
        decided = tg.run_review_bot(conn, client, chat_id, once=True)
        print(f"telegram-review (once): {decided} decision(s) processed")
        return 0

    print("telegram-review loop started (Ctrl-C to stop)", flush=True)
    tg.run_review_bot(conn, client, chat_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
