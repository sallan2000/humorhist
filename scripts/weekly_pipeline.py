#!/usr/bin/env python3
"""Weekly discovery pipeline: harvest -> screen -> draft net-new -> nudge.

Designed to run on a systemd --user timer (see humorhist-weekly.timer).
Each phase is idempotent and failure-isolated:

  - harvest  : upserts pool rows (INSERT OR IGNORE) -- never duplicates.
  - screen   : scores only NULL funny_score rows -- never overwrites.
  - draft    : selects status='new' pool items and draft_one now SKIPS any
               pool item that already has a draft, so re-running never
               overwrites an existing (possibly reviewed/approved) draft.
  - nudge    : best-effort Telegram ping if fresh pending drafts appeared.

The LLM token is re-read per network call so a long run survives the hourly
Nous token expiry (it only refreshes while Hermes runs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AUTH = Path.home() / ".hermes" / "auth.json"
LOG = REPO / "data" / "weekly_pipeline.log"

sys.path.insert(0, str(REPO))

# Load a local .env (gitignored) into os.environ so HUMORHIST_TELEGRAM_* work
# without manual exports.
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


def current_token() -> str:
    return json.loads(AUTH.read_text())["providers"]["nous"]["access_token"]


def _llm():
    from humorhist.llm import NousClient

    return NousClient(api_key=current_token(), max_retries=2, timeout=300.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "data" / "humorhist.sqlite"))
    ap.add_argument("--draft-count", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=7.0)
    ap.add_argument("--no-harvest", action="store_true", help="skip harvest (screen+draft only)")
    args = ap.parse_args()

    import humorhist.db as db
    from humorhist.drafting import draft_candidates
    from humorhist.harvest.screen import screen_pool
    from humorhist.harvest.seed import load_seed
    from humorhist.harvest.wikipedia_lists import harvest_wikipedia_lists

    conn = db.connect(args.db)
    db.migrate(conn)

    # --- 1. Harvest (idempotent upsert) -----------------------------------
    if not args.no_harvest:
        log("HARVEST: seed events")
        try:
            log(f"  seed: {load_seed(conn)}")
        except Exception as exc:  # noqa: BLE001
            log(f"  seed FAILED: {str(exc)[:200]}")
        log("HARVEST: wikipedia lists")
        try:
            log(f"  wikipedia: {harvest_wikipedia_lists(conn)}")
        except Exception as exc:  # noqa: BLE001
            log(f"  wikipedia FAILED: {str(exc)[:200]}")
    else:
        log("HARVEST: skipped (--no-harvest)")

    # --- 2. Screen unscored (idempotent: NULL funny_score only) ------------
    before = conn.execute("SELECT COUNT(*) n FROM pool WHERE funny_score IS NULL").fetchone()["n"]
    log(f"SCREEN: {before} unscored pool items")
    try:
        res = screen_pool(conn, _llm(), batch_size=20)
        log(f"  screen result: {res}")
    except Exception as exc:  # noqa: BLE001
        log(f"  screen FAILED: {str(exc)[:200]}")

    # --- 3. Draft net-new (draft_one skips existing drafts) ----------------
    pending_before = conn.execute(
        "SELECT COUNT(*) n FROM drafts WHERE status='pending'"
    ).fetchone()["n"]
    log(f"DRAFT: drafting up to {args.draft_count} net-new candidates")
    try:
        result = draft_candidates(
            conn, _llm(), count=args.draft_count, min_score=args.min_score
        )
        log(f"  draft result: {result}")
    except Exception as exc:  # noqa: BLE001
        log(f"  draft FAILED: {str(exc)[:200]}")
        result = {"drafted": 0}

    pending_after = conn.execute(
        "SELECT COUNT(*) n FROM drafts WHERE status='pending'"
    ).fetchone()["n"]
    fresh = pending_after - pending_before

    # --- 4. Nudge (best effort) --------------------------------------------
    chat_id = os.environ.get("HUMORHIST_TELEGRAM_CHAT_ID")
    if fresh and chat_id and os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN"):
        try:
            from humorhist import telegram as tg

            n = tg.notify_new_drafts(conn, tg.TelegramClient(), chat_id)
            log(f"NOTIFY: nudged Telegram with {n} pending draft(s)")
        except Exception as exc:  # noqa: BLE001 - nudge must never break the run
            log(f"NOTIFY: skipped ({str(exc)[:120]})")
    else:
        log(f"NOTIFY: skipped (fresh={fresh}, chat configured={bool(chat_id)})")

    log(f"DONE. fresh pending drafts this run: {fresh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
