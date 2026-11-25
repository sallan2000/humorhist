#!/usr/bin/env python3
"""Preserve existing drafts and draft fresh candidates.

Despite this script's historical name, it does not regenerate existing drafts:
replacing user-reviewed or edited content is not supported. The fresh phase uses
resilient_client() and selects pool rows with status='new' and sufficient score.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LOG = REPO / "data" / "regen_drafts.log"

# Load a local .env (gitignored) into os.environ so HUMORHIST_TELEGRAM_* work
# without manual exports.
import humorhist.env as env  # noqa: E402

env.load_env()

os.environ.setdefault("HUMORHIST_LLM_MODEL", "tencent/hy3:free")

logger = logging.getLogger("regen_drafts")


def log(msg: str) -> None:
    stamp = datetime.now(UTC).strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "data" / "humorhist.sqlite"))
    ap.add_argument("--fresh-count", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=7.0)
    args = ap.parse_args()

    import humorhist.db as db
    from humorhist.job_control import lock_path_for_db, try_acquire

    conn = db.connect(args.db)
    try:
        db.migrate(conn)
        lock = try_acquire(lock_path_for_db(conn))
        if lock is None:
            log("BUSY: another drafting job holds the lock")
            return 1
        with lock:
            return _run(conn, args)
    finally:
        conn.close()


def _run(conn, args) -> int:
    from humorhist.drafting import draft_candidates
    from humorhist.llm import LLMUnavailable, resilient_client

    # Existing drafts may contain editorial work; never replace them here.
    existing = conn.execute("SELECT COUNT(DISTINCT pool_id) n FROM drafts").fetchone()["n"]
    log(f"REGEN: skipped {existing} existing draft pool item(s); "
        "no regeneration occurred (replacement is not supported)")

    # --- Phase 2: fresh candidates ----------------------------------------
    log(f"FRESH: drafting {args.fresh_count} new candidates (min_score={args.min_score})")
    try:
        client = resilient_client(timeout=300.0)
    except LLMUnavailable as exc:
        log(f"FRESH: no LLM credential -- skipping ({exc})")
        result = {"drafted": 0, "failed": 1}
    else:
        log("  drafting...")
        result = draft_candidates(
            conn, client, count=args.fresh_count, min_score=args.min_score
        )
    log(f"FRESH result: {result}")

    # --- Summary ----------------------------------------------------------
    total = conn.execute("SELECT COUNT(*) n FROM drafts").fetchone()["n"]
    pending = conn.execute(
        "SELECT COUNT(*) n FROM drafts WHERE status='pending'"
    ).fetchone()["n"]
    log(f"DONE. total drafts={total} pending={pending}")

    # --- Phase 3.4 nudge: tell the reviewer new drafts await (best effort) ---
    chat_id = os.environ.get("HUMORHIST_TELEGRAM_CHAT_ID")
    if pending and chat_id and os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN"):
        try:
            from humorhist import telegram as tg

            n = tg.notify_new_drafts(conn, tg.TelegramClient(), chat_id)
            log(f"NOTIFY: nudged Telegram with {n} pending draft(s)")
        except Exception as exc:  # noqa: BLE001 - nudge must never break the run
            log(f"NOTIFY: skipped ({str(exc)[:120]})")
    return int(bool(result.get("failed")))


if __name__ == "__main__":
    raise SystemExit(main())
