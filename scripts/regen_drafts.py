#!/usr/bin/env python3
"""Regenerate existing drafts under the new prompts, then draft fresh candidates.

Both phases build their LLM client from the unattended credential path
(HUMORHIST_LLM_API_KEY, falling back to the live Nous OAuth token) via
resilient_client(), re-creating the client each item so a long run survives the
hourly token expiry.

  1. REGEN: re-run draft_one on the pool items backing the existing drafts,
     replacing their brief/angles with output from the current (post
     taste-filter-removal) prompts.
  2. FRESH: draft N new candidates from pool rows still status='new' with
     funny_score >= min_score.

Runs as a durable systemd --user unit.
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
    from humorhist.drafting import draft_candidates, draft_one
    from humorhist.llm import LLMUnavailable, resilient_client

    conn = db.connect(args.db)
    db.migrate(conn)

    # --- Phase 1: regenerate existing drafts -------------------------------
    existing = conn.execute(
        "SELECT DISTINCT pool_id FROM drafts"
    ).fetchall()
    pool_ids = [r["pool_id"] for r in existing]
    log(f"REGEN: {len(pool_ids)} existing draft pool items to regenerate")

    regen_ok, regen_fail = 0, 0
    for pid in pool_ids:
        row = db.get_pool_item(conn, pid)
        if row is None:
            log(f"  skip {pid}: pool row missing")
            continue
        title = row["title"]
        log(f"  regen {title[:50]!r}")
        try:
            client = resilient_client(timeout=300.0)
        except LLMUnavailable as exc:
            log(f"    SKIP {title[:40]}: no LLM credential ({exc})")
            regen_fail += 1
            continue
        try:
            did = draft_one(conn, client, row)
            regen_ok += 1
            log(f"    OK {did}")
        except Exception as exc:  # noqa: BLE001
            regen_fail += 1
            log(f"    FAILED: {str(exc)[:200]}")

    log(f"REGEN done: ok={regen_ok} failed={regen_fail}")

    # --- Phase 2: fresh candidates ----------------------------------------
    log(f"FRESH: drafting {args.fresh_count} new candidates (min_score={args.min_score})")
    try:
        client = resilient_client(timeout=300.0)
    except LLMUnavailable as exc:
        log(f"FRESH: no LLM credential -- skipping ({exc})")
        result = {"drafted": 0, "failed": 0}
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
