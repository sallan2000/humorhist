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

The LLM client is built via resilient_client() (HUMORHIST_LLM_API_KEY, falling
back to the live Nous OAuth token) — the same credential path the buffer timer
and Telegram bot use. Pass a 300s timeout since the hy3 model is slow.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "data" / "weekly_pipeline.log"

sys.path.insert(0, str(REPO))

# Load a local .env (gitignored) into os.environ so HUMORHIST_TELEGRAM_* work
# without manual exports.
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
    from humorhist.llm import resilient_client

    # Use the same unattended credential path as the buffer timer / Telegram bot
    # (HUMORHIST_LLM_API_KEY, falling back to the Nous OAuth token). The old
    # current_token() path read the OAuth token directly and 404'd on the hy3
    # endpoint, so the weekly harvest never produced drafts despite the key being
    # set in .env. Pass a 300s timeout since hy3 is slow.
    return resilient_client(timeout=300.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "data" / "humorhist.sqlite"))
    ap.add_argument("--draft-count", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=7.0)
    ap.add_argument("--no-harvest", action="store_true", help="skip harvest (screen+draft only)")
    args = ap.parse_args()

    import humorhist.db as db
    from humorhist.drafting import draft_candidates, format_failure_counts
    from humorhist.harvest.screen import format_screen_failures, screen_pool
    from humorhist.harvest.seed import load_seed
    from humorhist.harvest.wikipedia_lists import harvest_wikipedia_lists
    from humorhist.job_control import lock_path_for_db, try_acquire

    failures = []
    conn = None
    try:
        conn = db.connect(args.db)
        db.migrate(conn)
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__ if type(exc) in (RuntimeError, ValueError, OSError) else "Error"
        log(f"SETUP FAILED: {name}")
        if conn is not None:
            conn.close()
        return 1

    def failed(phase, exc):
        name = type(exc).__name__ if type(exc) in (RuntimeError, ValueError, OSError) else "Error"
        failures.append(f"{phase}:{name}")
        log(f"{phase} FAILED: {name}")

    try:
        if not args.no_harvest:
            for name, action in (("seed", load_seed), ("wikipedia", harvest_wikipedia_lists)):
                try:
                    if name == "wikipedia":
                        # Harvest logs raw provider exceptions; report only its counts.
                        logger = logging.getLogger("humorhist.harvest.wikipedia_lists")
                        was_disabled = logger.disabled
                        logger.disabled = True
                        try:
                            summary = action(conn)
                        finally:
                            logger.disabled = was_disabled
                    else:
                        summary = action(conn)
                    log(f"HARVEST {name}: inserted={summary['inserted']}"
                        f" pages_failed={summary.get('pages_failed', 0)}"
                        f" pages_unattempted={summary.get('pages_unattempted', 0)}")
                    if summary.get("pages_failed") or summary.get("pages_unattempted"):
                        failures.append(f"{name}:pages_failed")
                except Exception as exc:  # noqa: BLE001
                    failed(name, exc)
        else:
            log("HARVEST: skipped (--no-harvest)")

        try:
            before = conn.execute("SELECT COUNT(*) n FROM pool WHERE funny_score IS NULL").fetchone()["n"]
            log(f"SCREEN: {before} unscored pool items")
            res = screen_pool(conn, _llm(), batch_size=20)
            log(f"SCREEN: scored={res['scored']} batches={res['batches']} failed_batches={res['failed_batches']}")
            reasons = format_screen_failures(res.get("failures"))
            log(f"SCREEN: unscored={max(0, before - res['scored'])} skipped={res.get('skipped', 0)}"
                + (f" reasons={reasons}" if reasons else ""))
            if res["failed_batches"]:
                failures.append("screen:failed_batches")
            if res["scored"] < before:
                failures.append("screen:incomplete")
        except Exception as exc:  # noqa: BLE001
            failed("screen", exc)

        fresh = None
        try:
            lock = try_acquire(lock_path_for_db(conn))
            if lock is None:
                failures.append("draft:busy")
                log("DRAFT: busy (another drafting job holds the lock)")
            else:
                with lock:
                    result = draft_candidates(conn, _llm(), count=args.draft_count, min_score=args.min_score)
                    fresh = result["drafted"]
                    log(f"DRAFT: selected={result['selected']} drafted={fresh} failed={result['failed']}"
                        f" unattempted={result.get('unattempted', 0)}")
                    if result["failed"] or result.get("unattempted"):
                        failures.append("draft:failed")
                        diagnostics = format_failure_counts(result.get("failures"))
                        if diagnostics:
                            log(f"DRAFT failures: {diagnostics}")
        except Exception as exc:  # noqa: BLE001
            failed("draft", exc)

        chat_id = os.environ.get("HUMORHIST_TELEGRAM_CHAT_ID")
        if fresh and chat_id and os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN"):
            try:
                from humorhist import telegram as tg

                n = tg.notify_new_drafts(conn, tg.TelegramClient(), chat_id)
                log(f"NOTIFY: nudged Telegram with {n} pending draft(s)")
            except Exception as exc:  # noqa: BLE001
                failed("notify", exc)
        else:
            if fresh and not (chat_id and os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN")):
                failures.append("notify:NotConfigured")
            log(f"NOTIFY: skipped (fresh={fresh}, chat configured={bool(chat_id)})")
        log(f"DONE: fresh={fresh} failures={','.join(failures) or 'none'}")
        return int(bool(failures))
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
