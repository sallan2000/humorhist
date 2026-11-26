#!/usr/bin/env python3
"""Daily buffer-health monitor for humorhist (Phase 3.4).

Runs on a systemd --user timer (see humorhist-buffer.timer). Each run:

  1. Computes buffer depth (unpublished queued drafts == days of buffer).
  2. If pending drafts are scarce (< 5), auto-drafts more candidates
     (best-effort; uses HUMORHIST_LLM_API_KEY or the live Nous OAuth token).
  3. DMs a Telegram alert when the buffer is low or the job fails.

  The LLM credential is re-read per network call via resilient_client() so a long
  run survives the hourly Nous token expiry. A missing credential is a failed run
  if auto-drafting is needed; the
  report + best-effort Telegram alert still run. Both jobs return 1 for
  failures (including partial candidate failures or notification errors).
  Failure summaries are for this run only; no historical state is stored.
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
    from humorhist.job_control import lock_path_for_db, try_acquire

    failures = []
    diagnostics = ""
    draft_summary = ""
    draft_details = ""

    def failed(phase, exc):
        name = type(exc).__name__ if type(exc) in (RuntimeError, ValueError, OSError) else "Error"
        failures.append(f"{phase}:{name}")
        log(f"{phase.upper()} FAILED: {name}")

    def notify(summary):
        if args.no_notify:
            log("NOTIFY: skipped (--no-notify)")
            return
        chat_id = os.environ.get("HUMORHIST_TELEGRAM_CHAT_ID")
        if not (chat_id and os.environ.get("HUMORHIST_TELEGRAM_BOT_TOKEN")):
            failures.append("notify:NotConfigured")
            log("NOTIFY: skipped (Telegram not configured)")
            return
        try:
            message = summary + "\nFailures this run: " + (", ".join(failures) or "none")
            if draft_summary:
                message += "\n" + draft_summary
            if draft_details:
                message += "\nFailures: " + draft_details
            tg.TelegramClient().send_message(chat_id, message)
            log("NOTIFY: buffer/job alert sent to Telegram")
        except Exception as exc:  # noqa: BLE001
            failed("notify", exc)

    conn = None
    try:
        conn = db.connect(args.db)
        db.migrate(conn)
    except Exception as exc:  # noqa: BLE001
        failed("setup", exc)
        if conn is not None:
            conn.close()
        notify("HEALTH: unavailable")
        log(f"DONE: failures={','.join(failures)}")
        return 1

    try:
        try:
            health = buf.buffer_health(conn)
        except Exception as exc:  # noqa: BLE001
            failed("health", exc)
            notify("HEALTH: unavailable")
            log(f"DONE: failures={','.join(failures)}")
            return 1

        if not args.no_auto_draft and health["auto_draft"]:
            try:
                lock = try_acquire(lock_path_for_db(conn))
                if lock is None:
                    failures.append("draft:busy")
                    log("AUTO-DRAFT: busy (another drafting job holds the lock)")
                else:
                    with lock:
                        # Another run may have finished since the initial health check.
                        if not buf.buffer_health(conn)["auto_draft"]:
                            log("AUTO-DRAFT: skipped (buffer replenished)")
                        else:
                            client = _llm()
                            if client is None:
                                failures.append("draft:LLMUnavailable")
                                log("AUTO-DRAFT FAILED: LLMUnavailable")
                            else:
                                from humorhist.drafting import (
                                    describe_failure_counts,
                                    draft_candidates,
                                    format_failure_counts,
                                )

                                result = draft_candidates(conn, client, count=buf.AUTO_DRAFT_COUNT)
                                log(f"AUTO-DRAFT: selected={result['selected']} drafted={result['drafted']} "
                                    f"failed={result['failed']}")
                                if result["failed"]:
                                    failures.append("draft:failed")
                                    draft_summary = (f"Auto-draft: {result['drafted']} created, "
                                                     f"{result['failed']} failed out of {result['selected']} selected.")
                                    draft_details = describe_failure_counts(result.get("failures"))
                                    diagnostics = format_failure_counts(result.get("failures"))
                                    if diagnostics:
                                        log(f"AUTO-DRAFT failures: {diagnostics}")
            except Exception as exc:  # noqa: BLE001
                failed("draft", exc)
        else:
            log("AUTO-DRAFT: skipped")

        # Refresh after drafting, and distinguish future-deferred pending drafts.
        # Published drafts retain approved status; do not count them as available.
        try:
            now = datetime.now(UTC).isoformat()
            row = conn.execute(
                "SELECT SUM(CASE WHEN status='pending' AND (defer_until IS NULL OR defer_until <= ?) "
                "THEN 1 ELSE 0 END) AS ready, "
                "SUM(CASE WHEN status='pending' AND defer_until > ? THEN 1 ELSE 0 END) AS deferred, "
                "SUM(CASE WHEN status='approved' AND NOT EXISTS "
                "(SELECT 1 FROM queue q WHERE q.draft_id=drafts.id AND q.published=1) "
                "THEN 1 ELSE 0 END) AS approved, "
                "SUM(CASE WHEN status='approved' AND EXISTS "
                "(SELECT 1 FROM queue q WHERE q.draft_id=drafts.id AND q.published=1) "
                "THEN 1 ELSE 0 END) AS published FROM drafts", (now, now),
            ).fetchone()
            health = buf.buffer_health(conn)
            summary = (f"HEALTH: ready={row['ready'] or 0} deferred={row['deferred'] or 0} "
                       f"approved={row['approved'] or 0} queued={health['depth']} "
                       f"published={row['published'] or 0} level={health['level']}")
            log(summary)
        except Exception as exc:  # noqa: BLE001
            failed("summary", exc)
            summary = "HEALTH: summary unavailable"

        if health["level"] != "silent" or failures:
            notify(summary)
        else:
            log("NOTIFY: skipped (healthy buffer and no failures)")
        log(f"DONE: failures={','.join(failures) or 'none'}")
        return int(bool(failures))
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
