#!/usr/bin/env python3
"""Screen every NULL-scored pool row using the unattended LLM credential path.

Uses the same static-credential screening client as other screening entrypoints.
Re-reads the client each loop to pick up credential changes.
Designed to run as a durable systemd --user unit.

Usage:
    python3 scripts/screen_all.py [--batch-size 20] [--limit N] [--db PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "data" / "screen_all.log"

sys.path.insert(0, str(REPO))

import humorhist.env as env  # noqa: E402

env.load_env()


def log(msg: str) -> None:
    stamp = datetime.now(UTC).strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def _screen_ids(result: dict) -> str:
    """Return only generated identifiers, never arbitrary provider text."""
    values = [("run_id", result.get("run_id"))]
    batches = result.get("batch_ids")
    if isinstance(batches, list):
        values.extend(("batch_id", value) for value in batches)
    return " ".join(f"{key}={value}" for key, value in values
                    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "data" / "humorhist.sqlite"))
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import humorhist.db as db
    from humorhist.harvest.screen import format_screen_failures, screen_pool
    from humorhist.llm import LLMUnavailable, screening_client

    conn = db.connect(args.db)
    db.migrate(conn)

    # screen_pool only selects NULL rows, so once a batch is scored those rows
    # leave the pool and the next iteration naturally sees fewer. Loop until the
    # NULL set is empty (or a --limit cap is reached).
    n_unscored = conn.execute(
        "SELECT COUNT(*) AS n FROM pool WHERE funny_score IS NULL"
    ).fetchone()["n"]
    log(f"start: {n_unscored} unscored rows (batch={args.batch_size}, limit={args.limit})")

    scored_total = 0
    failed_total = 0
    client_unavailable = False
    preflight_failed = False
    selection_incomplete = False
    release_uncertain = False
    while True:
        if scored_total and args.limit is not None and scored_total >= args.limit:
            break
        try:
            client = screening_client(timeout=300.0)
        except LLMUnavailable:
            log("skipping batch: screening client unavailable")
            client_unavailable = True
            break
        if args.limit is not None and scored_total >= args.limit:
            break
        log(f"loop | scored_so_far={scored_total}")
        result = screen_pool(
            conn, client, batch_size=args.batch_size, limit=None if args.limit is None else args.limit - scored_total
        )
        reasons = format_screen_failures(result.get("failures"))
        uncertain = result.get("release_uncertain") is True
        release_uncertain |= uncertain
        log(f"  scored={result['scored']} failed_batches={result['failed_batches']}"
            + f" release_uncertain={uncertain}"
            + (f" reasons={reasons}" if reasons else "") + f" {_screen_ids(result)}")
        if uncertain:
            log("WARNING: reservation release uncertain; reservations may remain pending. Inspect screening status.")
        eligible = result.get("eligible_selected")
        if type(eligible) is int and eligible > result["scored"]:
            selection_incomplete = True
            log(f"selection_incomplete=True eligible_selected={eligible} scored={result['scored']}")
        if uncertain:
            scored_total += result["scored"]
            failed_total += result["failed_batches"]
            break
        if result.get("preflight_failed"):
            preflight_failed = True
            failed_total += result["failed_batches"]
            scored_total += result["scored"]
            break
        if result["failed_batches"]:
            failed_total += result["failed_batches"]
            scored_total += result["scored"]
            break  # do not continue after a systemic provider failure
        if result["scored"] == 0:
            break  # nothing left to score
        scored_total += result["scored"]
        failed_total += result["failed_batches"]

    left = conn.execute("SELECT COUNT(*) AS n FROM pool WHERE funny_score IS NULL").fetchone()["n"]
    avg = conn.execute(
        "SELECT round(avg(funny_score),2) AS a FROM pool WHERE funny_score IS NOT NULL"
    ).fetchone()["a"]
    log(f"DONE scored_total={scored_total} failed={failed_total} | unscored_left={left} | avg={avg}")
    has_attempts = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='screen_attempts'"
    ).fetchone()
    blocked = conn.execute(
        "SELECT COUNT(*) FROM screen_attempts a JOIN pool p ON p.id=a.pool_id WHERE p.funny_score IS NULL"
    ).fetchone()[0] if has_attempts else 0
    log(f"blocked={blocked}")
    return int(bool(client_unavailable or preflight_failed or failed_total or release_uncertain or
                    selection_incomplete or (args.limit is None and (blocked or left))))


if __name__ == "__main__":
    raise SystemExit(main())
