#!/usr/bin/env python3
"""Screen every NULL-scored pool row using the unattended LLM credential path.

Uses resilient_client() (HUMORHIST_LLM_API_KEY, falling back to the live Nous
OAuth token) -- the same credential path the weekly pipeline / buffer timer use.
Re-reads the client each loop so a long run survives the hourly token expiry.
Designed to run as a durable systemd --user unit.

Usage:
    python3 scripts/screen_all.py [--batch-size 20] [--limit N] [--db PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "data" / "screen_all.log"

sys.path.insert(0, str(REPO))


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
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.environ.setdefault("HUMORHIST_LLM_MODEL", "tencent/hy3:free")

    import humorhist.db as db
    from humorhist.harvest.screen import screen_pool
    from humorhist.llm import LLMUnavailable, resilient_client

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
    while True:
        if args.limit is not None and scored_total >= args.limit:
            break
        try:
            client = resilient_client(timeout=300.0)
        except LLMUnavailable as exc:
            log(f"skipping batch: no LLM credential ({exc})")
            break
        log(f"loop | scored_so_far={scored_total}")
        result = screen_pool(
            conn, client, batch_size=args.batch_size, limit=None if args.limit is None else args.limit - scored_total
        )
        if result["failed_batches"]:
            failed_total += result["failed_batches"]
            scored_total += result["scored"]
            break  # do not continue after a systemic provider failure
        if result["scored"] == 0:
            break  # nothing left to score
        scored_total += result["scored"]
        failed_total += result["failed_batches"]
        log(
            f"  scored={result['scored']} failed_batches={result['failed_batches']} | total={scored_total}"
        )

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
    return int(bool(failed_total or blocked or (left and (args.limit is None or scored_total < args.limit))))


if __name__ == "__main__":
    raise SystemExit(main())
