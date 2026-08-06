#!/usr/bin/env python3
"""Screen every NULL-scored pool row, re-reading the Nous OAuth token per batch.

The live token in ~/.hermes/auth.json expires hourly and only refreshes while
Hermes runs, so we re-read it before every batch instead of constructing one
client up front. Designed to run as a durable systemd --user unit.

Usage:
    python3 scripts/screen_all.py [--batch-size 20] [--limit N] [--db PATH]
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

sys.path.insert(0, str(REPO))

LOG = REPO / "data" / "screen_all.log"


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def current_token() -> str:
    return json.loads(AUTH.read_text())["providers"]["nous"]["access_token"]


def token_expiry_minutes() -> float:
    try:
        d = json.loads(AUTH.read_text())
        exp = datetime.fromisoformat(d["providers"]["nous"]["expires_at"].replace("Z", "+00:00"))
        return (exp - datetime.now(timezone.utc)).total_seconds() / 60
    except Exception:
        return -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "data" / "humorhist.sqlite"))
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.environ.setdefault("HUMORHIST_LLM_MODEL", "tencent/hy3:free")

    import humorhist.db as db
    from humorhist.harvest.screen import screen_pool
    from humorhist.llm import NousClient

    conn = db.connect(args.db)
    db.migrate(conn)

    # screen_pool only selects NULL rows, so once a batch is scored those rows
    # leave the pool and the next iteration naturally sees fewer. Loop until the
    # NULL set is empty (or a --limit cap is reached).
    log(f"start: {conn.execute('SELECT COUNT(*) AS n FROM pool WHERE funny_score IS NULL').fetchone()['n']} unscored rows (batch={args.batch_size}, limit={args.limit})")

    scored_total = 0
    failed_total = 0
    while True:
        if args.limit is not None and scored_total >= args.limit:
            break
        mins = token_expiry_minutes()
        log(f"loop | token {mins:.0f}m left | scored_so_far={scored_total}")
        token = current_token()
        client = NousClient(api_key=token, max_retries=2, timeout=300.0)
        # screen_pool scores all current NULL rows in batches; it returns when
        # none remain (or after --limit rows).
        result = screen_pool(
            conn, client, batch_size=args.batch_size, limit=args.limit
        )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
