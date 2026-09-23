#!/usr/bin/env python3
"""Long-running drafting worker for humorhist.

Builds its LLM client from the unattended credential path (HUMORHIST_LLM_API_KEY,
falling back to the live Nous OAuth token) via resilient_client(). Re-creates the
client each item so a long run survives the hourly token expiry. Usage:
    python3 scripts/run_drafts.py --count 5 --min-score 8

Writes progress to data/worker.log and exits non-zero on fatal error.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LOG = REPO / "data" / "worker.log"


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
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=8.0)
    args = ap.parse_args()

    import humorhist.db as db
    from humorhist.drafting import draft_one, select_candidates
    from humorhist.llm import LLMUnavailable, resilient_client

    conn = db.connect(args.db)
    db.migrate(conn)

    rows = select_candidates(conn, count=args.count, min_score=args.min_score)
    log(f"selected {len(rows)} candidates (min_score={args.min_score})")
    if not rows:
        log("nothing to draft - pool may need screening first")
        return 0

    drafted, failed = 0, 0
    for i, row in enumerate(rows, 1):
        title = row["title"]
        log(f"({i}/{len(rows)}) drafting: {title[:60]}")
        try:
            client = resilient_client(timeout=300.0)
        except LLMUnavailable as exc:
            log(f"    SKIP {title[:40]}: no LLM credential ({exc})")
            failed += 1
            continue
        started = time.time()
        try:
            draft_id = draft_one(conn, client, row)
            drafted += 1
            log(f"    OK {draft_id} in {time.time()-started:.0f}s")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log(f"    FAILED after {time.time()-started:.0f}s: {str(exc)[:200]}")

    log(f"DONE drafted={drafted} failed={failed}")
    return 0 if drafted else 1


if __name__ == "__main__":
    sys.exit(main())
