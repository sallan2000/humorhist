#!/usr/bin/env python3
"""Long-running drafting worker for humorhist.

Designed to survive disconnection. Re-reads the Nous OAuth token from
~/.hermes/auth.json before every LLM stage, so it picks up token refreshes
performed by the running Hermes process instead of dying when the token it
started with expires.

Usage:
    python3 scripts/run_drafts.py --count 5 --min-score 8

Writes progress to data/worker.log and exits non-zero on fatal error.
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
sys.path.insert(0, str(REPO))

AUTH = Path.home() / ".hermes" / "auth.json"
LOG = REPO / "data" / "worker.log"


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def current_token() -> str:
    """Read the freshest Nous access token from Hermes' auth store."""
    data = json.loads(AUTH.read_text())
    return data["providers"]["nous"]["access_token"]


def token_expiry_minutes() -> float:
    try:
        data = json.loads(AUTH.read_text())
        exp = data["providers"]["nous"]["expires_at"]
        t = datetime.fromisoformat(exp.replace("Z", "+00:00"))
        return (t - datetime.now(timezone.utc)).total_seconds() / 60
    except Exception:
        return -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "data" / "humorhist.sqlite"))
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=8.0)
    args = ap.parse_args()

    import humorhist.db as db
    from humorhist.drafting import draft_one, select_candidates
    from humorhist.llm import NousClient

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
        mins = token_expiry_minutes()
        log(f"({i}/{len(rows)}) drafting: {title[:60]} | token {mins:.0f}m left")

        # Fresh token per item so Hermes' background refresh is picked up.
        os.environ["HUMORHIST_LLM_API_KEY"] = current_token()
        client = NousClient(api_key=current_token(), max_retries=2, timeout=300.0)

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
