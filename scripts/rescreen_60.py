#!/usr/bin/env python3
"""One-off: re-screen the 60 previously-scored pool rows under the new prompt.

The death/suffering taste-penalty was removed from SCREEN_SYSTEM_PROMPT, but
the 60 rows scored under the old prompt keep their stale (low) scores because
screen_pool only touches NULL rows. This script:
  1. captures the ids of currently-scored rows,
  2. nulls their funny_score,
  3. re-scores exactly those rows via score_batch (the 681 never-screened
     NULL rows are left untouched),
  4. prints before/after for each.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AUTH = Path.home() / ".hermes" / "auth.json"
DB = REPO / "data" / "humorhist.sqlite"

os.environ.setdefault("HUMORHIST_LLM_MODEL", "tencent/hy3:free")

import humorhist.db as db
from humorhist.harvest.screen import score_batch
from humorhist.llm import NousClient

BATCH = 20


def main() -> int:
    token = json.loads(AUTH.read_text())["providers"]["nous"]["access_token"]
    client = NousClient(api_key=token, max_retries=2, timeout=300.0)

    conn = db.connect(str(DB))
    db.migrate(conn)

    before = {
        r["id"]: r["funny_score"]
        for r in conn.execute(
            "SELECT id, funny_score FROM pool WHERE funny_score IS NOT NULL"
        )
    }
    print(f"captured {len(before)} previously-scored rows")

    ids = list(before.keys())
    conn.execute(
        f"UPDATE pool SET funny_score = NULL WHERE id IN ({','.join('?' * len(ids))})",
        ids,
    )
    conn.commit()
    print(f"nulled {len(ids)} scores")

    rows = [
        dict(r)
        for r in conn.execute(
            f"SELECT * FROM pool WHERE id IN ({','.join('?' * len(ids))})", ids
        )
    ]

    re_scored = 0
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        scores = score_batch(client, batch)
        for pid, sc in scores.items():
            db.set_funny_score(conn, pid, sc)
            re_scored += 1
    print(f"re-scored {re_scored} rows")

    print("\nBEFORE -> AFTER  (title | old -> new)")
    for pid in ids:
        title = conn.execute("SELECT title FROM pool WHERE id=?", (pid,)).fetchone()[
            "title"
        ]
        new = conn.execute(
            "SELECT funny_score FROM pool WHERE id=?", (pid,)
        ).fetchone()["funny_score"]
        print(f"  {title[:55]:55} | {before[pid]:>4} -> {new}")

    n = conn.execute(
        "SELECT COUNT(*) AS n FROM pool WHERE funny_score IS NOT NULL"
    ).fetchone()["n"]
    avg = conn.execute(
        "SELECT round(avg(funny_score),2) AS a FROM pool WHERE funny_score IS NOT NULL"
    ).fetchone()["a"]
    print(f"\npool now scored: {n} (avg {avg})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
