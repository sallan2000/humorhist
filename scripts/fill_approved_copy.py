#!/usr/bin/env python3
"""One-shot: generate post copy for approved+queued drafts that lack it.

Builds its LLM client from the unattended credential path
(HUMORHIST_LLM_API_KEY, falling back to the live Nous OAuth token) via
resilient_client(). First runs db.migrate so the live DB gets the
queue.post_copy columns if absent.

Usage:
    python3 scripts/fill_approved_copy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DB = REPO / "data" / "humorhist.sqlite"


def main() -> int:
    import humorhist.copywriter as cw
    import humorhist.db as db
    from humorhist.llm import LLMUnavailable, resilient_client

    conn = db.connect(str(DB))
    db.migrate(conn)  # idempotent; adds queue.post_copy if missing

    try:
        client = resilient_client(timeout=300.0)
    except LLMUnavailable as exc:
        print(f"error: no LLM credential -- cannot generate post copy ({exc})")
        return 2

    filled = cw.fill_post_copy(conn, client, force=True)
    print(f"Generated post copy for {filled} draft(s).")

    # Report the result for the approved+queued rows.
    for r in conn.execute(
        """
        SELECT q.draft_id, q.post_copy, p.title
        FROM queue q
        LEFT JOIN drafts d ON d.id = q.draft_id
        LEFT JOIN pool p ON p.id = d.pool_id
        WHERE d.status = 'approved' AND q.published = 0
        ORDER BY q.id
        """
    ):
        copy = r["post_copy"]
        if copy:
            print(f"\n=== {r['title']} ({r['draft_id']}) -- {len(copy)}/280 ===\n{copy}")
        else:
            print(f"\n!!! {r['title']} ({r['draft_id']}) -- NO COPY GENERATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
