#!/usr/bin/env python3
"""One-shot: generate post copy for approved+queued drafts that lack it.

Mirrors the project's token-borrowing pattern (run_drafts.py): reads the
freshest Nous OAuth token from ~/.hermes/auth.json and sets
HUMORHIST_LLM_API_KEY per run. First runs db.migrate so the live DB gets the
queue.post_copy columns if absent.

Usage:
    python3 scripts/fill_approved_copy.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

AUTH = Path.home() / ".hermes" / "auth.json"
DB = REPO / "data" / "humorhist.sqlite"


def current_token() -> str:
    data = json.loads(AUTH.read_text())
    return data["providers"]["nous"]["access_token"]


def main() -> int:
    import humorhist.db as db
    import humorhist.copywriter as cw
    from humorhist.llm import NousClient

    conn = db.connect(str(DB))
    db.migrate(conn)  # idempotent; adds queue.post_copy if missing

    token = current_token()
    os.environ["HUMORHIST_LLM_API_KEY"] = token
    client = NousClient(api_key=token, max_retries=2, timeout=300.0)

    filled = cw.fill_post_copy(conn, client, force=True)
    print(f"Generated post copy for {filled} draft(s).")

    # Report the result for the 2 approved+queued rows.
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
            print(f"\n=== {r['title']} ({r['draft_id']}) — {len(copy)}/280 ===\n{copy}")
        else:
            print(f"\n!!! {r['title']} ({r['draft_id']}) — NO COPY GENERATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
