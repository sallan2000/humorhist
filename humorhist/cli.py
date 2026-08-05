"""Command-line interface for humorhist.

Subcommands:
    harvest   populate the candidate pool from seed CSV + Wikipedia lists
    screen    LLM-score unscored pool candidates for funniness
    draft     fact-check + generate comic angles for top candidates
    status    show pool/draft/queue health
    show      print a single draft in full (for eyeballing angle quality)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import humorhist.db as db

DEFAULT_DB = os.environ.get(
    "HUMORHIST_DB", str(Path.home() / "projects" / "humorhist" / "data" / "humorhist.sqlite")
)


def _open_db(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = db.connect(path)
    db.migrate(conn)
    return conn


def cmd_harvest(args: argparse.Namespace) -> int:
    from humorhist.harvest.seed import load_seed
    from humorhist.harvest.wikipedia_lists import harvest_wikipedia_lists

    conn = _open_db(args.db)

    if not args.wikipedia_only:
        print("Loading seed events...")
        print("  seed:", load_seed(conn))

    if not args.seed_only:
        print("Harvesting Wikipedia lists (this takes a moment)...")
        print("  wikipedia:", harvest_wikipedia_lists(conn))

    print("\nPool totals:", db.counts(conn))
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    from humorhist.harvest.screen import screen_pool
    from humorhist.llm import default_client

    conn = _open_db(args.db)
    client = default_client()
    print(f"Screening unscored candidates (batch={args.batch_size}, limit={args.limit})...")
    result = screen_pool(conn, client, batch_size=args.batch_size, limit=args.limit)
    print("Result:", result)

    cur = conn.execute(
        "SELECT count(*) AS n, round(avg(funny_score),2) AS avg FROM pool WHERE funny_score IS NOT NULL"
    )
    row = cur.fetchone()
    print(f"Scored so far: {row['n']} (avg {row['avg']})")
    return 0


def cmd_draft(args: argparse.Namespace) -> int:
    from humorhist.drafting import draft_candidates
    from humorhist.llm import default_client

    conn = _open_db(args.db)
    client = default_client()
    print(f"Drafting {args.count} candidates (min score {args.min_score})...")
    result = draft_candidates(
        conn, client, count=args.count, min_score=args.min_score
    )
    print("Result:", result)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    conn = _open_db(args.db)
    c = db.counts(conn)
    print("=== humorhist status ===")
    print(f"DB: {args.db}\n")
    print("Row counts:", c)

    bands = conn.execute(
        """
        SELECT
          sum(CASE WHEN funny_score IS NULL THEN 1 ELSE 0 END) AS unscored,
          sum(CASE WHEN funny_score >= 8 THEN 1 ELSE 0 END) AS band8,
          sum(CASE WHEN funny_score >= 7 AND funny_score < 8 THEN 1 ELSE 0 END) AS band7,
          sum(CASE WHEN funny_score >= 5 AND funny_score < 7 THEN 1 ELSE 0 END) AS band5,
          sum(CASE WHEN funny_score < 5 THEN 1 ELSE 0 END) AS below5
        FROM pool
        """
    ).fetchone()
    print("\nPool by score band:")
    print(f"  unscored : {bands['unscored'] or 0}")
    print(f"  >= 8     : {bands['band8'] or 0}")
    print(f"  7 - 8    : {bands['band7'] or 0}")
    print(f"  5 - 7    : {bands['band5'] or 0}")
    print(f"  < 5      : {bands['below5'] or 0}")

    drafts = conn.execute(
        "SELECT status, count(*) AS n FROM drafts GROUP BY status"
    ).fetchall()
    print("\nDrafts by status:")
    if not drafts:
        print("  (none yet)")
    for d in drafts:
        print(f"  {d['status']:10s}: {d['n']}")

    queued = conn.execute(
        "SELECT count(*) AS n FROM queue WHERE published = 0"
    ).fetchone()["n"]
    print(f"\nApproved and queued (unpublished): {queued}")
    if queued < 3:
        print("  ** BUFFER LOW ** run a review session soon")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    conn = _open_db(args.db)
    if args.draft_id:
        row = conn.execute(
            "SELECT * FROM drafts WHERE id = ?", (args.draft_id,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM drafts ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

    if row is None:
        print("No draft found.")
        return 1

    pool = db.get_pool_item(conn, row["pool_id"])
    brief = json.loads(row["brief_json"] or "{}")
    angles = json.loads(row["angles_json"] or "{}")

    title = pool["title"] if pool else "(unknown)"
    year = pool["year"] if pool else ""
    print("=" * 70)
    print(f"DRAFT {row['id']} — {title} ({year})")
    print(f"status: {row['status']}")
    print("=" * 70)

    print("\n--- VERIFIED FACTS ---")
    for f in brief.get("verified_facts", []):
        print(f"  • {f}")

    if brief.get("misconceptions"):
        print("\n--- MISCONCEPTIONS (popular version vs record) ---")
        for m in brief["misconceptions"]:
            print(f"  ! {m}")

    if brief.get("caveats"):
        print("\n--- CAVEATS ---")
        for c in brief["caveats"]:
            print(f"  ? {c}")

    flags = brief.get("sensitivity_flags", [])
    if flags and flags != ["none"]:
        print(f"\n--- SENSITIVITY: {', '.join(flags)} ---")

    print("\n--- COMIC ANGLES ---")
    for i, a in enumerate(angles.get("angles", []), 1):
        print(f"\n  {i}. {a.get('angle_name', '?')}")
        print(f"     setup   : {a.get('setup', '')}")
        print(f"     lands   : {a.get('why_it_lands', '')}")
        print(f"     pitfalls: {a.get('pitfalls', '')}")
        for rm in a.get("raw_material", []):
            print(f"     raw     : {rm}")

    if angles.get("strongest_single_detail"):
        print(f"\n--- STRONGEST DETAIL ---\n  {angles['strongest_single_detail']}")
    if angles.get("suggested_hook"):
        print(f"\n--- SUGGESTED HOOK (factual, not a joke) ---\n  {angles['suggested_hook']}")

    print("\n--- SOURCES ---")
    for s in brief.get("sources", []):
        print(f"  {s.get('title', '')} — {s.get('url', '')}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="humorhist", description=__doc__)
    p.add_argument("--db", default=DEFAULT_DB, help="path to the sqlite database")
    sub = p.add_subparsers(dest="command", required=True)

    h = sub.add_parser("harvest", help="populate the candidate pool")
    h.add_argument("--seed-only", action="store_true")
    h.add_argument("--wikipedia-only", action="store_true")
    h.set_defaults(func=cmd_harvest)

    s = sub.add_parser("screen", help="LLM-score unscored candidates")
    s.add_argument("--batch-size", type=int, default=20)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=cmd_screen)

    d = sub.add_parser("draft", help="fact-check + generate angles")
    d.add_argument("--count", type=int, default=3)
    d.add_argument("--min-score", type=float, default=7.0)
    d.set_defaults(func=cmd_draft)

    st = sub.add_parser("status", help="pool/draft/queue health")
    st.set_defaults(func=cmd_status)

    sh = sub.add_parser("show", help="print a draft in full")
    sh.add_argument("draft_id", nargs="?", default=None)
    sh.set_defaults(func=cmd_show)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
